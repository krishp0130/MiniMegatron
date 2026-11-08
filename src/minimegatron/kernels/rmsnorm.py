"""Fused RMSNorm in Triton, with the PyTorch reference as the arbiter of correctness.

    y = x / sqrt(mean(x^2) + eps) * w

PyTorch runs this as a sequence of kernels -- square, mean, add, rsqrt, multiply, multiply -- and
every one of them **reads its input from HBM and writes its output back**. The arithmetic is
trivial, so the whole operation is memory-bound and the time is all traffic. A fused kernel reads
the row once, does all six steps in registers, and writes once. That ratio of passes over memory is
the entire point; a correct but unfused kernel would be pointless.

**STATUS: the Triton kernels in this module have never been executed.** There is no NVIDIA GPU on
the development machine and Triton does not install on macOS (ISSUES #3). Everything around them --
the reference, the tests that will judge them, the dispatch, the fallback -- is tested. The kernels
themselves are *unverified code* until they have run on real hardware, and nothing in this project
claims otherwise.
"""

from __future__ import annotations

import torch

from minimegatron.config import ConfigError
from minimegatron.model.layers import NORM_EPS, rms_norm

try:  # pragma: no cover - depends on the machine, not on a branch
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False

NORM_BACKENDS = ("auto", "pytorch", "triton")
# Triton's block size must be a power of two; d_model need not be (the tiny preset uses 192).
MAX_FUSED_WIDTH = 65536


def triton_usable() -> bool:
    """Triton is only useful with a CUDA device to run it on."""
    return TRITON_AVAILABLE and torch.cuda.is_available()


def resolve_norm_backend(backend: str, device: torch.device | None = None) -> str:
    """Pick the RMSNorm implementation, erroring rather than silently substituting one.

    `triton` without Triton is a **hard error**, following the same rule as `device` and
    `precision`: a run that quietly used a different code path than its config says is worse than
    one that stops.

    `auto` is the single place a silent fallback is allowed, and only because the two paths are
    asserted numerically equivalent. If that assertion ever stops holding, `auto` becomes a
    liability and should be removed rather than patched.
    """
    if backend not in NORM_BACKENDS:
        raise ConfigError(f"norm_backend must be one of {NORM_BACKENDS}, got {backend!r}")

    if backend == "pytorch":
        return "pytorch"

    on_cuda = device is None or device.type == "cuda"
    if backend == "triton":
        if not TRITON_AVAILABLE:
            raise ConfigError(
                "norm_backend 'triton' requested but Triton is not installed "
                "(it needs an NVIDIA GPU and has no macOS wheels); use 'pytorch' or 'auto'"
            )
        if not torch.cuda.is_available():
            raise ConfigError(
                "norm_backend 'triton' requested but CUDA is not available; use 'pytorch' or 'auto'"
            )
        if not on_cuda:
            raise ConfigError(
                f"norm_backend 'triton' requires a CUDA device, got {device}; "
                "use 'pytorch' or 'auto'"
            )
        return "triton"

    return "triton" if (triton_usable() and on_cuda) else "pytorch"


if TRITON_AVAILABLE:  # pragma: no cover - requires an NVIDIA GPU

    @triton.jit
    def _rmsnorm_forward_kernel(
        x_ptr, w_ptr, y_ptr, rstd_ptr, stride_row, n_cols, eps, BLOCK: tl.constexpr
    ):
        """One program per row: read it once, do everything, write once."""
        row = tl.program_id(0)
        x_ptr += row * stride_row
        y_ptr += row * stride_row

        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols  # BLOCK is a power of two; n_cols need not be

        # Accumulate in float32 whatever the input dtype. A mean of squares in bf16 loses far too
        # much precision, and this is the one place in the op where that matters (ISSUES #4).
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_square + eps)
        tl.store(rstd_ptr + row, rstd)  # saved for the backward, rather than recomputed

        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + offsets, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_backward_dx_kernel(
        dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, stride_row, n_cols, BLOCK: tl.constexpr
    ):
        """dx only. `dw` is a cross-row reduction and is done in PyTorch (D31).

        dL/dx_i = r*w_i*dy_i - (r^3 * x_i / N) * sum_j(dy_j * w_j * x_j)

        The second term is why this needs a kernel at all: it couples every element of the row
        through one reduction, which is exactly what fuses well and what an unfused sequence would
        pay a full extra pass of memory traffic for.
        """
        row = tl.program_id(0)
        dy_ptr += row * stride_row
        x_ptr += row * stride_row
        dx_ptr += row * stride_row

        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols

        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)

        coefficient = tl.sum(dy * w * x, axis=0) / n_cols
        dx = rstd * w * dy - (rstd * rstd * rstd) * x * coefficient
        tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _block_size(n_cols: int) -> int:
    if n_cols > MAX_FUSED_WIDTH:
        raise ValueError(
            f"row width {n_cols} exceeds the fused kernel's limit of {MAX_FUSED_WIDTH}; "
            "use norm_backend 'pytorch'"
        )
    return max(16, triton.next_power_of_2(n_cols))


class _RMSNormTriton(torch.autograd.Function):  # pragma: no cover - requires an NVIDIA GPU
    """Autograd wrapper: Triton forward, Triton `dx`, PyTorch `dw` (D31)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        rows = rows if rows.is_contiguous() else rows.contiguous()
        n_rows, n_cols = rows.shape

        y = torch.empty_like(rows)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=rows.device)
        block = _block_size(n_cols)

        _rmsnorm_forward_kernel[(n_rows,)](
            rows, weight, y, rstd, rows.stride(0), n_cols, eps, BLOCK=block,
            num_warps=min(32, max(4, block // 256)),
        )
        ctx.save_for_backward(rows, weight, rstd)
        ctx.eps = eps
        ctx.block = block
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        rows, weight, rstd = ctx.saved_tensors
        shape = grad_output.shape
        dy = grad_output.reshape(-1, shape[-1])
        dy = dy if dy.is_contiguous() else dy.contiguous()
        n_rows, n_cols = dy.shape

        dx = torch.empty_like(dy)
        _rmsnorm_backward_dx_kernel[(n_rows,)](
            dy, rows, weight, rstd, dx, dy.stride(0), n_cols, BLOCK=ctx.block,
            num_warps=min(32, max(4, ctx.block // 256)),
        )

        # dw = sum over rows of dy * x * r. A plain reduction with nothing to fuse away, so it is
        # left to PyTorch rather than hand-rolled with locks or atomics (D31).
        dw = (dy.to(torch.float32) * rows.to(torch.float32) * rstd[:, None]).sum(0)
        return dx.reshape(shape), dw.to(weight.dtype), None


def rms_norm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS
) -> torch.Tensor:
    """The fused path. Raises if Triton is unavailable rather than falling back silently."""
    if not TRITON_AVAILABLE:
        raise ConfigError("Triton is not installed; use the 'pytorch' norm backend")
    return _RMSNormTriton.apply(x, weight, eps)


def rms_norm_dispatch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS, backend: str = "auto"
) -> torch.Tensor:
    """Run RMSNorm through the selected backend. The two must be numerically equivalent."""
    if resolve_norm_backend(backend, x.device) == "triton":
        return rms_norm_triton(x, weight, eps)
    return rms_norm(x, weight, eps)
