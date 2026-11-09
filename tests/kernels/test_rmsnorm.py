"""RMSNorm: the PyTorch reference, and the Triton kernel judged by the same assertions.

The tests are parameterized over backend so the kernel is held to exactly the reference's
standard, not a weaker one written later. On this machine only the `pytorch` parameters run --
Triton needs an NVIDIA GPU and has no macOS wheels (ISSUES #3) -- so **the Triton kernel has never
been executed and nothing here asserts that it works.**

What *is* verified locally: the reference across shapes, dtypes and extreme inputs; a float64
gradcheck; the unit-RMS property; and the dispatch and fallback logic that selects a backend.
"""

import math

import pytest
import torch

from minimegatron.config import ConfigError
from minimegatron.kernels.rmsnorm import (
    MAX_FUSED_WIDTH,
    NORM_BACKENDS,
    TRITON_AVAILABLE,
    resolve_norm_backend,
    rms_norm_dispatch,
    rms_norm_triton,
    triton_usable,
)
from minimegatron.model.config import ModelConfig
from minimegatron.model.layers import NORM_EPS, RMSNorm, make_norm, rms_norm

TRITON_REASON = "Triton + an NVIDIA GPU required (ISSUES #3)"

BACKENDS = [
    pytest.param("pytorch", id="pytorch"),
    pytest.param(
        "triton",
        id="triton",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not triton_usable(), reason=TRITON_REASON)],
    ),
]

# Non-powers-of-two are deliberate: the tiny preset's d_model is 192, and the kernel's BLOCK is
# the next power of two with a mask for the tail.
SHAPES = [(1, 1), (2, 16), (4, 192), (3, 193), (8, 64), (1, 1024), (7, 33), (16, 768)]


def apply_norm(backend: str, x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS):
    if backend == "triton":
        return rms_norm_triton(x, weight, eps)
    return rms_norm(x, weight, eps)


def device_for(backend: str) -> torch.device:
    return torch.device("cuda" if backend == "triton" else "cpu")


# --- against torch.nn.RMSNorm -------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows, cols", SHAPES)
def test_matches_torch_rmsnorm(backend, rows, cols):
    """The trusted third-party implementation, as the outermost check on both backends."""
    device = device_for(backend)
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=device)
    weight = torch.randn(cols, device=device)

    reference = torch.nn.RMSNorm(cols, eps=NORM_EPS, device=device)
    with torch.no_grad():
        reference.weight.copy_(weight)
        expected = reference(x)

    torch.testing.assert_close(apply_norm(backend, x, weight), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_the_closed_form(backend):
    device = device_for(backend)
    x = torch.tensor([[3.0, 4.0]], device=device)
    weight = torch.ones(2, device=device)
    # rms = sqrt((9 + 16) / 2) = 3.5355...
    expected = x / math.sqrt(12.5 + NORM_EPS)
    torch.testing.assert_close(apply_norm(backend, x, weight), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_has_unit_rms_when_weight_is_one(backend):
    """The property the operation exists to provide."""
    device = device_for(backend)
    torch.manual_seed(0)
    x = torch.randn(8, 128, device=device) * 37.0  # scale must not matter
    y = apply_norm(backend, x, torch.ones(128, device=device))
    rms = y.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(8, device=device), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_weight_scales_each_channel(backend):
    device = device_for(backend)
    x = torch.randn(4, 16, device=device, generator=None)
    plain = apply_norm(backend, x, torch.ones(16, device=device))
    scaled = apply_norm(backend, x, torch.full((16,), 3.0, device=device))
    torch.testing.assert_close(scaled, plain * 3.0, rtol=1e-5, atol=1e-6)


# --- dtypes -------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [(torch.float32, 1e-5, 1e-6), (torch.bfloat16, 2e-2, 1e-2), (torch.float16, 1e-2, 5e-3)],
)
def test_dtypes_with_dtype_appropriate_tolerances(backend, dtype, rtol, atol):
    """SCOPE.md section 8: dtype-appropriate tolerances, not exact float equality."""
    device = device_for(backend)
    torch.manual_seed(0)
    x = torch.randn(4, 64, device=device, dtype=dtype)
    weight = torch.randn(64, device=device, dtype=dtype)

    got = apply_norm(backend, x, weight)
    assert got.dtype == dtype  # the op must not silently promote its output

    # Compared against the same maths in float64, so the reference is not itself the thing under
    # test at this precision.
    exact = rms_norm(x.double(), weight.double())
    torch.testing.assert_close(got.double(), exact, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eps_is_actually_applied(backend):
    """eps must be inside the square root, and it is not always negligible.

    At x = 0.01 the mean of squares is ~1e-4, only ten times eps = 1e-5, so the output is 0.954
    rather than the 1.0 an eps-free formula would give. A kernel that dropped eps, or added it
    outside the sqrt, would pass every well-scaled test and fail here -- which is the point.
    """
    device = device_for(backend)
    x = torch.full((2, 256), 0.01, device=device)
    weight = torch.ones(256, device=device)

    y = apply_norm(backend, x, weight)
    exact = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + NORM_EPS)

    torch.testing.assert_close(y.double(), exact, rtol=1e-5, atol=1e-7)
    assert y.max().item() < 0.97  # not 1.0: eps genuinely moved the answer


@pytest.mark.parametrize("backend", BACKENDS)
def test_statistics_survive_a_bf16_input(backend):
    """bf16 has 8 mantissa bits, so the reduction is upcast to at least fp32 (ISSUES #4).

    Compared against the same maths in float64 on the same (already-rounded) bf16 values, so this
    isolates the *accumulation* precision rather than the input's representation error.
    """
    device = device_for(backend)
    torch.manual_seed(0)
    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    weight = torch.ones(512, device=device, dtype=torch.bfloat16)

    y = apply_norm(backend, x, weight)
    exact = rms_norm(x.double(), weight.double())
    torch.testing.assert_close(y.double(), exact, rtol=2e-2, atol=1e-2)


# --- extreme inputs -----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_zero_row_does_not_divide_by_zero(backend):
    """eps is what makes this finite; without it the row would be 0/0."""
    device = device_for(backend)
    y = apply_norm(backend, torch.zeros(2, 32, device=device), torch.ones(32, device=device))
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
def test_extreme_magnitudes_stay_finite(backend, scale):
    device = device_for(backend)
    torch.manual_seed(0)
    x = torch.randn(4, 64, device=device) * scale
    y = apply_norm(backend, x, torch.ones(64, device=device))
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_magnitudes_within_a_row(backend):
    device = device_for(backend)
    x = torch.tensor([[1e-6, 1.0, 1e6, -1e6]], device=device)
    y = apply_norm(backend, x, torch.ones(4, device=device))
    assert torch.isfinite(y).all()
    exact = rms_norm(x.double(), torch.ones(4, device=device, dtype=torch.float64))
    torch.testing.assert_close(y.double(), exact, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_element_rows(backend):
    """N = 1: rms = |x|, so the output is sign(x) scaled by the weight."""
    device = device_for(backend)
    x = torch.tensor([[-5.0], [2.0]], device=device)
    y = apply_norm(backend, x, torch.ones(1, device=device))
    torch.testing.assert_close(y, torch.tensor([[-1.0], [1.0]], device=device), rtol=1e-4, atol=1e-5)


# --- gradients ----------------------------------------------------------------------------------


def test_reference_passes_float64_gradcheck():
    """Finite differences against autograd, in float64. The Triton path has no fp64 kernel, so
    this judges the reference -- which is what the kernel is then compared against."""
    torch.manual_seed(0)
    x = torch.randn(3, 11, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, w: rms_norm(a, w), (x, weight), eps=1e-6, atol=1e-8)


@pytest.mark.gpu
@pytest.mark.skipif(not triton_usable(), reason=TRITON_REASON)
@pytest.mark.parametrize("rows, cols", SHAPES)
def test_triton_gradients_match_the_reference(rows, cols):
    """The kernel's whole correctness claim, in one test. NEVER EXECUTED on this machine."""
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda")
    weight = torch.randn(cols, device="cuda")

    xr, wr = x.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    xt, wt = x.clone().requires_grad_(True), weight.clone().requires_grad_(True)

    reference = rms_norm(xr, wr)
    fused = rms_norm_triton(xt, wt)
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)

    grad = torch.randn_like(reference)
    reference.backward(grad)
    fused.backward(grad)
    torch.testing.assert_close(xt.grad, xr.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(wt.grad, wr.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not triton_usable(), reason=TRITON_REASON)
def test_triton_handles_non_contiguous_input():
    """A transposed view is a realistic input and the kernel indexes rows by stride."""
    torch.manual_seed(0)
    x = torch.randn(64, 8, device="cuda").t()  # (8, 64), non-contiguous
    weight = torch.randn(64, device="cuda")
    torch.testing.assert_close(
        rms_norm_triton(x, weight), rms_norm(x, weight), rtol=1e-5, atol=1e-6
    )


@pytest.mark.gpu
@pytest.mark.skipif(not triton_usable(), reason=TRITON_REASON)
def test_triton_handles_three_dimensional_activations():
    """Real activations are (batch, seq, d); the kernel flattens to rows."""
    torch.manual_seed(0)
    x = torch.randn(2, 7, 192, device="cuda")
    weight = torch.randn(192, device="cuda")
    torch.testing.assert_close(
        rms_norm_triton(x, weight), rms_norm(x, weight), rtol=1e-5, atol=1e-6
    )


@pytest.mark.gpu
@pytest.mark.skipif(not TRITON_AVAILABLE, reason=TRITON_REASON)
def test_width_beyond_the_fused_limit_is_rejected():
    from minimegatron.kernels.rmsnorm import _block_size

    with pytest.raises(ValueError, match="exceeds the fused kernel's limit"):
        _block_size(MAX_FUSED_WIDTH + 1)


# --- dispatch and fallback (runs everywhere) ----------------------------------------------------


def test_pytorch_backend_always_resolves():
    assert resolve_norm_backend("pytorch") == "pytorch"
    assert resolve_norm_backend("pytorch", torch.device("cpu")) == "pytorch"


def test_auto_resolves_per_device_not_just_per_machine():
    """`auto` depends on the *tensor's* device, not merely on whether Triton is installed.

    The earlier version of this test asserted `triton` whenever `triton_usable()` was true, while
    passing `device=cpu` -- so it could only fail on a CUDA machine, and did (ISSUES #20). A CPU
    tensor must never be routed to the GPU kernel.
    """
    assert resolve_norm_backend("auto") == ("triton" if triton_usable() else "pytorch")
    assert resolve_norm_backend("auto", torch.device("cpu")) == "pytorch"


@pytest.mark.skipif(triton_usable(), reason="Triton is usable here, so this would not raise")
def test_explicit_triton_is_a_hard_error_without_triton():
    """No silent fallback: a run that quietly used a different code path than its config says is
    worse than one that stops -- the same rule as `device` and `precision`."""
    with pytest.raises(ConfigError, match="norm_backend 'triton' requested but"):
        resolve_norm_backend("triton")


def test_unknown_backend_lists_the_valid_ones():
    with pytest.raises(ConfigError, match="norm_backend must be one of"):
        resolve_norm_backend("cutlass")
    assert set(NORM_BACKENDS) == {"auto", "pytorch", "triton"}


def test_dispatch_matches_the_reference_on_the_pytorch_path():
    torch.manual_seed(0)
    x, weight = torch.randn(4, 48), torch.randn(48)
    torch.testing.assert_close(rms_norm_dispatch(x, weight, backend="pytorch"), rms_norm(x, weight))


def test_dispatch_through_auto_works_on_cpu():
    torch.manual_seed(0)
    x, weight = torch.randn(4, 48), torch.randn(48)
    if triton_usable():
        pytest.skip("auto would select the Triton path on this machine")
    torch.testing.assert_close(rms_norm_dispatch(x, weight, backend="auto"), rms_norm(x, weight))


# --- the module and the config switch -----------------------------------------------------------


def test_rmsnorm_module_resolves_its_backend_once():
    """'auto' must not be re-resolved per forward: that is a `torch.cuda.is_available()` call in a
    hot path, once per norm per step."""
    norm = RMSNorm(32, backend="auto")
    assert norm.backend == "auto"
    assert norm.resolved_backend in ("pytorch", "triton")


@pytest.mark.parametrize("requested", ["auto", "pytorch"])
def test_make_norm_passes_the_configured_backend(requested):
    """Asserts the *requested* backend propagates, not the resolved one.

    Checking `resolved_backend == "pytorch"` would be vacuous on a machine without Triton, since
    both `auto` and the module default resolve there anyway -- a mutation that ignored the config
    entirely would pass. The requested value is what distinguishes them.
    """
    cfg = ModelConfig(n_layers=1, n_heads=4, d_model=32, norm_type="rmsnorm",
                      norm_backend=requested)
    norm = make_norm(cfg)
    assert norm.backend == requested
    assert norm.resolved_backend in ("pytorch", "triton")


def test_layernorm_ignores_the_norm_backend():
    cfg = ModelConfig(n_layers=1, n_heads=4, d_model=32, norm_type="layernorm",
                      norm_backend="auto")
    assert isinstance(make_norm(cfg), torch.nn.LayerNorm)


def test_triton_backend_with_layernorm_is_rejected():
    """A config asking for a kernel it can never reach is a config whose author believes something
    false, so it is an error rather than silently ignored."""
    with pytest.raises(ConfigError, match="only applies to norm_type 'rmsnorm'"):
        ModelConfig(n_layers=1, n_heads=4, d_model=32, norm_type="layernorm",
                    norm_backend="triton")


@pytest.mark.skipif(triton_usable(), reason="Triton is usable here")
def test_module_construction_fails_fast_for_explicit_triton():
    """Fail at construction, not at the first training step an hour into a run."""
    with pytest.raises(ConfigError, match="Triton is not installed"):
        RMSNorm(32, backend="triton")


def test_the_reference_is_still_the_documented_reference():
    """M1 wrote `rms_norm` as the implementation this milestone is checked against. If it ever
    stops being importable from there, the kernel has nothing to be judged by."""
    from minimegatron.model.layers import rms_norm as reference

    assert callable(reference)
    x = torch.randn(2, 8)
    torch.testing.assert_close(reference(x, torch.ones(8)), rms_norm(x, torch.ones(8)))


# --- the kernel benchmark CLI -------------------------------------------------------------------


@pytest.mark.skipif(triton_usable(), reason="Triton is usable here, so the CLI would run")
def test_kernel_benchmark_refuses_clearly_without_a_gpu(capsys):
    """It must say *why* and exit non-zero, rather than measure the wrong device or crash."""
    from minimegatron.benchmarks.kernels import main

    assert main([]) == 2
    error = capsys.readouterr().err
    assert "need Triton and an NVIDIA GPU" in error
    assert "triton installed: False" in error
    assert "ISSUES #3" in error  # points at the recorded environment limitation


def test_kernel_benchmark_markdown_is_honest_about_what_it_counts():
    """The bandwidth column counts the fused ideal (one read, one write); the table says so, so a
    reader cannot mistake it for the reference's actual traffic."""
    from minimegatron.benchmarks.kernels import to_markdown

    record = {
        "rows": 4096, "cols": 768, "dtype": "float32", "direction": "fwd",
        "reference_ms": {"median": 1.0}, "triton_ms": {"median": 0.25},
        "speedup": 4.0, "triton_gbps": 900.0, "reference_gbps": 225.0,
    }
    table = to_markdown([record])
    assert "4.00x" in table and "900" in table
    assert "one read and one write" in table
    assert "No kernel benchmark records" in to_markdown([])
