"""Normalization, MLP, and the pre-norm transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimegatron.model.attention import CausalSelfAttention
from minimegatron.model.config import ModelConfig

NORM_EPS = 1e-5


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """y = x / sqrt(mean(x²) + eps) · weight, normalized over the last dimension.

    This is the reference implementation the Triton kernel (Milestone 5) is tested against.
    Statistics are computed in at least float32 even for low-precision inputs: a mean of squares in
    bf16 loses too much precision. (Promote, never downcast: float64 inputs stay float64.)
    """
    xf = x.to(torch.promote_types(x.dtype, torch.float32))
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms).to(x.dtype) * weight


class RMSNorm(nn.Module):
    """RMSNorm, dispatching to either the PyTorch reference or the fused Triton kernel.

    The backend is validated at construction so an explicit `triton` on a machine without it fails
    immediately rather than at the first forward pass. The `pytorch` path imports nothing from
    `kernels`, which keeps the default path free of the dependency entirely (and avoids an import
    cycle, since `kernels.rmsnorm` imports `rms_norm` from here).
    """

    def __init__(self, dim: int, eps: float = NORM_EPS, backend: str = "pytorch") -> None:
        super().__init__()
        self.eps = eps
        self.backend = backend
        self.weight = nn.Parameter(torch.ones(dim))

        # Resolved once, here, to a concrete 'pytorch' or 'triton'. Leaving it as 'auto' would
        # mean re-resolving -- including a `torch.cuda.is_available()` call -- on every forward
        # pass of every norm in the model, which is a hot path.
        if backend == "pytorch":
            self.resolved_backend = "pytorch"
        else:
            from minimegatron.kernels.rmsnorm import resolve_norm_backend

            self.resolved_backend = resolve_norm_backend(backend)  # raises now, not mid-training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resolved_backend == "pytorch":
            return rms_norm(x, self.weight, self.eps)

        if not x.is_cuda:
            raise RuntimeError(
                f"norm_backend {self.backend!r} resolved to the Triton kernel, which needs CUDA "
                f"tensors, but this input is on {x.device}. Set model.norm_backend: pytorch, or "
                "run on a CUDA device."
            )
        from minimegatron.kernels.rmsnorm import rms_norm_triton

        return rms_norm_triton(x, self.weight, self.eps)


def make_norm(cfg: ModelConfig) -> nn.Module:
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.d_model, backend=cfg.norm_backend)
    return nn.LayerNorm(cfg.d_model, eps=NORM_EPS, bias=cfg.bias)


class MLP(nn.Module):
    """Position-wise feed-forward network: fc2(GELU(fc1(x))), applied to each token independently."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)  # expand d -> d_ff
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)  # project back d_ff -> d
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))  # exact (erf) GELU


class Block(nn.Module):
    """Pre-norm transformer block. The residual stream x is only ever added to, never normalized:

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))  # tokens exchange information
        x = x + self.mlp(self.norm2(x))  # each token is processed on its own
        return x
