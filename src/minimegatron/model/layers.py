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
    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def make_norm(cfg: ModelConfig) -> nn.Module:
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.d_model)
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
