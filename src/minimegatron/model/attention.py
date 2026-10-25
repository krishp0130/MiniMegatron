"""Multi-head causal self-attention with an explicit reference path and an SDPA backend."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimegatron.model.config import ModelConfig


def split_qkv(qkv: torch.Tensor, n_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused projection (B, T, 3*d) into q, k, v, each (B, n_heads, T, head_dim).

    The last dimension is laid out per head, [h, 3, head_dim]: head 0's q, k, v, then head 1's,
    and so on. Any contiguous slice holding a whole number of heads is therefore self-contained,
    which is what tensor parallelism relies on when it splits this dimension across ranks (D6).
    """
    B, T, three_d = qkv.shape
    head_dim = three_d // (3 * n_heads)
    q, k, v = qkv.view(B, T, n_heads, 3, head_dim).unbind(dim=3)  # each (B, T, h, hd)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # each (B, h, T, hd)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """softmax(q·kᵀ / √d_h + mask) · v, written out op by op.

    q, k, v: (B, h, T, d_h). causal_mask: (T, T) bool, True where attention is allowed.
    Materializes the full (B, h, T, T) score matrix, so memory is quadratic in T.
    """
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # (B, h, T, T)
    scores = scores.masked_fill(~causal_mask, float("-inf"))  # exp(-inf) = 0: no future tokens
    probs = torch.softmax(scores, dim=-1)  # each row sums to 1 over allowed keys
    probs = F.dropout(probs, p=dropout_p, training=training)
    return probs @ v  # (B, h, T, d_h)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.seq_len = cfg.seq_len
        self.dropout_p = cfg.dropout
        self.backend = cfg.attention_backend

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular: query i may attend to keys 0..i. A buffer moves with .to(device);
        # persistent=False keeps it out of state_dict, so both backends share checkpoint keys.
        mask = torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        if T > self.seq_len:
            raise ValueError(f"sequence length {T} exceeds the model's seq_len {self.seq_len}")

        q, k, v = split_qkv(self.qkv(x), self.n_heads)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.backend == "reference":
            y = reference_attention(q, k, v, self.causal_mask[:T, :T], dropout_p, self.training)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        y = y.transpose(1, 2).reshape(B, T, d)  # concatenate heads: (B, h, T, hd) -> (B, T, d)
        return self.resid_dropout(self.out_proj(y))
