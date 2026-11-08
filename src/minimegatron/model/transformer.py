"""The full decoder-only GPT model: embeddings, a stack of blocks, a final norm, and an LM head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimegatron.model.config import ModelConfig
from minimegatron.model.layers import Block, make_norm

INIT_STD = 0.02  # GPT-2's initialization scale for every weight matrix
IGNORE_INDEX = -100  # positions with this target contribute nothing to the loss


class GPT(nn.Module):
    """A GPT-style decoder-only transformer.

        idx -> tok_emb + pos_emb -> dropout -> Block x L -> final norm -> lm_head -> logits

    `forward` returns `(logits, loss)`, where `loss` is None unless targets are supplied.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)  # learned absolute positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = make_norm(cfg)  # pre-norm never normalizes the residual stream itself
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # One matrix for token -> vector and vector -> token scores (D8). Assigning the
            # Parameter object itself (not a copy) makes the two uses literally the same tensor,
            # so one gradient accumulates from both and `parameters()` yields it once.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 init: N(0, 0.02) weights, zero biases, norms left at gamma=1, beta=0."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)

    def _scale_residual_projections(self) -> None:
        """Shrink the two projections that write into the residual stream by 1/sqrt(2L).

        Each block adds two sublayer outputs to the stream. If every one has variance s^2, the
        stream's variance grows like 2L*s^2 with depth, so activations at the top of a deep stack
        start out far larger than at the bottom. Scaling these weights by 1/sqrt(2L) keeps the
        stream's scale roughly constant with depth at initialization.
        """
        std = INIT_STD / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=std)

    def num_params(self, include_embeddings: bool = True) -> int:
        """Trainable parameter count. Shared (tied) tensors are counted once.

        Scaling-law papers report non-embedding counts because the token table dominates small
        models without doing any of the computation: it is 84% of the tiny preset.
        """
        total = sum(p.numel() for p in self.parameters())
        if include_embeddings:
            return total
        return total - self.tok_emb.weight.numel() - self.pos_emb.weight.numel()

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(
                f"sequence length {T} exceeds the model's seq_len {self.cfg.seq_len}"
            )

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, d); pos broadcasts over B
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_f(x))  # (B, T, V)

        loss = None
        if targets is not None:
            # Cross-entropy in at least float32: the log-sum-exp over V logits is the one place
            # low precision really hurts. Promote rather than cast, so float64 stays float64 (#4).
            flat = logits.view(-1, logits.size(-1))
            flat = flat.to(torch.promote_types(flat.dtype, torch.float32))
            loss = F.cross_entropy(flat, targets.reshape(-1), ignore_index=IGNORE_INDEX)
        return logits, loss
