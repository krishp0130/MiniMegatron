"""Model configuration: the single source of truth for GPT model shape."""

from __future__ import annotations

from dataclasses import dataclass

from minimegatron.config import ConfigError

ATTENTION_BACKENDS = ("reference", "sdpa")
NORM_TYPES = ("layernorm", "rmsnorm")
# BF16 arrives with mixed precision in Milestone 3.
SUPPORTED_DTYPES = ("float32",)


def _is_int(v: object) -> bool:
    # bool is a subclass of int in Python; `n_layers: true` should not pass as 1.
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class ModelConfig:
    """Shape and options of a decoder-only GPT model.

    Frozen so a config can't change after the model is built from it. Validated on construction,
    so an invalid config can't exist at all.
    """

    n_layers: int
    n_heads: int
    d_model: int
    vocab_size: int = 50304  # GPT-2 BPE (50257) padded to a multiple of 128
    seq_len: int = 256  # max context length; size of the learned position table
    d_ff: int | None = None  # defaults to 4 * d_model
    dropout: float = 0.0
    bias: bool = True  # biases in Linear and LayerNorm layers (GPT-2 style)
    tie_embeddings: bool = True  # LM head reuses the token-embedding matrix
    attention_backend: str = "reference"
    norm_type: str = "layernorm"
    dtype: str = "float32"  # parameter storage dtype

    def __post_init__(self) -> None:
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", 4 * self.d_model)  # frozen: bypass __setattr__

        errors = []
        for name in ("n_layers", "n_heads", "d_model", "vocab_size", "seq_len", "d_ff"):
            v = getattr(self, name)
            if not _is_int(v) or v <= 0:
                errors.append(f"{name} must be a positive integer, got {v!r}")
        if _is_int(self.d_model) and _is_int(self.n_heads) and self.n_heads > 0:
            if self.d_model % self.n_heads != 0:
                errors.append(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
                )

        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            errors.append(f"dropout must be a number, got {self.dropout!r}")
        elif not 0.0 <= self.dropout < 1.0:
            errors.append(f"dropout must be in [0, 1), got {self.dropout}")

        for name in ("bias", "tie_embeddings"):
            if not isinstance(getattr(self, name), bool):
                errors.append(f"{name} must be true or false, got {getattr(self, name)!r}")

        if self.attention_backend not in ATTENTION_BACKENDS:
            errors.append(
                f"attention_backend must be one of {ATTENTION_BACKENDS}, got {self.attention_backend!r}"
            )
        if self.norm_type not in NORM_TYPES:
            errors.append(f"norm_type must be one of {NORM_TYPES}, got {self.norm_type!r}")
        if self.dtype == "bfloat16":
            errors.append("dtype 'bfloat16' is not supported yet (mixed precision lands in Milestone 3)")
        elif self.dtype not in SUPPORTED_DTYPES:
            errors.append(f"dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype!r}")

        if errors:
            raise ConfigError("invalid model config:\n  - " + "\n  - ".join(errors))

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_params(self, include_embeddings: bool = True) -> int:
        """Exact parameter count, computed analytically from the config.

        The real model's count is tested against this, so the two must stay in sync.

        With `include_embeddings=False` the token and position tables are excluded. Scaling-law
        papers report that number because the embeddings hold a large share of a small model's
        parameters (84% of the tiny preset) while doing none of the per-token computation.
        """
        d, f, V, T, L = self.d_model, self.d_ff, self.vocab_size, self.seq_len, self.n_layers
        b = 1 if self.bias else 0
        norm = d * (2 if self.norm_type == "layernorm" and self.bias else 1)  # gamma (+ beta)

        attention = 4 * d * d + b * 4 * d  # qkv (d -> 3d) and out_proj (d -> d)
        mlp = 2 * d * f + b * (f + d)  # fc1 (d -> f) and fc2 (f -> d)
        per_layer = attention + mlp + 2 * norm  # two norms per block

        embeddings = V * d + T * d  # token + learned position
        lm_head = 0 if self.tie_embeddings else V * d
        total = L * per_layer + embeddings + norm + lm_head  # + final norm
        return total if include_embeddings else total - embeddings
