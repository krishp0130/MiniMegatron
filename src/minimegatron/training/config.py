"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass

from minimegatron.config import ConfigError

DEVICES = ("auto", "cpu", "cuda", "mps")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer, schedule, batching, logging, and checkpointing settings."""

    max_steps: int = 100
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    # Optional cross-check: micro_batch_size * grad_accum_steps * dp_world_size must equal it.
    global_batch_size: int | None = None

    lr: float = 3e-4
    min_lr: float | None = None  # defaults to lr / 10
    warmup_steps: int = 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0  # 0 disables clipping (the norm is still logged)

    device: str = "auto"
    seed: int = 1337
    out_dir: str = "runs/default"
    log_interval: int = 10
    eval_interval: int = 0  # 0 disables validation
    eval_steps: int = 10
    ckpt_interval: int = 0  # 0 saves only at the end

    def __post_init__(self) -> None:
        if self.min_lr is None and _is_num(self.lr):
            object.__setattr__(self, "min_lr", self.lr / 10)

        errors = []
        for name in ("max_steps", "micro_batch_size", "grad_accum_steps", "eval_steps"):
            v = getattr(self, name)
            if not _is_int(v) or v <= 0:
                errors.append(f"{name} must be a positive integer, got {v!r}")
        for name in ("warmup_steps", "log_interval", "eval_interval", "ckpt_interval", "seed"):
            v = getattr(self, name)
            if not _is_int(v) or v < 0:
                errors.append(f"{name} must be a non-negative integer, got {v!r}")
        if self.global_batch_size is not None and (
            not _is_int(self.global_batch_size) or self.global_batch_size <= 0
        ):
            errors.append(
                f"global_batch_size must be a positive integer or null, got {self.global_batch_size!r}"
            )

        if not _is_num(self.lr) or self.lr <= 0:
            errors.append(f"lr must be a positive number, got {self.lr!r}")
        if not _is_num(self.min_lr) or self.min_lr < 0:
            errors.append(f"min_lr must be a non-negative number, got {self.min_lr!r}")
        elif _is_num(self.lr) and self.min_lr > self.lr:
            errors.append(f"min_lr ({self.min_lr}) must not exceed lr ({self.lr})")
        for name in ("weight_decay", "eps", "grad_clip"):
            v = getattr(self, name)
            if not _is_num(v) or v < 0:
                errors.append(f"{name} must be a non-negative number, got {v!r}")
        for name in ("beta1", "beta2"):
            v = getattr(self, name)
            if not _is_num(v) or not 0.0 <= v < 1.0:
                errors.append(f"{name} must be in [0, 1), got {v!r}")

        if self.device not in DEVICES:
            errors.append(f"device must be one of {DEVICES}, got {self.device!r}")
        if not isinstance(self.out_dir, str) or not self.out_dir:
            errors.append(f"out_dir must be a non-empty string, got {self.out_dir!r}")

        if errors:
            raise ConfigError("invalid train config:\n  - " + "\n  - ".join(errors))

    @property
    def tokens_per_step_per_rank(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    def validate_global_batch(self, dp_world_size: int = 1) -> None:
        """Refuse to start when the declared global batch doesn't match the actual one.

        global_batch_size = micro_batch_size x grad_accum_steps x dp_world_size

        Checked at startup rather than ignored, because a silently wrong global batch size
        invalidates every number a run produces. `dp_world_size` is 1 until Milestone 2.
        """
        if self.global_batch_size is None:
            return
        actual = self.micro_batch_size * self.grad_accum_steps * dp_world_size
        if actual != self.global_batch_size:
            raise ConfigError(
                f"global_batch_size {self.global_batch_size} != micro_batch_size "
                f"{self.micro_batch_size} x grad_accum_steps {self.grad_accum_steps} "
                f"x dp_world_size {dp_world_size} = {actual}"
            )
