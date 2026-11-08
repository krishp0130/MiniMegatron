"""Data configuration."""

from __future__ import annotations

from dataclasses import dataclass

from minimegatron.config import ConfigError

DATASETS = ("random", "memorization")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class DataConfig:
    """Which synthetic dataset to train on, and the seeds that make it reproducible.

    `random` is unlearnable, so loss stays at ln V: the right choice for throughput benchmarks,
    where the data must cost nothing to produce. `memorization` repeats `n_sequences` fixed
    sequences, which the model *can* drive to ~0 loss: the right choice for proving the
    forward/backward/optimizer loop is correctly wired.
    """

    dataset: str = "random"
    n_items: int = 4096  # samples per epoch (the length of one index permutation)
    n_sequences: int = 8  # distinct sequences, `memorization` only
    seed: int = 0
    val_seed: int | None = None  # defaults to seed + 1: a disjoint validation split

    def __post_init__(self) -> None:
        if self.val_seed is None:
            object.__setattr__(self, "val_seed", self.seed + 1)

        errors = []
        if self.dataset not in DATASETS:
            errors.append(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        for name in ("n_items", "n_sequences"):
            v = getattr(self, name)
            if not _is_int(v) or v <= 0:
                errors.append(f"{name} must be a positive integer, got {v!r}")
        for name in ("seed", "val_seed"):
            v = getattr(self, name)
            if not _is_int(v) or v < 0:
                errors.append(f"{name} must be a non-negative integer, got {v!r}")
        if self.seed == self.val_seed:
            errors.append("val_seed must differ from seed, or train and val are the same data")

        if errors:
            raise ConfigError("invalid data config:\n  - " + "\n  - ".join(errors))
