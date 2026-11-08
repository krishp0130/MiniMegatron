"""Distributed configuration."""

from __future__ import annotations

from dataclasses import dataclass

from minimegatron.config import ConfigError

BACKENDS = ("auto", "gloo", "nccl")
# single: one replica. ddp: replicate everything, shard the batch. fsdp: shard parameters,
# gradients and optimizer state too. tp: shard each layer's weight matrices, so every rank
# computes a slice of every layer. `auto` picks single or ddp from the world size.
STRATEGIES = ("auto", "single", "ddp", "fsdp", "tp")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class DistributedConfig:
    """How to form the process group. Absent `torchrun`, none of this is used.

    `auto` picks nccl when CUDA is present and gloo otherwise, which is the right default: nccl is
    the only backend with fast GPU collectives, and gloo is the only one that works on CPU (and so
    the one every correctness test uses -- D2).
    """

    backend: str = "auto"
    strategy: str = "auto"
    # Ranks per tensor-parallel group. 0 means "the whole world", which is the basic version's
    # assumption (tp_size == world_size, no data parallelism alongside it). Validated against the
    # model: tensor parallelism partitions whole attention heads, so n_heads and d_ff must divide.
    tp_size: int = 0
    # Collectives block until every rank arrives. A finite timeout turns a dead peer into an
    # error instead of a job that hangs until a scheduler kills it.
    timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        errors = []
        if self.backend not in BACKENDS:
            errors.append(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.strategy not in STRATEGIES:
            errors.append(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if not _is_int(self.tp_size) or self.tp_size < 0:
            errors.append(
                f"tp_size must be a non-negative integer (0 = whole world), got {self.tp_size!r}"
            )
        if not _is_int(self.timeout_seconds) or self.timeout_seconds <= 0:
            errors.append(
                f"timeout_seconds must be a positive integer, got {self.timeout_seconds!r}"
            )
        if errors:
            raise ConfigError("invalid distributed config:\n  - " + "\n  - ".join(errors))
