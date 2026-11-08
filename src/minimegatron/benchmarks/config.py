"""Benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass

from minimegatron.config import ConfigError


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class BenchmarkConfig:
    """How to measure, as opposed to what to measure.

    The defaults encode the three things that make a number worth recording: discard warmup,
    measure a run of steps rather than one, and repeat so dispersion can be reported.
    """

    # Discarded. The first steps include allocator growth, kernel autotuning and lazy init;
    # timing them measures startup, not training.
    warmup_steps: int = 3
    measure_steps: int = 10  # steps per repeat
    repeats: int = 3  # a single measurement cannot distinguish a regression from a noisy neighbour
    label: str = ""  # free-form tag carried into the record

    def __post_init__(self) -> None:
        errors = []
        for name in ("measure_steps", "repeats"):
            v = getattr(self, name)
            if not _is_int(v) or v <= 0:
                errors.append(f"{name} must be a positive integer, got {v!r}")
        if not _is_int(self.warmup_steps) or self.warmup_steps < 0:
            errors.append(f"warmup_steps must be a non-negative integer, got {self.warmup_steps!r}")
        if not isinstance(self.label, str):
            errors.append(f"label must be a string, got {self.label!r}")
        if errors:
            raise ConfigError("invalid benchmark config:\n  - " + "\n  - ".join(errors))

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.measure_steps * self.repeats
