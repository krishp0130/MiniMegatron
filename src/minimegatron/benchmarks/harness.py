"""Benchmark harness: measure one configuration and report it honestly.

Three rules, from SCOPE.md section 9, and each of them is the difference between a number worth
recording and a number worth deleting:

1. **Warmup is discarded.** Early steps include allocator growth, kernel autotuning and lazy
   initialization.
2. **The device is synchronized at every boundary.** CUDA and MPS enqueue asynchronously, so an
   unsynchronized clock measures Python's enqueue rate, not compute.
3. **Repeats, reported as median and dispersion.** One measurement cannot tell a regression from a
   busy neighbour process.

Anything this machine cannot measure is recorded as `None` -- never 0, never estimated. On CPU and
MPS that means the GPU memory fields, because `torch.cuda.max_memory_allocated` is CUDA-only.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minimegatron.benchmarks.config import BenchmarkConfig
from minimegatron.training.trainer import Trainer, synchronize


def _git_commit() -> str | None:
    """The commit the measurement was taken at, so a recorded number is traceable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[3],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def environment_info() -> dict[str, Any]:
    """Hardware and library versions, as a dict. `scripts/check_env.py` prints this."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "gpu_name": None,
        "gpu_arch": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": None,
        "triton": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_arch"] = f"sm_{props.major}{props.minor}"
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        pass
    return info


@dataclass
class Measurement:
    """Median plus dispersion for one metric. Never a single number."""

    median: float
    minimum: float
    maximum: float
    iqr: float
    samples: list[float] = field(default_factory=list)

    @classmethod
    def of(cls, samples: list[float]) -> "Measurement":
        ordered = sorted(samples)
        if len(ordered) >= 4:
            spread = statistics.quantiles(ordered, n=4)
            iqr = spread[2] - spread[0]
        else:
            iqr = ordered[-1] - ordered[0]
        return cls(
            median=statistics.median(ordered),
            minimum=ordered[0],
            maximum=ordered[-1],
            iqr=iqr,
            samples=list(samples),
        )


def _peak_memory(device: torch.device) -> tuple[int | None, int | None]:
    """Peak allocated and reserved bytes, or `(None, None)` where it cannot be measured."""
    if device.type != "cuda":
        return None, None  # CUDA-only API; recorded as absent rather than as zero
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def benchmark(trainer: Trainer, cfg: BenchmarkConfig) -> dict[str, Any]:
    """Measure `trainer` and return a complete record.

    The trainer is stepped for real -- this measures the actual training loop, not a synthetic
    proxy -- so the model's weights move. Benchmarks therefore get their own trainer.
    """
    device = trainer.device
    train = trainer.cfg.train
    tokens_per_step = (
        train.micro_batch_size
        * train.grad_accum_steps
        * trainer.cfg.model.seq_len
        * trainer.dp_world_size
    )

    for _ in range(cfg.warmup_steps):
        trainer.train_step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)

    step_seconds: list[float] = []
    device_ms: list[float] = []
    for _ in range(cfg.repeats):
        synchronize(device)
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        wall_start = time.perf_counter()

        for _ in range(cfg.measure_steps):
            trainer.train_step()

        if device.type == "cuda":
            end.record()
        synchronize(device)
        wall = time.perf_counter() - wall_start

        step_seconds.append(wall / cfg.measure_steps)
        if device.type == "cuda":
            # Device-side time, which excludes Python overhead that wall clock includes.
            device_ms.append(start.elapsed_time(end) / cfg.measure_steps)

    allocated, reserved = _peak_memory(device)
    latency = Measurement.of([s * 1000 for s in step_seconds])
    throughput = Measurement.of([tokens_per_step / s for s in step_seconds])

    return {
        "label": cfg.label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        # --- what was measured ---
        "strategy": trainer.strategy,
        "precision": train.precision,
        "device": str(device),
        "world_size": trainer.dist.world_size,
        "dp_world_size": trainer.dp_world_size,
        "tp_size": trainer.tp_size,
        "micro_batch_size": train.micro_batch_size,
        "grad_accum_steps": train.grad_accum_steps,
        "effective_batch_size": train.micro_batch_size
        * train.grad_accum_steps
        * trainer.dp_world_size,
        "seq_len": trainer.cfg.model.seq_len,
        "tokens_per_step": tokens_per_step,
        "params": trainer.raw_model.num_params(),
        "non_embedding_params": trainer.raw_model.num_params(include_embeddings=False),
        # --- how it was measured ---
        "warmup_steps": cfg.warmup_steps,
        "measure_steps": cfg.measure_steps,
        "repeats": cfg.repeats,
        # --- results ---
        "step_latency_ms": dataclasses.asdict(latency),
        "tokens_per_sec": dataclasses.asdict(throughput),
        "device_step_ms": dataclasses.asdict(Measurement.of(device_ms)) if device_ms else None,
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_reserved_bytes": reserved,
        "environment": environment_info(),
    }


# --- output --------------------------------------------------------------------------------------

MARKDOWN_COLUMNS = [
    ("label", "Label"),
    ("strategy", "Strategy"),
    ("precision", "Precision"),
    ("device", "Device"),
    ("world_size", "Ranks"),
    ("tp_size", "TP"),
    ("effective_batch_size", "Batch"),
    ("seq_len", "Seq"),
    ("params", "Params"),
]


def _format_memory(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 2**20:,.0f} MiB"


def append_jsonl(record: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_markdown(records: list[dict[str, Any]]) -> str:
    """A table meant for pasting into PROGRESS.md, with dispersion shown, not hidden."""
    if not records:
        return "_No benchmark records._\n"

    headers = [title for _, title in MARKDOWN_COLUMNS] + [
        "tokens/s (median)",
        "tokens/s (IQR)",
        "step ms (median)",
        "peak alloc",
        "Commit",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for record in records:
        cells = [str(record.get(key, "")) for key, _ in MARKDOWN_COLUMNS]
        cells += [
            f"{record['tokens_per_sec']['median']:,.1f}",
            f"{record['tokens_per_sec']['iqr']:,.1f}",
            f"{record['step_latency_ms']['median']:,.2f}",
            _format_memory(record["peak_memory_allocated_bytes"]),
            str(record.get("commit") or ""),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    note = (
        "\n_Memory is `n/a` where it cannot be measured: "
        "`torch.cuda.max_memory_allocated` is CUDA-only. Never estimated._\n"
    )
    return "\n".join(lines) + "\n" + note


def write_markdown(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Benchmark results\n\n" + to_markdown(records))
    return path
