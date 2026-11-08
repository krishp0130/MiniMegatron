"""Kernel-level benchmark: the fused Triton RMSNorm against the PyTorch reference.

    python -m minimegatron.benchmarks.kernels

RMSNorm is memory-bound -- the arithmetic is trivial and the time is traffic -- so the headline
number is **achieved bandwidth**, not FLOPs. The reference makes roughly six passes over the row
(square, mean, add, rsqrt, multiply, multiply, each reading and writing HBM); the fused kernel
makes one read and one write. The ratio of those is what this is measuring.

Requires an NVIDIA GPU and Triton, and says so clearly rather than producing a number from the
wrong device (ISSUES #3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from typing import Any

import torch

from minimegatron.benchmarks.harness import Measurement, append_jsonl, environment_info
from minimegatron.kernels.rmsnorm import TRITON_AVAILABLE, rms_norm_triton, triton_usable
from minimegatron.model.layers import rms_norm

# (rows, cols): row counts spanning a realistic (batch x seq) range, widths including the tiny
# preset's non-power-of-two 192 and GPT-2 small's 768.
DEFAULT_SHAPES = [
    (4096, 192),
    (4096, 768),
    (4096, 1024),
    (16384, 768),
    (16384, 2048),
    (65536, 768),
]
# Read x once + write y once. The reference's extra passes are what the comparison exposes.
BYTES_PER_ELEMENT_IDEAL = 2


def _time_callable(fn, warmup: int, iters: int, repeats: int) -> Measurement:
    """Median per-call seconds, with CUDA synchronized at every boundary."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) / iters)
    return Measurement.of(samples)


def _bandwidth_gbps(rows: int, cols: int, seconds: float, element_size: int, passes: int) -> float:
    return rows * cols * element_size * passes / seconds / 1e9


def benchmark_shape(
    rows: int, cols: int, dtype: torch.dtype, warmup: int, iters: int, repeats: int, backward: bool
) -> dict[str, Any]:
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=backward)
    weight = torch.randn(cols, device="cuda", dtype=dtype, requires_grad=backward)
    grad = torch.randn(rows, cols, device="cuda", dtype=dtype) if backward else None

    def make(fn):
        if not backward:
            return lambda: fn(x, weight)

        def run():
            x.grad = weight.grad = None
            fn(x, weight).backward(grad)

        return run

    reference = _time_callable(make(rms_norm), warmup, iters, repeats)
    fused = _time_callable(make(rms_norm_triton), warmup, iters, repeats)

    element_size = torch.tensor([], dtype=dtype).element_size()
    return {
        "rows": rows,
        "cols": cols,
        "dtype": str(dtype).replace("torch.", ""),
        "direction": "fwd+bwd" if backward else "fwd",
        "reference_ms": dataclasses.asdict(Measurement.of([s * 1e3 for s in reference.samples])),
        "triton_ms": dataclasses.asdict(Measurement.of([s * 1e3 for s in fused.samples])),
        "speedup": reference.median / fused.median,
        "triton_gbps": _bandwidth_gbps(
            rows, cols, fused.median, element_size, BYTES_PER_ELEMENT_IDEAL
        ),
        "reference_gbps": _bandwidth_gbps(
            rows, cols, reference.median, element_size, BYTES_PER_ELEMENT_IDEAL
        ),
    }


def to_markdown(records: list[dict[str, Any]]) -> str:
    if not records:
        return "_No kernel benchmark records._\n"
    header = (
        "| rows | cols | dtype | dir | reference ms | triton ms | speedup | triton GB/s |"
    )
    lines = [header, "|" + "---|" * 8]
    for r in records:
        lines.append(
            f"| {r['rows']} | {r['cols']} | {r['dtype']} | {r['direction']} "
            f"| {r['reference_ms']['median']:.3f} | {r['triton_ms']['median']:.3f} "
            f"| {r['speedup']:.2f}x | {r['triton_gbps']:.0f} |"
        )
    lines.append(
        "\n_GB/s counts one read and one write of the activation (the fused ideal). The reference "
        "makes several passes, which is what the speedup column measures._\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimegatron.benchmarks.kernels", description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--backward", action="store_true", help="time forward+backward")
    parser.add_argument("--out", default=None, help="append JSON records to this JSONL file")
    args = parser.parse_args(argv)

    if not triton_usable():
        print(
            "kernel benchmarks need Triton and an NVIDIA GPU.\n"
            f"  triton installed: {TRITON_AVAILABLE}\n"
            f"  cuda available:   {torch.cuda.is_available()}\n"
            "Triton has no macOS wheels; run this on a Linux + NVIDIA machine (ISSUES #3).",
            file=sys.stderr,
        )
        return 2

    dtype = getattr(torch, args.dtype)
    records = [
        benchmark_shape(rows, cols, dtype, args.warmup, args.iters, args.repeats, args.backward)
        for rows, cols in DEFAULT_SHAPES
    ]
    environment = environment_info()
    print(to_markdown(records))
    print(f"\nGPU: {environment['gpu_name']} ({environment['gpu_arch']}), torch {environment['torch']}")

    if args.out:
        for record in records:
            append_jsonl({**record, "environment": environment}, args.out)
        print(f"appended {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
