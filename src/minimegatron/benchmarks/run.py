"""Benchmark entry point.

    python -m minimegatron.benchmarks.run --config configs/benchmark.yaml
    python -m minimegatron.benchmarks.run --config configs/benchmark.yaml --label fp32-cpu

Appends one JSON record per run to a JSONL file and rewrites a Markdown table from the whole file,
so the comparisons SCOPE.md section 9 asks for (FP32 vs BF16, 1 vs 2 ranks, DDP vs FSDP vs TP) are
made by running this repeatedly with different `--set` overrides and reading one table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from minimegatron.benchmarks.harness import (
    append_jsonl,
    benchmark,
    load_jsonl,
    to_markdown,
    write_markdown,
)
from minimegatron.config import ConfigError, load_yaml
from minimegatron.distributed.setup import destroy_distributed, init_distributed
from minimegatron.experiment import config_from_dict
from minimegatron.train import apply_overrides
from minimegatron.training.trainer import Trainer

DEFAULT_JSONL = "results/benchmarks.jsonl"
DEFAULT_MARKDOWN = "results/benchmarks.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minimegatron.benchmarks.run", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument("--label", default=None, help="override benchmark.label for this run")
    parser.add_argument("--out", default=DEFAULT_JSONL, help=f"JSONL file (default {DEFAULT_JSONL})")
    parser.add_argument(
        "--markdown", default=DEFAULT_MARKDOWN, help=f"Markdown table (default {DEFAULT_MARKDOWN})"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value",
        help="override a config value; repeatable",
    )
    parser.add_argument(
        "--no-store", action="store_true", help="print the record instead of writing any file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw = apply_overrides(load_yaml(args.config), args.overrides)
        cfg = config_from_dict(raw, source=args.config)
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2

    bench = cfg.benchmark
    if args.label is not None:
        bench = dataclasses.replace(bench, label=args.label)

    dist_info = init_distributed(cfg.distributed)
    try:
        # A fresh trainer: benchmarking steps the real training loop, so it moves weights.
        trainer = Trainer(cfg, dist_info=dist_info, out_dir=cfg.train.out_dir, echo=False)
        record = benchmark(trainer, bench)
    finally:
        destroy_distributed()

    # Only rank 0 writes, for the same reason it owns every other side effect.
    if not dist_info.is_main:
        return 0

    if args.no_store:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    jsonl = append_jsonl(record, args.out)
    markdown = write_markdown(load_jsonl(jsonl), args.markdown)
    print(to_markdown([record]))
    print(f"appended to {jsonl}\nrewrote {markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
