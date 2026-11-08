"""Compare a benchmark record against a stored baseline and warn on regressions.

Deliberately a **CLI tool, not a pytest test**. SCOPE.md section 8 is explicit: *"Avoid treating
noisy performance measurements as deterministic unit tests."* A throughput assertion inside the
test suite fails whenever the machine is busy, and a test that cries wolf gets deleted within a
week. Functional invariants are asserted hard, in `tests/regression/`; performance is a warning
against a baseline, here.

    python -m minimegatron.benchmarks.regression \\
        --baseline results/baselines/tiny_cpu.json --current results/benchmarks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from minimegatron.benchmarks.harness import load_jsonl

# Metrics where a *lower* number is better, so the sign of an acceptable change flips.
LOWER_IS_BETTER = {"step_latency_ms", "peak_memory_allocated_bytes", "peak_memory_reserved_bytes"}
DEFAULT_THRESHOLD = 10.0  # percent


def _value(record: dict[str, Any], metric: str) -> float | None:
    raw = record.get(metric)
    if raw is None:
        return None
    return float(raw["median"]) if isinstance(raw, dict) else float(raw)


def compare(
    baseline: dict[str, Any], current: dict[str, Any], threshold: float = DEFAULT_THRESHOLD
) -> list[dict[str, Any]]:
    """Percentage change per metric, flagged when it exceeds `threshold` in the bad direction."""
    findings = []
    for metric in (
        "tokens_per_sec",
        "step_latency_ms",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    ):
        before, after = _value(baseline, metric), _value(current, metric)
        if before is None or after is None or before == 0:
            continue  # unmeasurable on one side; silence beats a fabricated comparison
        change = (after - before) / before * 100.0
        # A regression is a change in the bad direction, whichever direction that is.
        regressed = (change < -threshold) if metric not in LOWER_IS_BETTER else (change > threshold)
        findings.append(
            {
                "metric": metric,
                "baseline": before,
                "current": after,
                "change_percent": change,
                "regressed": regressed,
            }
        )
    return findings


def format_report(findings: list[dict[str, Any]], threshold: float) -> str:
    if not findings:
        return "No comparable metrics (nothing measurable on both sides)."
    lines = [f"| Metric | Baseline | Current | Change | Verdict |", "|---|---|---|---|---|"]
    for f in findings:
        verdict = "REGRESSION" if f["regressed"] else "ok"
        lines.append(
            f"| {f['metric']} | {f['baseline']:,.2f} | {f['current']:,.2f} "
            f"| {f['change_percent']:+.1f}% | {verdict} |"
        )
    lines.append(f"\nThreshold: {threshold:.1f}% in the unfavourable direction.")
    return "\n".join(lines)


def _load_one(path: str | Path, label: str | None) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records = load_jsonl(path)
        if label:
            records = [r for r in records if r.get("label") == label]
        if not records:
            raise FileNotFoundError(f"no records in {path}" + (f" labelled {label!r}" if label else ""))
        return records[-1]  # most recent
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimegatron.benchmarks.regression", description=__doc__)
    parser.add_argument("--baseline", required=True, help="baseline JSON or JSONL")
    parser.add_argument("--current", required=True, help="current JSON or JSONL")
    parser.add_argument("--label", default=None, help="select records by label")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero on a regression; off by default, since these measurements are noisy",
    )
    args = parser.parse_args(argv)

    try:
        baseline = _load_one(args.baseline, args.label)
        current = _load_one(args.current, args.label)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"could not load records: {e}", file=sys.stderr)
        return 2

    findings = compare(baseline, current, args.threshold)
    print(format_report(findings, args.threshold))

    regressions = [f for f in findings if f["regressed"]]
    if regressions:
        names = ", ".join(f["metric"] for f in regressions)
        print(f"\nWARNING: {len(regressions)} regression(s): {names}", file=sys.stderr)
        return 1 if args.fail_on_regression else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
