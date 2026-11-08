"""The benchmark harness, the record it produces, and performance-regression comparison."""

import json

import pytest
import torch

from minimegatron.benchmarks.config import BenchmarkConfig
from minimegatron.benchmarks.harness import (
    Measurement,
    append_jsonl,
    benchmark,
    environment_info,
    load_jsonl,
    to_markdown,
    write_markdown,
)
from minimegatron.benchmarks.regression import DEFAULT_THRESHOLD, compare, format_report
from minimegatron.config import ConfigError
from minimegatron.data.config import DataConfig
from minimegatron.experiment import ExperimentConfig
from minimegatron.training.config import TrainConfig
from minimegatron.training.trainer import Trainer

# Every field SCOPE.md section 9 asks the harness to report.
REQUIRED_FIELDS = {
    "tokens_per_sec",
    "step_latency_ms",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "params",
    "effective_batch_size",
    "precision",
    "world_size",
    "strategy",
    "tp_size",
    "seq_len",
    "environment",
}


def tiny_trainer(nano_config, out_dir, **train_overrides) -> Trainer:
    train = dict(
        max_steps=100,
        micro_batch_size=4,
        grad_accum_steps=1,
        lr=1e-3,
        device="cpu",
        out_dir=str(out_dir),
        log_interval=0,
        eval_interval=0,
    )
    train.update(train_overrides)
    cfg = ExperimentConfig(
        model=nano_config,
        data=DataConfig(dataset="random", n_items=256, seed=0),
        train=TrainConfig(**train),
    )
    return Trainer(cfg, out_dir=out_dir, echo=False)


FAST = BenchmarkConfig(warmup_steps=1, measure_steps=2, repeats=2)


# --- statistics ---------------------------------------------------------------------------------


def test_measurement_reports_median_and_dispersion():
    m = Measurement.of([10.0, 12.0, 11.0, 50.0])
    assert m.median == pytest.approx(11.5)
    assert m.minimum == 10.0 and m.maximum == 50.0
    assert m.iqr > 0  # the outlier is visible as spread, not hidden in a mean
    assert m.samples == [10.0, 12.0, 11.0, 50.0]  # raw samples kept for re-analysis


def test_measurement_handles_few_samples():
    """Fewer than four samples cannot give quartiles; fall back to the full span."""
    m = Measurement.of([3.0, 5.0])
    assert m.median == pytest.approx(4.0)
    assert m.iqr == pytest.approx(2.0)


# --- environment --------------------------------------------------------------------------------


def test_environment_info_reports_hardware_and_versions():
    info = environment_info()
    assert {"platform", "processor", "python", "torch", "cuda_available", "mps_available"} <= set(
        info
    )
    assert info["torch"] == torch.__version__


def test_gpu_fields_are_none_rather_than_zero_without_a_gpu():
    """An unmeasurable field must be absent, not a plausible-looking 0."""
    info = environment_info()
    if not torch.cuda.is_available():
        assert info["gpu_name"] is None
        assert info["gpu_arch"] is None
        assert info["bf16_supported"] is None


# --- the record ---------------------------------------------------------------------------------


def test_benchmark_record_has_every_required_field(nano_config, tmp_path):
    record = benchmark(tiny_trainer(nano_config, tmp_path), FAST)
    assert REQUIRED_FIELDS <= set(record)
    assert record["tokens_per_sec"]["median"] > 0
    assert record["step_latency_ms"]["median"] > 0


def test_record_describes_how_it_was_measured(nano_config, tmp_path):
    """A number without its method is not reproducible."""
    record = benchmark(tiny_trainer(nano_config, tmp_path), FAST)
    assert record["warmup_steps"] == 1
    assert record["measure_steps"] == 2
    assert record["repeats"] == 2
    assert len(record["tokens_per_sec"]["samples"]) == 2  # one per repeat
    assert record["timestamp"] and "environment" in record


def test_warmup_steps_are_run_and_discarded(nano_config, tmp_path):
    """The trainer must advance by warmup + measure x repeats, with only the latter timed."""
    trainer = tiny_trainer(nano_config, tmp_path)
    benchmark(trainer, FAST)
    assert trainer.step == FAST.warmup_steps + FAST.measure_steps * FAST.repeats


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_memory_fields_are_none_on_cpu(nano_config, tmp_path):
    record = benchmark(tiny_trainer(nano_config, tmp_path), FAST)
    assert record["peak_memory_allocated_bytes"] is None
    assert record["peak_memory_reserved_bytes"] is None
    assert record["device_step_ms"] is None  # CUDA events only


def test_throughput_accounts_for_accumulation_and_data_parallel_width(nano_config, tmp_path):
    record = benchmark(
        tiny_trainer(nano_config, tmp_path, micro_batch_size=2, grad_accum_steps=4), FAST
    )
    assert record["tokens_per_step"] == 2 * 4 * nano_config.seq_len
    assert record["effective_batch_size"] == 8


# --- output -------------------------------------------------------------------------------------


def test_markdown_table_shows_dispersion_and_flags_unmeasurable_memory(nano_config, tmp_path):
    record = benchmark(tiny_trainer(nano_config, tmp_path), FAST)
    table = to_markdown([record])
    assert "tokens/s (median)" in table and "tokens/s (IQR)" in table
    if record["peak_memory_allocated_bytes"] is None:
        assert "n/a" in table
    assert "Never estimated" in table  # the honesty note travels with the table


def test_empty_table_says_so():
    assert "No benchmark records" in to_markdown([])


def test_jsonl_round_trip_and_append(nano_config, tmp_path):
    record = benchmark(tiny_trainer(nano_config, tmp_path), FAST)
    path = tmp_path / "out" / "bench.jsonl"
    append_jsonl(record, path)
    append_jsonl(record, path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["tokens_per_sec"]["median"] == record["tokens_per_sec"]["median"]


def test_loading_a_missing_jsonl_is_empty(tmp_path):
    assert load_jsonl(tmp_path / "nope.jsonl") == []


def test_write_markdown_creates_the_file(nano_config, tmp_path):
    record = benchmark(tiny_trainer(nano_config, tmp_path), FAST)
    path = write_markdown([record], tmp_path / "nested" / "bench.md")
    assert path.is_file() and "Benchmark results" in path.read_text()


# --- regression comparison ----------------------------------------------------------------------


def make_record(tokens: float, latency: float, memory: int | None = None) -> dict:
    return {
        "tokens_per_sec": {"median": tokens},
        "step_latency_ms": {"median": latency},
        "peak_memory_allocated_bytes": memory,
        "peak_memory_reserved_bytes": None,
    }


def test_identical_records_show_no_regression():
    findings = compare(make_record(100.0, 10.0), make_record(100.0, 10.0))
    assert findings and not any(f["regressed"] for f in findings)


def test_throughput_drop_is_a_regression_but_a_rise_is_not():
    worse = compare(make_record(100.0, 10.0), make_record(50.0, 10.0))
    assert next(f for f in worse if f["metric"] == "tokens_per_sec")["regressed"]

    better = compare(make_record(100.0, 10.0), make_record(200.0, 10.0))
    assert not next(f for f in better if f["metric"] == "tokens_per_sec")["regressed"]


def test_latency_and_memory_invert_the_sign():
    """Lower is better for latency and memory, so a *rise* is the regression."""
    findings = compare(make_record(100.0, 10.0, 1000), make_record(100.0, 20.0, 2000))
    assert next(f for f in findings if f["metric"] == "step_latency_ms")["regressed"]
    assert next(f for f in findings if f["metric"] == "peak_memory_allocated_bytes")["regressed"]

    improved = compare(make_record(100.0, 20.0, 2000), make_record(100.0, 10.0, 1000))
    assert not any(f["regressed"] for f in improved)


def test_change_within_the_threshold_is_not_a_regression():
    findings = compare(make_record(100.0, 10.0), make_record(95.0, 10.0), threshold=10.0)
    tokens = next(f for f in findings if f["metric"] == "tokens_per_sec")
    assert tokens["change_percent"] == pytest.approx(-5.0)
    assert not tokens["regressed"]


def test_unmeasurable_metrics_are_skipped_not_compared_as_zero():
    """Silence beats a fabricated comparison when one side has no number."""
    findings = compare(make_record(100.0, 10.0, None), make_record(100.0, 10.0, 2000))
    assert all(f["metric"] != "peak_memory_allocated_bytes" for f in findings)


def test_report_names_the_threshold_and_the_verdicts():
    report = format_report(compare(make_record(100.0, 10.0), make_record(50.0, 10.0)), 10.0)
    assert "REGRESSION" in report and "Threshold: 10.0%" in report


def test_report_handles_nothing_comparable():
    assert "No comparable metrics" in format_report([], DEFAULT_THRESHOLD)


# --- configuration ------------------------------------------------------------------------------


def test_benchmark_config_defaults_encode_the_method():
    cfg = BenchmarkConfig()
    assert cfg.warmup_steps > 0 and cfg.measure_steps > 1 and cfg.repeats > 1
    assert cfg.total_steps == cfg.warmup_steps + cfg.measure_steps * cfg.repeats


def test_zero_warmup_is_allowed_but_zero_repeats_is_not():
    assert BenchmarkConfig(warmup_steps=0).warmup_steps == 0
    with pytest.raises(ConfigError, match="repeats must be a positive integer"):
        BenchmarkConfig(repeats=0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"measure_steps": 0}, "measure_steps must be a positive integer"),
        ({"warmup_steps": -1}, "warmup_steps must be a non-negative integer"),
        ({"label": 3}, "label must be a string"),
    ],
)
def test_invalid_benchmark_config_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        BenchmarkConfig(**overrides)


def test_benchmark_section_loads_from_yaml():
    from minimegatron.experiment import load_config
    from tests.conftest import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "benchmark.yaml")
    assert cfg.benchmark.repeats == 3 and cfg.benchmark.warmup_steps == 3
