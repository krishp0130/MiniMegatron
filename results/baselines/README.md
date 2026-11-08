# Performance baselines

Versioned reference measurements, compared against by
`python -m minimegatron.benchmarks.regression`. Everything else under `results/` is a
gitignored ad-hoc run.

Baselines are **machine-specific**. Each file carries its own `environment` block (platform, CPU,
torch version, GPU name and architecture where applicable) and the `commit` it was taken at, so a
comparison across different hardware is visible as such rather than silently misleading.

Performance comparison is a **warning**, never a test assertion — SCOPE.md section 8: *"Avoid
treating noisy performance measurements as deterministic unit tests."* Functional invariants are
asserted hard in `tests/regression/`.

Re-record a baseline deliberately, with a commit message saying why, after a change that is
*expected* to move performance. Never re-record to silence a warning you have not explained.

| File | What it measures |
|---|---|
| `tiny_cpu_fp32.json` | tiny preset, single process, FP32, Apple M4 Pro CPU |
