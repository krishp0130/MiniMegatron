# Milestone 6 — Reliability Engineering

**Goal:** make the framework's correctness *and* its performance claims hold up over time, and make
its failures legible. Four milestones of optimizations are now checked against references, but
nothing yet (a) pins behaviour against recorded baselines across commits, (b) tests what happens
when things go *wrong*, (c) measures performance in a way worth recording, or (d) runs any of it
automatically.

**Not in this milestone:** Triton (M5, needs a GPU), the final documentation pass (M7), and any
GPU measurement (*pending hardware*, ISSUES #3).

**Where it runs:** entirely on this machine. This is the first milestone since M1 with no part
blocked on hardware — CPU throughput is genuinely measurable with warmup, repeats and dispersion;
only GPU memory and multi-GPU scaling are not.

## Exit criteria

1. `pytest -q` passes, including the new regression and fault-injection suites.
2. A deterministic regression suite pins functional invariants: golden loss trajectories, exact
   parameter counts, state-dict key sets, and cross-strategy agreement.
3. Failure injection covers every case SCOPE.md section 8 lists: CUDA OOM, invalid configuration
   (including invalid `tp_size`), corrupt checkpoints, missing files, unsupported GPU features, and
   **a worker dying mid-run**. Each produces a clear diagnostic and nothing hangs.
4. `python -m minimegatron.benchmarks.run --config configs/benchmark.yaml` works, emitting JSON
   *and* a Markdown table, with warmup, repeats, median and dispersion.
5. Performance regressions are a **warning against a stored baseline**, never a unit-test failure.
6. CI runs the CPU-safe suite on every push.
7. ISSUES #8 closed: validation loss is comparable across world sizes.
8. The documented CLI from SCOPE.md section 10 actually exists, including the strategy configs.

## Build order

| Step | Files | Tests |
|---|---|---|
| 1. Fix ISSUES #8 | `training/trainer.py` | `tests/distributed/test_eval_set.py` |
| 2. Benchmark harness | `benchmarks/{harness,run,regression}.py`, `configs/benchmark.yaml` | `tests/unit/test_benchmarks.py` |
| 3. Regression suite | `tests/regression/`, `results/baselines/` | itself |
| 4. Failure injection | `tests/fault_injection/` | itself |
| 5. CI + strategy configs | `.github/workflows/ci.yml`, `configs/*_2gpu.yaml` | CI run |

---

## Step 1 — Validation loss comparable across world sizes (ISSUES #8)

Today `eval_steps` counts batches **per rank**, so a 2-rank run evaluates twice the data and
reports a different number (10.8641 vs 10.8636 on the tiny preset). Training loss is exactly
comparable; validation is not, which makes it useless for exactly the cross-strategy comparison
this project is built to make.

**Fix.** Treat `eval_steps` as a count of **global** evaluation batches, and distribute those
batches across the data-parallel ranks rather than giving each rank its own shard of samples:

- the validation stream becomes **unsharded**, so micro-batch *j* is the same samples on every rank
- rank *r* evaluates `j = r, r + dp, r + 2·dp, …` while `j < eval_steps`
- the loss **sum** and the **count** are both all-reduced, and the mean is taken globally

The evaluation set is then exactly `eval_steps × micro_batch_size` samples whatever the world size,
and a 1-rank run is unchanged (it evaluates `j = 0 … eval_steps-1`). Reducing a sum and a count
rather than a mean-of-means is what makes it correct when `dp_world_size` does not divide
`eval_steps` — ranks then evaluate different numbers of batches, and averaging their means would
silently weight the short rank too heavily. That is the same trap as the unequal-shard problem in
D24, one level up.

## Step 2 — Benchmark harness

`benchmarks/harness.py` measures one configuration and returns a record with every field SCOPE.md
section 9 asks for: throughput (tokens/s), step latency, peak allocated and reserved memory,
parameter count, hardware and architecture, library versions, effective batch size, precision,
world size, strategy, `tp_size`.

Three things make the numbers worth recording rather than merely produced:

- **Warmup, discarded.** The first steps include allocator growth, kernel autotuning and lazy
  initialization. Timing them measures startup.
- **Device synchronization at boundaries**, reusing M1's `synchronize`: CUDA and MPS enqueue
  asynchronously, so an unsynchronized clock measures Python's enqueue rate. On CUDA, `cuda.Event`
  additionally gives device-side time rather than wall clock.
- **Repeats with median and dispersion**, never a single number. The harness reports median, min,
  max and IQR across repeats, because a single measurement cannot distinguish a regression from a
  noisy neighbour process.

Fields that cannot be measured on the current device are recorded as `null`, never as 0 and never
estimated — `max_memory_allocated` is CUDA-only, so on CPU and MPS those fields are explicitly
absent.

Output goes to JSON (for the baseline store and for machine comparison) *and* a Markdown table
(for pasting into PROGRESS.md), as section 9 requires.

## Step 3 — Regression suite

Two different things, deliberately kept apart, because conflating them is how performance tests
become flaky and then ignored:

### Functional regressions — hard assertions

`tests/regression/` pins deterministic facts that must never change silently:

- **Golden loss trajectories.** A fixed nano workload's first N losses, recorded to full precision.
  Legitimate because `torch==2.14.1` is pinned exactly (D1) and the data is index-deterministic
  (D9); a bump to either is expected to require re-recording, and the test says so in its failure
  message.
- **Exact parameter counts** for every preset, and non-embedding counts.
- **State-dict key sets**, so a checkpoint written today still loads tomorrow.
- **Cross-strategy agreement**: `single` and the strategies that can run in one process produce
  identical losses.

### Performance regressions — warnings against a stored baseline

Baselines live in `results/baselines/*.json`, **separate from the test suite** (section 8:
"Store baseline performance data separately"). `benchmarks/regression.py` compares a fresh run
against a baseline and reports a percentage change per metric, warning past a configurable
threshold. It is a CLI tool, not a pytest test, and `--fail-on-regression` is opt-in for CI use.

The reason for the split is stated plainly in the scope: *"Avoid treating noisy performance
measurements as deterministic unit tests."* A throughput assertion in `pytest` fails on a busy
laptop and gets deleted within a week.

## Step 4 — Failure injection

`tests/fault_injection/`, covering every case section 8 names. Each asserts a **clear diagnostic**,
not merely that something raised:

| Failure | How it is tested |
|---|---|
| Invalid configuration | unknown keys, bad values, mismatched `global_batch_size`, **invalid `tp_size`** (indivisible heads/`d_ff`, and `tp_size != world_size`) |
| Corrupt checkpoint | truncated file, wrong format version, missing keys, a directory with no `meta.pt` |
| Missing files | nonexistent `--config`, nonexistent `--resume` |
| Unsupported GPU features | `device: cuda` without CUDA, `precision: bf16` on a device without bf16 |
| CUDA OOM | the error *path*, by injecting `torch.cuda.OutOfMemoryError` — the handler is what we own, not the hardware |
| **Worker death mid-run** | a real `torchrun` with a rank that exits early: the job must fail fast with a non-zero code and a diagnostic, and **must not hang** |

The worker-death test is the one with teeth, and the one most likely to hang the suite if the
framework gets it wrong — which is precisely why it exists. Every distributed fault test uses
subprocess isolation and a hard Python-side timeout (ISSUES #1: macOS has no `timeout` binary, and
gloo hangs silently when misconfigured).

## Step 5 — CI and the documented CLI

`.github/workflows/ci.yml` on `ubuntu-latest`, Python 3.12, CPU-only torch, running the suite with
GPU-marked tests deselected. The 2-process gloo tests *do* run — they are the project's main
correctness evidence and they work on CPU (D2). Note gloo's loopback interface is `lo` on Linux and
`lo0` on macOS; the test harness already picks per platform.

SCOPE.md section 10 documents `configs/{ddp_2gpu,fsdp_2gpu,tp_2gpu}.yaml` and
`configs/benchmark.yaml`, none of which exist yet. A documented CLI that does not run is a
documentation bug, so this milestone creates them.

## Known traps

| Trap | Handling |
|---|---|
| Golden values that drift with library versions | torch pinned exactly (D1); the failure message says to re-record deliberately, not to loosen the test |
| Performance assertions in unit tests | kept out of pytest entirely; warnings against a stored baseline instead |
| A fault test that hangs instead of failing | subprocess isolation plus a hard Python-side timeout on every distributed fault test |
| Fabricated GPU numbers | unmeasurable fields are `null`, never 0; PROGRESS records only what ran |
| CI that silently skips the real tests | assert the collected test count in CI, so a misconfigured marker filter is visible |
