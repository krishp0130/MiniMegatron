# MiniMegatron

[![CI](https://github.com/krishp0130/MiniMegatron/actions/workflows/ci.yml/badge.svg)](https://github.com/krishp0130/MiniMegatron/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![PyTorch 2.14.1](https://img.shields.io/badge/pytorch-2.14.1-ee4c2c)
![tests 468](https://img.shields.io/badge/tests-468%20passing-brightgreen)
![GPU verified](https://img.shields.io/badge/GPU-verified%20on%202%C3%97%20T4-success)

An educational, engineering-quality distributed transformer training framework inspired by
[NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — built to understand how large models
are actually trained, not to wrap a library that already does it.

A GPT-style decoder trained four ways — single device, **DDP**, **FSDP2**, and Megatron-style
**tensor parallelism** — plus a hand-written **Triton** kernel, a reliability suite and a
benchmark harness. The communication primitives are written out by hand with explicit
`autograd.Function`s and collectives, so every byte of it is readable.

**The organising rule:** every optimization has a reference implementation, a correctness test and
a reproducible benchmark. Nothing claims a speedup it has not measured, and nothing claims to work
until it has run.

### What that bought

| | |
|---|---|
| **All four strategies agree** | single, DDP, FSDP2 and TP produce *identical* losses on the tiny preset |
| **Triton kernel** | **3.4–3.7×** faster forward than the PyTorch reference, ~73% of a T4's peak bandwidth |
| **Scaling** | **1.77×** from 1 to 2 GPUs (DDP). TP is **38% slower than DDP** on PCIe — exactly as predicted before measuring |
| **Tests** | 468 on CPU, 484 on GPU, across 6 suites |
| **Mutation testing** | 60 deliberate bugs injected, 59 caught |
| **Bugs this found** | 21 recorded in [ISSUES.md](docs/ISSUES.md), several that normal testing missed |

New here? **[OVERVIEW.md](docs/OVERVIEW.md) describes the entire project in one file** — what was
built, why each choice was made, and what was measured. Otherwise start with
[architecture.md](docs/architecture.md) for how it fits together, or [ISSUES.md](docs/ISSUES.md)
if you would rather read what went wrong and why.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0. Setup | Repo, environment, packaging, context docs | **done** |
| 1. Functional GPT baseline | Model, synthetic data, single-device training, unit tests | **done** |
| 2. Distributed data parallel | DDP, gradient sync validation, rank-aware logging | **done** (benchmarks pending GPU) |
| 3. Memory optimization | BF16, FSDP2, sharded checkpoints, memory measurements | **done** (memory numbers pending GPU) |
| 4. Tensor parallelism | Megatron-style column/row-parallel MLP and attention | **done** (benchmarks pending GPU) |
| 5. GPU kernel optimization | Triton RMSNorm, forward and backward, benchmarks | **done** — verified on a Tesla T4: 68 tests pass, **3.4–3.7x** faster forward than the reference |
| 6. Reliability engineering | Regression and distributed tests, failure injection, CI | **done** |
| 7. Documentation | Architecture, experiments, results tables | **done** |

**468 tests pass on CPU**; **484 passed on 2x Tesla T4**. **60 deliberate bugs injected across
the milestones; 59 caught.** The Triton kernel has run on real hardware and works — see
[measured results](docs/PROGRESS.md#measured-results).

## Documentation

| Read this | For |
|---|---|
| **[OVERVIEW.md](docs/OVERVIEW.md)** | **the whole project in one self-contained file — start here** |
| [architecture.md](docs/architecture.md) | how it is put together, and *why the seams are where they are* |
| [distributed_training.md](docs/distributed_training.md) | DDP, FSDP2 and TP side by side; how correctness is established |
| [fsdp.md](docs/fsdp.md) | all-gather vs reduce-scatter, sharding units, mixed precision, checkpoints |
| [tensor_parallel.md](docs/tensor_parallel.md) | the `f`/`g` pair, column→row pairing, head partitioning |
| [testing.md](docs/testing.md) | the six suites, mutation testing, and the test-design rules learned the hard way |
| [benchmarks.md](docs/benchmarks.md) | the measurement method, measured results, and what is still pending |
| [PROGRESS.md](docs/PROGRESS.md) | current state, per-milestone session log, measured results |
| [ISSUES.md](docs/ISSUES.md) | every problem hit (#1–#21) and every design decision (D1–D32) |
| [SCOPE.md](docs/SCOPE.md) | the authoritative specification |
| [milestones/](docs/milestones/) | the design doc written before each milestone was implemented |

## Install

Requires Python 3.11+ (3.12 recommended — D1).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/check_env.py   # what this machine can and cannot run
```

`dev` is pytest, and it is all you need: core is torch + numpy + pyyaml. Three further extras exist
for features that are optional or not yet built, and nothing in the project imports them today —
`gpu` (Triton; Linux + NVIDIA only), `text` (tokenizers, for the deferred real-text dataset) and
`logging` (tensorboard, since metrics are JSONL).

Verified from a cold start: a venv that had never seen this project installs with the line above
and runs the full suite — 455 passed, 37 skipped.

`check_env.py` prints the same environment probe the benchmark harness records, so an environment
report and a benchmark record can never disagree about the machine.

## Quick start

```bash
.venv/bin/pytest -q                                     # the full test suite (CPU)
.venv/bin/python -m minimegatron.train --config configs/tiny.yaml
.venv/bin/python -m minimegatron.train --config configs/tiny.yaml --set train.max_steps=20
.venv/bin/python -m minimegatron.train --config configs/tiny.yaml \
    --resume runs/tiny/ckpt_000050.pt
```

Training writes JSONL metrics (`step`, `loss`, `lr`, `grad_norm`, `tokens_seen`,
`tokens_per_sec`) and checkpoints to `train.out_dir`. `--set section.key=value` overrides any
config value and is validated exactly as strictly as the YAML file. Resuming is bit-for-bit
identical to an uninterrupted run on CPU, which a test asserts.

### Multi-process (DDP)

```bash
# 2 ranks at the same global batch as the single-process run above: micro=8 x accum=1 x 2 = 16.
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    -m minimegatron.train --config configs/tiny.yaml --set train.grad_accum_steps=1

# See the all-reduce DDP hides, done by hand:
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29501 \
    -m minimegatron.distributed.collectives_demo
```

### FSDP2 (sharded parameters, gradients and optimizer state)

```bash
# strategy: single | ddp | fsdp (default `auto`: single on one rank, ddp on several).
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29502 \
    -m minimegatron.train --config configs/tiny.yaml \
    --set distributed.strategy=fsdp --set train.grad_accum_steps=1

# BF16 mixed precision (autocast under single/ddp, MixedPrecisionPolicy under fsdp):
.venv/bin/python -m minimegatron.train --config configs/tiny.yaml --set train.precision=bf16
```

FSDP writes two checkpoints per step: a sharded `ckpt_NNNNNN/` directory
(`torch.distributed.checkpoint`, nothing gathered, used for resume) and a consolidated
`ckpt_NNNNNN_full.pt` that loads into a plain single-device model. See [docs/fsdp.md](docs/fsdp.md)
for all-gather vs reduce-scatter, sharding units, and the mixed-precision details.

`train.precision` is *compute* precision over fp32 master weights; `model.dtype` is parameter
storage and stays `float32`. bf16 needs Ampere or newer on CUDA and errors clearly otherwise.

Rank 0 does all the logging and checkpointing, and single-device/DDP checkpoints store the
unwrapped model, so a 2-rank checkpoint loads straight into a single-device run. If `global_batch_size` is declared it is
checked against `micro_batch_size x grad_accum_steps x world_size` and the run refuses to start on
a mismatch — to keep the global batch fixed while adding ranks, reduce `grad_accum_steps`.

### Tensor parallelism (Megatron-style)

```bash
# Splits each layer's weight matrices, so every rank computes a slice of every layer.
# tp_size defaults to the whole world. The preset's global batch needs no change: TP ranks are
# not data-parallel ranks, so the data-parallel world (and the batch) is unaffected.
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29503 \
    -m minimegatron.train --config configs/tiny.yaml --set distributed.strategy=tp
```

Implemented by hand with explicit autograd functions and collectives rather than DTensor, so the
`f`/`g` conjugate pair, the column→row pairing and the head partitioning are all readable in
`distributed/tensor_parallel.py`. See [docs/tensor_parallel.md](docs/tensor_parallel.md).

`n_heads % tp_size` and `d_ff % tp_size` are validated at startup. TP writes a per-rank checkpoint
for resume plus a gathered `_full.pt` that loads into a plain single-device model.

### Testing and benchmarking

```bash
.venv/bin/pytest -q                              # everything (~2 min, includes 2-process gloo)
.venv/bin/pytest -q -m "not distributed"         # fast inner loop
.venv/bin/pytest tests/regression -q             # golden trajectories and structural invariants
.venv/bin/pytest tests/fault_injection -q        # controlled failures and their diagnostics

# Measure. Appends JSON to results/benchmarks.jsonl and rewrites a Markdown table.
.venv/bin/python -m minimegatron.benchmarks.run --config configs/benchmark.yaml --label fp32-cpu

# Compare against a versioned baseline. A warning, not a test: these numbers are noisy.
.venv/bin/python -m minimegatron.benchmarks.regression \
    --baseline results/baselines/tiny_cpu_fp32.json --current results/benchmarks.jsonl
```

Functional invariants are asserted hard; performance is compared against
[`results/baselines/`](results/baselines/) and only *warns*. Unmeasurable fields (GPU memory on a
CPU box) are recorded as `null`, never as 0 and never estimated. CI runs the CPU-safe suite,
including the 2-process gloo tests, on every push.

### Custom kernels

```bash
# The RMSNorm implementation is selectable. `auto` uses Triton on CUDA and the reference elsewhere.
.venv/bin/python -m minimegatron.train --config configs/tiny.yaml \
    --set model.norm_type=rmsnorm --set model.norm_backend=pytorch

# Kernel-level comparison (needs Triton + an NVIDIA GPU; refuses clearly otherwise).
.venv/bin/python -m minimegatron.benchmarks.kernels
```

> **Verified on a Tesla T4** (2026-11-09): the 34 Triton-parameterized tests that skip on a
> CPU-only machine all pass, and the fused kernel is **3.4–3.7x** faster than the PyTorch
> reference on the forward pass at ~235 GB/s — roughly 73% of the T4's peak bandwidth, which is
> the right target for a memory-bound op. No kernel bugs were found on its first ever execution.
> On a machine without Triton the tests skip and an explicit `norm_backend: triton` fails fast
> rather than falling back silently.

## Hardware support

| Environment | What runs |
|---|---|
| CPU (any OS) | Model correctness tests, smoke training, **multi-process DDP/FSDP2/TP correctness via the `gloo` backend** |
| Apple Silicon (MPS) | Single-device training |
| 1× NVIDIA GPU | Single-GPU training, BF16, Triton kernels, benchmarks |
| 2+ NVIDIA GPUs | NCCL DDP/FSDP2/TP training and scaling benchmarks |

Multi-GPU (NCCL) and Triton results are **pending verification on NVIDIA hardware** until
explicitly recorded in `results/`.

### Running the GPU-gated work

Everything that needs CUDA is packaged into one script, so a GPU session is copy-paste:

```bash
bash scripts/gpu_verify.sh      # refuses clearly, exit 2, with no GPU
```

It runs the Triton kernel tests, the full suite, single-GPU training, the kernel benchmark
(forward and backward), FP32 throughput, and FP32-vs-BF16 where the GPU supports bf16 — and on a
2-GPU machine it additionally runs the DDP/FSDP/TP comparisons.

For Google Colab, [`scripts/colab_gpu_verify.ipynb`](scripts/colab_gpu_verify.ipynb) handles the
clone, the install and the run. Two caveats it states up front:

- **Colab gives one GPU per runtime**, so the 1-vs-2-GPU, DDP-vs-FSDP-vs-TP and
  peak-memory-per-rank comparisons (DoD #4, #5) cannot be done there. Those need a 2-GPU
  machine — Kaggle's free `T4 x2`, or a rented instance — where the same script's multi-GPU
  section runs automatically.
- **A T4 has no *native* bf16** (sm_75). It still runs — torch counts emulated bf16, so
  `precision: bf16` is permitted — but it is **slower** than fp32 there (measured: 23,438 vs
  29,648 tokens/s). The FP32-vs-BF16 *speedup* needs an L4 or A100. An earlier version of this
  README claimed a T4 would error on bf16; it does not (ISSUES #21).

The kernel tests **passed on their first ever run** on a T4 x2 (ISSUES #16 closed). That run did
find three other bugs the CPU-only suite could not: tensor-parallel layers built on the CPU
(#19), three tests encoding "no GPU here" as an invariant (#20), and the bf16 claim above (#21).

## Environment notes

### macOS, for multi-process runs

`gloo` can hang on macOS when it binds to the wrong network interface. Pin it to loopback:

```bash
export GLOO_SOCKET_IFNAME=lo0
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 ...
```

FSDP2's `fully_shard` must be given an explicit CPU device mesh on Macs
(`init_device_mesh("cpu", (world_size,))`). Without one, it auto-detects MPS and fails.

## Layout

```text
configs/              tiny/small/medium presets, ddp/fsdp/tp_2gpu, benchmark
src/minimegatron/
  config.py           ConfigError, strict dict->dataclass, YAML loading
  experiment.py       ExperimentConfig: model/data/train/distributed/benchmark sections
  train.py            CLI entry point
  model/              config, attention (reference + sdpa), layers, transformer
  data/               config, synthetic datasets + the deterministic index stream
  training/           trainer, optimizer and schedule, checkpoint, metrics
  distributed/        setup, fsdp, tensor_parallel, checkpoint, collectives_demo
  kernels/            rmsnorm (Triton + dispatch to the PyTorch reference)
  benchmarks/         harness, run, kernels, regression
tests/                unit / numerical / distributed / kernels / fault_injection / regression
scripts/              check_env.py
results/baselines/    versioned performance baselines (other results/ output is gitignored)
docs/                 OVERVIEW, architecture, distributed_training, fsdp, tensor_parallel, testing,
                      benchmarks, SCOPE, PROGRESS, ISSUES, milestones/
.github/workflows/    CI: the CPU-safe suite, gloo tests included
```

## Known limitations

Stated plainly, each with its reason. The fuller treatment is in
[architecture.md](docs/architecture.md).

- **The Triton kernel has never been executed** (ISSUES #16) — no NVIDIA GPU here, no macOS Triton
  wheels. Written, not working.
- **No GPU measurement exists anywhere in this repository.** Every throughput and memory comparison
  the scope asks for is pending hardware; [benchmarks.md](docs/benchmarks.md) lists the exact
  command for each.
- **Dropout is unsupported under tensor parallelism** for comparisons (D18); a model-parallel RNG
  tracker is needed first. All equivalence tests use `dropout=0`.
- **2D TP+DP is refused, not silently mistrained** — a `tp_size` below the world size would leave
  the data-parallel dimension without gradient synchronization.
- **Embeddings are replicated under TP**, so the tiny preset's 9.7M token table (84% of it) is not
  sharded. Vocab-parallel embeddings are the obvious next win.
- **No real text data** — synthetic tokens only, so no sample from the model would be meaningful.
- **Single node.** No elastic training, no pipeline parallelism, no FlashAttention (excluded from
  the first phase by the scope).

### Future work, in the order I would do it

1. Run the Triton kernel and close ISSUES #16 — everything needed to judge it already exists.
2. One 2-GPU session to fill every pending benchmark table.
3. Vocab-parallel embedding and cross-entropy: the largest remaining TP memory win.
4. A model-parallel RNG tracker, unblocking dropout under TP.
5. Real text data and sampling.
6. 2D TP+DP composition — the `(dp, tp)` mesh and the strategy switch are the seams for it.

## Definition of Done

Assessed against [SCOPE.md §13](docs/SCOPE.md), item by item. The scope's own fallback clause
applies throughout: *"If two GPUs are unavailable, implement the distributed interfaces, test what
is possible locally (gloo on CPU), and explicitly mark multi-GPU verification as pending. Do not
fabricate test results."*

| # | Criterion | Status |
|---|---|---|
| 1 | A GPT model trains and reduces loss on a small task | **met** — overfit test drives loss from ≈ln V to < 0.1 |
| 2 | CPU-safe correctness tests pass | **met** — 376 passed, 37 skipped |
| 3 | Single-GPU training works on CUDA | **met** — 20 steps on a Tesla T4, 29,648 tokens/s |
| 4 | DDP trains correctly across two GPUs, gradients verified | **met** — gradients verified against a single-process reference on gloo; measured on 2x T4 at **1.77x** scaling |
| 5 | FSDP2 trains correctly with valid checkpointing | **met** — correctness, DCP round trip, resume and consolidation on gloo; measured on 2x T4 |
| 6 | TP=2 matches the single-device reference in forward and gradients | **met** — sharded gradients match the reference slice and replicated ones match in full on gloo; **measured on 2x T4** at 32,545 tokens/s |
| 7 | A Triton kernel passes numerical and gradient tests | **met** — 68 passed on a Tesla T4, no kernel bugs found |
| 8 | Automated validation catches deliberately introduced errors | **met** — 60 mutations injected, 59 caught |
| 9 | Benchmark results reproducible from documented configs | **met** — `scripts/gpu_verify.sh` produced every GPU figure recorded |
| 10 | Performance claims cite measured configurations | **met** — every number names its hardware, commit and method |
| 11 | Documented well enough for another engineer to run it | **met** — verified by a cold install, and by this GPU run following the documented path |

**All eleven met.** The only measurement still missing anywhere in this project is the
FP32-vs-BF16 *speedup*, which needs Ampere or newer — a T4 emulates bf16 and is slower with it.
No number appears here that was not measured on the machine named beside it.
