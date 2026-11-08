# MiniMegatron

An educational, engineering-quality distributed transformer training framework inspired by
[NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM): a GPT-style decoder, a training
engine, DDP and FSDP2 data parallelism, Megatron-style tensor parallelism, a custom Triton
kernel, and a correctness/benchmarking
harness.

> **Guiding principle:** every optimization has a reference implementation, a correctness test,
> and a reproducible benchmark. No performance claim appears here without a measured config.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0. Setup | Repo, environment, packaging, context docs | **done** |
| 1. Functional GPT baseline | Model, synthetic data, single-device training, unit tests | **done** |
| 2. Distributed data parallel | DDP, gradient sync validation, rank-aware logging | **done** (benchmarks pending GPU) |
| 3. Memory optimization | BF16, FSDP2, sharded checkpoints, memory measurements | **done** (memory numbers pending GPU) |
| 4. Tensor parallelism | Megatron-style column/row-parallel MLP and attention | **done** (benchmarks pending GPU) |
| 5. GPU kernel optimization | Triton RMSNorm, forward and backward, benchmarks | not started |
| 6. Reliability engineering | Regression and distributed tests, failure injection, CI | not started |
| 7. Documentation | Architecture, experiments, results tables | not started |

Detailed status: [docs/PROGRESS.md](docs/PROGRESS.md) · Scope: [docs/SCOPE.md](docs/SCOPE.md) ·
Known issues and decisions: [docs/ISSUES.md](docs/ISSUES.md)

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

## Hardware support

| Environment | What runs |
|---|---|
| CPU (any OS) | Model correctness tests, smoke training, **multi-process DDP/FSDP2/TP correctness via the `gloo` backend** |
| Apple Silicon (MPS) | Single-device training |
| 1× NVIDIA GPU | Single-GPU training, BF16, Triton kernels, benchmarks |
| 2+ NVIDIA GPUs | NCCL DDP/FSDP2/TP training and scaling benchmarks |

Multi-GPU (NCCL) and Triton results are **pending verification on NVIDIA hardware** until
explicitly recorded in `results/`.

## Setup

Requires Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,text,logging]"     # add ,gpu on a Linux + NVIDIA machine for Triton
python scripts/check_env.py
```

### macOS notes for multi-process runs

`gloo` can hang on macOS when it binds to the wrong network interface. Pin it to loopback:

```bash
export GLOO_SOCKET_IFNAME=lo0
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 ...
```

FSDP2's `fully_shard` must be given an explicit CPU device mesh on Macs
(`init_device_mesh("cpu", (world_size,))`). Without one, it auto-detects MPS and fails.

## Layout

```text
configs/              YAML experiment configs
src/minimegatron/
  model/              GPT model: attention, blocks, config
  training/           trainer, optimizer and schedule, checkpointing, metrics
  distributed/        process-group setup, DDP, FSDP2, tensor parallelism, collectives demo
  kernels/            Triton kernels and their PyTorch references
  data/               synthetic and text datasets
  benchmarks/         benchmark harness and profiler helpers
tests/                unit / numerical / distributed / kernels / fault_injection
scripts/              utilities (environment check, experiment drivers)
results/              benchmark outputs (baselines are versioned, ad-hoc runs ignored)
docs/                 design and results documentation
```
