# MiniMegatron

An educational, engineering-quality distributed transformer training framework inspired by
[NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM): a GPT-style decoder, a training
engine, DDP and FSDP2 data parallelism, a custom Triton kernel, and a correctness/benchmarking
harness.

> **Guiding principle:** every optimization has a reference implementation, a correctness test,
> and a reproducible benchmark. No performance claim appears here without a measured config.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0. Setup | Repo, environment, packaging | done |
| 1. Functional GPT baseline | Model, synthetic data, single-device training, unit tests | not started |
| 2. Distributed training | DDP, gradient sync validation, rank-aware logging | not started |
| 3. Memory optimization | BF16, FSDP2, sharded checkpoints, memory measurements | not started |
| 4. GPU kernel optimization | Triton RMSNorm, forward and backward, benchmarks | not started |
| 5. Reliability engineering | Regression and distributed tests, failure injection, CI | not started |
| 6. Documentation | Architecture, experiments, results tables | not started |

## Hardware support

| Environment | What runs |
|---|---|
| CPU (any OS) | Model correctness tests, smoke training, **multi-process DDP/FSDP2 correctness via the `gloo` backend** |
| Apple Silicon (MPS) | Single-device training |
| 1× NVIDIA GPU | Single-GPU training, BF16, Triton kernels, benchmarks |
| 2+ NVIDIA GPUs | NCCL DDP/FSDP2 training and scaling benchmarks |

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
  distributed/        process-group setup, DDP, FSDP2, educational collectives demo
  kernels/            Triton kernels and their PyTorch references
  data/               synthetic and text datasets
  benchmarks/         benchmark harness and profiler helpers
tests/                unit / numerical / distributed / kernels / fault_injection
scripts/              utilities (environment check, experiment drivers)
results/              benchmark outputs (baselines are versioned, ad-hoc runs ignored)
docs/                 design and results documentation
```
