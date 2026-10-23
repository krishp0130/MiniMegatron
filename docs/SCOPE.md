# MiniMegatron — Project Scope

This is the authoritative project specification. Changes to scope get recorded here **and** in the
"Scope changes" log at the bottom. Day-to-day progress lives in [PROGRESS.md](PROGRESS.md);
problems and design decisions live in [ISSUES.md](ISSUES.md).

## Project Objective

Build **MiniMegatron**, an educational but engineering-quality distributed transformer training
framework inspired by NVIDIA Megatron-LM.

The goal is to demonstrate practical understanding of modern GPU-accelerated deep learning systems:
transformer architecture, distributed training (data, sharded, and tensor parallelism), collective
communication, GPU memory management, kernel optimization, automated validation, and performance
benchmarking.

This should be a real, executable project, not a collection of demonstrations. Prioritize
correctness, reproducibility, maintainability, and understandable abstractions.

The project should run on a single GPU and scale to at least two NVIDIA GPUs when hardware is
available. CPU mode should support model correctness tests and small smoke tests.

Do not claim speedups or memory reductions until measured against reproducible baselines.

**Guiding principle: every optimization must have a reference implementation, a correctness test,
and a reproducible benchmark.**

## 1. Core Technology

- Python 3.11+
- PyTorch 2.x, with the exact supported version pinned (currently `torch==2.14.1`)
- CUDA and NCCL for GPU execution
- Triton for custom GPU kernels
- PyTorch DistributedDataParallel (DDP)
- PyTorch fully_shard / FSDP2
- torch.distributed collectives (for manual tensor parallelism)
- Hugging Face tokenizers or a simple byte-level tokenizer
- NumPy
- pytest
- torch.profiler
- TensorBoard or structured JSONL logging
- YAML-based experiment configuration

Keep external dependencies minimal.

## 2. Transformer Architecture

Implement a configurable decoder-only GPT-style transformer using PyTorch.

Required components:

- Token and positional embeddings (learned positional embeddings for MVP)
- Multi-head causal self-attention
- QKV projections
- Scaled dot-product attention
- Causal attention masking
- Residual connections
- Pre-normalization transformer blocks
- LayerNorm or RMSNorm
- Feed-forward MLP using GELU
- Final normalization and language-model head
- Cross-entropy next-token prediction objective

Use standard PyTorch operations first. Correctness matters more than initial performance.

Implement a reference attention path explicitly enough to inspect its math. Optionally support
PyTorch SDPA as a separate optimized backend.

Configurable parameters: `vocab_size`, `seq_len`, `n_layers`, `n_heads`, `d_model`, `d_ff`,
`dropout`, `dtype`, `attention_backend`, `normalization_type`.

Provide three configurations (approximate ranges; report actual parameter counts):

- tiny: ~5–15M parameters
- small: ~30–60M parameters
- medium: ~100–150M parameters

Design note: keep linear layers in attention and MLP as separate, clearly named modules
(`qkv`, `out_proj`, `fc1`, `fc2`) so the tensor-parallel milestone can swap them for
column/row-parallel versions without restructuring the model.

## 3. Training Engine

Build a modular training engine supporting:

- AdamW optimizer
- Learning-rate warmup and cosine decay
- Gradient accumulation
- Gradient clipping
- BF16 mixed precision on supported hardware
- FP32 baseline
- Configurable batch size and sequence length
- Deterministic random seeds
- Training and validation loss logging
- Token count and tokens/second
- Model checkpointing
- Optimizer and scheduler state checkpointing
- RNG state restoration
- Resume from checkpoint

Define global batch size clearly:

```
global_batch_size = micro_batch_size × gradient_accumulation_steps × data_parallel_world_size
```

Require explicit validation of this relationship. Note that under tensor parallelism, TP ranks see
the *same* data, so `data_parallel_world_size = world_size / tp_size`.

Use synthetic token datasets for initial validation and benchmarking. Support a real text dataset
for language-model training after the infrastructure is functional.

## 4. Distributed Data Parallel Training

Implement multi-GPU training with torch.distributed and NCCL.

Requirements:

- Launch with torchrun.
- Initialize and destroy process groups correctly.
- Assign each process its local CUDA device.
- Wrap the model using DistributedDataParallel.
- Use DistributedSampler or equivalent deterministic sharding.
- Correctly synchronize gradients.
- Support gradient accumulation using no_sync where appropriate.
- Handle distributed metric aggregation.
- Log checkpoints from rank zero with appropriate synchronization.
- Ensure all ranks terminate cleanly on normal completion.

Do not reimplement production NCCL collectives in the core training path.

Create a separate educational module demonstrating how gradient synchronization works and how an
explicit all-reduce compares with DDP.

Measure throughput differences between single-GPU and two-GPU runs using equivalent per-device
workloads, and separately report comparisons with equivalent global batch sizes.

## 5. FSDP2 — Memory-Efficient Training

Add a selectable FSDP2 training strategy using PyTorch fully_shard.

Requirements:

- Shard model parameters, gradients, and optimizer states.
- Apply sharding at transformer-block boundaries where supported.
- Manage mixed precision consistently.
- Support valid distributed checkpoint and restore semantics for sharded models.
- Measure peak GPU memory per rank.
- Compare DDP versus FSDP under equivalent model, batch size, precision, and hardware conditions.

Include documentation explaining: parameter all-gather, reduce-scatter, gradient synchronization,
optimizer-state sharding, memory savings versus communication overhead.

Do not mix legacy FSDP1 and FSDP2 APIs without a documented compatibility reason.

## 6. Tensor Parallelism (basic, Megatron-style)

*Added 2026-10-23.* Implement basic intra-layer tensor parallelism (TP) following the Megatron-LM
paper (Shoeybi et al., 2019). Where DDP/FSDP split the **batch** across ranks, TP splits the
**weight matrices of each layer** across ranks, so each rank holds and computes a slice of every
layer.

Implement TP manually with `torch.distributed` collectives and custom `torch.autograd.Function`s,
so the communication is explicit and inspectable. PyTorch's DTensor-based
`torch.distributed.tensor.parallel` (`parallelize_module`, `ColwiseParallel`, `RowwiseParallel`)
may be used as a second, reference implementation for cross-checking, not as the primary path.

Required components:

- **Communication primitives** as autograd functions (Megatron's conjugate `f` / `g` pair):
  - `copy_to_tp_region`: identity in forward, all-reduce in backward
  - `reduce_from_tp_region`: all-reduce in forward, identity in backward
  - (optional) gather/scatter variants for gathering outputs along the last dimension
- **`ColumnParallelLinear`**: weight split along the output dimension; each rank produces a slice
  of the output features.
- **`RowParallelLinear`**: weight split along the input dimension; partial outputs are summed with
  an all-reduce.
- **Parallel MLP**: `fc1` column-parallel → GELU (local, no communication) → `fc2` row-parallel.
  Exactly one all-reduce in forward and one in backward per MLP.
- **Parallel attention**: QKV projection column-parallel, so attention **heads are partitioned**
  across ranks (each rank computes `n_heads / tp_size` heads independently); output projection
  row-parallel.
- **Embeddings and LM head**: replicated on every rank for the basic version. Vocab-parallel
  embedding and a vocab-parallel cross-entropy are optional stretch goals.
- **Weight initialization and loading**: TP shards must be derivable from a full, single-device
  model's weights (split a reference state dict into shards) so correctness can be compared
  exactly, and so a TP checkpoint can be gathered back to a full model.
- **Configuration**: `strategy: tp` with `tp_size`. Validate `n_heads % tp_size == 0` and
  `d_ff % tp_size == 0`, and fail clearly otherwise.
- **Process groups**: a TP group distinct from the (eventual) DP group, built with a device mesh,
  even though the basic version uses TP only (`tp_size == world_size`).

Correctness requirements:

- With identical weights and inputs, TP forward outputs (logits, loss) match the single-device
  reference within dtype-appropriate tolerances.
- Gradients of every sharded parameter match the corresponding slice of the reference gradient;
  gradients of replicated parameters (norms, embeddings) match the full reference gradient and are
  identical across TP ranks.
- A few optimizer steps under TP track the reference training trajectory.
- Unit tests for each primitive: forward and backward behavior of `f`/`g`, column/row linear
  against `nn.Linear`.
- All of the above runnable on CPU with the `gloo` backend (2 processes), so TP correctness does
  not depend on GPU availability.

Dropout: run correctness comparisons with `dropout=0`. Correct TP dropout requires a
model-parallel RNG tracker (different seeds for sharded regions, same seed for replicated ones);
document this, and implement it only if time permits.

Benchmarks (pending NVIDIA hardware): TP=2 vs DDP on 2 GPUs vs single GPU, reporting throughput,
step latency, peak memory per rank, and communication volume per step. Explain the expected
trade-off: TP cuts per-rank parameter and activation memory but adds communication in every layer's
forward and backward, so it is intended for fast intra-node interconnects.

Documentation (`docs/tensor_parallel.md`): column vs row partitioning, why column→row pairing
needs only one all-reduce per block, the `f`/`g` conjugate operators, head partitioning in
attention, and how TP composes with DP (2D parallelism) conceptually.

Out of scope for the basic version: sequence parallelism, pipeline parallelism, combined 2D TP+DP
training (document it as future work), and TP combined with FSDP.

## 7. Triton Kernel Optimization

Implement one meaningful custom GPU kernel in Triton. Start with fused RMSNorm or fused layer
normalization.

Requirements:

- A reference PyTorch implementation
- A Triton forward kernel
- A valid backward implementation or an explicit supported autograd strategy
- Numerical equivalence tests
- Gradient correctness tests where differentiable
- Tests across tensor shapes and dtypes
- GPU kernel benchmarking
- Memory and runtime comparisons

After RMSNorm works, optionally implement a fused GELU or another useful pointwise operation.

Do not attempt a custom FlashAttention implementation in the first phase.

The framework must support switching between baseline PyTorch and Triton implementations through
configuration.

## 8. Automated Correctness and QA Framework

This is a critical component.

### Unit tests

Test: attention tensor shapes and causal masking; transformer block forward pass; output logits
and loss; gradients; optimizer updates; tokenization and data batching; configuration validation;
checkpoint serialization.

### Numerical correctness

Compare optimized implementations with trusted PyTorch reference implementations. Test FP32 and
BF16 where supported. Use dtype-appropriate absolute and relative tolerances rather than exact
floating-point equality. Test extreme inputs, non-power-of-two dimensions, small and large tensor
shapes, and unusual batch sizes.

### Distributed correctness

Test:

- DDP gradients against a reference global-batch computation
- Correct dataset sharding
- Loss and gradient accumulation equivalence
- Rank synchronization
- FSDP training convergence against a reference
- TP forward/backward equivalence against a single-device reference
- Distributed checkpoint recovery
- Clean distributed shutdown

Preserve identical effective batches, objective scaling, and optimizer settings when making
correctness comparisons.

### Failure injection

Create controlled tests covering: CUDA out-of-memory errors, invalid configurations (including
invalid `tp_size`), corrupt checkpoints, missing dataset files, unsupported GPU features, and a
worker failing during distributed execution.

Ensure failures produce clear diagnostics and do not leave test processes hanging indefinitely.
Distributed failure tests must use subprocess isolation and explicit timeouts.

### Regression testing

Maintain a small suite of deterministic workloads. Verify functional invariants on every commit.
Store baseline performance data separately. Avoid treating noisy performance measurements as
deterministic unit tests. Add configurable warning thresholds for performance regressions, with
repeated measurements and statistical summaries.

## 9. GPU Performance Benchmarking

Build a benchmark harness that reports: training throughput (tokens/s), step latency, peak
allocated GPU memory, peak reserved GPU memory, GPU utilization (when supported), model parameter
count, hardware and GPU architecture, CUDA/driver/PyTorch/Triton versions, effective batch size,
precision, number of GPUs, distributed strategy, and TP size.

Use warmup iterations before timing. Synchronize CUDA operations at benchmark boundaries. Use CUDA
events for GPU execution time. Collect multiple runs and report median and dispersion rather than a
single measurement.

Explicit comparisons:

1. FP32 versus BF16
2. One GPU versus two GPUs
3. DDP versus FSDP
4. DDP versus TP (and FSDP versus TP) on two GPUs
5. PyTorch reference versus Triton kernel
6. Different sequence lengths and microbatch sizes

Output results in both JSON and human-readable Markdown tables. Include profiler traces where
appropriate.

## 10. CLI and Configuration

```bash
python -m minimegatron.train --config configs/tiny.yaml

torchrun --standalone --nproc_per_node=2 -m minimegatron.train --config configs/ddp_2gpu.yaml
torchrun --standalone --nproc_per_node=2 -m minimegatron.train --config configs/fsdp_2gpu.yaml
torchrun --standalone --nproc_per_node=2 -m minimegatron.train --config configs/tp_2gpu.yaml

python -m minimegatron.benchmarks.run --config configs/benchmark.yaml

pytest tests/unit -q
```

Provide meaningful errors for unsupported hardware and configuration options. Do not silently fall
back from a requested multi-GPU configuration to single-GPU training.

## 11. Repository Structure

```text
minimegatron/
├── CLAUDE.md                      # entry point for coding agents
├── configs/
│   ├── tiny.yaml  small.yaml  medium.yaml
│   ├── ddp_2gpu.yaml  fsdp_2gpu.yaml  tp_2gpu.yaml
│   └── benchmark.yaml
├── src/minimegatron/
│   ├── config.py       # ConfigError, strict dict->dataclass, YAML loading
│   ├── experiment.py   # ExperimentConfig: model/data/train/distributed sections
│   ├── model/          attention.py  transformer.py  layers.py  config.py
│   ├── training/       trainer.py  optimizer.py  checkpoint.py  metrics.py
│   ├── distributed/
│   │   ├── setup.py  ddp.py  fsdp.py  collectives_demo.py
│   │   └── tensor_parallel/
│   │       ├── mappings.py    # f/g autograd communication primitives
│   │       ├── layers.py      # ColumnParallelLinear, RowParallelLinear
│   │       └── model.py       # parallelize a GPT model, shard/gather state dicts
│   ├── kernels/        rmsnorm.py  reference.py
│   ├── data/           synthetic.py  text.py
│   ├── benchmarks/     run.py  profiler.py
│   └── train.py
├── tests/  unit/  numerical/  distributed/  kernels/  fault_injection/
├── scripts/
├── results/
├── docs/
│   ├── SCOPE.md  PROGRESS.md  ISSUES.md
│   ├── architecture.md  distributed_training.md  tensor_parallel.md
│   ├── testing.md  benchmarks.md
├── pyproject.toml
└── README.md
```

## 12. Implementation Milestones

### Milestone 1 — Functional GPT baseline
Model implementation; synthetic data; single-device training; CPU smoke tests; loss reduction on a
tiny overfitting task; unit tests.

### Milestone 2 — Distributed data parallel training
DDP implementation; two-GPU training; gradient synchronization validation; rank-aware logging;
distributed benchmarks; educational collectives demo.

### Milestone 3 — Memory optimization
BF16 training; FSDP2; peak memory measurements; sharded checkpoint support; DDP/FSDP comparisons.

### Milestone 4 — Tensor parallelism
`f`/`g` communication primitives; column/row-parallel linear layers; parallel MLP and attention;
state-dict sharding/gathering; TP config and validation; CPU (gloo) correctness tests against the
single-device reference; TP benchmarks (pending hardware); `docs/tensor_parallel.md`.

### Milestone 5 — GPU kernel optimization
Triton RMSNorm; reference implementation; correct forward and backward behavior; kernel
benchmarks; end-to-end integration.

### Milestone 6 — Reliability engineering
Numerical regression tests; distributed correctness tests; failure injection; automated
benchmarks; CI pipeline running CPU-safe tests; GPU test workflow documented for supported
hardware.

### Milestone 7 — Final documentation
Architecture overview; installation instructions; reproducible experiments; performance comparison
tables; profiling results; known limitations and future work.

## 13. Definition of Done

1. A GPT-style model trains successfully and reduces loss on a small task.
2. CPU-safe correctness tests pass.
3. Single-GPU training works on supported CUDA hardware.
4. DDP trains correctly across two GPUs, with verified gradients.
5. FSDP2 trains correctly and supports valid checkpointing.
6. Tensor parallelism (TP=2) matches the single-device reference in forward outputs and gradients,
   and trains correctly.
7. An optimized Triton kernel passes numerical and gradient tests.
8. Automated validation catches deliberately introduced errors.
9. Benchmark results can be reproduced using documented configurations.
10. Performance claims cite actual measured configurations and results.
11. The repository is documented well enough for another engineer to run the project.

If two GPUs are unavailable, implement the distributed interfaces, test what is possible locally
(gloo on CPU), and explicitly mark multi-GPU verification as pending. Do not fabricate test
results.

## Scope changes

| Date | Change |
|---|---|
| 2026-10-23 | Initial scope adopted. |
| 2026-10-23 | Added basic Megatron-style tensor parallelism as Milestone 4 (after FSDP2). Triton, reliability, and documentation milestones renumbered to 5, 6, 7. |
