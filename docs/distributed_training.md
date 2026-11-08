# Distributed training

Three ways to use more than one device, what each actually splits, and how each is proved correct.
Deep dives: [fsdp.md](fsdp.md) and [tensor_parallel.md](tensor_parallel.md). DDP is covered in full
here, since it has no separate document.

## The three strategies at a glance

| | Splits | Communicates | Buys | Per-rank memory |
|---|---|---|---|---|
| **DDP** | the **batch** | one all-reduce of gradients per step | throughput | unchanged (`16P` bytes) |
| **FSDP2** | parameter / gradient / optimizer **storage** | all-gather + reduce-scatter per unit per step | capacity for one large model | `16P/R` + one unit |
| **TP** | each layer's **weight matrices** | two all-reduces per block, each direction | capacity, never gathering a full layer | shards of every matmul weight |

`16P` is the FP32 + AdamW cost per parameter: 4 bytes each for parameters, gradients, Adam's `m`
and Adam's `v` — before a single activation. For the medium preset (124M) that is ~2 GB per rank.
DDP replicates all of it, so **adding GPUs under DDP buys throughput and nothing at all in
capacity**. That is the problem the other two attack.

Selected by `distributed.strategy: auto | single | ddp | fsdp | tp`. `auto` is `single` on one rank
and `ddp` on several, which preserves every pre-M3 default. Asking for a strategy that needs peers
on a single rank is a hard error, not a silent downgrade.

## Shared infrastructure

`distributed/setup.py` owns everything that is not strategy-specific:

- **`DistInfo`** — `rank`, `local_rank`, `world_size`, `is_distributed`, `is_main`. Its defaults
  describe a single process, so every code path works unlaunched.
- **`init_distributed`** reads the environment `torchrun` sets and **does nothing when it is
  absent**. One entry point therefore serves single-process and distributed runs, and `train.py` is
  identical either way.
- **`destroy_distributed`** barriers then destroys the group, called from a `finally`. A rank that
  exits while its peers are inside a collective leaves them blocked until the timeout, turning one
  failure into a hung job.
- **`build_device_mesh` / `build_tp_mesh`** always pass an **explicit** device type. Letting torch
  build a default mesh means auto-detecting the accelerator, which finds MPS on a Mac and crashes
  (ISSUES #2).
- **Rank 0 owns every side effect.** Logging goes through a `NullLogger` elsewhere; checkpoints are
  written by rank 0 and followed by a barrier so no rank races past a file that is not on disk yet.

### Batches, and the one arithmetic rule

```
global_batch_size = micro_batch_size × grad_accum_steps × dp_world_size
dp_world_size     = world_size // tp_size
```

If `global_batch_size` is declared, the run **refuses to start** when that identity does not hold
(D16). It is validated rather than derived: deriving `grad_accum_steps` would make one config mean
different things at different world sizes, which is exactly what this project's strict
configuration exists to prevent. The error states the arithmetic:

```
ConfigError: global_batch_size 16 != micro_batch_size 8 x grad_accum_steps 2 x dp_world_size 2 = 32
```

The lesson it teaches is the right one: **to hold the global batch fixed while adding ranks, reduce
`grad_accum_steps`.** And note `dp_world_size` divides out `tp_size` — tensor-parallel ranks are
not data-parallel ranks (D24), so a preset's declared global batch needs **no change** when TP is
switched on, unlike DDP or FSDP.

## DDP: averaging gradients is one large batch

Data parallelism rests on the loss being a **mean over samples**. For a global batch `B` split into
equal shards:

```
L(B) = (1/R) Σᵣ L(Bᵣ)        ⟹        ∇L(B) = (1/R) Σᵣ ∇L(Bᵣ)
```

Each rank computes `∇L(Bᵣ)` from its own shard, and an **all-reduce with mean** turns those `R`
different gradients into the single gradient of the global batch, present identically everywhere.
Every replica then takes the same step from the same weights, so they never diverge.

Two consequences that drove real code:

- **Equal shard sizes are mandatory.** The mean of per-rank means is only the mean over the global
  batch when the shards are the same size. A short shard silently reweights the gradient, so the
  sharding hands every rank exactly `micro_batch_size` samples and never pads a tail.
- **Nothing about the maths changes with world size.** Throughput changes; the update does not. Any
  divergence between a 1-rank and a 2-rank run at equal global batch is therefore a bug.

### What `DistributedDataParallel` does

At construction it broadcasts rank 0's parameters, so replicas start identical. Then it hooks every
parameter: as `backward()` produces each gradient, a **reducer** groups them into buckets and fires
an async all-reduce per bucket as soon as one fills. That bucketing is the performance idea —
communication for early gradients **overlaps** with the backward still computing later ones.

It also explains why DDP wants every parameter to receive a gradient: one the reducer waits for and
never gets will hang the step. We keep `find_unused_parameters=False` and *assert* that the tied
embedding (used twice per forward) is handled, rather than papering over it.

### `no_sync` for accumulation

With `grad_accum_steps = k`, only the gradient after the *k*-th micro-batch is used. All-reducing
the intermediates costs `k×` the communication and discards `k−1` of the results. `model.no_sync()`
suppresses the reducer for all but the last micro-batch, giving exactly one all-reduce per
optimizer step.

Correctness is unaffected because local accumulation and all-reduce are both sums, and sums
commute: `allreduce(Σⱼ gⱼ) == Σⱼ allreduce(gⱼ)`.

FSDP has its own equivalent (`set_requires_gradient_sync(False)`) with the *opposite* trade-off:
unreduced FSDP gradients accumulate **unsharded**, so suppressing the collective costs memory to
save bandwidth.

### The educational all-reduce

`distributed/collectives_demo.py` does the averaging by hand — `dist.all_reduce(SUM)` then
`/= world_size` — deliberately outside the training path, and prints per-rank gradients before and
after so the operation DDP hides behind an autograd hook is visible once. A test asserts it
reproduces DDP's gradients, which keeps the demo from quietly ceasing to describe reality.

```bash
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29501 \
    -m minimegatron.distributed.collectives_demo
```

## How correctness is established

Every strategy is checked against the **single-device reference**, never against another strategy
and never against itself. The claims, all verified on 2 gloo ranks on CPU:

| Claim | Where |
|---|---|
| DDP gradients == 1 process at the same global batch, every parameter | `test_ddp_gradients.py` |
| `no_sync` accumulation == one large batch | `test_ddp_accumulation.py` |
| Hand-rolled all-reduce == DDP's gradients | `test_collectives_demo.py` |
| FSDP forward, loss, gradients and **gradient norm** == reference | `test_fsdp_equivalence.py` |
| Each rank holds ~1/R of parameters *and* optimizer state | `test_fsdp_sharding.py` |
| TP sharded gradients == the reference's matching slice | `test_tp_model.py` |
| TP **replicated** gradients == the full reference gradient, identical across ranks | `test_tp_model.py` |
| Multi-step trajectories track the reference | `test_*_training.py`, `test_fsdp_equivalence.py` |
| Checkpoints from any strategy load into a plain `GPT` | `test_*_checkpoint.py` |
| A dying worker fails the job and does not hang | `tests/fault_injection/` |

The headline result, on the tiny preset at a global batch of 16 — all four agree:

| Run | Step losses |
|---|---|
| 1 rank, single | 10.8652, 10.8624, 10.8589, 10.8704 |
| 2 ranks, DDP | identical |
| 2 ranks, FSDP2 | identical |
| 2 ranks, TP=2 | identical |

## Running it

All correctness work runs on CPU with gloo (D2), so none of this needs a GPU.

```bash
# gloo picks an unreachable interface from the hostname unless pinned: lo0 on macOS, lo on Linux.
export GLOO_SOCKET_IFNAME=lo0

TORCHRUN="--nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500"

# DDP. grad_accum_steps=1 keeps the global batch at the preset's 16 (8 x 1 x 2).
.venv/bin/torchrun $TORCHRUN -m minimegatron.train \
    --config configs/ddp_2gpu.yaml

# FSDP2.
.venv/bin/torchrun $TORCHRUN -m minimegatron.train \
    --config configs/fsdp_2gpu.yaml

# Tensor parallelism. tp_size defaults to the whole world; the global batch is unaffected.
.venv/bin/torchrun $TORCHRUN -m minimegatron.train \
    --config configs/tp_2gpu.yaml
```

On NVIDIA hardware the backend resolves to `nccl` automatically, `--standalone` works, and
`GLOO_SOCKET_IFNAME` is unnecessary.

**Always use a hard timeout around a distributed subprocess.** macOS has no `timeout` binary, and a
misconfigured gloo hangs silently rather than erroring (ISSUES #1). Every distributed test in this
repository enforces one in Python — which is how the FSDP evaluation deadlock (ISSUES #13) became a
three-minute diagnosable failure instead of a stalled session.

## Gotchas worth knowing before you extend this

| Gotcha | Detail |
|---|---|
| Collectives inside the forward | Under FSDP the forward communicates, so **the number of forward passes is part of the contract between ranks** (ISSUES #13), not just the data |
| `clip_grad_norm_` under TP | Computes a rank-local norm, because TP parameters are ordinary tensors that happen to be slices. FSDP's `DTensor`s carry their sharding; TP's do not (ISSUES #12) |
| Optimizer construction order | Under FSDP it must come **after** `fully_shard`, which replaces parameters with `DTensor`s in place |
| Dropout | Identically seeded ranks draw identical masks (wrong); differently seeded ranks cannot match a single-device run. All comparisons use `dropout=0` (D18) |
| One global RNG | Tests comparing two runs in one process must drive them **sequentially** (ISSUES #6) |
| Tensors through a `Queue` | Deadlocks when the sender exits; the test harness hands results over via files (ISSUES #7) |
