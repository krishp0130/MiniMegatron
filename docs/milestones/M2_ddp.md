# Milestone 2 — Distributed Data Parallel

**Goal:** train one model across several processes, each holding a full replica and a different
slice of the global batch, and prove the result is *mathematically identical* to the M1
single-device run. That proof is the deliverable. It is what makes M3's FSDP2 memory numbers and
M4's tensor-parallel numbers believable, because each will be checked against this same reference.

**Not in this milestone:** parameter sharding (M3), BF16 (M3), splitting weight matrices (M4),
Triton (M5), elastic or multi-node training, pipeline parallelism, a general `strategy:` switch
(deferred to M3, where a second strategy will show what the seam should be).

**Where it runs:** correctness on **CPU with gloo, 2 processes** (D2). There is no NVIDIA GPU on
the dev machine (ISSUES #3), so NCCL and every throughput number stay *pending hardware*.

## Exit criteria

1. `pytest -q` passes on CPU, including the new 2-process gloo tests.
2. DDP gradients equal single-process global-batch gradients: 2 ranks x micro=4 matches
   1 rank x micro=8, for every parameter, within float tolerance.
3. `no_sync` accumulation is equivalent: 2 ranks x accum=2 x micro=2 matches 1 rank x micro=8.
4. The educational manual all-reduce reproduces DDP's gradients.
5. A 2-rank checkpoint loads into a plain single-device `GPT` with no key munging.
6. `torchrun --nproc_per_node=2 -m minimegatron.train --config configs/tiny.yaml` runs end to end,
   logs from rank 0 only, and **all ranks exit 0**.
7. Throughput recorded only if measured; otherwise marked *pending hardware*.

## Build order

| Step | Files | Tests |
|---|---|---|
| 1. Process groups | `distributed/{config,setup}.py` | `tests/distributed/test_setup.py` |
| 2. Rank sharding | `data/synthetic.py` | `tests/unit/test_data.py` |
| 3. DDP in the trainer | `training/trainer.py`, `train.py` | `tests/distributed/test_ddp_*.py` |
| 4. Collectives demo | `distributed/collectives_demo.py` | `tests/distributed/test_collectives_demo.py` |
| 5. Harness + torchrun | `tests/distributed/conftest.py` | `tests/distributed/test_torchrun_smoke.py` |

---

## The one idea: averaging gradients == one large batch

Data parallelism rests on the loss being a **mean over samples**. For a global batch `B` split into
`R` equal shards `B_1..B_R`, one per rank:

```
L(B) = (1/|B|) * sum over all samples   = (1/R) * sum_r L(B_r)      (when |B_r| are equal)
```

Gradients are linear, so

```
grad L(B) = (1/R) * sum_r grad L(B_r)
```

Each rank computes `grad L(B_r)` from its own shard, and an **all-reduce with mean** turns those
`R` different gradients into the single gradient of the global batch, present identically on every
rank. Every replica then takes the same optimizer step from the same weights, so the replicas stay
bit-identical without ever exchanging weights again after startup.

Two consequences worth internalizing:

- **Equal shard sizes matter.** If one rank gets fewer samples, `(1/R) * sum_r L(B_r)` is no longer
  the mean over the global batch, and the gradient is silently wrong — weighted toward the samples
  on the smaller shard. This is why our sharding hands every rank exactly `micro_batch_size`
  samples and never pads a short tail.
- **Nothing about the *math* changes with world size.** Throughput changes; the update does not.
  So any divergence between a 1-rank and a 2-rank run at equal global batch is a bug, which is
  exactly what exit criterion 2 asserts.

### What `DistributedDataParallel` actually does

1. At construction, it **broadcasts rank 0's parameters and buffers** to every rank, so replicas
   start identical even if their RNG differed.
2. It registers an autograd hook on every parameter. As `backward()` produces each gradient, the
   hook hands it to a **reducer** that groups gradients into buckets (25 MB by default) and fires
   an async all-reduce per bucket as soon as it is full.
3. The all-reduce is `SUM` divided by world size, i.e. a mean — the `(1/R) sum_r` above.

The bucketing is the performance idea: communication for early-computed gradients **overlaps** with
the backward pass still computing later ones, so most of the all-reduce cost hides behind compute
that was happening anyway. It also explains why gradients arrive in reverse-layer order and why DDP
wants every parameter to receive a gradient — a parameter the reducer waits for and never gets will
hang the step (the reason `find_unused_parameters` exists, and why we keep it `False` and prove our
tied embedding is fine rather than papering over it).

### Why `no_sync` for gradient accumulation

With `grad_accum_steps = k`, only the gradient after the *k*-th micro-batch is used. All-reducing
the intermediate `k-1` partial sums costs `k` times the communication and throws `k-1` of the
results away. `model.no_sync()` suppresses the reducer, so gradients accumulate locally, and the
final micro-batch runs outside the context to trigger exactly one all-reduce per optimizer step.

Correctness is unaffected because local accumulation and all-reduce are both sums, and sums commute:
`allreduce(sum_j g_j) == sum_j allreduce(g_j)`.

### Gradient clipping must come after synchronization

`clip_grad_norm_` computes a global L2 norm over all gradients and rescales if it exceeds the
threshold. Run on *pre-sync* per-rank gradients, each rank would compute a different norm from its
own shard, rescale by a different factor, and the subsequent all-reduce would average inconsistently
scaled gradients — making the effective clipping threshold depend on world size and the data split.
Clipping after the synchronizing backward, on the underlying module's parameters, computes the one
true global norm. (Our `grad_clip` is logged pre-clip every step, which stays an instability signal.)

### Metric aggregation

A loss computed on rank *r* describes `B_r`, not `B`. For logging to mean what it says, the loss is
all-reduced with mean across ranks. Same for validation loss. Without this, a logged curve is one
rank's view and the two-rank run looks noisier than it is.

### Rank-0-only side effects

Logs and checkpoints are written by rank 0 alone: every rank holds identical weights after the
optimizer step, so `R` writes of the same bytes is pure waste and interleaved JSONL lines would
corrupt the log. A **barrier after the write** keeps the other ranks from racing ahead of a
checkpoint that is not on disk yet. Checkpoints store `model.module.state_dict()` — the unwrapped
module — so they stay interchangeable with single-device runs rather than growing a `module.`
prefix that every future loader would have to strip.

---

## Step 1 — Process groups

`DistributedConfig`: `backend` (`auto` | `gloo` | `nccl`) and `timeout_seconds`. `auto` resolves to
nccl when CUDA is available, else gloo. Same frozen, collect-every-error validation as the other
config sections (D10).

`DistInfo`: `rank`, `local_rank`, `world_size`, `is_distributed`, `is_main`.

`init_distributed(cfg)` reads the environment `torchrun` sets (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`).
When those are absent it returns a single-process `DistInfo` and initializes nothing, so **every M1
code path keeps working unchanged** and the same `train.py` serves both. `local_rank` selects the
CUDA device, so each process owns exactly one GPU.

`destroy_distributed()` barriers and then destroys the group, and is called from a `finally` so a
crashing rank does not leave its peers blocked in a collective waiting for a process that is gone.
**Every distributed subprocess in tests gets a hard Python-side timeout** — gloo on macOS hangs
silently without loopback pinning (ISSUES #1) and macOS has no `timeout` binary.

## Step 2 — Rank sharding

`BatchStream` gains `rank` and `world_size`. Micro-batch *m* on rank *r* reads global stream
positions starting at:

```
offset = m * micro_batch_size * world_size + r * micro_batch_size
```

The union over ranks of micro-batch *m* is therefore exactly the single-process micro-batch of size
`micro_batch_size * world_size` at index *m*. This is what keeps exit criteria 2 and 3 exact rather
than statistical, and it follows for free from M1's decision to define the index stream over
*samples* rather than batches (D12). Every rank gets the same number of samples (see "equal shard
sizes" above), every sample is still seen once per epoch across the cluster, and mid-epoch resume
still needs nothing saved beyond the step counter.

## Step 3 — DDP in the trainer

Wrap the model in `DistributedDataParallel` (`device_ids=[local_rank]` on CUDA, omitted on
CPU/gloo). `no_sync()` for all but the last micro-batch; clip after the synchronizing backward on
`model.module.parameters()`; all-reduce the logged losses; count `tokens_seen` across the cluster;
pass the real `dp_world_size` into the `global_batch_size` cross-check that M1 already built.

## Step 4 — Collectives demo

`distributed/collectives_demo.py`, deliberately outside the training path (scope §4 forbids
reimplementing collectives there). It computes gradients locally, then averages them by hand —
`dist.all_reduce(g, op=SUM)` then `g /= world_size` — printing per-rank values before and after so
the mechanism DDP hides is visible once. A test asserts it reproduces DDP's gradients, which keeps
the demo from rotting into something that no longer reflects reality.

---

## Known traps

| Trap | Handling |
|---|---|
| Dropout under DDP | Identically seeded ranks draw identical masks (wrong for DP); differently seeded ranks can't match a single-device run. **All equivalence tests use `dropout=0`**; correct per-rank dropout needs a model-parallel RNG tracker, documented and deferred (the same call the TP scope makes). |
| One global RNG, now across processes | Ranks seed identically for identical init (DDP's broadcast enforces it anyway). Tests comparing a 1-rank and a 2-rank run must drive them **sequentially** (ISSUES #6). |
| Tied embedding used twice per forward | DDP's reducer handles shared parameters, but assert it with `find_unused_parameters=False` rather than assume, and check the tied gradient against the reference. |
| gloo hangs on macOS | `GLOO_SOCKET_IFNAME=lo0`, explicit `--master_addr/--master_port`, hard timeouts (ISSUES #1). |
| Unequal shards | Never pad or drop a tail into a short shard; it silently reweights the gradient. |

## Benchmarks

*Pending hardware.* 1 vs 2 GPU throughput at equal per-device workload, and separately at equal
global batch size, is meaningless on a single CPU — two gloo ranks on one machine add communication
and buy no parallelism, so a local measurement would be actively misleading. To be run on 2 NVIDIA
GPUs and recorded in PROGRESS.md with config, hardware and commit.
