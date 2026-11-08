# Milestone 3 — Memory Optimization (BF16 + FSDP2)

**Goal:** make the model's *memory* footprint scale with the number of ranks, instead of being
capped by one device. DDP (M2) replicates parameters, gradients and optimizer state on every rank,
so the biggest trainable model is whatever fits on one GPU. FSDP2 shards all three, trading
communication for capacity. BF16 mixed precision attacks the same problem from the other side.

**Not in this milestone:** tensor parallelism (M4), Triton (M5), FP8, CPU offload, activation
checkpointing, FSDP1 APIs (SCOPE.md forbids mixing them without a documented reason), and 2D
FSDP+TP composition.

**Where it runs:** correctness on **CPU with gloo, 2 processes** (D2). Peak-memory *bytes* and
throughput need NVIDIA hardware (ISSUES #3) and stay *pending*.

## Exit criteria

1. `pytest -q` passes on CPU, including the new 2-process FSDP tests.
2. FSDP forward, loss and gradients match the single-device reference (dropout=0).
3. A short FSDP run tracks the reference loss curve across several optimizer steps.
4. Parameters **and** optimizer state are demonstrably sharded: each rank holds ~1/world_size.
5. BF16: parameters stay fp32, activations are bf16, the loss is fp32, and a bf16 run converges.
6. DCP round trip restores an FSDP run exactly; the consolidated checkpoint loads into a plain
   single-device `GPT`.
7. Peak memory and DDP-vs-FSDP throughput marked *pending hardware* — never estimated.

## Build order

| Step | Files | Tests |
|---|---|---|
| 1. Precision | `training/config.py`, `training/trainer.py` | `tests/unit/test_precision.py` |
| 2. Strategy switch | `distributed/config.py`, `training/trainer.py` | `tests/unit/test_strategy.py` |
| 3. FSDP2 wrapping | `distributed/fsdp.py`, `distributed/setup.py` | `tests/distributed/test_fsdp_*.py` |
| 4. Sharded checkpoints | `distributed/checkpoint.py` | `tests/distributed/test_fsdp_checkpoint.py` |

---

## Where the memory actually goes

For a model with `P` parameters trained in FP32 with AdamW, per rank:

```
parameters   4P bytes
gradients    4P bytes
Adam m       4P bytes
Adam v       4P bytes
-----------------------
total       16P bytes    before a single activation
```

For the medium preset (124M params) that is ~2 GB before activations. DDP replicates all 16P on
every rank: adding GPUs buys throughput and **nothing** in capacity.

FSDP2 shards all four across `R` ranks, so each holds `16P/R` bytes, plus transient space for
whichever unit is currently gathered. That is the entire proposition — and the reason the sharding
*unit* matters so much (below).

### All-gather and reduce-scatter

DDP's collective is an **all-reduce** on gradients: every rank ends up with the full averaged
gradient, because every rank owns a full copy of the parameters it needs to update.

FSDP owns only a *slice* of each parameter, so its collectives are different:

```
forward (per unit):   all-gather parameter shards -> full params -> compute -> free the full params
backward (per unit):  all-gather params again -> compute grads -> reduce-scatter grads
```

**Reduce-scatter**, not all-reduce: the reduction and the scattering happen in one step, so each
rank receives only the averaged gradient *for its own shard* — which is all it needs, since it only
updates its own shard. An all-reduce would hand every rank the whole gradient, most of which it
would throw away.

Communication volume per step is roughly `2P` (gather in forward + gather in backward) plus `P`
(reduce-scatter) versus DDP's `2P` for one all-reduce. So FSDP moves more data; it is a capacity
trade, not a free win, which is why it is used with fast interconnects and why the benchmark must
report both.

### Why the sharding unit is per-block

`fully_shard` is applied to **each transformer `Block` individually, then to the root module**.
Each `fully_shard` call creates one *unit* — the granularity at which parameters are gathered and
freed. With per-block units, peak transient memory is one block's parameters; with only a root
unit, FSDP would shard *storage* but still gather the entire model at once in the forward, which
defeats the purpose.

The trade-off is communication count: more units mean more, smaller collectives, which is why the
unit is a block rather than a single `Linear`.

**The tied embedding must stay in the root unit** (D8). `tok_emb` and `lm_head` are the same
tensor; a shared parameter split across two different sharding units gets gathered and freed on two
schedules, which is the kind of thing that fails quietly. Wrapping only `Block`s and the root gives
this for free — verified by test, not assumed.

### Ordering trap: build the optimizer *after* `fully_shard`

`fully_shard` replaces each parameter with a sharded `DTensor` **in place**. An optimizer
constructed beforehand would hold references to the pre-shard tensors and silently update nothing
that matters. The trainer therefore wraps first, then builds the optimizer.

Verified in a spike: after `fully_shard`, `named_parameters()` yields `DTensor`s whose
`to_local().numel()` sums to exactly `P/R` (10,880 of 21,760 on the nano config at R=2), and
`torch.nn.utils.clip_grad_norm_` computes the correct *global* norm across shards — both ranks
reported an identical 1.36097407341, so no hand-rolled collective is needed.

---

## Mixed precision: two mechanisms, one config field

`train.precision: fp32 | bf16` drives both. The distinction that matters:

- **`ModelConfig.dtype` is parameter *storage*** and stays `float32`.
- **`train.precision` is *compute*.**

Mixed precision means bf16 math with **fp32 master weights**: the optimizer step, weight decay and
gradient clipping all stay in fp32. That is what keeps small updates from being rounded away —
bf16 has 8 mantissa bits, so `w + small_update == w` far more often than in fp32.

### Single-device and DDP: `torch.autocast`

`torch.autocast(device_type, dtype=bfloat16)` around the forward. Matmuls and convolutions run in
bf16; reductions and anything numerically delicate stay fp32 by autocast's own op table.
Parameters are untouched, so the optimizer sees fp32 throughout.

**The loss needs no new code.** `GPT.forward` already promotes logits via
`torch.promote_types(dtype, float32)` before cross-entropy — the ISSUES #4 pattern from M1 — so
bf16 logits are upcast and the log-sum-exp over the vocabulary happens in fp32. That is the most
precision-sensitive op in the model, and the M1 decision covers it for free. Tested explicitly
rather than assumed.

### FSDP: `MixedPrecisionPolicy`

```python
MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
```

- `param_dtype=bf16`: shards are stored fp32 but **all-gathered as bf16**, halving the
  communication volume of the dominant collective. Autocast cannot do this — it converts at the op
  boundary, after the gather has already moved fp32 bytes.
- `reduce_dtype=fp32`: the gradient reduce-scatter accumulates in fp32. Summing many small bf16
  gradients loses precision exactly where it hurts, and the reduction is cheap relative to the
  gather.

This is why the milestone implements both mechanisms rather than using autocast everywhere: under
FSDP, the gather dtype *is* the memory and bandwidth story.

### Hardware note, recorded because it will bite

bf16 needs Ampere (sm_80) or newer. The planned Kaggle `T4 x2` is **sm_75 and has no bf16**, so
`precision: bf16` there is a hard `ConfigError` rather than a silent fallback to a different
numeric path. The FP32-vs-BF16 comparison the scope asks for needs an A100/L4-class card.
Locally, bf16 autocast works on both CPU and MPS (probed), so the plumbing is fully testable here.

---

## Checkpoints: two paths, deliberately

FSDP's state dict holds `DTensor`s, so M1's single-file `torch.save` of `model.state_dict()` is not
enough on its own. Both of these exist:

1. **Sharded (DCP)** — `torch.distributed.checkpoint.save`/`load` to a *directory*; each rank
   writes its own shard. Nothing is ever gathered, so saving a model too large for one device still
   works. This is what FSDP resume uses. A spike confirms the layout: `.metadata` plus one
   `__{rank}_0.distcp` per rank.
2. **Consolidated** — `StateDictOptions(full_state_dict=True, cpu_offload=True)` gathers a full
   fp32 state dict on rank 0 (and returns `{}` elsewhere, so only rank 0 writes), saved through
   M1's existing `save_checkpoint`.

The second path is not redundant. It keeps checkpoints **interchangeable across strategies**, which
is what lets an FSDP run be compared against the M1 single-device reference at all — and that
comparison is the milestone's actual evidence. It also means a sharded run's result can be loaded
for inference anywhere.

Note that `state_dict()` does not deduplicate tied weights: both `tok_emb.weight` and
`lm_head.weight` appear, pointing at the same values. That is true of plain `GPT.state_dict()` too,
which is precisely why the consolidated file loads into a single-device model unchanged.

---

## Known traps

| Trap | Handling |
|---|---|
| Device mesh auto-detection | Always `init_device_mesh(<explicit type>, (world_size,))`. Auto-detection finds MPS and crashes (ISSUES #2). |
| Optimizer built before `fully_shard` | Wrap first, then build. Covered by a mutation. |
| Tied weight split across units | Wrap `Block`s and the root only; assert the tying survives. |
| bf16 on sm_75 | Hard error, no fallback. |
| Dropout under data parallelism | `dropout=0` in every equivalence test (D18). |
| `max_memory_allocated` is CUDA-only | Peak bytes stay *pending hardware*; assert per-rank **parameter and optimizer-state counts** instead, which is the mechanism rather than a proxy. |

## Benchmarks

*Pending hardware.* Peak allocated/reserved memory per rank, DDP vs FSDP at equal model, batch and
precision, and FP32 vs BF16. A 2-rank gloo run on one CPU is slower than one process and cannot
measure GPU memory at all, so a local number would mislead rather than approximate. To be run on 2
NVIDIA GPUs and recorded in PROGRESS.md with config, hardware and commit.
