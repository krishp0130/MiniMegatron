# FSDP2: sharding parameters, gradients and optimizer state

How MiniMegatron's `strategy: fsdp` works, and why each piece is the way it is. The milestone
design notes are in [milestones/M3_fsdp.md](milestones/M3_fsdp.md); this is the standalone
explanation SCOPE.md section 5 asks for.

## The problem DDP does not solve

Data parallelism (M2) replicates the model on every rank and splits the *batch*. Every rank holds
a full copy of everything, so for `P` parameters in FP32 with AdamW, each rank needs:

| What | Bytes per parameter |
|---|---|
| parameters | 4 |
| gradients | 4 |
| Adam `m` (first moment) | 4 |
| Adam `v` (second moment) | 4 |
| **total** | **16** |

…before a single activation. The medium preset (124M parameters) needs ~2 GB per rank just to
exist. Adding GPUs under DDP buys throughput and **nothing at all** in capacity: the largest model
you can train is whatever fits on one device.

FSDP2 shards all four across `R` ranks, so each holds `16P/R` bytes plus transient space for
whichever unit is currently in use. That is the whole proposition, and it is a *trade*: the memory
comes back as communication.

## All-gather and reduce-scatter

A rank under FSDP owns only a **slice** of each parameter, which changes both collectives.

```
forward, per unit:    all-gather parameter shards -> full params -> compute -> free the full params
backward, per unit:   all-gather params again -> compute grads -> reduce-scatter grads
```

**Why reduce-scatter and not all-reduce.** DDP all-reduces gradients: every rank ends up with the
complete averaged gradient, which it needs because it updates every parameter. An FSDP rank only
updates *its own shard*, so it only needs the averaged gradient for that shard. Reduce-scatter
does the reduction and the distribution in one step, delivering exactly that. An all-reduce would
hand every rank the entire gradient, and every rank would discard almost all of it.

**The cost.** Per optimizer step, FSDP moves roughly `2P` (gather in forward, gather again in
backward) plus `P` (reduce-scatter), versus DDP's `2P` for a single all-reduce. FSDP moves *more*
data. It is a capacity trade, which is why it belongs on fast interconnects and why the benchmark
must report throughput alongside memory rather than claiming a free win.

The second gather in backward is avoidable by keeping parameters unsharded between forward and
backward (`reshard_after_forward=False`), trading memory back for bandwidth. We keep the default —
resharding — because memory is the point of the milestone.

## Sharding units, and why they are transformer blocks

`fully_shard` is applied to **each `Block` individually, then to the root module**. Each call
creates one *unit*: the granularity at which parameters are gathered before use and freed after.

- **Per-block units:** peak transient memory is one block's parameters. This is what makes the
  saving real.
- **Root unit only:** parameters would be *stored* sharded, but the forward would gather the
  entire model at once, so peak memory would be as bad as DDP. Storage sharding without gather
  granularity buys nothing.
- **Finer than a block** (every `Linear` its own unit): more, smaller collectives, each with fixed
  overhead, for a saving that is already small relative to one block.

A block is the natural unit because it is the repeating structure, and because its parameters are
used together within a contiguous stretch of the forward.

### The tied embedding must stay in the root unit

`tok_emb.weight` and `lm_head.weight` are the *same tensor* (D8). A shared parameter split across
two different sharding units would be gathered and freed on two schedules — which does not crash,
it produces subtly wrong gradients. Wrapping only `Block`s and the root gives the right answer for
free, because neither embedding is inside a block. `tests/distributed/test_fsdp_sharding.py`
asserts the tying survives rather than trusting that argument.

## Ordering trap: build the optimizer after sharding

`fully_shard` replaces each parameter with a sharded `DTensor` **in place**. An optimizer
constructed before the call holds references to the pre-shard tensors, so it would allocate
full-size `m` and `v` and update tensors the model no longer uses. The failure mode is quiet: the
run trains, and the memory saving silently does not happen. `Trainer` therefore wraps first and
builds the optimizer second, and a test measures per-rank optimizer-state size to pin it.

## Gradient clipping and accumulation

**Clipping** must compute a global L2 norm across shards. `torch.nn.utils.clip_grad_norm_` handles
`DTensor` gradients and performs the reduction itself — verified, not assumed: a test asserts the
FSDP gradient norm equals the single-device reference's.

**Accumulation** uses FSDP's own switch, `set_requires_gradient_sync(False)` (plus
`set_is_last_backward(False)`), for every micro-batch but the last, so there is one reduce-scatter
per optimizer step rather than one per micro-batch. The trade-off differs from DDP's `no_sync`:
unreduced FSDP gradients accumulate **unsharded**, so suppressing the collective costs memory to
save bandwidth. Correctness is unaffected either way, since reduction and accumulation are both
sums and sums commute.

## Mixed precision: why FSDP needs its own mechanism

`train.precision: bf16` means two different things depending on strategy, and they are not
interchangeable.

**Single-device and DDP — `torch.autocast`.** Matmuls run in bf16; normalization and other
reduction-heavy ops stay fp32 by autocast's op table; `Embedding` is untouched. Because the
residual stream is `fp32 + bf16`, it promotes back to fp32 and is never downcast. Parameters stay
fp32, so the optimizer sees fp32 throughout.

**FSDP — `MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)`.**

- `param_dtype=bf16`: shards are *stored* fp32 but **all-gathered as bf16**, halving the volume of
  FSDP's dominant collective. Autocast cannot do this — it converts at the op boundary, by which
  point the gather has already moved fp32 bytes. Under FSDP the gather dtype *is* the bandwidth
  story.
- `reduce_dtype=fp32`: the gradient reduce-scatter accumulates in fp32. Summing many small bf16
  gradients loses precision exactly where it matters, and the reduction is cheap next to the
  gather.

Because the policy also casts forward inputs, autocast is redundant under FSDP and the trainer
skips it there.

**Either way, master weights are fp32.** The optimizer step, weight decay and gradient clipping
all happen in fp32. bf16 has 8 mantissa bits, so `w + small_update == w` far more often than in
fp32 — keeping updates in low precision is how mixed-precision training quietly stops converging.

**The loss is always fp32.** `GPT.forward` promotes logits with
`torch.promote_types(dtype, float32)` before cross-entropy, so the log-sum-exp over the vocabulary
— the most precision-sensitive operation in the model — never runs in bf16. That came from a
Milestone 1 bug fix (ISSUES #4) and covers Milestone 3 for free.

**Hardware.** bf16 needs compute capability 8.0+ (Ampere). On an older card such as a T4 (sm_75),
`precision: bf16` is a hard error rather than a silent fallback to different numerics. Locally,
bf16 autocast works on both CPU and MPS, so the plumbing is fully testable without a GPU.

## Checkpoints: two formats, both written

An FSDP `state_dict` holds `DTensor`s, so the single-file `torch.save` used by single-device and
DDP runs is not sufficient. `strategy: fsdp` writes both of these per checkpoint:

### `ckpt_NNNNNN/` — sharded, via `torch.distributed.checkpoint`

```
ckpt_000002/
  .metadata          how the shards fit together
  __0_0.distcp       rank 0's shard of params + optimizer state
  __1_0.distcp       rank 1's shard
  meta.pt            step, tokens_seen, scheduler state, resolved config
  rng_0.pt           rank 0's RNG streams
  rng_1.pt           rank 1's RNG streams
```

Nothing is ever gathered, so a model too large for one device still saves. This is what resume
uses. **RNG is stored per rank**, not just from rank 0: random state differs by rank, so restoring
one rank's streams everywhere would make a resumed run diverge from an uninterrupted one — the
same reasoning as M1's RNG capture, extended to a cluster.

### `ckpt_NNNNNN_full.pt` — consolidated

`StateDictOptions(full_state_dict=True, cpu_offload=True)` all-gathers every shard onto rank 0 as
plain fp32 tensors (other ranks get an empty dict, so only rank 0 writes), saved through the same
writer single-device runs use.

This is not redundant. It keeps checkpoints **interchangeable across strategies**, which is what
lets an FSDP run be compared against the single-device reference at all — and that comparison is
the milestone's actual evidence. It also means a sharded run's result can be loaded for inference
anywhere, with no FSDP in the loading process.

Note that `state_dict` does not deduplicate tied weights: both `tok_emb.weight` and
`lm_head.weight` appear. That is true of a plain `GPT.state_dict()` too, which is precisely why the
consolidated file loads into a single-device model with no key munging.

## Running it

```bash
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    -m minimegatron.train --config configs/tiny.yaml \
    --set distributed.strategy=fsdp --set train.grad_accum_steps=1
```

`grad_accum_steps=1` keeps the global batch at the preset's declared 16 across two ranks
(8 × 1 × 2); the trainer refuses to start otherwise (D16).

Device meshes are always built with an **explicit** device type
(`init_device_mesh("cpu", (world_size,))`). Letting FSDP build a default mesh means auto-detecting
the accelerator, which finds MPS on an Apple machine and crashes (ISSUES #2).

## What is measured, and what is not

On a single CPU, `torch.cuda.max_memory_allocated` does not exist and two gloo ranks are slower
than one process, so **peak-memory bytes and throughput are pending NVIDIA hardware** and are not
estimated here.

What *is* verified, on CPU, is the mechanism rather than a proxy for it: each rank's **parameter
and optimizer-state element counts**. At `world_size=2` on the nano config, each rank holds 10,880
of 21,760 parameters and 21,760 of 43,520 optimizer-state elements — exactly half of each, with the
shards partitioning the model and nothing duplicated.
