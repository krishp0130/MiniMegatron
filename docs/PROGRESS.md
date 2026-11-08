# Progress

Living status of the project. **Update this at the end of every working session**: tick off
checklist items, add a session-log entry, and rewrite "Next up". Scope lives in
[SCOPE.md](SCOPE.md); problems and design decisions live in [ISSUES.md](ISSUES.md).

## Current state

- **Active milestone:** 4 (Tensor parallelism) **complete** — all exit criteria met except the
  benchmarks, which are *pending NVIDIA hardware*. Next: Milestone 5 (Triton kernels), which
  **cannot be written on this machine at all** (no CUDA/Triton, ISSUES #3).
- **Last updated:** 2026-11-08
- **Test status:** 344 passed, 1 skipped (`pytest -q`, CPU, ~124 s), including 83 two-process gloo
  tests. The skip needs a CUDA device without bf16 support.
- **Verified hardware:** Apple M4 Pro (CPU + MPS), 2-process gloo. No NVIDIA GPU verification yet.

## Next up

**Milestone 5 — Triton kernels.** This is the first milestone that **cannot be developed locally**:
Triton needs an NVIDIA GPU and does not install on macOS (ISSUES #3). The reference implementation
it is checked against already exists — `rms_norm` in `model/layers.py` was written in M1 as that
reference — so the work splits cleanly:

- *Possible here:* the numerical test suite against `rms_norm` (shapes, dtypes, extreme inputs,
  float64 gradcheck), the config switch between the PyTorch and Triton paths, and the benchmark
  harness, all written and exercised against the reference with the kernel stubbed out.
- *Needs a GPU session:* the kernel itself, its backward, and every measurement.

Given that, Milestone 6 (reliability: regression suite, failure injection, CI) is the better next
target if no GPU session is booked — it is fully local, and it is where ISSUES #8 (pinning the
evaluation set so validation loss is comparable across world sizes) belongs.

Still open from earlier milestones and worth folding into whichever comes next:
- Every benchmark table is *pending hardware*: M2's 1-vs-2-rank throughput, M3's peak memory and
  FP32-vs-BF16, M4's TP-vs-DDP-vs-FSDP. One 2-GPU session covers nearly all of them; bf16 needs
  Ampere or newer, so a T4 will not do.
- A model-parallel RNG tracker, which is what dropout under TP needs (D18).

## Milestone checklist

### Milestone 0 — Setup ✅
- [x] GitHub repo (`krishp0130/MiniMegatron`, private)
- [x] Python 3.12 venv, `torch==2.14.1` pinned, editable install
- [x] Directory skeleton, `scripts/check_env.py`
- [x] Verified 2-process DDP and FSDP2 on CPU via gloo (see ISSUES.md for macOS quirks)
- [x] Context docs: CLAUDE.md, SCOPE.md, PROGRESS.md, ISSUES.md

### Milestone 1 — Functional GPT baseline ✅
- [x] Milestone scoped ([M1_gpt_baseline.md](milestones/M1_gpt_baseline.md))
- [x] Model config + validation (`config.py`, `model/config.py`, `experiment.py`, `configs/*.yaml`)
- [x] Reference attention + SDPA backend (`model/attention.py`)
- [x] Norms, MLP, transformer block (`model/layers.py`)
- [x] Full GPT model; param counts match formula (tiny 11,487,360 / small 51,237,888 / medium 124,475,904)
- [x] Synthetic dataset
- [x] Single-device training loop (AdamW, warmup + cosine, grad accumulation, clipping)
- [x] Checkpoint save/resume (model, optimizer, scheduler, RNG)
- [x] Loss-at-init ≈ ln V test, overfit test, bitwise resume test, CPU smoke test
- [x] `python -m minimegatron.train --config configs/tiny.yaml` works (CPU and MPS)
- [ ] *(stretch, deferred)* Real text data (`data/text.py`) — not required for sign-off

### Milestone 2 — Distributed data parallel ✅
- [x] Milestone scoped ([M2_ddp.md](milestones/M2_ddp.md))
- [x] Process-group setup/teardown (`distributed/{config,setup}.py`)
- [x] DDP wrapper, rank sharding via `IndexStream` (D17), `no_sync` grad accumulation
- [x] Rank-aware logging, metric aggregation, rank-0 checkpointing
- [x] DDP gradients == reference global-batch gradients (gloo test)
- [x] Educational explicit all-reduce demo (`collectives_demo.py`)
- [ ] 1 vs 2 GPU throughput benchmark *(pending hardware)*

### Milestone 3 — Memory optimization ✅
- [x] Milestone scoped ([M3_fsdp.md](milestones/M3_fsdp.md))
- [x] BF16 mixed precision (autocast for single/DDP, `MixedPrecisionPolicy` for FSDP — D22)
- [x] `strategy: auto|single|ddp|fsdp` switch (deferred from M2 by design)
- [x] FSDP2 with per-block `fully_shard` (D20)
- [x] Sharded checkpoint save/restore (torch.distributed.checkpoint) + consolidation path (D21)
- [x] FSDP convergence vs reference (gloo test)
- [x] Per-rank parameter and optimizer-state sharding verified on CPU
- [x] `docs/fsdp.md` (all-gather, reduce-scatter, sharding units, mixed precision, checkpoints)
- [ ] Peak memory per rank, DDP vs FSDP comparison *(pending hardware)*

### Milestone 4 — Tensor parallelism ✅
- [x] Milestone scoped ([M4_tp.md](milestones/M4_tp.md))
- [x] `f`/`g` autograd communication primitives + scatter/gather variants + unit tests
- [x] ColumnParallelLinear / RowParallelLinear vs `nn.Linear` tests
- [x] Parallel MLP and parallel attention (head partitioning)
- [x] Shard a full state dict into TP shards / gather back
- [x] TP forward + gradient equivalence vs single-device reference (gloo test)
- [x] TP-aware global gradient clipping (ISSUES #12)
- [x] TP config validation (`n_heads`, `d_ff` divisible by `tp_size`)
- [x] [`docs/tensor_parallel.md`](../docs/tensor_parallel.md)
- [ ] TP vs DDP vs FSDP benchmarks *(pending hardware)*

### Milestone 5 — Triton kernels *(requires NVIDIA GPU)*
- [ ] RMSNorm reference + Triton forward/backward
- [ ] Numerical + gradient tests across shapes/dtypes
- [ ] Kernel benchmarks, config switch, end-to-end integration

### Milestone 6 — Reliability engineering
- [ ] Numerical regression suite
- [ ] Failure injection (OOM, bad config, corrupt checkpoint, missing data, worker death)
- [ ] Automated benchmarks with regression thresholds
- [ ] CI (GitHub Actions) running CPU-safe tests

### Milestone 7 — Final documentation
- [ ] architecture.md, distributed_training.md, testing.md, benchmarks.md
- [ ] Reproducible experiment instructions + measured results tables

## Measured results

**No throughput numbers recorded**, deliberately. The loop prints `tokens_per_sec`, but short runs
include warmup and have no repeats or variance, and more importantly a 2-rank gloo run on one CPU
is *slower* than one process — it adds communication and buys no parallelism, so a local
measurement would be actively misleading rather than merely imprecise. The benchmark harness is
Milestone 6; the 1-vs-2-GPU comparison needs two NVIDIA GPUs (Kaggle's free `T4 x2` is the planned
venue).

**Correctness results**, which are the milestones' actual claims. tiny preset, CPU, 4 steps,
global batch 16 in every row:

| Run | Config | Step losses | Grad norms |
|---|---|---|---|
| 1 rank, single | micro=8 x accum=2 | 10.8652, 10.8624, 10.8589, 10.8704 | 0.901, 0.914, 0.866, 0.831 |
| 2 ranks, DDP | micro=8 x accum=1 | 10.8652, 10.8624, 10.8589, 10.8704 | 0.901, 0.914, 0.866, 0.831 |
| 2 ranks, FSDP2 | micro=8 x accum=1 | 10.8652, 10.8624, 10.8589, 10.8704 | 0.901, 0.914, 0.866, 0.832 |
| 2 ranks, TP=2 | micro=8 x accum=2 | 10.8652, 10.8624, 10.8589, 10.8704 | 0.901, 0.915, 0.866, 0.832 |

Identical, as every form of parallelism here requires. (Grad norms differ in the third or fourth
decimal from different reduction orders.) Only throughput differed. Note the TP row keeps the
preset's `accum=2` unchanged: TP ranks are not data-parallel ranks, so the global batch is
unaffected (D24).

**Tensor-parallel parameter counts**, tiny preset at `tp_size=2` — what is and is not split:

| | Full model | Per rank, TP=2 |
|---|---|---|
| total | 11,487,360 | 10,599,936 |
| non-embedding | 1,779,840 | 892,416 (0.5014x) |

Non-embedding nearly halves; it is not *exactly* half because norms and row-parallel biases stay
replicated. Total barely moves, because the 9.7M-parameter embedding table — 84% of this preset —
is replicated in the basic version. That is the argument for vocab-parallel embeddings, which
remain a documented stretch goal.

**Sharding, measured on CPU** (nano config, world_size 2) — the mechanism behind the memory
saving, which is measurable here even though bytes are not:

| Quantity | Full model | Per rank under FSDP2 |
|---|---|---|
| parameters | 21,760 | **10,880** (exactly half; shards partition it, nothing duplicated) |
| optimizer state (Adam m + v) | 43,520 | **21,760** |

DDP would hold all 65,280 elements on every rank; FSDP2 holds 32,640. That is the 16-bytes-per-
parameter figure being divided by the world size.

Record only numbers that were actually measured, with the config, hardware, and commit that
produced them.

## Session log

Newest first.

### 2026-11-08 — M4: tensor parallelism (complete, benchmarks pending hardware)

Whole-milestone pass. Scoped first in [milestones/M4_tp.md](milestones/M4_tp.md), which worked
through both ways to split a matmul, why the column->row pairing needs only one all-reduce per
pair, and why the `f`/`g` asymmetry falls out of the calculus rather than being a convention.

**Built.** `distributed/tensor_parallel.py`: the `f`/`g` conjugate pair plus scatter/gather
variants as `torch.autograd.Function`s over raw `torch.distributed` collectives (D23);
`ColumnParallelLinear` / `RowParallelLinear` with `from_linear` constructors that take this rank's
slice; `parallelize_model` surgery that preserves module names (D25); `tp_shard_dim` /
`shard_state_dict` / `gather_state_dict` / `gather_tp_state_dict`; `clip_grad_norm_tp`;
`validate_tp`. `distributed.strategy` gained a `tp` case and a `tp_size`; `setup.py` gained
`build_tp_mesh` (a 2-D `(dp, tp)` mesh even though dp is 1 here, as the seam for later).
`Trainer` dispatches on strategy, shards data over the *data-parallel* world only (D24), and
writes a per-rank checkpoint plus a gathered `_full.pt`.
[`docs/tensor_parallel.md`](../docs/tensor_parallel.md) written as the scope requires.

**Results: 344 passed, 1 skipped** (`pytest -q`, CPU, ~124 s), up from 285, including **83
two-process gloo tests**. Every exit criterion met but the benchmarks (*pending hardware*). This
also satisfies SCOPE.md success criterion 6 verbatim:
- Each primitive tested in both directions; both parallel layers tested against `nn.Linear` for
  output, input gradient and weight-gradient slice, with and without bias.
- TP loss and gradient norm match the single-device reference; **sharded** parameter gradients
  match the reference slice and **replicated** ones match in full and agree across ranks.
- A 5-step run tracks the reference loss curve exactly and ends at matching weights.
- Full state dict shards and gathers back exactly, for tp_size 1, 2 and 4.
- The real `torchrun` TP run matches the 1-rank run on the tiny preset (table above).

**Mutation check** — 11 injected, **11 caught**:

| Mutation | Failures |
|---|---|
| row-parallel bias added BEFORE the all-reduce | 1 |
| row-parallel bias split instead of replicated | 1 |
| row-parallel bias treated as sharded in `tp_shard_dim` | 1 |
| `f`: no all-reduce in the backward | 1 |
| `g`: no all-reduce in the forward | 1 |
| do not set the local head count | 1 |
| column-parallel splits the wrong dimension | 1 |
| TP clip: do not reduce the sharded squared norms | 1 |
| TP clip: also reduce the replicated norms (double counting) | 1 |
| shard data across the whole world, not the dp world | 1 |
| attention output reshaped to the full `d_model` | 1 |

(Counts are from `pytest -x`, so they mean "caught", not a total. First milestone where every
mutation was caught on the first pass.)

**Three things found:**
- **The attention reshape assumed the full `d_model`** (ISSUES #11). Every TP test failed
  instantly with a shape error. `CausalSelfAttention.forward` reshaped concatenated heads back to
  its input's width, but under TP this rank holds only `n_heads/tp_size` heads. It survived three
  milestones because nothing before M4 ever gave a layer a width different from `d_model` — D4
  made the *layers* replaceable, but one shape assumption outside them was not.
- **`clip_grad_norm_` was silently wrong** (ISSUES #12): 1.26595 per rank against a true global
  norm of 1.296081, while losses matched to six decimals. Under TP `parameters()` yields ordinary
  tensors that happen to be slices, with nothing marking them as sharded — unlike FSDP's
  `DTensor`s, so M3's experience actively misled me. `clip_grad_norm_tp` all-reduces the sharded
  squared norms and counts replicated ones once. **A correct loss does not imply correct
  clipping.**
- **`tp_size < world_size` would have trained silently wrong**, and was spotted by reasoning about
  the mesh rather than by a test. It implies a real data-parallel dimension with *no* gradient
  synchronization over it, so the dp groups would diverge. Since 2D TP+DP is out of scope, it is
  now refused with an error that says why.

A weight-comparison tolerance was also widened deliberately, after measuring rather than until
green: the parallel matmuls sum in a different order and Adam divides by sqrt(v), so a ~1e-8
gradient difference becomes ~1.4e-6 in a bias that started at exactly zero. The floor is absolute
(1e-5), because a relative tolerance buys nothing for elements near zero, and it is still 0.25% of
the largest bias.

### 2026-11-08 — M3: BF16 + FSDP2 (complete, memory/throughput benchmarks pending hardware)

Whole-milestone pass. Scoped first in [milestones/M3_fsdp.md](milestones/M3_fsdp.md), and the
riskiest parts were **spiked before any code was written** — a throwaway 2-rank script confirmed
that `fully_shard` shards to exactly P/R, that `clip_grad_norm_` computes a correct global norm
over `DTensor`s (both ranks: 1.36097407341), that tying survives sharding, and what a DCP
directory actually contains. That removed every open risk from the plan.

**Built.** `train.precision: fp32|bf16` with autocast in the trainer; `distributed.strategy:
auto|single|ddp|fsdp` (`auto` preserves every prior default); `distributed/fsdp.py` (`build_fsdp`
with per-block units, `MixedPrecisionPolicy`, and sharding introspection helpers);
`distributed/setup.py` gained `resolve_strategy` and `build_device_mesh` (always an explicit device
type — ISSUES #2); `distributed/checkpoint.py` with both the DCP sharded path and the consolidation
path; `Trainer` dispatches on strategy, builds the optimizer **after** wrapping, and uses FSDP's
`set_requires_gradient_sync` for accumulation. `docs/fsdp.md` written as the scope requires.

**Results: 285 passed, 1 skipped** (`pytest -q`, CPU, ~92 s), up from 241, including **54
two-process gloo tests**. All exit criteria met but the benchmarks (*pending hardware*):
- FSDP gradients, loss and gradient norm match the single-device reference; a 5-step run tracks the
  reference loss curve and ends at matching weights.
- Each rank holds exactly half the parameters *and* half of Adam's state (table above).
- bf16: parameters, gradients and optimizer state stay fp32; `Linear` runs in bf16 while norms,
  embeddings and the residual stream stay fp32; the loss is fp32; a bf16 nano model still overfits
  a fixed batch; bf16 also verified on MPS.
- Under FSDP, parameters are **gathered** as bf16 while **stored** as fp32 — the saving autocast
  cannot provide.
- DCP round trip resumes exactly (2N steps straight == N + checkpoint + fresh trainer + resume + N);
  the consolidated file loads into a plain single-device `GPT` with tying intact.
- The real `torchrun` FSDP run matches the 1-rank run on the tiny preset (table above).

**Mutation check** — 10 injected, 9 caught, 1 unverifiable on this hardware:

| Mutation | Failures |
|---|---|
| shard only the root, not each block | 1 |
| drop the FSDP mixed-precision policy | 1 |
| reduce gradients in bf16 instead of fp32 | 1 |
| build the optimizer before `fully_shard` | 1 |
| consolidate without gathering the full state dict | 1 |
| skip autocast entirely | 1 |
| compute the loss in bf16 (drop the promote) | 1 |
| allow ddp/fsdp with a single rank | 1 |
| never suppress FSDP's reduce-scatter | 1 |
| accept bf16 on CUDA without bf16 support | **0 — needs a CUDA device without bf16; test is skipped here** |

(Counts are from `pytest -x`, so they mean "caught", not a total.)

**Four things found, in the order they bit:**
- A bf16 run printed losses *identical to fp32 at four decimals* while running 100x slower, which
  looked like autocast was inert. It is not: autocast keeps norms, embeddings and the residual
  stream in fp32 and only downcasts matmuls, and at initialization that moves the loss by <1e-4.
  Recorded as ISSUES #10, and the test now pins the dtype at four points rather than checking one
  activation. **Lesson: verify a precision mode by dtype per op, never by whether the loss moved.**
- The first `no_sync` test for FSDP assumed `param.grad` would hold a partial mid-group. It is
  `None`: FSDP accumulates in an internal *unsharded* buffer and only populates `.grad` at the
  reduce-scatter. Probed both branches, then asserted the discriminating fact (suppressed →
  `None`, synced → `DTensor`).
- The first mutation pass had **two surviving mutations and one bad mutation**. The bad one built
  an extra optimizer while leaving the real one in place, so it tested nothing — rewritten and then
  caught. The two real gaps: nothing asserted the loss dtype under FSDP+bf16 (where the ISSUES #4
  promote is genuinely load-bearing, since autocast — which has `cross_entropy` on its own fp32
  list — is not active), and nothing tested FSDP's sync suppression at all. Both now covered.
- Autocast under FSDP would be redundant double-casting, since `MixedPrecisionPolicy` already
  casts parameters and forward inputs. `_autocast()` now returns a null context under FSDP (D22).

Third milestone running where mutation testing found something the suite missed — and the first
where it also caught a flaw in a *test of mine*, which is a useful reminder that a mutation that
survives is as likely to be a bad mutation as a missing test.

### 2026-11-08 — M2: distributed data parallel (complete, benchmarks pending hardware)

Whole-milestone pass, as authorized for M1–M2. Scoped first in
[milestones/M2_ddp.md](milestones/M2_ddp.md) (the all-reduce/large-batch equivalence worked
through, DDP's bucketing and overlap, why `no_sync` is safe, why clipping must follow
synchronization).

**Built.** `distributed/config.py` (`DistributedConfig`: backend `auto|gloo|nccl`, explicit
timeout) and `distributed/setup.py` (`DistInfo`, `init_distributed` / `destroy_distributed`,
`barrier`, `all_reduce_mean`). `init_distributed` is a no-op without torchrun's environment, so
one entry point serves both single-process and distributed runs and **every M1 path is unchanged**.
`BatchStream` gained rank sharding (D17). `Trainer` gained a `DistInfo`, DDP wrapping with
`raw_model` kept separate from the wrapper, `no_sync` on all but the last micro-batch, clipping
after the synchronizing backward, loss/eval aggregation, cluster-wide token counting, and
rank-0-only logging (`NullLogger` elsewhere) and checkpointing with a barrier.
`distributed/collectives_demo.py` does the averaging by hand, outside the training path.

**Results: 241 passed** (`pytest -q`, CPU, ~47 s), up from 201, including **30 two-process gloo
tests**. All exit criteria met but the benchmark (*pending hardware*):
- 2 ranks x micro=4 gradients match 1 process x micro=8 for every parameter, the tied embedding
  included; every rank holds identical gradients after the all-reduce.
- `no_sync` 2 ranks x accum=2 x micro=2 matches 1 process x micro=8.
- The hand-rolled all-reduce reproduces DDP's gradients exactly.
- A 2-rank checkpoint loads straight into a plain single-device `GPT`.
- `torchrun --nproc_per_node=2` runs end to end, logs from rank 0 alone, all ranks exit 0.
- On the **tiny preset**, 1 rank and 2 ranks at equal global batch give identical losses and
  gradient norms (table above).

**Mutation check** — 8 bugs injected, each caught, then reverted:

| Mutation | Failures |
|---|---|
| give every rank the same shard | 11 |
| log the rank-local loss instead of the global one | 3 |
| count tokens for one rank only | 3 |
| never suppress the all-reduce (drop `no_sync`) | 2 |
| let every rank write the log | 2 |
| sum gradients without dividing by world size | 1 |
| checkpoint the DDP wrapper instead of the module | 1 |
| average the eval loss per rank only | 1 *(0 on the first pass — see below)* |

**Three things found, in the order they bit:**
- The 2-process tests hung. DDP was fine — the *harness* was: torch hands tensors between
  processes via shared memory, which needs the sender alive until the parent maps the storage, and
  the workers exited right after `queue.put`. Switched to file-based results (ISSUES #7).
- A test that counted DDP's synchronizations by patching `reducer.prepare_for_backward` failed:
  it is a read-only C++ binding. Replaced with a behavioural test that is stronger — mid-group the
  ranks' gradients must disagree, at the end they must agree (ISSUES #9).
- The first mutation run showed **eval-loss aggregation was untested** (0 failures when removed).
  Added a test asserting the per-rank losses genuinely differ and the reported value is their mean;
  the mutation now fails. This is the second milestone in a row where mutation testing found a gap
  the suite missed.

**Known limitation recorded, not papered over** (ISSUES #8): `eval_steps` counts batches per rank,
so a 2-rank run evaluates twice the validation data and reports a slightly different number
(10.8641 vs 10.8636). Training loss and gradients are exactly comparable, which is what the
equivalence tests pin; pinning the eval set to a fixed sample count belongs with M6's regression
suite.

Also decided: `global_batch_size` is **validated, not derived** (D16) — deriving would make one
config mean different things at different world sizes. The error message teaches the right fix
(reduce `grad_accum_steps` to hold the global batch fixed while adding ranks).

### 2026-11-07 — M1 steps 4–6: GPT model, synthetic data, training engine (M1 complete)

Run as a whole-milestone pass at the owner's request, rather than the usual step-at-a-time
cadence. The `data/text.py` stretch item was deliberately skipped.

**Step 4 — `model/transformer.py`.** `GPT`: token + learned position embeddings, `Block` × L,
final norm, tied LM head, cross-entropy with `ignore_index=-100` computed in at least float32 via
`torch.promote_types` (the #4 pattern). GPT-2 init, with `attn.out_proj` and `mlp.fc2` re-inited
at `0.02/√(2L)` in a separate pass over the blocks. `num_params(include_embeddings=...)` on both
the model and `ModelConfig`, so the doc's non-embedding column is testable.

**Step 5 — `data/{synthetic,config}.py`.** `RandomTokenDataset` (unlearnable; for throughput
work) and `MemorizationDataset` (fixed sequences; for the overfit test), both pure functions of
`(seed, index)` through a SplitMix64 mixer. `IndexStream` + `BatchStream` define the sample order
over *samples rather than batches* (D12), which is what makes the accumulation-equivalence and
resume tests exact.

**Step 6 — `training/` and `train.py`.** `TrainConfig`; decay/no-decay param groups; `lr_at` as a
pure function behind `LambdaLR`; atomic checkpoints (`.tmp` + `os.replace`) carrying optimizer
moments, scheduler, step, tokens, resolved config and all four RNGs; `JSONLLogger`; `Trainer`
with 1/accum-scaled gradient accumulation, pre-clip grad-norm logging, device sync before timing,
and no silent CPU fallback; a `--config/--resume/--set` CLI whose overrides are validated by the
same path as the YAML. `data:`/`train:` sections added to `experiment.py` (both optional) and to
the three presets.

**Results: 201 passed** (`pytest -q`, CPU, ~15 s), up from 92. tiny reports exactly 11,487,360
parameters / 1,779,840 non-embedding, matching the design doc's table. Loss at init 10.86 vs
ln 50304 = 10.83. End-to-end runs verified on **both CPU and MPS**. All six M1 exit criteria met.

**Mutation check** — 9 bugs injected by hand, each caught, then reverted:

| Mutation | Failures |
|---|---|
| drop the causal mask | 15 |
| untie the embedding and LM head | 15 |
| restore the model but not the optimizer moments | 3 |
| shift the targets the wrong way | 2 |
| drop the 1/grad_accum loss scaling | 1 |
| remove the residual 1/√(2L) init scaling | 1 |
| skip RNG restore on resume | 1 |
| ignore the live schedule on resume | 1 |
| drop the global-batch-size cross-check | 1 |

**Two bugs found while building, both by tests rather than by reading:**
- An import cycle between `experiment.py` and `training/__init__.py` (ISSUES #5), fixed with a
  `TYPE_CHECKING`-only import.
- The learning rate after a resume came from the *saved* schedule, not the live one, so resuming
  with a changed `max_steps` (which the CLI test does) took one step on the old curve (D14). The
  first attempted fix was indistinguishable from doing nothing and was caught by the mutation
  run, not by the test suite.

Also recorded: a test-design rule after `evaluate()` was wrongly suspected of perturbing training
— two trainers in one process share one global torch RNG and must be driven sequentially
(ISSUES #6). M2's DDP equivalence tests will meet the same trap.

### 2026-10-25 — M1 step 3: norms, MLP, block
- `model/layers.py`: functional `rms_norm` (the M5 Triton reference) + `RMSNorm` module,
  `make_norm` (LayerNorm/RMSNorm per config), position-wise GELU `MLP` (`fc1`/`fc2`), pre-norm `Block`.
- Tests: `tests/unit/test_layers.py` (norm factory, MLP is position-wise, block param count ==
  config formula for 4 variants, block is identity when sublayer outputs are zeroed, causality,
  gradients reach all params) and `tests/numerical/test_norms.py` (vs. `torch.nn.RMSNorm` on 5
  shapes incl. non-power-of-two widths, bf16 vs. fp32 computation, unit-RMS property,
  zero/tiny/huge inputs, float64 gradcheck, exact-vs-tanh GELU < 1e-3).
- **Bug found by gradcheck:** `x.float()` downcast float64 inputs to float32. Fixed by promoting
  to at least float32 (ISSUES.md #4).
- **92 passed.** Mutation check: post-norm block → 1 failure (identity test); RMSNorm without
  eps → 5 failures.

### 2026-10-25 — M1 step 2: causal self-attention
- `model/attention.py`: `split_qkv` (per-head `[h, 3, d_h]` layout), `reference_attention`
  (explicit math, non-persistent causal-mask buffer), and `CausalSelfAttention` with
  `reference`/`sdpa` backends sharing identical state_dict keys.
- Tests: `tests/unit/test_attention.py` (shapes, behavioral causality, seq_len guard, state_dict
  keys, whole-head split for TP, dropout train/eval) and `tests/numerical/test_attention_backends.py`
  (reference vs. SDPA forward and backward across 3 shapes including head_dim 48 and 1 head,
  extreme input scales, float64 gradcheck of the function and module, MPS vs. CPU).
- **63 passed** (CPU; the 2 MPS tests ran on the M4 Pro).
- Mutation check (bugs injected by hand, then reverted): deleting the mask → 13 failures;
  dropping 1/√d_h → 9 failures; GPT-2 `[3, h, d_h]` layout → 1 failure (only the layout test,
  because both backends share `split_qkv`; see ISSUES.md D11).

### 2026-10-23 — M1 step 1: model config
- `ModelConfig` (frozen, validates on construction, reports all errors at once, analytic
  `num_params()`), strict `from_dict` (rejects unknown keys with "did you mean"), `load_yaml`,
  `load_config` → `ExperimentConfig(model)`, and tiny/small/medium YAML presets.
- `tests/conftest.py` (`REPO_ROOT`, `nano_config` fixture), `tests/unit/test_config.py`.
- **31 tests passed** on CPU. Preset counts in code: tiny 11,487,360 / small 51,237,888 /
  medium 124,475,904, and the GPT-2 anchor 124,439,808 is now asserted in a test.
- Removed Claude co-author trailers from history (force-push); CLAUDE.md now forbids them.

### 2026-10-23 — Milestone 1 scoped
- Wrote `docs/milestones/M1_gpt_baseline.md`: build order, attention/norm/MLP/init math,
  optimizer and schedule, checkpoint and resume design, test list, exit criteria.
- Fixed presets (vocab 50304): tiny 11.49M, small 51.24M, medium 124.48M params. The analytic
  formula reproduces GPT-2 small's 124,439,808 at V=50257 (checked by hand, not yet in code).
- New decisions D6–D9 in ISSUES.md (per-head QKV layout, vocab padding, weight tying,
  index-deterministic data).

### 2026-10-23 — Project setup
- Created repo, venv (Python 3.12.14, torch 2.14.1, numpy 2.5.4, pytest 9.1.1, tokenizers 0.23.3,
  tensorboard), editable install, directory skeleton, env check script.
- Hardware audit: Apple M4 Pro, 24 GB, MPS available, **no CUDA/NCCL/Triton**.
- Probed distributed on CPU: 2-process DDP and FSDP2 both work under gloo after two macOS
  workarounds (ISSUES.md #1, #2).
- Added tensor parallelism to scope as Milestone 4.
- Wrote CLAUDE.md, SCOPE.md, PROGRESS.md, ISSUES.md.
