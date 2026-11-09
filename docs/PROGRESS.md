# Progress

Living status of the project. **Update this at the end of every working session**: tick off
checklist items, add a session-log entry, and rewrite "Next up". Scope lives in
[SCOPE.md](SCOPE.md); problems and design decisions live in [ISSUES.md](ISSUES.md).

## Current state

- **All seven milestones addressed.** M1–M4, M6 and M7 are complete. M5 is complete except for the
  kernel itself, which is **written but has never been executed** (ISSUES #16) — there is no NVIDIA
  GPU on this machine.
- **Remaining work is hardware-gated, not unwritten.** Four of SCOPE.md §13's eleven
  Definition-of-Done criteria need NVIDIA GPUs; see the table in the README and the commands in
  [benchmarks.md](../docs/benchmarks.md). The harness, configs, baseline store and
  backend-parameterized kernel tests all exist, so each pending item is a command.
- **Last updated:** 2026-11-08
- **Test status:** 463 passed, 37 skipped (`pytest -q`, CPU, ~125 s). By suite: 261 unit, 89
  distributed (two-process gloo), 73 kernels, 34 fault-injection, 31 numerical, 12 regression.
  All 37 skips need CUDA hardware; 34 of them are the Triton-parameterized kernel tests
  (ISSUES #16).
- **Verified from a cold install:** a fresh venv that had never seen this project installs with
  `pip install -e ".[dev]"` and runs the full suite to the same result.
- **CI is passing, not merely configured.** Recent GitHub Actions runs on `main`:
  `42536192235` (M7, 6m39s), `42532323921` (M5, 7m37s), `42525629959` (M6, 8m28s) — all success.
- **Verified hardware:** Apple M4 Pro (CPU + MPS), 2-process gloo. No NVIDIA GPU verification yet.
- **CI:** GitHub Actions runs the CPU-safe suite — the gloo tests included — on every push.

## Next up

The project is wrapped. What is left needs hardware, and none of it needs new code:

| Pending | Command | Needs |
|---|---|---|
| Triton kernel correctness (ISSUES #16) | `pytest tests/kernels -q` | 1 NVIDIA GPU |
| Triton vs reference runtime/bandwidth | `python -m minimegatron.benchmarks.kernels` | 1 NVIDIA GPU |
| Single-GPU training (DoD #3) | `python -m minimegatron.train --config configs/tiny.yaml` | 1 NVIDIA GPU |
| 1 vs 2 GPU throughput (DoD #4) | `benchmarks.run` ×2 | 2 NVIDIA GPUs |
| Peak memory, DDP vs FSDP (M3) | `benchmarks.run` with `*_2gpu.yaml` | 2 NVIDIA GPUs |
| FP32 vs BF16 (M3) | `--set train.precision=bf16` | **Ampere+** (a T4 is sm_75, no bf16) |
| TP vs DDP vs FSDP (M4) | `benchmarks.run` ×3 | 2 GPUs, ideally ± NVLink |

Expect the kernel's first real run to surface bugs; that is the normal outcome for code written
without ever executing it, and the suite is arranged so they land as test failures rather than as
silently wrong training.

Deferred by choice, in the order worth doing them:
1. Vocab-parallel embedding and cross-entropy — the largest remaining TP memory win (84% of the
   tiny preset's parameters are a replicated token table).
2. A model-parallel RNG tracker, which unblocks dropout under TP (D18).
3. Real text data and sampling (`data/text.py`, an M1 stretch item).
4. 2D TP+DP composition — the `(dp, tp)` mesh and the strategy switch are the seams for it.
5. Profiler traces through the benchmark harness (`torch.profiler` is already a dependency).

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

### Milestone 5 — Triton kernels *(partial: kernel unverified, ISSUES #16)*
- [x] Milestone scoped ([M5_triton.md](milestones/M5_triton.md))
- [x] RMSNorm PyTorch reference (written in M1 for exactly this)
- [x] Triton forward kernel + `dx` backward kernel — **written, NEVER EXECUTED**
- [x] Explicit autograd strategy: Triton `dx`, PyTorch `dw`, documented as a decision (D31)
- [x] Numerical + gradient tests across shapes/dtypes/extremes, parameterized over backend —
      the reference passes; the 34 Triton parameters skip, stating why
- [x] float64 `gradcheck` of the reference
- [x] Config switch `model.norm_backend: auto|pytorch|triton` with no silent fallback (D32)
- [x] Kernel benchmark CLI (`benchmarks.kernels`), which refuses clearly without a GPU
- [ ] Triton numerical equivalence vs the reference *(pending hardware)*
- [ ] Kernel runtime/bandwidth measurements *(pending hardware)*

### Milestone 6 — Reliability engineering ✅
- [x] Milestone scoped ([M6_reliability.md](milestones/M6_reliability.md))
- [x] Validation loss comparable across world sizes (ISSUES #8 closed; exposed #13)
- [x] Benchmark harness: warmup, repeats, median + dispersion, JSON *and* Markdown (D28)
- [x] Numerical regression suite: golden losses, gradient norms, param counts, state-dict keys
- [x] Performance regression tool with a versioned baseline store, warning not asserting (D27)
- [x] Failure injection (OOM, bad config, invalid `tp_size`, corrupt checkpoint, missing files,
      unsupported hardware, **worker death mid-run**)
- [x] Actionable diagnostics, asserted as text rather than just "it raised" (D29)
- [x] CI (GitHub Actions) running CPU-safe tests, gated on the collected test count (D30)
- [x] The CLI documented in SCOPE.md §10 now exists (`configs/{benchmark,ddp_2gpu,fsdp_2gpu,tp_2gpu}.yaml`)

### Milestone 7 — Final documentation ✅
- [x] Milestone scoped ([M7_documentation.md](milestones/M7_documentation.md))
- [x] [architecture.md](../docs/architecture.md) — layering, the model, config, the loop, the seams
- [x] [distributed_training.md](../docs/distributed_training.md) — the three strategies side by side
- [x] [testing.md](../docs/testing.md) — six suites, mutation testing, test-design rules
- [x] [benchmarks.md](../docs/benchmarks.md) — method, measured CPU results, pending commands
- [x] Installation instructions and a documentation index in the README
- [x] Known limitations and future work, stated plainly
- [x] Honest Definition-of-Done assessment against SCOPE.md §13, item by item

## Measured results

### CPU throughput (measured, Milestone 6)

First recorded numbers. tiny preset, Apple M4 Pro **CPU**, commit `7b8313e`, 3 warmup steps
discarded, 3 repeats of 10 steps, median reported with IQR. Produced by
`python -m minimegatron.benchmarks.run`.

| Label | Strategy | Precision | Ranks | Batch | tokens/s (median) | IQR | ms/step |
|---|---|---|---|---|---|---|---|
| tiny-cpu-fp32 | single | fp32 | 1 | 8 | **18,166.7** | 389.8 | 112.73 |
| tiny-cpu-bf16 | single | bf16 | 1 | 8 | **151.4** | 0.7 | 13,523.86 |
| tiny-cpu-ddp2 | ddp | fp32 | 2 | 16 | **9,811.2** | 26.0 | 417.48 |
| tiny-cpu-fsdp2 | fsdp | fp32 | 2 | 16 | **8,673.3** | 448.9 | 472.25 |
| tiny-cpu-tp2 | tp | fp32 | 2 | 16 | **5,175.2** | 1,030.6 | 791.47 |

**Read these as relative overhead, not as scaling.** Two ranks on one CPU contend for the same
cores *and* add communication, so a 2-rank row being slower than the 1-rank row says nothing about
what two GPUs would do. What the table does show is the **ordering** — single > DDP > FSDP > TP —
which matches the prediction recorded in each milestone's design doc *before* measuring: FSDP adds
gathers and a reduce-scatter on top of DDP's single all-reduce, and TP all-reduces twice per block
per direction, so TP is expected to be last without NVLink-class bandwidth.

The bf16 row is a real measurement of **emulated** bf16 on Apple silicon (~120x slower) and
implies nothing about bf16 on a GPU, where it is a speedup. GPU memory is absent from the table
because `torch.cuda.max_memory_allocated` is CUDA-only; it is recorded as `null`, never estimated.

### Still pending hardware

Everything in the "Next up" table. The harness, the strategy configs and the baseline store are in
place, so these are a handful of commands on a 2-GPU box rather than new work.

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

### 2026-11-08 — End-to-end verification (two real bugs found and fixed)

The first single pass over the whole project: a cold install, the full suite, **every documented
command run verbatim**, the cross-strategy checkpoint path through the CLI, the failure modes from
the command line, and determinism at the CLI boundary. 26 scripted checks, plus the suite.

Everything the plan listed was run. **It found two real bugs that 60 mutations and 455 tests had
both missed**, and both lived in seams between units rather than inside one:

- **ISSUES #17 — a TP checkpoint gathered the model but not the optimizer.** Resuming a 2-rank TP
  run's `_full.pt` in a single-device run loaded cleanly and then died on the *first optimizer
  step*: `The size of tensor a (288) must match the size of tensor b (576)`. The file held full
  weights beside rank 0's **sharded** Adam moments — 10,599,936 elements against the model's
  11,487,360. FSDP's consolidation had always gathered both; TP's branch passed only
  `model_state=`. Fixed with `gather_tp_optimizer_state_dict`, which all-gathers each moment along
  the dimension its parameter is sharded on. The unit tests had asserted "the consolidated file
  loads into a plain `GPT`" — and it did, because `load_state_dict` only checks the model.
  **"It loads" is not "it works":** the regression test now takes an optimizer step, which is the
  line that used to raise.
- **ISSUES #18 — the CLI leaked tracebacks for setup errors.** Five documented failure modes
  printed the correct message and then a full traceback, because `train.py` caught `ConfigError`
  only around config *loading*; anything from `Trainer.__init__` or `--resume` escaped. Fixed by
  extending the handler to trainer construction and resume (and the same in `benchmarks/run.py`),
  while deliberately leaving `fit()` outside — a crash during training is a bug and must stay
  loud. A test asserts that distinction.

**Results.** 26/26 scripted checks pass. Cold install: a venv that had never seen this project
installs with `pip install -e ".[dev]"` and runs the suite to **455 passed, 37 skipped** (463 now,
after the tests added here). Every documented command exits 0, or non-zero *by design* for the
seven refusals. A TP checkpoint and an FSDP checkpoint each resume a single-device run — and both
produce identical continuation losses, which is further cross-strategy evidence. Two identical CLI
runs produce identical metrics.

**One false alarm worth recording:** the `global_batch_size` mismatch check appeared to leak a
traceback under torchrun. It was torchrun's own — it raises `ChildFailedError` and prints a
traceback plus a "Root Cause" block for any non-zero worker exit. Our ranks each printed one clean
line. The harness check was tightened to look for a frame in *our* source rather than the word
"Traceback" anywhere.

**Housekeeping.** The README told readers to `pip install -e ".[dev,text,logging]"`, pulling in
`tokenizers` and `tensorboard` — neither of which the project imports, since the real-text dataset
was deferred and metrics are JSONL. The extras stay declared (they are the hook for planned work)
but the install line is now `[dev]`, with the others explained. Three stale `.gitkeep` files
removed from directories that now have content. CI's three passing runs recorded above with their
IDs, since PROGRESS had been claiming CI ran on every push without citing one.

### 2026-11-08 — M7: final documentation (project wrapped)

Scoped in [milestones/M7_documentation.md](milestones/M7_documentation.md), whose single rule was
**no claim may exceed its evidence**: six milestones of discipline about that get undone by one
summary document that rounds "written but never executed" up to "implemented", or quotes a CPU
figure without saying it is CPU.

**Written.** The four documents SCOPE.md §11 names, each linking to code rather than restating it:
- [architecture.md](../docs/architecture.md) — layering and the dependency rule (`model/` knows
  nothing about `distributed/`, which is what makes the single-device model usable as *the*
  reference), the config system's strictness rules, the training loop's non-obvious parts, the
  three checkpoint formats and their one invariant, and a table of the four M1 decisions that each
  paid off in a later milestone.
- [distributed_training.md](../docs/distributed_training.md) — the three strategies side by side
  with what each splits, communicates and buys; DDP in full (it had no standalone doc); the
  `16P`-bytes-per-parameter argument for why FSDP and TP exist at all; and the table of every
  correctness claim with the test that establishes it.
- [testing.md](../docs/testing.md) — the six suites, the reference discipline, **60 mutations with
  59 caught**, the six things mutation testing found that the suite could not, and the eight
  test-design rules learned the hard way with their issue numbers.
- [benchmarks.md](../docs/benchmarks.md) — the four rules that make a number worth recording, how
  to reproduce any figure, the measured CPU table with its "relative overhead, not scaling" caveat,
  and every pending measurement paired with the command and hardware that would produce it.

**README restructured** into a front door: status, a documentation index, install, quick starts per
strategy, known limitations, future work in priority order, and an **item-by-item Definition-of-Done
assessment**. Four of the eleven criteria depend on NVIDIA hardware and are marked pending rather
than done — including #7 (the Triton kernel), which is honestly **not met**.

**Verified:** all internal documentation links resolve; `pytest -q` unaffected (455 passed, 37
skipped). Also corrected the README's repository layout, which still described files the scope had
sketched but the project never created (`distributed/ddp.py`, `kernels/reference.py`,
`data/text.py`, `benchmarks/profiler.py`) and omitted the ones it did.

**Final state.** 455 tests passing on CPU across six suites; 60 injected bugs with 59 caught; the
first measured performance numbers recorded with their caveats; and every remaining gap reduced to
a named command plus the hardware it needs. The one thing this project does not have is a GPU, and
that fact is visible in every document rather than papered over in any of them.

### 2026-11-08 — M5: Triton RMSNorm (kernel written but UNVERIFIED)

Scoped in [milestones/M5_triton.md](milestones/M5_triton.md), which states the constraint before
anything else: **Triton needs an NVIDIA GPU and has no macOS wheels (ISSUES #3)**, so the kernel
cannot be run here. The milestone therefore delivers everything *around* the kernel and records the
kernel as unverified code (ISSUES #16). Nothing in this project claims it works.

**Built.**
- `kernels/rmsnorm.py`: a fused forward kernel (one program per row, `BLOCK` = next power of two
  with a mask, fp32 accumulation) and a `dx` backward kernel. Triton is imported under a guard and
  the kernels are defined inside it, so the module imports fine without Triton.
- **Explicit autograd strategy** rather than a full Triton backward (D31): Triton for `dx`, PyTorch
  for `dw`. `dw` is a cross-row reduction needing atomics or locks — the most error-prone code in
  the kernel for the least benefit, and a decisively bad trade when it cannot be tested here.
  SCOPE.md §7 permits exactly this ("a valid backward implementation *or* an explicit supported
  autograd strategy").
- `model.norm_backend: auto|pytorch|triton` (D32), mirroring the existing `attention_backend`.
  Explicit `triton` is a hard error without Triton/CUDA, and fails at *construction*. `auto` is the
  only silent fallback in the project, defensible solely because the two paths are asserted
  equivalent — and it is resolved **once**, not per forward, since re-resolving would put a
  `torch.cuda.is_available()` call in a hot path.
- `tests/kernels/test_rmsnorm.py`: 73 tests parameterized over backend, so the kernel will be
  judged by the reference's own assertions with no new test code. 8 shapes (including
  non-powers-of-two 192, 193, 33), 3 dtypes with dtype-appropriate tolerances, extreme magnitudes,
  all-zero rows, single-element rows, mixed magnitudes, a float64 `gradcheck`, and the unit-RMS
  property.
- `benchmarks/kernels.py`: reference vs Triton across 6 shapes, reporting runtime and achieved
  bandwidth, since RMSNorm is memory-bound and bandwidth is the honest headline. Exits 2 with a
  clear explanation without a GPU.

**Results: 455 passed, 37 skipped** (`pytest -q`, CPU, ~126 s), up from 452. **34 of the skips are
the Triton parameters**, each stating the reason. The reference is fully verified; the kernel has
executed zero times.

**Mutation check** — 10 injected, 10 caught, but **two needed real fixes first**:

| Mutation | Caught |
|---|---|
| resolve: explicit triton falls back silently | 1 |
| resolve: auto errors instead of falling back | 1 |
| resolve: accept any backend string | 1 |
| module: do not resolve the backend at construction | 1 |
| config: allow norm_backend triton with layernorm | 1 |
| config: accept any norm_backend value | 1 *(0 first — real gap)* |
| reference: drop eps entirely | 1 |
| reference: add eps outside the square root | 1 |
| reference: downcast instead of promote (ISSUES #4 regression) | 1 |
| make_norm: ignore the configured backend | 1 *(0 first — vacuous test)* |

**Three things found:**
- **A test of mine was vacuous.** `test_make_norm_passes_the_configured_backend` asserted the
  *resolved* backend was `pytorch` — true on this machine whether or not the config was read at
  all, so a mutation that ignored `cfg.norm_backend` entirely passed. It now asserts the
  **requested** backend propagates, which is the part that distinguishes them. A test that cannot
  fail on the machine it runs on is not a test.
- **`ModelConfig.norm_backend`'s value validation was untested** — `test_config.py`'s
  invalid-values list never got a case, and the kernels test exercised `resolve_norm_backend`
  directly instead. Both now covered.
- **`eps` is not negligible at small activation scales** (ISSUES #15). A test asserted a constant
  row of 0.01 normalizes to exactly 1.0 and measured 0.954. With `mean(x²) ≈ 1e-4` against
  `eps = 1e-5`, 0.954 is *correct*. The mistake became a better test: it now compares against the
  exact float64 formula including eps and asserts the output is below 0.97, so a kernel that
  dropped eps or put it outside the square root fails there and nowhere else. Two of the ten
  mutations exist because of it.

**What closing ISSUES #16 takes:** `pytest tests/kernels -q` on a Linux + NVIDIA box, then
`python -m minimegatron.benchmarks.kernels`. Expect the first run to surface real bugs — that is
the normal outcome for a kernel written without ever executing it, and the suite is arranged so
they land as test failures rather than as silently wrong training.

### 2026-11-08 — M6: reliability engineering (complete)

Whole-milestone pass, scoped first in
[milestones/M6_reliability.md](milestones/M6_reliability.md). The first milestone since M1 with no
part blocked on hardware.

**Built.**
- **ISSUES #8 closed.** `eval_steps` now counts *global* evaluation batches over an unsharded
  validation stream, with the loss **sum** and the **count** both reduced. The evaluation set is
  exactly `eval_steps x micro_batch_size` samples at any world size, so validation loss is finally
  comparable across strategies.
- **Benchmark harness** (`benchmarks/{config,harness,run,regression}.py`): discards warmup,
  synchronizes at every boundary, repeats and reports median with min/max/IQR, uses CUDA events on
  CUDA, records the commit and full environment, and emits JSON *and* Markdown (D28).
  `scripts/check_env.py` now prints the same probe the harness records.
- **Regression suite** (`tests/regression/`): golden loss trajectory, golden gradient norms, golden
  LR schedule, exact preset parameter counts, state-dict key sets, and that every config in the
  repo still loads. Gradient norms are pinned *separately* from losses, because ISSUES #12 showed
  a gradient bug can leave the loss plausible while the norms are already wrong.
- **Performance regressions are warnings**, compared against versioned baselines in
  `results/baselines/` by a CLI tool, never asserted in pytest (D27).
- **Failure injection** (`tests/fault_injection/`, 27 tests) covering every case SCOPE.md §8 names,
  each asserting the *diagnostic text* rather than just that something raised (D29) — including a
  real `torchrun` job whose rank 1 dies via `os._exit` while rank 0 sits in an all-reduce.
- **OOM diagnostics**: the trainer re-raises a CUDA OOM (never swallows it) with the current
  settings and the four levers in order of disruption.
- **CI** (`.github/workflows/ci.yml`): CPU-only torch, the gloo tests included, gated on collecting
  at least 300 tests so a broken marker filter cannot pass silently (D30).
- **The documented CLI now exists**: `configs/{benchmark,ddp_2gpu,fsdp_2gpu,tp_2gpu}.yaml`, which
  SCOPE.md §10 had been describing without them being real.

**Results: 414 passed, 3 skipped** (`pytest -q`, CPU, ~130 s), up from 344. **First recorded
performance numbers** are in "Measured results" above — CPU only, and labelled as relative
overhead rather than scaling, because two ranks on one CPU contend for the same cores.

**Mutation check** — 12 injected, **12 caught** (one needed correcting first):

| Mutation | Caught |
|---|---|
| eval: shard the validation stream again | 1 |
| eval: unbalanced forward passes (the #13 deadlock) | 1 |
| eval: average per-rank means instead of sum/count | 1 |
| benchmark: do not discard warmup | 1 |
| benchmark: report peak memory as 0 instead of None | 1 |
| benchmark: single sample, no dispersion | 1 |
| benchmark: ignore grad accumulation in tokens/step | 1 |
| regression: treat higher latency as an improvement | 1 |
| regression: compare unmeasurable metrics as zero | 1 |
| OOM: swallow the error instead of re-raising | 1 |
| OOM: remove the advice lines | 1 *(0 on the first attempt — see below)* |
| strategy: allow tp on a single rank | 1 |

**Four things found:**
- **A deadlock, immediately** (ISSUES #13). Making the evaluation set a fixed global size left one
  rank with 3 batches and the other with 2 — and under FSDP the *forward itself* contains
  collectives, so the short rank left the loop and the other blocked forever in an all-gather.
  DDP and TP passed, which is what made it interesting: a `no_grad` DDP forward has no collective,
  and TP's `dp_world_size` is 1. Fixed by running `ceil(eval_steps / dp)` passes on every rank with
  the surplus weighted zero. **The number of forward passes is part of the contract between
  ranks, not just the data.** The harness's hard timeout turned the hang into a diagnosable
  failure in three minutes.
- **`strategy: tp` on a single rank was never rejected** — the needs-peers guard listed only `ddp`
  and `fsdp`, so TP fell through to `init_device_mesh` and died with a torch error about an unset
  `RANK`. Found by a fault-injection test written to check the *diagnostic*, which is exactly what
  that suite is for.
- **Loss at init is below ln(V) when the targets are the inputs** (ISSUES #14): 4.46 against 4.85.
  With tied embeddings, "predict the token at this position" is already easy at initialization,
  because the residual stream carries `tok_emb(x[t])` and the tied head scores it highly. A real
  property of weight tying, and a reminder that a loss-at-init check needs independent targets.
- **One mutation survived and was wrong, not revealing.** It deleted the header line of the OOM
  advice rather than the advice, which the test does not assert (and need not). Re-run with the
  advice actually removed, it was caught. Second time in the project that a surviving mutation
  turned out to be a bad mutation — worth checking that before concluding there is a test gap.

Also obsoleted one M2 test by design: `test_validation_loss_is_aggregated_across_ranks` asserted
that each rank's eval loss *differed*, which was true only because the validation stream used to be
sharded. Rewritten to assert the new, stronger invariant — each rank handles a different subset of
the global batches, and the reported number is the mean over their union, identical everywhere.

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
