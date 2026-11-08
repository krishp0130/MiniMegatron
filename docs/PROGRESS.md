# Progress

Living status of the project. **Update this at the end of every working session**: tick off
checklist items, add a session-log entry, and rewrite "Next up". Scope lives in
[SCOPE.md](SCOPE.md); problems and design decisions live in [ISSUES.md](ISSUES.md).

## Current state

- **Active milestone:** 1 (Functional GPT baseline) **complete** — all 6 steps and all exit
  criteria met. Next: Milestone 2 (DDP).
- **Last updated:** 2026-11-07
- **Test status:** 201 passed (`pytest -q`, CPU; the 2 MPS tests ran on the M4 Pro)
- **Verified hardware:** Apple M4 Pro (CPU + MPS). No NVIDIA GPU verification yet.

## Next up

**Milestone 2 — distributed data parallel.** Scope it in `docs/milestones/M2_ddp.md` first
(process-group lifecycle, DDP gradient synchronization, `no_sync` accumulation, rank-aware
logging and rank-0 checkpointing, the gloo correctness test), then implement.

The M1 pieces M2 builds directly on:
- `training/trainer.py` takes a `dp_world_size` argument already; it is 1 in M1 and feeds the
  `global_batch_size` cross-check (`micro × accum × dp_world_size`).
- `data/synthetic.py`'s `IndexStream` is defined over samples, not batches (D12), so rank
  sharding slots in without disturbing the single-device sample order — which is what makes
  "DDP gradients == reference global-batch gradients" a meaningful comparison.
- `tests/unit/test_train.py::test_accumulation_matches_one_large_batch` is the single-process
  form of exactly the equivalence the DDP test must show across ranks.

Two environment constraints to plan around (ISSUES #1, #3): gloo on macOS needs
`GLOO_SOCKET_IFNAME=lo0` and an explicit master address, and every distributed subprocess needs a
hard Python-side timeout. NCCL and all scaling benchmarks stay *pending hardware*.

Full M1 design, for reference: [milestones/M1_gpt_baseline.md](milestones/M1_gpt_baseline.md).

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

### Milestone 2 — Distributed data parallel
- [ ] Process-group setup/teardown (`distributed/setup.py`)
- [ ] DDP wrapper, DistributedSampler, `no_sync` grad accumulation
- [ ] Rank-aware logging, metric aggregation, rank-0 checkpointing
- [ ] DDP gradients == reference global-batch gradients (gloo test)
- [ ] Educational explicit all-reduce demo (`collectives_demo.py`)
- [ ] 1 vs 2 GPU throughput benchmark *(pending hardware)*

### Milestone 3 — Memory optimization
- [ ] BF16 mixed precision
- [ ] FSDP2 with per-block `fully_shard`
- [ ] Sharded checkpoint save/restore (torch.distributed.checkpoint)
- [ ] FSDP convergence vs reference (gloo test)
- [ ] Peak memory per rank, DDP vs FSDP comparison *(pending hardware)*

### Milestone 4 — Tensor parallelism
- [ ] `f`/`g` autograd communication primitives + unit tests
- [ ] ColumnParallelLinear / RowParallelLinear vs `nn.Linear` tests
- [ ] Parallel MLP and parallel attention (head partitioning)
- [ ] Shard a full state dict into TP shards / gather back
- [ ] TP forward + gradient equivalence vs single-device reference (gloo test)
- [ ] TP config validation (`n_heads`, `d_ff` divisible by `tp_size`)
- [ ] `docs/tensor_parallel.md`
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

None recorded yet, deliberately. The tiny preset trains end to end on CPU and MPS and the loop
prints `tokens_per_sec`, but those short runs include warmup and are not a benchmark: no warmup
discard, no repeats, no variance. The benchmark harness is Milestone 6, and the first numbers
worth recording here are M2's 1-vs-2-rank comparisons.

Record only numbers that were actually measured, with the config, hardware, and commit that
produced them.

## Session log

Newest first.

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
