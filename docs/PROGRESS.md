# Progress

Living status of the project. **Update this at the end of every working session**: tick off
checklist items, add a session-log entry, and rewrite "Next up". Scope lives in
[SCOPE.md](SCOPE.md); problems and design decisions live in [ISSUES.md](ISSUES.md).

## Current state

- **Active milestone:** 1 (Functional GPT baseline), steps 1–3 of 6 done
- **Last updated:** 2026-10-23
- **Test status:** 92 passed (`pytest tests/unit tests/numerical -q`, CPU + MPS)
- **Verified hardware:** Apple M4 Pro (CPU + MPS). No NVIDIA GPU verification yet.

## Next up

Milestone 1, step 4: `src/minimegatron/model/transformer.py`. The full `GPT` (token + position
embeddings, `Block` × L, final norm, tied LM head, cross-entropy), GPT-2 init with residual
projections scaled by 1/√(2L), and tests: param count == formula, loss at init ≈ ln V, tied
weights share storage, gradients reach every parameter, reference vs. SDPA logits match.

Full M1 design (math, decisions, tests, exit criteria):
[milestones/M1_gpt_baseline.md](milestones/M1_gpt_baseline.md).
Build order: config → attention → layers → GPT model → synthetic data → training engine.

## Milestone checklist

### Milestone 0 — Setup ✅
- [x] GitHub repo (`krishp0130/MiniMegatron`, private)
- [x] Python 3.12 venv, `torch==2.14.1` pinned, editable install
- [x] Directory skeleton, `scripts/check_env.py`
- [x] Verified 2-process DDP and FSDP2 on CPU via gloo (see ISSUES.md for macOS quirks)
- [x] Context docs: CLAUDE.md, SCOPE.md, PROGRESS.md, ISSUES.md

### Milestone 1 — Functional GPT baseline
- [x] Milestone scoped ([M1_gpt_baseline.md](milestones/M1_gpt_baseline.md))
- [x] Model config + validation (`config.py`, `model/config.py`, `experiment.py`, `configs/*.yaml`)
- [x] Reference attention + SDPA backend (`model/attention.py`)
- [x] Norms, MLP, transformer block (`model/layers.py`)
- [ ] Full GPT model; param counts match formula (tiny 11,487,360 / small 51,237,888 / medium 124,475,904)
- [ ] Synthetic dataset
- [ ] Single-device training loop (AdamW, warmup + cosine, grad accumulation, clipping)
- [ ] Checkpoint save/resume (model, optimizer, scheduler, RNG)
- [ ] Loss-at-init ≈ ln V test, overfit test, bitwise resume test, CPU smoke test
- [ ] `python -m minimegatron.train --config configs/tiny.yaml` works

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

None yet. Record only numbers that were actually measured, with the config, hardware, and
commit that produced them.

## Session log

Newest first.

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
