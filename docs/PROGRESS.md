# Progress

Living status of the project. **Update this at the end of every working session**: tick off
checklist items, add a session-log entry, and rewrite "Next up". Scope lives in
[SCOPE.md](SCOPE.md); problems and design decisions live in [ISSUES.md](ISSUES.md).

## Current state

- **Active milestone:** 1 (Functional GPT baseline), not yet started
- **Last updated:** 2026-10-23
- **Test status:** no tests yet (`pytest` collects 0)
- **Verified hardware:** Apple M4 Pro (CPU + MPS). No NVIDIA GPU verification yet.

## Next up

Milestone 1, step 1: `src/minimegatron/model/config.py`, a validated model config dataclass with
tiny/small/medium presets and unit tests for validation (e.g. `d_model % n_heads == 0`).

Planned order within Milestone 1:
1. `model/config.py` + config validation tests
2. `model/attention.py`: explicit reference attention + SDPA backend, equivalence and causal-mask tests
3. `model/layers.py`: LayerNorm/RMSNorm, MLP, pre-norm block
4. `model/transformer.py`: full GPT, parameter-count tests for presets
5. `data/synthetic.py`, `training/` loop, CPU smoke test, overfit-a-tiny-batch test

## Milestone checklist

### Milestone 0 — Setup ✅
- [x] GitHub repo (`krishp0130/MiniMegatron`, private)
- [x] Python 3.12 venv, `torch==2.14.1` pinned, editable install
- [x] Directory skeleton, `scripts/check_env.py`
- [x] Verified 2-process DDP and FSDP2 on CPU via gloo (see ISSUES.md for macOS quirks)
- [x] Context docs: CLAUDE.md, SCOPE.md, PROGRESS.md, ISSUES.md

### Milestone 1 — Functional GPT baseline
- [ ] Model config + validation
- [ ] Reference attention + SDPA backend
- [ ] Norms, MLP, transformer block
- [ ] Full GPT model + tiny/small/medium configs with reported parameter counts
- [ ] Synthetic dataset
- [ ] Single-device training loop (AdamW, warmup + cosine, grad accumulation, clipping)
- [ ] Checkpoint save/resume (model, optimizer, scheduler, RNG)
- [ ] CPU smoke test + overfit test
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

### 2026-10-23 — Project setup
- Created repo, venv (Python 3.12.14, torch 2.14.1, numpy 2.5.4, pytest 9.1.1, tokenizers 0.23.3,
  tensorboard), editable install, directory skeleton, env check script.
- Hardware audit: Apple M4 Pro, 24 GB, MPS available, **no CUDA/NCCL/Triton**.
- Probed distributed on CPU: 2-process DDP and FSDP2 both work under gloo after two macOS
  workarounds (ISSUES.md #1, #2).
- Added tensor parallelism to scope as Milestone 4.
- Wrote CLAUDE.md, SCOPE.md, PROGRESS.md, ISSUES.md.
