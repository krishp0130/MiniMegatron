# MiniMegatron — complete project overview

*A single self-contained description of what this project is, what was built, why each choice was
made, and what was measured. Written so that a person or an LLM reading only this file understands
the project without opening the source.*

**Last updated:** 2026-11-09, at commit `f8f7e82`.

---

## 1. What this is

MiniMegatron is an **educational, engineering-quality distributed transformer training
framework**. It trains a GPT-style decoder four different ways — on one device, with DDP, with
FSDP2, and with Megatron-style tensor parallelism — and includes a hand-written Triton kernel.

Its organising principle, which explains nearly every design decision in the repository:

> **Every optimization is checked against a single-device reference implementation, with a
> correctness test and a reproducible benchmark.**

This is the thing production frameworks cannot offer you. Megatron-LM and PyTorch's own FSDP are
faster and more complete, but at their scale the single-device reference *no longer fits in
memory*, so the equivalence can only be asserted, not demonstrated. Here the models are small
enough that every parallel path can be diffed against a plain one, numerically, in CI.

### What it is not

- **Not a production training framework.** Use Megatron-LM, DeepSpeed or PyTorch FSDP for that.
- **Not a CLI tool.** The CLI (`train.py`) is 123 lines out of 4,116 — about 3% of the source. No
  console entry point is installed; you invoke `python -m minimegatron.train`.
- **Not a library with a stable public API.** No versioning policy, no distribution packaging.
  Those were deliberately skipped: cost with no learning in it.

The most accurate one-line description is **a reference implementation with an executable proof
attached** — closest in spirit to a textbook whose examples compile.

### The shape of the repository tells the story

| | lines | note |
|---|---|---|
| `src/minimegatron/` | 4,116 | the library |
| ├─ `train.py` (CLI) | 123 | 3% of source |
| `tests/` | **5,525** | **more code than the thing it tests** |
| `docs/` | 4,503 | roughly equal to the source |
| `scripts/` | 36 | |

A product has far more source than tests. A reference implementation has the inverse, because its
deliverable is not the feature — it is *the evidence that the feature is correct*.

---

## 2. Quick facts

| | |
|---|---|
| **Language / runtime** | Python 3.12, `torch==2.14.1` (pinned), Triton 3.x |
| **Tests** | **505 collected** — 266 unit, 89 distributed, 73 kernels, 34 fault-injection, 31 numerical, 12 regression |
| **Test status** | 468 passed / 37 skipped on CPU; 104 passed / 7 skipped / **0 failed** on 2× T4 |
| **Mutation testing** | 60 deliberate bugs injected across six milestones, **59 caught** |
| **Dev hardware** | Apple M4 Pro (CPU + MPS) — **no CUDA, NCCL or Triton locally** |
| **Verification hardware** | Kaggle `T4 x2` (2× Tesla T4, sm_75), NCCL |
| **CI** | GitHub Actions, CPU-safe suite on every push, gated on collected test count |
| **Definition of Done** | **11 of 11 met** |
| **Repository** | github.com/krishp0130/MiniMegatron (public) |

---

## 3. Repository map

```
src/minimegatron/
  config.py            strict loader: unknown keys rejected with "did you mean"
  experiment.py        YAML -> validated config objects
  train.py             the CLI (123 lines)
  model/
    config.py          ModelConfig, frozen dataclass, validates everything at once
    attention.py       causal self-attention; reference + SDPA backends
    layers.py          RMSNorm (+ the PyTorch reference M5 is measured against), GELU MLP, Block
    transformer.py     the GPT: embeddings, blocks, final norm, tied LM head
  data/
    synthetic.py       index-deterministic dataset; rank sharding
  training/
    trainer.py         (578 lines) the training loop, strategy dispatch, DDP, eval, checkpoints
    optimizer.py       AdamW + cosine schedule with warmup
    checkpoint.py      save/load, RNG state, resume
    metrics.py         JSONL logging
  distributed/
    setup.py           process group init, backend/device resolution
    fsdp.py            FSDP2 fully_shard wiring, mixed-precision policy
    tensor_parallel.py (503 lines) ColumnParallelLinear, RowParallelLinear, f/g, clip, gather
    checkpoint.py      distributed checkpointing, consolidation to single-device format
    collectives_demo.py teaching demo of the collectives
  kernels/
    rmsnorm.py         Triton forward + dx backward; backend resolution
  benchmarks/
    harness.py         timing, warmup, IQR, memory, commit stamping
    run.py             end-to-end training benchmarks
    kernels.py         kernel microbenchmarks
    regression.py      compares against recorded baselines

tests/        unit, numerical, distributed, fault_injection, regression, kernels
docs/         SCOPE, PROGRESS, ISSUES, architecture, distributed_training, fsdp,
              tensor_parallel, benchmarks, testing, milestones/M1-M7
configs/      tiny, small, medium, benchmark, ddp_2gpu, fsdp_2gpu, tp_2gpu
```

---

## 4. The model (Milestone 1)

A standard **pre-norm GPT decoder**:

```
tokens -> embeddings -> Block x L -> final norm -> tied LM head -> cross-entropy
```

A `Block` is:

```python
x = x + attn(norm1(x))
x = x + mlp(norm2(x))
```

**Pre-norm matters.** The residual stream `x` is only ever *added to*, never normalized. That
leaves an unobstructed path for gradients from the loss back to the embeddings, which is what
makes deep stacks trainable without delicate warmup schedules. Post-norm (the original 2017
arrangement) puts a LayerNorm *on* the residual path and is materially harder to train deep.

### Four M1 decisions that paid off much later

| Decision | ID | Paid off in |
|---|---|---|
| QKV laid out **per head** `[h, 3, d_h]`, not GPT-2's `[3, h, d_h]` | D6 | **M4** — a contiguous column-parallel slice then holds *complete heads*, so TP attention needs no communication inside itself |
| **Tied** input embedding / LM head | D8 | M3/M4 — forced sharding units and state-dict handling to be correct |
| Loss promoted to **>= fp32** before cross-entropy (`torch.promote_types`) | — | **M3** — under FSDP there is no autocast, so this is the *only* thing keeping the loss in fp32 |
| Four **separately named** `nn.Linear` modules, not fused blobs | D4 | M4 — TP swaps exactly these by name |

The QKV layout is the clearest example of the project's method: a choice made in M1 for reasons
that only became visible in M4. Writing the design doc before the implementation is what made that
possible.

Other M1 details: residual projections initialized at `0.02/sqrt(2L)` (so the residual stream's
variance does not grow with depth), vocab padded to 50304 for alignment (D7), RMSNorm written in
PyTorch in M1 *specifically* so that M5 would have a reference to measure the Triton kernel
against.

---

## 5. Configuration and the training engine

### Strict configuration (D10)

YAML loads into frozen dataclasses. Unknown keys are **rejected**, with a suggestion:

```
unknown key 'n_layer' in section 'model'; did you mean 'n_layers'?
```

All validation errors are reported **at once**, not one per run. A typo in a config is the most
common way a long training run is silently wasted, so it is treated as a first-class failure.

An important distinction, D19: **`model.dtype` is parameter *storage*; `train.precision` is
*compute*.** Setting `model.dtype: bfloat16` is an error that tells you to use
`train.precision: bf16` instead. Conflating the two is a common and expensive confusion.

### Determinism (D9, D12, D17)

- Datasets are **index-deterministic**: sample `i` is always the same, independent of batch size,
  world size or iteration order.
- The index stream is defined over **samples, not batches** (D12). This is subtle and important:
  defining it over batches makes the data seen depend on the batch size, so a gradient-accumulation
  change silently alters the run.
- Rank sharding formula (D17): rank `r` takes index `m*micro*R + r*micro` for micro-step `m`.
- RNG state is checkpointed, NumPy's as a tensor (D13) so checkpoints load under
  `weights_only=True`.

Two identical runs produce **byte-identical `log.jsonl`** — asserted from outside the test suite,
at the CLI boundary, because that is where a reader would check it.

### The optimizer and schedule

AdamW with a cosine schedule and linear warmup. On resume, **the learning rate is recomputed from
the live schedule, not restored from the checkpoint** (D14). This bug is described in section 11 —
it is the single best story in the project.

---

## 6. The three parallelism strategies

This is the core of the project.

### Why more than one exists

FP32 + AdamW costs **16 bytes per parameter** before any activations: 4 for the weight, 4 for the
gradient, 4 each for Adam's `m` and `v`. DDP replicates all of it on every rank, so adding GPUs
buys throughput and **nothing** in capacity. FSDP and TP exist to buy capacity instead.

| | Splits | Communicates | Buys |
|---|---|---|---|
| **DDP** | the **batch** | 1 all-reduce per step | throughput |
| **FSDP2** | parameter/gradient/optimizer **storage** | all-gather + reduce-scatter per unit | capacity |
| **TP** | each layer's **weight matrices** | 2 all-reduces per block, each direction | capacity, never gathering a full layer |

### DDP (Milestone 2)

Rests on a single identity: because the loss is a **mean**,

```
grad L(B) = (1/R) * sum_r grad L(B_r)
```

So one all-reduce converts R different gradients into the one correct gradient. Consequences:

- **Equal shard sizes are mandatory.** A short shard silently reweights the gradient — it does not
  crash, it just trains something slightly different.
- `no_sync` is used for gradient accumulation, so only the final micro-step communicates.
- Bucketing overlaps communication with the backward pass.

DDP is implemented in `training/trainer.py` (wrapping with `DistributedDataParallel`), with
`distributed/setup.py` handling process-group initialization and backend resolution.

### FSDP2 (Milestone 3)

Uses PyTorch's `fully_shard`. Key decisions:

- **Sharding units are transformer blocks** (D20), applied per block *and* to the root. Sharding
  only the root gathers the whole model at once and defeats the purpose.
- `MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)` — compute in bf16, but
  **reduce gradients in fp32**. Reducing in bf16 loses precision in exactly the place it matters.
- **Autocast is skipped under FSDP** (D22) — the policy already handles dtype. This is why the
  M1 loss promotion is load-bearing: nothing else keeps the loss in fp32.
- The **optimizer must be built after `fully_shard`** (D20), so it sees sharded parameters.
- Gradients use **reduce-scatter**, not all-reduce: each rank owns only a slice, so it only needs
  the averaged gradient for that slice.
- Two checkpoint formats are written (D21): DCP sharded, plus a consolidated single-device file.

### Tensor parallelism (Milestone 4)

Hand-written, Megatron-style (D23 — DTensor used only as a cross-check). This splits the matmuls
themselves.

**Column-parallel** (`W = [W1 | W2]`, split along output):
needs all of X, produces a *slice* of Y. No forward communication.

**Row-parallel** (`W = [W1 ; W2]`, split along input):
needs a *slice* of X, produces a **partial sum**. Must all-reduce.

**The trick:** chain column-parallel into row-parallel and the intermediate activation is never
gathered — one all-reduce per pair. Both the MLP and attention are built as exactly one such pair:

```
MLP:        fc  (column) -> GELU -> proj (row)  ->  1 all-reduce
Attention:  qkv (column) -> SDPA -> proj (row)  ->  1 all-reduce
```

Attention needs no internal communication **because of the D6 head layout** — a contiguous column
slice holds whole heads.

**The `f` / `g` operators** are implemented as `autograd.Function`s and are *conjugates*:

| | forward | backward |
|---|---|---|
| `f` (entering a TP region) | identity | all-reduce |
| `g` (leaving a TP region) | all-reduce | identity |

The asymmetry is not a convention — it falls out of the calculus. Each rank's output depends on
all of the input, so the input's gradient is a sum over ranks.

Other TP decisions:

- **Row-parallel biases are replicated and added after the all-reduce** (D26). Adding before would
  add the bias R times. Splitting it would be wrong in a different way. Both were tested by
  mutation.
- Parallel layers are **installed by surgery with names preserved** (D25), so state dicts stay
  compatible.
- **TP ranks are not data-parallel ranks** (D24). The global batch is `micro × accum × 1`, and the
  data stream is sharded over the *data-parallel* world, not the whole world.
- `clip_grad_norm_tp` — see section 11, bug #12.

---

## 7. The Triton kernel (Milestone 5)

A hand-written Triton RMSNorm: **forward kernel plus a `dx` backward kernel**, with `dw` left in
PyTorch (D31, an explicit and documented strategy — `dw` requires a cross-row reduction that is
awkward to do well in a single kernel, and it is not the bottleneck).

`model.norm_backend: auto | pytorch | triton`. The `auto` setting is **the one place in the
project where a silent fallback is permitted** (D32); an explicit `triton` with no Triton
available is a hard error, because silently not running the thing you asked to measure is how
benchmark claims become fiction.

**Measured on a Tesla T4, float32**, against the PyTorch reference written in M1:

| rows × cols | reference | Triton | speedup | GB/s |
|---|---|---|---|---|
| 4096 × 192 | 0.114 ms | 0.073 ms | 1.56× | 87 |
| 4096 × 768 | 0.405 ms | 0.111 ms | **3.66×** | 228 |
| 16384 × 768 | 1.475 ms | 0.430 ms | **3.43×** | 234 |
| 65536 × 768 | 5.839 ms | 1.707 ms | **3.42×** | 236 |

234 GB/s is about **73% of the T4's 320 GB/s peak** — near the hardware ceiling for a memory-bound
operation. The small shape (192 cols) wins less because there is not enough work per row to hide
launch overhead.

**Honest framing:** the kernel is 3.4× faster. **Training is not.** RMSNorm is a small slice of a
step; a large speedup on a small fraction is a small speedup overall. The project states this
explicitly rather than quoting 3.4× as if it were an end-to-end number.

---

## 8. Testing (Milestone 6)

Six suites, **505 tests collected**:

| suite | tests | what it does |
|---|---|---|
| `unit` | 266 | config validation, shapes, invariants |
| `distributed` | 89 | DDP/FSDP/TP equivalence against single-device, over gloo |
| `kernels` | 73 | Triton vs reference, numerics, gradients, float64 `gradcheck` |
| `fault_injection` | 34 | deliberate breakage must be caught and reported clearly |
| `numerical` | 31 | numerical properties, extremes |
| `regression` | 12 | compares against recorded baselines |

**gloo-on-CPU is the distributed correctness backend** (D2). This is what made a GPU-free
development machine viable: collectives semantics are identical, so equivalence can be proven
locally and only *performance* needs real hardware.

### Mutation testing: 60 injected bugs, 59 caught

The suite was validated by deliberately breaking the source and confirming the tests fail. Example
mutations from M4:

```
row-parallel bias added BEFORE the all-reduce         -> caught
row-parallel bias split instead of replicated         -> caught
f: no all-reduce in the backward                      -> caught
g: no all-reduce in the forward                       -> caught
column-parallel splits the wrong dimension            -> caught
TP clip: do not reduce the sharded squared norms      -> caught
TP clip: also reduce the replicated norms (double)    -> caught
attention output reshaped to the full d_model         -> caught
```

The single survivor needs a CUDA device without bf16 support — hardware that was not available.

**Two "surviving" mutations turned out to be bad mutations, not missing tests** (M3's
optimizer-ordering and M6's OOM-header mutations changed nothing observable). That is itself a
lesson: a mutation that changes nothing proves nothing about the suite, and the honest response is
to discard the mutation rather than claim a gap.

**What mutation testing did not catch:** the end-to-end run later found two real bugs that 60
mutations and 455 tests had both missed (see #17 and #18 in section 11). Mutation testing proves
the tests cover the code that *exists*; it cannot find a seam that no test crosses at all.

---

## 9. Measured results

All numbers below were measured on the hardware named. **Nothing in this project is estimated.**

### End-to-end training, 2× Tesla T4, fp32, 10.6M-parameter model

| label | strategy | GPUs | tokens/s | vs 1 GPU | peak memory |
|---|---|---|---|---|---|
| gpu-single | single | 1 | 29,648 | — | 1,862 MiB |
| **gpu-ddp-2** | ddp | 2 | **52,493** | **1.77×** | 1,906 MiB |
| gpu-fsdp-2 | fsdp | 2 | 51,925 | 1.75× | 1,834 MiB |
| gpu-tp-2 | tp | 2 | 32,545 | 1.10× | 1,841 MiB |

Three things worth reading off this table:

1. **1.77× from 2 GPUs is good.** Perfect is 2.0×; the missing 12% is the all-reduce. DDP's
   communication is one operation per step, so it amortizes well.
2. **FSDP is 1% slower than DDP, as expected.** FSDP is not for speed. It trades throughput for
   the ability to fit a model DDP cannot hold. At this model size there is nothing to fit, so you
   see only the cost.
3. **TP is 38% slower than DDP and barely faster than one GPU** — and this was **predicted in
   writing before any GPU was available to measure it**, in M4's design doc and `benchmarks.md`.
   TP all-reduces twice per block per direction, so its communication scales with *depth* rather
   than being one operation per step, and two T4s on a PCIe bus cannot hide it. This is exactly
   why Megatron confines tensor parallelism to within a node (on NVLink) and composes it with data
   parallelism across nodes.

**Caveat, stated plainly:** these are two T4s on PCIe with a tiny model. They demonstrate that the
implementations are correct and behave sensibly. They do not extrapolate to scale, and the TP
result in particular would change completely on NVLink.

### Benchmark methodology (D28)

A number is recorded only with: warmup, median of repeated runs, the IQR, peak memory, the config,
the hardware, and the **git commit**. Memory reports `n/a` where it cannot be measured rather than
guessing (`torch.cuda.max_memory_allocated` is CUDA-only). Functional regressions are assertions;
performance regressions are warnings (D27), because a slow machine should not fail CI.

---

## 10. Do the strategies combine?

**In this project: no — you choose exactly one.**

```yaml
distributed:
  strategy: tp       # auto | single | ddp | fsdp | tp
```

They are mutually exclusive by design. That is a *teaching* decision, not a technical limit: each
strategy is compared against the same single-device reference in isolation, so when a number moves
you know precisely what moved it.

**In production they compose, and that is the whole game.** Real large-model training uses **3D
parallelism** — TP × PP × DP simultaneously:

- **TP within a node** (8 GPUs on NVLink), because TP is bandwidth-hungry and NVLink is where the
  bandwidth is.
- **Pipeline parallelism across a few nodes** (splitting layers — not implemented here).
- **DDP/FSDP across everything else**, because one all-reduce per step is cheap enough to cross a
  slower fabric.

The ranks form a grid and each axis communicates over the fabric it can afford. Nobody uses TP
alone; the 32,545 number is precisely why.

**How you choose in practice** — driven by *what does not fit*, not by what is fastest:

| situation | use |
|---|---|
| model fits on one GPU | **DDP** — simplest and fastest |
| model does not fit, but a single layer does | **FSDP** |
| a single layer does not fit | **TP** inside a node, composed with DP outside |

**The hook for composition already exists**: `tp_size` is separate from world size, and data
sharding is defined over the data-parallel world (D24), not the whole world. `tp_size: 0` means
"the whole world is one TP group". With 4 ranks and `tp_size: 2` you would want TP=2 × DP=2 — the
config can express it, but the strategy dispatch does not yet build the 2D mesh. That is the
natural next milestone.

---

## 11. The bugs — the real payload

`docs/ISSUES.md` holds 21 numbered issues and 32 design decisions. The issues are not a changelog;
they are the lessons, preserved with their symptoms. The most instructive:

**#14 / D14 — the learning rate after resume.** On resume, the LR was taken from the *saved*
schedule rather than recomputed. The first fix pushed the stored value back into the scheduler —
which was **indistinguishable from doing nothing**. Mutation testing caught it. Fixed properly
with `set_lr_for_step`. *Lesson: a fix that cannot fail a test is not a fix.*

**#12 — `clip_grad_norm_` computes a rank-local norm under TP.** The norm came out 1.26595 instead
of 1.296081, while the **losses matched to six decimal places**. A sharded parameter's full norm
requires summing squared norms across ranks; a replicated parameter must *not* be summed, or it is
counted R times. *Lesson: a correct loss does not imply correct clipping.* Fixed with
`clip_grad_norm_tp`.

**#13 — unbalanced evaluation batches deadlock FSDP.** Fixing the eval-set size caused both ranks
to time out at 180s. Under FSDP the forward pass *contains collectives*, so **the number of
forward passes is part of the contract between ranks**. Fixed by running
`ceil(eval_steps / dp_world_size)` passes on every rank, with the surplus zero-weighted.

**#17 — a TP checkpoint gathered the model but not the optimizer.** It loaded without complaint
and died one optimizer step later. *Lesson: "it loads" is not "it works."* Found only by the
end-to-end run, which crossed a seam no unit test crossed.

**#19 — TP layers were built on the CPU** regardless of the model's device. Completely invisible
on a CPU-only machine, where everything is on the CPU. Surfaced on the first GPU run as
`mat1 is on cuda:0, different from other tensors on cpu`. Fixed by threading `device`/`dtype`
through both parallel constructors, plus an `assert_single_device` guard.

**#20 — three tests encoded "there is no GPU here" as a fact.** They asserted the *absence* of
CUDA rather than skipping. They passed for months and failed the moment a GPU appeared.
*Lesson, and the one most worth generalizing: **a test suite written on hardware unlike the target
quietly encodes that hardware's limits as invariants.***

**#21 — a documentation error I made.** I claimed a T4 "has no bf16 at all".
`torch.cuda.is_bf16_supported()` returns True on a T4 — bf16 *runs*, via emulation, and is
therefore **slower** than fp32. Corrected.

Also recorded: #11 (attention reshape assumed full `d_model`), #5 (import cycle), #7 (torch tensors
through a `multiprocessing.Queue` deadlock the parent), #9 (DDP's reducer is a read-only C++
binding, so test its *observable consequences* instead), #15 (my own test premise was wrong — `eps`
is not negligible at small activation scales), #18 (the CLI leaked tracebacks for setup errors).

**Still open:** #3 (no NVIDIA GPU on the dev machine — environmental) and #10 (autocast leaves the
residual stream in fp32, so bf16 losses barely move — understood and documented, not yet acted on).

---

## 12. How to run it

```bash
pip install -e ".[dev]"            # text/logging/gpu extras are for optional + planned features
python scripts/check_env.py        # environment report

# single device
python -m minimegatron.train --config configs/tiny.yaml
python -m minimegatron.train --config configs/tiny.yaml --set train.max_steps=50
python -m minimegatron.train --config configs/tiny.yaml --resume runs/tiny/ckpt_000050.pt

# distributed (macOS gloo needs loopback pinning, ISSUES #1)
GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m minimegatron.train --config configs/ddp_2gpu.yaml       # or fsdp_2gpu / tp_2gpu

# on 2 NVIDIA GPUs --standalone works and the backend resolves to nccl automatically
torchrun --standalone --nproc_per_node=2 -m minimegatron.train --config configs/tp_2gpu.yaml

# tests
pytest -q                               # all
pytest -q -m "not gpu and not multigpu" # what CI runs

# benchmarks
python -m minimegatron.benchmarks.run --config configs/benchmark.yaml --label my-run
python -m minimegatron.benchmarks.kernels     # refuses clearly without a GPU
bash scripts/gpu_verify.sh                    # 7-section GPU battery
```

GPU verification without local hardware: `scripts/colab_gpu_verify.ipynb`, or Kaggle's free
`T4 x2` (2 GPUs, 30h/week). **Colab gives only one GPU per runtime**, so multi-GPU work needs
Kaggle.

---

## 13. Milestones

| | milestone | status |
|---|---|---|
| M1 | Functional GPT baseline | complete |
| M2 | Distributed data parallel | complete |
| M3 | FSDP2 + BF16 mixed precision | complete |
| M4 | Megatron-style tensor parallelism | complete |
| M5 | Triton RMSNorm kernel | complete, verified on a T4 |
| M6 | Reliability engineering | complete |
| M7 | Final documentation | complete |

Each milestone was **scoped in its own design doc before implementation** (`docs/milestones/`),
covering the math, the decisions, the tests and the exit criteria. The QKV-layout story in section
4 is what that process buys.

### Definition of Done — 11 of 11

All eleven criteria in `docs/SCOPE.md` §13 are met. The single measurement still missing anywhere
in the project is the **FP32-vs-BF16 speedup**, which needs **Ampere or newer**: a T4 emulates
bf16 and is slower with it, so the number cannot be obtained on the available hardware. It is
documented as pending with the exact command and hardware it requires, rather than estimated.

---

## 14. Limits and what is deliberately absent

- **No pipeline parallelism**, and no composition of strategies into a 2D/3D mesh (section 10).
- **No real tokenizer / text data** — `data/text.py` was deferred; the `tokenizers` extra is the
  hook for it. Training uses an index-deterministic synthetic dataset.
- **No profiler integration** (`benchmarks/profiler.py` was scoped, not built).
- **Metrics are JSONL**, not TensorBoard; the `tensorboard` extra is unused.
- **Benchmarks are small-model, 2-GPU, PCIe.** They validate correctness and relative behaviour,
  not scaling claims.
- **No stable public API or distribution packaging** — deliberate, as stated in section 1.

---

## 15. If you are an LLM reading this to work on the project

Read, in order: `docs/PROGRESS.md` (current state and session log), `docs/ISSUES.md` (problems,
workarounds, the D-numbered decisions), `docs/SCOPE.md` (the source of truth for scope), then the
design doc for whichever milestone is active.

Three standing rules from `CLAUDE.md`, which are not negotiable:

1. **The owner is building this to learn.** Work one step at a time and explain the concepts and
   design choices as you go.
2. **Every optimization needs a reference implementation, a correctness test, and a reproducible
   benchmark.**
3. **Never fabricate or estimate results.** GPU/NCCL/Triton work stays marked *pending* until
   actually run on real hardware. Record measured numbers with config, hardware and commit.

The development machine has **no CUDA, NCCL or Triton** (Apple M4 Pro). `torchrun` with gloo
**hangs silently** on macOS without `GLOO_SOCKET_IFNAME=lo0`, so always use a hard timeout. FSDP2
and device meshes need an explicit device type (`init_device_mesh("cpu", ...)`); auto-detection
picks MPS and crashes.

Commits are authored by the owner only — **no AI attribution trailers**.
