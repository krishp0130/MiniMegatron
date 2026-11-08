# Architecture

How MiniMegatron is put together, and why the seams are where they are. This is the orientation
document; the deep dives are [fsdp.md](fsdp.md), [tensor_parallel.md](tensor_parallel.md) and the
per-milestone design docs under [milestones/](milestones/).

## Layering

```
train.py                  CLI: config -> overrides -> Trainer -> fit
  │
  ├── experiment.py       one YAML file -> one validated ExperimentConfig
  │     model / data / train / distributed / benchmark sections
  │
  ├── model/              the GPT itself, device- and strategy-agnostic
  │     config.py         ModelConfig: shape and options, validated on construction
  │     attention.py      causal self-attention, `reference` and `sdpa` backends
  │     layers.py         norms, GELU MLP, pre-norm Block
  │     transformer.py    GPT: embeddings -> Block x L -> norm -> tied LM head -> loss
  │
  ├── data/               index-deterministic synthetic datasets + batching
  ├── training/           optimizer, LR schedule, checkpoints, metrics, the loop
  ├── distributed/        process groups, DDP, FSDP2, tensor parallelism, collectives demo
  ├── kernels/            Triton kernels and the PyTorch references they are judged against
  └── benchmarks/         measurement harness, runner, regression comparison
```

The dependency direction is strict: `model/` knows nothing about `distributed/`, `training/` or
`benchmarks/`. A `GPT` is an ordinary `nn.Module` that can be built and run with no process group,
no strategy and no config beyond `ModelConfig`. Everything that makes it distributed is applied
*from outside* — which is what makes the single-device model usable as the reference that every
parallel strategy is checked against.

## The model

```
idx (B x T, int64)
 -> tok_emb(idx) + pos_emb(arange(T))        B x T x d
 -> dropout
 -> Block x L                                 pre-norm: x = x + attn(norm(x)); x = x + mlp(norm(x))
 -> final norm
 -> lm_head (weight tied to tok_emb)          B x T x V
 -> cross-entropy vs targets                  computed in at least fp32
```

Four decisions in here are load-bearing far beyond M1, and each one paid off in a later milestone:

| Decision | Why | Where it paid off |
|---|---|---|
| **Per-head fused QKV layout `[h, 3, d_h]`** (D6) rather than GPT-2's `[3, h, d_h]` | a contiguous column-parallel slice then holds *complete* heads | M4: tensor-parallel attention needs no communication inside itself |
| **Tied embedding / LM head** (D8) | saves `V·d` parameters; matches GPT-2 | M3/M4: forced the sharding units and the state-dict handling to be got right |
| **Loss promoted to ≥fp32** before cross-entropy (ISSUES #4) | log-sum-exp over the vocabulary is the most precision-sensitive op in the model | M3: bf16 needed no new code, and under FSDP (where autocast is absent) it is the only thing keeping the loss in fp32 |
| **Separate named `nn.Linear`s** `qkv`/`out_proj`/`fc1`/`fc2` (D4) | replaceable without restructuring | M4: TP swaps exactly these four per block |

The fourth was *nearly* enough on its own: M4 still found one shape assumption outside the layers
(`attention.py` reshaped concatenated heads to the full `d_model`, ISSUES #11). Replaceable layers
are not the same as a layer-width-agnostic model.

## Configuration

One YAML file becomes one frozen, validated `ExperimentConfig` with five sections. The design rules
(D10) are stricter than usual, deliberately:

- **Frozen dataclasses**, validated in `__post_init__`, so an invalid config cannot exist once a
  model is built from it.
- **Every error reported at once**, so a broken config takes one pass to fix, not one error per run.
- **Unknown keys are rejected** with a "did you mean" hint. A typo like `n_layer:` would otherwise
  silently keep the default and train the wrong model — the single nastiest class of bug this
  project could ship.
- **`bool` rejected where an int is expected**, since `True` is an `int` subclass in Python.
- **The YAML files in `configs/` are the only source of truth** for presets; there is no duplicate
  table in code, and tests load the YAML and assert exact parameter counts.
- **`--set section.key=value`** overrides are applied to the raw mapping *before* validation, so a
  command-line override is checked exactly as strictly as a file.

Two distinctions in the config surface are worth stating because both are easy to conflate:

- `model.dtype` is parameter **storage** (always `float32`); `train.precision` is **compute**
  (`fp32`/`bf16`). Mixed precision means bf16 maths over fp32 master weights (D19).
- `model.attention_backend` and `model.norm_backend` choose an **implementation**, not the maths —
  which is why they sit together and why both are expected to be numerically equivalent (D32).

## The training loop

`training/trainer.py` owns one `train_step`: `grad_accum_steps` micro-batches, each loss scaled by
`1/accum`, then clip, optimizer step, scheduler step. The non-obvious parts:

- **`1/accum` scaling** makes the accumulated gradient exactly the gradient of the *mean* loss over
  the global batch, so accumulation is identical to one large batch up to float rounding.
- **`raw_model` is kept separate from `model`.** `model` may be a DDP wrapper, an FSDP-sharded
  module or a TP-surgered one; `raw_model` is always the real `GPT`. Parameter access, clipping and
  checkpointing therefore never depend on the wrapper, and adding the third and fourth strategies
  touched none of them.
- **Clipping happens after the synchronizing backward**, on the unwrapped parameters. Clipping
  per-rank gradients first would make the effective threshold depend on world size — and under TP
  it needs a hand-written global norm entirely (ISSUES #12).
- **No silent fallback, anywhere.** An unavailable device, precision or kernel is a hard error, at
  construction where possible. The one exception is `norm_backend: auto` (D32).
- **The device is synchronized before any clock is read.** CUDA and MPS enqueue asynchronously, so
  an unsynchronized measurement reports Python's enqueue rate.

## Data

Synthetic datasets that are a pure function of `(seed, index)` through a SplitMix64 mixer (D9), so
nothing is stored and any sampler gets reproducible shards. `RandomTokenDataset` is unlearnable —
ideal for throughput work, since there is no learning curve to confuse the measurement — and
`MemorizationDataset` repeats N fixed sequences, which is what makes the overfit test a proof that
the whole loop is wired correctly.

The decision that mattered most: **the index stream is defined over samples, not batches** (D12).
`IndexStream` is one infinite sequence — concatenated per-epoch permutations — and a batch is a
slice of it. Three things fall out for free:

1. `accum=4 × micro=2` consumes byte-identical samples to `accum=1 × micro=8`.
2. "Resume at step *s*" needs **no dataloader state in the checkpoint at all**.
3. Rank sharding is just a different slice of the same stream (D17), which is what made DDP's and
   FSDP's equivalence tests *exact* rather than statistical.

## Checkpoints

Three formats, because the strategies genuinely differ, but with one invariant across all of them:
**a checkpoint can always be loaded into a plain single-device `GPT`.** That portability is what
makes an FSDP or TP run comparable to the reference at all, which is the project's whole evidential
basis.

| Strategy | Resume format | Portable format |
|---|---|---|
| single / DDP | one file (rank 0) | the same file |
| FSDP2 | `ckpt_NNNNNN/` directory, one DCP shard per rank, RNG per rank | `ckpt_NNNNNN_full.pt`, gathered on rank 0 |
| TP | `ckpt_NNNNNN_tp{rank}.pt`, per rank | `ckpt_NNNNNN_full.pt`, gathered on rank 0 |

Writes are atomic (`.tmp` then `os.replace`), so a crash during a checkpoint cannot destroy the
last good one. All four RNG streams are captured, per rank where ranks differ — without them a
resumed run diverges and a resume bug becomes indistinguishable from noise.

## Known limitations

Honest, and each with the reason:

- **The Triton kernel has never been executed** (ISSUES #16). No NVIDIA GPU on the development
  machine and no macOS Triton wheels. It is written, not working.
- **No GPU measurement exists.** Every throughput and memory comparison the scope asks for is
  pending hardware; see [benchmarks.md](benchmarks.md) for the exact commands.
- **Dropout is unsupported under tensor parallelism** for comparisons (D18). Correct per-rank
  dropout needs a model-parallel RNG tracker; all equivalence tests use `dropout=0`.
- **2D TP+DP is not supported** and is explicitly refused rather than silently mistrained: a
  `tp_size` below the world size would leave the data-parallel dimension without gradient
  synchronization.
- **Embeddings are replicated under TP**, so the tiny preset's 9.7M-parameter token table (84% of
  it) is not sharded. Vocab-parallel embeddings and cross-entropy are the obvious next step.
- **Validation loss is comparable across world sizes, but the evaluation set is small** by default;
  it is a sanity signal, not a benchmark.
- **No real text data.** `data/text.py` was scoped as an M1 stretch item and deferred; everything
  trains on synthetic tokens, so no sample from the model would be meaningful.
- **Single node only.** No elastic training, no pipeline parallelism, no FlashAttention (the scope
  excludes the last one from the first phase).

## Future work, in the order I would do it

1. **Run the Triton kernel** and close ISSUES #16. Everything needed to judge it already exists.
2. **One 2-GPU session** to fill every pending benchmark table.
3. **Vocab-parallel embedding and cross-entropy** — the largest remaining memory win for TP.
4. **A model-parallel RNG tracker**, which unblocks dropout under TP and FSDP comparisons.
5. **Real text data** and sampling, so the model produces something a human can read.
6. **2D TP+DP composition.** The `(dp, tp)` device mesh and the strategy switch are the seams it
   would attach to.
