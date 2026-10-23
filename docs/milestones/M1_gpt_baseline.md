# Milestone 1 — Functional GPT Baseline

**Goal:** a correct, tested, single-device GPT training pipeline that everything later (DDP, FSDP2,
tensor parallelism, Triton) builds on and is checked against. Everything in this milestone runs on
CPU, and on MPS for speed. No GPU is required.

**Not in this milestone:** BF16/autocast (M3), distributed anything (M2+), `torch.compile`, custom
kernels (M5).

## Exit criteria

M1 is done when all of these hold, verified and recorded in PROGRESS.md:

1. `pytest tests/unit tests/numerical -q` passes on CPU.
2. Reference attention and SDPA attention agree within FP32 tolerance; `gradcheck` passes in FP64.
3. Model parameter counts for tiny/small/medium equal the analytic formula below, exactly.
4. Loss at initialization is ≈ ln(V) (sanity check on init and on the loss).
5. Overfit test: a nano model memorizes a fixed batch (loss falls from ≈ln V to < 0.1).
6. Resume test: training 2N steps straight equals training N steps, checkpointing, resuming, and
   training N more, **bit-for-bit on CPU** (weights, losses, LR).
7. `python -m minimegatron.train --config configs/tiny.yaml` runs end to end on CPU and MPS,
   logging loss, LR, grad norm, and tokens/s to JSONL. Throughput is recorded only as measured.

## Build order

Six steps. Each ends with passing tests, a check-in, a PROGRESS.md update, and a commit.

| Step | Files | Tests |
|---|---|---|
| 1. Config | `model/config.py`, `configs/{tiny,small,medium}.yaml` | `tests/unit/test_config.py` |
| 2. Attention | `model/attention.py` | `tests/unit/test_attention.py`, `tests/numerical/test_attention_backends.py` |
| 3. Layers | `model/layers.py` (norms, MLP, block) | `tests/unit/test_layers.py`, `tests/numerical/test_norms.py` |
| 4. GPT model | `model/transformer.py` | `tests/unit/test_model.py` |
| 5. Data | `data/synthetic.py` | `tests/unit/test_data.py` |
| 6. Training engine | `training/{optimizer,checkpoint,metrics,trainer}.py`, `train.py` | `tests/unit/test_{optimizer,checkpoint,train}.py` |
| (stretch) Real text | `data/text.py` | `tests/unit/test_text_data.py` |

---

## Step 1 — Model config

One frozen dataclass, `ModelConfig`, that is the single source of truth for model shape and
validates itself in `__post_init__` (so an invalid model can't be built):

| Field | Default | Validation |
|---|---|---|
| `vocab_size` | 50304 | > 0 |
| `seq_len` | 256 | > 0 (max context; learned position table size) |
| `n_layers` | — | > 0 |
| `n_heads` | — | > 0, `d_model % n_heads == 0` |
| `d_model` | — | > 0 |
| `d_ff` | `4 * d_model` | > 0 |
| `dropout` | 0.0 | 0 ≤ p < 1 |
| `bias` | True | Linear and LayerNorm biases (GPT-2 style) |
| `tie_embeddings` | True | LM head shares the token-embedding matrix |
| `attention_backend` | `"reference"` | in {`reference`, `sdpa`} |
| `norm_type` | `"layernorm"` | in {`layernorm`, `rmsnorm`} |
| `dtype` | `"float32"` | parameter storage dtype; M1 supports `float32` only (BF16 arrives in M3, which errors clearly until then) |

Derived: `head_dim = d_model // n_heads`.

YAML experiment files have top-level sections `model:`, `data:`, `train:` (later `distributed:`).
The loader **rejects unknown keys**, so a typo like `n_layer:` fails instead of being silently
ignored, which is a common and nasty bug class.

### Presets and exact parameter counts

**Vocab = 50304:** the GPT-2 BPE vocabulary (50257) padded up to a multiple of 128. Padding costs
nothing semantically (the extra tokens are never produced by the tokenizer) but gives GPUs nicely
aligned matmul shapes, and makes the vocab divisible by any small TP size.

| Preset | L | d_model | heads | head_dim | d_ff | seq_len | **Total params** | Non-embedding |
|---|---|---|---|---|---|---|---|---|
| tiny | 4 | 192 | 4 | 48 | 768 | 256 | **11,487,360** | 1,779,840 |
| small | 8 | 512 | 8 | 64 | 2048 | 512 | **51,237,888** | 25,220,096 |
| medium | 12 | 768 | 12 | 64 | 3072 | 1024 | **124,475,904** | 85,056,000 |

Formula (LayerNorm, biases on, tied embeddings), with V = vocab, T = seq_len, d = d_model, f = d_ff:

```
attention   = 4d² + 4d           (qkv: d×3d + 3d,  out_proj: d×d + d)
mlp         = 2df + f + d        (fc1: d×f + f,    fc2: f×d + d)
norms       = 2 · 2d             (two LayerNorms, weight + bias each)
per_layer   = attention + mlp + norms
total       = L · per_layer + V·d (token emb) + T·d (pos emb) + 2d (final norm) + 0 (tied head)
```

Sanity anchor: with V = 50257 the medium preset gives 124,439,808, **exactly GPT-2 small**. Our
medium is GPT-2 small with a padded vocab. A unit test asserts the model's real count equals this
formula for every preset and both norm types.

Notice how embedding-heavy the tiny model is: 84% of its parameters are the token table. This is
why scaling-law papers report non-embedding parameter counts.

Tests also use a **nano** config (V=128, T=16, L=2, d=32, h=4, f=64), which is not a preset, so the
whole suite runs in seconds on CPU.

---

## Step 2 — Causal self-attention

### The math

Input `x ∈ ℝ^{B×T×d}`. One fused projection gives queries, keys, and values:

```
[Q K V] = x·W_qkv + b_qkv           W_qkv ∈ ℝ^{d×3d}
```

Split into `h` heads of size `d_h = d/h` and move heads to a batch-like dimension:
`Q, K, V ∈ ℝ^{B×h×T×d_h}`. For each head independently:

```
S = Q·Kᵀ / √d_h                     S ∈ ℝ^{B×h×T×T}    (score of query i against key j)
S[i, j] = −∞  for j > i             causal mask: position i may only look at positions ≤ i
P = softmax(S, dim=−1)              each row is a probability distribution over keys
P = dropout(P)
O = P·V                             O ∈ ℝ^{B×h×T×d_h}  (weighted average of value vectors)
```

Then concatenate heads back to `B×T×d` and apply the output projection `y = O·W_o + b_o`.

**Why divide by √d_h.** If the components of q and k are independent with mean 0 and variance 1,
then `q·k = Σᵢ qᵢkᵢ` is a sum of d_h terms each with variance 1, so `Var(q·k) = d_h`. Unscaled,
scores grow like √d_h, softmax saturates toward one-hot, and its gradient (which is
`P(1−P)`-shaped) vanishes. Dividing by √d_h restores unit variance regardless of head size.

**Why −∞ works for masking.** `exp(−∞) = 0`, so masked keys get exactly zero probability and the
row still sums to 1 over the allowed keys. A fully masked row would give 0/0 = NaN, but the causal
mask always leaves the diagonal (a token can attend to itself), so this never happens here. Worth
knowing for padding masks later.

**Softmax stability.** `softmax(s) = softmax(s − max(s))`, so implementations subtract the row max
before `exp` to avoid overflow. `torch.softmax` already does this. We won't hand-roll it, but the
same identity is the heart of FlashAttention's online softmax.

**Cost.** `S` and `P` are `B·h·T²` elements each, so memory and compute are quadratic in sequence
length. At T=1024, h=12, B=8 that is ~100M floats per layer just for P, which is the exact problem
FlashAttention solves by never materializing P.

### Two backends, one interface

- `reference`: the math above, written out op by op with a precomputed lower-triangular mask
  buffer (`register_buffer(..., persistent=False)`, so it moves with `.to(device)` but isn't saved
  in checkpoints). This is the trusted path.
- `sdpa`: `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)`. PyTorch picks
  a fused kernel (FlashAttention/memory-efficient on CUDA, others on CPU/MPS).

### Design decision: per-head QKV layout (sets up tensor parallelism)

The fused `qkv` output of width 3d can be laid out two ways:

- `[3, h, d_h]`: all Qs, then all Ks, then all Vs (GPT-2's layout).
- `[h, 3, d_h]`: head 0's q,k,v, then head 1's q,k,v, … (Megatron's layout).

In M4, column-parallel TP splits the `qkv` output dimension into `tp_size` **contiguous** chunks.
With `[3, h, d_h]`, rank 0 of 2 would get all the queries and half the keys, which is wrong. With
`[h, 3, d_h]`, each rank gets complete heads (q, k and v together) and can run attention with no
communication. We use `[h, 3, d_h]` from day one: `qkv(x).view(B, T, h, 3, d_h)`. The cost is that
GPT-2 pretrained weights would need permuting to load, which we don't plan to do. (Recorded as D6.)

### Tests

- Shapes: output `B×T×d` for odd B, T < seq_len, T = 1.
- **Causality, tested behaviorally**: perturb the input at position t and assert outputs at
  positions < t are unchanged, while position ≥ t changes. This tests the property directly instead
  of inspecting the mask.
- Reference vs. SDPA: same weights, `dropout=0`, `torch.testing.assert_close` (dtype-aware default
  tolerances), including non-power-of-two `head_dim` (tiny uses 48).
- `torch.autograd.gradcheck` on the reference path in float64: compares autograd gradients against
  finite differences.
- Input longer than `seq_len` raises a clear error.

---

## Step 3 — Norms, MLP, transformer block

### LayerNorm vs. RMSNorm

For a vector `x ∈ ℝ^d` (normalized over the feature dimension, independently per token):

```
LayerNorm:  μ = mean(x),  σ² = mean((x − μ)²)
            y = (x − μ) / √(σ² + ε) · γ + β
RMSNorm:    rms = √(mean(x²) + ε)
            y = x / rms · γ
```

RMSNorm drops the mean-centering and β. It is cheaper (one reduction instead of two), works
as well in practice (Llama, Mistral, etc.), and is the kernel we'll fuse in Triton in M5. Our
hand-written `RMSNorm` becomes the reference implementation for that kernel, and is tested here
against `torch.nn.RMSNorm`. For LayerNorm we use `nn.LayerNorm`.

Implementation note: compute the normalization statistics in FP32 even when activations are BF16
(upcast, normalize, downcast). Mean-of-squares in BF16 loses too much precision. This does nothing
in M1 (all FP32) but is the right structure for M3.

### MLP

```
MLP(x) = dropout( GELU(x·W₁ + b₁) · W₂ + b₂ )      W₁ ∈ ℝ^{d×f}, W₂ ∈ ℝ^{f×d}, f = 4d
GELU(u) = u · Φ(u)                                   Φ = standard normal CDF
```

GELU is a smooth ReLU: it weights each input by the probability that a standard normal falls
below it. We use the exact (erf) form; GPT-2 used the tanh approximation. The two differ by < 1e-3.
The MLP holds 2/3 of each block's parameters (8d² of 12d²).

### Pre-norm block

```
x = x + Attention(Norm₁(x))
x = x + MLP(Norm₂(x))
```

The **residual stream** `x` passes through every block via pure addition. In pre-norm (GPT-2
onward), normalization is applied only to each sublayer's *input*, so there is an unobstructed
identity path from output to input. Gradients flow through it without passing through any norm,
which makes deep stacks stable without delicate LR warmup. The original Transformer used post-norm
(`x = Norm(x + Sublayer(x))`), which puts a norm on the main path and is harder to train deep.
Consequence: the residual stream is never normalized inside the stack, so the model needs a
**final norm** before the LM head.

---

## Step 4 — GPT model

```
idx (B×T, int64)
 → tok_emb(idx) + pos_emb(arange(T))       B×T×d
 → dropout
 → Block × L
 → final norm
 → lm_head  (weight tied to tok_emb)        B×T×V logits
 → cross-entropy vs. targets (if given)
```

`forward(idx, targets=None) -> (logits, loss | None)`.

**Next-token objective.** The dataset yields `x = tokens[0:T]`, `y = tokens[1:T+1]`, so position
i's target is the token at i+1. Thanks to the causal mask, one forward pass trains T prediction
problems at once. Loss = mean over all B·T positions of `−log softmax(logits)[target]`, computed in
FP32, with `ignore_index=-100` reserved for future padding.

**Weight tying.** The input embedding maps token → vector; the output head maps vector → token
scores. Sharing one V×d matrix saves V·d parameters (9.7M of tiny's 21M untied total) and tends to
help small models. Implications later: FSDP2 must keep both uses in the same sharding unit (the
root module), and TP keeps both replicated in the basic version.

**Initialization (GPT-2 scheme).**
- Linear and embedding weights `~ N(0, 0.02)`, biases 0, norm γ = 1, β = 0.
- **Residual projections** (`out_proj`, `fc2`) get std `0.02 / √(2L)`. Each block adds two
  outputs into the residual stream; if each has variance σ², after L blocks the stream variance
  grows by ~2L·σ². Scaling these weights by 1/√(2L) keeps the stream variance roughly constant with
  depth at init.

**Loss-at-init check.** With small random weights the logits are nearly uniform, so
`loss ≈ −log(1/V) = ln V = ln 50304 ≈ 10.83`. A test asserts the initial loss is within a small
margin of ln V. A big deviation means broken init, a broken loss, or misaligned targets, all of
which can otherwise go unnoticed while training still "kind of works".

**Tests:** logits shape; loss is a finite scalar; parameter count == formula (all presets, both
norms); tied weights really are the same tensor; loss at init ≈ ln V; every parameter receives a
non-None, finite gradient after `backward()`; `reference` and `sdpa` models with the same weights
give the same logits.

---

## Step 5 — Synthetic data

Two map-style datasets, deterministic **by index** (item i is generated from `hash(seed, i)`), so
any sampler, including M2's `DistributedSampler`, gets reproducible shards, and resuming
mid-epoch is trivial:

- `RandomTokenDataset`: uniform random tokens. Nothing is learnable (loss stays ≈ ln V), which
  makes it ideal for **throughput benchmarks** because the data costs nothing to produce.
- `MemorizationDataset`: N fixed random sequences. The model *can* drive loss to ~0 by memorizing
  them, which makes it ideal for the **overfit test** that proves the whole forward/backward/optimizer
  loop is wired correctly.

Batching: a small deterministic sampler where the batch for global step `s` is a pure function of
`(seed, s)` (seeded permutation per epoch). This makes "resume at step s" exact without
saving dataloader state.

**Tests:** same seed + index → same tokens; different seed → different; `y` is `x` shifted by
one; tokens are in `[0, V)`; the batch at step s is identical whether reached directly or after a
resume.

---

## Step 6 — Training engine

### Optimizer: AdamW with parameter groups

Adam keeps per-parameter running averages of the gradient (m, momentum) and squared gradient (v),
and steps by `m̂ / (√v̂ + ε)`: roughly a per-parameter-normalized gradient. **AdamW** applies weight
decay directly to the weights (`w ← w − lr·λ·w`) instead of adding `λw` to the gradient, where
Adam's normalization would distort it.

Two groups:
- **decay**: all ≥2-D weights (matmul weights, embeddings)
- **no decay**: biases and norm γ/β (1-D). Decaying these just fights the normalization and has
  no regularizing benefit.

Memory note for M3: Adam stores m and v in FP32, which is 2 extra copies of the model. With FP32
params and grads, that's **16 bytes per parameter** before activations: the number FSDP's
optimizer-state sharding attacks.

### LR schedule: linear warmup → cosine decay

```
step < warmup:          lr = max_lr · (step + 1) / warmup
warmup ≤ step ≤ max:    progress = (step − warmup) / (max − warmup)
                        lr = min_lr + ½(max_lr − min_lr)(1 + cos(π · progress))
```

Warmup matters because Adam's v estimate is noisy in the first steps, so early updates are
unreliable and large. Cosine decay anneals smoothly to `min_lr` (typically max_lr/10). Implemented
as a pure function (unit-tested at boundaries) wrapped in `LambdaLR`, so the scheduler has a
standard `state_dict()` for checkpointing.

### Gradient accumulation and global batch

```
global_batch_size = micro_batch_size × grad_accum_steps × dp_world_size   (dp_world_size = 1 in M1)
tokens_per_step   = global_batch_size × seq_len
```

We run `grad_accum_steps` forward/backward passes before one optimizer step, scaling each
micro-batch loss by `1/grad_accum_steps`. Gradients **sum** across `backward()` calls, so the
scaling makes the accumulated gradient the gradient of the *mean* loss over the whole global batch,
which is identical to one big batch (up to float rounding). If the config specifies
`global_batch_size`, the trainer checks the equation and refuses to start if it doesn't hold. M2's
DDP correctness test leans on exactly this equivalence.

### Gradient clipping

`clip_grad_norm_(params, max_norm)` computes the global L2 norm over *all* gradients; if it exceeds
`max_norm` (1.0), every gradient is scaled by `max_norm / norm`. This preserves direction and caps
step size against loss spikes. The pre-clip norm is logged every step because it's one of the best
early-warning signals of instability.

### Device selection (no silent fallback)

`device: auto` picks cuda → mps → cpu. An explicit `device: cuda` on a machine without CUDA is a
hard error, not a quiet CPU run.

### Timing and throughput

GPU and MPS execution is **asynchronous**: Python enqueues kernels and returns immediately.
Timing a step without synchronizing measures enqueue time, not compute. The trainer synchronizes
the device at log boundaries before reading the clock. (CUDA events come in the M2+ benchmark
harness.) Logged: `step, loss, lr, grad_norm, tokens_seen, tokens_per_sec, elapsed`, as JSONL,
one object per line, which is trivial to parse and plot later.

Validation: every `eval_interval` steps, `model.eval()` + `torch.no_grad()`, average loss over
`eval_steps` batches from a separate split, then back to `train()` (which matters for dropout).

### Checkpointing and exact resume

A checkpoint contains: model state, optimizer state (m, v, step counts), scheduler state, global
step, tokens seen, the full resolved config, and **RNG states** (Python, NumPy, torch CPU, and
device).

- **Atomic writes:** save to `ckpt.tmp`, then `os.replace` it to the final name. A crash
  mid-write never corrupts the last good checkpoint. (M6 tests the corrupt-checkpoint path.)
- **Why RNG state:** dropout masks and data order come from RNGs. Without restoring them, a
  resumed run diverges from the uninterrupted one, and you can no longer tell a resume bug from
  noise.
- **The resume test:** run A trains 2N steps. Run B trains N, saves, starts a fresh process
  state, loads, and trains N more. Assert identical losses at every step and **bit-identical**
  final weights on CPU. This only passes if model, optimizer, scheduler, data position, and RNG
  are all restored correctly, so it is a strong single test. (MPS isn't guaranteed deterministic,
  so the bitwise assertion is CPU-only.)

### CLI

```bash
python -m minimegatron.train --config configs/tiny.yaml
python -m minimegatron.train --config configs/tiny.yaml --resume runs/tiny/ckpt_000500.pt
python -m minimegatron.train --config configs/tiny.yaml --set train.max_steps=20 --set train.device=cpu
```

`--set` takes dotted overrides that are validated like the YAML itself; it is used heavily by
smoke tests.

### Tests

- LR schedule values at step 0, end of warmup, midpoint, and the final step.
- Param groups: every parameter in exactly one group; 1-D params have no decay.
- Grad accumulation: accum=4 × micro=2 gives the same gradients as accum=1 × micro=8.
- Global batch validation rejects a mismatched config.
- Checkpoint round-trip, the resume equivalence test, atomic-write behavior.
- **Overfit test:** nano model on `MemorizationDataset` (1 batch), ~200 steps on CPU, loss < 0.1.
- CLI smoke: subprocess run of `train.py` for a few steps with a nano config; exit code 0 and a
  well-formed JSONL log.

---

## Stretch: real text

`data/text.py`: tokenize a small public corpus (e.g. TinyShakespeare) with the GPT-2 tokenizer via
Hugging Face `tokenizers`, store as a flat `uint16` NumPy memmap (50304 < 65536), and sample random
windows. This produces a genuinely learning language model to look at samples from before moving
on. Not required for M1 sign-off.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| MPS op coverage/determinism gaps | CPU is the correctness platform; MPS is used for speed only. Log any MPS-specific failures in ISSUES.md. |
| Tied weights complicating FSDP2 (M3) | Keep embedding + head in the root sharding unit; noted in D8. |
| Fused QKV layout vs. TP (M4) | Per-head `[h, 3, d_h]` layout from the start (D6). |
| Hidden nondeterminism breaking the resume test | Index-deterministic data, full RNG capture, CPU-only bitwise check. |
| Medium preset is slow on a laptop | Only tiny is trained locally; small/medium are param-count-tested and run on GPU later. |
