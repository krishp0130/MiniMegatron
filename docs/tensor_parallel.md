# Tensor parallelism

How MiniMegatron's `strategy: tp` works. Design notes are in
[milestones/M4_tp.md](milestones/M4_tp.md); this is the standalone explanation SCOPE.md section 6
asks for.

## What it is, next to the other two

| Strategy | What it splits | Communication | Buys you |
|---|---|---|---|
| DDP | the **batch** | one all-reduce of gradients per step | throughput |
| FSDP2 | parameter/gradient/optimizer **storage** | all-gather + reduce-scatter per unit per step | capacity for one big model |
| **TP** | each layer's **weight matrices** | two all-reduces per block, each direction | capacity, with no gather of full layers ever |

DDP replicates everything. FSDP shards storage but still gathers a whole unit to compute it. TP
splits the **computation itself**: a rank holds a slice of every weight matrix and never
materializes a full one. The price is communication *inside* every block's forward and backward,
not once per step.

## The two ways to split a matmul

Take `Y = XW`, `X ∈ ℝ^{n×d}`, `W ∈ ℝ^{d×f}`.

### Column parallel — split `W` by columns

```
W = [W₁ | W₂]                   W₁, W₂ ∈ ℝ^{d×f/2}
Y = X[W₁ | W₂] = [XW₁ | XW₂]
```

Each rank needs **all of X** and produces a **slice of Y**. No forward communication: the slices
*are* different columns of the same answer.

Backward: `∂L/∂X = Σᵢ (∂L/∂Yᵢ)Wᵢᵀ`. Each rank computes a **partial** input gradient, so the
partials must be summed.

### Row parallel — split `W` by rows

```
W = [W₁ ; W₂]                   W₁, W₂ ∈ ℝ^{d/2×f}
X = [X₁ | X₂]                   X must already be split by columns
Y = X₁W₁ + X₂W₂
```

Each rank needs a **slice of X** and produces a **partial sum** of the whole `Y`, so the forward
needs an all-reduce.

Backward: `∂L/∂Xᵢ = (∂L/∂Y)Wᵢᵀ` needs only the full output gradient, which every rank already has.
No backward communication.

### The pairing is the whole trick

Column parallel leaves its output split along the feature dimension. Row parallel *wants* its input
split that way. Chain **column → row** and the intermediate is never gathered:

```
X ──[column: no comm]──▶ sharded hidden ──[row: ONE all-reduce]──▶ Y
```

One all-reduce per pair in the forward and one in the backward, no matter how many matmuls sit
inside. Both the MLP and attention are built as exactly one such pair.

## The `f` / `g` conjugate pair

Megatron names the two communication operators `f` and `g`. They are conjugates — each is the
identity in one direction and an all-reduce in the other:

| Operator | Forward | Backward | Sits at |
|---|---|---|---|
| `f` = `copy_to_tp_region` | identity | all-reduce (sum) | **input** of a column-parallel layer |
| `g` = `reduce_from_tp_region` | all-reduce (sum) | identity | **output** of a row-parallel layer |

The asymmetry is not a convention; it falls straight out of the calculus above:

- `f`'s input is **replicated** and consumed by every rank, so its gradient is a sum of per-rank
  contributions. Nothing to do forward; sum backward.
- `g`'s output is a **partial sum** needing completion, so sum forward. Afterwards every rank holds
  the identical full output, so the incoming gradient is already complete — nothing to do backward.

Two more, for moving between replicated and sharded layouts (used to compare a parallel layer
against `nn.Linear`, and the seam sequence parallelism would later use):

| Operator | Forward | Backward |
|---|---|---|
| `scatter_to_tp_region` | split along the last dim, keep this rank's chunk | all-gather |
| `gather_from_tp_region` | all-gather along the last dim | split |

All four are `torch.autograd.Function`s, because the backward of a collective is a *different
collective* and autograd cannot infer that. They are implemented by hand with
`torch.distributed` calls rather than via DTensor (D3), so the communication is visible in our own
code.

## Parallel MLP

```
fc1: column parallel   d → d_ff/tp     no communication
GELU                                    elementwise, so local
fc2: row parallel      d_ff/tp → d     one all-reduce
```

GELU being elementwise is what makes this work: a nonlinearity applied to a *slice* of the hidden
dimension equals the matching slice of the nonlinearity applied to the whole. If the activation
mixed features across `d_ff` — a softmax, say — the hidden state would have to be gathered first.

## Parallel attention: partition the heads

QKV is column parallel, `out_proj` is row parallel — the same pair. What makes it clean is that
multi-head attention is *already* `n_heads` independent computations, so splitting the QKV output
dimension by **whole heads** lets each rank run `n_heads/tp_size` complete heads with **no
communication inside attention at all**.

This is why Milestone 1 chose the per-head fused QKV layout `[h, 3, d_h]` over GPT-2's
`[3, h, d_h]` (D6). Column parallel splits the output dimension into *contiguous* chunks:

```
[h, 3, d_h], tp=2, h=4:   rank 0 → heads 0,1 (q,k,v each);  rank 1 → heads 2,3        ✓
[3, h, d_h], tp=2:        rank 0 → all 4 q's + 2 k's;       rank 1 → 2 k's + 4 v's    ✗
```

A decision made three milestones earlier, for this.

Two code consequences, both small and both load-bearing:

- `CausalSelfAttention.n_heads` becomes the **local** count during sharding, or `split_qkv` would
  reshape this rank's slice with the wrong head count.
- The reshape that concatenates heads derives its width from the heads this module owns, not from
  its input's `d_model`. Hard-coding the input width breaks TP — it did, and that is ISSUES #11.

## What stays replicated

The token embedding, position embedding, the tied LM head and every norm are replicated on every
rank in this basic version (D8 keeps the tying). Their gradients are therefore already complete on
each rank and must match the single-device reference **exactly** and be identical across ranks —
a sharper test than it sounds, because it only holds if `f`'s backward all-reduce is right.

The practical consequence is visible in the parameter count. The tiny preset, `tp_size=2`:

| | Full model | Per rank, TP=2 |
|---|---|---|
| total | 11,487,360 | 10,599,936 |
| non-embedding | 1,779,840 | 892,416 |

Non-embedding nearly halves, as expected — it is not *exactly* half because the norms and the
row-parallel biases stay replicated. Total barely moves, because the 9.7M-parameter embedding table
(84% of this preset) is replicated. **That** is the argument for vocab-parallel embeddings and a
vocab-parallel cross-entropy, which are a documented stretch goal rather than part of the basic
version.

## Gradient clipping needs its own collective

`torch.nn.utils.clip_grad_norm_` computes the wrong thing under TP, and quietly.

Under FSDP the gradients are `DTensor`s that carry their sharding, so the stock function reduces
across shards correctly. Under TP they are **ordinary tensors that happen to be slices**, with
nothing marking them as such — so each rank computes a norm over its own slices plus the replicated
parameters, which is neither the global norm nor even the same on every rank. Measured: 1.26595 per
rank against a true global norm of 1.296081.

`clip_grad_norm_tp` treats the two kinds separately:

```
‖g‖² = all_reduce( Σ_sharded ‖g_shard‖² )  +  Σ_replicated ‖g‖²
```

- **sharded** — each rank holds a different slice, so `‖g_full‖² = Σᵣ ‖g_shardᵣ‖²`; the squared
  norms must be all-reduced.
- **replicated** — every rank already holds the identical full gradient, so it is counted **once**
  and must *not* be reduced, or it would be counted `tp_size` times.

Both halves of that are mutation-tested, because each produces a plausible-looking wrong number.

## Data, batches and process groups

**Every TP rank processes the same batch.** TP is not data parallelism: the ranks collectively
compute one forward pass, so they must agree on the input. The loader is therefore sharded across
the *data-parallel* world:

```
dp_world_size = world_size // tp_size      (= 1 in the basic version, tp_size == world_size)
dp_rank       = rank // tp_size
```

`tokens_seen` counts the data-parallel world only, and `global_batch_size` is validated against
`micro × accum × dp_world_size`. Getting this wrong would silently inflate every throughput figure
by `tp_size` — which is why a preset's declared global batch needs no change when TP is switched
on, unlike DDP or FSDP.

The TP group comes from a 2-D `(dp, tp)` device mesh, even though the `dp` dimension has extent 1
here. Building both named dimensions now is what makes 2D TP+DP composition a configuration change
later rather than a rewrite.

## Checkpoints

Each checkpoint writes:

- `ckpt_NNNNNN_tp{rank}.pt` — **per rank**. Every rank owns different slices *and* its own Adam
  moments, so there is no single file that resumes the run. This is what resume reads.
- `ckpt_NNNNNN_full.pt` — the shards all-gathered into a full state dict and written from rank 0 in
  the single-device format, so a TP result loads into a plain `GPT` anywhere. Same reasoning as the
  FSDP consolidation path (D21).

Sharding is a pure function of the parameter name, which is why module names are preserved during
surgery:

| Parameter | Split |
|---|---|
| `attn.qkv.{weight,bias}`, `mlp.fc1.{weight,bias}` | column: dim 0 |
| `attn.out_proj.weight`, `mlp.fc2.weight` | row: dim 1 |
| `attn.out_proj.bias`, `mlp.fc2.bias`, norms, embeddings | replicated |

**A row-parallel layer's bias is replicated and added after the all-reduce.** Adding it before
would include it once per rank — `tp_size` times. That asymmetry (column-parallel biases split,
row-parallel biases not) is the easiest thing to get wrong here, so it has its own test and two
mutations.

## Running it

```bash
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    -m minimegatron.train --config configs/tiny.yaml --set distributed.strategy=tp
```

`tp_size` defaults to the whole world. `n_heads % tp_size` and `d_ff % tp_size` are validated at
startup with an error that explains the head constraint.

Dropout must be 0 for any comparison against the reference. Correct TP dropout needs a
model-parallel RNG tracker — different seeds inside sharded regions, the same seed in replicated
ones — which is documented and deferred (D18).

## Benchmarks

*Pending hardware.* TP=2 vs DDP vs FSDP vs one GPU: throughput, step latency, peak memory per rank,
communication volume.

The expected result is worth stating **before** measuring, so it is not misread as a bug: **TP
should lose to DDP on PCIe.** It all-reduces twice per block per direction, so without
NVLink-class bandwidth the communication dominates. That is the honest, instructive result, and the
reason Megatron confines tensor parallelism to within a single node and composes it with data
parallelism across nodes.

## Out of scope here

Sequence parallelism, pipeline parallelism, vocab-parallel embedding and cross-entropy, 2D TP+DP
training, and TP combined with FSDP. The `(dp, tp)` mesh and the strategy switch are the seams
those would attach to.
