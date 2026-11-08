# Milestone 4 — Tensor Parallelism (Megatron-style)

**Goal:** split each layer's **weight matrices** across ranks, so every rank holds and computes a
slice of every layer. DDP (M2) splits the batch; FSDP (M3) shards storage but still computes whole
layers after gathering them. TP splits the *computation* itself, which is a different trade: it
needs communication inside every block's forward and backward, and in exchange no rank ever holds
a full layer.

Implemented **manually** with explicit `torch.autograd.Function`s and `torch.distributed`
collectives (D3), so every byte of communication is visible in our own code rather than inside
DTensor's dispatcher.

**Not in this milestone:** sequence parallelism, pipeline parallelism, combined 2D TP+DP training
(documented as future work), TP combined with FSDP, vocab-parallel embedding/cross-entropy
(optional stretch, skipped).

**Where it runs:** correctness on **CPU with gloo, 2 processes** (D2). Benchmarks need NVIDIA
hardware (ISSUES #3) and stay *pending*.

## Exit criteria

1. `pytest -q` passes on CPU, including the new 2-process TP tests.
2. Each primitive and each parallel layer is unit-tested against its single-device equivalent.
3. TP forward (logits and loss) matches the single-device reference within FP32 tolerance.
4. Gradients of **sharded** parameters match the corresponding slice of the reference gradient;
   gradients of **replicated** parameters match the full reference gradient and are identical
   across ranks.
5. A few optimizer steps under TP track the reference training trajectory.
6. A full state dict shards into TP shards and gathers back exactly (round trip).
7. `n_heads % tp_size` and `d_ff % tp_size` are validated with a clear error.
8. `docs/tensor_parallel.md` written; benchmarks marked *pending hardware*.

---

## The two ways to split a matmul

Take `Y = XW` with `X ∈ ℝ^{n×d}` and `W ∈ ℝ^{d×f}`.

### Column parallel: split `W` by columns

```
W = [W₁ | W₂]            W₁, W₂ ∈ ℝ^{d×f/2}
Y = X[W₁ | W₂] = [XW₁ | XW₂] = [Y₁ | Y₂]
```

Each rank needs **all of X** and produces a **slice of Y**. No communication in the forward: the
outputs are simply different columns of the same answer. In the backward, `∂L/∂X = Σᵢ (∂L/∂Yᵢ)Wᵢᵀ`
— each rank computes a *partial* input gradient, so the partials must be **summed**.

### Row parallel: split `W` by rows

```
W = [W₁ ; W₂]            W₁, W₂ ∈ ℝ^{d/2×f}
X = [X₁ | X₂]            X must already be split by columns
Y = X₁W₁ + X₂W₂
```

Each rank needs a **slice of X** and produces a **partial sum** of the whole `Y`, so the forward
needs an **all-reduce**. In the backward, `∂L/∂Xᵢ = (∂L/∂Y)Wᵢᵀ` needs only the full output
gradient, which every rank already has — so no communication.

### Why the pairing is the whole trick

Column parallel leaves its output split along the feature dimension. Row parallel *wants* its
input split along the feature dimension. Chaining **column → row** means the intermediate never
has to be gathered:

```
X --[column: no comm]--> sharded hidden --[row: one all-reduce]--> Y
```

That is **one all-reduce per pair in the forward and one in the backward**, regardless of how many
matmuls are inside. Both the MLP and attention are built as exactly one such pair.

## The `f` / `g` conjugate pair

Megatron names the two communication operators `f` and `g`. They are conjugates: each is the
identity in one direction and an all-reduce in the other.

| Operator | Forward | Backward | Sits at |
|---|---|---|---|
| `f` = `copy_to_tp_region` | identity | all-reduce (sum) | the **input** of a column-parallel layer |
| `g` = `reduce_from_tp_region` | all-reduce (sum) | identity | the **output** of a row-parallel layer |

The asymmetry is not arbitrary — it falls straight out of the calculus above:

- `f`'s input is **replicated** and consumed by every rank, so its gradient is a sum of each
  rank's contribution. Forward needs nothing; backward must sum.
- `g`'s output is a **partial sum** that must be completed, so forward must sum; and because every
  rank then holds the identical full output, the incoming gradient is already complete and
  backward needs nothing.

Two more, for moving between replicated and sharded representations (needed to compare a parallel
layer against `nn.Linear`, and the seam sequence parallelism would later use):

| Operator | Forward | Backward |
|---|---|---|
| `scatter_to_tp_region` | split along the last dim, keep this rank's chunk | all-gather |
| `gather_from_tp_region` | all-gather along the last dim | split |

Implemented as `torch.autograd.Function`s, because the backward of a collective is a *different
collective* and autograd cannot infer that.

## Parallel MLP

```
fc1: column parallel   d -> d_ff/tp      (no communication; GELU is elementwise, so local)
fc2: row parallel      d_ff/tp -> d      (one all-reduce)
```

GELU being elementwise is what makes this work: a nonlinearity applied to a *slice* of the hidden
dimension equals the corresponding slice of the nonlinearity applied to the whole thing. If the
activation mixed features (a softmax over `d_ff`, say), the hidden state would have to be gathered.

## Parallel attention: partition the heads

QKV is column parallel and `out_proj` is row parallel — the same pair. What makes it clean is that
multi-head attention is already `n_heads` independent computations, so splitting the QKV output
dimension by *whole heads* means each rank runs `n_heads/tp_size` complete heads with **no
communication inside attention at all**.

This is exactly why M1 chose the per-head fused QKV layout `[h, 3, d_h]` over GPT-2's
`[3, h, d_h]` (D6). Column-parallel splits the output dimension into *contiguous* chunks:

```
[h, 3, d_h] layout, tp=2, h=4:   rank 0 gets heads 0,1 (q,k,v each);  rank 1 gets heads 2,3   ✓
[3, h, d_h] layout, tp=2:        rank 0 gets all 4 q's and 2 k's; rank 1 gets 2 k's and 4 v's  ✗
```

A decision made three milestones ago, for this. The practical consequence in code is small but
essential: after sharding, `CausalSelfAttention.n_heads` must become the **local** head count, or
`split_qkv` will reshape the local slice with the wrong head count.

## Embeddings, LM head and norms: replicated

The basic version replicates the token embedding, position embedding, the tied LM head and every
norm on every rank (D8 keeps the tying). Their gradients are therefore already complete on each
rank and must match the single-device reference exactly — and be identical across ranks, which is
a sharper test than it sounds, because it only holds if `f`'s backward all-reduce is correct.

Vocab-parallel embedding and a vocab-parallel cross-entropy would shard the `V·d` table (9.7M
parameters of the tiny preset, 84% of it) — real savings, but out of scope here.

## Data, batches and process groups

**Every TP rank processes the same batch.** TP is not data parallelism: the ranks collectively
compute one forward pass, so they must agree on the input. The data loader therefore gets
`rank=dp_rank, world_size=dp_world_size`, where

```
dp_world_size = world_size // tp_size      (= 1 in the basic version, where tp_size == world_size)
dp_rank       = rank // tp_size
```

`tokens_seen` counts the *data-parallel* world only, and `global_batch_size` is validated against
`micro × accum × dp_world_size`. Getting this wrong would silently inflate every throughput figure
by `tp_size`.

The TP group is built from a device mesh with a named dimension, even though it spans the whole
world in the basic version — that is the seam 2D TP+DP composition would use later.

## Checkpoints

A TP checkpoint is **gathered to a full state dict on rank 0** and written in the Milestone 1
single-file format, so it loads into a plain `GPT` like every other strategy's (D21's reasoning).
Resume loads the full state dict on every rank and shards it.

This requires knowing, per parameter, how it was split:

| Parameter | Split |
|---|---|
| `attn.qkv.{weight,bias}`, `mlp.fc1.{weight,bias}` | column: dim 0 |
| `attn.out_proj.weight`, `mlp.fc2.weight` | row: dim 1 |
| `attn.out_proj.bias`, `mlp.fc2.bias`, norms, embeddings | replicated |

The bias of a row-parallel layer is replicated and added **after** the all-reduce — adding it
before would add it `tp_size` times. That asymmetry (column-parallel biases are split, row-parallel
biases are not) is the single easiest thing to get wrong here, so it gets its own test.

## Known traps

| Trap | Handling |
|---|---|
| Row-parallel bias added before the all-reduce | Add after. Directly tested, and a planned mutation. |
| `n_heads` not updated to the local count after sharding | Set during surgery; tested by comparing attention output against the reference. |
| Dropout under TP | Needs a model-parallel RNG tracker (different seeds in sharded regions, same in replicated ones). Documented and deferred; all comparisons use `dropout=0` (D18). |
| Equivalence tests hiding bugs in shared code | D11: the parallel and reference paths share `split_qkv` and the attention math, so the primitives and the state-dict round trip get their own direct property tests. |
| `tp_size` not dividing `n_heads` or `d_ff` | Validated up front with a clear error (exit criterion 7). |

## Benchmarks

*Pending hardware.* TP=2 vs DDP vs FSDP vs single GPU: throughput, step latency, peak memory per
rank, communication volume per step. The expected result is worth stating in advance so the
measurement is not read as a failure: **TP should look bad on PCIe.** It all-reduces twice per
block per direction, so without NVLink-class bandwidth it will likely lose to DDP. That is the
honest, instructive result, and the reason Megatron confines TP to within a node.
