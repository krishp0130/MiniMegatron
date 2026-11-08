# Milestone 5 — Triton Kernel Optimization

**Goal:** replace one meaningful operation with a hand-written fused GPU kernel, and prove it
numerically equivalent to the PyTorch reference it replaces. RMSNorm is the target, chosen in
Milestone 1 — `rms_norm` in `model/layers.py` was written *then* as the reference this milestone
is checked against.

**The honest constraint, stated first.** Triton needs an NVIDIA GPU and does not install on macOS
(ISSUES #3). So this milestone splits in two, and the split is recorded rather than blurred:

| Part | Status |
|---|---|
| Reference implementation | done in M1 (`rms_norm`) |
| Numerical/gradient test suite against the reference | **written and passing locally** |
| Config switch between PyTorch and Triton | **written and tested locally** |
| Triton kernels (forward, `dx` backward) | **written, NEVER EXECUTED** |
| Kernel benchmarks, memory/runtime comparison | *pending hardware* |
| Triton-vs-reference numerical equivalence | *pending hardware* |

**Nothing in this milestone may claim the kernel works.** It has never run. Every Triton test is
GPU-marked and skips here, and PROGRESS.md records the kernel as unverified until it has executed
on real hardware. The value delivered locally is everything *around* the kernel: the reference, the
tests that will judge it, the dispatch that selects it, and the fallback that makes its absence a
non-event.

**Not in this milestone:** FlashAttention (SCOPE.md forbids it in the first phase), fused GELU
(optional, deferred), and anything requiring measurement.

## Exit criteria

1. `pytest -q` passes locally, with the Triton tests skipping cleanly and saying why.
2. The numerical suite exercises the **reference** fully: shapes including non-powers-of-two,
   dtypes, extreme inputs, unusual batch shapes, and a float64 `gradcheck`.
3. The same suite compares Triton against the reference when a GPU is present — written once,
   parameterized over backend.
4. `model.norm_backend: auto | pytorch | triton` switches implementations, with `auto` falling back
   silently *only* because PyTorch and Triton are meant to be numerically equivalent; an explicit
   `triton` without Triton is a hard error.
5. A kernel benchmark exists and runs through the M6 harness, marked *pending hardware*.
6. The kernel is recorded as **written but unverified** everywhere it is mentioned.

## Why RMSNorm is the right first kernel

```
y = x / sqrt(mean(x²) + eps) · w
```

PyTorch's version of this launches a sequence of kernels — square, mean, add, rsqrt, multiply,
multiply — and each one **reads its input from HBM and writes its output back**. For a
`(B·T, d)` activation the intermediates are the same size as the input, so the op is entirely
memory-bound: the arithmetic is trivial and the time is all traffic.

A fused kernel reads the row once, keeps it in registers/SRAM, does all six operations there, and
writes the result once. The theoretical win is the ratio of passes over memory, which is why
normalization layers are the standard first Triton exercise and why a *correct but unfused* kernel
would be pointless.

It is also the right shape for a first kernel: one program per row, no cross-row communication in
the forward, and a reduction small enough to hold in one block for every `d_model` this project
uses.

## Design

### Forward

One Triton program per row. `BLOCK` is the next power of two at or above `d_model`, with a mask for
the tail, so a non-power-of-two width like the tiny preset's 192 works without a separate path.

Statistics are accumulated in **float32 regardless of the input dtype**, mirroring the reference
(and ISSUES #4): a mean of squares in bf16 loses far too much precision, and this is the one place
in the op where it matters.

### Backward: Triton for `dx`, PyTorch for `dw`

With `r = 1/sqrt(mean(x²) + eps)` and `y_i = x_i · r · w_i`:

```
dL/dx_i = r·w_i·dy_i − (r³·x_i / N) · Σⱼ(dyⱼ·wⱼ·xⱼ)
dL/dw_i = Σ_rows (dy_i · x_i · r)
```

`dx` is per-row and fuses well: one program per row, one pass, the `Σⱼ` term computed in-block.

`dw` is a reduction **across rows**, which in Triton needs either atomics or a two-stage partial
buffer with locks. SCOPE.md permits "a valid backward implementation **or an explicit supported
autograd strategy**", so this takes the second option deliberately: `dw` is computed as
`(dy * x * r).sum(0)` in PyTorch, which is a single well-optimized reduction with nothing to fuse
away. The fused win is in `dx`; adding a lock-based accumulator for `dw` would add the most
error-prone code in the kernel for the least benefit — and *especially* so when it cannot be
tested here.

That trade-off is a decision, not an omission, and it is recorded as such (D31).

### Dispatch and fallback

`model.norm_backend` mirrors the existing `attention_backend` switch, which is the same kind of
choice (which implementation, not which maths):

- `pytorch` — always the reference.
- `triton` — a hard error if Triton or CUDA is missing. No silent fallback, following the same rule
  as `device` and `precision`: a run that quietly used a different code path than its config says
  is worse than one that stops.
- `auto` — Triton on CUDA when importable, else the reference. This is the *one* place a silent
  fallback is acceptable, and only because the two paths are asserted numerically equivalent; if
  that assertion ever fails, `auto` becomes a liability and should be removed.

## Tests

`tests/kernels/test_rmsnorm.py`, parameterized over backend so the reference and the kernel are
judged by **the same assertions**:

| Test | Runs locally? |
|---|---|
| vs `torch.nn.RMSNorm`, 8 shapes incl. non-powers-of-two | yes (reference) |
| fp32 / bf16 / fp16, dtype-appropriate tolerances | yes (reference) |
| extreme inputs: zeros, tiny, huge, mixed magnitude | yes (reference) |
| float64 `gradcheck` of the function | yes (reference only — Triton has no fp64 path) |
| unit-RMS property of the output | yes |
| Triton vs reference, forward and both gradients | **no — GPU required** |
| kernel benchmark vs reference | **no — GPU required** |

M1's `tests/numerical/test_norms.py` already covers the reference against `torch.nn.RMSNorm`; this
milestone's suite adds the backend parameterization and the extreme-shape sweep, and is where the
Triton comparison will light up on a GPU.

## Known traps

| Trap | Handling |
|---|---|
| Claiming an unexecuted kernel works | every Triton test is GPU-marked and skips with a reason; PROGRESS records it as unverified |
| `@triton.jit` at import time without Triton | kernels defined inside a guarded block; the module imports fine without Triton |
| Silent fallback hiding a broken kernel | explicit `triton` is a hard error; only `auto` falls back |
| bf16 statistics | accumulate in fp32 in the kernel, as the reference does (ISSUES #4) |
| Non-power-of-two `d_model` | `BLOCK = next_power_of_2(d)` with a mask; the tiny preset's 192 exercises it |
| A kernel that is correct but not fused | the point is passes over memory, so the benchmark must compare against the reference, not against nothing |

## Benchmarks

*Pending hardware.* Kernel-level: reference vs Triton forward, backward, and forward+backward
across widths and row counts, reporting runtime and achieved bandwidth. End-to-end: the tiny and
small presets with `norm_type: rmsnorm` under both backends, through the Milestone 6 harness.

The harness, the config switch and the baseline store already exist, so this is a handful of
commands on a GPU box rather than new work.
