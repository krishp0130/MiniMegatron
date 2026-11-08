# Testing

The testing strategy, what each suite is for, and what it has actually caught. SCOPE.md calls the
QA framework "a critical component"; this is the account of it.

## Current state

**455 passed, 37 skipped** (`pytest -q`, CPU, ~2 min). All 37 skips need CUDA hardware this machine
does not have, and each states its reason.

| Suite | Tests | What it is for |
|---|---|---|
| `tests/unit` | 261 | shapes, config validation, param groups, schedules, checkpoint round trips, data determinism |
| `tests/distributed` | 88 | DDP, FSDP2 and TP correctness on **2 gloo ranks on CPU** |
| `tests/kernels` | 73 | RMSNorm, parameterized over backend (34 Triton parameters skip — ISSUES #16) |
| `tests/numerical` | 31 | reference vs optimized implementations, dtype-appropriate tolerances |
| `tests/fault_injection` | 27 | controlled failures, asserting the **diagnostic**, not just that something raised |
| `tests/regression` | 12 | golden trajectories and structural invariants |

```bash
.venv/bin/pytest -q                        # everything
.venv/bin/pytest -q -m "not distributed"   # fast inner loop (~20 s)
.venv/bin/pytest -q -m "not gpu"           # what CI runs
.venv/bin/pytest tests/regression -q       # golden values
.venv/bin/pytest tests/fault_injection -q  # failure modes
```

CI runs the CPU-safe suite on every push, **including the 2-process gloo tests** — they are the
project's main correctness evidence and need no GPU (D2). It also asserts that at least 300 tests
were collected, because a typo in a `-m` expression otherwise runs almost nothing and still reports
success (D30). A green tick that means nothing is worse than a red one.

## The central discipline: everything has a reference

> Every optimization must have a reference implementation, a correctness test, and a reproducible
> benchmark. — SCOPE.md

In practice this means **no optimization is ever compared against another optimization**. The
single-device FP32 `GPT` is the reference, and DDP, FSDP2, tensor parallelism and the Triton kernel
are each checked against *it*:

- `model/attention.py` has a `reference` path written out op by op, and `sdpa` is checked against it.
- `model/layers.py::rms_norm` was written in M1 **specifically** as the reference the M5 Triton
  kernel would be judged by — three milestones before the kernel existed.
- Every distributed strategy is checked against a single-process run at the same global batch.

### Equivalence tests cannot catch bugs in shared code (D11)

A reference-vs-optimized comparison only tests the part that *differs*. Both attention backends
share `split_qkv` and the projections, so a wrong QKV layout passes every equivalence test —
confirmed by mutation in M1.

So shared code gets its own direct property tests. The rule applied throughout: every comparison
test needs an independent property test for whatever both sides share. Examples: behavioural
causality (perturb position *t*, assert positions `< t` are unchanged), the QKV-layout test that
asserts a contiguous slice holds whole heads, and the `f`/`g` primitives tested numerically in both
directions rather than only through the model.

## Mutation testing

Every milestone ends by injecting deliberate bugs, confirming the suite catches each, and reverting.
**60 mutations across six milestones; 59 caught.** The one survivor needs a CUDA device without
bf16 support to reproduce — honestly unverifiable here, not a gap.

| Milestone | Injected | Caught |
|---|---|---|
| M1 — GPT baseline | 9 | 9 |
| M2 — DDP | 8 | 8 |
| M3 — BF16 + FSDP2 | 10 | 9 *(1 needs CUDA without bf16)* |
| M4 — tensor parallelism | 11 | 11 |
| M5 — Triton | 10 | 10 |
| M6 — reliability | 12 | 12 |

This is the single highest-value practice in the project, because it found things the test suite
could not:

| Found by mutation | Milestone |
|---|---|
| A *fix* that was indistinguishable from doing nothing — the first LR-on-resume fix pushed back the same value the optimizer had already restored | M1 |
| Validation-loss aggregation across ranks was entirely untested | M2 |
| The loss dtype under FSDP+bf16 was untested — where the ISSUES #4 promote is genuinely load-bearing, since autocast is absent | M3 |
| FSDP's gradient-sync suppression was untested | M3 |
| `ModelConfig.norm_backend` value validation was untested | M5 |
| One of my own tests was **vacuous**: it asserted the *resolved* backend, which is `pytorch` on this machine whether or not the config was read at all | M5 |

**Two surviving mutations turned out to be bad mutations, not missing tests** (M3's optimizer
ordering, M6's OOM message). That is worth checking before concluding a test is missing: a weak
mutation that changes nothing observable proves nothing about the suite.

## Functional invariants vs performance (D27)

Two mechanisms, deliberately separate:

- **`tests/regression/` asserts hard.** Golden loss trajectory, golden gradient norms, golden LR
  schedule, exact preset parameter counts, state-dict key sets, and that every config in the repo
  still loads. Golden values are legitimate because both things they depend on are pinned —
  `torch==2.14.1` exactly (D1) and index-deterministic data (D9) — and the failure message says to
  **re-record deliberately**, never to loosen the tolerance.
- **`benchmarks/regression.py` warns.** It compares against a versioned baseline in
  `results/baselines/` and is a CLI tool, not a pytest test. `--fail-on-regression` is opt-in.

The reason is SCOPE.md's own words: *"Avoid treating noisy performance measurements as deterministic
unit tests."* A throughput assertion inside the suite fails on a busy laptop and gets deleted
within a week, taking its real signal with it.

Gradient norms are pinned *separately* from losses, because ISSUES #12 showed a gradient bug can
leave the loss plausible to six decimal places while the norms are already wrong.

## Failure injection

Every case SCOPE.md §8 names, and each test asserts the **diagnostic text** rather than merely that
an exception occurred (D29). An error saying `RuntimeError: shape mismatch` is a failure; one naming
the setting to change is a feature.

| Failure | How |
|---|---|
| Invalid configuration | unknown keys (with "did you mean"), bad values, mismatched `global_batch_size`, **invalid `tp_size`** |
| Corrupt checkpoint | truncated, empty, wrong format version, missing parameter, wrong shape, directory with no metadata |
| Missing files | nonexistent `--config`, nonexistent `--resume` |
| Unsupported hardware | `device: cuda` without CUDA, `precision: bf16` without bf16, strategies needing peers on one rank |
| CUDA OOM | the error *path*, by injecting `torch.OutOfMemoryError` — the handler is what we own |
| **A worker dying mid-run** | a real `torchrun` job whose rank 1 exits via `os._exit` while rank 0 sits in an all-reduce |

The last one is the most likely to hang a suite, which is exactly why it exists. It passes in 1.5 s:
torchrun detects the casualty and tears the job down promptly.

The OOM diagnostic is worth quoting, as the standard the rest aim for:

```
out of memory on cuda:0 during training step 412.
current settings: strategy=ddp, precision=fp32, micro_batch_size=8, grad_accum_steps=2, seq_len=1024
options, least disruptive first:
  - halve train.micro_batch_size and double train.grad_accum_steps (same global batch, same maths, less peak memory)
  - set train.precision: bf16 (needs Ampere or newer on CUDA)
  - set distributed.strategy: fsdp to shard parameters, gradients and optimizer state across ranks
  - reduce model.seq_len, remembering attention memory is quadratic in it
```

## Test-design rules learned the hard way

Each of these cost real debugging time and is recorded in ISSUES so it is not rediscovered:

| Rule | Why | Issue |
|---|---|---|
| Two runs in one process must be driven **sequentially** | there is one global torch RNG, and `Trainer.__init__` reseeds it — so constructing the second trainer early rewinds the stream the first is drawing from | #6 |
| Never return tensors from a subprocess through a `Queue` | torch hands them over via shared memory, which needs the sender alive until the parent maps the storage; the harness uses files | #7 |
| Test the *consequence*, not the internals | DDP's reducer is a read-only C++ binding and cannot be patched; asserting that mid-group gradients disagree and end-group gradients agree turned out stronger anyway | #9 |
| Every distributed subprocess needs a hard Python-side timeout | macOS has no `timeout` binary and gloo hangs silently; this is what turned the FSDP deadlock into a 3-minute diagnosable failure | #1, #13 |
| A loss-at-init check needs **independent** targets | with tied embeddings, "predict the current token" is already easy at init, so `targets=idx` measures 4.46 against ln(128) = 4.85 | #14 |
| In a normalization test, `eps` is part of the expected value | at small activations `mean(x²)` can be only ~10× `eps`; a constant row of 0.01 normalizes to 0.954, not 1.0 | #15 |
| Verify a precision mode by the **dtype at each op** | autocast keeps norms, embeddings and the residual stream in fp32, so a bf16 loss barely moves — "the loss changed" is not a test | #10 |
| A test that cannot fail on the machine it runs on is not a test | the vacuous `make_norm` assertion passed whether or not the config was read | #16 context |

## What is not tested, and why

- **The Triton kernels** have never executed (ISSUES #16). 34 tests are written and parameterized
  over backend so they will judge the kernel by the reference's own assertions, and they skip here.
- **Everything GPU** — NCCL, real device memory, bf16 arithmetic on tensor cores, multi-GPU scaling.
  The gloo-on-CPU tests cover the *logic*; they cannot cover the hardware.
- **Performance**, by design, as above.
- **Real text data**, since there is none (deferred M1 stretch item).
