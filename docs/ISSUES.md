# Issues, Gotchas, and Decisions

Log every non-obvious problem hit and every design decision made, so nobody (human or agent)
rediscovers the same thing twice. Number entries; never renumber. Mark issues **resolved** with
the fix rather than deleting them.

## Open issues

### #16 — The Triton RMSNorm kernel has NEVER BEEN EXECUTED
- **Status:** open (blocked on hardware)
- **Detail:** `kernels/rmsnorm.py` contains a fused forward kernel and a `dx` backward kernel
  written for Triton. **Neither has ever run.** There is no NVIDIA GPU on this machine and Triton
  has no macOS wheels (#3), so the code has been type-checked by nothing stronger than reading it.
  It is unverified code, not working code, and no document in this project may say otherwise.
- **What *is* verified:** the PyTorch reference (`rms_norm`, written in M1 for this purpose) across
  8 shapes, 3 dtypes, extreme inputs and a float64 gradcheck; the eps handling; the backend
  dispatch and fallback; and that an explicit `norm_backend: triton` fails fast here rather than
  falling back silently. 34 Triton-parameterized tests exist and **skip**, stating the reason.
- **To close:** run `pytest tests/kernels -q` on a Linux + NVIDIA machine. The tests are
  parameterized over backend, so the kernel is judged by the reference's own assertions with no
  new test code. Then `python -m minimegatron.benchmarks.kernels` for the runtime and bandwidth
  comparison. Expect the first run to surface real bugs; that is the normal outcome for a kernel
  written blind.

### #3 — No NVIDIA GPU on the development machine
- **Status:** open (environmental)
- **Impact:** CUDA, NCCL, BF16-on-GPU, Triton (Milestone 5), and every GPU benchmark cannot be run
  locally. Triton does not install on macOS.
- **Plan:** develop and verify correctness on CPU (gloo for multi-process). Run GPU verification
  and benchmarks on a rented Linux + NVIDIA machine (1 GPU for Triton/BF16, 2 GPUs for
  DDP/FSDP/TP scaling). All GPU results stay marked *pending* in PROGRESS.md until measured.

### #10 — Autocast leaves the residual stream in FP32, so BF16 losses barely move
- **Status:** open (documented behaviour, not a bug)
- **Symptom:** a `precision: bf16` run on the tiny preset printed losses *identical to FP32 at four
  decimal places* (10.8624, 10.8704) with identical gradient norms, which looked like autocast was
  not active at all. Throughput told a different story: ~153 vs ~18,000 tokens/s, a ~100x slowdown
  from emulated bf16 on Apple silicon.
- **Cause:** autocast has an op table. `Linear` runs in bf16; normalization stays fp32 (its
  reductions need the range); `Embedding` is untouched. The residual stream is therefore
  `fp32 + bf16`, which **promotes back to fp32** and is never downcast. Only the matmuls are low
  precision, and at initialization — where logits are near-uniform and the loss is dominated by
  ln(V) — their rounding moves the loss by less than 1e-4.
- **Resolution:** this is autocast working as designed, and the reason mixed precision is safe.
  `tests/unit/test_precision.py::test_matmuls_run_in_bf16_while_parameters_stay_fp32` pins the
  dtype at four points (linear bf16, norm fp32, embedding fp32, block output fp32) instead of
  checking a single activation, so the distinction is asserted rather than inferred. The lesson:
  verify a precision mode by the dtype at each op, never by whether the loss changed.

## Resolved issues

### #15 — `eps` is not negligible at small activation scales
- **Status:** resolved (test-design rule), 2026-11-08
- **Symptom:** a kernel test asserted that a constant row normalizes to exactly 1.0 and measured
  0.954 instead — a 4.7% discrepancy that looked like a precision bug.
- **Cause:** the row's value was 0.01, so `mean(x²) ≈ 1e-4`, only ten times `eps = 1e-5`. With eps
  inside the square root — where it belongs — the correct answer *is* 0.954. The test's premise
  that eps could be ignored was simply false at that scale.
- **Fix:** the test now compares against the full float64 formula *including* eps, and additionally
  asserts the output is below 0.97 — so it fails if eps is dropped or moved outside the square
  root. A mistake in a test turned into a test that pins something real: a kernel that omitted eps
  would pass every well-scaled case and fail only here.
- **Lesson:** when a normalization test uses small activations, eps is part of the expected value,
  not a rounding detail. Compare against the exact formula rather than an idealized limit.

### #14 — Loss at init is *below* ln(V) when the targets are the inputs
- **Status:** resolved (test-design rule), 2026-11-08
- **Symptom:** a loss-at-init regression test measured 4.46 against the expected
  ln(128) = 4.85 and failed, suggesting a broken initialization.
- **Cause:** the test passed `targets=idx`. With **tied embeddings** the residual stream carries
  `tok_emb(x[t])`, and the tied LM head scores exactly that token highly -- so "predict the token
  at this position" is already easy at initialization. Nothing was broken; the task was.
- **Fix:** draw the targets independently of the inputs, as the Milestone 1 test already did. Any
  loss-at-init check must use independent targets, or it measures the model's ability to copy
  rather than its initialization.

### #13 — Unbalanced evaluation batches deadlock FSDP
- **Status:** resolved, 2026-11-08
- **Symptom:** after making the evaluation set a fixed global size (#8), the FSDP validation test
  **hung**, and both ranks were killed by the harness timeout after 180 s. DDP and TP passed.
- **Cause:** distributing `eval_steps = 5` global batches over 2 ranks gives one rank 3 batches and
  the other 2. Under FSDP the **forward itself contains collectives** (the per-unit all-gather),
  so the rank with fewer batches left the loop and the other blocked forever in an all-gather with
  no peer. DDP was fine because a `no_grad` forward has no collective, and TP was fine because
  `dp_world_size` is 1 there so every rank ran every batch.
- **Fix:** every rank runs `ceil(eval_steps / dp_world_size)` forward passes, and the surplus ones
  are computed and weighted **zero**. The collectives stay balanced and the arithmetic is still a
  mean over exactly batches 0..eval_steps-1.
- **Lesson:** under FSDP (or any strategy whose forward communicates), the *number of forward
  passes* is part of the contract between ranks, not just the data. Any per-rank loop bound must be
  rank-independent. The hard timeout in the test harness is what turned a hang into a diagnosable
  failure within three minutes -- the ISSUES #1 discipline paying for itself.

### #8 — Validation loss was not comparable across world sizes
- **Status:** resolved, 2026-11-08 (Milestone 6)
- **Symptom:** `eval_steps` counted batches *per rank*, so a 2-rank run evaluated twice the data
  and reported a different number (10.8641 vs 10.8636 on the tiny preset). Training loss was
  exactly comparable; validation was not -- useless for precisely the cross-strategy comparison
  this project exists to make.
- **Fix:** `eval_steps` now counts **global** evaluation batches. The validation stream is
  unsharded, so micro-batch *j* is the same samples on every rank; rank *r* handles
  `j = r, r + dp, ...`; and the loss **sum** and the **count** are both all-reduced so the mean is
  taken globally. The evaluation set is exactly `eval_steps x micro_batch_size` samples at any
  world size, and a 1-rank run is unchanged.
- Reducing a sum and a count rather than a mean-of-means is what makes it correct when
  `dp_world_size` does not divide `eval_steps`: the ranks then evaluate different numbers of
  batches, and averaging their means would weight the short rank too heavily -- the unequal-shard
  trap of D24, one level up. Fixing this also exposed #13.

### #12 — `clip_grad_norm_` computes a rank-local norm under tensor parallelism
- **Status:** resolved, 2026-11-08
- **Symptom:** a TP run's logged gradient norm was 1.26595 where the single-device reference
  reported 1.296081 — close enough to look like float noise, which is what makes it dangerous.
  Losses matched to six decimal places, so nothing else hinted at a problem.
- **Cause:** under TP, `model.parameters()` yields **ordinary tensors that happen to be slices**,
  with nothing marking them as sharded. So each rank computed a norm over its own slices plus the
  replicated parameters: neither the global norm nor even the same value on every rank. This is
  precisely where TP differs from FSDP, whose `DTensor` gradients carry their sharding and let the
  stock function reduce correctly — so the M3 experience ("clipping just works") was misleading.
- **Fix:** `clip_grad_norm_tp` splits the parameters by `tp_shard_dim` and treats them
  differently: sharded squared norms are **all-reduced** (`‖g_full‖² = Σᵣ ‖g_shardᵣ‖²`), replicated
  ones are counted **once** and not reduced (reducing them would count them `tp_size` times). Both
  halves are mutation-tested, because each produces a plausible-looking wrong number rather than
  a crash.
- **Lesson:** a correct loss does not imply correct clipping. Anything that aggregates *across all
  parameters* — norms, global statistics, logging — needs its own treatment under any form of
  model parallelism.

### #11 — Attention's head-concatenation reshape assumed the full `d_model`
- **Status:** resolved, 2026-11-08
- **Symptom:** every TP model test failed with
  `RuntimeError: shape '[8, 16, 32]' is invalid for input of size 2048`.
- **Cause:** `CausalSelfAttention.forward` took `B, T, d = x.shape` and reshaped the concatenated
  heads back to `d`. Under TP this rank holds only `n_heads/tp_size` heads, so its attention output
  is `d_model/tp_size` wide and must feed a row-parallel `out_proj` at that width — but the reshape
  demanded the full `d_model`.
- **Fix:** `y.transpose(1, 2).reshape(B, T, -1)`. The width is now derived from the heads the
  module actually owns rather than from its input's width, which is correct for both the
  single-device and the tensor-parallel case.
- **Why it survived three milestones:** nothing before M4 ever gave a layer a different width than
  the model's `d_model`, so the assumption was invisible. D4 ("linear layers are separate named
  modules so TP can replace them without restructuring the model") was *nearly* enough — the
  layers were replaceable, but one shape assumption outside them was not.

### #9 — DDP's reducer cannot be monkeypatched, so count its effects instead
- **Status:** resolved (test-design rule), 2026-11-08
- **Symptom:** a test that wrapped `trainer.model.reducer.prepare_for_backward` to count
  synchronizations failed with
  `AttributeError: 'torch._C._distributed_c10d.Reducer' object attribute 'prepare_for_backward' is read-only`.
- **Cause:** the reducer is a C++ binding, and DDP's all-reduces go through the ProcessGroup
  directly rather than through Python's `dist.all_reduce`, so neither the object nor the call can
  be patched from Python.
- **Fix:** test the *observable consequence* instead, which turned out to be a stronger test
  anyway: after the first micro-batch of a `no_sync` group the ranks' gradients must **disagree**
  (each holds only its own shard's partial), and after the last they must agree exactly. That is
  what "one synchronization per optimizer step" means operationally, and it does not depend on
  DDP's internals.

### #7 — Passing torch tensors through a `multiprocessing.Queue` deadlocks the parent
- **Status:** resolved, 2026-11-08
- **Symptom:** the 2-process DDP tests hung indefinitely. The same work, run from a standalone
  script, completed in under a second with both ranks reporting identical losses and exit code 0 —
  so DDP was fine and the *harness* was at fault.
- **Cause:** `torch.multiprocessing` serializes tensors by handing over shared-memory file
  descriptors, which requires the **sending process to stay alive** until the receiver has mapped
  the storage. The workers put their results (dicts of gradient tensors) on the queue and exited
  immediately, so the parent's `get()` waited forever.
- **Fix:** workers `torch.save` their result to a per-rank file in a temp directory and the parent
  loads it after joining. `torch.save` copies bytes to disk, so the handoff survives the sender's
  exit; it also makes a dead rank diagnosable (a missing file plus its exit code) rather than an
  indefinite wait. Any future harness that returns tensors from a subprocess must do the same.

### #6 — One global torch RNG makes interleaved trainers unsafe in tests
- **Status:** resolved (test-design rule), 2026-11-07
- **Symptom:** a test that built two `Trainer`s, then stepped them alternately, saw different
  losses from identical configs, which looked like a bug in `Trainer.evaluate`.
- **Cause:** there is one global torch RNG. `Trainer.__init__` calls `seed_everything`, so
  constructing the second trainer rewound the stream the first one was drawing dropout masks
  from. The trainers were identical; the test was not.
- **Fix:** any test comparing two runs in one process must drive them **sequentially** — build
  and run A to completion, then build and run B (whose construction reseeds). Verified directly:
  `evaluate()` leaves the torch RNG, the weights, and `model.training` untouched.
  Milestone 2's DDP equivalence tests will hit this same trap.

### #5 — Import cycle between `experiment.py` and `training/`
- **Status:** resolved, 2026-11-07
- **Symptom:** `ImportError: cannot import name 'ExperimentConfig' from partially initialized
  module 'minimegatron.experiment'`.
- **Cause:** `experiment.py` imports `training.config.TrainConfig`, which executes
  `training/__init__.py`, which imports `trainer.py`, which imported `experiment.py`.
- **Fix:** `trainer.py` needs `ExperimentConfig` only as a type annotation, so it is imported
  under `if TYPE_CHECKING:` (`from __future__ import annotations` is already on, so annotations
  are never evaluated at runtime). Package `__init__.py` files that re-export eagerly make a
  cycle out of any config module a leaf imports; keep type-only imports deferred.

### #4 — `rms_norm` silently downcast float64 inputs to float32
- **Status:** resolved (2026-10-25)
- **Symptom:** float64 `gradcheck` of `rms_norm` failed: numerical and analytical Jacobians
  differed around the 3rd decimal place.
- **Cause:** "compute statistics in fp32" was written as `x.float()`, which *converts to* float32,
  so it is an upcast for bf16 but a downcast for float64.
- **Fix:** `x.to(torch.promote_types(x.dtype, torch.float32))`: at least float32, never lower. Use
  the same pattern for every "upcast for numerical stability" site (the Triton kernel and its
  wrappers in M5 included).

### #1 — torchrun + gloo hangs on macOS
- **Status:** resolved (workaround)
- **Symptom:** `torchrun --standalone --nproc_per_node=2 ...` hangs indefinitely at process-group
  init on macOS; no error.
- **Cause:** gloo picks a non-loopback/IPv6 interface from the hostname
  (`1.0.0.0...ip6.arpa`) that the processes can't reach each other on.
- **Fix:** pin gloo to loopback and use an explicit address instead of `--standalone`:
  ```bash
  export GLOO_SOCKET_IFNAME=lo0
  torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 ...
  ```
  Test helpers that spawn distributed subprocesses must set these and use a hard timeout
  (macOS has no `timeout` binary; enforce it in Python with `subprocess.run(timeout=...)`).

### #2 — FSDP2 `fully_shard` crashes on Mac: `torch.mps has no attribute 'is_initialized'`
- **Status:** resolved (workaround), torch 2.14.1
- **Symptom:** `fully_shard(module)` without a mesh raises
  `AttributeError: module 'torch.mps' has no attribute 'is_initialized'` from
  `device_mesh._setup_world_group_and_device`.
- **Cause:** with no mesh given, FSDP2 builds a default mesh for the auto-detected accelerator,
  finds MPS, and hits an incomplete MPS code path.
- **Fix:** always pass an explicit mesh on CPU:
  `mesh = init_device_mesh("cpu", (world_size,)); fully_shard(m, mesh=mesh)`.
  The framework's device/mesh setup should always construct meshes explicitly from the configured
  device type rather than relying on auto-detection.

## Design decisions

### D1 — Python 3.12, torch pinned to 2.14.1
3.12 has the broadest wheel coverage across torch/triton/tokenizers while satisfying the 3.11+
requirement. Torch is pinned exactly in `pyproject.toml`. Bump deliberately and re-run the full
test suite when changing it.

### D2 — gloo-on-CPU as the distributed correctness backend
Distributed correctness tests (DDP, FSDP2, TP) run with 2 processes on CPU via gloo, so they work
in CI and on machines without GPUs. NCCL-specific behavior and performance are verified separately
on GPU hardware.

### D3 — Manual tensor parallelism, DTensor TP as cross-check only
TP is implemented with explicit autograd functions and `torch.distributed` collectives (Megatron
style) for educational transparency. `torch.distributed.tensor.parallel` may be used as an
independent reference implementation in tests, not as the primary path.

### D4 — Model linear layers are separate named modules
Attention uses `qkv`/`out_proj` and the MLP uses `fc1`/`fc2` as distinct `nn.Linear` modules, so TP
can replace them with column/row-parallel layers without restructuring the model.

### D5 — Optional dependency extras
Core install is torch + numpy + pyyaml. `dev` (pytest), `text` (tokenizers), `logging`
(tensorboard), and `gpu` (triton, Linux only) are extras.

### D6 — Per-head fused QKV layout `[h, 3, d_h]`
The fused `qkv` projection output is viewed as `(B, T, n_heads, 3, head_dim)`, i.e. each head's
q, k, v are contiguous. Column-parallel TP (M4) splits that dimension into contiguous chunks, so
each rank then owns complete heads. GPT-2's `[3, h, d_h]` layout would hand rank 0 all the queries
and half the keys. Trade-off: loading GPT-2 pretrained weights would require a permutation (not
planned).

### D7 — Vocab padded to 50304
GPT-2 BPE (50257) is padded to a multiple of 128 for aligned GPU matmuls and TP divisibility. The
padded token ids are never produced by the tokenizer.

### D8 — Tied input embedding / LM head
Saves V·d params (9.7M in tiny) and matches GPT-2. FSDP2 must keep both uses in the root sharding
unit; TP keeps both replicated in the basic version.

### D9 — Index-deterministic datasets and step-deterministic batching
Synthetic dataset items are a pure function of `(seed, index)`, and the batch at global step `s` is
a pure function of `(seed, s)`. This makes resume exact without saving dataloader state and makes
DDP sharding reproducible.

### D10 — Strict configuration loading
- Config dataclasses are **frozen** and validate in `__post_init__`, so an invalid or mutated
  config can't exist once a model is built from it.
- Validation **collects every error** and raises one `ConfigError` (a `ValueError` subclass), so a
  broken YAML is fixed in one pass, not one error per run.
- **Unknown keys are rejected** (with a difflib "did you mean" hint): a typo like `n_layer` would
  otherwise silently keep the default. Missing required keys are rejected too.
- `bool` is rejected where an int is expected (`True` is an `int` subclass in Python).
- YAML files in `configs/` are the **single source of truth** for presets; there is no duplicate
  preset table in code. Tests load the YAML and assert exact parameter counts.
- Module split: generic helpers (`ConfigError`, `from_dict`, `load_yaml`) live in
  `minimegatron/config.py`; section composition (`ExperimentConfig`, `load_config`) lives in
  `minimegatron/experiment.py`. Any section's config (model, train, data) imports the helpers with
  no import cycle.

### D11 — Equivalence tests can't catch bugs in shared code
Reference-vs-optimized comparisons only test the part that differs between the two paths. Both
attention backends share `split_qkv` and the projections, so a wrong QKV layout passes every
equivalence test (confirmed by mutation, 2026-10-25). Shared code needs its own direct
property tests (here: `test_qkv_layout_splits_into_whole_heads`, plus behavioral causality and
gradcheck). Apply the same rule to TP, FSDP, and Triton: every comparison test needs an
independent property test for the code both sides share.


### D12 — The dataset index stream is defined over samples, not batches
`IndexStream` is one infinite deterministic sequence of dataset indices — the concatenation of a
seeded permutation per epoch — and a micro-batch is a slice of it at
`micro_index * micro_batch_size`. So the sample *order* is a pure function of
`(seed, sample_offset)` and does not depend on `micro_batch_size` or `grad_accum_steps`.

Two things fall out, both of which would otherwise need extra machinery:
- `grad_accum_steps=4, micro_batch_size=2` consumes byte-identical samples to
  `grad_accum_steps=1, micro_batch_size=8`, so the gradient-accumulation equivalence test is
  exact rather than statistical. Milestone 2's DDP test rests on the same property.
- "Resume at step s" needs no dataloader state in the checkpoint at all: the step counter is the
  stream position.

### D13 — NumPy RNG state is stored as a tensor so checkpoints load with `weights_only=True`
`torch.load(weights_only=True)` refuses to unpickle arbitrary objects, which is the right default
for a file that may not be ours. NumPy's MT19937 state tuple contains a `uint32` ndarray, which
that loader rejects, so `checkpoint.py` converts the key array to an int64 tensor on save and
rebuilds the tuple on load. The alternative — `weights_only=False` — would mean every checkpoint
load is an arbitrary-code-execution path, which is a poor trade for a framework whose Milestone 6
deliberately feeds it corrupt checkpoints.

### D14 — On resume, the learning rate is recomputed from the live schedule
Restoring a run turned out to have a subtlety with two halves:
- `LRScheduler.load_state_dict` restores `last_epoch` but does **not** write a learning rate into
  the optimizer; that only happens inside `step()`.
- The rate nevertheless survives, because the *optimizer's* `state_dict` carries its param groups,
  `lr` included.

So the rate comes back — but it is the rate the **previous** run's schedule produced. Resuming
with a changed schedule (`--set train.max_steps=...`, a longer warmup) would silently take one
step on the old curve. `Trainer.resume` therefore calls `set_lr_for_step(optimizer, cfg, step)`,
computing the rate from the live config. Found by mutation testing: the first version of this fix
pushed the scheduler's *stored* rate back instead, which no test could distinguish from doing
nothing, because the optimizer had already restored the same value.

### D15 — Validation measurements are never dropped to fit the logging cadence
The trainer logs every `log_interval` steps, and an evaluation that lands between two logging
boundaries gets a JSONL line of its own rather than being discarded or deferred. Logs are keyed by
`step`, so an extra line is harmless to parse, whereas a silently skipped validation point is a
hole in the curve.

### D16 — `global_batch_size` is validated, not derived
Given `global_batch_size`, the trainer could *derive* `grad_accum_steps` as
`global / (micro x world_size)`. It deliberately does not: it checks the identity and refuses to
start on a mismatch.

Deriving would silently change the accumulation count when the world size changed, so the same
config would mean different things on different launches — the class of thing this project's strict
config loading exists to prevent (D10). Validation instead makes the contract explicit and the
error actionable:

```
ConfigError: global_batch_size 16 != micro_batch_size 8 x grad_accum_steps 2 x dp_world_size 2 = 32
```

The lesson it teaches is the right one: **to hold the global batch fixed while adding ranks, reduce
`grad_accum_steps`.** Verified on the tiny preset — 1 rank at micro=8 x accum=2 and 2 ranks at
micro=8 x accum=1 produce *identical* losses and gradient norms, differing only in throughput.

### D17 — Rank sharding formula
Micro-batch *m* on rank *r* of *R* reads the global index stream starting at
`m * micro_batch_size * R + r * micro_batch_size`.

So the union over ranks of micro-batch *m* is exactly the single-process micro-batch of size
`micro_batch_size * R` at the same index. This is what keeps the DDP equivalence tests exact rather
than statistical, and it follows directly from M1's choice to define the stream over samples rather
than batches (D12). Every rank gets the same number of samples, with no padded or dropped tail:
unequal shards would silently reweight the averaged gradient, because the mean of per-rank means is
only the mean over the global batch when the shards are the same size.

### D18 — DDP equivalence tests run with `dropout=0`
Identically seeded ranks draw identical dropout masks, which is wrong for data parallelism (the
replicas should sample independently). Differently seeded ranks can't be compared against a
single-device run at all. Correct per-rank dropout needs a model-parallel RNG tracker — different
seeds in sharded regions, the same seed in replicated ones — which is deferred, exactly as the
tensor-parallelism scope already defers it. Every equivalence test therefore sets `dropout=0`, and
`grad_clip=0` so the comparison is on raw gradients.

### D19 — `dtype` is parameter storage; `precision` is compute
Two separate settings, deliberately:
- `ModelConfig.dtype` — parameter **storage**, always `float32`.
- `TrainConfig.precision` — **compute**, `fp32` or `bf16`.

Mixed precision means bf16 math over **fp32 master weights**: the optimizer step, weight decay and
gradient clipping all stay fp32. bf16 has 8 mantissa bits, so `w + small_update == w` far more
often than in fp32, and keeping updates in low precision is how mixed-precision training quietly
stops converging. `ModelConfig`'s rejection of `dtype: bfloat16` now says this and points at
`train.precision` rather than at "Milestone 3".

The loss needs no precision-specific code at all: `GPT.forward` already promotes logits with
`torch.promote_types(dtype, float32)` before cross-entropy (ISSUES #4), so the log-sum-exp over the
vocabulary is fp32 under autocast for free. An M1 bug fix paid for an M3 requirement.

### D20 — FSDP sharding units are transformer blocks
`fully_shard` is applied to each `Block`, then to the root. Each call creates one *unit* — the
granularity at which parameters are all-gathered before use and freed after.

- Per-block: peak transient memory is one block's parameters. This is the actual saving.
- Root only: parameters would be *stored* sharded but the forward would gather the whole model at
  once, so peak memory would be no better than DDP. Storage sharding without gather granularity
  buys nothing.
- Finer (per `Linear`): more, smaller collectives with fixed overhead each, for a saving already
  small next to one block.

Two consequences recorded because both fail *quietly* rather than loudly:
- **The optimizer must be built after `fully_shard`**, which replaces parameters with `DTensor`s in
  place. An optimizer built earlier allocates full-size `m` and `v` and updates tensors the model
  no longer uses — the run trains fine and the memory saving simply does not happen.
- **The tied embedding stays in the root unit** (D8). Wrapping only blocks and the root gives this
  for free, but it is asserted rather than argued: a shared parameter gathered and freed on two
  schedules produces subtly wrong gradients, not a crash.

Verified in a spike before any code was written: after sharding, `to_local().numel()` sums to
exactly `P/R` (10,880 of 21,760 at R=2), and `clip_grad_norm_` computes the correct **global** norm
across `DTensor` shards — both ranks reported an identical 1.36097407341, so no hand-rolled
collective was needed.

### D21 — FSDP writes two checkpoint formats, and RNG per rank
Every FSDP checkpoint produces both:
1. `ckpt_NNNNNN/` — sharded via `torch.distributed.checkpoint`; each rank writes its own shard and
   nothing is gathered, so a model too large for one device still saves. Resume uses this.
2. `ckpt_NNNNNN_full.pt` — a full fp32 state dict gathered on rank 0 and written through M1's
   single-file writer.

The second is not redundant. It keeps checkpoints **interchangeable across strategies**, which is
what lets an FSDP run be compared against the single-device reference at all — the milestone's
actual evidence — and lets a sharded run's weights be loaded for inference with no FSDP present.
It works unchanged because `state_dict` does not deduplicate tied weights: both `tok_emb.weight`
and `lm_head.weight` appear, exactly as in a plain `GPT.state_dict()`.

**RNG state is saved per rank** (`rng_0.pt`, `rng_1.pt`), not only from rank 0. Random state differs
by rank, so restoring one rank's streams everywhere would make a resumed run diverge from an
uninterrupted one — M1's reasoning for capturing RNG at all, extended to a cluster.

### D22 — Autocast is skipped under FSDP
`precision: bf16` uses `torch.autocast` for `single` and `ddp`, and FSDP2's
`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` under `fsdp` — never both.

They are not interchangeable. `param_dtype=bf16` stores shards in fp32 but **all-gathers them as
bf16**, halving the volume of FSDP's dominant collective; autocast converts at the op boundary,
after the gather has already moved fp32 bytes. Under FSDP the gather dtype *is* the bandwidth
story. `reduce_dtype=fp32` then keeps the gradient reduce-scatter accumulating in fp32, because
summing many small bf16 gradients loses precision where it matters and the reduction is cheap next
to the gather.

Since the policy also casts forward inputs, autocast on top would be redundant double-casting, so
`Trainer._autocast()` returns a null context under FSDP.

### D23 — Tensor parallelism is hand-written, not DTensor
Confirms D3 in code: the `f`/`g` conjugate pair and the scatter/gather variants are
`torch.autograd.Function`s calling `torch.distributed` collectives directly. The backward of a
collective is a *different* collective, which autograd cannot infer, so each one is spelled out.

The point is pedagogical — every byte of communication is visible in our own code — but it also
means TP gets none of DTensor's conveniences, which is exactly how ISSUES #12 happened.

### D24 — TP ranks are not data-parallel ranks
Tensor-parallel ranks jointly compute **one** forward pass over **one** batch, so they must all
see the same input. The data-parallel world is therefore

```
dp_world_size = world_size // tp_size      (1 in the basic version)
dp_rank       = rank // tp_size
```

and `BatchStream` is sharded over *that*, `tokens_seen` counts *that*, and `global_batch_size` is
validated against `micro × accum × dp_world_size`. Getting it wrong would silently inflate every
throughput figure by `tp_size`.

A pleasant consequence: a preset's declared `global_batch_size` needs **no change** when TP is
switched on, unlike DDP or FSDP where ranks are data-parallel and `grad_accum_steps` has to come
down to compensate (D16).

### D25 — Parallel layers are installed by surgery, with names preserved
`parallelize_model` replaces the four `nn.Linear`s inside each block in place and leaves the module
*names* untouched, so `state_dict` keys are identical to a single-device model's. That makes
sharding a pure function of the parameter name (`tp_shard_dim`), which in turn is what lets the
same function drive the state-dict round trip, the checkpoint gather, and the test that compares
gradients against the reference. One source of truth for "what is split where".

Two things surgery has to fix up beyond swapping the layers, both easy to miss and both tested:
`attn.n_heads` must become the local count, and the attention output reshape must not assume the
full `d_model` (ISSUES #11).

### D26 — Row-parallel biases are replicated and added after the all-reduce
A column-parallel layer's bias is split along the output dimension, like its weight. A
row-parallel layer's bias is **not** split: every rank holds a full copy, and it is added *after*
the `g` all-reduce. Adding it before would include it once per rank, i.e. `tp_size` times.

This asymmetry is the single easiest thing to get wrong in a TP implementation, so it gets a direct
test (the bias is a full copy and its gradient matches the reference in full, not as a slice) plus
two mutations (bias added before the reduction; bias split instead of replicated).

### D27 — Functional regressions are assertions; performance regressions are warnings
Two different mechanisms, deliberately kept apart:

- **`tests/regression/`** asserts *functional* invariants hard: golden loss trajectories, golden
  gradient norms, exact preset parameter counts, state-dict key sets, and that every config in the
  repo still loads. Golden values are legitimate because both things they depend on are pinned --
  `torch==2.14.1` exactly (D1) and index-deterministic data (D9) -- and the failure message says
  to **re-record deliberately**, never to loosen the tolerance.
- **`benchmarks/regression.py`** compares a fresh measurement against a versioned baseline in
  `results/baselines/` and *warns* past a threshold. It is a CLI tool, not a pytest test, and
  `--fail-on-regression` is opt-in.

The reason is SCOPE.md section 8's own words: *"Avoid treating noisy performance measurements as
deterministic unit tests."* A throughput assertion inside the suite fails on a busy laptop and
gets deleted within a week, taking its real signal with it. Gradient norms are pinned separately
from losses, because ISSUES #12 showed a gradient bug can leave the loss plausible for several
steps while the norms are already wrong.

### D28 — What makes a benchmark number worth recording
The harness enforces four things, and each is the difference between a number worth keeping and a
number worth deleting:

1. **Warmup discarded** — early steps include allocator growth, kernel autotuning and lazy init.
2. **Device synchronized at every boundary** — CUDA and MPS enqueue asynchronously, so an
   unsynchronized clock measures Python's enqueue rate. On CUDA, `cuda.Event` additionally gives
   device-side time.
3. **Repeats, reported as median with min/max/IQR** — one measurement cannot distinguish a
   regression from a busy neighbour process.
4. **Unmeasurable fields are `null`, never 0 and never estimated** — `max_memory_allocated` is
   CUDA-only, so on CPU and MPS the memory fields are explicitly absent, and the Markdown table
   carries that note with it.

Every record also stores the commit and the full environment, so a number is traceable to what
produced it. `scripts/check_env.py` prints the same probe the harness records, so an environment
report and a benchmark record cannot disagree about the machine.

### D29 — Failures name the knob, not just the symptom
A CUDA OOM's native message says how many bytes failed to allocate, which is not actionable. The
trainer re-raises it (never swallows it -- the run must still fail) with the current
`strategy`/`precision`/`micro_batch_size`/`grad_accum_steps`/`seq_len` and the four levers in
increasing order of disruption: halve the micro-batch and double accumulation, switch to bf16,
switch to FSDP, reduce the sequence length.

The same standard applies throughout, and the fault-injection suite asserts the *diagnostic* text
rather than merely that something raised: unknown config keys suggest the intended key, an invalid
`tp_size` explains that whole attention heads are partitioned, a bad `global_batch_size` prints the
arithmetic, and a wrong checkpoint version prints both versions.

### D30 — CI runs the distributed tests, and gates on the collected count
The 2-process gloo tests run in CI rather than being deselected: they are the project's main
correctness evidence and need no GPU (D2). Only genuinely GPU-marked tests are skipped.

CI also asserts that at least 300 tests were collected. A typo in a `-m` marker expression
otherwise runs almost nothing and still reports success -- a green tick that means nothing is worse
than a red one. gloo's loopback interface is `lo` on Linux and `lo0` on macOS; the test harness
picks per platform and the workflow sets it for anything launched through torchrun directly.

### D31 — Triton for `dx`, PyTorch for `dw`
RMSNorm's backward splits into two unlike halves:

```
dL/dx_i = r·w_i·dy_i − (r³·x_i / N) · Σⱼ(dyⱼ·wⱼ·xⱼ)
dL/dw_i = Σ_rows (dy_i · x_i · r)
```

`dx` is per-row with an in-row reduction, which is exactly what fuses well — one program per row,
one pass over memory. `dw` is a reduction **across rows**, which in Triton needs either atomics or
a two-stage partial buffer with locks.

SCOPE.md section 7 permits "a valid backward implementation **or an explicit supported autograd
strategy**", and this takes the second option deliberately: `dw` is computed as
`(dy * x * r).sum(0)` in PyTorch, a single well-optimized reduction with nothing to fuse away.

The reasoning is risk against benefit. The fused win lives in `dx`; a lock-based `dw` accumulator
would be the most error-prone code in the kernel for the smallest gain — and that calculus is
decisively worse when the code **cannot be executed or tested on the development machine** (#16).
Writing the riskiest possible code blind is not a trade worth making. If a GPU benchmark later
shows `dw` dominating, it can be fused then, with a working test suite to catch the mistakes.

### D32 — `norm_backend`: one place where a silent fallback is allowed
`model.norm_backend` mirrors the existing `attention_backend` switch, since both choose an
*implementation* rather than the maths:

- **`pytorch`** — always the reference.
- **`triton`** — a hard error if Triton or CUDA is missing, and a hard error at the first forward
  if the tensors are not on CUDA. The same rule as `device` and `precision` (D29 and the no-silent-
  fallback principle): a run that quietly used a different code path than its config says is worse
  than one that stops. It fails at *construction*, not an hour into training.
- **`auto`** — Triton on CUDA when importable, else the reference.

`auto` is the only silent fallback anywhere in this project, and it is defensible for exactly one
reason: the two paths are asserted numerically equivalent by the same test suite. **If that
assertion ever stops holding, `auto` becomes a liability and should be removed rather than
patched.** Until the kernel has actually run (#16), `auto` resolves to `pytorch` everywhere it has
been exercised.

`auto` is resolved **once, at module construction**, to a concrete backend. Leaving it as `"auto"`
would mean re-resolving — including a `torch.cuda.is_available()` call — on every forward pass of
every norm in the model, which is a hot path.

`norm_backend: triton` with `norm_type: layernorm` is rejected rather than ignored: there is no
Triton LayerNorm here, so such a config asks for a kernel it can never reach, and its author
believes something false.
