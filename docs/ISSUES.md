# Issues, Gotchas, and Decisions

Log every non-obvious problem hit and every design decision made, so nobody (human or agent)
rediscovers the same thing twice. Number entries; never renumber. Mark issues **resolved** with
the fix rather than deleting them.

## Open issues

### #3 — No NVIDIA GPU on the development machine
- **Status:** open (environmental)
- **Impact:** CUDA, NCCL, BF16-on-GPU, Triton (Milestone 5), and every GPU benchmark cannot be run
  locally. Triton does not install on macOS.
- **Plan:** develop and verify correctness on CPU (gloo for multi-process). Run GPU verification
  and benchmarks on a rented Linux + NVIDIA machine (1 GPU for Triton/BF16, 2 GPUs for
  DDP/FSDP/TP scaling). All GPU results stay marked *pending* in PROGRESS.md until measured.

### #8 — Validation loss is not comparable across world sizes
- **Status:** open (documented semantics, not a bug)
- **Detail:** `eval_steps` counts batches *per rank*, so the validation set covers
  `eval_steps x micro_batch_size x world_size` samples. A 2-rank run therefore evaluates twice as
  much data as a 1-rank run and reports a slightly different (better-estimated) number: 10.8641 vs
  10.8636 on the tiny preset. Training loss and gradients *are* exactly comparable, which is what
  the equivalence tests pin.
- **Plan:** if validation curves are ever compared across world sizes, pin the evaluation set to a
  fixed number of samples rather than a number of per-rank batches. Milestone 6's regression suite
  is the natural place.

## Resolved issues

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
