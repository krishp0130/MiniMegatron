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

## Resolved issues

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

