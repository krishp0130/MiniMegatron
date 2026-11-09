# Benchmarks

How performance is measured, how to reproduce it, what has actually been measured, and what has
not. The last two are kept strictly apart.

## The rules (D28)

Four things make a number worth recording. Each is the difference between a measurement and a
guess:

1. **Warmup is discarded.** The first steps include allocator growth, kernel autotuning and lazy
   initialization. Timing them measures startup.
2. **The device is synchronized at every boundary.** CUDA and MPS enqueue work asynchronously, so
   an unsynchronized clock measures Python's enqueue rate, not compute. On CUDA, `cuda.Event`
   additionally gives device-side time.
3. **Repeats, reported as median with min/max/IQR.** One measurement cannot distinguish a
   regression from a busy neighbour process.
4. **Unmeasurable fields are `null` — never 0, never estimated.** `max_memory_allocated` is
   CUDA-only, so on CPU and MPS the memory columns read `n/a`, and the generated table carries that
   note with it.

Every record also stores the **commit** and the full **environment** (platform, CPU, torch version,
GPU name and architecture where applicable), so a number is always traceable to what produced it.
`scripts/check_env.py` prints the same probe the harness records, so an environment report and a
benchmark record cannot disagree about the machine.

## Running it

```bash
# One configuration. Appends a JSON record to results/benchmarks.jsonl and rewrites the Markdown
# table from the whole file, so comparisons accumulate.
.venv/bin/python -m minimegatron.benchmarks.run --config configs/benchmark.yaml --label fp32-cpu

# The comparisons SCOPE.md section 9 asks for are the same command with overrides:
.venv/bin/python -m minimegatron.benchmarks.run --config configs/benchmark.yaml \
    --label bf16 --set train.precision=bf16

GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29540 \
    -m minimegatron.benchmarks.run --config configs/ddp_2gpu.yaml --label ddp2

# Kernel level (needs Triton + an NVIDIA GPU; exits 2 with an explanation otherwise).
.venv/bin/python -m minimegatron.benchmarks.kernels

# Compare against a stored baseline. A WARNING, not a test (D27).
.venv/bin/python -m minimegatron.benchmarks.regression \
    --baseline results/baselines/tiny_cpu_fp32.json --current results/benchmarks.jsonl
```

`--no-store` prints the record instead of writing files. `benchmark.warmup_steps`,
`benchmark.measure_steps` and `benchmark.repeats` are config, so a quick run is
`--set benchmark.repeats=1`.

Baselines in [`results/baselines/`](../results/baselines/) are **versioned**; everything else under
`results/` is gitignored ad-hoc output. Baselines are machine-specific and carry their own
environment block, so a cross-hardware comparison is visible as such rather than silently
misleading. Re-record one deliberately, with a commit message saying why — never to silence a
warning you have not explained.

## Measured: GPU

**Kaggle `T4 x2`**, 2026-11-09 — 2x Tesla T4 (sm_75, 320 GB/s peak each), torch 2.11.0+cu130,
triton 3.6.0, source `010ee80`. Produced by `bash scripts/gpu_verify.sh`.

| Label | Strategy | Precision | GPUs | tokens/s | peak allocated |
|---|---|---|---|---|---|
| gpu-fp32 | single | fp32 | 1 | 29,648 | 1,862 MiB |
| gpu-bf16 | single | bf16 | 1 | 23,438 | 1,660 MiB |
| gpu-ddp-2 | ddp | fp32 | 2 | **52,493** | 1,906 MiB |
| gpu-fsdp-2 | fsdp | fp32 | 2 | 51,925 | 1,834 MiB |

**1 to 2 GPUs is 1.77x.** FSDP costs 1.1% throughput and saves 72 MiB against DDP — the right
direction, but the tiny preset is far too small for sharding to pay. **bf16 is slower than fp32**
on this hardware (see the note below). Tensor parallelism is missing because it failed on this run
(ISSUES #19), now fixed and awaiting a re-measurement.

Triton RMSNorm against the PyTorch reference, float32 — the full table is in
[PROGRESS.md](PROGRESS.md#measured-results):

| rows x cols | direction | speedup | GB/s |
|---|---|---|---|
| 4096 x 768 | fwd | **3.66x** | 228 |
| 65536 x 768 | fwd | **3.42x** | 236 |
| 4096 x 1024 | fwd+bwd | **2.74x** | 42 |
| 4096 x 192 | fwd+bwd | 1.07x | 13 |

~235 GB/s is about **73% of the T4's theoretical peak**, the right target for a memory-bound op.
The narrow 192-column case barely improves: launch overhead dominates and `dw` stays in PyTorch by
design (D31).

## Measured: CPU

tiny preset, Apple M4 Pro **CPU**, commit `7b8313e`, 3 warmup steps discarded, 3 repeats of 10
steps, median reported.

| Label | Strategy | Precision | Ranks | Batch | tokens/s (median) | IQR | ms/step | peak mem |
|---|---|---|---|---|---|---|---|---|
| tiny-cpu-fp32 | single | fp32 | 1 | 8 | **18,166.7** | 389.8 | 112.73 | n/a |
| tiny-cpu-bf16 | single | bf16 | 1 | 8 | **151.4** | 0.7 | 13,523.86 | n/a |
| tiny-cpu-ddp2 | ddp | fp32 | 2 | 16 | **9,811.2** | 26.0 | 417.48 | n/a |
| tiny-cpu-fsdp2 | fsdp | fp32 | 2 | 16 | **8,673.3** | 448.9 | 472.25 | n/a |
| tiny-cpu-tp2 | tp | fp32 | 2 | 16 | **5,175.2** | 1,030.6 | 791.47 | n/a |

**Read this as relative overhead, not as scaling.** Two ranks on one CPU contend for the same cores
*and* add communication, so a 2-rank row being slower than the 1-rank row says nothing about what
two GPUs would do.

What it does show is the **ordering** — single > DDP > FSDP > TP — which matches the prediction
recorded in each milestone's design doc *before* anything was measured:

- FSDP adds parameter all-gathers and a reduce-scatter on top of DDP's single all-reduce, moving
  roughly `3P` per step against DDP's `2P`.
- TP all-reduces twice per block per direction, so its communication scales with depth rather than
  being one operation per step.

The bf16 row is a real measurement of **emulated** bf16 on Apple silicon (~120× slower) and implies
nothing about bf16 on a GPU, where it is a speedup. It is included because deleting an inconvenient
measurement is worse than labelling it.

### Sharding, measured without a GPU

Peak-memory *bytes* need CUDA, but the per-rank **parameter and optimizer-state counts** are the
mechanism itself and are asserted by tests on CPU:

| Quantity (nano config, world_size 2) | Full model | Per rank, FSDP2 |
|---|---|---|
| parameters | 21,760 | **10,880** |
| optimizer state (Adam `m` + `v`) | 43,520 | **21,760** |

DDP would hold all 65,280 elements on every rank; FSDP2 holds 32,640. That is the 16-bytes-per-
parameter figure being divided by the world size.

Under TP, the tiny preset at `tp_size=2`:

| | Full model | Per rank |
|---|---|---|
| total | 11,487,360 | 10,599,936 |
| non-embedding | 1,779,840 | **892,416** (0.5014×) |

Non-embedding nearly halves — not exactly, because norms and row-parallel biases stay replicated.
Total barely moves, because the 9.7M-parameter embedding table (84% of this preset) is replicated in
the basic version. That is the argument for vocab-parallel embeddings.

## Not measured: everything requiring a GPU

No GPU figure appears anywhere in this repository. The harness, the configs and the baseline store
all exist, so each of these is a command rather than new work — and they are bundled into
`scripts/gpu_verify.sh`, with [`scripts/colab_gpu_verify.ipynb`](../scripts/colab_gpu_verify.ipynb)
wrapping it for Colab:

| Comparison (SCOPE.md §9) | Command | Needs |
|---|---|---|
| 1 vs 2 GPU throughput | `benchmarks.run` with `configs/benchmark.yaml`, then under `torchrun` with `ddp_2gpu.yaml` | 2 NVIDIA GPUs |
| DDP vs FSDP vs TP | `benchmarks.run` with each `*_2gpu.yaml` | 2 NVIDIA GPUs, ideally ± NVLink |
| Peak allocated / reserved memory per rank | any of the above (the fields populate automatically on CUDA) | 2 NVIDIA GPUs |
| FP32 vs BF16 | `--set train.precision=bf16` | **Ampere or newer** — a T4 is sm_75 and has no bf16 at all |
| PyTorch reference vs Triton kernel | `benchmarks.kernels` | 1 NVIDIA GPU + Triton |
| Sequence lengths and micro-batch sizes | `--set model.seq_len=... --set train.micro_batch_size=...` | any GPU |
| Profiler traces | not yet wired; `torch.profiler` is a dependency and the harness is the place for it | any GPU |

Note that **Colab cannot cover the two-GPU rows** — it provides one GPU per runtime. Those need a
2-GPU machine (Kaggle's free `T4 x2`, which is what produced the table above, or a rented
instance), where `gpu_verify.sh` runs its multi-GPU section automatically.

**On bf16 and the T4:** a T4 is sm_75 and has no *native* bf16, but torch counts emulated bf16, so
`torch.cuda.is_bf16_supported()` is True and `precision: bf16` is permitted. It runs, and it is
**slower than fp32** — 23,438 vs 29,648 tokens/s measured — while saving memory. The FP32-vs-BF16
*speedup* therefore still needs Ampere or newer. Earlier versions of these docs said a T4 would
error on bf16; that was wrong (ISSUES #21).

Two expectations recorded **in advance**, so that measuring them is not mistaken for finding a bug:

- **TP should lose to DDP on PCIe.** Two all-reduces per block per direction means communication
  dominates without NVLink-class bandwidth. This is why Megatron confines tensor parallelism to
  within a node and composes it with data parallelism across nodes.
- **The Triton kernel's first run will probably surface bugs.** — *Resolved 2026-11-09: it did
  not.* The kernel passed all 68 tests on its first execution. The run did surface three other
  bugs (ISSUES #19, #20, #21), all in code and tests *around* the kernel rather than in it.

## Reproducing a figure from this document

1. `python scripts/check_env.py` — if the hardware differs from the table's, the numbers will too.
2. `git checkout <commit>` from the table's Commit column.
3. Run the command above for that row.
4. `python -m minimegatron.benchmarks.regression --baseline results/baselines/<file>.json
   --current results/benchmarks.jsonl` to see the delta as a percentage per metric.

If the comparison is across different hardware, the environment blocks in both records will say so.
That is the intended way to notice it.
