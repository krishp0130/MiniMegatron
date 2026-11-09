#!/usr/bin/env bash
# Run every check this project cannot run without an NVIDIA GPU, and print a summary to paste
# back into docs/PROGRESS.md.
#
#   bash scripts/gpu_verify.sh
#
# Works on any Linux + NVIDIA machine: Colab, Kaggle, or a rented instance. Refuses clearly
# without a GPU rather than producing numbers from the wrong device.
#
# WHAT THIS CLOSES
#   ISSUES #16   the Triton RMSNorm kernel has never been executed
#   DoD #3       single-GPU training works on CUDA
#   DoD #7       an optimized Triton kernel passes numerical and gradient tests
#   M5           kernel runtime and bandwidth vs the PyTorch reference
#   M3 (part)    FP32 vs BF16 -- only on Ampere or newer; a T4 is sm_75 and has NO bf16
#
# WHAT IT CANNOT CLOSE
#   Anything needing TWO GPUs: 1-vs-2-GPU throughput, DDP vs FSDP vs TP, peak memory per rank
#   (DoD #4, #5). Google Colab provides ONE GPU per runtime, so those need a 2-GPU machine --
#   Kaggle's free `T4 x2`, or a rented instance. Re-run this script there and it will also run
#   the multi-GPU section below.

set -u
PY=${PY:-python}
FAIL=0

hr() { printf '\n== %s ==\n' "$1"; }

hr "environment"
if ! $PY -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "No CUDA device visible. This script needs an NVIDIA GPU (ISSUES #3)."
    echo "On Colab: Runtime -> Change runtime type -> GPU."
    exit 2
fi
$PY scripts/check_env.py
GPUS=$($PY -c 'import torch; print(torch.cuda.device_count())')
BF16=$($PY -c 'import torch; print(torch.cuda.is_bf16_supported())')
echo "gpus=$GPUS bf16=$BF16"

hr "1. Triton kernel correctness  (ISSUES #16, DoD #7)"
# The kernel tests are parameterized over backend, so they judge the kernel by the PyTorch
# reference's own assertions. EXPECT FAILURES on the first ever run: this code has never executed.
$PY -m pytest tests/kernels -q || { echo ">>> kernel tests FAILED (expected on a first run; this is the point)"; FAIL=1; }

hr "2. Full suite on GPU"
$PY -m pytest -q || FAIL=1

hr "3. Single-GPU training  (DoD #3)"
$PY -m minimegatron.train --config configs/tiny.yaml \
    --set train.max_steps=20 --set train.device=cuda --out-dir runs/gpu_smoke || FAIL=1

hr "4. Kernel benchmark: reference vs Triton  (M5)"
$PY -m minimegatron.benchmarks.kernels --out results/kernels_gpu.jsonl || FAIL=1
echo "--- backward pass ---"
$PY -m minimegatron.benchmarks.kernels --backward --out results/kernels_gpu.jsonl || FAIL=1

hr "5. End-to-end throughput: FP32  (M2 baseline)"
$PY -m minimegatron.benchmarks.run --config configs/benchmark.yaml \
    --label gpu-fp32 --set train.device=cuda || FAIL=1

hr "6. FP32 vs BF16  (M3)"
if [ "$BF16" = "True" ]; then
    $PY -m minimegatron.benchmarks.run --config configs/benchmark.yaml \
        --label gpu-bf16 --set train.device=cuda --set train.precision=bf16 || FAIL=1
else
    echo "SKIPPED: this GPU has no bf16 (needs compute capability 8.0+, i.e. Ampere or newer)."
    echo "A T4 is sm_75. The project errors clearly on bf16 here rather than silently changing"
    echo "numerics -- which is itself worth confirming:"
    $PY -m minimegatron.train --config configs/tiny.yaml --set train.max_steps=1 \
        --set train.device=cuda --set train.precision=bf16 2>&1 | tail -2
fi

if [ "$GPUS" -ge 2 ]; then
    hr "7. Multi-GPU  (DoD #4, #5; M2/M3/M4 benchmarks)"
    TR="$PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2"
    for STRATEGY in ddp fsdp tp; do
        echo "--- $STRATEGY ---"
        $TR -m minimegatron.benchmarks.run --config "configs/${STRATEGY}_2gpu.yaml" \
            --label "gpu-$STRATEGY-2" --set train.device=cuda || FAIL=1
    done
else
    hr "7. Multi-GPU  (DoD #4, #5)"
    echo "SKIPPED: only $GPUS GPU visible. Colab provides one GPU per runtime."
    echo "For the 1-vs-2-GPU, DDP-vs-FSDP-vs-TP and peak-memory comparisons, run this script on a"
    echo "2-GPU machine (Kaggle's free 'T4 x2', or a rented instance) and section 7 will run."
fi

hr "summary"
echo "Results appended to results/*.jsonl; the Markdown table is in results/benchmarks.md."
echo "Paste the tables into docs/PROGRESS.md under 'Measured results', with the GPU name and"
echo "commit, and update the ISSUES #16 / Definition-of-Done rows."
echo
[ "$FAIL" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED -- see above (expected if the kernel has never run)"
exit $FAIL
