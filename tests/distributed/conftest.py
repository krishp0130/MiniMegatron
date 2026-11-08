"""Harness for running a function in several processes under gloo.

Two rules this encodes, both learned the hard way (ISSUES #1):

* gloo on macOS must be pinned to loopback and given an explicit master address, or
  `init_process_group` hangs forever with no error.
* every distributed subprocess needs a **hard timeout enforced in Python** -- macOS has no
  `timeout` binary, and a hung collective otherwise hangs the whole test session.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
import torch.multiprocessing as mp

from minimegatron.distributed.setup import destroy_distributed

# gloo picks an unreachable interface from the hostname unless told otherwise.
LOOPBACK_IFNAME = "lo0" if sys.platform == "darwin" else "lo"
DEFAULT_TIMEOUT = 180


def free_port() -> int:
    """Ask the OS for an unused port, so parallel test runs don't collide on a fixed one."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank, world_size, port, fn, args, results_dir):
    """Set up the environment `torchrun` would have provided, then run `fn`.

    Results go to a file rather than a `multiprocessing.Queue`. Torch serializes tensors through
    shared memory, which requires the *sending* process to stay alive until the parent has mapped
    the storage -- a worker that puts its result and exits immediately deadlocks the parent's
    `get()` (ISSUES #7). `torch.save` copies the bytes to disk, so the handoff survives the
    sender's exit.
    """
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        GLOO_SOCKET_IFNAME=LOOPBACK_IFNAME,
    )
    target = Path(results_dir) / f"rank{rank}.pt"
    try:
        payload = {"ok": True, "result": fn(rank, world_size, *args)}
    except BaseException:  # noqa: BLE001 - reported by the parent with this traceback
        payload = {"ok": False, "error": traceback.format_exc()}
    finally:
        destroy_distributed()
    torch.save(payload, target)


def run_distributed(
    fn: Callable[..., Any],
    world_size: int = 2,
    args: tuple = (),
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[int, Any]:
    """Run `fn(rank, world_size, *args)` in `world_size` spawned processes.

    `fn` must be importable by name (a module-level function), because spawn re-imports it in the
    child. Returns `{rank: result}`; a failure in any rank fails the test with that rank's
    traceback, and a rank that outlives `timeout` is terminated rather than left to hang the
    session.
    """
    ctx = mp.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as results_dir:
        procs = [
            ctx.Process(target=_worker, args=(rank, world_size, port, fn, args, results_dir))
            for rank in range(world_size)
        ]
        for proc in procs:
            proc.start()

        failures = []
        deadline = time.monotonic() + timeout
        for rank, proc in enumerate(procs):
            proc.join(timeout=max(1.0, deadline - time.monotonic()))
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
                failures.append(f"rank {rank} timed out after {timeout}s and was terminated")

        results = {}
        for rank, proc in enumerate(procs):
            target = Path(results_dir) / f"rank{rank}.pt"
            if not target.is_file():
                failures.append(
                    f"rank {rank} wrote no result (exit code {proc.exitcode}); it died early"
                )
                continue
            payload = torch.load(target, map_location="cpu", weights_only=False)
            if payload["ok"]:
                results[rank] = payload["result"]
            else:
                failures.append(f"rank {rank} failed:\n{payload['error']}")

    if failures:
        pytest.fail("\n\n".join(failures))
    exit_codes = [proc.exitcode for proc in procs]
    assert all(code == 0 for code in exit_codes), f"non-zero exit codes: {exit_codes}"
    return results


# --- shared fixtures for the DDP tests ----------------------------------------------------------

NANO_MODEL = dict(vocab_size=128, seq_len=16, n_layers=2, n_heads=4, d_model=32, d_ff=64)


def nano_experiment(out_dir: str, **train_overrides):
    """A tiny experiment config, identical in the parent and in every spawned rank.

    `dropout=0` throughout: identically seeded ranks would draw identical masks (wrong for data
    parallelism), and differently seeded ranks could never match a single-process run. Correct
    per-rank dropout needs a model-parallel RNG tracker, which is documented and deferred.
    `grad_clip=0` keeps the comparison on raw gradients.
    """
    from minimegatron.data.config import DataConfig
    from minimegatron.experiment import ExperimentConfig
    from minimegatron.model.config import ModelConfig
    from minimegatron.training.config import TrainConfig

    train = dict(
        max_steps=8,
        micro_batch_size=4,
        grad_accum_steps=1,
        lr=1e-3,
        warmup_steps=0,
        grad_clip=0.0,
        device="cpu",
        seed=1337,
        out_dir=out_dir,
        log_interval=0,
        eval_interval=0,
    )
    train.update(train_overrides)
    return ExperimentConfig(
        model=ModelConfig(dropout=0.0, **NANO_MODEL),
        data=DataConfig(dataset="memorization", n_items=256, n_sequences=16, seed=0),
        train=TrainConfig(**train),
    )


def single_process_reference(out_dir: str, steps: int = 1, **train_overrides):
    """The Milestone 1 trainer, run in this process: the thing DDP must reproduce."""
    from minimegatron.training.trainer import Trainer

    trainer = Trainer(nano_experiment(out_dir, **train_overrides), out_dir=out_dir, echo=False)
    records = [trainer.train_step() for _ in range(steps)]
    grads = {name: p.grad.clone() for name, p in trainer.raw_model.named_parameters()}
    return records, grads, trainer
