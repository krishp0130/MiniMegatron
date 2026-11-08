"""Process-group lifecycle and the handful of collectives the trainer needs.

Nothing here reimplements a collective: `torch.distributed` owns the communication, and this
module owns the bookkeeping around it (who am I, which device is mine, tear down cleanly). The
educational hand-rolled all-reduce lives in `collectives_demo.py`, deliberately outside the
training path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from minimegatron.config import ConfigError
from minimegatron.distributed.config import DistributedConfig


@dataclass(frozen=True)
class DistInfo:
    """Who this process is. The defaults describe a single-process (Milestone 1) run."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_distributed: bool = False

    @property
    def is_main(self) -> bool:
        """Rank 0, which owns every side effect: logging, checkpoints, stdout."""
        return self.rank == 0


def resolve_backend(spec: str) -> str:
    """Resolve a backend name, erroring rather than silently substituting one that works."""
    if spec == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    if spec == "nccl" and not dist.is_nccl_available():
        raise ConfigError("backend 'nccl' requested but NCCL is not available in this build")
    if spec == "gloo" and not dist.is_gloo_available():
        raise ConfigError("backend 'gloo' requested but gloo is not available in this build")
    return spec


def init_distributed(cfg: DistributedConfig | None = None) -> DistInfo:
    """Join the process group `torchrun` launched us into, if there is one.

    When `RANK`/`WORLD_SIZE` are absent we are a plain single process: nothing is initialized and a
    default `DistInfo` comes back, so every Milestone 1 path -- and `train.py` itself -- works
    unchanged whether or not it was launched under torchrun.

    On macOS, gloo also needs `GLOO_SOCKET_IFNAME=lo0` and an explicit master address in the
    environment, or process-group init hangs with no error (ISSUES #1).
    """
    cfg = cfg or DistributedConfig()

    if dist.is_available() and dist.is_initialized():
        return _info_from_group()  # already joined, e.g. by a test harness
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo()

    backend = resolve_backend(cfg.backend)
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=cfg.timeout_seconds))
    info = _info_from_group()
    if backend == "nccl":
        # One process owns one GPU. Without this every rank would default to cuda:0.
        torch.cuda.set_device(info.local_rank)
    return info


def _info_from_group() -> DistInfo:
    rank = dist.get_rank()
    return DistInfo(
        rank=rank,
        local_rank=int(os.environ.get("LOCAL_RANK", rank)),
        world_size=dist.get_world_size(),
        is_distributed=True,
    )


def destroy_distributed() -> None:
    """Leave the process group, after waiting for everyone else to finish.

    Call this from a `finally`. A rank that exits while its peers are inside a collective leaves
    them blocked until the timeout, which turns one failure into a hung job.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    """Wait for every rank. A no-op in a single process."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks, in place. A no-op in a single process.

    Used for *metrics*: a loss computed on one rank describes that rank's shard, not the global
    batch, so a logged number only means what it says after this. Gradient averaging is DDP's job,
    not ours.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def resolve_strategy(spec: str, world_size: int) -> str:
    """Resolve the parallelism strategy, erroring on combinations that cannot work.

    `auto` keeps every existing default: one process trains single-replica, several use DDP. A
    strategy that needs peers is an error at world size 1 rather than a silent downgrade -- the
    same no-silent-fallback rule the device and precision checks follow, and for the same reason:
    a run that quietly trains differently from what the config says is worse than one that stops.
    """
    if spec == "auto":
        return "single" if world_size == 1 else "ddp"
    if spec in ("ddp", "fsdp") and world_size == 1:
        raise ConfigError(
            f"strategy {spec!r} needs more than one rank, but world_size is 1; "
            "launch with torchrun --nproc_per_node=N, or use strategy 'single'"
        )
    return spec


def build_device_mesh(device_type: str, world_size: int):
    """A 1-D device mesh over every rank, built with an **explicit** device type.

    FSDP2 and DTensor need a mesh. Letting them build a default one means auto-detecting the
    accelerator, which finds MPS on this machine and crashes inside
    `device_mesh._setup_world_group_and_device` (ISSUES #2). Passing the type we actually resolved
    is both correct and one fewer piece of magic.
    """
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(device_type, (world_size,))
