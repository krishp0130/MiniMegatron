"""Distributed training: process groups, DDP, and an educational collectives demo."""

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import (
    DistInfo,
    all_reduce_mean,
    barrier,
    build_device_mesh,
    destroy_distributed,
    init_distributed,
    resolve_backend,
    resolve_strategy,
)

__all__ = [
    "DistInfo",
    "DistributedConfig",
    "all_reduce_mean",
    "barrier",
    "build_device_mesh",
    "destroy_distributed",
    "init_distributed",
    "resolve_backend",
    "resolve_strategy",
]
