"""Process-group lifecycle."""

import pytest
import torch
import torch.distributed as dist

from minimegatron.config import ConfigError
from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import (
    DistInfo,
    all_reduce_mean,
    init_distributed,
    resolve_backend,
)
from tests.distributed.conftest import run_distributed

pytestmark = pytest.mark.distributed


def _identity_and_collectives(rank, world_size):
    info = init_distributed(DistributedConfig(backend="gloo"))
    value = all_reduce_mean(torch.tensor([float(rank)]))
    return {
        "rank": info.rank,
        "local_rank": info.local_rank,
        "world_size": info.world_size,
        "is_distributed": info.is_distributed,
        "is_main": info.is_main,
        "initialized": dist.is_initialized(),
        "mean": value.item(),
    }


def test_each_rank_knows_who_it_is():
    results = run_distributed(_identity_and_collectives, world_size=2)
    assert sorted(results) == [0, 1]
    for rank, result in results.items():
        assert result["rank"] == rank
        assert result["local_rank"] == rank
        assert result["world_size"] == 2
        assert result["is_distributed"] is True
        assert result["initialized"] is True
    assert results[0]["is_main"] is True
    assert results[1]["is_main"] is False


def test_all_reduce_mean_averages_across_ranks():
    results = run_distributed(_identity_and_collectives, world_size=2)
    for result in results.values():
        assert result["mean"] == pytest.approx(0.5)  # mean of rank 0 and rank 1


# --- single-process behaviour (no processes needed) ---------------------------------------------


def test_without_torchrun_env_nothing_is_initialized(monkeypatch):
    """Milestone 1's path must keep working: one entry point, launched either way."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = init_distributed(DistributedConfig(backend="gloo"))
    assert info == DistInfo(rank=0, local_rank=0, world_size=1, is_distributed=False)
    assert info.is_main
    assert not dist.is_initialized()


def test_collectives_are_no_ops_in_a_single_process():
    value = torch.tensor([3.0])
    assert all_reduce_mean(value).item() == 3.0


def test_backend_auto_resolves_to_gloo_without_cuda():
    expected = "nccl" if torch.cuda.is_available() else "gloo"
    assert resolve_backend("auto") == expected


@pytest.mark.skipif(dist.is_nccl_available(), reason="NCCL is available in this build")
def test_requesting_nccl_without_nccl_is_a_hard_error():
    with pytest.raises(ConfigError, match="NCCL is not available"):
        resolve_backend("nccl")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"backend": "mpi"}, "backend must be one of"),
        ({"timeout_seconds": 0}, "timeout_seconds must be a positive integer"),
    ],
)
def test_invalid_distributed_config_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        DistributedConfig(**overrides)
