"""The educational hand-rolled all-reduce must agree with the real thing.

This is what keeps `collectives_demo.py` honest: a demo nobody tests quietly stops describing
what the framework actually does.
"""

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from minimegatron.data.synthetic import BatchStream, RandomTokenDataset
from minimegatron.distributed.collectives_demo import (
    DEMO_CONFIG,
    local_gradients,
    manual_average_gradients,
)
from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.model.transformer import GPT
from tests.distributed.conftest import run_distributed

pytestmark = pytest.mark.distributed

PROBE = "blocks.0.mlp.fc1.weight"


def _manual_vs_ddp(rank, world_size):
    info = init_distributed(DistributedConfig(backend="gloo"))

    torch.manual_seed(0)
    by_hand = GPT(DEMO_CONFIG)
    torch.manual_seed(0)
    by_ddp = DistributedDataParallel(GPT(DEMO_CONFIG))

    dataset = RandomTokenDataset(DEMO_CONFIG.vocab_size, DEMO_CONFIG.seq_len, 256, seed=0)
    stream = BatchStream(dataset, 4, seed=0, rank=info.rank, world_size=info.world_size)
    x, y = stream.micro_batch(0)

    local_gradients(by_hand, x, y)
    before = dict(by_hand.named_parameters())[PROBE].grad.clone()
    manual_average_gradients(by_hand, info.world_size)

    local_gradients(by_ddp, x, y)  # DDP synchronizes inside backward()

    return {
        "before_average": before,
        "manual": {name: p.grad.clone() for name, p in by_hand.named_parameters()},
        "ddp": {name: p.grad.clone() for name, p in by_ddp.module.named_parameters()},
    }


def test_manual_all_reduce_reproduces_ddp_gradients(tmp_path):
    """Exit criterion 4. Same arithmetic; DDP just overlaps it with the backward pass."""
    results = run_distributed(_manual_vs_ddp, 2)
    for rank in (0, 1):
        for name, manual in results[rank]["manual"].items():
            torch.testing.assert_close(
                manual, results[rank]["ddp"][name], msg=f"{name} differs on rank {rank}"
            )


def test_ranks_disagree_before_the_all_reduce_and_agree_after(tmp_path):
    """The whole point, in one assertion: different data in, one shared gradient out."""
    results = run_distributed(_manual_vs_ddp, 2)
    assert not torch.allclose(results[0]["before_average"], results[1]["before_average"])
    torch.testing.assert_close(results[0]["manual"][PROBE], results[1]["manual"][PROBE])
