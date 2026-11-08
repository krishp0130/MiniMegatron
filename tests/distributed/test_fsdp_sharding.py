"""Where FSDP's memory saving actually comes from, measured rather than assumed.

Peak-memory *bytes* need CUDA (`max_memory_allocated`), so they are pending hardware. The
per-rank parameter and optimizer-state *counts* are the mechanism itself, and they are measurable
right here on CPU.
"""

import pytest
import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.fsdp import (
    is_sharded,
    local_optimizer_state_count,
    local_parameter_count,
)
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import nano_experiment, run_distributed

pytestmark = pytest.mark.distributed

NANO_PARAMS = 21_760  # the nano config's full parameter count (asserted in tests/unit/test_config)


def _sharding_facts(rank, world_size, out_dir):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, strategy="fsdp", micro_batch_size=4)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    trainer.train_step()  # populates Adam's m and v

    model = trainer.raw_model
    return {
        "is_sharded": is_sharded(model),
        "local_params": local_parameter_count(model),
        "local_optim_state": local_optimizer_state_count(trainer.optimizer),
        "blocks_are_units": [isinstance(b, FSDPModule) for b in model.blocks],
        "root_is_unit": isinstance(trainer.model, FSDPModule),
        "tied": model.lm_head.weight is model.tok_emb.weight,
        "all_dtensors": all(isinstance(p, DTensor) for p in model.parameters()),
        "tied_grad": model.tok_emb.weight.grad.full_tensor().clone(),
    }


def test_parameters_are_sharded_across_ranks(tmp_path):
    """Exit criterion 4, first half: each rank stores ~1/world_size of the parameters."""
    results = run_distributed(_sharding_facts, 2, args=(str(tmp_path / "fsdp"),))
    for rank in (0, 1):
        assert results[rank]["is_sharded"]
        assert results[rank]["all_dtensors"]
        # Exactly half here; in general within one shard's rounding of P/R.
        assert results[rank]["local_params"] == NANO_PARAMS // 2
    total = sum(results[r]["local_params"] for r in (0, 1))
    assert total == NANO_PARAMS  # the shards partition the model, with nothing duplicated


def test_optimizer_state_is_sharded_too(tmp_path):
    """Exit criterion 4, second half -- and the half that fails quietly.

    Adam's m and v are 8 of the 16 bytes per parameter. An optimizer built before `fully_shard`
    would hold full-size state and still train perfectly well, just without the saving.
    """
    results = run_distributed(_sharding_facts, 2, args=(str(tmp_path / "fsdp"),))
    for rank in (0, 1):
        # m and v, each the size of this rank's parameter shard.
        assert results[rank]["local_optim_state"] == 2 * (NANO_PARAMS // 2)


def test_every_block_is_its_own_sharding_unit(tmp_path):
    """Per-block units are the point: FSDP gathers one block at a time and frees it after, so
    peak transient memory is one block rather than the whole model."""
    results = run_distributed(_sharding_facts, 2, args=(str(tmp_path / "fsdp"),))
    assert results[0]["blocks_are_units"] == [True, True]
    assert results[0]["root_is_unit"]


def test_the_tied_embedding_survives_sharding(tmp_path):
    """D8: the tied weight must stay in the root unit. A shared parameter split across two units
    would be gathered and freed on two schedules -- the kind of bug that produces subtly wrong
    gradients rather than a crash."""
    results = run_distributed(_sharding_facts, 2, args=(str(tmp_path / "fsdp"),))
    for rank in (0, 1):
        assert results[rank]["tied"], "tok_emb and lm_head stopped sharing storage"
        grad = results[rank]["tied_grad"]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_fsdp_holds_less_per_rank_than_ddp(tmp_path):
    """The comparison the milestone exists to make, in the one currency measurable on CPU."""
    results = run_distributed(_sharding_facts, 2, args=(str(tmp_path / "fsdp"),))
    fsdp_per_rank = results[0]["local_params"] + results[0]["local_optim_state"]
    ddp_per_rank = NANO_PARAMS + 2 * NANO_PARAMS  # DDP replicates params and all of Adam's state
    assert fsdp_per_rank * 2 == ddp_per_rank  # exactly halved at world_size=2
