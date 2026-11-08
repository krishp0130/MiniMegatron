"""Rank-0-only side effects, and checkpoints that outlive the strategy that wrote them."""

from pathlib import Path

import pytest

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import GPT
from minimegatron.training.checkpoint import load_checkpoint
from tests.distributed.conftest import NANO_MODEL, nano_experiment, run_distributed

pytestmark = pytest.mark.distributed


def _train_and_save(rank, world_size, out_dir):
    from minimegatron.training.trainer import Trainer

    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, max_steps=3, micro_batch_size=4, log_interval=1)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    trainer.fit()
    return {
        "step": trainer.step,
        "is_main": info.is_main,
        "logger": type(trainer.logger).__name__,
        "files": sorted(p.name for p in Path(out_dir).iterdir()),
    }


def test_only_rank_zero_writes(tmp_path):
    out_dir = tmp_path / "run"
    results = run_distributed(_train_and_save, 2, args=(str(out_dir),))

    assert results[0]["logger"] == "JSONLLogger"
    assert results[1]["logger"] == "NullLogger"
    # One log and one checkpoint, not one per rank.
    assert sorted(p.name for p in out_dir.iterdir()) == ["ckpt_000003.pt", "log.jsonl"]
    assert len((out_dir / "log.jsonl").read_text().splitlines()) == 3


def test_both_ranks_see_the_checkpoint_after_the_barrier(tmp_path):
    """Rank 1 must not race past a checkpoint that is not on disk yet."""
    out_dir = tmp_path / "run"
    results = run_distributed(_train_and_save, 2, args=(str(out_dir),))
    for rank in (0, 1):
        assert "ckpt_000003.pt" in results[rank]["files"]
        assert results[rank]["step"] == 3


def test_a_ddp_checkpoint_loads_into_a_single_device_model(tmp_path):
    """Exit criterion 5: the saved keys carry no `module.` prefix, so a Milestone 1 model loads
    it directly. Checkpoints should outlive the parallelism strategy that produced them."""
    out_dir = tmp_path / "run"
    run_distributed(_train_and_save, 2, args=(str(out_dir),))

    plain = GPT(ModelConfig(dropout=0.0, **NANO_MODEL))
    meta = load_checkpoint(out_dir / "ckpt_000003.pt", model=plain)
    assert meta["step"] == 3
    assert meta["tokens_seen"] == 3 * 4 * 2 * 16  # steps x micro x ranks x seq_len
