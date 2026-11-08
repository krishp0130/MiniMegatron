"""Sharded (DCP) and consolidated checkpoints for FSDP runs."""

from pathlib import Path

import pytest
import torch
from torch.distributed.tensor import DTensor

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import GPT
from minimegatron.training.checkpoint import load_checkpoint
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import NANO_MODEL, nano_experiment, run_distributed

pytestmark = pytest.mark.distributed

N = 3


def _full(t):
    return t.full_tensor() if isinstance(t, DTensor) else t


def _trainer(out_dir, rank_info, **overrides):
    cfg = nano_experiment(out_dir, strategy="fsdp", micro_batch_size=4, **overrides)
    return Trainer(cfg, dist_info=rank_info, out_dir=out_dir, echo=False)


def _train_and_save(rank, world_size, out_dir):
    info = init_distributed(DistributedConfig(backend="gloo"))
    trainer = _trainer(out_dir, info, max_steps=N)
    trainer.fit()
    return {
        "step": trainer.step,
        "files": sorted(p.name for p in Path(out_dir).iterdir()),
        "shard_files": sorted(p.name for p in (Path(out_dir) / f"ckpt_{N:06d}").iterdir()),
        "weights": _full(trainer.raw_model.tok_emb.weight.detach()).clone(),
    }


def test_sharded_checkpoint_has_one_shard_per_rank(tmp_path):
    out_dir = tmp_path / "run"
    results = run_distributed(_train_and_save, 2, args=(str(out_dir),))

    files = results[0]["shard_files"]
    assert ".metadata" in files
    assert "__0_0.distcp" in files and "__1_0.distcp" in files  # nothing was gathered
    assert "meta.pt" in files  # step, tokens, scheduler, resolved config
    # RNG per rank, not just rank 0: random state differs by rank, so one rank's streams
    # restored everywhere would make a resumed run diverge.
    assert "rng_0.pt" in files and "rng_1.pt" in files


def test_both_checkpoint_formats_are_written(tmp_path):
    out_dir = tmp_path / "run"
    results = run_distributed(_train_and_save, 2, args=(str(out_dir),))
    assert results[0]["files"] == [f"ckpt_{N:06d}", f"ckpt_{N:06d}_full.pt", "log.jsonl"]


def test_consolidated_checkpoint_loads_into_a_single_device_model(tmp_path):
    """Exit criterion 6, second half: a sharded run's result is loadable anywhere.

    This is what makes an FSDP run comparable to the Milestone 1 reference at all, and it works
    because `state_dict` lists both `tok_emb.weight` and `lm_head.weight` -- exactly as a plain
    `GPT.state_dict()` does -- so the keys line up with no munging.
    """
    out_dir = tmp_path / "run"
    results = run_distributed(_train_and_save, 2, args=(str(out_dir),))

    plain = GPT(ModelConfig(dropout=0.0, **NANO_MODEL))
    meta = load_checkpoint(out_dir / f"ckpt_{N:06d}_full.pt", model=plain)

    assert meta["step"] == N
    assert meta["tokens_seen"] == N * 4 * 2 * 16  # steps x micro x ranks x seq_len
    torch.testing.assert_close(plain.tok_emb.weight.detach(), results[0]["weights"])
    assert plain.lm_head.weight is plain.tok_emb.weight  # tying intact after the round trip


def _resume_from_sharded(rank, world_size, out_dir):
    """Train 2N steps straight, versus N + sharded checkpoint + fresh trainer + resume + N."""
    info = init_distributed(DistributedConfig(backend="gloo"))

    straight = _trainer(f"{out_dir}/a", info, max_steps=2 * N)
    straight_losses = [straight.train_step()["loss"] for _ in range(2 * N)]
    straight_weights = _full(straight.raw_model.tok_emb.weight.detach()).clone()

    first = _trainer(f"{out_dir}/b", info, max_steps=2 * N)
    resumed_losses = [first.train_step()["loss"] for _ in range(N)]
    ckpt = first.save()

    second = _trainer(f"{out_dir}/c", info, max_steps=2 * N)
    before = _full(second.raw_model.tok_emb.weight.detach()).clone()
    meta = second.resume(ckpt)
    resumed_losses += [second.train_step()["loss"] for _ in range(N)]

    return {
        "straight": straight_losses,
        "resumed": resumed_losses,
        "step_after_resume": meta["step"],
        "differed_before_resume": not torch.allclose(before, straight_weights),
        "straight_weights": straight_weights,
        "resumed_weights": _full(second.raw_model.tok_emb.weight.detach()).clone(),
    }


def test_resuming_from_a_sharded_checkpoint_continues_the_run(tmp_path):
    """Exit criterion 6, first half. Passes only if weights, Adam's moments, the schedule, the
    data position and the RNG are all restored on every rank."""
    results = run_distributed(_resume_from_sharded, 2, args=(str(tmp_path),))
    r = results[0]

    assert r["step_after_resume"] == N
    assert r["differed_before_resume"], "the fresh trainer already matched; the test proves nothing"
    assert r["resumed"] == pytest.approx(r["straight"], rel=1e-6)
    torch.testing.assert_close(r["resumed_weights"], r["straight_weights"], rtol=1e-5, atol=1e-7)


def test_loading_a_directory_without_metadata_is_rejected(tmp_path):
    empty = tmp_path / "not-a-checkpoint"
    empty.mkdir()
    from minimegatron.distributed.checkpoint import load_sharded

    with pytest.raises(FileNotFoundError, match="not a sharded checkpoint"):
        load_sharded(empty, model=GPT(ModelConfig(dropout=0.0, **NANO_MODEL)))
