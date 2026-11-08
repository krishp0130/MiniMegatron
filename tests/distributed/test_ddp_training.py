"""A short two-rank run must track the equivalent single-process run."""

import pytest

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import (
    nano_experiment,
    run_distributed,
    single_process_reference,
)

pytestmark = pytest.mark.distributed

STEPS = 5


def _ddp_losses(rank, world_size, out_dir):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, max_steps=STEPS, micro_batch_size=4, warmup_steps=2)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    records = [trainer.train_step() for _ in range(STEPS)]
    return {
        "losses": [r["loss"] for r in records],
        "lrs": [r["lr"] for r in records],
        "weights": trainer.raw_model.tok_emb.weight.detach().clone(),
    }


def test_two_rank_run_tracks_the_single_process_run(tmp_path):
    """Several optimizer steps, not just one: a synchronization bug that only shows up after the
    replicas have had a chance to drift would be invisible in a single-step test."""
    results = run_distributed(_ddp_losses, 2, args=(str(tmp_path / "ddp"),))
    reference, _, ref_trainer = single_process_reference(
        str(tmp_path / "ref"), steps=STEPS, micro_batch_size=8, max_steps=STEPS, warmup_steps=2
    )

    for step, (got, want) in enumerate(zip(results[0]["losses"], [r["loss"] for r in reference])):
        assert got == pytest.approx(want, rel=1e-4), f"diverged at step {step}"
    assert results[0]["lrs"] == pytest.approx([r["lr"] for r in reference])


def test_replicas_stay_identical_across_steps(tmp_path):
    """Identical gradients and identical optimizer state means identical weights, forever."""
    results = run_distributed(_ddp_losses, 2, args=(str(tmp_path / "ddp"),))
    assert results[0]["weights"].equal(results[1]["weights"])


def test_both_ranks_report_the_same_loss(tmp_path):
    results = run_distributed(_ddp_losses, 2, args=(str(tmp_path / "ddp"),))
    assert results[0]["losses"] == pytest.approx(results[1]["losses"])


def _eval_losses(rank, world_size, out_dir):
    """The aggregated validation loss, next to this rank's unaggregated one."""
    import torch

    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, eval_steps=2)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)

    trainer.raw_model.eval()
    with torch.no_grad():
        total = sum(
            trainer.raw_model(*(t.to(trainer.device) for t in trainer.val_stream.micro_batch(j)))[1]
            for j in range(cfg.train.eval_steps)
        )
    local = (total / cfg.train.eval_steps).item()
    trainer.raw_model.train()

    return {"aggregated": trainer.evaluate(), "local": local}


def test_validation_loss_is_aggregated_across_ranks(tmp_path):
    """Each rank evaluates its own shard, so an unaggregated number describes only that shard."""
    results = run_distributed(_eval_losses, 2, args=(str(tmp_path / "ddp"),))

    # The shards genuinely differ, so this test has teeth.
    assert results[0]["local"] != pytest.approx(results[1]["local"])
    # ...and the reported value is the mean over the union, identical on every rank.
    expected = (results[0]["local"] + results[1]["local"]) / 2
    for rank in (0, 1):
        assert results[rank]["aggregated"] == pytest.approx(expected, rel=1e-6)
