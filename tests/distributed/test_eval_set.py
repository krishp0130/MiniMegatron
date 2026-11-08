"""Validation loss must mean the same thing at any world size (ISSUES #8)."""

import pytest

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import nano_experiment, run_distributed

pytestmark = pytest.mark.distributed

EVAL_STEPS = 5  # deliberately odd, so it does not divide by 2


def _eval_loss(rank, world_size, out_dir, strategy):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(
        out_dir, strategy=strategy, micro_batch_size=4, eval_steps=EVAL_STEPS
    )
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    return {
        "eval_loss": trainer.evaluate(),
        "dp_world_size": trainer.dp_world_size,
        "weighted_batches": len(
            range(trainer.dp_rank, EVAL_STEPS, trainer.dp_world_size)
        ),
        # Every rank runs the same number of forwards, whatever its share of the weighted ones.
        "forward_passes": -(-EVAL_STEPS // trainer.dp_world_size),
    }


def _single_eval_loss(out_dir):
    """The one-process reference, run in this process."""
    from minimegatron.training.trainer import Trainer as T

    cfg = nano_experiment(out_dir, micro_batch_size=4, eval_steps=EVAL_STEPS)
    return T(cfg, out_dir=out_dir, echo=False).evaluate()


def test_ddp_validation_loss_equals_the_single_process_value(tmp_path):
    """The headline fix: the same number, not a differently-sized sample of the same data."""
    results = run_distributed(_eval_loss, 2, args=(str(tmp_path / "ddp"), "ddp"))
    reference = _single_eval_loss(str(tmp_path / "ref"))

    for rank in (0, 1):
        assert results[rank]["eval_loss"] == pytest.approx(reference, rel=1e-6)


def test_the_ranks_split_an_odd_number_of_batches_unevenly(tmp_path):
    """5 batches over 2 ranks is 3 and 2 -- which is why the sum and the count are both reduced,
    instead of averaging each rank's mean."""
    results = run_distributed(_eval_loss, 2, args=(str(tmp_path / "ddp"), "ddp"))
    assert sorted(results[r]["weighted_batches"] for r in (0, 1)) == [2, 3]
    assert results[0]["eval_loss"] == pytest.approx(results[1]["eval_loss"], rel=1e-9)


def test_every_rank_runs_the_same_number_of_forward_passes(tmp_path):
    """The uneven split must not become an uneven number of *forwards*.

    Under FSDP the forward contains collectives, so a rank that leaves the loop early strands its
    peers in an all-gather that never completes. That is a deadlock rather than a wrong number,
    and it is what ISSUES #13 records. The surplus pass is computed and weighted zero.
    """
    results = run_distributed(_eval_loss, 2, args=(str(tmp_path / "fsdp"), "fsdp"))
    assert results[0]["forward_passes"] == results[1]["forward_passes"] == 3  # ceil(5 / 2)


def test_fsdp_validation_loss_also_matches(tmp_path):
    results = run_distributed(_eval_loss, 2, args=(str(tmp_path / "fsdp"), "fsdp"))
    reference = _single_eval_loss(str(tmp_path / "ref"))
    assert results[0]["eval_loss"] == pytest.approx(reference, rel=1e-5)


def test_tp_validation_loss_also_matches(tmp_path):
    """TP ranks are not data-parallel, so each evaluates all 5 batches and agrees already."""
    results = run_distributed(_eval_loss, 2, args=(str(tmp_path / "tp"), "tp"))
    reference = _single_eval_loss(str(tmp_path / "ref"))
    for rank in (0, 1):
        assert results[rank]["dp_world_size"] == 1
        assert results[rank]["weighted_batches"] == EVAL_STEPS
        assert results[rank]["eval_loss"] == pytest.approx(reference, rel=1e-5)
