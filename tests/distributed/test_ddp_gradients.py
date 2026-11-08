"""The milestone's central claim: DDP changes the speed, not the math."""

import pytest
import torch

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import (
    nano_experiment,
    run_distributed,
    single_process_reference,
)

pytestmark = pytest.mark.distributed


def _one_ddp_step(rank, world_size, out_dir, micro, accum):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, micro_batch_size=micro, grad_accum_steps=accum)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    record = trainer.train_step()
    return {
        "record": record,
        "grads": {name: p.grad.clone() for name, p in trainer.raw_model.named_parameters()},
        "is_ddp": trainer.model is not trainer.raw_model,
        "tokens_seen": trainer.tokens_seen,
    }


def test_ddp_gradients_match_a_single_process_global_batch(tmp_path):
    """2 ranks x micro=4 must equal 1 process x micro=8, parameter by parameter.

    This is exit criterion 2, and the reason is that the loss is a mean over samples: for equal
    shards, the mean of the per-rank gradients *is* the gradient of the global batch. Any
    divergence here is a bug, not noise.
    """
    results = run_distributed(_one_ddp_step, 2, args=(str(tmp_path / "ddp"), 4, 1))
    _, reference_grads, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )

    assert results[0]["is_ddp"], "the model was not actually wrapped in DDP"
    assert set(results[0]["grads"]) == set(reference_grads)
    for name, grad in results[0]["grads"].items():
        # Not bitwise: the all-reduce sums in a different order than one big reduction.
        torch.testing.assert_close(
            grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=f"{name} differs"
        )


def test_every_rank_holds_the_same_gradients_after_the_all_reduce(tmp_path):
    """The point of synchronization: replicas take an identical step, so they never diverge."""
    results = run_distributed(_one_ddp_step, 2, args=(str(tmp_path / "ddp"), 4, 1))
    for name, grad in results[0]["grads"].items():
        torch.testing.assert_close(grad, results[1]["grads"][name], msg=f"{name} differs by rank")


def test_the_tied_embedding_gradient_survives_ddp(tmp_path):
    """The tied weight is used twice per forward; DDP's reducer must handle the shared parameter.

    Asserted rather than assumed, because `find_unused_parameters=False` is kept on, and a
    parameter the reducer mishandles would show up here and nowhere else.
    """
    results = run_distributed(_one_ddp_step, 2, args=(str(tmp_path / "ddp"), 4, 1))
    _, reference_grads, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )
    tied = results[0]["grads"]["tok_emb.weight"]
    assert torch.isfinite(tied).all() and tied.abs().sum() > 0
    torch.testing.assert_close(tied, reference_grads["tok_emb.weight"], rtol=1e-4, atol=1e-6)


def test_logged_loss_describes_the_global_batch(tmp_path):
    """A loss computed on one rank describes that rank's shard; aggregation makes it honest."""
    results = run_distributed(_one_ddp_step, 2, args=(str(tmp_path / "ddp"), 4, 1))
    reference_records, _, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )
    for rank in (0, 1):
        assert results[rank]["record"]["loss"] == pytest.approx(
            reference_records[0]["loss"], rel=1e-5
        )


def test_tokens_seen_counts_the_whole_cluster(tmp_path):
    results = run_distributed(_one_ddp_step, 2, args=(str(tmp_path / "ddp"), 4, 1))
    assert results[0]["tokens_seen"] == 4 * 2 * 16  # micro x ranks x seq_len
