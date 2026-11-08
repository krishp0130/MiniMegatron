"""`no_sync` gradient accumulation under DDP."""

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import (
    nano_experiment,
    run_distributed,
    single_process_reference,
)

pytestmark = pytest.mark.distributed


def _accumulated_step(rank, world_size, out_dir, micro, accum):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, micro_batch_size=micro, grad_accum_steps=accum)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    record = trainer.train_step()
    return {
        "record": record,
        "grads": {name: p.grad.clone() for name, p in trainer.raw_model.named_parameters()},
    }


def test_accumulation_across_ranks_matches_one_large_batch(tmp_path):
    """2 ranks x accum=2 x micro=2 must equal 1 process x micro=8 (exit criterion 3).

    Local accumulation and the all-reduce are both sums, and sums commute, so suppressing the
    intermediate synchronizations cannot change the result -- only the communication volume.
    """
    results = run_distributed(_accumulated_step, 2, args=(str(tmp_path / "ddp"), 2, 2))
    _, reference_grads, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )
    for name, grad in results[0]["grads"].items():
        torch.testing.assert_close(
            grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=f"{name} differs"
        )


def test_accumulated_gradients_are_synchronized_across_ranks(tmp_path):
    results = run_distributed(_accumulated_step, 2, args=(str(tmp_path / "ddp"), 2, 2))
    for name, grad in results[0]["grads"].items():
        torch.testing.assert_close(grad, results[1]["grads"][name], msg=f"{name} differs by rank")


def _grads_between_micro_batches(rank, world_size, out_dir):
    """Snapshot one parameter's gradient after each micro-batch of an accumulation group."""
    info = init_distributed(DistributedConfig(backend="gloo"))
    accum = 2
    cfg = nano_experiment(out_dir, micro_batch_size=2, grad_accum_steps=accum)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    probe = trainer.raw_model.blocks[0].mlp.fc1.weight

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    snapshots = []
    for j in range(accum):
        with trainer._sync_context(is_last_micro_batch=j == accum - 1):
            x, y = trainer.train_stream.micro_batch(j)
            _, loss = trainer.model(x, y)
            (loss / accum).backward()
        snapshots.append(probe.grad.clone())
    return {"after_first": snapshots[0], "after_last": snapshots[1]}


def test_no_sync_suppresses_the_intermediate_all_reduce(tmp_path):
    """Observed directly in the gradients: unsynchronized mid-group, synchronized at the end.

    After the first micro-batch the ranks must *disagree* -- each holds only its own shard's
    partial gradient, because `no_sync` suppressed the reducer. After the last micro-batch, which
    runs outside the context, they must agree exactly. That pair of facts is what "exactly one
    synchronization per optimizer step" means operationally, and it does not depend on poking at
    DDP's internals.
    """
    results = run_distributed(_grads_between_micro_batches, 2, args=(str(tmp_path / "a"),))

    assert not torch.allclose(results[0]["after_first"], results[1]["after_first"]), (
        "ranks already agreed mid-group: no_sync did not suppress the all-reduce"
    )
    torch.testing.assert_close(results[0]["after_last"], results[1]["after_last"])


def _sync_context_types(rank, world_size, out_dir):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, micro_batch_size=2, grad_accum_steps=3)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    assert isinstance(trainer.model, DistributedDataParallel)
    return [
        type(trainer._sync_context(is_last_micro_batch=j == 2)).__name__ for j in range(3)
    ]


def test_only_the_last_micro_batch_synchronizes(tmp_path):
    results = run_distributed(_sync_context_types, 2, args=(str(tmp_path / "a"),))
    kinds = results[0]
    assert kinds[0] == kinds[1] != "nullcontext"  # suppressed
    assert kinds[2] == "nullcontext"  # the synchronizing one
