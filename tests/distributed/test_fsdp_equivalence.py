"""FSDP2 must reproduce the single-device reference: sharding is a memory trade, not a maths one."""

import pytest
import torch
from torch.distributed.tensor import DTensor

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import (
    nano_experiment,
    run_distributed,
    single_process_reference,
)

pytestmark = pytest.mark.distributed


def _full(tensor):
    """Gather a possibly-sharded tensor into a plain one for comparison."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _one_fsdp_step(rank, world_size, out_dir, micro, accum):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, strategy="fsdp", micro_batch_size=micro, grad_accum_steps=accum)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    record = trainer.train_step()
    return {
        "record": record,
        # full_tensor() all-gathers the shards, so these are comparable with the reference.
        "grads": {name: _full(p.grad).clone() for name, p in trainer.raw_model.named_parameters()},
        "weights": {
            name: _full(p.detach()).clone() for name, p in trainer.raw_model.named_parameters()
        },
        "strategy": trainer.strategy,
    }


def test_fsdp_gradients_match_the_single_device_reference(tmp_path):
    """Exit criterion 2. 2 ranks x micro=4 against 1 process x micro=8."""
    results = run_distributed(_one_fsdp_step, 2, args=(str(tmp_path / "fsdp"), 4, 1))
    _, reference_grads, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )

    assert results[0]["strategy"] == "fsdp"
    assert set(results[0]["grads"]) == set(reference_grads)
    for name, grad in results[0]["grads"].items():
        torch.testing.assert_close(
            grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=f"{name} differs"
        )


def test_fsdp_loss_matches_the_reference(tmp_path):
    results = run_distributed(_one_fsdp_step, 2, args=(str(tmp_path / "fsdp"), 4, 1))
    reference_records, _, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )
    for rank in (0, 1):
        assert results[rank]["record"]["loss"] == pytest.approx(
            reference_records[0]["loss"], rel=1e-5
        )


def test_gradient_norm_is_global_across_shards(tmp_path):
    """`clip_grad_norm_` has to reduce across shards, or the effective clip would depend on the
    world size. Asserted rather than trusted -- it is the one part of the FSDP path that silently
    does the wrong thing if DTensor support regresses."""
    results = run_distributed(_one_fsdp_step, 2, args=(str(tmp_path / "fsdp"), 4, 1))
    reference_records, _, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )
    assert results[0]["record"]["grad_norm"] == pytest.approx(
        reference_records[0]["grad_norm"], rel=1e-4
    )
    assert results[0]["record"]["grad_norm"] == pytest.approx(results[1]["record"]["grad_norm"])


def test_accumulation_under_fsdp_matches_one_large_batch(tmp_path):
    """FSDP's `set_requires_gradient_sync(False)` is its `no_sync`; the sums still commute."""
    results = run_distributed(_one_fsdp_step, 2, args=(str(tmp_path / "fsdp"), 2, 2))
    _, reference_grads, _ = single_process_reference(
        str(tmp_path / "ref"), steps=1, micro_batch_size=8
    )
    for name, grad in results[0]["grads"].items():
        torch.testing.assert_close(
            grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=f"{name} differs"
        )


def _fsdp_losses(rank, world_size, out_dir, steps):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, strategy="fsdp", micro_batch_size=4, warmup_steps=2,
                          max_steps=steps)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    records = [trainer.train_step() for _ in range(steps)]
    return {
        "losses": [r["loss"] for r in records],
        "weights": _full(trainer.raw_model.tok_emb.weight.detach()).clone(),
    }


def test_fsdp_run_tracks_the_reference_loss_curve(tmp_path):
    """Exit criterion 3: several optimizer steps, so a drift that needs a few steps to show up
    cannot hide."""
    steps = 5
    results = run_distributed(_fsdp_losses, 2, args=(str(tmp_path / "fsdp"), steps))
    reference, _, _ = single_process_reference(
        str(tmp_path / "ref"), steps=steps, micro_batch_size=8, max_steps=steps, warmup_steps=2
    )
    for step, (got, want) in enumerate(zip(results[0]["losses"], [r["loss"] for r in reference])):
        assert got == pytest.approx(want, rel=1e-4), f"diverged at step {step}"


def test_fsdp_weights_match_the_reference_after_several_steps(tmp_path):
    steps = 5
    results = run_distributed(_fsdp_losses, 2, args=(str(tmp_path / "fsdp"), steps))
    _, _, ref_trainer = single_process_reference(
        str(tmp_path / "ref"), steps=steps, micro_batch_size=8, max_steps=steps, warmup_steps=2
    )
    torch.testing.assert_close(
        results[0]["weights"], ref_trainer.raw_model.tok_emb.weight.detach(),
        rtol=1e-4, atol=1e-6,
    )


def _grad_presence_mid_group(rank, world_size, out_dir, suppress):
    """Is a sharded gradient available part-way through an accumulation group?"""
    import contextlib

    info = init_distributed(DistributedConfig(backend="gloo"))
    accum = 2
    cfg = nano_experiment(out_dir, strategy="fsdp", micro_batch_size=2, grad_accum_steps=accum)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    probe = trainer.raw_model.blocks[0].mlp.fc1.weight

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    mid_kind = None
    for j in range(accum):
        context = (
            trainer._sync_context(is_last_micro_batch=j == accum - 1)
            if suppress
            else contextlib.nullcontext()
        )
        with context:
            x, y = trainer.train_stream.micro_batch(j)
            _, loss = trainer.model(x, y)
            (loss / accum).backward()
        if j == 0:
            mid_kind = type(probe.grad).__name__
    return {"mid": mid_kind, "final": type(probe.grad).__name__}


def test_fsdp_no_sync_suppresses_the_intermediate_reduce_scatter(tmp_path):
    """FSDP's accumulation switch, observed rather than assumed.

    FSDP accumulates unreduced gradients in an internal **unsharded** buffer and only populates
    `param.grad` when the reduce-scatter runs. So with the sync suppressed, there is simply no
    sharded gradient mid-group (`param.grad is None`), and with it enabled there is one. Both
    branches are measured in the same test so the comparison is real rather than a claim about
    one of them.

    That internal buffer is also the trade-off: suppressing the collective costs memory to save
    bandwidth, the opposite balance from DDP's `no_sync`. Correctness is unaffected either way,
    since reduction and accumulation are both sums.
    """
    suppressed = run_distributed(_grad_presence_mid_group, 2, args=(str(tmp_path / "a"), True))
    synced = run_distributed(_grad_presence_mid_group, 2, args=(str(tmp_path / "b"), False))

    for rank in (0, 1):
        assert suppressed[rank]["mid"] == "NoneType", "the reduce-scatter was not suppressed"
        assert synced[rank]["mid"] == "DTensor"  # ...and it does happen when not suppressed
        assert suppressed[rank]["final"] == "DTensor"  # the last micro-batch always reduces
