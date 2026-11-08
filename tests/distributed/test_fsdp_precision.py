"""BF16 under FSDP, which uses MixedPrecisionPolicy rather than autocast."""

import pytest
import torch
from torch.distributed.tensor import DTensor

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.fsdp import mixed_precision_policy
from minimegatron.distributed.setup import init_distributed
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import nano_experiment, run_distributed

pytestmark = pytest.mark.distributed


def _precision_facts(rank, world_size, out_dir, precision):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, strategy="fsdp", micro_batch_size=4, precision=precision)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)

    seen = {}
    fc1 = trainer.raw_model.blocks[0].mlp.fc1
    # A pre-hook sees the parameter as the forward sees it: after FSDP's all-gather, in whatever
    # dtype the policy gathered it as.
    fc1.register_forward_pre_hook(lambda m, args: seen.__setitem__("gathered", m.weight.dtype))
    fc1.register_forward_hook(lambda m, a, o: seen.__setitem__("activation", o.dtype))

    record = trainer.train_step()

    # The loss dtype, taken from a direct forward so the tensor itself can be inspected.
    x, y = trainer.train_stream.micro_batch(0)
    with torch.no_grad():
        logits, loss = trainer.model(x.to(trainer.device), y.to(trainer.device))

    stored = trainer.raw_model.blocks[0].mlp.fc1.weight
    return {
        "logits_dtype": logits.dtype,
        "loss_dtype": loss.dtype,
        "gathered_dtype": seen["gathered"],
        "activation_dtype": seen["activation"],
        "stored_dtype": stored.to_local().dtype if isinstance(stored, DTensor) else stored.dtype,
        "grad_dtype": stored.grad.to_local().dtype
        if isinstance(stored.grad, DTensor)
        else stored.grad.dtype,
        "uses_autocast": trainer._autocast().__class__.__name__ != "nullcontext",
        "loss": record["loss"],
    }


def test_bf16_gathers_parameters_in_bf16_but_stores_them_in_fp32(tmp_path):
    """Exit criterion 5, and the reason FSDP needs its own mechanism.

    The shard on disk/in memory is fp32 (master weights), but the all-gather that feeds the
    forward moves bf16 -- halving the volume of FSDP's dominant collective. Autocast cannot do
    this: it converts after the gather has already moved fp32 bytes.
    """
    results = run_distributed(_precision_facts, 2, args=(str(tmp_path / "bf16"), "bf16"))
    for rank in (0, 1):
        assert results[rank]["gathered_dtype"] == torch.bfloat16
        assert results[rank]["activation_dtype"] == torch.bfloat16
        assert results[rank]["stored_dtype"] == torch.float32  # fp32 master weights
        # reduce_dtype=fp32: summing many small bf16 gradients across ranks loses precision
        # exactly where it matters, and the reduction is cheap next to the gather.
        assert results[rank]["grad_dtype"] == torch.float32


def test_fsdp_does_not_stack_autocast_on_top_of_the_policy(tmp_path):
    """Two mechanisms doing the same cast would be redundant; the policy is the right one here."""
    results = run_distributed(_precision_facts, 2, args=(str(tmp_path / "bf16"), "bf16"))
    assert results[0]["uses_autocast"] is False


def test_fp32_under_fsdp_stays_fp32_throughout(tmp_path):
    results = run_distributed(_precision_facts, 2, args=(str(tmp_path / "fp32"), "fp32"))
    for rank in (0, 1):
        assert results[rank]["gathered_dtype"] == torch.float32
        assert results[rank]["activation_dtype"] == torch.float32
        assert results[rank]["stored_dtype"] == torch.float32


def test_bf16_and_fp32_fsdp_losses_agree_within_bf16_tolerance(tmp_path):
    bf16 = run_distributed(_precision_facts, 2, args=(str(tmp_path / "b"), "bf16"))
    fp32 = run_distributed(_precision_facts, 2, args=(str(tmp_path / "f"), "fp32"))
    assert bf16[0]["loss"] == pytest.approx(fp32[0]["loss"], rel=2e-2)


def _bf16_converges(rank, world_size, out_dir, steps):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(
        out_dir, strategy="fsdp", micro_batch_size=2, precision="bf16",
        max_steps=steps, lr=3e-3, warmup_steps=10,
    )
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    first = trainer.train_step()["loss"]
    for _ in range(steps - 1):
        last = trainer.train_step()["loss"]
    return {"first": first, "last": last}


@pytest.mark.slow
def test_bf16_fsdp_still_trains(tmp_path):
    """The only test of a precision mode that really matters: does it still learn?"""
    results = run_distributed(_bf16_converges, 2, args=(str(tmp_path / "c"), 300))
    assert results[0]["last"] < results[0]["first"] / 2
    assert results[0]["last"] < 0.5, f"bf16 FSDP training stalled at {results[0]['last']:.4f}"


def test_no_policy_for_fp32():
    assert mixed_precision_policy("fp32") is None
    assert mixed_precision_policy("bf16").param_dtype == torch.bfloat16
    assert mixed_precision_policy("bf16").reduce_dtype == torch.float32


def test_the_loss_is_fp32_under_fsdp_bf16(tmp_path):
    """Load-bearing here in a way it is not under autocast.

    Autocast has `cross_entropy` on its fp32 promotion list, so the explicit upcast in
    `GPT.forward` is redundant for the single-device and DDP paths. Under FSDP there is no
    autocast -- `MixedPrecisionPolicy` does the casting -- so bf16 logits would reach
    `cross_entropy` in bf16 and the log-sum-exp over the vocabulary, the most precision-sensitive
    op in the model, would run in 8 mantissa bits. The `torch.promote_types` call from ISSUES #4
    is what prevents that, and this is the test that notices if it goes away.
    """
    results = run_distributed(_precision_facts, 2, args=(str(tmp_path / "bf16"), "bf16"))
    for rank in (0, 1):
        assert results[rank]["logits_dtype"] == torch.bfloat16  # the forward really was bf16
        assert results[rank]["loss_dtype"] == torch.float32  # ...and the loss was not
