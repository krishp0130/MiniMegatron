"""BF16 mixed precision: bf16 compute over fp32 master weights."""

import dataclasses

import pytest
import torch

from minimegatron.config import ConfigError
from minimegatron.data.config import DataConfig
from minimegatron.experiment import ExperimentConfig
from minimegatron.training.config import TrainConfig
from minimegatron.training.trainer import Trainer, validate_precision


def experiment(nano_config, out_dir, **train_overrides) -> ExperimentConfig:
    train = dict(
        max_steps=4,
        micro_batch_size=4,
        grad_accum_steps=1,
        lr=1e-3,
        warmup_steps=0,
        grad_clip=0.0,
        device="cpu",
        seed=1337,
        out_dir=str(out_dir),
        log_interval=0,
        eval_interval=0,
    )
    train.update(train_overrides)
    return ExperimentConfig(
        model=nano_config,
        data=DataConfig(dataset="memorization", n_items=16, n_sequences=4, seed=0),
        train=TrainConfig(**train),
    )


def trainer_for(nano_config, out_dir, **overrides) -> Trainer:
    return Trainer(experiment(nano_config, out_dir, **overrides), out_dir=out_dir, echo=False)


# --- what is bf16 and what stays fp32 -----------------------------------------------------------


def test_matmuls_run_in_bf16_while_parameters_stay_fp32(nano_config, tmp_path):
    """The definition of mixed precision: bf16 math, fp32 master weights.

    Note *where* the bf16 is. Autocast has an op table: `Linear` runs in bf16, normalization stays
    fp32 (its reductions need the range), and `Embedding` is untouched. So the residual stream
    stays fp32 too -- `fp32 + bf16` promotes to fp32 -- and only the matmuls are low precision.
    That is the design, not an accident, and it is why a bf16 run's loss barely moves: the
    numerically delicate parts were never downcast.
    """
    trainer = trainer_for(nano_config, tmp_path, precision="bf16")
    seen = {}

    def record(name):
        return lambda module, args, output: seen.__setitem__(name, output.dtype)

    block = trainer.raw_model.blocks[0]
    block.mlp.fc1.register_forward_hook(record("linear"))
    block.norm1.register_forward_hook(record("norm"))
    block.register_forward_hook(record("block"))
    trainer.raw_model.tok_emb.register_forward_hook(record("embedding"))
    trainer.train_step()

    assert seen["linear"] == torch.bfloat16  # matmuls: low precision
    assert seen["norm"] == torch.float32  # reductions: full precision
    assert seen["embedding"] == torch.float32
    assert seen["block"] == torch.float32  # the residual stream is never downcast

    for name, p in trainer.raw_model.named_parameters():
        assert p.dtype == torch.float32, f"{name} is not fp32"


def test_fp32_run_has_no_bf16_activations(nano_config, tmp_path):
    trainer = trainer_for(nano_config, tmp_path, precision="fp32")
    seen = {}
    trainer.raw_model.blocks[0].mlp.fc1.register_forward_hook(
        lambda m, a, o: seen.__setitem__("dtype", o.dtype)
    )
    trainer.train_step()
    assert seen["dtype"] == torch.float32  # the same hook that sees bf16 under precision=bf16
    assert trainer.autocast_dtype is None


def test_loss_is_computed_in_fp32_under_autocast(nano_config, tmp_path):
    """The log-sum-exp over the vocabulary is the most precision-sensitive op in the model.

    M1 promotes logits with `torch.promote_types(dtype, float32)` before cross-entropy (the
    ISSUES #4 pattern), so this holds without any precision-specific code -- but it is worth
    pinning, because losing it would be invisible except as slightly worse training.
    """
    trainer = trainer_for(nano_config, tmp_path, precision="bf16")
    idx = torch.randint(0, nano_config.vocab_size, (2, 8))
    with trainer._autocast():
        logits, loss = trainer.raw_model(idx, idx)
    assert logits.dtype == torch.bfloat16  # the forward really did run in bf16
    assert loss.dtype == torch.float32  # ...and the loss did not


def test_gradients_and_optimizer_state_stay_fp32(nano_config, tmp_path):
    """Updates are applied in fp32: bf16 has 8 mantissa bits, so `w + small_update == w` far too
    often to keep master weights in it."""
    trainer = trainer_for(nano_config, tmp_path, precision="bf16")
    trainer.train_step()
    for name, p in trainer.raw_model.named_parameters():
        assert p.grad.dtype == torch.float32, f"{name} gradient is not fp32"
    for state in trainer.optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


# --- numerics -----------------------------------------------------------------------------------


def test_bf16_and_fp32_losses_agree_within_bf16_tolerance(nano_config, tmp_path):
    """Run sequentially: there is one global RNG and Trainer.__init__ reseeds it (ISSUES #6)."""
    fp32 = trainer_for(nano_config, tmp_path / "a", precision="fp32")
    fp32_losses = [fp32.train_step()["loss"] for _ in range(3)]

    bf16 = trainer_for(nano_config, tmp_path / "b", precision="bf16")
    bf16_losses = [bf16.train_step()["loss"] for _ in range(3)]

    # bf16 has ~3 decimal digits of precision, so this is a loose but meaningful bound.
    for step, (a, b) in enumerate(zip(fp32_losses, bf16_losses)):
        assert b == pytest.approx(a, rel=2e-2), f"diverged at step {step}"


def test_bf16_still_overfits_a_fixed_batch(nano_config, tmp_path):
    """The real test of a precision mode: can it still train?"""
    trainer = trainer_for(
        nano_config, tmp_path, precision="bf16", max_steps=300, lr=3e-3, warmup_steps=10
    )
    for _ in range(300):
        record = trainer.train_step()
    assert record["loss"] < 0.1, f"bf16 training stalled at {record['loss']:.4f}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_bf16_works_on_mps(nano_config, tmp_path):
    trainer = trainer_for(nano_config, tmp_path, precision="bf16", device="mps")
    record = trainer.train_step()
    assert torch.isfinite(torch.tensor(record["loss"]))


# --- configuration ------------------------------------------------------------------------------


def test_invalid_precision_rejected():
    with pytest.raises(ConfigError, match="precision must be one of"):
        TrainConfig(precision="fp16")


def test_bf16_on_cpu_and_mps_is_allowed():
    validate_precision("bf16", torch.device("cpu"))  # both support bf16 here
    validate_precision("fp32", torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.is_bf16_supported(),
    reason="needs a CUDA device without bf16 support",
)
def test_bf16_on_unsupported_cuda_is_a_hard_error():
    with pytest.raises(ConfigError, match="does not support bfloat16"):
        validate_precision("bf16", torch.device("cuda"))


def test_model_dtype_points_at_train_precision(nano_config):
    """`ModelConfig.dtype` is storage; bf16 is a compute setting, so the error must say so."""
    with pytest.raises(ConfigError, match="train.precision: bf16"):
        dataclasses.replace(nano_config, dtype="bfloat16")
