"""Tests for the training loop: batching equivalence, convergence, exact resume, and the CLI."""

import dataclasses
import json
import subprocess
import sys

import pytest
import torch

from minimegatron.config import ConfigError
from minimegatron.data.config import DataConfig
from minimegatron.experiment import ExperimentConfig
from minimegatron.train import apply_overrides, main, parse_override
from minimegatron.training.config import TrainConfig
from minimegatron.training.optimizer import lr_at
from minimegatron.training.trainer import Trainer, resolve_device

CLI_TIMEOUT = 300  # macOS has no `timeout` binary; every subprocess gets a hard limit (ISSUES #1)


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


def trainer_for(cfg, out_dir) -> Trainer:
    return Trainer(cfg, out_dir=out_dir, echo=False)


# --- gradient accumulation ----------------------------------------------------------------------


def test_accumulation_matches_one_large_batch(nano_config, tmp_path):
    """accum=4 x micro=2 must produce the same gradients as accum=1 x micro=8.

    Gradients sum across backward() calls, so each micro-loss is scaled by 1/accum to make the
    accumulated gradient the gradient of the *mean* loss over the global batch. This equivalence
    is also what Milestone 2's DDP correctness test is built on.
    """
    split = trainer_for(experiment(nano_config, tmp_path / "a", micro_batch_size=2,
                                   grad_accum_steps=4), tmp_path / "a")
    whole = trainer_for(experiment(nano_config, tmp_path / "b", micro_batch_size=8,
                                   grad_accum_steps=1), tmp_path / "b")

    split_record = split.train_step()
    whole_record = whole.train_step()

    assert split_record["loss"] == pytest.approx(whole_record["loss"], rel=1e-5)
    assert split_record["grad_norm"] == pytest.approx(whole_record["grad_norm"], rel=1e-4)
    for (name, a), (_, b) in zip(split.model.named_parameters(), whole.model.named_parameters()):
        # Not bitwise: summing four partial gradients rounds differently from one reduction.
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6, msg=f"{name} differs")


def test_accumulation_counts_tokens_per_global_batch(nano_config, tmp_path):
    t = trainer_for(experiment(nano_config, tmp_path, micro_batch_size=2, grad_accum_steps=4),
                    tmp_path)
    t.train_step()
    assert t.tokens_seen == 2 * 4 * nano_config.seq_len


# --- global batch validation --------------------------------------------------------------------


def test_mismatched_global_batch_size_is_rejected(nano_config, tmp_path):
    cfg = experiment(nano_config, tmp_path, micro_batch_size=4, grad_accum_steps=2,
                     global_batch_size=99)
    with pytest.raises(ConfigError, match="global_batch_size 99"):
        trainer_for(cfg, tmp_path)


def test_matching_global_batch_size_is_accepted(nano_config, tmp_path):
    cfg = experiment(nano_config, tmp_path, micro_batch_size=4, grad_accum_steps=2,
                     global_batch_size=8)
    assert trainer_for(cfg, tmp_path).step == 0


# --- convergence --------------------------------------------------------------------------------


@pytest.mark.slow
def test_model_overfits_a_fixed_batch(nano_config, tmp_path):
    """The end-to-end proof that forward, backward, optimizer and target alignment are correct.

    A model that cannot memorize four sequences it has seen hundreds of times has something
    genuinely broken -- and this catches whole classes of bug that leave training looking fine.
    """
    cfg = experiment(nano_config, tmp_path, max_steps=300, micro_batch_size=4,
                     lr=3e-3, warmup_steps=10)
    trainer = trainer_for(cfg, tmp_path)

    first = trainer.train_step()["loss"]
    assert first == pytest.approx(torch.log(torch.tensor(float(nano_config.vocab_size))).item(),
                                  abs=0.5)
    for _ in range(cfg.train.max_steps - 1):
        record = trainer.train_step()
    assert record["loss"] < 0.1, f"loss stalled at {record['loss']:.4f}"


def test_random_data_stays_near_ln_vocab(nano_config, tmp_path):
    """The flip side: uniform random tokens are unlearnable, so loss must *not* fall.

    This is what makes RandomTokenDataset the right choice for throughput benchmarks -- there is
    no learning curve to confuse the measurement.
    """
    cfg = dataclasses.replace(
        experiment(nano_config, tmp_path, max_steps=30, lr=1e-3),
        data=DataConfig(dataset="random", n_items=256, seed=0),
    )
    trainer = trainer_for(cfg, tmp_path)
    for _ in range(30):
        record = trainer.train_step()
    expected = torch.log(torch.tensor(float(nano_config.vocab_size))).item()
    assert record["loss"] > expected - 0.5


# --- exact resume -------------------------------------------------------------------------------


@pytest.mark.slow
def test_resume_is_bit_for_bit_identical_to_an_uninterrupted_run(nano_config, tmp_path):
    """Train 2N steps, versus N + checkpoint + fresh process state + resume + N.

    Dropout is deliberately non-zero so the test actually exercises RNG restoration. This passes
    only if weights, Adam's moments, the schedule, the data position and every RNG are all
    restored, which makes it the single strongest test in the milestone.
    """
    model_cfg = dataclasses.replace(nano_config, dropout=0.1)
    N = 4

    straight = trainer_for(experiment(model_cfg, tmp_path / "a", max_steps=2 * N),
                           tmp_path / "a")
    losses_straight = [straight.train_step()["loss"] for _ in range(2 * N)]

    first_half = trainer_for(experiment(model_cfg, tmp_path / "b", max_steps=2 * N),
                             tmp_path / "b")
    losses_resumed = [first_half.train_step()["loss"] for _ in range(N)]
    ckpt = first_half.save()

    # A fresh Trainer: new weights, empty optimizer state, schedule at step 0, different RNG.
    second_half = trainer_for(experiment(model_cfg, tmp_path / "c", max_steps=2 * N),
                              tmp_path / "c")
    assert not torch.equal(second_half.model.tok_emb.weight, first_half.model.tok_emb.weight)
    second_half.resume(ckpt)
    assert second_half.step == N
    losses_resumed += [second_half.train_step()["loss"] for _ in range(N)]

    assert losses_resumed == losses_straight  # exact equality, not approx
    for (name, a), (_, b) in zip(
        second_half.model.state_dict().items(), straight.model.state_dict().items()
    ):
        assert torch.equal(a, b), f"{name} diverged after resume"


def test_resume_restores_the_learning_rate_mid_schedule(nano_config, tmp_path):
    """Regression test: a restored scheduler must not take one step at the warmup rate."""
    cfg = experiment(nano_config, tmp_path / "a", max_steps=50, warmup_steps=5, lr=1e-2)
    trainer = trainer_for(cfg, tmp_path / "a")
    for _ in range(20):
        trainer.train_step()
    ckpt = trainer.save()
    expected_next_lr = trainer.optimizer.param_groups[0]["lr"]

    resumed = trainer_for(experiment(nano_config, tmp_path / "b", max_steps=50,
                                     warmup_steps=5, lr=1e-2), tmp_path / "b")
    resumed.resume(ckpt)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(expected_next_lr)
    assert resumed.train_step()["lr"] == pytest.approx(expected_next_lr)


def test_resume_with_a_changed_schedule_follows_the_new_one(nano_config, tmp_path):
    """Extending a run on resume (`--set train.max_steps=...`) must take effect immediately.

    The learning rate rides back in on the optimizer's state dict, so without recomputing it from
    the live config the first step after resuming would silently use the *old* schedule's value.
    """
    short = experiment(nano_config, tmp_path / "a", max_steps=10, warmup_steps=2, lr=1e-2)
    trainer = trainer_for(short, tmp_path / "a")
    for _ in range(10):
        trainer.train_step()
    ckpt = trainer.save()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(short.train.min_lr)

    extended = experiment(nano_config, tmp_path / "b", max_steps=100, warmup_steps=2, lr=1e-2)
    resumed = trainer_for(extended, tmp_path / "b")
    resumed.resume(ckpt)

    expected = lr_at(10, extended.train.lr, extended.train.min_lr, 2, 100)
    assert resumed.train_step()["lr"] == pytest.approx(expected)
    assert expected > short.train.min_lr * 2  # the two schedules really do differ here


def test_resume_continues_the_data_stream(nano_config, tmp_path):
    """No dataloader state is saved; the step counter alone must put us back in the stream."""
    cfg = experiment(nano_config, tmp_path / "a", max_steps=10)
    trainer = trainer_for(cfg, tmp_path / "a")
    for _ in range(3):
        trainer.train_step()
    expected = trainer.train_stream.indices(3)

    resumed = trainer_for(experiment(nano_config, tmp_path / "b", max_steps=10), tmp_path / "b")
    resumed.resume(trainer.save())
    torch.testing.assert_close(resumed.train_stream.indices(resumed.step), expected)


# --- the run loop and logging -------------------------------------------------------------------


def test_fit_runs_the_configured_number_of_steps_and_logs_jsonl(nano_config, tmp_path):
    cfg = experiment(nano_config, tmp_path, max_steps=6, log_interval=2, eval_interval=3,
                     eval_steps=2)
    trainer = trainer_for(cfg, tmp_path)
    trainer.fit()
    assert trainer.step == 6

    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    # Every log_interval step, plus a line of its own for an evaluation that falls between them:
    # a validation measurement is never dropped just because it missed a logging boundary.
    assert [r["step"] for r in lines] == [2, 3, 4, 6]
    assert [r["step"] for r in lines if "val_loss" in r] == [3, 6]
    for record in lines:
        assert {"loss", "lr", "grad_norm", "tokens_seen", "tokens_per_sec"} <= set(record)
        assert record["tokens_per_sec"] > 0


def test_fit_writes_a_final_checkpoint(nano_config, tmp_path):
    trainer = trainer_for(experiment(nano_config, tmp_path, max_steps=3), tmp_path)
    trainer.fit()
    assert (tmp_path / "ckpt_000003.pt").is_file()


def test_evaluation_does_not_disturb_training(nano_config, tmp_path):
    """eval() turns dropout off, so evaluating must consume no RNG and change no weights.

    The two runs are built and driven one after the other, never interleaved: there is a single
    global torch RNG, and `Trainer.__init__` reseeds it, so a second trainer constructed early
    would leave the first one drawing dropout masks from the wrong point in the stream. Any test
    comparing two runs in one process has to keep them sequential.
    """
    model_cfg = dataclasses.replace(nano_config, dropout=0.1)

    a = trainer_for(experiment(model_cfg, tmp_path / "a", eval_steps=2), tmp_path / "a")
    rng_before = torch.get_rng_state().clone()
    weights_before = a.model.tok_emb.weight.detach().clone()
    a.train_step()
    a.evaluate()  # the only difference between the two runs
    loss_a = a.train_step()["loss"]

    b = trainer_for(experiment(model_cfg, tmp_path / "b", eval_steps=2), tmp_path / "b")
    assert torch.equal(torch.get_rng_state(), rng_before)  # construction reseeds identically
    assert torch.equal(b.model.tok_emb.weight, weights_before)
    b.train_step()
    assert b.train_step()["loss"] == loss_a


def test_training_mode_is_restored_after_evaluation(nano_config, tmp_path):
    trainer = trainer_for(experiment(nano_config, tmp_path, eval_steps=1), tmp_path)
    trainer.evaluate()
    assert trainer.model.training


# --- device resolution --------------------------------------------------------------------------


def test_auto_device_resolves_to_something_available():
    assert resolve_device("auto").type in ("cuda", "mps", "cpu")


def test_explicit_cpu_is_honoured():
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_requesting_cuda_without_cuda_is_a_hard_error():
    """A run that quietly falls back to CPU reports throughput that is wrong by orders of
    magnitude, so the fallback must not be silent."""
    with pytest.raises(ConfigError, match="CUDA is not available"):
        resolve_device("cuda")


# --- the CLI ------------------------------------------------------------------------------------


def test_parse_override_uses_yaml_types():
    assert parse_override("train.max_steps=20") == (["train", "max_steps"], 20)
    assert parse_override("model.bias=false") == (["model", "bias"], False)
    assert parse_override("train.lr=6.0e-4") == (["train", "lr"], 6.0e-4)
    assert parse_override("train.global_batch_size=null") == (
        ["train", "global_batch_size"], None
    )


@pytest.mark.parametrize("text", ["train.max_steps", "maxsteps=3", "=3"])
def test_malformed_override_is_rejected(text):
    with pytest.raises(ConfigError):
        parse_override(text)


def test_overrides_do_not_mutate_the_loaded_config():
    raw = {"model": {"n_layers": 2}, "train": {"max_steps": 5}}
    merged = apply_overrides(raw, ["train.max_steps=9"])
    assert merged["train"]["max_steps"] == 9
    assert raw["train"]["max_steps"] == 5  # the original mapping is untouched


def test_override_typo_is_rejected_like_a_file_typo(nano_config, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("model: {n_layers: 2, n_heads: 4, d_model: 32}\n")
    assert main(["--config", str(config), "--set", "train.max_step=3"]) == 2


def write_nano_config(path, out_dir) -> None:
    path.write_text(
        "model:\n"
        "  vocab_size: 128\n  seq_len: 16\n  n_layers: 2\n  n_heads: 4\n  d_model: 32\n"
        "  d_ff: 64\n"
        "data:\n"
        "  dataset: memorization\n  n_items: 16\n  n_sequences: 4\n  seed: 0\n"
        "train:\n"
        f"  max_steps: 4\n  micro_batch_size: 4\n  lr: 1.0e-3\n  device: cpu\n"
        f"  log_interval: 2\n  out_dir: {out_dir}\n"
    )


def test_cli_smoke_run(tmp_path):
    """Exercise the real entry point in a subprocess: argument parsing, config loading,
    device selection, the loop, logging and the final checkpoint."""
    config = tmp_path / "nano.yaml"
    out_dir = tmp_path / "run"
    write_nano_config(config, out_dir)

    result = subprocess.run(
        [sys.executable, "-m", "minimegatron.train", "--config", str(config)],
        capture_output=True, text=True, timeout=CLI_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    assert "parameters" in result.stdout

    lines = [json.loads(line) for line in (out_dir / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [2, 4]
    assert (out_dir / "ckpt_000004.pt").is_file()


def test_cli_resume_smoke_run(tmp_path):
    config = tmp_path / "nano.yaml"
    out_dir = tmp_path / "run"
    write_nano_config(config, out_dir)
    base = [sys.executable, "-m", "minimegatron.train", "--config", str(config)]

    first = subprocess.run(base, capture_output=True, text=True, timeout=CLI_TIMEOUT)
    assert first.returncode == 0, first.stderr

    second = subprocess.run(
        base + ["--set", "train.max_steps=8", "--resume", str(out_dir / "ckpt_000004.pt")],
        capture_output=True, text=True, timeout=CLI_TIMEOUT,
    )
    assert second.returncode == 0, second.stderr
    assert "resumed from" in second.stdout
    assert (out_dir / "ckpt_000008.pt").is_file()


def test_cli_reports_a_bad_config_without_a_traceback(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("model: {n_layer: 2, n_heads: 4, d_model: 32}\n")
    result = subprocess.run(
        [sys.executable, "-m", "minimegatron.train", "--config", str(config)],
        capture_output=True, text=True, timeout=CLI_TIMEOUT,
    )
    assert result.returncode == 2
    assert "did you mean 'n_layers'" in result.stderr
    assert "Traceback" not in result.stderr
