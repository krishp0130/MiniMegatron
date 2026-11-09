"""Controlled failure injection: every case SCOPE.md section 8 names.

Each test asserts a **clear diagnostic**, not merely that something raised. An exception that says
`RuntimeError: shape mismatch` is a failure; one that says which setting to change is a feature.

Nothing here may hang. Distributed cases use subprocess isolation with a hard Python-side timeout
(ISSUES #1: macOS has no `timeout` binary and gloo hangs silently when misconfigured).
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minimegatron.config import ConfigError
from minimegatron.data.config import DataConfig
from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.tensor_parallel import validate_tp
from minimegatron.experiment import ExperimentConfig, load_config
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import GPT
from minimegatron.train import main as train_main
from minimegatron.training.checkpoint import load_checkpoint, save_checkpoint
from minimegatron.training.config import TrainConfig
from minimegatron.training.optimizer import build_optimizer, build_scheduler
from minimegatron.training.trainer import Trainer, resolve_device, validate_precision

TIMEOUT = 300


def experiment(nano_config, out_dir, **train_overrides) -> ExperimentConfig:
    train = dict(
        max_steps=2, micro_batch_size=4, grad_accum_steps=1, lr=1e-3, device="cpu",
        out_dir=str(out_dir), log_interval=0, eval_interval=0,
    )
    train.update(train_overrides)
    return ExperimentConfig(
        model=nano_config,
        data=DataConfig(dataset="random", n_items=64, seed=0),
        train=TrainConfig(**train),
    )


# --- invalid configuration ----------------------------------------------------------------------


def test_unknown_config_key_names_the_likely_intent(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("model: {n_layer: 2, n_heads: 4, d_model: 32}\n")
    with pytest.raises(ConfigError, match="did you mean 'n_layers'"):
        load_config(config)


def test_unknown_section_lists_the_valid_ones(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("model: {n_layers: 2, n_heads: 4, d_model: 32}\ntrianing: {}\n")
    with pytest.raises(ConfigError, match=r"unknown section.*expected \['model'"):
        load_config(config)


def test_every_config_error_is_reported_at_once():
    """One pass to fix a broken config, not one error per run."""
    with pytest.raises(ConfigError) as exc:
        ModelConfig(n_layers=0, n_heads=4, d_model=30, norm_type="batchnorm", dropout=2.0)
    message = str(exc.value)
    for fragment in ("n_layers", "divisible", "norm_type", "dropout"):
        assert fragment in message


def test_mismatched_global_batch_size_refuses_to_start(nano_config, tmp_path):
    """Caught before the first step: a silently wrong global batch invalidates an entire run."""
    cfg = experiment(nano_config, tmp_path, micro_batch_size=4, grad_accum_steps=2,
                     global_batch_size=99)
    with pytest.raises(ConfigError, match="global_batch_size 99 != micro_batch_size 4"):
        Trainer(cfg, out_dir=tmp_path, echo=False)


@pytest.mark.parametrize(
    "tp_size, world_size, message",
    [
        (3, 3, "n_heads .4. must be divisible by tp_size .3."),
        (2, 3, "world_size .3. must be divisible by tp_size .2."),
        (2, 4, "is not supported yet"),  # tp_size < world_size: 2D TP+DP, out of scope
        (0, 2, "tp_size must be at least 1"),
    ],
)
def test_invalid_tp_size_is_rejected_with_the_reason(nano_config, tp_size, world_size, message):
    """Explicitly required by SCOPE.md section 8."""
    with pytest.raises(ConfigError, match=message):
        validate_tp(nano_config, tp_size=tp_size, world_size=world_size)


def test_invalid_strategy_and_backend_rejected():
    with pytest.raises(ConfigError, match="strategy must be one of"):
        DistributedConfig(strategy="pipeline")
    with pytest.raises(ConfigError, match="backend must be one of"):
        DistributedConfig(backend="mpi")


# --- missing files ------------------------------------------------------------------------------


def test_missing_config_file_exits_cleanly_without_a_traceback(tmp_path, capsys):
    assert train_main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_missing_checkpoint_names_the_path(nano_config, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pt", model=GPT(nano_config))


def test_resuming_from_a_missing_checkpoint_fails_loudly(nano_config, tmp_path):
    trainer = Trainer(experiment(nano_config, tmp_path), out_dir=tmp_path, echo=False)
    with pytest.raises(FileNotFoundError):
        trainer.resume(tmp_path / "ghost.pt")


def test_sharded_checkpoint_directory_without_metadata_is_rejected(nano_config, tmp_path):
    from minimegatron.distributed.checkpoint import load_sharded

    empty = tmp_path / "ckpt_000001"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="not a sharded checkpoint"):
        load_sharded(empty, model=GPT(nano_config))


# --- corrupt checkpoints ------------------------------------------------------------------------


@pytest.fixture
def saved_checkpoint(nano_config, tmp_path):
    cfg = TrainConfig(lr=1e-3, out_dir=str(tmp_path))
    model = GPT(nano_config)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    path = save_checkpoint(
        tmp_path / "ckpt.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        step=1, tokens_seen=10, config=cfg,
    )
    return path, model


def test_truncated_checkpoint_fails_rather_than_loading_garbage(saved_checkpoint):
    path, model = saved_checkpoint
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(Exception) as exc:  # torch raises its own unpickling/zip errors here
        load_checkpoint(path, model=model)
    assert not isinstance(exc.value, AssertionError)  # a real error, not a silent success


def test_empty_checkpoint_file_fails(saved_checkpoint):
    path, model = saved_checkpoint
    path.write_bytes(b"")
    with pytest.raises(Exception):
        load_checkpoint(path, model=model)


def test_wrong_format_version_names_both_versions(saved_checkpoint):
    path, model = saved_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint format version 999, this build writes"):
        load_checkpoint(path, model=model)


def test_checkpoint_missing_a_parameter_is_rejected(saved_checkpoint):
    path, model = saved_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["model"]["blocks.0.mlp.fc1.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key|missing keys"):
        load_checkpoint(path, model=model)


def test_checkpoint_with_a_wrong_shape_is_rejected(saved_checkpoint, nano_config):
    """Loading a checkpoint into a differently shaped model must fail, not silently truncate."""
    path, _ = saved_checkpoint
    bigger = GPT(dataclasses.replace(nano_config, d_ff=128))
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        load_checkpoint(path, model=bigger)


def test_a_crash_during_save_leaves_the_previous_checkpoint_intact(saved_checkpoint, monkeypatch):
    """The failure this guards against: losing a long run to a crash during its own checkpoint."""
    path, model = saved_checkpoint
    good = path.read_bytes()
    cfg = TrainConfig(lr=1e-3)
    optimizer = build_optimizer(model, cfg)

    real_save = torch.save

    def exploding(payload, target, *a, **kw):
        real_save(payload, target, *a, **kw)
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", exploding)
    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint(path, model=model, optimizer=optimizer,
                        scheduler=build_scheduler(optimizer, cfg), step=2, tokens_seen=0,
                        config=cfg)
    assert path.read_bytes() == good


# --- unsupported hardware -----------------------------------------------------------------------


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_requesting_cuda_without_cuda_is_a_hard_error():
    """No silent fallback: a run that quietly uses the CPU reports throughput wrong by orders of
    magnitude and looks like it worked."""
    with pytest.raises(ConfigError, match="device 'cuda' requested but CUDA is not available"):
        resolve_device("cuda")


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS is available here")
def test_requesting_mps_without_mps_is_a_hard_error():
    with pytest.raises(ConfigError, match="MPS is not available"):
        resolve_device("mps")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.is_bf16_supported(),
    reason="needs a CUDA device without bf16",
)
def test_bf16_on_unsupported_cuda_names_the_requirement():
    with pytest.raises(ConfigError, match="compute capability 8.0"):
        validate_precision("bf16", torch.device("cuda"))


def test_strategies_needing_peers_are_rejected_in_one_process(nano_config, tmp_path):
    for strategy in ("ddp", "fsdp", "tp"):
        cfg = dataclasses.replace(
            experiment(nano_config, tmp_path),
            distributed=DistributedConfig(strategy=strategy),
        )
        with pytest.raises(ConfigError, match="needs more than one rank"):
            Trainer(cfg, out_dir=tmp_path, echo=False)


# --- CUDA out of memory -------------------------------------------------------------------------


def test_out_of_memory_is_reported_with_the_knobs_that_help(nano_config, tmp_path, monkeypatch):
    """The error *path* is what we own; the hardware is not, so the OOM is injected.

    The raw message says how many bytes failed to allocate, which tells you nothing actionable.
    This asserts the diagnostic names the current settings and the levers, in order.
    """
    trainer = Trainer(
        experiment(nano_config, tmp_path, micro_batch_size=4, grad_accum_steps=2),
        out_dir=tmp_path, echo=False,
    )

    def explode(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")

    # Patch `forward`, not the module itself: train_step legitimately calls `self.model.train()`.
    monkeypatch.setattr(trainer.raw_model, "forward", explode)
    with pytest.raises(torch.OutOfMemoryError) as exc:
        trainer.train_step()

    message = str(exc.value)
    assert "micro_batch_size=4" in message and "grad_accum_steps=2" in message
    assert "halve train.micro_batch_size" in message
    assert "distributed.strategy: fsdp" in message
    assert "Tried to allocate 20.00 GiB" in message  # the original cause is preserved
    assert isinstance(exc.value, RuntimeError)  # still a real failure, not downgraded


def test_out_of_memory_still_fails_the_run(nano_config, tmp_path, monkeypatch):
    """A better message must not become a swallowed error."""
    trainer = Trainer(experiment(nano_config, tmp_path), out_dir=tmp_path, echo=False)

    def explode(*args, **kwargs):
        raise torch.OutOfMemoryError("oom")

    monkeypatch.setattr(trainer.raw_model, "forward", explode)
    with pytest.raises(torch.OutOfMemoryError):
        trainer.train_step()
    assert trainer.step == 0  # the step did not complete


# --- a worker dying mid-run ---------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.slow
def test_a_dying_worker_fails_the_job_without_hanging():
    """The case most likely to hang a test suite, which is exactly why it exists.

    Rank 1 exits via `os._exit` (no cleanup, like a real crash) while rank 0 is inside an
    all-reduce. torchrun must notice, tear the job down, and exit non-zero -- within the timeout,
    not after the collective's own 30-minute default.
    """
    from tests.distributed.conftest import LOOPBACK_IFNAME, free_port

    script = Path(__file__).parent / "_dying_worker.py"
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes=1", "--nproc_per_node=2",
            "--master_addr=127.0.0.1", f"--master_port={free_port()}",
            str(script),
        ],
        env={**os.environ, "GLOO_SOCKET_IFNAME": LOOPBACK_IFNAME},
        capture_output=True,
        text=True,
        timeout=TIMEOUT,  # a hang fails the test here rather than stalling the session
    )

    assert result.returncode != 0, "a dead worker must fail the job, not be ignored"
    combined = result.stdout + result.stderr
    assert "exiting deliberately" in combined  # the casualty got as far as we intended
    # And the survivor must not have sailed on as though nothing happened.
    assert "finished unexpectedly" not in combined


@pytest.mark.distributed
def test_the_distributed_harness_reports_a_failing_rank(tmp_path):
    """The harness itself must surface a rank's traceback rather than time out silently --
    otherwise every distributed test failure looks identical to a hang."""
    from tests.distributed.conftest import run_distributed

    with pytest.raises(BaseException) as exc:
        run_distributed(_always_fails, 2, timeout=120)
    assert "deliberate failure" in str(exc.value)


def _always_fails(rank, world_size):
    raise RuntimeError("deliberate failure for the harness test")


# --- the CLI's own error handling ---------------------------------------------------------------
#
# Regression for ISSUES #18: `train.py` used to catch ConfigError only around config *loading*, so
# anything raised later -- in Trainer.__init__ or by --resume -- escaped as a bare traceback. The
# message was right; the presentation was not, and SCOPE.md section 10 asks for meaningful errors.


def write_config(path, **sections) -> None:
    lines = [
        "model: {vocab_size: 128, seq_len: 16, n_layers: 2, n_heads: 4, d_model: 32, d_ff: 64}",
        "data: {dataset: random, n_items: 64, seed: 0}",
    ]
    for name, body in sections.items():
        lines.append(f"{name}: {{{body}}}")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "sections, extra, fragment",
    [
        # Raised by resolve_strategy, inside Trainer.__init__.
        ({"distributed": "strategy: ddp"}, [], "needs more than one rank"),
        ({"distributed": "strategy: fsdp"}, [], "needs more than one rank"),
        ({"distributed": "strategy: tp"}, [], "needs more than one rank"),
        # Raised by TrainConfig.validate_global_batch.
        (
            {"train": "max_steps: 1, micro_batch_size: 4, grad_accum_steps: 1, "
                      "global_batch_size: 99, device: cpu"},
            [],
            "global_batch_size 99",
        ),
        # Raised by RMSNorm's constructor, via make_norm.
        ({}, ["--set", "model.norm_type=rmsnorm", "--set", "model.norm_backend=triton"],
         "Triton is not installed"),
    ],
)
def test_setup_errors_exit_cleanly_rather_than_raising(tmp_path, capsys, sections, extra, fragment):
    config = tmp_path / "c.yaml"
    write_config(config, **sections)

    assert train_main(["--config", str(config), "--out-dir", str(tmp_path / "run")] + extra) == 2

    error = capsys.readouterr().err
    assert "configuration error:" in error
    assert fragment in error


def test_missing_resume_target_exits_cleanly(tmp_path, capsys):
    config = tmp_path / "c.yaml"
    write_config(config, train="max_steps: 1, device: cpu")

    code = train_main(
        [
            "--config", str(config),
            "--out-dir", str(tmp_path / "run"),
            "--resume", str(tmp_path / "ghost.pt"),
        ]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "file not found:" in error and "checkpoint not found" in error


def test_a_failure_during_training_is_still_loud(tmp_path, monkeypatch):
    """The narrowness matters: only *setup* errors are turned into clean exits.

    A crash inside the training loop is a bug, and must still propagate as an exception rather
    than be flattened into a tidy exit code that a caller could mistake for a clean refusal.
    """
    from minimegatron.training.trainer import Trainer

    config = tmp_path / "c.yaml"
    write_config(config, train="max_steps: 1, device: cpu")

    def explode(self):
        raise RuntimeError("something genuinely broke mid-training")

    monkeypatch.setattr(Trainer, "fit", explode)
    with pytest.raises(RuntimeError, match="something genuinely broke"):
        train_main(["--config", str(config), "--out-dir", str(tmp_path / "run")])
