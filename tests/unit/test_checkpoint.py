"""Tests for checkpoint contents, atomic writes, and RNG capture."""

import random

import numpy as np
import pytest
import torch

from minimegatron.model.transformer import GPT
from minimegatron.training.checkpoint import (
    FORMAT_VERSION,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)
from minimegatron.training.config import TrainConfig
from minimegatron.training.optimizer import build_optimizer, build_scheduler


@pytest.fixture
def run(nano_config):
    """A model that has taken a few real optimizer steps, so its state is non-trivial."""
    torch.manual_seed(0)
    cfg = TrainConfig(lr=1e-2, warmup_steps=5, max_steps=50)
    model = GPT(nano_config)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    for _ in range(3):
        idx = torch.randint(0, nano_config.vocab_size, (2, 8))
        _, loss = model(idx, idx)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return cfg, model, optimizer, scheduler


def fresh(nano_config, cfg):
    torch.manual_seed(999)  # deliberately different, so a failed restore is obvious
    model = GPT(nano_config)
    optimizer = build_optimizer(model, cfg)
    return model, optimizer, build_scheduler(optimizer, cfg)


# --- round trip ---------------------------------------------------------------------------------


def test_round_trip_restores_weights_optimizer_and_schedule(nano_config, run, tmp_path):
    cfg, model, optimizer, scheduler = run
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=3,
        tokens_seen=384,
        config=cfg,
    )

    new_model, new_optimizer, new_scheduler = fresh(nano_config, cfg)
    assert not torch.equal(new_model.tok_emb.weight, model.tok_emb.weight)

    meta = load_checkpoint(
        path, model=new_model, optimizer=new_optimizer, scheduler=new_scheduler
    )

    assert meta["step"] == 3 and meta["tokens_seen"] == 384
    for (name, a), (_, b) in zip(new_model.state_dict().items(), model.state_dict().items()):
        assert torch.equal(a, b), f"{name} was not restored"
    # Adam's m and v, not just the weights: resuming without them restarts the moment estimates.
    assert new_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    assert new_scheduler.last_epoch == scheduler.last_epoch


def test_optimizer_moments_are_restored_exactly(nano_config, run, tmp_path):
    cfg, model, optimizer, scheduler = run
    save_checkpoint(
        tmp_path / "c.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        step=3, tokens_seen=0, config=cfg,
    )
    new_model, new_optimizer, new_scheduler = fresh(nano_config, cfg)
    load_checkpoint(tmp_path / "c.pt", model=new_model, optimizer=new_optimizer,
                    scheduler=new_scheduler)

    for key, state in optimizer.state_dict()["state"].items():
        restored = new_optimizer.state_dict()["state"][key]
        torch.testing.assert_close(restored["exp_avg"], state["exp_avg"])  # m
        torch.testing.assert_close(restored["exp_avg_sq"], state["exp_avg_sq"])  # v
        assert restored["step"] == state["step"]


def test_resolved_config_is_stored(nano_config, run, tmp_path):
    cfg, model, optimizer, scheduler = run
    save_checkpoint(
        tmp_path / "c.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        step=1, tokens_seen=0, config=cfg,
    )
    meta = load_checkpoint(tmp_path / "c.pt", model=model)
    assert meta["config"]["lr"] == cfg.lr  # a checkpoint says what produced it


def test_checkpoint_loads_without_arbitrary_unpickling(nano_config, run, tmp_path):
    """Every field must survive torch.load(weights_only=True), which refuses to unpickle
    arbitrary objects. NumPy's RNG state contains a uint32 array that would otherwise fail."""
    cfg, model, optimizer, scheduler = run
    save_checkpoint(
        tmp_path / "c.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        step=1, tokens_seen=0, config=cfg,
    )
    payload = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=True)
    assert payload["format_version"] == FORMAT_VERSION
    assert set(payload["rng"]) >= {"python", "numpy", "torch_cpu"}


def test_mismatched_format_version_is_rejected(nano_config, run, tmp_path):
    cfg, model, optimizer, scheduler = run
    save_checkpoint(
        tmp_path / "c.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        step=1, tokens_seen=0, config=cfg,
    )
    payload = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=True)
    payload["format_version"] = FORMAT_VERSION + 1
    torch.save(payload, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="checkpoint format version"):
        load_checkpoint(tmp_path / "c.pt", model=model)


def test_missing_checkpoint_raises(nano_config, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pt", model=GPT(nano_config))


# --- atomic writes ------------------------------------------------------------------------------


def test_save_leaves_no_temporary_file(nano_config, run, tmp_path):
    cfg, model, optimizer, scheduler = run
    save_checkpoint(
        tmp_path / "c.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        step=1, tokens_seen=0, config=cfg,
    )
    assert (tmp_path / "c.pt").is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_crash_mid_write_preserves_the_previous_checkpoint(
    nano_config, run, tmp_path, monkeypatch
):
    """The failure this guards against: losing a long run to a crash during its own checkpoint."""
    cfg, model, optimizer, scheduler = run
    path = tmp_path / "c.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                    step=1, tokens_seen=100, config=cfg)
    good = path.read_bytes()

    real_save = torch.save

    def exploding_save(payload, target, *a, **kw):
        real_save(payload, target, *a, **kw)  # a partial write lands in the temp file
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                        step=2, tokens_seen=200, config=cfg)

    assert path.read_bytes() == good  # untouched: the rename never happened
    monkeypatch.undo()
    assert load_checkpoint(path, model=model)["step"] == 1


def test_save_creates_missing_directories(nano_config, run, tmp_path):
    cfg, model, optimizer, scheduler = run
    path = save_checkpoint(
        tmp_path / "deep" / "nested" / "c.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, step=1, tokens_seen=0, config=cfg,
    )
    assert path.is_file()


# --- RNG capture --------------------------------------------------------------------------------


def test_rng_state_round_trip_reproduces_every_stream():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = rng_state(torch.device("cpu"))

    expected = (random.random(), np.random.rand(), torch.rand(3))
    for _ in range(5):  # advance all three well past the saved point
        random.random(), np.random.rand(), torch.rand(10)

    restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2])


def test_dropout_masks_repeat_after_an_rng_restore(nano_config):
    """The practical reason RNG state is in the checkpoint at all."""
    import dataclasses

    model = GPT(dataclasses.replace(nano_config, dropout=0.5)).train()
    idx = torch.randint(0, nano_config.vocab_size, (2, 8))

    state = rng_state(torch.device("cpu"))
    first, _ = model(idx)
    restore_rng_state(state)
    second, _ = model(idx)
    torch.testing.assert_close(first, second)
