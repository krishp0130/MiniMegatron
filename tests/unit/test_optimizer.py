"""Tests for parameter groups and the learning-rate schedule."""

import math

import pytest
import torch

from minimegatron.model.transformer import GPT
from minimegatron.training.config import TrainConfig
from minimegatron.training.optimizer import (
    build_optimizer,
    build_scheduler,
    lr_at,
    param_groups,
    set_lr_for_step,
)

MAX_LR, MIN_LR, WARMUP, MAX_STEPS = 1.0, 0.1, 10, 110


def lr(step: int) -> float:
    return lr_at(step, MAX_LR, MIN_LR, WARMUP, MAX_STEPS)


# --- the schedule, as a pure function -----------------------------------------------------------


def test_warmup_starts_at_one_step_worth_and_reaches_max():
    assert lr(0) == pytest.approx(MAX_LR / WARMUP)  # never starts at exactly 0
    assert lr(WARMUP - 1) == pytest.approx(MAX_LR)


def test_cosine_begins_at_max_and_ends_at_min():
    assert lr(WARMUP) == pytest.approx(MAX_LR)
    assert lr(MAX_STEPS) == pytest.approx(MIN_LR)


def test_cosine_midpoint_is_the_average_of_max_and_min():
    midpoint = WARMUP + (MAX_STEPS - WARMUP) // 2
    assert lr(midpoint) == pytest.approx((MAX_LR + MIN_LR) / 2)


def test_schedule_is_monotonically_decreasing_after_warmup():
    values = [lr(s) for s in range(WARMUP, MAX_STEPS + 1)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_schedule_is_clamped_past_the_end():
    assert lr(MAX_STEPS + 50) == pytest.approx(MIN_LR)


def test_no_warmup_starts_the_cosine_immediately():
    assert lr_at(0, MAX_LR, MIN_LR, 0, 100) == pytest.approx(MAX_LR)


def test_degenerate_schedule_where_warmup_covers_the_run():
    """warmup_steps >= max_steps would divide by zero in the cosine; it must stay at max_lr."""
    assert lr_at(10, MAX_LR, MIN_LR, 10, 10) == pytest.approx(MAX_LR)
    assert lr_at(50, MAX_LR, MIN_LR, 100, 100) == pytest.approx(MAX_LR * 51 / 100)


def test_cosine_matches_the_closed_form():
    step = 60
    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    expected = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))
    assert lr(step) == pytest.approx(expected)


# --- parameter groups ---------------------------------------------------------------------------


def test_every_parameter_lands_in_exactly_one_group(nano_config):
    model = GPT(nano_config)
    groups = param_groups(model, weight_decay=0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))  # no parameter in two groups
    assert set(ids) == {id(p) for p in model.parameters()}  # and none missing


def test_only_multidimensional_weights_are_decayed(nano_config):
    model = GPT(nano_config)
    decay, no_decay = param_groups(model, weight_decay=0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])  # matmul weights and embeddings
    assert all(p.dim() == 1 for p in no_decay["params"])  # biases and norm gamma/beta


def test_tied_weight_is_decayed_only_once(nano_config):
    """A tied embedding/LM head is one tensor; decaying it twice would be a silent 2x."""
    model = GPT(nano_config)
    decay, _ = param_groups(model, weight_decay=0.1)
    assert sum(1 for p in decay["params"] if p is model.tok_emb.weight) == 1


def test_norm_and_bias_parameters_are_present_in_the_undecayed_group(nano_config):
    model = GPT(nano_config)
    _, no_decay = param_groups(model, weight_decay=0.1)
    assert any(p is model.norm_f.weight for p in no_decay["params"])
    assert any(p is model.blocks[0].attn.qkv.bias for p in no_decay["params"])


# --- wiring -------------------------------------------------------------------------------------


def test_scheduler_drives_the_optimizer_learning_rate(nano_config):
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    model = GPT(nano_config)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    optimizer.step()  # marks the correct optimizer-then-scheduler order, silencing torch's warning

    for step in range(12):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr(step))
        scheduler.step()


def test_scheduler_state_survives_a_round_trip(nano_config):
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    model = GPT(nano_config)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    optimizer.step()
    for _ in range(25):
        scheduler.step()
    state = scheduler.state_dict()

    fresh_optimizer = build_optimizer(model, cfg)
    fresh = build_scheduler(fresh_optimizer, cfg)
    fresh.load_state_dict(state)
    assert fresh.last_epoch == 25
    # Documents the subtlety that `set_lr_for_step` exists for: load_state_dict restores
    # last_epoch but leaves the optimizer on the rate its constructor computed for step 0.
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(lr(0))
    fresh.step()  # the next step() does put it back on the curve
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(lr(26))


def test_optimizer_uses_the_configured_betas(nano_config):
    cfg = TrainConfig(beta1=0.8, beta2=0.99)
    optimizer = build_optimizer(GPT(nano_config), cfg)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.99)


def test_optimizer_actually_changes_the_weights(nano_config):
    model = GPT(nano_config)
    cfg = TrainConfig(lr=1e-2)
    optimizer = build_optimizer(model, cfg)
    before = model.blocks[0].mlp.fc1.weight.detach().clone()

    idx = torch.randint(0, nano_config.vocab_size, (2, 8))
    _, loss = model(idx, idx)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.blocks[0].mlp.fc1.weight)


def test_set_lr_for_step_uses_the_live_schedule(nano_config):
    """On resume the learning rate must come from *this* run's schedule, not the saved one."""
    model = GPT(nano_config)
    old_cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=5, max_steps=50)
    optimizer = build_optimizer(model, old_cfg)

    new_cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=5, max_steps=200)  # extended run
    set_lr_for_step(optimizer, new_cfg, step=20)

    expected = lr_at(20, new_cfg.lr, new_cfg.min_lr, new_cfg.warmup_steps, new_cfg.max_steps)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    # The two schedules genuinely disagree at this step, so the test has teeth.
    assert expected != pytest.approx(
        lr_at(20, old_cfg.lr, old_cfg.min_lr, old_cfg.warmup_steps, old_cfg.max_steps)
    )
