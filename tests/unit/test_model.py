"""Tests for the full GPT model: shapes, parameter accounting, initialization, and gradients."""

import dataclasses
import math

import pytest
import torch

from minimegatron.experiment import load_config
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import INIT_STD, GPT
from tests.conftest import REPO_ROOT

PRESETS = ("tiny", "small", "medium")


def preset_config(name: str) -> ModelConfig:
    return load_config(REPO_ROOT / "configs" / f"{name}.yaml").model


# --- shapes and the forward contract ------------------------------------------------------------


@pytest.mark.parametrize("B, T", [(3, 8), (1, 1), (2, 16)])
def test_logits_shape(nano_config, B, T):
    model = GPT(nano_config)
    idx = torch.randint(0, nano_config.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, nano_config.vocab_size)
    assert loss is None  # no targets given


def test_loss_is_a_finite_scalar(nano_config):
    model = GPT(nano_config)
    idx = torch.randint(0, nano_config.vocab_size, (2, 8))
    targets = torch.randint(0, nano_config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    assert loss.shape == ()  # reduced over every one of the B*T positions
    assert torch.isfinite(loss)


def test_sequence_longer_than_seq_len_raises(nano_config):
    model = GPT(nano_config)
    idx = torch.randint(0, nano_config.vocab_size, (1, nano_config.seq_len + 1))
    with pytest.raises(ValueError, match="exceeds the model's seq_len"):
        model(idx)


# --- parameter accounting -----------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
def test_preset_param_count_matches_formula(preset, norm_type):
    """The real model's parameter count must equal the analytic formula, exactly."""
    cfg = dataclasses.replace(preset_config(preset), norm_type=norm_type)
    model = GPT(cfg)
    assert model.num_params() == cfg.num_params()


@pytest.mark.parametrize(
    "overrides",
    [{}, {"tie_embeddings": False}, {"bias": False}, {"norm_type": "rmsnorm"}, {"dropout": 0.1}],
)
def test_param_count_matches_formula_for_variants(nano_config, overrides):
    cfg = dataclasses.replace(nano_config, **overrides)
    assert GPT(cfg).num_params() == cfg.num_params()


@pytest.mark.parametrize(
    "preset, non_embedding",
    [("tiny", 1_779_840), ("small", 25_220_096), ("medium", 85_056_000)],
)
def test_non_embedding_param_counts(preset, non_embedding):
    """The figures published in the milestone doc's preset table."""
    cfg = preset_config(preset)
    model = GPT(cfg)
    assert model.num_params(include_embeddings=False) == non_embedding
    assert cfg.num_params(include_embeddings=False) == non_embedding


def test_tiny_is_dominated_by_the_token_embedding():
    cfg = preset_config("tiny")
    embedding_share = (cfg.vocab_size * cfg.d_model) / cfg.num_params()
    assert embedding_share > 0.80  # 84%: why scaling-law papers quote non-embedding counts


# --- weight tying -------------------------------------------------------------------------------


def test_tied_head_is_the_same_tensor(nano_config):
    model = GPT(nano_config)
    assert nano_config.tie_embeddings
    assert model.lm_head.weight is model.tok_emb.weight
    # Counted once by `parameters()`, which is what makes the analytic formula's `lm_head = 0`
    # term agree with a real numel() sum.
    assert sum(p.numel() for p in model.parameters()) == nano_config.num_params()


def test_tied_head_accumulates_gradient_from_both_uses(nano_config):
    """A tied weight is used twice per forward, so one gradient must reflect both paths."""
    tied = GPT(nano_config)
    untied = GPT(dataclasses.replace(nano_config, tie_embeddings=False))
    assert untied.lm_head.weight is not untied.tok_emb.weight

    idx = torch.randint(0, nano_config.vocab_size, (2, 8))
    _, loss = tied(idx, idx)
    loss.backward()
    grad = tied.tok_emb.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    # The head path touches every row of the vocabulary; the embedding path only the rows that
    # appear in `idx`. A gradient that is non-zero on unused rows proves both contributed.
    unused = sorted(set(range(nano_config.vocab_size)) - set(idx.flatten().tolist()))
    assert grad[unused].abs().sum() > 0


# --- initialization -----------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["tiny", "small"])
def test_loss_at_init_is_about_ln_vocab(preset):
    """Near-uniform logits at init give loss = -log(1/V) = ln V.

    A large deviation means broken init, a broken loss, or misaligned targets -- all of which can
    otherwise go unnoticed while training still "kind of works".
    """
    cfg = preset_config(preset)
    torch.manual_seed(0)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    targets = torch.randint(0, cfg.vocab_size, (4, 32))
    _, loss = model(idx, targets)
    assert loss.item() == pytest.approx(math.log(cfg.vocab_size), abs=0.3)


def test_residual_projections_are_scaled_by_depth():
    """out_proj and fc2 write into the residual stream, so they init at 0.02/sqrt(2L)."""
    cfg = preset_config("tiny")  # L = 4, d = 192: 36,864 weights per projection
    torch.manual_seed(0)
    model = GPT(cfg)
    expected = INIT_STD / math.sqrt(2 * cfg.n_layers)

    for block in model.blocks:
        assert block.attn.out_proj.weight.std().item() == pytest.approx(expected, rel=0.05)
        assert block.mlp.fc2.weight.std().item() == pytest.approx(expected, rel=0.05)

    # Non-residual projections keep the unscaled 0.02.
    for block in model.blocks:
        assert block.attn.qkv.weight.std().item() == pytest.approx(INIT_STD, rel=0.05)
        assert block.mlp.fc1.weight.std().item() == pytest.approx(INIT_STD, rel=0.05)


def test_biases_and_norms_init_to_the_identity(nano_config):
    model = GPT(nano_config)
    assert torch.all(model.blocks[0].attn.qkv.bias == 0)
    assert torch.all(model.blocks[0].mlp.fc1.bias == 0)
    assert torch.all(model.norm_f.weight == 1)
    assert torch.all(model.norm_f.bias == 0)


def test_residual_stream_scale_is_stable_with_depth(nano_config):
    """The point of the 1/sqrt(2L) scaling: activations should not blow up through the stack."""
    cfg = dataclasses.replace(nano_config, n_layers=8, d_model=64, n_heads=4, d_ff=256)
    torch.manual_seed(0)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))

    x = model.drop(model.tok_emb(idx) + model.pos_emb(torch.arange(16)))
    first = x.std().item()
    for block in model.blocks:
        x = block(x)
    assert x.std().item() < 4 * first  # grows slowly, not geometrically


# --- gradients and backend equivalence ----------------------------------------------------------


def test_every_parameter_receives_a_finite_gradient(nano_config):
    model = GPT(dataclasses.replace(nano_config, dropout=0.1))
    idx = torch.randint(0, nano_config.vocab_size, (2, 8))
    _, loss = model(idx, idx)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


def test_reference_and_sdpa_models_agree(nano_config):
    ref = GPT(dataclasses.replace(nano_config, attention_backend="reference")).eval()
    sdpa = GPT(dataclasses.replace(nano_config, attention_backend="sdpa")).eval()
    sdpa.load_state_dict(ref.state_dict())  # identical weights; the mask buffer is non-persistent

    idx = torch.randint(0, nano_config.vocab_size, (3, 12))
    with torch.no_grad():
        ref_logits, ref_loss = ref(idx, idx)
        sdpa_logits, sdpa_loss = sdpa(idx, idx)
    torch.testing.assert_close(ref_logits, sdpa_logits)
    torch.testing.assert_close(ref_loss, sdpa_loss)


def test_ignore_index_excludes_positions_from_the_loss(nano_config):
    """-100 targets are reserved for padding; they must not contribute to the loss."""
    model = GPT(nano_config).eval()
    idx = torch.randint(0, nano_config.vocab_size, (2, 8))
    targets = idx.clone()
    masked = targets.clone()
    masked[:, 4:] = -100

    with torch.no_grad():
        _, full = model(idx, targets)
        _, partial = model(idx, masked)
        _, first_half = model(idx[:, :4], targets[:, :4])
    assert not torch.isclose(full, partial)
    torch.testing.assert_close(partial, first_half)  # same positions, same mean
