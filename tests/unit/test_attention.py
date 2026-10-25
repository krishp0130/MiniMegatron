import dataclasses

import pytest
import torch

from minimegatron.model.attention import CausalSelfAttention, split_qkv

BACKENDS = ["reference", "sdpa"]


def make_attention(cfg, backend="reference", seed=0):
    torch.manual_seed(seed)
    return CausalSelfAttention(dataclasses.replace(cfg, attention_backend=backend))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("B, T", [(1, 1), (3, 7), (2, 16)])
def test_output_shape(nano_config, backend, B, T):
    attn = make_attention(nano_config, backend)
    x = torch.randn(B, T, nano_config.d_model)
    assert attn(x).shape == (B, T, nano_config.d_model)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("t", [0, 5, 15])
def test_causality(nano_config, backend, t):
    """Changing the token at position t must not affect outputs at positions < t."""
    attn = make_attention(nano_config, backend).eval()
    x = torch.randn(2, nano_config.seq_len, nano_config.d_model)
    x_perturbed = x.clone()
    x_perturbed[:, t] += torch.randn(nano_config.d_model)

    y, y_perturbed = attn(x), attn(x_perturbed)
    torch.testing.assert_close(y[:, :t], y_perturbed[:, :t])  # past: unchanged
    assert not torch.allclose(y[:, t:], y_perturbed[:, t:])  # present and future: changed


def test_sequence_longer_than_seq_len_raises(nano_config):
    attn = make_attention(nano_config)
    with pytest.raises(ValueError, match="exceeds the model's seq_len"):
        attn(torch.randn(1, nano_config.seq_len + 1, nano_config.d_model))


def test_state_dict_keys_identical_across_backends(nano_config):
    # The mask buffer is non-persistent, so checkpoints don't depend on the backend.
    expected = {"qkv.weight", "qkv.bias", "out_proj.weight", "out_proj.bias"}
    for backend in BACKENDS:
        assert set(make_attention(nano_config, backend).state_dict()) == expected


def test_qkv_layout_splits_into_whole_heads(nano_config):
    """The first half of the fused qkv output must hold exactly the first half of the heads.

    This is the property column-parallel tensor parallelism needs (D6).
    """
    attn = make_attention(nano_config)
    h, d = nano_config.n_heads, nano_config.d_model
    qkv = attn.qkv(torch.randn(2, 5, d))

    q, k, v = split_qkv(qkv, h)
    q_half, k_half, v_half = split_qkv(qkv[..., : 3 * d // 2], h // 2)
    torch.testing.assert_close(q_half, q[:, : h // 2])
    torch.testing.assert_close(k_half, k[:, : h // 2])
    torch.testing.assert_close(v_half, v[:, : h // 2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_dropout_active_only_in_training(nano_config, backend):
    attn = make_attention(dataclasses.replace(nano_config, dropout=0.5), backend)
    x = torch.randn(2, 8, nano_config.d_model)

    attn.train()
    assert not torch.allclose(attn(x), attn(x))  # fresh dropout masks each call
    attn.eval()
    torch.testing.assert_close(attn(x), attn(x), rtol=0, atol=0)  # deterministic
