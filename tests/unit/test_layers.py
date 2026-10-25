import dataclasses

import pytest
import torch
import torch.nn as nn

from minimegatron.model.layers import MLP, Block, RMSNorm, make_norm


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize(
    "norm_type, bias, expected_type, expected_params",
    [
        ("layernorm", True, nn.LayerNorm, 64),  # gamma + beta
        ("layernorm", False, nn.LayerNorm, 32),  # gamma only
        ("rmsnorm", True, RMSNorm, 32),  # RMSNorm never has beta
    ],
)
def test_make_norm(nano_config, norm_type, bias, expected_type, expected_params):
    norm = make_norm(dataclasses.replace(nano_config, norm_type=norm_type, bias=bias))
    assert isinstance(norm, expected_type)
    assert num_params(norm) == expected_params


def test_mlp_shapes(nano_config):
    mlp = MLP(nano_config)
    assert mlp.fc1.out_features == nano_config.d_ff
    assert mlp(torch.randn(3, 7, nano_config.d_model)).shape == (3, 7, nano_config.d_model)


def test_mlp_is_position_wise(nano_config):
    """The MLP never mixes tokens: running it on one token alone gives the same result."""
    torch.manual_seed(0)
    mlp = MLP(nano_config)
    x = torch.randn(2, 10, nano_config.d_model)
    y = mlp(x)
    for t in range(10):
        torch.testing.assert_close(mlp(x[:, t : t + 1]), y[:, t : t + 1])


@pytest.mark.parametrize(
    "overrides",
    [{}, {"norm_type": "rmsnorm"}, {"bias": False}, {"d_ff": 100}],
)
def test_block_param_count_matches_config_formula(nano_config, overrides):
    cfg = dataclasses.replace(nano_config, **overrides)
    # One block's share of the formula = num_params(L=2) - num_params(L=1).
    per_layer = (
        dataclasses.replace(cfg, n_layers=2).num_params()
        - dataclasses.replace(cfg, n_layers=1).num_params()
    )
    assert num_params(Block(cfg)) == per_layer


def test_block_shape(nano_config):
    assert Block(nano_config)(torch.randn(3, 7, 32)).shape == (3, 7, 32)


def test_block_is_identity_when_sublayer_outputs_are_zero(nano_config):
    """With the output projections zeroed, only the residual path remains: block(x) == x.

    This is the pre-norm property: the residual stream passes through untouched by any norm.
    """
    block = Block(nano_config)
    with torch.no_grad():
        for layer in (block.attn.out_proj, block.mlp.fc2):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(2, 9, nano_config.d_model)
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_block_is_causal(nano_config):
    torch.manual_seed(0)
    block = Block(nano_config).eval()
    x = torch.randn(2, nano_config.seq_len, nano_config.d_model)
    x_perturbed = x.clone()
    x_perturbed[:, 6] += 1.0
    torch.testing.assert_close(block(x)[:, :6], block(x_perturbed)[:, :6])
    assert not torch.allclose(block(x)[:, 6:], block(x_perturbed)[:, 6:])


def test_block_gradients_reach_every_parameter(nano_config):
    block = Block(nano_config)
    block(torch.randn(2, 8, nano_config.d_model)).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
