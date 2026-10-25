"""The optimized SDPA backend must agree with the explicit reference implementation."""

import dataclasses

import pytest
import torch

from minimegatron.model.attention import CausalSelfAttention, reference_attention
from minimegatron.model.config import ModelConfig

CONFIGS = {
    "nano": ModelConfig(vocab_size=128, seq_len=16, n_layers=2, n_heads=4, d_model=32, d_ff=64),
    # head_dim 48 (tiny's shape): not a power of two
    "hd48": ModelConfig(vocab_size=128, seq_len=64, n_layers=1, n_heads=4, d_model=192),
    "one_head": ModelConfig(vocab_size=128, seq_len=33, n_layers=1, n_heads=1, d_model=24),
}


def make_pair(cfg, seed=0):
    """Reference and SDPA attention modules with identical weights."""
    torch.manual_seed(seed)
    ref = CausalSelfAttention(dataclasses.replace(cfg, attention_backend="reference"))
    sdpa = CausalSelfAttention(dataclasses.replace(cfg, attention_backend="sdpa"))
    sdpa.load_state_dict(ref.state_dict())
    return ref, sdpa


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("B", [1, 3])
def test_forward_matches(name, B):
    cfg = CONFIGS[name]
    ref, sdpa = make_pair(cfg)
    x = torch.randn(B, cfg.seq_len, cfg.d_model)
    torch.testing.assert_close(sdpa(x), ref(x))


@pytest.mark.parametrize("name", CONFIGS)
def test_backward_matches(name):
    cfg = CONFIGS[name]
    ref, sdpa = make_pair(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    x_ref, x_sdpa = x.clone().requires_grad_(), x.clone().requires_grad_()

    ref(x_ref).square().sum().backward()
    sdpa(x_sdpa).square().sum().backward()

    torch.testing.assert_close(x_sdpa.grad, x_ref.grad)
    for (name_, p_ref), (_, p_sdpa) in zip(ref.named_parameters(), sdpa.named_parameters()):
        torch.testing.assert_close(p_sdpa.grad, p_ref.grad, msg=lambda m: f"{name_}: {m}")


@pytest.mark.parametrize("scale", [1e-4, 1e2])
def test_extreme_input_scales(scale):
    """Tiny inputs give near-uniform attention; huge ones saturate softmax to one-hot."""
    cfg = CONFIGS["nano"]
    ref, sdpa = make_pair(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.d_model) * scale
    y_ref = ref(x)
    assert torch.isfinite(y_ref).all()
    torch.testing.assert_close(sdpa(x), y_ref)


def test_reference_attention_gradcheck():
    """Autograd gradients of the reference math match finite differences (float64)."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 5, 3, dtype=torch.float64, requires_grad=True) for _ in range(3))
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    assert torch.autograd.gradcheck(lambda q, k, v: reference_attention(q, k, v, mask), (q, k, v))


def test_module_gradcheck():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CONFIGS["nano"]).double()
    x = torch.randn(1, 6, 32, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(attn, (x,))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_mps_matches_cpu(backend):
    cfg = dataclasses.replace(CONFIGS["hd48"], attention_backend=backend)
    torch.manual_seed(0)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)
    y_cpu = attn(x)
    y_mps = attn.to("mps")(x.to("mps")).cpu()
    torch.testing.assert_close(y_mps, y_cpu, atol=1e-5, rtol=1e-4)
