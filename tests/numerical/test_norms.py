"""Our RMSNorm is the reference for the Milestone 5 Triton kernel, so it gets checked hard."""

import pytest
import torch
import torch.nn.functional as F

from minimegatron.model.layers import NORM_EPS, RMSNorm, rms_norm

# Non-power-of-two widths matter: Triton kernels tile in powers of two and must mask the tail.
SHAPES = [(1, 1), (4, 7), (2, 3, 192), (8, 1000), (2, 16, 768)]


def random_weight(dim: int) -> torch.Tensor:
    return 1.0 + 0.1 * torch.randn(dim)


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch_rmsnorm(shape):
    torch.manual_seed(0)
    x = torch.randn(shape)
    ours = RMSNorm(shape[-1])
    theirs = torch.nn.RMSNorm(shape[-1], eps=NORM_EPS)
    with torch.no_grad():
        ours.weight.copy_(random_weight(shape[-1]))
        theirs.weight.copy_(ours.weight)
    torch.testing.assert_close(ours(x), theirs(x))


@pytest.mark.parametrize("shape", SHAPES)
def test_bf16_matches_fp32_computation(shape):
    torch.manual_seed(0)
    x, w = torch.randn(shape), random_weight(shape[-1])
    expected = rms_norm(x, w).bfloat16()
    actual = rms_norm(x.bfloat16(), w.bfloat16())
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)  # bf16 default tolerances


def test_output_has_unit_rms():
    x = torch.randn(4, 256) * 37.0
    y = rms_norm(x, torch.ones(256))
    torch.testing.assert_close(y.pow(2).mean(dim=-1).sqrt(), torch.ones(4))


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e4])
def test_extreme_scales_stay_finite(scale):
    """eps keeps all-zero and tiny inputs from dividing by zero."""
    y = rms_norm(torch.randn(3, 64) * scale, torch.ones(64))
    assert torch.isfinite(y).all()


def test_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7, dtype=torch.float64, requires_grad=True)
    w = random_weight(7).double().requires_grad_()
    assert torch.autograd.gradcheck(rms_norm, (x, w))


def test_exact_and_tanh_gelu_differ_by_less_than_1e3():
    # The M1 design doc claims this; keep the claim tested.
    u = torch.linspace(-10, 10, 100_001, dtype=torch.float64)
    diff = (F.gelu(u) - F.gelu(u, approximate="tanh")).abs().max()
    assert diff < 1e-3
