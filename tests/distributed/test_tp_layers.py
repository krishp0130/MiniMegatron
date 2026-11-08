"""ColumnParallelLinear and RowParallelLinear against plain `nn.Linear`."""

import pytest
import torch
import torch.nn as nn

from minimegatron.config import ConfigError
from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    group_rank,
    group_size,
)
from tests.distributed.conftest import run_distributed

pytestmark = pytest.mark.distributed

IN, OUT, BATCH = 8, 12, 4


def _reference_linear(bias=True):
    """Identical on every rank, because every rank seeds the same way."""
    torch.manual_seed(0)
    return nn.Linear(IN, OUT, bias=bias)


def _column_vs_linear(rank, world_size, bias):
    init_distributed(DistributedConfig(backend="gloo"))
    linear = _reference_linear(bias)
    parallel = ColumnParallelLinear.from_linear(linear, gather_output=True)

    torch.manual_seed(1)
    x = torch.randn(BATCH, IN)
    xr, xp = x.clone().requires_grad_(True), x.clone().requires_grad_(True)

    reference = linear(xr)
    out = parallel(xp)
    reference.sum().backward()
    out.sum().backward()

    size, local = group_size(), group_rank()
    return {
        "out": out.detach().clone(),
        "reference": reference.detach().clone(),
        "input_grad": xp.grad.clone(),
        "reference_input_grad": xr.grad.clone(),
        "weight_grad": parallel.weight.grad.clone(),
        "reference_weight_grad_slice": linear.weight.grad.chunk(size, dim=0)[local].clone(),
        "local_out_features": parallel.out_features_per_rank,
        "weight_shape": tuple(parallel.weight.shape),
    }


@pytest.mark.parametrize("bias", [True, False])
def test_column_parallel_matches_nn_linear(bias):
    """Gathered output must equal the unsharded layer's, and so must the input gradient.

    The input gradient is the interesting half: each rank computes only a partial
    `dL/dX = (dL/dY_i) W_i^T`, so this passes only if `f`'s backward all-reduce is right.
    """
    results = run_distributed(_column_vs_linear, 2, args=(bias,))
    for rank in (0, 1):
        r = results[rank]
        torch.testing.assert_close(r["out"], r["reference"])
        torch.testing.assert_close(r["input_grad"], r["reference_input_grad"])
        # Each rank's weight gradient is the matching slice of the reference's.
        torch.testing.assert_close(r["weight_grad"], r["reference_weight_grad_slice"])


def test_column_parallel_splits_the_output_dimension():
    results = run_distributed(_column_vs_linear, 2, args=(True,))
    for rank in (0, 1):
        assert results[rank]["local_out_features"] == OUT // 2
        assert results[rank]["weight_shape"] == (OUT // 2, IN)  # full input, half the outputs


def _row_vs_linear(rank, world_size, bias):
    init_distributed(DistributedConfig(backend="gloo"))
    linear = _reference_linear(bias)
    # input_is_parallel=False makes the layer scatter the replicated input itself, which is what
    # lets it be compared against nn.Linear on the same full input.
    parallel = RowParallelLinear.from_linear(linear, input_is_parallel=False)

    torch.manual_seed(1)
    x = torch.randn(BATCH, IN)
    xr, xp = x.clone().requires_grad_(True), x.clone().requires_grad_(True)

    reference = linear(xr)
    out = parallel(xp)
    reference.sum().backward()
    out.sum().backward()

    size, local = group_size(), group_rank()
    return {
        "out": out.detach().clone(),
        "reference": reference.detach().clone(),
        "input_grad": xp.grad.clone(),
        "reference_input_grad": xr.grad.clone(),
        "weight_grad": parallel.weight.grad.clone(),
        "reference_weight_grad_slice": linear.weight.grad.chunk(size, dim=1)[local].clone(),
        "bias": None if parallel.bias is None else parallel.bias.detach().clone(),
        "reference_bias": None if linear.bias is None else linear.bias.detach().clone(),
        "bias_grad": None if parallel.bias is None else parallel.bias.grad.clone(),
        "reference_bias_grad": None if linear.bias is None else linear.bias.grad.clone(),
        "weight_shape": tuple(parallel.weight.shape),
    }


@pytest.mark.parametrize("bias", [True, False])
def test_row_parallel_matches_nn_linear(bias):
    results = run_distributed(_row_vs_linear, 2, args=(bias,))
    for rank in (0, 1):
        r = results[rank]
        torch.testing.assert_close(r["out"], r["reference"])
        torch.testing.assert_close(r["input_grad"], r["reference_input_grad"])
        torch.testing.assert_close(r["weight_grad"], r["reference_weight_grad_slice"])


def test_row_parallel_bias_is_replicated_and_added_once():
    """The classic TP bug: a bias added before the all-reduce is added tp_size times.

    The output comparison above would already catch it, but this pins the cause rather than the
    symptom -- the bias is a full, unsplit copy on every rank, and its gradient is the reference's
    in full (not a slice), because every rank's partial output passes through the same bias.
    """
    results = run_distributed(_row_vs_linear, 2, args=(True,))
    for rank in (0, 1):
        r = results[rank]
        assert r["bias"].shape == (OUT,)  # replicated, not split
        torch.testing.assert_close(r["bias"], r["reference_bias"])
        torch.testing.assert_close(r["bias_grad"], r["reference_bias_grad"])


def test_row_parallel_splits_the_input_dimension():
    results = run_distributed(_row_vs_linear, 2, args=(True,))
    for rank in (0, 1):
        assert results[rank]["weight_shape"] == (OUT, IN // 2)  # all outputs, half the inputs


def _column_then_row(rank, world_size):
    """The pairing used by the MLP and by attention: no gather in between."""
    init_distributed(DistributedConfig(backend="gloo"))
    torch.manual_seed(0)
    first, second = nn.Linear(IN, OUT), nn.Linear(OUT, IN)

    column = ColumnParallelLinear.from_linear(first)  # gather_output=False
    row = RowParallelLinear.from_linear(second)  # input_is_parallel=True

    torch.manual_seed(1)
    x = torch.randn(BATCH, IN)
    xr, xp = x.clone().requires_grad_(True), x.clone().requires_grad_(True)

    reference = second(torch.nn.functional.gelu(first(xr)))
    out = row(torch.nn.functional.gelu(column(xp)))
    reference.sum().backward()
    out.sum().backward()

    return {
        "out": out.detach().clone(),
        "reference": reference.detach().clone(),
        "input_grad": xp.grad.clone(),
        "reference_input_grad": xr.grad.clone(),
        "hidden_width": column(xp).shape[-1],
    }


def test_column_then_row_needs_no_intermediate_gather():
    """Column parallel leaves its output split; row parallel wants its input split. Chaining them
    means the hidden state is never gathered -- one all-reduce per pair, not per matmul."""
    results = run_distributed(_column_then_row, 2)
    for rank in (0, 1):
        r = results[rank]
        torch.testing.assert_close(r["out"], r["reference"])
        torch.testing.assert_close(r["input_grad"], r["reference_input_grad"])
        assert r["hidden_width"] == OUT // 2  # the intermediate really is sharded


def _indivisible(rank, world_size):
    init_distributed(DistributedConfig(backend="gloo"))
    errors = []
    for build in (
        lambda: ColumnParallelLinear(IN, 7),  # 7 outputs over 2 ranks
        lambda: RowParallelLinear(7, OUT),  # 7 inputs over 2 ranks
    ):
        try:
            build()
        except ConfigError as e:
            errors.append(str(e))
    return errors


def test_indivisible_dimensions_are_rejected():
    results = run_distributed(_indivisible, 2)
    assert len(results[0]) == 2
    assert "out_features (7) must be divisible by tp_size (2)" in results[0][0]
    assert "in_features (7) must be divisible by tp_size (2)" in results[0][1]
