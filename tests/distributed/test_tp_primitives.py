"""The f / g conjugate pair and the scatter/gather variants, tested in both directions.

These are the only places tensor parallelism communicates, so they get direct numeric tests rather
than being covered incidentally by the model-level comparison (D11: equivalence tests cannot catch
bugs in code both paths share).
"""

import pytest
import torch

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.distributed.tensor_parallel import (
    copy_to_tp_region,
    gather_from_tp_region,
    group_rank,
    group_size,
    reduce_from_tp_region,
    scatter_to_tp_region,
)
from tests.distributed.conftest import run_distributed

pytestmark = pytest.mark.distributed


def _copy_region(rank, world_size):
    """`f`: identity forward, all-reduce backward."""
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.ones(2, 4, requires_grad=True)
    y = copy_to_tp_region(x)
    forward_unchanged = torch.equal(y.detach(), torch.ones(2, 4))

    # Weight each rank's contribution differently, so a missing all-reduce is visible: the
    # gradient must end up as the SUM over ranks of (rank + 1) = 1 + 2 = 3.
    (y * (rank + 1)).sum().backward()
    return {"forward_unchanged": forward_unchanged, "grad": x.grad.clone()}


def test_f_is_identity_forward_and_all_reduce_backward():
    results = run_distributed(_copy_region, 2)
    for rank in (0, 1):
        assert results[rank]["forward_unchanged"], "f changed its input in the forward"
        # Both ranks see the summed gradient, not just their own contribution.
        torch.testing.assert_close(results[rank]["grad"], torch.full((2, 4), 3.0))


def _reduce_region(rank, world_size):
    """`g`: all-reduce forward, identity backward."""
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.full((2, 4), float(rank + 1), requires_grad=True)
    y = reduce_from_tp_region(x)
    y.sum().backward()
    return {"forward": y.detach().clone(), "grad": x.grad.clone()}


def test_g_is_all_reduce_forward_and_identity_backward():
    results = run_distributed(_reduce_region, 2)
    for rank in (0, 1):
        # 1 + 2 = 3: the partial sums are completed, identically on every rank.
        torch.testing.assert_close(results[rank]["forward"], torch.full((2, 4), 3.0))
        # The incoming gradient is already complete, so backward must not touch it.
        torch.testing.assert_close(results[rank]["grad"], torch.ones(2, 4))


def _conjugate(rank, world_size):
    """f and g are conjugates: composing them scales by tp_size in both directions."""
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.ones(3, requires_grad=True)
    reduce_from_tp_region(copy_to_tp_region(x)).sum().backward()
    return {"grad": x.grad.clone(), "world": group_size()}


def test_f_and_g_compose_as_conjugates():
    results = run_distributed(_conjugate, 2)
    # g sums across ranks in the forward; f sums across ranks in the backward.
    torch.testing.assert_close(results[0]["grad"], torch.full((3,), 2.0))


def _scatter_region(rank, world_size):
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.arange(8.0).repeat(2, 1).requires_grad_(True)  # (2, 8), identical on both ranks
    y = scatter_to_tp_region(x)
    (y * (rank + 1)).sum().backward()
    return {"forward": y.detach().clone(), "grad": x.grad.clone(), "rank": group_rank()}


def test_scatter_splits_the_last_dimension_and_gathers_the_gradient():
    results = run_distributed(_scatter_region, 2)
    torch.testing.assert_close(results[0]["forward"], torch.arange(0.0, 4.0).repeat(2, 1))
    torch.testing.assert_close(results[1]["forward"], torch.arange(4.0, 8.0).repeat(2, 1))

    # The backward all-gathers, so each half of the gradient carries its own rank's weight:
    # columns 0-3 scaled by 1 (rank 0) and columns 4-7 by 2 (rank 1).
    expected = torch.cat([torch.ones(2, 4), torch.full((2, 4), 2.0)], dim=-1)
    for rank in (0, 1):
        torch.testing.assert_close(results[rank]["grad"], expected)


def _gather_region(rank, world_size):
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.full((2, 3), float(rank + 1), requires_grad=True)
    y = gather_from_tp_region(x)
    (y * torch.arange(1.0, y.shape[-1] + 1)).sum().backward()
    return {"forward": y.detach().clone(), "grad": x.grad.clone()}


def test_gather_concatenates_the_last_dimension_and_splits_the_gradient():
    results = run_distributed(_gather_region, 2)
    expected_forward = torch.cat([torch.ones(2, 3), torch.full((2, 3), 2.0)], dim=-1)
    for rank in (0, 1):
        torch.testing.assert_close(results[rank]["forward"], expected_forward)

    # The backward splits, so each rank keeps only the columns it contributed.
    torch.testing.assert_close(results[0]["grad"], torch.tensor([1.0, 2.0, 3.0]).repeat(2, 1))
    torch.testing.assert_close(results[1]["grad"], torch.tensor([4.0, 5.0, 6.0]).repeat(2, 1))


def _scatter_gather_identity(rank, world_size):
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(0), requires_grad=True)
    y = gather_from_tp_region(scatter_to_tp_region(x))
    y.sum().backward()
    return {"round_trip": torch.equal(y.detach(), x.detach()), "grad": x.grad.clone()}


def test_scatter_then_gather_is_the_identity():
    results = run_distributed(_scatter_gather_identity, 2)
    for rank in (0, 1):
        assert results[rank]["round_trip"]
        torch.testing.assert_close(results[rank]["grad"], torch.ones(2, 8))


def _no_op_in_one_process(rank, world_size):
    """A group of one must skip every collective, so single-device code paths stay unchanged."""
    init_distributed(DistributedConfig(backend="gloo"))
    x = torch.ones(2, 4, requires_grad=True)
    return {"size": group_size(), "scatter_shape": tuple(scatter_to_tp_region(x).shape)}


def test_collectives_are_skipped_for_a_group_of_one():
    results = run_distributed(_no_op_in_one_process, 1)
    assert results[0]["size"] == 1
    assert results[0]["scatter_shape"] == (2, 4)  # nothing split
