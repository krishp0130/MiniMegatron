"""Tests for synthetic datasets and deterministic batching."""

import pytest
import torch

from minimegatron.config import ConfigError
from minimegatron.data.config import DataConfig
from minimegatron.data.synthetic import (
    BatchStream,
    IndexStream,
    MemorizationDataset,
    RandomTokenDataset,
    build_dataset,
    _mix,
)

V, T, N = 128, 16, 64


def random_ds(seed: int = 0, n_items: int = N) -> RandomTokenDataset:
    return RandomTokenDataset(vocab_size=V, seq_len=T, n_items=n_items, seed=seed)


# --- index determinism --------------------------------------------------------------------------


def test_same_seed_and_index_give_identical_tokens():
    a, b = random_ds(seed=7), random_ds(seed=7)
    for i in (0, 1, 37):
        torch.testing.assert_close(a[i][0], b[i][0])


def test_different_seed_gives_different_tokens():
    assert not torch.equal(random_ds(seed=1)[0][0], random_ds(seed=2)[0][0])


def test_neighbouring_indices_are_not_correlated():
    """Seeding with `seed + index` would make adjacent items near-identical; _mix prevents that."""
    ds = random_ds(seed=0)
    assert not torch.equal(ds[0][0], ds[1][0])
    overlap = (ds[0][0] == ds[1][0]).float().mean().item()
    assert overlap < 0.5  # chance agreement is 1/V


def test_mix_avalanches_adjacent_inputs():
    a, b = _mix(0, 0), _mix(0, 1)
    differing_bits = bin(a ^ b).count("1")
    assert differing_bits > 10  # a good mixer flips about half of the 63 bits


# --- the next-token objective -------------------------------------------------------------------


def test_targets_are_inputs_shifted_by_one():
    x, y = random_ds()[3]
    torch.testing.assert_close(x[1:], y[:-1])  # y[t] is the token after x[t]


def test_shapes_dtype_and_token_range():
    x, y = random_ds()[0]
    assert x.shape == (T,) and y.shape == (T,)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert 0 <= x.min() and x.max() < V


def test_out_of_range_index_raises():
    with pytest.raises(IndexError):
        random_ds(n_items=4)[4]


# --- memorization dataset -----------------------------------------------------------------------


def test_memorization_repeats_with_period_n_sequences():
    ds = MemorizationDataset(vocab_size=V, seq_len=T, n_items=N, n_sequences=4, seed=0)
    torch.testing.assert_close(ds[0][0], ds[4][0])
    torch.testing.assert_close(ds[1][0], ds[9][0])
    assert not torch.equal(ds[0][0], ds[1][0])
    assert len({tuple(ds[i][0].tolist()) for i in range(N)}) == 4  # exactly 4 distinct sequences


# --- the index stream ---------------------------------------------------------------------------


def test_each_epoch_is_a_true_permutation():
    stream = IndexStream(n_items=N, seed=0)
    for epoch in (0, 1, 5):
        assert sorted(stream.permutation(epoch).tolist()) == list(range(N))


def test_epochs_differ_from_each_other():
    stream = IndexStream(n_items=N, seed=0)
    assert not torch.equal(stream.permutation(0), stream.permutation(1))


def test_take_wraps_across_the_epoch_boundary():
    stream = IndexStream(n_items=8, seed=0)
    taken = stream.take(6, 4)  # 2 from epoch 0, then 2 from epoch 1
    expected = torch.cat([stream.permutation(0)[6:], stream.permutation(1)[:2]])
    torch.testing.assert_close(taken, expected)


def test_stream_is_resumable_at_any_offset():
    """Reading from offset k must match reading straight through -- this is what makes resume
    exact without saving any dataloader state."""
    a = IndexStream(n_items=N, seed=3).take(0, 200)
    b = IndexStream(n_items=N, seed=3).take(100, 100)  # a fresh stream, started mid-run
    torch.testing.assert_close(a[100:], b)


def test_stream_order_is_independent_of_how_it_is_chopped():
    stream = IndexStream(n_items=N, seed=0)
    whole = stream.take(0, 60)
    pieces = torch.cat([stream.take(0, 7), stream.take(7, 13), stream.take(20, 40)])
    torch.testing.assert_close(whole, pieces)


# --- batching -----------------------------------------------------------------------------------


def test_micro_batch_shapes():
    bs = BatchStream(random_ds(), micro_batch_size=4, seed=0)
    x, y = bs.micro_batch(0)
    assert x.shape == (4, T) and y.shape == (4, T)


def test_micro_batch_is_a_pure_function_of_seed_and_index():
    a = BatchStream(random_ds(), micro_batch_size=4, seed=0).micro_batch(11)
    b = BatchStream(random_ds(), micro_batch_size=4, seed=0).micro_batch(11)
    torch.testing.assert_close(a[0], b[0])


def test_accumulation_sees_the_same_samples_as_one_large_batch():
    """accum=4 x micro=2 must consume exactly the samples of accum=1 x micro=8, in order.

    This is the property the gradient-accumulation equivalence test rests on (and Milestone 2's
    DDP equivalence test after it). It holds because the index stream is defined over samples,
    not batches.
    """
    ds = random_ds()
    small = BatchStream(ds, micro_batch_size=2, seed=0)
    large = BatchStream(ds, micro_batch_size=8, seed=0)

    accumulated = torch.cat([small.micro_batch(j)[0] for j in range(4)])
    torch.testing.assert_close(accumulated, large.micro_batch(0)[0])


def test_batching_is_consistent_across_a_step_boundary():
    ds = random_ds()
    small = BatchStream(ds, micro_batch_size=2, seed=0)
    large = BatchStream(ds, micro_batch_size=8, seed=0)
    # Step 1 with accum=4 -> micro indices 4..7; step 1 with accum=1 -> micro index 1.
    accumulated = torch.cat([small.micro_batch(j)[0] for j in range(4, 8)])
    torch.testing.assert_close(accumulated, large.micro_batch(1)[0])


# --- config and construction --------------------------------------------------------------------


def test_val_seed_defaults_to_a_different_split():
    cfg = DataConfig(seed=5)
    assert cfg.val_seed == 6


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dataset": "shakespeare"}, "dataset must be one of"),
        ({"n_items": 0}, "n_items must be a positive integer"),
        ({"n_sequences": -1}, "n_sequences must be a positive integer"),
        ({"seed": True}, "seed must be a non-negative integer"),
        ({"seed": 5, "val_seed": 5}, "val_seed must differ"),
    ],
)
def test_invalid_data_config_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        DataConfig(**overrides)


def test_build_dataset_honours_dataset_and_split(nano_config):
    data_cfg = DataConfig(dataset="memorization", n_items=32, n_sequences=4, seed=0)
    train = build_dataset(data_cfg, nano_config, split="train")
    val = build_dataset(data_cfg, nano_config, split="val")
    assert isinstance(train, MemorizationDataset)
    assert train.seq_len == nano_config.seq_len and train.vocab_size == nano_config.vocab_size
    assert not torch.equal(train[0][0], val[0][0])  # disjoint splits


def test_build_dataset_rejects_unknown_split(nano_config):
    with pytest.raises(ValueError, match="split must be"):
        build_dataset(DataConfig(), nano_config, split="test")


# --- rank sharding (Milestone 2) ----------------------------------------------------------------


def shards(micro: int, world_size: int, micro_index: int = 0, seed: int = 0):
    ds = random_ds()
    return [
        BatchStream(ds, micro, seed, rank=r, world_size=world_size).indices(micro_index)
        for r in range(world_size)
    ]


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_ranks_together_consume_the_single_process_batch(world_size):
    """The union over ranks of micro-batch m is the single-process batch of size micro x R.

    This is what makes the DDP equivalence test exact: two ranks of 4 see precisely the samples
    one process of 8 would, in the same order.
    """
    micro = 4
    union = torch.cat(shards(micro, world_size))
    whole = BatchStream(random_ds(), micro * world_size, 0).indices(0)
    torch.testing.assert_close(union, whole)


def test_ranks_never_see_the_same_sample():
    a, b = shards(micro=4, world_size=2)
    assert set(a.tolist()).isdisjoint(b.tolist())


def test_every_rank_gets_the_same_number_of_samples():
    """Unequal shards would silently reweight the averaged gradient: the mean of per-rank means
    is only the mean over the global batch when the shards are the same size."""
    for shard in shards(micro=4, world_size=4):
        assert len(shard) == 4


def test_sharding_still_covers_each_epoch_exactly_once():
    micro, world_size = 4, 2
    per_step = micro * world_size
    seen = []
    for micro_index in range(N // per_step):
        seen += [int(i) for shard in shards(micro, world_size, micro_index) for i in shard]
    assert sorted(seen) == list(range(N))


def test_single_rank_sharding_is_the_milestone_1_behaviour():
    """world_size=1 must change nothing, so the single-process path stays the reference."""
    ds = random_ds()
    sharded = BatchStream(ds, 4, 0, rank=0, world_size=1).indices(3)
    plain = BatchStream(ds, 4, 0).indices(3)
    torch.testing.assert_close(sharded, plain)


@pytest.mark.parametrize("rank, world_size", [(2, 2), (-1, 2), (0, 0)])
def test_invalid_rank_or_world_size_rejected(rank, world_size):
    with pytest.raises(ValueError):
        BatchStream(random_ds(), 4, 0, rank=rank, world_size=world_size)
