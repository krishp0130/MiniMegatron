"""TP state-dict sharding and configuration validation. No processes needed."""

import dataclasses

import pytest
import torch

from minimegatron.config import ConfigError
from minimegatron.distributed.tensor_parallel import (
    gather_state_dict,
    shard_state_dict,
    tp_shard_dim,
    validate_tp,
)
from minimegatron.model.transformer import GPT


# --- which parameters are split, and along which dimension --------------------------------------


@pytest.mark.parametrize(
    "key, dim",
    [
        # Column parallel: split along the output dimension, biases included.
        ("blocks.0.attn.qkv.weight", 0),
        ("blocks.0.attn.qkv.bias", 0),
        ("blocks.3.mlp.fc1.weight", 0),
        ("blocks.3.mlp.fc1.bias", 0),
        # Row parallel: the weight is split along the INPUT dimension...
        ("blocks.0.attn.out_proj.weight", 1),
        ("blocks.0.mlp.fc2.weight", 1),
        # ...but its bias is replicated, because it is added after the all-reduce.
        ("blocks.0.attn.out_proj.bias", None),
        ("blocks.0.mlp.fc2.bias", None),
        # Replicated in the basic version.
        ("tok_emb.weight", None),
        ("pos_emb.weight", None),
        ("lm_head.weight", None),
        ("norm_f.weight", None),
        ("blocks.0.norm1.weight", None),
        ("blocks.0.norm2.bias", None),
    ],
)
def test_shard_dim_classification(key, dim):
    assert tp_shard_dim(key) == dim


def test_row_parallel_bias_is_replicated_not_split():
    """The easiest mistake in TP: a row-parallel bias added per rank is added tp_size times."""
    assert tp_shard_dim("blocks.0.mlp.fc2.weight") == 1
    assert tp_shard_dim("blocks.0.mlp.fc2.bias") is None


# --- the round trip -----------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_sharding_and_gathering_a_state_dict_round_trips(nano_config, tp_size):
    """Exit criterion 6. Shards must reassemble into exactly the model they came from, or a TP
    checkpoint could not be compared against the reference or loaded anywhere else."""
    full = GPT(nano_config).state_dict()
    shards = [shard_state_dict(full, rank, tp_size) for rank in range(tp_size)]
    gathered = gather_state_dict(shards)

    assert set(gathered) == set(full)
    for key, value in full.items():
        assert torch.equal(gathered[key], value), f"{key} did not survive the round trip"


def test_shards_partition_the_split_parameters(nano_config):
    """Nothing duplicated, nothing lost: the whole point of splitting rather than replicating."""
    full = GPT(nano_config).state_dict()
    shards = [shard_state_dict(full, rank, 2) for rank in range(2)]

    for key, value in full.items():
        dim = tp_shard_dim(key)
        local = shards[0][key]
        if dim is None:
            assert local.shape == value.shape  # replicated
            assert torch.equal(local, shards[1][key])
        else:
            assert local.shape[dim] == value.shape[dim] // 2
            assert sum(s[key].shape[dim] for s in shards) == value.shape[dim]


def test_qkv_shard_holds_whole_heads(nano_config):
    """Why M1 chose the per-head [h, 3, d_h] QKV layout (D6).

    A contiguous slice of the column-parallel QKV output must contain complete heads -- all of
    q, k and v for each -- or a rank could not run attention without communication. With GPT-2's
    [3, h, d_h] layout, rank 0 would get every query and only half the keys.
    """
    full = GPT(nano_config).state_dict()
    shard = shard_state_dict(full, 0, 2)["blocks.0.attn.qkv.weight"]

    head_dim = nano_config.head_dim
    local_heads = nano_config.n_heads // 2
    assert shard.shape[0] == local_heads * 3 * head_dim

    # The first local_heads heads of the full matrix, q/k/v interleaved per head.
    expected = full["blocks.0.attn.qkv.weight"][: local_heads * 3 * head_dim]
    assert torch.equal(shard, expected)


def test_sharding_does_not_alias_the_source(nano_config):
    """Shards are copies: mutating one must not reach back into the full state dict."""
    full = GPT(nano_config).state_dict()
    original = full["blocks.0.mlp.fc1.weight"].clone()
    shard = shard_state_dict(full, 0, 2)
    shard["blocks.0.mlp.fc1.weight"].zero_()
    assert torch.equal(full["blocks.0.mlp.fc1.weight"], original)


def test_gathering_nothing_is_an_error():
    with pytest.raises(ValueError, match="no shards"):
        gather_state_dict([])


# --- validation ---------------------------------------------------------------------------------


def test_valid_tp_config_passes(nano_config):
    validate_tp(nano_config, tp_size=2, world_size=2)  # 4 heads, d_ff 64
    validate_tp(nano_config, tp_size=4, world_size=4)


def test_indivisible_heads_rejected(nano_config):
    cfg = dataclasses.replace(nano_config, n_heads=4, d_model=32)
    with pytest.raises(ConfigError, match="n_heads .4. must be divisible by tp_size .3."):
        validate_tp(cfg, tp_size=3, world_size=3)


def test_indivisible_d_ff_rejected(nano_config):
    cfg = dataclasses.replace(nano_config, d_ff=30, n_heads=4, d_model=32)
    with pytest.raises(ConfigError, match="d_ff .30. must be divisible by tp_size .4."):
        validate_tp(cfg, tp_size=4, world_size=4)  # heads and world size divide; d_ff does not


def test_world_size_not_divisible_by_tp_size_rejected(nano_config):
    with pytest.raises(ConfigError, match="world_size .3. must be divisible by tp_size .2."):
        validate_tp(nano_config, tp_size=2, world_size=3)


def test_all_tp_errors_reported_at_once(nano_config):
    cfg = dataclasses.replace(nano_config, n_heads=4, d_model=32, d_ff=32)
    with pytest.raises(ConfigError) as exc:
        validate_tp(cfg, tp_size=3, world_size=4)  # 4%3, 32%3 and 4%3 all fail
    message = str(exc.value)
    assert "n_heads" in message and "d_ff" in message and "world_size" in message


def test_error_explains_the_head_constraint(nano_config):
    cfg = dataclasses.replace(nano_config, n_heads=4, d_model=32)
    with pytest.raises(ConfigError, match="partitions whole attention heads"):
        validate_tp(cfg, tp_size=3, world_size=3)


def test_tp_size_smaller_than_the_world_is_rejected(nano_config):
    """2D TP+DP is out of scope, and allowing it would train silently wrong.

    With tp_size < world_size there is a real data-parallel dimension, but nothing synchronizes
    gradients across it -- each data-parallel group would see different batches and diverge. A
    clear refusal beats a run that looks fine and is not.
    """
    with pytest.raises(ConfigError, match="is not supported yet"):
        validate_tp(nano_config, tp_size=2, world_size=4)
    with pytest.raises(ConfigError, match="without gradient synchronization"):
        validate_tp(nano_config, tp_size=2, world_size=4)


def test_tp_size_equal_to_the_world_is_accepted(nano_config):
    validate_tp(nano_config, tp_size=2, world_size=2)
    validate_tp(nano_config, tp_size=4, world_size=4)


# --- parallel layers must inherit placement from what they replace (ISSUES #19) -----------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_parallel_layers_inherit_dtype_from_the_layer_they_replace(dtype):
    """Catches the missing factory kwargs on a CPU-only machine.

    The real bug was `torch.empty(...)` with no `device=`, which put every parallel shard on the
    CPU while the rest of the model sat on a GPU — invisible here, fatal on a GPU. `dtype` travels
    through exactly the same factory arguments, so pinning it pins the device too.
    """
    import torch.nn as nn

    from minimegatron.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    linear = nn.Linear(8, 12).to(dtype)
    for built in (
        ColumnParallelLinear.from_linear(linear),
        RowParallelLinear.from_linear(linear),
    ):
        assert built.weight.dtype == dtype
        assert built.weight.device == linear.weight.device
        if built.bias is not None:
            assert built.bias.dtype == dtype
            assert built.bias.device == linear.bias.device


def test_assert_single_device_accepts_a_uniform_model(nano_config):
    from minimegatron.distributed.tensor_parallel import assert_single_device

    assert assert_single_device(GPT(nano_config)) == torch.device("cpu")


def test_assert_single_device_names_the_offenders(nano_config):
    """The guard that turns a silent split into a clear failure at construction."""
    import torch.nn as nn

    from minimegatron.distributed.tensor_parallel import assert_single_device

    model = GPT(nano_config)
    # `meta` stands in for "a different device" without needing a second real one.
    model.blocks[0].mlp.fc1.weight = nn.Parameter(
        torch.empty_like(model.blocks[0].mlp.fc1.weight, device="meta")
    )
    with pytest.raises(RuntimeError, match="span multiple devices"):
        assert_single_device(model)
