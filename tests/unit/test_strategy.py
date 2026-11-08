"""The `distributed.strategy` switch: resolution and the combinations it refuses."""

import pytest

from minimegatron.config import ConfigError
from minimegatron.distributed.config import STRATEGIES, DistributedConfig
from minimegatron.distributed.setup import resolve_strategy


def test_auto_preserves_existing_defaults():
    """`auto` must not change how any pre-Milestone-3 config behaves."""
    assert resolve_strategy("auto", 1) == "single"
    assert resolve_strategy("auto", 2) == "ddp"
    assert resolve_strategy("auto", 8) == "ddp"


@pytest.mark.parametrize("strategy", ["single", "ddp", "fsdp"])
def test_explicit_strategies_pass_through(strategy):
    assert resolve_strategy(strategy, 2) == strategy


def test_single_is_valid_in_one_process():
    assert resolve_strategy("single", 1) == "single"


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_strategies_needing_peers_are_rejected_in_one_process(strategy):
    """No silent downgrade: a run that quietly trains differently from its config is worse than
    one that stops and says so."""
    with pytest.raises(ConfigError, match=f"strategy '{strategy}' needs more than one rank"):
        resolve_strategy(strategy, 1)


def test_error_names_the_fix():
    with pytest.raises(ConfigError, match="torchrun --nproc_per_node"):
        resolve_strategy("fsdp", 1)


def test_invalid_strategy_rejected():
    with pytest.raises(ConfigError, match="strategy must be one of"):
        DistributedConfig(strategy="pipeline")


def test_default_strategy_is_auto():
    assert DistributedConfig().strategy == "auto"
    assert "auto" in STRATEGIES
