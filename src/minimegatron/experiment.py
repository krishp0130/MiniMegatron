"""Top-level experiment config: one YAML file -> one validated ExperimentConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimegatron.config import ConfigError, from_dict, load_yaml
from minimegatron.data.config import DataConfig
from minimegatron.model.config import ModelConfig
from minimegatron.training.config import TrainConfig

# `distributed` arrives in Milestone 2.
SECTIONS = ("model", "data", "train")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    # `data` and `train` are optional, so a file that only describes a model shape stays valid.
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def config_from_dict(raw: dict[str, Any], source: str = "config") -> ExperimentConfig:
    """Validate a raw config mapping. Used for YAML files and for `--set` overrides alike, so a
    command-line override is checked exactly as strictly as a file."""
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"{source}: unknown section(s) {unknown}; expected {list(SECTIONS)}")
    if "model" not in raw:
        raise ConfigError(f"{source}: missing required section 'model'")
    return ExperimentConfig(
        model=from_dict(ModelConfig, raw["model"], section="model"),
        data=from_dict(DataConfig, raw.get("data", {}), section="data"),
        train=from_dict(TrainConfig, raw.get("train", {}), section="train"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(load_yaml(path), source=str(path))
