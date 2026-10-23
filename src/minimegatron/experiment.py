"""Top-level experiment config: one YAML file -> one validated ExperimentConfig."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minimegatron.config import ConfigError, from_dict, load_yaml
from minimegatron.model.config import ModelConfig

# Sections grow as milestones land: `data` and `train` in M1 steps 5-6, `distributed` in M2.
SECTIONS = ("model",)


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"{path}: unknown section(s) {unknown}; expected {list(SECTIONS)}")
    if "model" not in raw:
        raise ConfigError(f"{path}: missing required section 'model'")
    return ExperimentConfig(model=from_dict(ModelConfig, raw["model"], section="model"))
