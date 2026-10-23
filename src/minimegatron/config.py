"""Shared configuration helpers: strict dict -> dataclass conversion and YAML loading."""

from __future__ import annotations

import difflib
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """An invalid configuration. Messages list every problem found, not just the first."""


def from_dict(cls: type[T], data: Any, section: str) -> T:
    """Build dataclass `cls` from a mapping, rejecting unknown and missing keys.

    Unknown keys are an error rather than being ignored: a typo like `n_layer: 12` would otherwise
    silently leave the default in place and train the wrong model.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"[{section}] must be a mapping, got {type(data).__name__}")

    known = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(known))
    if unknown:
        problems = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), known, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            problems.append(f"unknown key {key!r}{hint}")
        raise ConfigError(f"[{section}] " + "; ".join(problems))

    required = [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]
    missing = [name for name in required if name not in data]
    if missing:
        raise ConfigError(f"[{section}] missing required key(s): {', '.join(missing)}")

    return cls(**data)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, turning I/O and parse failures into ConfigErrors."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"could not parse {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data
