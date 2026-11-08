"""Checkpoint save/load, written so that resuming is bit-for-bit identical to never stopping."""

from __future__ import annotations

import dataclasses
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

FORMAT_VERSION = 1


def _numpy_rng_state() -> dict[str, Any]:
    """NumPy's MT19937 state, in types `torch.load(weights_only=True)` will accept.

    The raw state tuple contains a uint32 ndarray, and the safe loader rejects ndarrays. Storing
    it as an int64 tensor keeps checkpoints loadable without `weights_only=False`, which would
    mean unpickling arbitrary objects out of a file that may not be ours.
    """
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": torch.from_numpy(keys.astype(np.int64)),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(
        (
            state["name"],
            state["keys"].numpy().astype(np.uint32),
            state["pos"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )


def rng_state(device: torch.device | None = None) -> dict[str, Any]:
    """Capture every RNG that can influence a training step.

    Dropout masks come from the torch RNG; without restoring it a resumed run diverges from the
    uninterrupted one and you can no longer tell a resume bug from ordinary noise.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device is not None and device.type == "mps":
        # The MPS surface has gaps on this torch build (ISSUES #2); never let it break a save.
        try:
            state["mps"] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):
            pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    # Serialization may hand tuples back as lists, and `random.setstate` insists on a tuple of
    # ints, so rebuild the shape it wants rather than trusting what came off disk.
    version, internal, gauss = state["python"]
    random.setstate((int(version), tuple(int(v) for v in internal), gauss))
    _restore_numpy_rng_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        try:
            torch.mps.set_rng_state(state["mps"])
        except (AttributeError, RuntimeError):
            pass


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    config: Any,
    device: torch.device | None = None,
) -> Path:
    """Write a checkpoint atomically: to a temp file, then rename it into place.

    `os.replace` is atomic on POSIX, so a crash (or a killed job) part-way through a multi-hundred
    megabyte write can never leave a half-written file where the last good checkpoint was. The
    failure mode it prevents -- losing a long run to a crash during its own checkpoint -- is the
    kind that only ever shows up at the worst moment.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config,
        "rng": rng_state(device),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint in place and return its bookkeeping fields."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)

    version = ckpt.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format version {version!r}, this build writes {FORMAT_VERSION}"
        )

    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        # Restores `last_epoch`; the learning rate itself rides along in the optimizer's state.
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        restore_rng_state(ckpt["rng"])

    return {
        "step": ckpt["step"],
        "tokens_seen": ckpt["tokens_seen"],
        "config": ckpt["config"],
    }
