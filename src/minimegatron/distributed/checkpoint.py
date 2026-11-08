"""Checkpointing for sharded (FSDP) runs, in two deliberately different formats.

An FSDP model's `state_dict` holds `DTensor`s, so Milestone 1's single-file `torch.save` is not
sufficient on its own. Two paths exist, and the second is not redundant:

1. **Sharded** (`torch.distributed.checkpoint`): a *directory*, each rank writing its own shard.
   Nothing is ever gathered, so a model too large for one device still saves. FSDP resume uses it.
2. **Consolidated**: a full fp32 state dict gathered on rank 0 and written through Milestone 1's
   `save_checkpoint`. This keeps checkpoints **interchangeable across strategies**, which is what
   lets an FSDP run be compared against the single-device reference at all -- the milestone's
   actual evidence -- and lets a sharded run's result be loaded anywhere for inference.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from minimegatron.training.checkpoint import (
    FORMAT_VERSION,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)

META_FILE = "meta.pt"


def _rng_file(rank: int) -> str:
    return f"rng_{rank}.pt"


def _atomic_save(payload: Any, target: Path) -> None:
    """Milestone 1's discipline: write to .tmp, then rename, so a crash can't corrupt the file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def save_sharded(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    config: Any,
    rank: int,
    is_main: bool,
    device: torch.device | None = None,
) -> Path:
    """Write a sharded checkpoint directory.

    `get_model_state_dict` / `get_optimizer_state_dict` know how to describe `DTensor`s, so DCP
    can write each rank's shard and record how they fit together. The small, non-tensor state goes
    in `meta.pt` beside the shards, which keeps the directory self-contained and readable without
    DCP.

    RNG state is written **per rank** rather than only from rank 0: dropout masks and anything
    else random differ by rank, so restoring one rank's streams everywhere would make a resumed
    run diverge from the uninterrupted one -- the same reasoning as Milestone 1's RNG capture,
    extended to a cluster.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    dcp.save(
        {
            "model": get_model_state_dict(model),
            "optim": get_optimizer_state_dict(model, optimizer),
        },
        checkpoint_id=str(directory),
    )

    _atomic_save(rng_state(device), directory / _rng_file(rank))
    if is_main:
        _atomic_save(
            {
                "format_version": FORMAT_VERSION,
                "step": step,
                "tokens_seen": tokens_seen,
                "scheduler": scheduler.state_dict(),
                "config": dataclasses.asdict(config)
                if dataclasses.is_dataclass(config)
                else config,
            },
            directory / META_FILE,
        )
    return directory


def load_sharded(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    rank: int = 0,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a sharded checkpoint in place and return its bookkeeping fields."""
    directory = Path(directory)
    meta_path = directory / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"not a sharded checkpoint (no {META_FILE}): {directory}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{directory}: checkpoint format version {meta.get('format_version')!r}, "
            f"this build writes {FORMAT_VERSION}"
        )

    # DCP loads *into* a template state dict, which tells it each tensor's sharding, so the
    # template must come from the already-sharded model rather than be an empty dict.
    state: dict[str, Any] = {"model": get_model_state_dict(model)}
    if optimizer is not None:
        state["optim"] = get_optimizer_state_dict(model, optimizer)
    dcp.load(state, checkpoint_id=str(directory))

    set_model_state_dict(model, state["model"])
    if optimizer is not None:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state["optim"])
    if scheduler is not None:
        scheduler.load_state_dict(meta["scheduler"])

    rng_path = directory / _rng_file(rank)
    if restore_rng and rng_path.is_file():
        restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=True))

    return {"step": meta["step"], "tokens_seen": meta["tokens_seen"], "config": meta["config"]}


def full_state_dicts(
    model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gather full fp32 state dicts on rank 0; other ranks get empty dicts.

    `full_state_dict=True` all-gathers every shard and `cpu_offload=True` moves the result to host
    memory, so rank 0 ends up holding a plain, unsharded state dict -- the same keys a
    single-device `GPT.state_dict()` would produce, tied weights included (`state_dict` lists both
    `tok_emb.weight` and `lm_head.weight`, which is exactly why the file loads unchanged).
    """
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    optimizer_state = (
        get_optimizer_state_dict(model, optimizer, options=options) if optimizer is not None else {}
    )
    return model_state, optimizer_state


def save_consolidated(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    config: Any,
    is_main: bool,
    device: torch.device | None = None,
) -> Path:
    """Gather the shards and write one single-device-compatible file from rank 0."""
    model_state, optimizer_state = full_state_dicts(model, optimizer)
    path = Path(path)
    if is_main:
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            tokens_seen=tokens_seen,
            config=config,
            device=device,
            model_state=model_state,
            optimizer_state=optimizer_state,
        )
    return path
