"""FSDP2 wrapping: shard parameters, gradients and optimizer state across ranks.

DDP replicates all of it on every rank, so the largest trainable model is capped by one device.
For `P` parameters in FP32 with AdamW that is 16 bytes per parameter before a single activation
(params + grads + Adam's m and v), which FSDP divides by the world size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

PRECISION_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,  # no autocast, no mixed-precision policy
    "bf16": torch.bfloat16,
}


def precision_dtype(precision: str) -> torch.dtype | None:
    """The compute dtype for a `train.precision` value, or None for full fp32."""
    return PRECISION_DTYPES[precision]


def mixed_precision_policy(precision: str) -> MixedPrecisionPolicy | None:
    """FSDP's own mixed precision, which `torch.autocast` cannot replicate.

    `param_dtype=bf16` stores shards in fp32 but **all-gathers them as bf16**, halving the volume
    of FSDP's dominant collective. Autocast converts at the op boundary, by which point the gather
    has already moved fp32 bytes -- so under FSDP the gather dtype *is* the bandwidth story.

    `reduce_dtype=fp32` keeps the gradient reduce-scatter accumulating in fp32: summing many small
    bf16 gradients loses precision exactly where it matters, and the reduction is cheap next to
    the gather.
    """
    dtype = precision_dtype(precision)
    if dtype is None:
        return None
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)


def build_fsdp(model: nn.Module, mesh, precision: str = "fp32") -> nn.Module:
    """Shard `model` in place: one unit per transformer block, then the root.

    Each `fully_shard` call creates a *unit* -- the granularity at which parameters are
    all-gathered before use and freed after. Per-block units mean peak transient memory is one
    block's parameters. Wrapping only the root would shard storage but still gather the whole
    model at once in the forward, which defeats the purpose. The cost of more units is more,
    smaller collectives, which is why the unit is a block rather than an individual Linear.

    The root call must come last, and it keeps the tied embedding/LM head in the root unit (D8):
    they are the same tensor, and a shared parameter split across two units would be gathered and
    freed on two schedules.

    Callers must build the optimizer **after** this returns -- `fully_shard` replaces parameters
    with sharded DTensors in place, so an optimizer created earlier would hold the pre-shard
    tensors.
    """
    kwargs = {"mesh": mesh}
    policy = mixed_precision_policy(precision)
    if policy is not None:
        kwargs["mp_policy"] = policy

    for block in model.blocks:
        fully_shard(block, **kwargs)
    return fully_shard(model, **kwargs)


# --- introspection, used by tests and by memory reporting ---------------------------------------


def is_sharded(model: nn.Module) -> bool:
    return any(isinstance(p, DTensor) for p in model.parameters())


def local_parameter_count(model: nn.Module) -> int:
    """Parameters this rank actually stores. Equals the full count when unsharded."""
    return sum(
        p.to_local().numel() if isinstance(p, DTensor) else p.numel() for p in model.parameters()
    )


def local_optimizer_state_count(optimizer: torch.optim.Optimizer) -> int:
    """Optimizer-state elements this rank stores: Adam's m and v, sharded along with the params.

    Measured rather than assumed, because optimizer-state sharding is half of FSDP's memory
    saving (8 of the 16 bytes per parameter) and the easiest half to get silently wrong -- an
    optimizer built before `fully_shard` would hold full-size state and still train.
    """
    total = 0
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is None:
                continue
            total += value.to_local().numel() if isinstance(value, DTensor) else value.numel()
    return total
