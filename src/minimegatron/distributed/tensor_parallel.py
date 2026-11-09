"""Megatron-style tensor parallelism, written out by hand.

DDP splits the batch; FSDP shards storage but computes whole layers. Tensor parallelism splits
each layer's **weight matrices**, so every rank holds and computes a slice of every layer and no
rank ever materializes a full one.

The communication is implemented with explicit `torch.autograd.Function`s and
`torch.distributed` collectives (D3) rather than DTensor, so every byte of it is visible here.
See `docs/tensor_parallel.md` for the derivation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from minimegatron.config import ConfigError
from minimegatron.model.config import ModelConfig

# Linear layers whose weights are split along the OUTPUT dimension (dim 0), biases included.
COLUMN_PARALLEL = ("attn.qkv", "mlp.fc1")
# Linear layers whose weights are split along the INPUT dimension (dim 1). Biases stay replicated.
ROW_PARALLEL = ("attn.out_proj", "mlp.fc2")


def group_size(group=None) -> int:
    """Ranks in a process group; 1 when distributed is not in use."""
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group=group)


def group_rank(group=None) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank(group=group)


def _all_reduce(tensor: torch.Tensor, group) -> torch.Tensor:
    """Sum a tensor across the group, in place. A no-op for a group of one."""
    if group_size(group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def _all_gather_last_dim(tensor: torch.Tensor, group) -> torch.Tensor:
    size = group_size(group)
    if size == 1:
        return tensor
    tensor = tensor.contiguous()
    chunks = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=-1)


def _split_last_dim(tensor: torch.Tensor, group) -> torch.Tensor:
    size = group_size(group)
    if size == 1:
        return tensor
    return tensor.chunk(size, dim=-1)[group_rank(group)].contiguous()


# --- the f / g conjugate pair -------------------------------------------------------------------
#
# Each is the identity in one direction and an all-reduce in the other. The asymmetry falls out of
# the calculus, not from convention -- see docs/tensor_parallel.md.


class _CopyToTPRegion(torch.autograd.Function):
    """`f`: identity forward, all-reduce backward.

    Sits at the **input** of a column-parallel layer. The input is replicated and consumed by
    every rank, so each rank computes only a partial input gradient
    (`dL/dX = sum_i (dL/dY_i) W_i^T`) and the partials have to be summed.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Clone before reducing in place: autograd may hand us a buffer it still references.
        return _all_reduce(grad_output.clone(), ctx.group), None


class _ReduceFromTPRegion(torch.autograd.Function):
    """`g`: all-reduce forward, identity backward.

    Sits at the **output** of a row-parallel layer, whose output is a partial sum that must be
    completed. Because every rank then holds the identical full output, the incoming gradient is
    already complete and the backward needs no communication.
    """

    @staticmethod
    def forward(ctx, x, group):
        return _all_reduce(x.clone(), group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToTPRegion(torch.autograd.Function):
    """Split a replicated tensor along its last dimension; all-gather the gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _split_last_dim(x, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_gather_last_dim(grad_output, ctx.group), None


class _GatherFromTPRegion(torch.autograd.Function):
    """All-gather a sharded tensor along its last dimension; split the gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _all_gather_last_dim(x, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_last_dim(grad_output, ctx.group), None


def copy_to_tp_region(x: torch.Tensor, group=None) -> torch.Tensor:
    return _CopyToTPRegion.apply(x, group)


def reduce_from_tp_region(x: torch.Tensor, group=None) -> torch.Tensor:
    return _ReduceFromTPRegion.apply(x, group)


def scatter_to_tp_region(x: torch.Tensor, group=None) -> torch.Tensor:
    return _ScatterToTPRegion.apply(x, group)


def gather_from_tp_region(x: torch.Tensor, group=None) -> torch.Tensor:
    return _GatherFromTPRegion.apply(x, group)


# --- parallel linear layers ---------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """`Y = XW + b` with `W` split by columns, so each rank produces a slice of the output.

    No communication in the forward: the slices *are* different columns of the same answer. The
    `f` operator on the input supplies the backward all-reduce that the replicated input needs.

    `gather_output=False` is the useful case -- the next layer is row-parallel and wants a sharded
    input, so the intermediate is never gathered. `gather_output=True` exists to compare a single
    layer against `nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group=None,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        size = group_size(group)
        if out_features % size != 0:
            raise ConfigError(
                f"out_features ({out_features}) must be divisible by tp_size ({size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // size
        self.group = group
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_rank)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tp_region(x, self.group)  # f
        y = F.linear(x, self.weight, self.bias)
        return gather_from_tp_region(y, self.group) if self.gather_output else y

    @classmethod
    def from_linear(cls, linear: nn.Linear, group=None, gather_output: bool = False):
        """Build the sharded layer and copy this rank's slice out of the full weights."""
        size, rank = group_size(group), group_rank(group)
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group=group,
            gather_output=gather_output,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight.chunk(size, dim=0)[rank])
            if layer.bias is not None:
                layer.bias.copy_(linear.bias.chunk(size, dim=0)[rank])
        return layer


class RowParallelLinear(nn.Module):
    """`Y = X₁W₁ + X₂W₂ + b` with `W` split by rows, so each rank produces a partial sum.

    The forward needs one all-reduce (`g`) to complete the sum; the backward needs none, because
    the full output gradient is already on every rank.

    **The bias is replicated and added after the all-reduce.** Adding it before would include it
    once per rank, i.e. `tp_size` times -- the easiest mistake to make here, and the one that
    produces a plausible-looking but wrong model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group=None,
        input_is_parallel: bool = True,
    ) -> None:
        super().__init__()
        size = group_size(group)
        if in_features % size != 0:
            raise ConfigError(f"in_features ({in_features}) must be divisible by tp_size ({size})")
        self.in_features = in_features
        self.in_features_per_rank = in_features // size
        self.out_features = out_features
        self.group = group
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None  # replicated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = scatter_to_tp_region(x, self.group)
        y = F.linear(x, self.weight)  # no bias here
        y = reduce_from_tp_region(y, self.group)  # g
        return y if self.bias is None else y + self.bias  # bias after the reduction

    @classmethod
    def from_linear(cls, linear: nn.Linear, group=None, input_is_parallel: bool = True):
        size, rank = group_size(group), group_rank(group)
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group=group,
            input_is_parallel=input_is_parallel,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight.chunk(size, dim=1)[rank])
            if layer.bias is not None:
                layer.bias.copy_(linear.bias)  # replicated, not split
        return layer


# --- model surgery ------------------------------------------------------------------------------


def validate_tp(model_cfg: ModelConfig, tp_size: int, world_size: int = 1) -> None:
    """Check up front that the model can be split `tp_size` ways, reporting every problem."""
    errors = []
    if tp_size < 1:
        errors.append(f"tp_size must be at least 1, got {tp_size}")
    else:
        if model_cfg.n_heads % tp_size != 0:
            errors.append(
                f"n_heads ({model_cfg.n_heads}) must be divisible by tp_size ({tp_size}): "
                "tensor parallelism partitions whole attention heads"
            )
        if model_cfg.d_ff % tp_size != 0:
            errors.append(f"d_ff ({model_cfg.d_ff}) must be divisible by tp_size ({tp_size})")
        if world_size % tp_size != 0:
            errors.append(
                f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
            )
        elif tp_size != world_size:
            # tp_size < world_size means a real data-parallel dimension alongside TP -- 2D
            # TP+DP, which SCOPE.md puts out of scope for the basic version. Allowing it would
            # train *silently wrong*: the data would be sharded across the dp dimension with no
            # gradient all-reduce over it, so each dp group would diverge. Refuse instead.
            errors.append(
                f"tp_size ({tp_size}) must equal world_size ({world_size}): combining tensor "
                "and data parallelism (2D TP+DP) is not supported yet, and a smaller tp_size "
                "would leave the data-parallel dimension without gradient synchronization"
            )
    if errors:
        raise ConfigError("invalid tensor-parallel config:\n  - " + "\n  - ".join(errors))


def parallelize_model(model: nn.Module, group=None) -> nn.Module:
    """Replace each block's linear layers with their parallel equivalents, in place.

    Only the four `nn.Linear`s inside each block change; embeddings, the LM head and the norms
    stay replicated (the basic version, D8). Module *names* are preserved, so `state_dict` keys
    are identical to a single-device model's and sharding is a pure function of the key.

    The model must be identically initialized on every rank before this is called -- it is, since
    every rank seeds the same way -- so the slices taken here fit together into exactly the
    reference model.
    """
    size = group_size(group)
    for block in model.blocks:
        block.attn.qkv = ColumnParallelLinear.from_linear(block.attn.qkv, group)
        block.attn.out_proj = RowParallelLinear.from_linear(block.attn.out_proj, group)
        block.mlp.fc1 = ColumnParallelLinear.from_linear(block.mlp.fc1, group)
        block.mlp.fc2 = RowParallelLinear.from_linear(block.mlp.fc2, group)

        # Essential and easy to miss: after column-parallel QKV this rank holds only
        # n_heads/tp_size *whole* heads (which is why M1 chose the [h, 3, d_h] layout, D6), so
        # `split_qkv` must reshape with the local head count or it will mangle the slice.
        block.attn.n_heads = block.attn.n_heads // size
    return model


# --- gradient clipping ---------------------------------------------------------------------------


def clip_grad_norm_tp(
    model: nn.Module, max_norm: float, group=None, eps: float = 1e-6
) -> torch.Tensor:
    """Clip by the **global** gradient norm, across TP shards.

    `torch.nn.utils.clip_grad_norm_` cannot do this. Under tensor parallelism `parameters()`
    yields ordinary tensors that happen to be slices, with nothing marking them as such -- unlike
    FSDP, whose `DTensor` gradients carry their sharding and let the stock function reduce
    correctly. So each rank would compute a norm over its own slices plus the replicated
    parameters, which is neither the global norm nor the same on every rank.

    The global norm needs the two kinds of parameter treated differently:

    * **sharded** -- each rank holds a different slice, so `||g_full||^2 = sum_r ||g_shard_r||^2`
      and the squared norms must be all-reduced.
    * **replicated** -- every rank already holds the identical full gradient, so it is counted
      **once** and must not be reduced, or it would be counted `tp_size` times.

    Returns the pre-clip global norm, which is what the trainer logs.
    """
    sharded_sq: list[torch.Tensor] = []
    replicated_sq: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grads.append(param.grad)
        squared = param.grad.detach().float().pow(2).sum()
        (sharded_sq if tp_shard_dim(name) is not None else replicated_sq).append(squared)

    if not grads:
        return torch.zeros(())

    device = grads[0].device
    sharded_total = (
        torch.stack(sharded_sq).sum() if sharded_sq else torch.zeros((), device=device)
    )
    _all_reduce(sharded_total, group)  # sum the slices' contributions
    replicated_total = (
        torch.stack(replicated_sq).sum() if replicated_sq else torch.zeros((), device=device)
    )
    total_norm = (sharded_total + replicated_total).sqrt()

    if max_norm > 0 and math.isfinite(max_norm):
        coefficient = (max_norm / (total_norm + eps)).clamp(max=1.0)
        for grad in grads:
            grad.mul_(coefficient)
    return total_norm


# --- state dict sharding ------------------------------------------------------------------------


def tp_shard_dim(key: str) -> int | None:
    """Which dimension a parameter is split along under TP, or None if it is replicated."""
    if any(f"{prefix}." in key for prefix in COLUMN_PARALLEL):
        return 0  # both weight and bias are split along the output dimension
    if any(f"{prefix}." in key for prefix in ROW_PARALLEL):
        return 1 if key.endswith(".weight") else None  # row-parallel biases are replicated
    return None


def shard_state_dict(full: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> dict:
    """Split a full (single-device) state dict into one rank's TP shard."""
    out = {}
    for key, value in full.items():
        dim = tp_shard_dim(key)
        out[key] = value if dim is None else value.chunk(tp_size, dim=dim)[tp_rank].clone()
    return out


def gather_state_dict(shards: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Reassemble TP shards into a full state dict, the inverse of `shard_state_dict`."""
    if not shards:
        raise ValueError("no shards to gather")
    out = {}
    for key in shards[0]:
        dim = tp_shard_dim(key)
        out[key] = shards[0][key] if dim is None else torch.cat([s[key] for s in shards], dim=dim)
    return out


def gather_tp_optimizer_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer, group=None
) -> dict[str, Any]:
    """Gather a TP-sharded optimizer state into its full, single-device shape.

    Adam's `exp_avg` and `exp_avg_sq` have the same shape as the parameter they track, so under
    tensor parallelism they are sharded exactly as the parameters are -- and a consolidated
    checkpoint that gathered only the *model* would be internally inconsistent: full weights beside
    half-width moments. That file loads without complaint and then dies on the first optimizer
    step with a shape error naming nothing actionable (ISSUES #17).

    `torch` keys optimizer state by position in the flattened `param_groups` order, so the mapping
    back to parameter names -- and therefore to shard dimensions -- is reconstructed from the
    optimizer's own param groups rather than guessed.
    """
    size = group_size(group)
    state_dict = optimizer.state_dict()
    if size == 1:
        return state_dict

    name_of = {id(param): name for name, param in model.named_parameters()}
    flat = [param for pg in optimizer.param_groups for param in pg["params"]]

    gathered: dict[Any, Any] = {}
    for index, param in enumerate(flat):
        # Every rank holds the identical parameter structure (surgery is deterministic), so this
        # loop issues the same sequence of collectives on every rank. That matters: a rank-
        # dependent number of all-gathers would deadlock, which is the ISSUES #13 lesson.
        entry = state_dict["state"].get(index)
        if entry is None:
            continue  # no state yet: true on every rank before the first optimizer step
        dim = tp_shard_dim(name_of.get(id(param), ""))
        rebuilt = {}
        for key, value in entry.items():
            if dim is not None and torch.is_tensor(value) and value.dim() > dim:
                shards: list[Any] = [None] * size
                dist.all_gather_object(shards, value.detach().cpu(), group=group)
                rebuilt[key] = torch.cat(shards, dim=dim)
            else:
                # Replicated moments, and scalars like Adam's step counter, pass through.
                rebuilt[key] = value
        gathered[index] = rebuilt

    return {"state": gathered, "param_groups": state_dict["param_groups"]}


def gather_tp_state_dict(model: nn.Module, group=None) -> dict[str, torch.Tensor]:
    """All-gather this model's sharded parameters into a full state dict on every rank.

    Used for checkpointing: a TP checkpoint is written in the single-device format so it loads
    into a plain `GPT`, exactly like the FSDP consolidation path (D21).
    """
    size = group_size(group)
    local = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if size == 1:
        return local
    shards: list[dict] = [None] * size  # type: ignore[list-item]
    dist.all_gather_object(shards, local, group=group)
    return gather_state_dict(shards)
