"""What DDP's gradient synchronization actually does, written out by hand.

Run it:

    GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 \
      --master_addr=127.0.0.1 --master_port=29501 \
      -m minimegatron.distributed.collectives_demo

This module is deliberately **outside** the training path: the real trainer uses
`DistributedDataParallel`, because reimplementing production collectives in the code that does
the work would be a bad trade (SCOPE.md section 4). The point here is to make visible the one
operation DDP hides behind an autograd hook.

The claim being demonstrated: for a global batch split into equal shards, averaging each rank's
gradient gives exactly the gradient of the whole batch, because the loss is a mean over samples
and differentiation is linear:

    grad L(B) = (1/R) * sum_r grad L(B_r)

So each rank computes its own gradient and a single all-reduce turns the R different gradients
into one shared, correct gradient. After it, every rank holds identical gradients, takes an
identical optimizer step, and the replicas stay in lockstep without ever exchanging weights
again.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from minimegatron.data.synthetic import BatchStream, RandomTokenDataset
from minimegatron.distributed.setup import destroy_distributed, init_distributed
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import GPT

DEMO_CONFIG = ModelConfig(vocab_size=128, seq_len=16, n_layers=2, n_heads=4, d_model=32, d_ff=64)


def manual_average_gradients(model: nn.Module, world_size: int) -> None:
    """Average every gradient across ranks, in place: DDP's job, done explicitly.

    `all_reduce(SUM)` leaves every rank holding the sum over all ranks, so dividing by the world
    size turns it into the mean. This is a *synchronous* pass over parameters after the whole
    backward has finished; DDP instead buckets gradients and fires each bucket's all-reduce as
    soon as it fills, so communication overlaps with the rest of the backward. Same arithmetic,
    much better wall clock -- which is the entire reason to use the real thing.
    """
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size


def local_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One forward/backward on this rank's shard. Returns the loss."""
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    return loss.detach()


def main() -> int:
    info = init_distributed()
    try:
        if not info.is_distributed:
            print(
                "This demo needs at least 2 processes. Launch it with torchrun:\n"
                "  GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 \\\n"
                "    --master_addr=127.0.0.1 --master_port=29501 \\\n"
                "    -m minimegatron.distributed.collectives_demo"
            )
            return 1

        # nccl collectives require CUDA tensors, so the demo must follow the backend onto the
        # device. Under gloo on CPU this is a no-op; under nccl it is the difference between
        # working and `no support for CPU tensors` (ISSUES #20).
        device = (
            torch.device(f"cuda:{info.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        torch.manual_seed(0)  # identical weights on every rank
        model = GPT(DEMO_CONFIG).to(device)

        # Each rank takes its own shard of one global batch of 4 * world_size sequences.
        dataset = RandomTokenDataset(DEMO_CONFIG.vocab_size, DEMO_CONFIG.seq_len, 256, seed=0)
        stream = BatchStream(dataset, 4, seed=0, rank=info.rank, world_size=info.world_size)
        x, y = stream.micro_batch(0)

        loss = local_gradients(model, x.to(device), y.to(device))
        probe = model.blocks[0].mlp.fc1.weight  # one parameter, followed all the way through
        before = probe.grad.norm().item()

        manual_average_gradients(model, info.world_size)
        after = probe.grad.norm().item()

        print(
            f"[rank {info.rank}/{info.world_size}] {device} | loss {loss.item():.6f} | "
            f"grad norm local {before:.6f} -> averaged {after:.6f}",
            flush=True,
        )
        dist.barrier()
        if info.is_main:
            print(
                "\nThe local norms differ: each rank saw different data.\n"
                "The averaged norms are identical on every rank, and equal what one process\n"
                "would have computed over the whole batch. That is the whole of data parallelism.",
                flush=True,
            )
        return 0
    finally:
        destroy_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
