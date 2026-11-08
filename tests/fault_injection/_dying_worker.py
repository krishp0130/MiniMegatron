"""A worker that dies mid-run, for the distributed failure test.

Run under torchrun. Rank 0 loops long enough to still be inside a collective when rank 1 exits,
so the test exercises what happens to the *survivors*, not just the exit code of the casualty.
"""

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    if rank == 1:
        print(f"[rank {rank}] exiting deliberately", flush=True)
        # os._exit skips cleanup, which is the point: a real crash does not say goodbye.
        os._exit(17)

    print(f"[rank {rank}] entering collectives", flush=True)
    for i in range(1000):
        dist.all_reduce(torch.ones(1))  # rank 0 will block here once rank 1 is gone
    print(f"[rank {rank}] finished unexpectedly", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
