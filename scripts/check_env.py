"""Print the hardware and software environment MiniMegatron will run on.

The same probe the benchmark harness records into every result, so an environment report and a
benchmark record can never disagree about what the machine is.
"""

import torch.distributed as dist

from minimegatron.benchmarks.harness import environment_info


def main() -> None:
    info = environment_info()

    print(f"platform     {info['platform']}")
    print(f"processor    {info['processor']}")
    print(f"python       {info['python']}")
    print(f"torch        {info['torch']}")

    print(f"cuda         {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"  cuda ver   {info['cuda_version']}")
        print(f"  gpus       {info['gpu_count']}")
        print(f"  gpu 0      {info['gpu_name']}  {info['gpu_arch']}")
        print(f"  bf16       {info['bf16_supported']}")
    print(f"mps          {info['mps_available']}")

    print(
        f"distributed  {dist.is_available()}  "
        f"(nccl={dist.is_nccl_available()}, gloo={dist.is_gloo_available()})"
    )
    print(f"triton       {info['triton'] or 'not installed'}")


if __name__ == "__main__":
    main()
