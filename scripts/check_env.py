"""Print the hardware and software environment MiniMegatron will run on."""

import platform
import sys

import torch
import torch.distributed as dist


def main() -> None:
    print(f"platform     {platform.platform()}")
    print(f"processor    {platform.processor() or platform.machine()}")
    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")

    print(f"cuda         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda ver   {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu {i}      {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB")
        print(f"  bf16       {torch.cuda.is_bf16_supported()}")
    print(f"mps          {torch.backends.mps.is_available()}")

    print(f"distributed  {dist.is_available()}  (nccl={dist.is_nccl_available()}, gloo={dist.is_gloo_available()})")

    try:
        import triton

        print(f"triton       {triton.__version__}")
    except ImportError:
        print("triton       not installed")


if __name__ == "__main__":
    main()
