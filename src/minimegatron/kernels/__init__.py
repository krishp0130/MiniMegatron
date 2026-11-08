"""Custom GPU kernels and the PyTorch references they are checked against."""

from minimegatron.kernels.rmsnorm import (
    NORM_BACKENDS,
    TRITON_AVAILABLE,
    resolve_norm_backend,
    rms_norm_dispatch,
    rms_norm_triton,
    triton_usable,
)

__all__ = [
    "NORM_BACKENDS",
    "TRITON_AVAILABLE",
    "resolve_norm_backend",
    "rms_norm_dispatch",
    "rms_norm_triton",
    "triton_usable",
]
