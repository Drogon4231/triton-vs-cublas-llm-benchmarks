# src/bytes_model.py
from __future__ import annotations

import torch
from .utils import dtype_bytes


def bytes_io_fused(M: int, N: int, K: int, dtype: torch.dtype) -> float:
    """
    Approx global traffic for a *fused* kernel:
      - read A, read B, read bias, write C once

    Assumes epilogue is done in registers before final store.
    """
    b = dtype_bytes(dtype)
    bytes_A = M * K * b
    bytes_B = K * N * b
    bytes_bias = N * b
    bytes_Cw = M * N * b
    return float(bytes_A + bytes_B + bytes_bias + bytes_Cw)


def bytes_io_baseline(M: int, N: int, K: int, dtype: torch.dtype, epilogue: str) -> float:
    """
    Approx global traffic for baseline: cuBLAS GEMM + separate epilogue kernels.

    Baseline assumptions:
      - GEMM: read A, read B, write C once
      - bias add: read C + write C + read bias
      - activation (SiLU/fastGeLU): read C + write C
    """
    b = dtype_bytes(dtype)

    # GEMM pass
    bytes_A = M * K * b
    bytes_B = K * N * b
    bytes_Cw = M * N * b

    # Bias kernel pass (always for linear+bias)
    bytes_bias_kernel = 2 * (M * N * b) + (N * b)

    # Activation kernel pass (optional)
    bytes_act_kernel = 2 * (M * N * b) if epilogue in ("silu", "gelu") else 0

    return float(bytes_A + bytes_B + bytes_Cw + bytes_bias_kernel + bytes_act_kernel)


def gbps_from_ms(bytes_io: float, ms: float) -> float:
    return float((bytes_io / max(ms * 1e-3, 1e-12)) / 1e9)


def ai_flops_per_byte(flops: float, bytes_io: float) -> float:
    return float(flops / max(bytes_io, 1e-12))