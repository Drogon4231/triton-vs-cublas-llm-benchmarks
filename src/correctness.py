# src/correctness.py
from __future__ import annotations

import torch
import torch.nn.functional as tF

from .utils import require_cuda
from .kernels.matmul import triton_matmul
from .kernels.linear_bias import triton_linear_bias
from .kernels.linear_bias import gelu_fast_torch  # keep definition consistent


def default_tolerances(dtype: torch.dtype, K: int, epilogue: str = "bias") -> tuple[float, float]:
    """
    Numerical tolerances for GEMM + epilogue (heuristic).

    - Accumulation error grows with K
    - Nonlinear activations amplify differences
    """
    if epilogue == "silu":
        act_scale = 2.0
    elif epilogue == "gelu":
        act_scale = 2.5
    else:
        act_scale = 1.0

    if dtype == torch.float16:
        base = 2e-2
        atol = base * (K ** 0.5) / 32.0
        atol = max(atol, 5e-2) * act_scale
        rtol = 2e-2
        return float(atol), float(rtol)

    if dtype == torch.bfloat16:
        base = 6e-2
        atol = base * (K ** 0.5) / 32.0
        atol = max(atol, 1e-1) * act_scale
        rtol = 3e-2
        return float(atol), float(rtol)

    return 1e-5, 1e-5


@torch.no_grad()
def check_gemm_correctness(M: int, N: int, K: int, dtype: torch.dtype, atol: float, rtol: float, seed: int = 0):
    require_cuda()
    torch.manual_seed(seed)
    A = torch.randn((M, K), device="cuda", dtype=dtype)
    B = torch.randn((K, N), device="cuda", dtype=dtype)

    C_ref = torch.matmul(A, B)
    C_out = torch.empty((M, N), device="cuda", dtype=dtype)
    triton_matmul(A, B, out=C_out)

    diff = (C_out.float() - C_ref.float()).abs()
    max_abs = float(diff.max().item())
    denom = C_ref.float().abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item())
    ok = torch.allclose(C_out, C_ref, atol=atol, rtol=rtol)
    return ok, max_abs, max_rel


@torch.no_grad()
def check_linear_bias_correctness(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    epilogue: str,
    atol: float,
    rtol: float,
    seed: int = 0,
):
    require_cuda()
    torch.manual_seed(seed)
    A = torch.randn((M, K), device="cuda", dtype=dtype) * 0.1
    B = torch.randn((K, N), device="cuda", dtype=dtype) * 0.1
    bias = torch.randn((N,), device="cuda", dtype=dtype) * 0.1

    # Reference: GEMM + bias + epilogue (consistent w/ Triton)
    C_ref = torch.matmul(A, B)
    C_ref = C_ref + bias
    if epilogue == "silu":
        C_ref = tF.silu(C_ref)
    elif epilogue == "gelu":
        C_ref = gelu_fast_torch(C_ref)

    C_out = torch.empty((M, N), device="cuda", dtype=dtype)
    triton_linear_bias(A, B, bias, out=C_out, epilogue=epilogue)

    diff = (C_out.float() - C_ref.float()).abs()
    max_abs = float(diff.max().item())
    scale = C_ref.float().abs().mean().clamp_min(1e-6)
    max_rel = float((diff / scale).max().item())
    ok = torch.allclose(C_out, C_ref, atol=atol, rtol=rtol)
    return ok, max_abs, max_rel