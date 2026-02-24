# src/kernels/linear_bias.py
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---- Keep Torch ref + Triton epilogue consistent ----
def gelu_fast_torch(x: torch.Tensor) -> torch.Tensor:
    # gelu(x) ≈ x * sigmoid(1.702 * x)
    return x * torch.sigmoid(1.702 * x)


@triton.jit
def gelu_fast_triton(x):
    return x * tl.sigmoid(1.702 * x)


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32},  num_warps=4,  num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=8,  num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=8,  num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8,  num_stages=5),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8,  num_stages=6),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8,  num_stages=5),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8,  num_stages=5),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=16, num_stages=7),
        triton.Config({"BM": 256, "BN": 256, "BK": 128}, num_warps=16, num_stages=8),
    ],
    key=["M", "N", "K", "dtype_id", "DO_SILU", "DO_GELU"],
)
@triton.jit
def linear_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    dtype_id: tl.constexpr,
    DO_SILU: tl.constexpr,
    DO_GELU: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        k = k0 + offs_k

        ptrs_a = A_ptr + (offs_m[:, None] * stride_am + k[None, :] * stride_ak)
        ptrs_b = B_ptr + (k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(ptrs_a, mask=mask_a, other=0.0)
        b = tl.load(ptrs_b, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    # Fused bias: bias shape (N,)
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Optional epilogues
    if DO_SILU:
        acc = acc * tl.sigmoid(acc)
    elif DO_GELU:
        acc = gelu_fast_triton(acc)

    out = acc.to(tl.float16) if dtype_id == 0 else acc.to(tl.bfloat16)

    ptrs_c = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptrs_c, out, mask=mask_c)


@torch.no_grad()
def triton_linear_bias(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    epilogue: str = "bias",
) -> torch.Tensor:
    """
    Fused: C = A @ B + bias; optional epilogue in {"bias","silu","gelu"}.

    A: (M,K), B: (K,N), bias: (N,), out: (M,N)
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda and out.is_cuda
    assert A.dtype == B.dtype == bias.dtype == out.dtype
    assert A.shape[1] == B.shape[0]
    assert bias.ndim == 1 and bias.shape[0] == B.shape[1]
    assert out.shape[0] == A.shape[0] and out.shape[1] == B.shape[1]

    A = A.contiguous()
    B = B.contiguous()
    bias = bias.contiguous()

    M, K = A.shape
    _, N = B.shape
    dtype_id = 0 if A.dtype == torch.float16 else 1

    DO_SILU = (epilogue == "silu")
    DO_GELU = (epilogue == "gelu")
    if epilogue not in ("bias", "silu", "gelu"):
        raise ValueError(f"Unsupported epilogue: {epilogue}")

    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    linear_bias_kernel[grid](
        A, B, bias, out,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        dtype_id=dtype_id,
        DO_SILU=DO_SILU,
        DO_GELU=DO_GELU,
    )
    return out