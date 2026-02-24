# src/kernels/matmul.py
from __future__ import annotations

import torch
import triton
import triton.language as tl


def early_prune(configs, meta):
    M, N, K = meta["M"], meta["N"], meta["K"]
    pruned = []
    for c in configs:
        BM = c.kwargs["BM"]
        BN = c.kwargs["BN"]
        BK = c.kwargs["BK"]
        nw = c.num_warps

        # Skip tiles bigger than the problem for small shapes (wasteful)
        if BM > M and M <= 256:
            continue
        if BN > N and N <= 256:
            continue

        # For small K, large BK is wasteful
        if K <= 1024 and BK > 64:
            continue

        # Avoid warp-heavy configs for small problems
        if (M * N) <= (256 * 256) and nw >= 16:
            continue

        pruned.append(c)

    return pruned if pruned else configs


def simple_perf_model(meta, num_warps, num_stages):
    """
    Very rough "lower is better" score to prune autotune configs.
    It's intentionally simple and only used to reduce benchmarking time.
    """
    M, N, K = meta["M"], meta["N"], meta["K"]
    BM, BN, BK = meta["BM"], meta["BN"], meta["BK"]

    tile_flops = 2.0 * BM * BN * K
    bytes_per_kstep = 2.0 * (BM * BK + BK * BN)  # fp16/bf16 bytes
    bytes_c = 2.0 * (BM * BN)
    steps = (K + BK - 1) // BK
    tile_bytes = steps * bytes_per_kstep + bytes_c

    stage_bonus = 1.0 + 0.15 * min(num_stages, 6)
    warp_penalty = 1.0 + 0.08 * max(0, num_warps - 4)

    score = (tile_bytes / stage_bonus) * warp_penalty / max(tile_flops, 1.0)
    return score


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32},  num_warps=4,  num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=8,  num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=8,  num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8,  num_stages=5),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8,  num_stages=6),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8,  num_stages=5),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8,  num_stages=5),
        triton.Config({"BM": 256, "BN": 128, "BK": 64}, num_warps=8,  num_stages=6),
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8,  num_stages=6),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=16, num_stages=7),
        triton.Config({"BM": 256, "BN": 256, "BK": 128}, num_warps=16, num_stages=8),
    ],
    key=["M", "N", "K", "dtype_id"],
    prune_configs_by={
        "early_config_prune": early_prune,
        "perf_model": simple_perf_model,
        "top_k": 6,
    },
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    dtype_id: tl.constexpr,
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

    out = acc.to(tl.float16) if dtype_id == 0 else acc.to(tl.bfloat16)

    ptrs_c = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptrs_c, out, mask=mask_c)


@torch.no_grad()
def triton_matmul(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    A: (M,K), B: (K,N), out: (M,N)
    """
    assert A.is_cuda and B.is_cuda and out.is_cuda
    assert A.dtype == B.dtype == out.dtype
    assert A.shape[1] == B.shape[0]
    assert out.shape[0] == A.shape[0] and out.shape[1] == B.shape[1]

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    _, N = B.shape
    dtype_id = 0 if A.dtype == torch.float16 else 1

    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    matmul_kernel[grid](
        A, B, out,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        dtype_id=dtype_id,
    )
    return out