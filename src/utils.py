# src/utils.py
from __future__ import annotations

import numpy as np
import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This repo requires a CUDA-capable GPU.")


def cuda_timer(fn, warmup: int = 30, iters: int = 200) -> float:
    """
    Time `fn()` on GPU using CUDA events.
    Returns average milliseconds per call.
    """
    require_cuda()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return float(total_ms) / float(iters)


def gemm_flops(M: int, N: int, K: int) -> float:
    return 2.0 * float(M) * float(N) * float(K)


def tflops_from_ms(flops: float, ms: float) -> float:
    sec = ms / 1000.0
    return float(flops) / max(sec, 1e-12) / 1e12


def iters_for_shape(M: int, N: int, K: int) -> int:
    """
    Heuristic iteration policy to keep total timed work roughly comparable.
    Tune for your runtime budget.
    """
    flops = float(gemm_flops(M, N, K))
    if flops < 1e10:        # < ~10 GFLOP
        return 2000
    if flops < 5e10:        # 10–50 GFLOP
        return 1000
    if flops < 2e11:        # 50–200 GFLOP
        return 400
    if flops < 8e11:        # 200–800 GFLOP
        return 200
    return 120


def summarize_ms(ms_list: list[float]) -> dict[str, float]:
    arr = np.array(ms_list, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    jitter = float(p90 / p50) if p50 > 0 else float("inf")
    return {"p50": p50, "p90": p90, "std": std, "jitter_p90_over_p50": jitter}


def dtype_bytes(dtype: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dtype).element_size())