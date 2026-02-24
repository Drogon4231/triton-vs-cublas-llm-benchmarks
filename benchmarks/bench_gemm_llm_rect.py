# benchmarks/bench_gemm_llm_rect.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.utils import require_cuda, cuda_timer, gemm_flops, tflops_from_ms, summarize_ms
from src.kernels.matmul import triton_matmul
from src.shapes import make_llm_rect_shapes


@torch.no_grad()
def bench_torch_gemm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, warmup: int, iters: int) -> float:
    def fn():
        torch.matmul(A, B, out=out)
    return cuda_timer(fn, warmup=warmup, iters=iters)


@torch.no_grad()
def bench_triton_gemm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, warmup: int, iters: int) -> float:
    def fn():
        triton_matmul(A, B, out=out)
    return cuda_timer(fn, warmup=warmup, iters=iters)


def main():
    parser = argparse.ArgumentParser(description="LLM-rect GEMM benchmark: Triton vs cuBLAS (PyTorch matmul).")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtypes", type=str, nargs="+", default=["fp16", "bf16"], choices=["fp16", "bf16"])
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--plot_dir", type=str, default="plots")
    args = parser.parse_args()

    require_cuda()
    gpu_name = torch.cuda.get_device_name(0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    shapes = make_llm_rect_shapes()
    print(f"Total shapes: {len(shapes)}")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}

    rows = []
    for (M, N, K) in shapes:
        for dtype_key in args.dtypes:
            dtype = dtype_map[dtype_key]

            A = torch.randn((M, K), device="cuda", dtype=dtype)
            B = torch.randn((K, N), device="cuda", dtype=dtype)
            C_torch = torch.empty((M, N), device="cuda", dtype=dtype)
            C_triton = torch.empty((M, N), device="cuda", dtype=dtype)

            # Force compile/tune outside timing
            _ = bench_triton_gemm(A, B, C_triton, warmup=0, iters=1)

            ms_torch = []
            ms_triton = []

            for t in range(args.trials):
                mt = bench_torch_gemm(A, B, C_torch, warmup=args.warmup, iters=args.iters)
                mr = bench_triton_gemm(A, B, C_triton, warmup=args.warmup, iters=args.iters)
                ms_torch.append(mt)
                ms_triton.append(mr)
                print(f"trial {t:02d} | {M}x{N}x{K} {dtype_key:>4} | torch {mt:7.3f} ms | triton {mr:7.3f} ms")

            torch_sum = summarize_ms(ms_torch)
            triton_sum = summarize_ms(ms_triton)

            flops = float(gemm_flops(M, N, K))
            torch_tfl = float(tflops_from_ms(flops, torch_sum["p50"]))
            triton_tfl = float(tflops_from_ms(flops, triton_sum["p50"]))
            speedup = float(torch_sum["p50"] / triton_sum["p50"])

            rows.append({
                "gpu": gpu_name,
                "op": "gemm_llm_rect",
                "backend": "cuBLAS",
                "dtype": dtype_key,
                "M": M, "N": N, "K": K,
                "shape": f"{M}x{N}x{K}",
                "flops": flops,
                "trials": args.trials,
                "warmup": args.warmup,
                "iters": args.iters,
                "latency_p50_ms": torch_sum["p50"],
                "latency_p90_ms": torch_sum["p90"],
                "jitter_p90_over_p50": torch_sum["jitter_p90_over_p50"],
                "gemm_equiv_tflops_p50": torch_tfl,
            })
            rows.append({
                "gpu": gpu_name,
                "op": "gemm_llm_rect",
                "backend": "Triton",
                "dtype": dtype_key,
                "M": M, "N": N, "K": K,
                "shape": f"{M}x{N}x{K}",
                "flops": flops,
                "trials": args.trials,
                "warmup": args.warmup,
                "iters": args.iters,
                "latency_p50_ms": triton_sum["p50"],
                "latency_p90_ms": triton_sum["p90"],
                "jitter_p90_over_p50": triton_sum["jitter_p90_over_p50"],
                "gemm_equiv_tflops_p50": triton_tfl,
                "speedup_vs_cublas_p50": speedup,
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"gemm_llm_rect_bench_{gpu_name.replace(' ','_')}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # Plot: TFLOPs vs FLOPs
    plt.figure(figsize=(12, 6))
    for dtype_key in args.dtypes:
        for backend in ["cuBLAS", "Triton"]:
            sub = df[(df["dtype"] == dtype_key) & (df["backend"] == backend)].sort_values("flops")
            plt.plot(sub["flops"], sub["gemm_equiv_tflops_p50"], marker="o", label=f"{backend} {dtype_key}")
    plt.xlabel("Work (FLOPs)")
    plt.ylabel("GEMM Throughput (TFLOP/s, p50)")
    plt.title(f"LLM-Rect GEMM Throughput (p50) on {gpu_name}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    out_png = plot_dir / "llm_rect_gemm_tflops_p50.png"
    plt.savefig(out_png)
    plt.show()
    print("Saved:", out_png)

    # Plot: Speedup vs FLOPs
    plt.figure(figsize=(12, 6))
    for dtype_key in args.dtypes:
        cublas = df[(df["dtype"] == dtype_key) & (df["backend"] == "cuBLAS")].sort_values("flops")
        triton = df[(df["dtype"] == dtype_key) & (df["backend"] == "Triton")].sort_values("flops")
        merged = cublas.merge(triton, on=["gpu", "op", "dtype", "M", "N", "K", "shape", "flops"], suffixes=("_c", "_t"))
        sp = merged["latency_p50_ms_c"] / merged["latency_p50_ms_t"]
        plt.plot(merged["flops"], sp, marker="o", label=f"{dtype_key}")
    plt.axhline(1.0, linewidth=1.0)
    plt.xlabel("Work (FLOPs)")
    plt.ylabel("Speedup (cuBLAS p50 / Triton p50)")
    plt.title(f"Triton Speedup (p50) for LLM-Rect GEMM on {gpu_name}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    out_png = plot_dir / "llm_rect_gemm_speedup_p50.png"
    plt.savefig(out_png)
    plt.show()
    print("Saved:", out_png)


if __name__ == "__main__":
    main()