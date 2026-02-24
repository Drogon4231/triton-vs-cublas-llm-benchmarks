# benchmarks/bench_linear_bias_epilogue.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as tF

from src.utils import require_cuda, cuda_timer, gemm_flops, tflops_from_ms, iters_for_shape
from src.utils import summarize_ms
from src.bytes_model import bytes_io_baseline, bytes_io_fused, gbps_from_ms, ai_flops_per_byte
from src.shapes import make_llm_rect_shapes
from src.correctness import default_tolerances, check_linear_bias_correctness
from src.kernels.linear_bias import triton_linear_bias, gelu_fast_torch


@torch.no_grad()
def bench_torch_linear_bias(A, B, bias, out, epilogue: str, warmup: int, iters: int) -> float:
    assert bias.ndim == 1 and bias.shape[0] == B.shape[1]

    def fn():
        torch.matmul(A, B, out=out)
        out.add_(bias)
        if epilogue == "silu":
            out.copy_(tF.silu(out))
        elif epilogue == "gelu":
            out.copy_(gelu_fast_torch(out))

    return cuda_timer(fn, warmup=warmup, iters=iters)


@torch.no_grad()
def bench_triton_linear_bias(A, B, bias, out, epilogue: str, warmup: int, iters: int) -> float:
    def fn():
        triton_linear_bias(A, B, bias, out=out, epilogue=epilogue)
    return cuda_timer(fn, warmup=warmup, iters=iters)


def main():
    parser = argparse.ArgumentParser(description="LLM-shaped fused linear+bias(+GELU/SiLU) benchmark.")
    parser.add_argument("--epilogue", type=str, default="gelu", choices=["bias", "gelu", "silu"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--dtypes", type=str, nargs="+", default=["fp16", "bf16"], choices=["fp16", "bf16"])
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    require_cuda()
    gpu_name = torch.cuda.get_device_name(0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    shapes = make_llm_rect_shapes()
    op_name = f"linear+bias+{args.epilogue}" if args.epilogue != "bias" else "linear+bias"

    print("GPU:", gpu_name)
    print("Op :", op_name)
    print(f"Total shapes: {len(shapes)}")

    # Quick correctness sanity check on one representative shape per dtype
    for dtype_key in args.dtypes:
        dtype = dtype_map[dtype_key]
        atol, rtol = default_tolerances(dtype, K=4096, epilogue=args.epilogue)
        ok, max_abs, max_rel = check_linear_bias_correctness(
            M=512, N=4096, K=4096, dtype=dtype, epilogue=args.epilogue, atol=atol, rtol=rtol
        )
        print(f"Correctness {dtype_key}: ok={ok} max_abs={max_abs:.3e} max_rel={max_rel:.3e} atol/rtol=({atol},{rtol})")
        assert ok, "Correctness check failed."

    rows = []
    TRIALS = args.trials
    WARMUP = args.warmup

    for (M, N, K) in shapes:
        for dtype_key in args.dtypes:
            dtype = dtype_map[dtype_key]
            iters = iters_for_shape(M, N, K)

            # Allocate once per (shape, dtype)
            A = torch.randn((M, K), device="cuda", dtype=dtype)
            B = torch.randn((K, N), device="cuda", dtype=dtype)
            bias = torch.randn((N,), device="cuda", dtype=dtype)
            out_torch = torch.empty((M, N), device="cuda", dtype=dtype)
            out_triton = torch.empty((M, N), device="cuda", dtype=dtype)

            # Force compile/tune outside timing
            _ = bench_triton_linear_bias(A, B, bias, out_triton, args.epilogue, warmup=0, iters=1)

            # Trial timing
            ms_cublas = []
            ms_triton = []

            for t in range(TRIALS):
                mt = bench_torch_linear_bias(A, B, bias, out_torch, args.epilogue, warmup=WARMUP, iters=iters)
                mr = bench_triton_linear_bias(A, B, bias, out_triton, args.epilogue, warmup=WARMUP, iters=iters)
                ms_cublas.append(mt)
                ms_triton.append(mr)

                print(
                    f"trial {t:02d} | {M}x{N}x{K} {dtype_key:>4} | "
                    f"cuBLAS {mt:7.3f} ms | Triton {mr:7.3f} ms"
                )

            c = summarize_ms(ms_cublas)
            r = summarize_ms(ms_triton)

            flops = float(gemm_flops(M, N, K))  # GEMM flops only
            c_tfl = float(tflops_from_ms(flops, c["p50"]))
            r_tfl = float(tflops_from_ms(flops, r["p50"]))

            c_bytes = bytes_io_baseline(M, N, K, dtype, epilogue=args.epilogue)
            r_bytes = bytes_io_fused(M, N, K, dtype)
            c_bw = gbps_from_ms(c_bytes, c["p50"])
            r_bw = gbps_from_ms(r_bytes, r["p50"])
            c_ai = ai_flops_per_byte(flops, c_bytes)
            r_ai = ai_flops_per_byte(flops, r_bytes)

            speedup = float(c["p50"] / r["p50"])
            shape_label = f"{M}x{N}x{K}"

            rows.append({
                "gpu": gpu_name,
                "backend": "cuBLAS",
                "op": op_name,
                "epilogue": args.epilogue,
                "M": M, "N": N, "K": K,
                "shape": shape_label,
                "dtype": dtype_key,
                "trials": TRIALS,
                "warmup": WARMUP,
                "iters": iters,
                "flops": flops,
                "latency_p50_ms": c["p50"],
                "latency_p90_ms": c["p90"],
                "latency_std_ms": c["std"],
                "jitter_p90_over_p50": c["jitter_p90_over_p50"],
                "gemm_equiv_tflops_p50": c_tfl,
                "bytes_io": c_bytes,
                "bandwidth_gbps_p50": c_bw,
                "ai_flops_per_byte": c_ai,
            })

            rows.append({
                "gpu": gpu_name,
                "backend": "Triton",
                "op": op_name,
                "epilogue": args.epilogue,
                "M": M, "N": N, "K": K,
                "shape": shape_label,
                "dtype": dtype_key,
                "trials": TRIALS,
                "warmup": WARMUP,
                "iters": iters,
                "flops": flops,
                "latency_p50_ms": r["p50"],
                "latency_p90_ms": r["p90"],
                "latency_std_ms": r["std"],
                "jitter_p90_over_p50": r["jitter_p90_over_p50"],
                "gemm_equiv_tflops_p50": r_tfl,
                "bytes_io": r_bytes,
                "bandwidth_gbps_p50": r_bw,
                "ai_flops_per_byte": r_ai,
                "speedup_vs_cublas_p50": speedup,
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"llm_rect_{op_name.replace('+','_')}_bench_{gpu_name.replace(' ','_')}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    if args.no_plots:
        return

    # (A) Throughput vs FLOPs
    plt.figure(figsize=(12, 6))
    for dtype_key in sorted(df["dtype"].unique()):
        for backend in ["cuBLAS", "Triton"]:
            sub = df[(df["dtype"] == dtype_key) & (df["backend"] == backend)].sort_values("flops")
            plt.plot(sub["flops"], sub["gemm_equiv_tflops_p50"], marker="o", label=f"{backend} {dtype_key}")
    plt.xlabel("Work (FLOPs per GEMM)")
    plt.ylabel("GEMM-Equivalent Throughput (TFLOP/s, p50)")
    plt.title(f"LLM-Shaped {op_name} Throughput on {gpu_name} (p50)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    out_png = plot_dir / f"llm_rect_{op_name.replace('+','_')}_throughput_p50.png"
    plt.savefig(out_png)
    plt.show()
    print("Saved:", out_png)

    # (B) Speedup vs FLOPs
    plt.figure(figsize=(12, 6))
    for dtype_key in sorted(df["dtype"].unique()):
        cublas_sub = df[(df["dtype"] == dtype_key) & (df["backend"] == "cuBLAS")].sort_values("flops")
        triton_sub = df[(df["dtype"] == dtype_key) & (df["backend"] == "Triton")].sort_values("flops")

        merged = cublas_sub.merge(
            triton_sub,
            on=["gpu", "op", "epilogue", "M", "N", "K", "shape", "dtype", "trials", "warmup", "iters", "flops"],
            suffixes=("_cublas", "_triton"),
            how="inner",
        ).sort_values("flops")

        speedup = merged["latency_p50_ms_cublas"] / merged["latency_p50_ms_triton"]
        plt.plot(merged["flops"], speedup, marker="o", label=f"{dtype_key}")

    plt.axhline(1.0, linewidth=1.0)
    plt.xlabel("Work (FLOPs per GEMM)")
    plt.ylabel("Speedup (cuBLAS p50 latency / Triton p50 latency)")
    plt.title(f"Triton Speedup over cuBLAS for LLM-Shaped {op_name} on {gpu_name} (p50)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    out_png = plot_dir / f"llm_rect_{op_name.replace('+','_')}_speedup_p50.png"
    plt.savefig(out_png)
    plt.show()
    print("Saved:", out_png)

    # (C) Bandwidth vs FLOPs (model)
    plt.figure(figsize=(12, 6))
    for dtype_key in sorted(df["dtype"].unique()):
        for backend in ["cuBLAS", "Triton"]:
            sub = df[(df["dtype"] == dtype_key) & (df["backend"] == backend)].sort_values("flops")
            plt.plot(sub["flops"], sub["bandwidth_gbps_p50"], marker="o", label=f"{backend} {dtype_key}")
    plt.xlabel("Work (FLOPs per GEMM)")
    plt.ylabel("Effective Bandwidth (GB/s, p50) [model]")
    plt.title(f"LLM-Shaped {op_name} Effective Bandwidth on {gpu_name} (p50)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    out_png = plot_dir / f"llm_rect_{op_name.replace('+','_')}_bandwidth_p50.png"
    plt.savefig(out_png)
    plt.show()
    print("Saved:", out_png)

    # (D) Jitter vs FLOPs
    plt.figure(figsize=(12, 6))
    for dtype_key in sorted(df["dtype"].unique()):
        for backend in ["cuBLAS", "Triton"]:
            sub = df[(df["dtype"] == dtype_key) & (df["backend"] == backend)].sort_values("flops")
            plt.plot(sub["flops"], sub["jitter_p90_over_p50"], marker="o", label=f"{backend} {dtype_key}")
    plt.xlabel("Work (FLOPs per GEMM)")
    plt.ylabel("Jitter (p90 / p50)")
    plt.title(f"LLM-Shaped {op_name} Latency Jitter on {gpu_name}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    out_png = plot_dir / f"llm_rect_{op_name.replace('+','_')}_jitter_p90_over_p50.png"
    plt.savefig(out_png)
    plt.show()
    print("Saved:", out_png)


if __name__ == "__main__":
    main()