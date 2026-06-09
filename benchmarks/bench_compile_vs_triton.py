# benchmarks/bench_compile_vs_triton.py
"""Does the hand-fused Triton kernel still beat torch.compile?

bench_linear_bias_epilogue.py compares the fused Triton linear+bias+act kernel
against UNFUSED eager PyTorch (matmul + add + act = three kernel launches). The
sharper question a reviewer asks first: does the hand kernel beat what the
COMPILER gives you for free? torch.compile (inductor) fuses the pointwise
epilogue (bias + activation) into/after the matmul, so it is the realistic
baseline a 2026 practitioner would actually ship.

Three backends per (shape, dtype):
  eager    : torch.matmul + add + act    (unfused, 3 launches)
  compile  : torch.compile(eager_fn)     (inductor fuses the epilogue)
  triton   : triton_linear_bias          (hand-fused autotuned kernel)

The column that matters is speedup_vs_compile. If triton > compile, the hand
kernel earns its keep; if not, that is an honest, sophisticated finding (the
compiler already closes the gap), which is a *better* portfolio story than
beating only the naive eager baseline.

NOTE: needs a CUDA GPU; NOT validated on the author's M3 Pro — run on the
A100/Colab/H100 and report back. torch.compile recompiles on shape change, so
the first touch per shape is slow; compilation is triggered OUTSIDE the timed
region. With --compile-mode max-autotune the per-shape compile is much slower
but the baseline is stronger.

Usage:
  python benchmarks/bench_compile_vs_triton.py --epilogue gelu --dtypes fp16
  python benchmarks/bench_compile_vs_triton.py --compile-mode max-autotune
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root so `src` imports

import pandas as pd
import torch
import torch.nn.functional as tF

from src.utils import (
    require_cuda, cuda_timer, gemm_flops, tflops_from_ms,
    iters_for_shape, summarize_ms,
)
from src.shapes import make_llm_rect_shapes
from src.kernels.linear_bias import triton_linear_bias, gelu_fast_torch


def eager_linear_bias(A, B, bias, epilogue: str):
    out = torch.matmul(A, B)
    out = out + bias
    if epilogue == "gelu":
        out = gelu_fast_torch(out)
    elif epilogue == "silu":
        out = tF.silu(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epilogue", default="gelu", choices=["bias", "gelu", "silu"])
    ap.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"], choices=["fp16", "bf16"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--compile-mode", dest="compile_mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--out-dir", dest="out_dir", default="data")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N shapes (quick smoke)")
    args = ap.parse_args()

    require_cuda()
    gpu = torch.cuda.get_device_name(0)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dtmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
    shapes = make_llm_rect_shapes()
    if args.limit:
        shapes = shapes[:args.limit]
    print(f"GPU: {gpu} | epilogue: {args.epilogue} | "
          f"compile-mode: {args.compile_mode} | shapes: {len(shapes)}")

    compiled = torch.compile(eager_linear_bias, mode=args.compile_mode)

    rows = []
    for (M, N, K) in shapes:
        for dk in args.dtypes:
            dt = dtmap[dk]
            iters = iters_for_shape(M, N, K)
            A = torch.randn((M, K), device="cuda", dtype=dt)
            B = torch.randn((K, N), device="cuda", dtype=dt)
            bias = torch.randn((N,), device="cuda", dtype=dt)
            out_tri = torch.empty((M, N), device="cuda", dtype=dt)

            def eager_fn():
                return eager_linear_bias(A, B, bias, args.epilogue)

            def comp_fn():
                return compiled(A, B, bias, args.epilogue)

            def tri_fn():
                return triton_linear_bias(A, B, bias, out=out_tri, epilogue=args.epilogue)

            # Trigger compile + autotune OUTSIDE the timed region.
            ref = eager_fn()
            for _ in range(3):
                comp_fn()
                tri_fn()
            torch.cuda.synchronize()

            # One-time correctness vs the eager reference.
            comp_out = comp_fn()
            atol = 2e-2 if dk == "fp16" else 3e-2
            comp_ok = bool(torch.allclose(comp_out.float(), ref.float(), atol=atol, rtol=atol))
            tri_ok = bool(torch.allclose(out_tri.float(), ref.float(), atol=atol, rtol=atol))

            e_ms, c_ms, t_ms = [], [], []
            for _ in range(args.trials):
                e_ms.append(cuda_timer(eager_fn, warmup=args.warmup, iters=iters))
                c_ms.append(cuda_timer(comp_fn, warmup=args.warmup, iters=iters))
                t_ms.append(cuda_timer(tri_fn, warmup=args.warmup, iters=iters))
            es, cs, tts = summarize_ms(e_ms), summarize_ms(c_ms), summarize_ms(t_ms)
            flops = gemm_flops(M, N, K)

            rows.append({
                "gpu": gpu, "epilogue": args.epilogue, "compile_mode": args.compile_mode,
                "M": M, "N": N, "K": K, "shape": f"{M}x{N}x{K}", "dtype": dk, "flops": flops,
                "eager_p50_ms": es["p50"], "compile_p50_ms": cs["p50"], "triton_p50_ms": tts["p50"],
                "triton_tflops_p50": tflops_from_ms(flops, tts["p50"]),
                "compile_tflops_p50": tflops_from_ms(flops, cs["p50"]),
                "speedup_vs_eager": es["p50"] / tts["p50"],
                "speedup_vs_compile": cs["p50"] / tts["p50"],
                "compile_speedup_vs_eager": es["p50"] / cs["p50"],
                "compile_correct": comp_ok, "triton_correct": tri_ok,
            })
            print(f"{M}x{N}x{K} {dk:>4} | eager {es['p50']:7.3f}  compile {cs['p50']:7.3f}  "
                  f"triton {tts['p50']:7.3f} ms | tri/compile {rows[-1]['speedup_vs_compile']:.2f}x"
                  f"{'' if (comp_ok and tri_ok) else '  [CORRECTNESS WARN]'}")

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"compile_vs_triton_{args.epilogue}_{gpu.replace(' ', '_')}_{ts}.csv"
    df.to_csv(p, index=False)

    med_ve = df["speedup_vs_eager"].median()
    med_vc = df["speedup_vs_compile"].median()
    wins = (df["speedup_vs_compile"] > 1.0).mean() * 100
    idx = df["speedup_vs_compile"].idxmax()
    print(f"\nSaved: {p}")
    print(f"median Triton vs eager:   {med_ve:.3f}x")
    print(f"median Triton vs compile: {med_vc:.3f}x   "
          f"(Triton wins on {wins:.0f}% of shapes)")
    print(f"max Triton vs compile:    {df['speedup_vs_compile'].max():.3f}x "
          f"at {df.loc[idx, 'shape']} {df.loc[idx, 'dtype']}")
    if not (df["compile_correct"].all() and df["triton_correct"].all()):
        print("WARNING: some correctness checks failed — investigate before trusting timings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
