# scripts/roofline.py
"""Roofline analysis for the LLM-shaped GEMM benchmarks.

Builds an A100 roofline from the *committed measurements* (no GPU required):
each shape's achieved GEMM throughput (TFLOP/s, p50) is plotted against its
operational intensity (FLOP / byte of DRAM traffic), against the A100's
compute and memory ceilings. This is what explains the two headline results
at once:

  - small-batch LLM GEMMs sit LEFT of the ridge point  -> memory-bound, so
    cutting an HBM round-trip via epilogue fusion is what wins there (the
    1.73x fused result);
  - large-batch GEMMs sit near the compute roof          -> compute-bound, so
    cuBLAS's tuned tensor-core scheduling dominates a bare Triton GEMM
    (the ~0.78x bare-GEMM median).

Operational intensity uses the algorithmic-minimum DRAM traffic for a GEMM
(read A once, read B once, write C once); real kernels move more (tile
re-reads), so this is the *ideal* OI and the achieved-vs-roof gap is the
efficiency headroom.

Usage:
    python scripts/roofline.py            # auto-find newest bare-GEMM CSV
    python scripts/roofline.py --csv data/gemm_llm_rect_bench_*.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── A100-SXM4-80GB hardware ceilings ─────────────────────────────────
# Dense fp16/bf16 tensor-core peak (no sparsity) and HBM2e bandwidth.
A100_PEAK_TFLOPS = 312.0      # TFLOP/s, fp16/bf16 tensor core, dense
A100_HBM_GBPS = 2039.0        # GB/s, 80 GB SXM4
RIDGE_OI = A100_PEAK_TFLOPS * 1e3 / A100_HBM_GBPS   # FLOP/byte at the ridge

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "fp16": 2, "bf16": 2}


def operational_intensity(M, N, K, dtype) -> float:
    """FLOP per byte of algorithmic-minimum DRAM traffic (A,B read once; C written once)."""
    b = DTYPE_BYTES.get(str(dtype), 2)
    flops = 2.0 * M * N * K
    dram_bytes = (M * K + K * N + M * N) * b
    return flops / dram_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="bare-GEMM benchmark CSV (default: newest data/gemm_llm_rect*.csv)")
    ap.add_argument("--out", type=str, default="plots/roofline_a100_gemm.png")
    args = ap.parse_args()

    csv = args.csv or sorted(glob.glob("data/gemm_llm_rect*.csv"))[-1]
    df = pd.read_csv(csv)
    df["oi"] = [operational_intensity(m, n, k, d)
                for m, n, k, d in zip(df.M, df.N, df.K, df.dtype)]

    tcol = "tflops_p50"
    backends = sorted(df.backend.unique())               # e.g. ['torch', 'triton']
    label = {"triton": "Triton", "torch": "cuBLAS (PyTorch)",
             "cuBLAS": "cuBLAS (PyTorch)", "Triton": "Triton"}
    color = {"triton": "#FF6A4D", "Triton": "#FF6A4D",
             "torch": "#41D6C3", "cuBLAS": "#41D6C3"}

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── ceilings ──
    oi = np.logspace(np.log10(df.oi.min() * 0.5), np.log10(df.oi.max() * 2), 200)
    mem_roof = A100_HBM_GBPS * oi / 1e3                   # GB/s * FLOP/byte -> TFLOP/s
    roof = np.minimum(mem_roof, A100_PEAK_TFLOPS)
    ax.plot(oi, roof, "k-", lw=2, zorder=5)
    ax.axhline(A100_PEAK_TFLOPS, ls="--", color="#888", lw=1)
    ax.text(oi[-1], A100_PEAK_TFLOPS * 1.02, f"compute roof {A100_PEAK_TFLOPS:.0f} TFLOP/s",
            ha="right", va="bottom", fontsize=9, color="#555")
    ax.axvline(RIDGE_OI, ls=":", color="#888", lw=1)
    xtr = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    ax.text(RIDGE_OI * 1.06, 0.04, f"ridge {RIDGE_OI:.0f} FLOP/byte",
            transform=xtr, rotation=90, va="bottom", ha="left",
            fontsize=8.5, color="#555")
    ax.text(RIDGE_OI * 0.92, 0.96, "memory-bound", transform=xtr,
            ha="right", va="top", fontsize=9, color="#777")
    ax.text(RIDGE_OI * 1.10, 0.96, "compute-bound", transform=xtr,
            ha="left", va="top", fontsize=9, color="#777")

    # ── measured points ──
    for be in backends:
        sub = df[df.backend == be]
        ax.scatter(sub.oi, sub[tcol], s=34, alpha=0.8,
                   color=color.get(be, "#999"), edgecolor="white", lw=0.4,
                   label=label.get(be, be), zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational intensity (FLOP / byte of DRAM, algorithmic min)")
    ax.set_ylabel("Achieved GEMM throughput (TFLOP/s, p50)")
    ax.set_title("A100-SXM4-80GB roofline — LLM-shaped GEMMs (fp16/bf16)")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}")

    # ── textual breakdown (the narrative) ──
    df["regime"] = np.where(df.oi < RIDGE_OI, "memory-bound", "compute-bound")
    tri = df[df.backend.str.lower() == "triton"]
    print(f"\nCSV: {csv}   ({len(df)} rows, {df[['M','N','K']].drop_duplicates().shape[0]} shapes)")
    print(f"Ridge point: {RIDGE_OI:.0f} FLOP/byte "
          f"(peak {A100_PEAK_TFLOPS:.0f} TFLOP/s / {A100_HBM_GBPS:.0f} GB/s)")
    for regime in ("memory-bound", "compute-bound"):
        t = tri[tri.regime == regime]
        if len(t):
            frac = t[tcol] / A100_PEAK_TFLOPS
            print(f"  {regime:>14}: {len(t):3d} Triton shapes | "
                  f"median {t[tcol].median():6.1f} TFLOP/s "
                  f"({100*frac.median():4.1f}% of peak) | "
                  f"OI {t.oi.min():.0f}-{t.oi.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
