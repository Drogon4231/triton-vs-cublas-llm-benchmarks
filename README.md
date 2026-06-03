# Triton vs cuBLAS: LLM Kernel Benchmarking Suite

Benchmarks **Triton fused GPU kernels** against **cuBLAS / PyTorch** baselines for the matmul shapes that dominate LLM inference — projection and FFN GEMMs — and measures where hand-written fusion actually wins. Run on an **NVIDIA A100-SXM4-80GB**.

It reports **p50 latency, GEMM-equivalent TFLOP/s, effective memory bandwidth, and p90/p50 tail jitter** across **76 LLM-shaped GEMM shapes**, and isolates the regime where a fused `linear + bias + activation` Triton kernel beats a separate cuBLAS GEMM + epilogue.

## Key results (A100-SXM4-80GB, fp16)

| Result | Value | Where |
|---|---|---|
| Fused `linear+bias+GeLU` vs cuBLAS, best small-batch FFN shape | **1.73× lower p50 latency** | M=128, N=11008, K=4096 |
| Triton GEMM peak throughput | **215 TFLOP/s (~69% of A100 fp16 tensor-core peak)** | rectangular FFN shapes |
| Shapes benchmarked | **76** LLaMA/Mistral-style projection + FFN GEMMs | `src/shapes.py` |
| Metrics per shape | p50/p90 latency, TFLOP/s, GB/s, p90/p50 jitter | `data/*.csv` |

**The honest story (this is the interesting part):** cuBLAS is extremely well-tuned for large dense GEMM, so a *bare* Triton GEMM is generally **not** faster (median ≈ 0.78× vs PyTorch/cuBLAS across the 76 shapes). The win comes from **fusion**: folding `linear + bias + GeLU/SiLU` into one Triton kernel removes an intermediate HBM round-trip and a kernel launch, which pays off at **small batch / memory-bound FFN projections** — up to **1.73×** there, with the fused kernels at ~parity (median ≈ 0.96×) elsewhere. The suite is built to show exactly *where* the crossover is, not to cherry-pick one number.

## Plots

![Fused linear+bias speedup vs cuBLAS (p50)](plots/llm_rect_linear_bias_speedup_p50.png)

![Bare GEMM speedup vs cuBLAS (p50)](plots/llm_rect_gemm_speedup_p50.png)

## Methodology

- **Shapes** (`src/shapes.py`): 76 rectangular GEMMs matching LLaMA-7B/13B/70B and Mistral-7B projection/FFN dimensions, swept over batch/sequence sizes.
- **Kernels** (`src/kernels/`): a tiled Triton `matmul`, and a fused `linear_bias` kernel with optional GeLU/SiLU epilogue.
- **Timing** (`src/utils.py`): CUDA-event timing with warmup, multiple trials, and p50/p90/std + a p90/p50 jitter ratio to capture tail stability.
- **Throughput / bandwidth** (`src/bytes_model.py`): GEMM-equivalent TFLOP/s and a fused-vs-unfused HBM-traffic model to attribute speedups to eliminated global-memory passes. (Bandwidth is a comparative IO model, not hardware counters — intended for relative fused-vs-baseline comparison.)
- **Correctness** (`src/correctness.py`): every kernel is checked against the reference with explicit tolerances before timing.
- **GeLU**: fast GeLU, `gelu(x) = x · σ(1.702x)`, used consistently across reference + Triton so correctness checks stay meaningful.

## Reproduce

```bash
pip install -r requirements.txt          # CUDA GPU + driver required (results here: A100-SXM4-80GB)
bash scripts/run_all_a100.sh             # runs all benchmarks -> data/*.csv
python scripts/plot_results.py           # regenerates plots/ from data/
```

Single benchmarks: `bench_gemm_square.py`, `bench_gemm_llm_rect.py`, `bench_linear_bias_epilogue.py --epilogue {gelu,silu,bias}` (under `benchmarks/`).

## Environment

NVIDIA A100-SXM4-80GB · CUDA-capable driver · Python 3.9+ · PyTorch ≥ 2.1 · Triton ≥ 2.1. Raw measurements are committed under `data/` (filenames stamp the GPU and date); scripts are portable to other CUDA GPUs but absolute numbers will differ by architecture/driver.

## Limitations & future work

- Single-GPU, inference-shaped GEMMs; no distributed or training-loop measurement.
- Absolute throughput is hardware/driver specific; the fusion *speedup* is the portable finding.
- The 1.73× is the best small-batch fused shape, not a blanket claim — see "the honest story" above.
- Next: fused QKV projection + attention epilogues, KV-cache read/write kernels, and an edge-inference comparison on Jetson Orin (see [edge-llm-gpu-profiling](https://github.com/Drogon4231/edge-llm-gpu-profiling)).

## Author

**Harshith Kantamneni** — MS ECE, UW-Madison. GPU performance engineering, Triton/CUDA kernel optimization, systems benchmarking.
