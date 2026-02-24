# Triton vs cuBLAS: LLM Kernel Benchmarking Suite

Benchmarking suite comparing **Triton fused GPU kernels** against **PyTorch/cuBLAS baselines** for operators commonly used in **LLM inference**, including:

- **Square GEMM** (M = N = K)
- **LLM-shaped rectangular GEMM** (projection/FFN-like shapes)
- **Fused linear + bias** (single Triton kernel)
- **Fused linear + bias + fast GeLU** (`x * sigmoid(1.702x)`)
- **Fused linear + bias + SiLU** (`x * sigmoid(x)`)

Benchmarks report **p50 latency**, **GEMM-equivalent throughput (TFLOP/s)**, a simple **effective bandwidth model (GB/s)**, and **tail-latency jitter (p90/p50)**.

> **Hardware used for results in this repo:** NVIDIA A100-SXM4-80GB  
> (Scripts are portable to other CUDA GPUs; results will vary by architecture/driver.)

---

## Key Takeaways

- **cuBLAS dominates raw GEMM** throughput on many large dense shapes (as expected).
- For **LLM inference operators**, performance can be **memory/launch bound** rather than compute bound.
- **Triton fusion can reduce end-to-end latency** (especially for `linear+bias+activation`) by:
  - eliminating intermediate global memory reads/writes
  - reducing kernel launch overhead
- Tail latency is tracked via **p90/p50 jitter** to capture stability under repeated runs.

---

## Repository Layout

src/
kernels/          Triton kernels (GEMM, fused linear+bias(+epilogue))
utils.py          CUDA timing + FLOPs helpers
bytes_model.py    Comparative IO/bandwidth model
shapes.py         LLM-ish shape generator
correctness.py    correctness checks + tolerances

benchmarks/
bench_gemm_square.py
bench_gemm_llm_rect.py
bench_linear_bias_epilogue.py

scripts/
run_all_a100.sh
plot_results.py

plots/              curated plots (committed)
data/               CSV outputs (gitignored by default)

---

## Setup

### Requirements
- CUDA-capable GPU + compatible NVIDIA driver
- Python 3.9+
- PyTorch + Triton

Install dependencies:

```bash
pip install -r requirements.txt


⸻

Running Benchmarks

1) Run everything (recommended)

bash scripts/run_all_a100.sh

Outputs:
	•	CSVs → data/
	•	Plots → plots/

2) Run a single benchmark

Square GEMM

python benchmarks/bench_gemm_square.py

LLM-shaped GEMM

python benchmarks/bench_gemm_llm_rect.py

Fused linear+bias(+epilogue)

python benchmarks/bench_linear_bias_epilogue.py --epilogue gelu
python benchmarks/bench_linear_bias_epilogue.py --epilogue silu
python benchmarks/bench_linear_bias_epilogue.py --epilogue bias


⸻

Metrics Reported
	•	Latency (p50, p90): median and tail latency in ms
	•	Jitter: p90 / p50
	•	GEMM-equivalent TFLOP/s: computed from GEMM FLOPs divided by p50 latency
(useful for comparing shapes even when fused epilogues are present)
	•	Effective bandwidth (GB/s) [model]: comparative IO model to show memory pressure
(not hardware counters; intended for relative comparisons between fused vs baseline)

⸻

Notes on GeLU Definition

This repo uses fast GeLU consistently across reference + Triton:

[
\text{gelu_fast}(x) = x \cdot \sigma(1.702x)
]

This avoids mixing different GeLU approximations and keeps correctness checks meaningful.

⸻

Re-plotting From a CSV

python scripts/plot_results.py --csv data/<your_file>.csv --out_dir plots


⸻

Future Work
	•	Fuse QKV projection and attention epilogues
	•	KV-cache read/write optimized kernels
	•	Extend benchmark runs to Jetson Orin for edge inference comparisons

⸻

Author

Harshith Kantamneni
MS ECE, UW–Madison
Focus: GPU performance engineering, Triton/CUDA kernel optimization, systems benchmarking
