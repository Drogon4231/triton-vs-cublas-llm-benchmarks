#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_all_a100.sh
#
# Notes:
# - Assumes you've installed dependencies:
#     pip install -r requirements.txt
# - Outputs:
#     data/*.csv
#     plots/*.png

mkdir -p data plots

echo "[INFO] GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo "[INFO] Running square GEMM benchmarks..."
python benchmarks/bench_gemm_square.py \
  --sizes 256 512 1024 2048 4096 8192 \
  --trials 10 --warmup 30 --iters 200 \
  --dtypes fp16 bf16 \
  --out_dir data --plot_dir plots

echo "[INFO] Running LLM-rect GEMM benchmarks..."
python benchmarks/bench_gemm_llm_rect.py \
  --trials 10 --warmup 30 --iters 200 \
  --dtypes fp16 bf16 \
  --out_dir data --plot_dir plots

echo "[INFO] Running fused linear+bias (bias only)..."
python benchmarks/bench_linear_bias_epilogue.py \
  --epilogue bias \
  --trials 10 --warmup 30 \
  --dtypes fp16 bf16 \
  --out_dir data --plot_dir plots

echo "[INFO] Running fused linear+bias+GELU (fast GELU)..."
python benchmarks/bench_linear_bias_epilogue.py \
  --epilogue gelu \
  --trials 10 --warmup 30 \
  --dtypes fp16 bf16 \
  --out_dir data --plot_dir plots

echo "[INFO] Running fused linear+bias+SiLU..."
python benchmarks/bench_linear_bias_epilogue.py \
  --epilogue silu \
  --trials 10 --warmup 30 \
  --dtypes fp16 bf16 \
  --out_dir data --plot_dir plots

echo "[DONE] All benchmarks finished."
echo "       CSVs  -> data/"
echo "       Plots -> plots/"