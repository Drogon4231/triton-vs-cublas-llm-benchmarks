"""
Triton vs cuBLAS LLM kernel benchmarks.

This package contains:
- Triton kernels (GEMM, fused linear+bias(+GELU/SiLU))
- Benchmark drivers and utilities
- Shape generators and simple bandwidth/AI models
"""

__all__ = [
    "utils",
    "bytes_model",
    "shapes",
    "correctness",
    "kernels",
]