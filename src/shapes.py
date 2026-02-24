# src/shapes.py
from __future__ import annotations

from .utils import gemm_flops


def make_llm_rect_shapes(
    tokens_list: list[int] | None = None,
    hidden_list: list[int] | None = None,
    ffn_map: dict[int, int] | None = None,
    use_ffn_4x: bool = True,
    include_stress: bool = True,
) -> list[tuple[int, int, int]]:
    """
    Returns a sorted list of (M, N, K) GEMM shapes that resemble LLM inference projections:
      - attention projections: (M, H) x (H, H)
      - gated FFN up/down projections
      - optional 4x FFN
      - optional long-K stress shapes

    M represents tokens (batch * seq) for a given micro-batch.
    """
    if tokens_list is None:
        tokens_list = [128, 256, 512, 1024, 2048]
    if hidden_list is None:
        hidden_list = [4096, 5120, 8192]
    if ffn_map is None:
        ffn_map = {4096: 11008, 5120: 13824, 8192: 28672}

    shapes: list[tuple[int, int, int]] = []

    # Attention projections (Q,K,V,O): (M,H) @ (H,H) => (M,H)
    for M in tokens_list:
        for H in hidden_list:
            shapes.append((M, H, H))  # (M x K=H) @ (K=H x N=H) => (M,N=H)

    # Gated FFN up/down: (M,H)@(H,F) and (M,F)@(F,H)
    for M in tokens_list:
        for H in hidden_list:
            F = ffn_map[H]
            shapes.append((M, F, H))  # up-proj
            shapes.append((M, H, F))  # down-proj

    # Classic 4x FFN
    if use_ffn_4x:
        for M in tokens_list:
            for H in hidden_list:
                F4 = 4 * H
                shapes.append((M, F4, H))
                shapes.append((M, H, F4))

    if include_stress:
        shapes += [
            (512, 4096, 16384),
            (1024, 4096, 16384),
            (2048, 4096, 16384),
            (1024, 8192, 16384),
        ]

    shapes = sorted(set(shapes), key=lambda s: gemm_flops(*s))
    return shapes