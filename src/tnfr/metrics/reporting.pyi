from __future__ import annotations

from ..types import NodeId, TNFRGraph

__all__ = [
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
    "build_metrics_summary",
]

def Tg_global(G: TNFRGraph, normalize: bool = True) -> dict[str, float]: ...
def Tg_by_node(
    G: TNFRGraph, n: NodeId, normalize: bool = False
) -> dict[str, float] | dict[str, list[float]]: ...
def latency_series(G: TNFRGraph) -> dict[str, list[float]]: ...
def glyphogram_series(G: TNFRGraph) -> dict[str, list[float]]: ...
def glyph_top(G: TNFRGraph, k: int = 3) -> list[tuple[str, float]]: ...
def build_metrics_summary(
    G: TNFRGraph, *, series_limit: int | None = None
) -> tuple[dict[str, float | dict[str, float] | dict[str, list[float]] | dict[str, int]], bool]: ...
