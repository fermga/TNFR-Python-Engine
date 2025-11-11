from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .types import GlyphLoadDistribution, TNFRGraph

__all__: tuple[str, ...]

DEFAULT_GLYPH_LOAD_SPAN: Final[int]
DEFAULT_WBAR_SPAN: Final[int]

def _std_log(kind: str, G: TNFRGraph, ctx: Mapping[str, object]) -> None: ...
def attach_standard_observer(G: TNFRGraph) -> TNFRGraph: ...
def _ensure_nodes(G: TNFRGraph) -> bool: ...
def kuramoto_metrics(G: TNFRGraph) -> tuple[float, float]: ...
def phase_sync(
    G: TNFRGraph,
    R: float | None = ...,
    psi: float | None = ...,
) -> float: ...
def kuramoto_order(
    G: TNFRGraph,
    R: float | None = ...,
    psi: float | None = ...,
) -> float: ...
def glyph_load(
    G: TNFRGraph,
    window: int | None = ...,
) -> GlyphLoadDistribution: ...
def wbar(G: TNFRGraph, window: int | None = ...) -> float: ...
