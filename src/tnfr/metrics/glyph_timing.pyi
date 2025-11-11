from __future__ import annotations

from ..types import (
    GlyphCounts,
    GlyphMetricsHistory,
    GlyphTimingByNode as GlyphTimingByNode,
    GlyphTimingTotals as GlyphTimingTotals,
    GlyphogramRow as GlyphogramRow,
    GraphLike,
    SigmaTrace as SigmaTrace,
)
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

__all__ = [
    "LATENT_GLYPH",
    "GlyphTiming",
    "SigmaTrace",
    "GlyphogramRow",
    "GlyphTimingTotals",
    "GlyphTimingByNode",
    "_tg_state",
    "for_each_glyph",
    "_update_tg_node",
    "_update_tg",
    "_update_glyphogram",
    "_update_latency_index",
    "_update_epi_support",
    "_update_morph_metrics",
    "_compute_advanced_metrics",
]

LATENT_GLYPH: str

@dataclass
class GlyphTiming:
    curr: str | None = ...
    run: float = ...

def _tg_state(nd: MutableMapping[str, Any]) -> GlyphTiming: ...
def for_each_glyph(fn: Callable[[str], None]) -> None: ...
def _update_tg_node(
    n: Any,
    nd: MutableMapping[str, Any],
    dt: float,
    tg_total: GlyphTimingTotals,
    tg_by_node: GlyphTimingByNode | None,
) -> tuple[str | None, bool]: ...
def _update_tg(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    dt: float,
    save_by_node: bool,
    n_jobs: int | None = None,
) -> tuple[Counter[str], int, int]: ...
def _update_glyphogram(
    G: GraphLike, hist: GlyphMetricsHistory, counts: GlyphCounts, t: float, n_total: int
) -> None: ...
def _update_latency_index(
    G: GraphLike, hist: GlyphMetricsHistory, n_total: int, n_latent: int, t: float
) -> None: ...
def _update_epi_support(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    t: float,
    threshold: float = ...,
    n_jobs: int | None = None,
) -> None: ...
def _update_morph_metrics(
    G: GraphLike, hist: GlyphMetricsHistory, counts: GlyphCounts, t: float
) -> None: ...
def _compute_advanced_metrics(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    t: float,
    dt: float,
    cfg: Mapping[str, Any],
    threshold: float = ...,
    n_jobs: int | None = None,
) -> None: ...
