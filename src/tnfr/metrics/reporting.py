"""Reporting helpers for collected metrics."""

from __future__ import annotations

from heapq import nlargest
from statistics import mean, median

from ..glyph_history import ensure_history
from .glyph_timing import for_each_glyph

__all__ = [
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
    "glyph_dwell_stats",
]


# ---------------------------------------------------------------------------
# Reporting functions
# ---------------------------------------------------------------------------


def Tg_global(G, normalize: bool = True) -> dict[str, float]:
    """Total glyph dwell time per class."""

    hist = ensure_history(G)
    tg_total: dict[str, float] = hist.get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    out: dict[str, float] = {}

    def add(g):
        val = float(tg_total.get(g, 0.0))
        out[g] = val / total if normalize else val

    for_each_glyph(add)
    return out


def Tg_by_node(G, n, normalize: bool = False) -> dict[str, float | list[float]]:
    """Per-node glyph dwell summary."""

    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    if not normalize:
        out: dict[str, list[float]] = {}

        def copy_runs(g):
            out[g] = list(rec.get(g, []))

        for_each_glyph(copy_runs)
        return out
    out: dict[str, float] = {}

    def add(g):
        runs = rec.get(g, [])
        out[g] = float(mean(runs)) if runs else 0.0

    for_each_glyph(add)
    return out


def latency_series(G) -> dict[str, list[float]]:
    hist = ensure_history(G)
    xs = hist.get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glyphogram_series(G) -> dict[str, list[float]]:
    hist = ensure_history(G)
    xs = hist.get("glyphogram", [])
    if not xs:
        return {"t": []}
    out: dict[str, list[float]] = {"t": [float(x.get("t", i)) for i, x in enumerate(xs)]}

    def add(g):
        out[g] = [float(x.get(g, 0.0)) for x in xs]

    for_each_glyph(add)
    return out


def glyph_top(G, k: int = 3) -> list[tuple[str, float]]:
    """Top-k structural operators by ``Tg_global`` fraction."""

    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    tg = Tg_global(G, normalize=True)
    return nlargest(k, tg.items(), key=lambda kv: kv[1])


def glyph_dwell_stats(G, n) -> dict[str, dict[str, float]]:
    """Per-node dwell time statistics for each glyph."""

    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    out: dict[str, dict[str, float]] = {}

    def add(g):
        runs = list(rec.get(g, []))
        if not runs:
            out[g] = {"mean": 0.0, "median": 0.0, "max": 0.0, "count": 0}
        else:
            out[g] = {
                "mean": float(mean(runs)),
                "median": float(median(runs)),
                "max": float(max(runs)),
                "count": int(len(runs)),
            }

    for_each_glyph(add)
    return out
