"""Reporting helpers for collected metrics."""

from __future__ import annotations

from collections.abc import Sequence
from heapq import nlargest
from statistics import StatisticsError, fmean, mean
from typing import Any

from ..glyph_history import ensure_history
from ..sense import sigma_rose
from ..types import NodeId, TNFRGraph
from .glyph_timing import for_each_glyph

__all__ = [
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
    "build_metrics_summary",
]

# ---------------------------------------------------------------------------
# Reporting functions
# ---------------------------------------------------------------------------


def Tg_global(G: TNFRGraph, normalize: bool = True) -> dict[str, float]:
    """Total glyph dwell time per class."""

    hist = ensure_history(G)
    tg_total: dict[str, float] = hist.get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    out: dict[str, float] = {}

    def add(g: str) -> None:
        val = float(tg_total.get(g, 0.0))
        out[g] = val / total if normalize else val

    for_each_glyph(add)
    return out


def Tg_by_node(
    G: TNFRGraph, n: NodeId, normalize: bool = False
) -> dict[str, float] | dict[str, list[float]]:
    """Per-node glyph dwell summary."""

    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    if not normalize:
        runs_out: dict[str, list[float]] = {}

        def copy_runs(g: str) -> None:
            runs_out[g] = list(rec.get(g, []))

        for_each_glyph(copy_runs)
        return runs_out
    mean_out: dict[str, float] = {}

    def add(g: str) -> None:
        runs = rec.get(g, [])
        mean_out[g] = float(mean(runs)) if runs else 0.0

    for_each_glyph(add)
    return mean_out


def latency_series(G: TNFRGraph) -> dict[str, list[float]]:
    """Return latency samples as ``{"t": [...], "value": [...]}``."""

    hist = ensure_history(G)
    xs = hist.get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glyphogram_series(G: TNFRGraph) -> dict[str, list[float]]:
    """Return glyphogram time series keyed by glyph label."""

    hist = ensure_history(G)
    xs = hist.get("glyphogram", [])
    if not xs:
        return {"t": []}
    out: dict[str, list[float]] = {"t": [float(x.get("t", i)) for i, x in enumerate(xs)]}

    def add(g: str) -> None:
        out[g] = [float(x.get(g, 0.0)) for x in xs]

    for_each_glyph(add)
    return out


def glyph_top(G: TNFRGraph, k: int = 3) -> list[tuple[str, float]]:
    """Top-k structural operators by ``Tg_global`` fraction."""

    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    tg = Tg_global(G, normalize=True)
    return nlargest(k, tg.items(), key=lambda kv: kv[1])


def build_metrics_summary(
    G: TNFRGraph, *, series_limit: int | None = None
) -> tuple[dict[str, float | dict[str, float] | dict[str, list[float]] | dict[str, int]], bool]:
    """Collect a compact metrics summary for CLI reporting.

    This factory aggregates various TNFR metrics into a unified summary
    structure suitable for command-line display and analysis. It combines
    glyph timing statistics, latency measurements, and coherence indicators.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the recorded metrics history.
    series_limit : int | None, optional
        Maximum number of samples to keep for each glyphogram series.
        When ``None`` or non-positive, returns the full history without
        trimming. Default is None.

    Returns
    -------
    tuple[dict, bool]
        A two-element tuple containing:

        - **summary** (dict): Metrics dictionary with the following keys:

          - ``Tg_global``: Normalized glyph dwell time per class
          - ``latency_mean``: Mean latency across all samples
          - ``rose``: Sigma rose coherence indicator
          - ``glyphogram``: Time series of glyph activity (trimmed if limit set)

        - **has_data** (bool): True if latency data is available, False otherwise

    Notes
    -----
    The series trimming feature is useful for limiting memory usage when
    tracking long-running simulations. Trimming only affects the glyphogram
    time series; aggregate statistics remain computed from the full history.
    """

    tg = Tg_global(G, normalize=True)
    latency = latency_series(G)
    glyph = glyphogram_series(G)
    rose = sigma_rose(G)

    latency_values = latency.get("value", [])
    try:
        latency_mean = fmean(latency_values)
    except StatisticsError:
        latency_mean = 0.0

    limit: int | None
    if series_limit is None:
        limit = None
    else:
        limit = int(series_limit)
        if limit <= 0:
            limit = None

    def _trim(values: Sequence[Any]) -> list[Any]:
        seq = list(values)
        if limit is None:
            return seq
        return seq[:limit]

    glyph_summary = {k: _trim(v) for k, v in glyph.items()}

    summary = {
        "Tg_global": tg,
        "latency_mean": latency_mean,
        "rose": rose,
        "glyphogram": glyph_summary,
    }
    return summary, bool(latency_values)
