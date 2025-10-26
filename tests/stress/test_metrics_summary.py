"""Stress tests for the metrics summary reporting pipeline."""

from __future__ import annotations

import time
from statistics import fmean
from typing import NamedTuple

import networkx as nx
import pytest

try:  # pragma: no cover - optional plugin detection
    import pytest_timeout  # noqa: F401
except ImportError:  # pragma: no cover - fallback when plugin missing
    def timeout_mark(_: float):
        def decorator(func):
            return func

        return decorator
else:  # pragma: no cover - executed when plugin available
    timeout_mark = pytest.mark.timeout

from tnfr.config.constants import GLYPHS_CANONICAL
from tnfr.constants import inject_defaults
from tnfr.glyph_history import ensure_history
from tnfr.metrics.reporting import build_metrics_summary

pytestmark = [pytest.mark.slow, pytest.mark.stress]


class SeededMetrics(NamedTuple):
    """Container storing seeded graph data and expected aggregates."""

    graph: nx.Graph
    tg_normalized: dict[str, float]
    latency_mean: float
    glyphogram_series: dict[str, list[float]]
    rose_totals: dict[str, int]


def _seed_metrics_graph(
    *,
    nodes: int,
    probability: float,
    seed: int,
    glyph_samples: int,
    latency_samples: int,
    rose_samples: int,
) -> SeededMetrics:
    """Build a large graph with extensive metrics history for stress testing."""

    graph = nx.gnp_random_graph(nodes, probability, seed=seed)
    inject_defaults(graph)
    graph.graph["HISTORY_MAXLEN"] = 0

    history = ensure_history(graph)

    totals: dict[str, float] = {}
    totals_sum = 0.0
    for idx, glyph in enumerate(GLYPHS_CANONICAL):
        value = float((idx + 1) * 250.0 + (seed % 19))
        totals[glyph] = value
        totals_sum += value
    history["Tg_total"] = totals
    tg_normalized = {
        glyph: value / totals_sum if totals_sum else 0.0
        for glyph, value in totals.items()
    }

    glyphogram_rows: list[dict[str, float]] = []
    glyphogram_series: dict[str, list[float]] = {"t": []}
    phase_offset = (seed % 11) * 0.05
    for step in range(glyph_samples):
        t_value = float(step * 0.5)
        glyphogram_series.setdefault("t", []).append(t_value)
        row: dict[str, float] = {"t": t_value}
        modulation = (step % 23) * 0.02
        for idx, glyph in enumerate(GLYPHS_CANONICAL):
            value = (idx + 1) * 0.25 + phase_offset + modulation
            row[glyph] = value
            glyphogram_series.setdefault(glyph, []).append(value)
        glyphogram_rows.append(row)
    history["glyphogram"] = glyphogram_rows

    latency_rows: list[dict[str, float]] = []
    latency_values: list[float] = []
    for i in range(latency_samples):
        t_value = float(i * 0.2)
        latency_value = 0.75 + ((i * 7 + seed) % 41) * 0.01
        latency_rows.append({"t": t_value, "value": latency_value})
        latency_values.append(latency_value)
    history["latency_index"] = latency_rows
    latency_mean = fmean(latency_values) if latency_values else 0.0

    sigma_rows: list[dict[str, int]] = []
    rose_totals = {glyph: 0 for glyph in GLYPHS_CANONICAL}
    for step in range(rose_samples):
        row = {"t": step}
        for idx, glyph in enumerate(GLYPHS_CANONICAL):
            count = int(((step + idx) * (idx + 2)) % 13 + idx)
            row[glyph] = count
            rose_totals[glyph] += count
        sigma_rows.append(row)
    history["sigma_counts"] = sigma_rows

    return SeededMetrics(
        graph=graph,
        tg_normalized=tg_normalized,
        latency_mean=latency_mean,
        glyphogram_series=glyphogram_series,
        rose_totals=rose_totals,
    )


@timeout_mark(30)
def test_build_metrics_summary_handles_large_histories() -> None:
    """``build_metrics_summary`` must stay fast and accurate on large histories."""

    seeded = _seed_metrics_graph(
        nodes=420,
        probability=0.14,
        seed=5179,
        glyph_samples=720,
        latency_samples=600,
        rose_samples=512,
    )

    start = time.perf_counter()
    summary, has_latency = build_metrics_summary(seeded.graph)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0
    assert has_latency is True

    assert summary["Tg_global"].keys() == seeded.tg_normalized.keys()
    for glyph, expected_value in seeded.tg_normalized.items():
        assert summary["Tg_global"][glyph] == pytest.approx(expected_value)

    assert summary["latency_mean"] == pytest.approx(seeded.latency_mean)
    assert summary["rose"] == seeded.rose_totals

    glyphogram = summary["glyphogram"]
    assert glyphogram["t"] == seeded.glyphogram_series["t"]
    for glyph in GLYPHS_CANONICAL:
        assert glyphogram[glyph] == seeded.glyphogram_series[glyph]

    limit = 37
    trimmed_summary, trimmed_has_latency = build_metrics_summary(
        seeded.graph, series_limit=limit
    )
    assert trimmed_has_latency is True
    trimmed_glyphs = trimmed_summary["glyphogram"]
    assert len(trimmed_glyphs["t"]) == limit
    assert trimmed_glyphs["t"] == seeded.glyphogram_series["t"][:limit]
    for glyph in GLYPHS_CANONICAL:
        assert trimmed_glyphs[glyph] == seeded.glyphogram_series[glyph][:limit]

    trimmed_summary_repeat, trimmed_has_latency_repeat = build_metrics_summary(
        seeded.graph, series_limit=limit
    )
    assert trimmed_has_latency_repeat is True
    assert trimmed_summary_repeat == trimmed_summary
    assert trimmed_summary_repeat["latency_mean"] == pytest.approx(summary["latency_mean"])
    assert trimmed_summary_repeat["Tg_global"] == summary["Tg_global"]
    assert trimmed_summary_repeat["rose"] == summary["rose"]
