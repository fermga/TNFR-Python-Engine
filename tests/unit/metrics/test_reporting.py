"""Tests for reporting metrics helpers."""

from __future__ import annotations

import pytest

from tnfr.glyph_history import ensure_history
from tnfr.metrics.reporting import (
    Tg_by_node,
    Tg_global,
    glyph_top,
    glyphogram_series,
    latency_series,
)


def _graph_with_history(graph_canon, history):
    G = graph_canon()
    hist = ensure_history(G)
    hist.update(history)
    return G


def test_tg_global_normalization_and_raw(graph_canon):
    G = _graph_with_history(
        graph_canon,
        {"Tg_total": {"AL": 2.0, "EN": 1.0}},
    )

    normalized = Tg_global(G, normalize=True)
    raw = Tg_global(G, normalize=False)

    assert pytest.approx(2.0 / 3.0) == normalized["AL"]
    assert pytest.approx(1.0 / 3.0) == normalized["EN"]
    assert normalized["IL"] == 0.0
    assert raw["AL"] == 2.0
    assert raw["EN"] == 1.0
    assert raw["IL"] == 0.0


def test_tg_by_node_copies_runs_and_computes_means(graph_canon):
    runs = {"AL": [1.0, 3.0], "EN": [2.0]}
    G = _graph_with_history(graph_canon, {"Tg_by_node": {0: runs}})

    raw_runs = Tg_by_node(G, 0, normalize=False)
    assert raw_runs["AL"] == runs["AL"]
    assert raw_runs["AL"] is not runs["AL"]

    means = Tg_by_node(G, 0, normalize=True)
    assert means["AL"] == pytest.approx(2.0)
    assert means["EN"] == pytest.approx(2.0)
    assert means["IL"] == 0.0

    runs["AL"].append(9.0)
    assert raw_runs["AL"] == [1.0, 3.0]
    updated_means = Tg_by_node(G, 0, normalize=True)
    assert updated_means["AL"] == pytest.approx((1.0 + 3.0 + 9.0) / 3.0)


def test_latency_and_glyphogram_series_index_fallback(graph_canon):
    G = _graph_with_history(
        graph_canon,
        {
            "latency_index": [
                {"value": 2.5},
                {"t": 5, "value": 3.0},
            ],
            "glyphogram": [
                {"AL": 0.1},
                {"t": 2, "AL": 0.2},
            ],
        },
    )

    latency = latency_series(G)
    assert latency == {"t": [0.0, 5.0], "value": [2.5, 3.0]}

    glyph = glyphogram_series(G)
    assert glyph["t"] == [0.0, 2.0]
    assert glyph["AL"] == [0.1, 0.2]
    assert glyph["EN"] == [0.0, 0.0]

    G_empty = graph_canon()
    glyph_empty = glyphogram_series(G_empty)
    assert glyph_empty == {"t": []}


def test_glyph_top_validates_k_and_returns_largest(graph_canon):
    G = _graph_with_history(
        graph_canon,
        {"Tg_total": {"AL": 3.0, "EN": 1.0, "IL": 4.0}},
    )

    top_two = glyph_top(G, k=2)
    assert top_two[0][0] == "IL"
    assert top_two[1][0] == "AL"
    assert top_two[0][1] == pytest.approx(4.0 / 8.0)
    assert top_two[1][1] == pytest.approx(3.0 / 8.0)

    with pytest.raises(ValueError):
        glyph_top(G, k=0)
