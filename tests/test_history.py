"""Pruebas de history."""

import pytest

from tnfr.metrics.core import _metrics_step
from tnfr.glyph_history import push_glyph, ensure_history


def test_phase_sync_and_kuramoto_recorded(graph_canon):
    G = graph_canon()
    G.add_node(1, theta=0.0)
    G.add_node(2, theta=0.0)
    _metrics_step(G)
    hist = ensure_history(G)
    assert hist["phase_sync"][-1] == pytest.approx(1.0)
    assert "kuramoto_R" in hist
    assert hist["kuramoto_R"][-1] == pytest.approx(1.0)


def test_string_history_is_discarded():
    nd = {"glyph_history": "ABC"}
    push_glyph(nd, "Z", 5)
    assert list(nd["glyph_history"]) == ["Z"]
