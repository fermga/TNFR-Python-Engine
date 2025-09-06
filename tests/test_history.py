"""Pruebas de history."""

import pytest

from tnfr.metrics import _metrics_step


def test_phase_sync_and_kuramoto_recorded(graph_canon):
    G = graph_canon()
    G.add_node(1, theta=0.0)
    G.add_node(2, theta=0.0)
    _metrics_step(G)
    hist = G.graph.get("history", {})
    assert hist["phase_sync"][-1] == pytest.approx(1.0)
    assert "kuramoto_R" in hist
    assert hist["kuramoto_R"][-1] == pytest.approx(1.0)
