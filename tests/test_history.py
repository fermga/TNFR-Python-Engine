import networkx as nx
import pytest

from tnfr.dynamics import _update_history


def test_phase_sync_and_kuramoto_recorded():
    G = nx.Graph()
    G.add_node(1, theta=0.0)
    G.add_node(2, theta=0.0)
    _update_history(G)
    hist = G.graph.get("history", {})
    assert hist["phase_sync"][-1] == pytest.approx(1.0)
    assert "kuramoto_R" in hist
    assert hist["kuramoto_R"][-1] == pytest.approx(1.0)
