import pytest

from tnfr.helpers.numeric import neighbor_phase_mean
from tnfr.constants import get_aliases
from tnfr.alias import set_attr


class DummyNeighbor:
    pass


class DummyNode:
    def __init__(self):
        self.theta = 0.5
        self._neigh = [DummyNeighbor()]

    def neighbors(self):
        return self._neigh


def test_neighbor_phase_mean_requires_graph():
    node = DummyNode()
    with pytest.raises(TypeError):
        neighbor_phase_mean(node)


def test_neighbor_phase_mean_uses_core(monkeypatch, graph_canon):
    ALIAS_THETA = get_aliases("THETA")
    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)

    calls = []

    def fake_core(neigh, cos_map, sin_map, np, fallback):
        calls.append((neigh, cos_map, sin_map, np, fallback))
        return 0.0

    monkeypatch.setattr("tnfr.helpers.numeric._neighbor_phase_mean_core", fake_core)
    neighbor_phase_mean(G, 1)
    assert len(calls) == 1
    assert list(calls[0][0]) == list(G[1])
