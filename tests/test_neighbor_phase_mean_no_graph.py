import pytest

from tnfr.metrics.trig import neighbor_phase_mean
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


def test_neighbor_phase_mean_uses_generic(monkeypatch, graph_canon):
    ALIAS_THETA = get_aliases("THETA")
    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)

    calls = []

    def fake_generic(obj, cos_map=None, sin_map=None, np=None, fallback=0.0):
        calls.append((obj, cos_map, sin_map, np, fallback))
        return 0.0

    monkeypatch.setattr("tnfr.metrics.trig._neighbor_phase_mean_generic", fake_generic)
    neighbor_phase_mean(G, 1)
    assert len(calls) == 1
    node_arg, cos_map, sin_map, np_arg, fallback = calls[0]
    assert node_arg.n == 1 and node_arg.G is G
    assert cos_map is None and sin_map is None and np_arg is None and fallback == 0.0
