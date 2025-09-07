import networkx as nx
from tnfr.constants import ALIAS_THETA, ALIAS_VF, ALIAS_DNFR
from tnfr.alias import set_attr
from tnfr.helpers import neighbor_phase_mean
import tnfr.helpers as helpers


def test_import_nodonx_called_once(monkeypatch):
    calls = 0

    def mock_import_nodonx():
        nonlocal calls
        calls += 1

        class DummyNode:
            def __init__(self, G, n):
                self.G = G
                self.n = n

            def neighbors(self):
                return self.G.neighbors(self.n)

        return DummyNode

    monkeypatch.setattr(helpers, "import_nodonx", mock_import_nodonx)
    helpers._get_nodonx.cache_clear()

    G = nx.Graph()
    G.add_edge(1, 2)
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, 0.0)
    set_attr(G.nodes[1], ALIAS_VF, 0.0)
    set_attr(G.nodes[2], ALIAS_VF, 0.0)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[2], ALIAS_DNFR, 0.0)

    neighbor_phase_mean(G, 1)
    neighbor_phase_mean(G, 1)

    assert calls == 1
    helpers._get_nodonx.cache_clear()
