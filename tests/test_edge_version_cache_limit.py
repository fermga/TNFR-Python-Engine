import networkx as nx

from tnfr.helpers import edge_version_cache


def test_edge_version_cache_limit():
    G = nx.Graph()
    edge_version_cache(G, "a", lambda: 1, max_entries=2)
    edge_version_cache(G, "b", lambda: 2, max_entries=2)
    edge_version_cache(G, "c", lambda: 3, max_entries=2)
    assert "a_cache" not in G.graph
    assert "b_cache" in G.graph and "c_cache" in G.graph
