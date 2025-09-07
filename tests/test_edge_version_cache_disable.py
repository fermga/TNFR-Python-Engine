import networkx as nx

from tnfr.helpers import edge_version_cache


def test_edge_version_cache_disable():
    G = nx.Graph()
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return object()

    first = edge_version_cache(G, "k", builder, max_entries=0)
    second = edge_version_cache(G, "k", builder, max_entries=0)

    assert calls == 2
    assert first is not second
    assert "_edge_version_cache" not in G.graph
