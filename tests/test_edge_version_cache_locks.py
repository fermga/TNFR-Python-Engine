import networkx as nx

from tnfr.helpers.cache import edge_version_cache


def test_edge_version_cache_prunes_locks(graph_canon):
    G = graph_canon()
    for i in range(5):
        edge_version_cache(G, str(i), lambda i=i: i, max_entries=2)
    locks = G.graph["_edge_version_cache_locks"]
    assert len(locks) <= 2
