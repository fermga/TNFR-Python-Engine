import networkx as nx

from tnfr.helpers.cache import edge_version_cache


def test_edge_version_cache_limit():
    G = nx.Graph()
    edge_version_cache(G, "a", lambda: 1, max_entries=2)
    edge_version_cache(G, "b", lambda: 2, max_entries=2)
    edge_version_cache(G, "c", lambda: 3, max_entries=2)
    cache = G.graph["_edge_version_cache"]
    assert "a" not in cache
    assert "b" in cache and "c" in cache


def test_edge_version_cache_lock_cleanup():
    G = nx.Graph()
    for i in range(10):
        edge_version_cache(G, str(i), lambda i=i: i, max_entries=2)
    cache = G.graph["_edge_version_cache"]
    locks = G.graph["_edge_version_cache_locks"]
    assert len(cache) <= 2
    assert set(locks) == set(cache)
