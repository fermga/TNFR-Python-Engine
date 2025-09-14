from tnfr.helpers import EdgeCacheManager, edge_version_cache


def test_edge_version_cache_limit(graph_canon):
    G = graph_canon()
    edge_version_cache(G, "a", lambda: 1, max_entries=2)
    edge_version_cache(G, "b", lambda: 2, max_entries=2)
    edge_version_cache(G, "c", lambda: 3, max_entries=2)
    cache, _ = EdgeCacheManager(G.graph).get_cache(2)
    assert "a" not in cache
    assert "b" in cache and "c" in cache


def test_edge_version_cache_lock_cleanup(graph_canon):
    G = graph_canon()
    for i in range(10):
        edge_version_cache(G, str(i), lambda i=i: i, max_entries=2)
    cache, locks = EdgeCacheManager(G.graph).get_cache(2)
    assert len(cache) <= 2
    assert set(locks) == set(cache)
