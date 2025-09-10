from tnfr.helpers.cache import EdgeCacheManager, edge_version_cache


def test_edge_version_cache_prunes_locks(graph_canon):
    G = graph_canon()
    for i in range(5):
        edge_version_cache(G, str(i), lambda i=i: i, max_entries=2)
    _, locks = EdgeCacheManager(G.graph).get_cache(2)
    assert len(locks) <= 2


def test_edge_version_cache_lock_cleanup_unbounded(graph_canon):
    G = graph_canon()
    edge_version_cache(G, "a", lambda: 1, max_entries=None)
    edge_version_cache(G, "b", lambda: 2, max_entries=None)
    cache, locks = EdgeCacheManager(G.graph).get_cache(None)
    cache.pop("a")
    assert "a" in locks
    edge_version_cache(G, "c", lambda: 3, max_entries=None)
    assert "a" not in locks
    assert set(locks) == set(cache)
