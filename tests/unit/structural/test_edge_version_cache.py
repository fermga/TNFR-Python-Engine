import pytest
from concurrent.futures import ThreadPoolExecutor

from tnfr.utils import (
    EdgeCacheManager,
    edge_version_cache,
    increment_edge_version,
)


@pytest.fixture
def graph_and_manager(graph_canon):
    """Return a factory that builds a graph and its cache manager."""

    def _factory():
        G = graph_canon()
        manager = EdgeCacheManager(G.graph)
        return G, manager

    return _factory


def test_edge_version_cache_disable(graph_and_manager):
    G, _ = graph_and_manager()
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


def test_edge_version_cache_limit(graph_and_manager):
    G, _ = graph_and_manager()
    edge_version_cache(G, "a", lambda: 1, max_entries=2)
    edge_version_cache(G, "b", lambda: 2, max_entries=2)
    edge_version_cache(G, "c", lambda: 3, max_entries=2)
    cache, _ = EdgeCacheManager(G.graph).get_cache(2)
    assert "a" not in cache
    assert "b" in cache and "c" in cache


@pytest.mark.parametrize("max_entries", [2, None])
def test_edge_version_cache_lock_cleanup(graph_and_manager, max_entries):
    G, _ = graph_and_manager()
    if max_entries is None:
        edge_version_cache(G, "a", lambda: 1, max_entries=None)
        edge_version_cache(G, "b", lambda: 2, max_entries=None)
        cache, locks = EdgeCacheManager(G.graph).get_cache(None)
        cache.pop("a")
        assert "a" in locks
        edge_version_cache(G, "c", lambda: 3, max_entries=None)
        cache, locks = EdgeCacheManager(G.graph).get_cache(None)
        assert "a" not in locks
        assert set(locks) == set(cache)
    else:
        for i in range(5):
            edge_version_cache(G, str(i), lambda i=i: i, max_entries=max_entries)
        cache, locks = EdgeCacheManager(G.graph).get_cache(max_entries)
        assert len(cache) <= max_entries
        assert set(locks) == set(cache)


def test_edge_version_cache_manager(graph_and_manager):
    G, _ = graph_and_manager()
    assert "_edge_cache_manager" not in G.graph

    edge_version_cache(G, "a", lambda: 1)
    manager = G.graph.get("_edge_cache_manager")
    assert isinstance(manager, EdgeCacheManager)
    assert manager.graph is G.graph

    edge_version_cache(G, "b", lambda: 2)
    assert G.graph.get("_edge_cache_manager") is manager


def test_edge_version_cache_reentrant(graph_and_manager):
    G, _ = graph_and_manager()
    calls = []

    def builder():
        if not calls:
            calls.append("outer")
            return edge_version_cache(G, "k", builder)
        calls.append("inner")
        return "ok"

    assert edge_version_cache(G, "k", builder) == "ok"
    assert calls == ["outer", "inner"]


def test_edge_version_cache_thread_safety(graph_and_manager):
    G, _ = graph_and_manager()
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return object()

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: edge_version_cache(G, "k", builder), range(32)))
    first = results[0]
    assert all(r is first for r in results)
    assert calls >= 1

    calls_after_first = calls

    increment_edge_version(G)
    with ThreadPoolExecutor(max_workers=16) as ex:
        results2 = list(ex.map(lambda _: edge_version_cache(G, "k", builder), range(32)))
    second = results2[0]
    assert all(r is second for r in results2)
    assert second is not first
    assert calls > calls_after_first


def test_edge_version_cache_metrics(graph_and_manager):
    G, manager = graph_and_manager()
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return calls

    assert edge_version_cache(G, "k", builder) == 1
    assert edge_version_cache(G, "k", builder) == 1

    stats = manager._manager.get_metrics(manager._STATE_KEY)
    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.evictions == 0
