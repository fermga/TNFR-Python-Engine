from tnfr.helpers.edge_cache import EdgeCacheManager, edge_version_cache


def test_edge_version_cache_stores_manager(graph_canon):
    G = graph_canon()
    edge_version_cache(G, "a", lambda: 1)
    manager = G.graph.get("_edge_cache_manager")
    assert isinstance(manager, EdgeCacheManager)
    edge_version_cache(G, "b", lambda: 2)
    assert G.graph.get("_edge_cache_manager") is manager


def test_edge_version_cache_accepts_manager(graph_canon):
    G = graph_canon()
    manager = EdgeCacheManager(G.graph)
    edge_version_cache(G, "a", lambda: 1, manager=manager)
    assert G.graph.get("_edge_cache_manager") is manager
