from tnfr.helpers.cache import edge_version_cache


def test_edge_version_cache_reentrant(graph_canon):
    G = graph_canon()
    calls = []

    def builder():
        if not calls:
            calls.append("outer")
            return edge_version_cache(G, "k", builder)
        calls.append("inner")
        return "ok"

    assert edge_version_cache(G, "k", builder) == "ok"
    assert calls == ["outer", "inner"]
