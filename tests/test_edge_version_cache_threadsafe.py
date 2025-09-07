import networkx as nx
from concurrent.futures import ThreadPoolExecutor

from tnfr.helpers import edge_version_cache, increment_edge_version


def test_edge_version_cache_thread_safety():
    G = nx.Graph()
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
