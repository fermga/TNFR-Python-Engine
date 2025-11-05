"""Performance tests for NodeNX.from_graph() cache optimization.

This test validates the performance improvements from the optimized
NodeNX.from_graph() implementation which uses:
1. Lock-free fast path for cache hits (common case)
2. Per-node locking instead of per-graph (reduced contention)
3. Simplified cache management (no double-caching)
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tnfr.node import NodeNX


@pytest.fixture
def graph_with_nodes():
    """Create a graph with nodes for testing."""
    pytest.importorskip("networkx")
    import networkx as nx

    G = nx.Graph()
    for i in range(50):
        G.add_node(
            i,
            theta=0.0,
            vf=1.0,
            EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)},
        )
    return G


def test_cache_hit_is_fast(graph_with_nodes):
    """Verify that cache hits are fast (lock-free read path)."""
    G = graph_with_nodes

    # Warm up the cache
    for i in range(50):
        NodeNX.from_graph(G, i)

    # Measure cache hit performance
    start = time.perf_counter()
    iterations = 10000
    for _ in range(iterations):
        for i in range(50):
            node = NodeNX.from_graph(G, i)
    end = time.perf_counter()

    # Cache hits should be very fast (< 1 µs per access)
    time_per_access = (end - start) / (iterations * 50)
    assert time_per_access < 1e-6, f"Cache hit too slow: {time_per_access * 1e6:.2f} µs"


def test_parallel_access_has_low_contention(graph_with_nodes):
    """Verify that parallel access has minimal lock contention."""
    G = graph_with_nodes

    # Warm up the cache
    for i in range(50):
        NodeNX.from_graph(G, i)

    def access_nodes():
        """Access all nodes multiple times."""
        for _ in range(100):
            for i in range(50):
                node = NodeNX.from_graph(G, i)

    # Measure parallel access
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(access_nodes) for _ in range(8)]
        for f in futures:
            f.result()
    end = time.perf_counter()

    # Parallel access should not have significant overhead
    # With lock-free reads, parallel should be close to sequential speed
    time_per_access = (end - start) / (8 * 100 * 50)
    assert time_per_access < 2e-6, f"Parallel access too slow: {time_per_access * 1e6:.2f} µs"


def test_no_duplicate_node_instances_sequential(graph_with_nodes):
    """Verify that sequential access returns the same cached instance."""
    G = graph_with_nodes

    node1 = NodeNX.from_graph(G, 0)
    node2 = NodeNX.from_graph(G, 0)
    node3 = NodeNX.from_graph(G, 0)

    assert node1 is node2
    assert node2 is node3


def test_no_duplicate_node_instances_parallel(graph_with_nodes):
    """Verify that parallel access returns the same cached instance."""
    G = graph_with_nodes

    def get_node():
        return NodeNX.from_graph(G, 0)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(get_node) for _ in range(32)]
        nodes = [f.result() for f in futures]

    # All threads should get the same cached instance
    first = nodes[0]
    assert all(node is first for node in nodes)


def test_cache_initialization_is_thread_safe(graph_with_nodes):
    """Verify that cache initialization with multiple threads is safe."""
    G = graph_with_nodes

    # Clear any existing cache
    G.graph.pop("_node_cache", None)
    G.graph.pop("_node_cache_weak", None)

    def get_node(node_id):
        return NodeNX.from_graph(G, node_id)

    # Access different nodes concurrently (cache doesn't exist yet)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_node, i % 50) for i in range(100)]
        nodes = [f.result() for f in futures]

    # Verify cache was created
    assert "_node_cache" in G.graph

    # Verify each node ID has exactly one cached instance
    for i in range(50):
        matching_nodes = [n for n in nodes if n.n == i]
        if matching_nodes:
            first = matching_nodes[0]
            assert all(n is first for n in matching_nodes)


def test_weak_cache_separate_from_strong_cache(graph_with_nodes):
    """Verify that weak and strong caches remain independent."""
    G = graph_with_nodes

    # Get with strong cache
    node_strong = NodeNX.from_graph(G, 0, use_weak_cache=False)

    # Get with weak cache
    node_weak = NodeNX.from_graph(G, 0, use_weak_cache=True)

    # They should be different instances due to separate caches
    assert node_strong is not node_weak

    # Both caches should exist
    assert "_node_cache" in G.graph
    assert "_node_cache_weak" in G.graph
