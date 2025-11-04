"""Test NodeNX.from_graph with WeakValueDictionary for memory management."""

import gc
import weakref

import pytest

from tnfr.node import NodeNX

def test_node_weak_cache_releases_unused_instances():
    """Verify that use_weak_cache=True allows garbage collection of NodeNX instances."""
    pytest.importorskip("networkx")
    import networkx as nx

    # Create a graph
    G = nx.Graph()
    G.add_node(0, theta=0.0, vf=1.0, EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)})
    G.add_node(1, theta=0.1, vf=1.0, EPI={"continuous": (0.6,), "discrete": (), "grid": (0.0, 1.0)})

    # Get node with weak cache
    node0 = NodeNX.from_graph(G, 0, use_weak_cache=True)
    assert node0.n == 0

    # Create a weak reference to track garbage collection
    weak_ref = weakref.ref(node0)
    assert weak_ref() is not None

    # The node should be in the weak cache
    assert "_node_cache_weak" in G.graph
    weak_cache = G.graph["_node_cache_weak"]
    assert 0 in weak_cache

    # Delete the strong reference
    del node0

    # Force garbage collection
    gc.collect()

    # The weak reference should now be None (object was collected)
    assert weak_ref() is None

    # The weak cache may or may not contain the key (depends on WeakValueDict implementation)
    # but getting the node again should create a new instance
    node0_new = NodeNX.from_graph(G, 0, use_weak_cache=True)
    assert node0_new.n == 0

def test_node_strong_cache_retains_instances():
    """Verify that default (strong) cache retains NodeNX instances."""
    pytest.importorskip("networkx")
    import networkx as nx

    # Create a graph
    G = nx.Graph()
    G.add_node(0, theta=0.0, vf=1.0, EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)})

    # Get node with strong cache (default)
    node0 = NodeNX.from_graph(G, 0)
    assert node0.n == 0

    # Create a weak reference
    weak_ref = weakref.ref(node0)
    assert weak_ref() is not None

    # Delete the local reference
    del node0

    # Force garbage collection
    gc.collect()

    # The weak reference should still be valid (object retained by cache)
    assert weak_ref() is not None

    # Getting the node again should return the same cached instance
    node0_again = NodeNX.from_graph(G, 0)
    assert weak_ref() is node0_again

def test_node_cache_separate_strong_and_weak():
    """Verify that strong and weak caches are independent."""
    pytest.importorskip("networkx")
    import networkx as nx

    # Create a graph
    G = nx.Graph()
    G.add_node(0, theta=0.0, vf=1.0, EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)})

    # Get node with strong cache
    node_strong = NodeNX.from_graph(G, 0, use_weak_cache=False)

    # Get node with weak cache
    node_weak = NodeNX.from_graph(G, 0, use_weak_cache=True)

    # They should be different instances
    assert node_strong is not node_weak

    # Both caches should exist
    assert "_node_cache" in G.graph
    assert "_node_cache_weak" in G.graph

def test_node_weak_cache_reuses_live_instance():
    """Verify that weak cache returns the same instance when it's still alive."""
    pytest.importorskip("networkx")
    import networkx as nx

    # Create a graph
    G = nx.Graph()
    G.add_node(0, theta=0.0, vf=1.0, EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)})

    # Get node with weak cache
    node0_first = NodeNX.from_graph(G, 0, use_weak_cache=True)
    node0_second = NodeNX.from_graph(G, 0, use_weak_cache=True)

    # Should be the same instance
    assert node0_first is node0_second
