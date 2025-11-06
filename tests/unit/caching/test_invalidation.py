"""Tests for graph change tracking and cache invalidation."""

from __future__ import annotations

import pytest

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from tnfr.cache import (
    CacheLevel,
    TNFRHierarchicalCache,
    GraphChangeTracker,
    track_node_property_update,
)


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX required")
class TestGraphChangeTracker:
    """Test graph modification tracking."""

    def test_init(self):
        """Test tracker initialization."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        assert tracker.topology_changes == 0
        assert tracker.property_changes == 0

    def test_track_add_node(self):
        """Test that adding nodes invalidates topology cache."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()

        # Cache something dependent on topology
        cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})

        # Track graph
        tracker.track_graph_changes(G)

        # Add node
        G.add_node("n1")

        # Should invalidate topology cache
        assert cache.get("key1", CacheLevel.GRAPH_STRUCTURE) is None
        assert tracker.topology_changes == 1

    def test_track_remove_node(self):
        """Test that removing nodes invalidates topology cache."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()
        G.add_node("n1")

        cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})

        tracker.track_graph_changes(G)
        G.remove_node("n1")

        assert cache.get("key1", CacheLevel.GRAPH_STRUCTURE) is None
        assert tracker.topology_changes == 1

    def test_track_add_edge(self):
        """Test that adding edges invalidates topology cache."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()
        G.add_nodes_from(["n1", "n2"])

        cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})

        tracker.track_graph_changes(G)
        G.add_edge("n1", "n2")

        assert cache.get("key1", CacheLevel.GRAPH_STRUCTURE) is None
        assert tracker.topology_changes == 1

    def test_track_remove_edge(self):
        """Test that removing edges invalidates topology cache."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()
        G.add_edge("n1", "n2")

        cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})

        tracker.track_graph_changes(G)
        G.remove_edge("n1", "n2")

        assert cache.get("key1", CacheLevel.GRAPH_STRUCTURE) is None
        assert tracker.topology_changes == 1

    def test_topology_change_invalidates_neighbors(self):
        """Test that topology changes invalidate neighbor caches."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()

        cache.set("neighbors", [], CacheLevel.GRAPH_STRUCTURE, {"node_neighbors"})

        tracker.track_graph_changes(G)
        G.add_node("n1")

        assert cache.get("neighbors", CacheLevel.GRAPH_STRUCTURE) is None

    def test_node_property_change(self):
        """Test explicit node property change notification."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)

        # Cache entry dependent on node property
        cache.set("node_metric", 0.5, CacheLevel.DERIVED_METRICS, {"node_epi_n1"})

        # Notify of property change
        tracker.on_node_property_change("n1", "epi", 0.5, 0.7)

        # Should invalidate
        assert cache.get("node_metric", CacheLevel.DERIVED_METRICS) is None
        assert tracker.property_changes == 1

    def test_property_change_invalidates_global(self):
        """Test that property changes invalidate global dependencies."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)

        cache.set("global", 1, CacheLevel.DERIVED_METRICS, {"all_node_epi"})

        tracker.on_node_property_change("n1", "epi")

        assert cache.get("global", CacheLevel.DERIVED_METRICS) is None

    def test_structural_property_invalidates_derived(self):
        """Test that structural property changes invalidate derived metrics."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)

        cache.set("derived", 1, CacheLevel.DERIVED_METRICS, {"derived_metrics_n1"})

        tracker.on_node_property_change("n1", "vf")

        assert cache.get("derived", CacheLevel.DERIVED_METRICS) is None

    def test_reset_counters(self):
        """Test resetting change counters."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()

        tracker.track_graph_changes(G)
        G.add_node("n1")
        tracker.on_node_property_change("n1", "epi")

        assert tracker.topology_changes == 1
        assert tracker.property_changes == 1

        tracker.reset_counters()

        assert tracker.topology_changes == 0
        assert tracker.property_changes == 0

    def test_track_same_graph_twice_ignored(self):
        """Test that tracking same graph twice doesn't duplicate hooks."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()

        tracker.track_graph_changes(G)
        tracker.track_graph_changes(G)  # Should be no-op

        cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})
        G.add_node("n1")

        # Should only increment once
        assert tracker.topology_changes == 1

    def test_preserves_independent_caches(self):
        """Test that changes don't invalidate independent caches."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()

        # Cache dependent on topology
        cache.set("topo", 1, CacheLevel.GRAPH_STRUCTURE, {"graph_topology"})

        # Cache independent of topology
        cache.set("independent", 2, CacheLevel.TEMPORARY, {"other_dep"})

        tracker.track_graph_changes(G)
        G.add_node("n1")

        # Topology cache invalidated
        assert cache.get("topo", CacheLevel.GRAPH_STRUCTURE) is None

        # Independent cache preserved
        assert cache.get("independent", CacheLevel.TEMPORARY) == 2


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX required")
class TestTrackNodePropertyUpdate:
    """Test helper function for tracking property updates."""

    def test_updates_and_notifies(self):
        """Test that helper updates property and notifies tracker."""
        cache = TNFRHierarchicalCache()
        tracker = GraphChangeTracker(cache)
        G = nx.Graph()
        G.add_node("n1", epi=0.5)

        # Attach tracker to graph
        G.graph["_tnfr_change_tracker"] = tracker

        cache.set("key1", 1, CacheLevel.DERIVED_METRICS, {"node_epi_n1"})

        # Update property
        track_node_property_update(G, "n1", "epi", 0.7)

        # Property updated
        assert G.nodes["n1"]["epi"] == 0.7

        # Cache invalidated
        assert cache.get("key1", CacheLevel.DERIVED_METRICS) is None
        assert tracker.property_changes == 1

    def test_works_without_tracker(self):
        """Test that helper works even without tracker attached."""
        G = nx.Graph()
        G.add_node("n1", epi=0.5)

        # Should not raise even without tracker
        track_node_property_update(G, "n1", "epi", 0.7)

        assert G.nodes["n1"]["epi"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
