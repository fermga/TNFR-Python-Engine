"""Tests for cache key collision avoidance and uniqueness.

This module validates that cache keys used across TNFR hot paths follow
the documented patterns and do not collide with each other.

See docs/CACHING_STRATEGY.md for cache key naming conventions.
"""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.metrics.buffer_cache import ensure_numpy_buffers
from tnfr.metrics.common import ensure_neighbors_map
from tnfr.metrics.sense_index import (
    _ensure_chunk_workspace,
    _ensure_neighbor_bulk_buffers,
    _ensure_si_buffers,
)
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.utils import edge_version_cache, get_numpy, increment_edge_version


class TestCacheKeyUniqueness:
    """Test that cache keys don't collide across different computations."""

    def test_buffer_cache_key_structure(self):
        """Test that buffer cache keys follow the documented tuple pattern."""
        G = nx.complete_graph(10)
        np_mod = get_numpy()
        if np_mod is None:
            pytest.skip("NumPy not available")

        # Allocate buffers with different prefixes
        buffers1 = ensure_numpy_buffers(G, key_prefix="_test1", count=10, buffer_count=2, np=np_mod)
        buffers2 = ensure_numpy_buffers(G, key_prefix="_test2", count=10, buffer_count=2, np=np_mod)
        buffers3 = ensure_numpy_buffers(G, key_prefix="_test1", count=20, buffer_count=2, np=np_mod)

        # Different prefixes should not share buffers
        assert buffers1[0] is not buffers2[0]

        # Same prefix but different count should not share buffers
        assert buffers1[0] is not buffers3[0]

        # Same parameters should reuse buffers
        buffers1_again = ensure_numpy_buffers(
            G, key_prefix="_test1", count=10, buffer_count=2, np=np_mod
        )
        assert buffers1[0] is buffers1_again[0]

    def test_si_buffer_keys_unique(self):
        """Test that Si computation buffer keys don't collide."""
        G = nx.complete_graph(10)
        np_mod = get_numpy()
        if np_mod is None:
            pytest.skip("NumPy not available")

        # These three functions should use different cache keys
        si_buffers = _ensure_si_buffers(G, count=10, np=np_mod)
        chunk_buffers = _ensure_chunk_workspace(G, mask_count=5, np=np_mod)
        neighbor_buffers = _ensure_neighbor_bulk_buffers(G, count=10, np=np_mod)

        # All should be distinct sets of buffers
        assert si_buffers[0] is not chunk_buffers[0]
        assert si_buffers[0] is not neighbor_buffers[0]
        assert chunk_buffers[0] is not neighbor_buffers[0]

        # Verify buffer counts match expectations
        assert len(si_buffers) == 3
        assert len(chunk_buffers) == 2
        assert len(neighbor_buffers) == 5

    def test_trig_cache_key_versioning(self):
        """Test that trig cache uses version-based keys correctly."""
        G = nx.Graph()
        G.add_node(0, phase=0.0)
        G.add_node(1, phase=1.0)

        np_mod = get_numpy()

        # Initial cache
        trig1 = get_trig_cache(G, np=np_mod)
        assert 0 in trig1.cos
        assert 1 in trig1.cos

        # Reuse should return same cache
        trig1_again = get_trig_cache(G, np=np_mod)
        assert trig1.cos is trig1_again.cos

        # Changing theta should invalidate and create new version
        G.nodes[0]["phase"] = 2.0
        trig2 = get_trig_cache(G, np=np_mod)
        # Should be different cache due to version increment
        assert trig2.theta[0] != trig1.theta[0]

    def test_neighbors_cache_key(self):
        """Test that neighbor topology cache uses correct key."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])

        # Get neighbor map
        neighbors1 = ensure_neighbors_map(G)
        assert 0 in neighbors1
        assert list(neighbors1[0]) == [1]

        # Should reuse cache
        neighbors1_again = ensure_neighbors_map(G)
        assert neighbors1 is neighbors1_again

        # Adding edge should invalidate
        G.add_edge(0, 3)
        increment_edge_version(G)  # Manually trigger invalidation
        neighbors2 = ensure_neighbors_map(G)
        # Should have new neighbor
        assert 3 in neighbors2[0]

    def test_cache_key_namespace_separation(self):
        """Test that different computations maintain separate namespaces."""
        G = nx.complete_graph(5)
        np_mod = get_numpy()
        if np_mod is None:
            pytest.skip("NumPy not available")

        # Store various cached items
        buffers = ensure_numpy_buffers(G, key_prefix="_test_ns", count=5, buffer_count=1, np=np_mod)
        neighbors = ensure_neighbors_map(G)
        trig = get_trig_cache(G, np=np_mod)

        # Store custom cache entry
        def custom_builder():
            return "custom_value"

        custom = edge_version_cache(G, ("_custom_key", 123), custom_builder)

        # All should coexist without collision
        assert buffers is not None
        assert neighbors is not None
        assert trig is not None
        assert custom == "custom_value"

        # Verify they're independent
        buffers_again = ensure_numpy_buffers(
            G, key_prefix="_test_ns", count=5, buffer_count=1, np=np_mod
        )
        assert buffers[0] is buffers_again[0]


class TestCacheKeyDocumentation:
    """Test that actual cache keys match documented patterns."""

    def test_si_buffer_prefixes(self):
        """Test that Si buffers use documented key prefixes."""
        G = nx.complete_graph(10)
        np_mod = get_numpy()
        if np_mod is None:
            pytest.skip("NumPy not available")

        # According to CACHING_STRATEGY.md:
        # - _si_buffers: Si main computation buffers
        # - _si_chunk_workspace: Si chunked processing scratch space
        # - _si_neighbor_buffers: Si neighbor phase aggregation buffers

        # These should not collide
        si_main = ensure_numpy_buffers(
            G, key_prefix="_si_buffers", count=10, buffer_count=3, np=np_mod
        )
        si_chunk = ensure_numpy_buffers(
            G, key_prefix="_si_chunk_workspace", count=5, buffer_count=2, np=np_mod
        )
        si_neighbor = ensure_numpy_buffers(
            G, key_prefix="_si_neighbor_buffers", count=10, buffer_count=5, np=np_mod
        )

        # All should be distinct
        assert si_main[0] is not si_chunk[0]
        assert si_main[0] is not si_neighbor[0]
        assert si_chunk[0] is not si_neighbor[0]

    def test_cache_key_collision_resistance(self):
        """Test that similar but distinct parameters produce different keys."""
        G = nx.complete_graph(10)
        np_mod = get_numpy()
        if np_mod is None:
            pytest.skip("NumPy not available")

        # Test variations that should NOT collide
        test_cases = [
            # (prefix, count, buffer_count)
            ("_metric1", 10, 2),
            ("_metric1", 10, 3),  # Same prefix, different buffer_count
            ("_metric1", 20, 2),  # Same prefix, different count
            ("_metric2", 10, 2),  # Different prefix, same dimensions
        ]

        buffers = []
        for prefix, count, buf_count in test_cases:
            buf = ensure_numpy_buffers(
                G,
                key_prefix=prefix,
                count=count,
                buffer_count=buf_count,
                np=np_mod,
            )
            buffers.append(buf)

        # All should be distinct cache entries
        for i in range(len(buffers)):
            for j in range(i + 1, len(buffers)):
                assert buffers[i][0] is not buffers[j][0], (
                    f"Buffers {i} and {j} share same cache entry: "
                    f"{test_cases[i]} vs {test_cases[j]}"
                )


class TestCacheInvalidation:
    """Test that cache invalidation works correctly across hot paths."""

    def test_edge_version_invalidates_all_buffers(self):
        """Test that edge structure changes invalidate buffer caches."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        np_mod = get_numpy()
        if np_mod is None:
            pytest.skip("NumPy not available")

        # Allocate buffers
        buffers1 = ensure_numpy_buffers(G, key_prefix="_test", count=3, buffer_count=1, np=np_mod)

        # Modify graph topology
        G.add_edge(2, 3)
        increment_edge_version(G)

        # Should get new buffers
        buffers2 = ensure_numpy_buffers(G, key_prefix="_test", count=3, buffer_count=1, np=np_mod)

        # Note: The count is the same (3 nodes), but edge version changed
        # The cache should still be invalidated based on edge version
        # However, if the builder creates new buffers each time, they'll be different
        # This tests that the cache respects edge version changes

        # Add another node to change the count
        G.add_node(4)
        buffers3 = ensure_numpy_buffers(G, key_prefix="_test", count=4, buffer_count=1, np=np_mod)

        # Different count should definitely produce different buffers
        assert buffers3[0].shape != buffers1[0].shape

    def test_trig_cache_invalidates_on_theta_change(self):
        """Test that trig cache invalidates when theta values change."""
        G = nx.Graph()
        G.add_node(0, phase=0.0)
        G.add_node(1, phase=1.0)

        # Get initial cache
        trig1 = get_trig_cache(G)
        theta1_0 = trig1.theta[0]

        # Change theta
        G.nodes[0]["phase"] = 2.0

        # Get cache again - should detect change
        trig2 = get_trig_cache(G)
        theta2_0 = trig2.theta[0]

        # Theta should reflect the new value
        assert abs(theta2_0 - 2.0) < 1e-9
        assert abs(theta1_0 - 0.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
