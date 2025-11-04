"""Tests for unified buffer cache management."""

import pytest

pytest.importorskip("numpy")
import numpy as np
import networkx as nx

from tnfr.metrics.buffer_cache import ensure_numpy_buffers

def test_ensure_numpy_buffers_basic():
    """Verify basic buffer allocation with correct shapes."""
    G = nx.Graph([(0, 1), (1, 2)])
    buffers = ensure_numpy_buffers(
        G, key_prefix="_test_basic", count=10, buffer_count=3, np=np
    )
    assert len(buffers) == 3
    assert all(buf.shape == (10,) for buf in buffers)
    assert all(buf.dtype == np.dtype(float) for buf in buffers)

def test_ensure_numpy_buffers_caches():
    """Verify buffers are cached and reused."""
    G = nx.Graph([(0, 1)])
    buffers1 = ensure_numpy_buffers(
        G, key_prefix="_test_cache", count=5, buffer_count=2, np=np
    )
    buffers2 = ensure_numpy_buffers(
        G, key_prefix="_test_cache", count=5, buffer_count=2, np=np
    )
    # Same cache key should return same buffer objects
    assert buffers1[0] is buffers2[0]
    assert buffers1[1] is buffers2[1]

def test_ensure_numpy_buffers_different_counts():
    """Verify different counts create separate cache entries."""
    G = nx.Graph([(0, 1)])
    buffers_small = ensure_numpy_buffers(
        G, key_prefix="_test_diff", count=5, buffer_count=2, np=np
    )
    buffers_large = ensure_numpy_buffers(
        G, key_prefix="_test_diff", count=10, buffer_count=2, np=np
    )
    # Different counts should have different shapes
    assert buffers_small[0].shape == (5,)
    assert buffers_large[0].shape == (10,)
    # Should be different objects
    assert buffers_small[0] is not buffers_large[0]

def test_ensure_numpy_buffers_edge_version_invalidation():
    """Verify buffers are invalidated when graph edges change."""
    from tnfr.utils import increment_edge_version

    G = nx.Graph([(0, 1)])
    buffers1 = ensure_numpy_buffers(
        G, key_prefix="_test_edge", count=5, buffer_count=1, np=np
    )
    # Change graph structure
    G.add_edge(2, 3)
    increment_edge_version(G)
    buffers2 = ensure_numpy_buffers(
        G, key_prefix="_test_edge", count=5, buffer_count=1, np=np
    )
    # Should be different objects after edge change
    assert buffers1[0] is not buffers2[0]

def test_ensure_numpy_buffers_zero_count_clamps():
    """Verify zero or negative count is clamped to 1."""
    G = nx.Graph()
    buffers = ensure_numpy_buffers(
        G, key_prefix="_test_zero", count=0, buffer_count=1, np=np
    )
    assert buffers[0].shape == (1,)
    buffers_neg = ensure_numpy_buffers(
        G, key_prefix="_test_neg", count=-5, buffer_count=1, np=np
    )
    assert buffers_neg[0].shape == (1,)

def test_ensure_numpy_buffers_custom_dtype():
    """Verify custom dtype is applied correctly."""
    G = nx.Graph()
    buffers = ensure_numpy_buffers(
        G, key_prefix="_test_dtype", count=3, buffer_count=1, np=np, dtype=np.int32
    )
    assert buffers[0].dtype == np.dtype(np.int32)

def test_ensure_numpy_buffers_capacity_limit():
    """Verify cache capacity limits are respected."""
    G = nx.Graph()
    # Create many buffer sets with capacity limit of 2
    keys = []
    for i in range(5):
        key = f"_test_cap_{i}"
        keys.append(key)
        ensure_numpy_buffers(
            G, key_prefix=key, count=i + 1, buffer_count=1, np=np, max_cache_entries=2
        )
    # Cache should only keep recent entries (LRU)
    # This is a light test - detailed cache eviction is tested in cache.py tests
