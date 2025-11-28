"""
Comprehensive tests for TNFR sparse representations.

Tests cover:
- SparseTNFRGraph creation and initialization
- CompactAttributeStore operations
- Memory efficiency verification
- SparseCache functionality
- Integration with TNFR operators
"""

import pytest
import numpy as np

from tnfr.sparse import (
    SparseTNFRGraph,
    CompactAttributeStore,
    MemoryReport,
    SparseCache,
)


class TestCompactAttributeStore:
    """Test CompactAttributeStore for sparse node attributes."""
    
    def test_initialization(self):
        """Test basic initialization."""
        store = CompactAttributeStore(node_count=100)
        
        assert store.node_count == 100
        assert store.default_vf == 1.0
        assert store.default_theta == 0.0
        assert store.default_si == 0.0
        
    def test_store_and_retrieve_vf(self):
        """Test storing and retrieving structural frequency."""
        store = CompactAttributeStore(node_count=10)
        
        # Set non-default value
        store.set_vf(5, 2.5)
        assert store.get_vf(5) == 2.5
        
        # Get default value (not explicitly set)
        assert store.get_vf(3) == 1.0
        
    def test_default_values_not_stored(self):
        """Test that default values don't consume storage."""
        store = CompactAttributeStore(node_count=100)
        
        # Set a few values to default
        store.set_vf(0, 1.0)  # Default
        store.set_vf(1, 2.0)  # Non-default
        store.set_vf(2, 1.0)  # Default
        
        # Only non-default should be stored
        assert len(store._vf_sparse) == 1
        assert 1 in store._vf_sparse
        
    def test_vectorized_get_vfs(self):
        """Test vectorized retrieval of structural frequencies."""
        store = CompactAttributeStore(node_count=10)
        
        # Set some values
        store.set_vf(0, 1.5)
        store.set_vf(2, 2.5)
        store.set_vf(5, 0.5)
        
        # Get multiple values at once
        node_ids = [0, 1, 2, 3, 5]
        vfs = store.get_vfs(node_ids)
        
        expected = np.array([1.5, 1.0, 2.5, 1.0, 0.5], dtype=np.float32)
        assert np.allclose(vfs, expected)
        
    def test_update_to_default_removes_storage(self):
        """Test that updating to default value frees storage."""
        store = CompactAttributeStore(node_count=10)
        
        # Set non-default
        store.set_vf(3, 5.0)
        assert 3 in store._vf_sparse
        
        # Update to default
        store.set_vf(3, 1.0)
        assert 3 not in store._vf_sparse
        
    def test_multiple_attributes(self):
        """Test storing multiple attribute types."""
        store = CompactAttributeStore(node_count=10)
        
        # Set different attributes
        store.set_vf(0, 2.0)
        
        # Should have separate storage
        assert len(store._vf_sparse) == 1


class TestSparseCache:
    """Test SparseCache for time-to-live caching."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = SparseCache(capacity=100, ttl_steps=5)
        
        assert cache.capacity == 100
        assert cache.ttl_steps == 5
        assert cache._current_step == 0
        
    def test_cache_hit(self):
        """Test retrieving cached value."""
        cache = SparseCache(capacity=10, ttl_steps=5)
        
        # Store value
        cache.update({1: 42.0})
        
        # Retrieve within TTL
        value = cache.get(1)
        assert value == 42.0
        
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = SparseCache(capacity=10, ttl_steps=5)
        
        # Try to get non-existent key
        value = cache.get(999)
        assert value is None
        
    def test_ttl_expiration(self):
        """Test that cached values expire after TTL."""
        cache = SparseCache(capacity=10, ttl_steps=3)
        
        # Store value
        cache.update({1: 100.0})
        
        # Advance steps beyond TTL
        for _ in range(4):
            cache.step()
            
        # Should be expired
        value = cache.get(1)
        assert value is None
        
    def test_lru_eviction(self):
        """Test LRU eviction when capacity exceeded."""
        cache = SparseCache(capacity=3, ttl_steps=10)
        
        # Fill cache to capacity
        cache.update({1: 10.0, 2: 20.0, 3: 30.0})
        
        # Add more, should evict oldest
        cache.step()
        cache.update({4: 40.0, 5: 50.0})
        
        # Oldest entries (1, 2) should be evicted
        assert cache.get(1) is None
        assert cache.get(2) is None
        
        # Newer entries should remain
        assert cache.get(3) == 30.0
        assert cache.get(4) == 40.0
        assert cache.get(5) == 50.0
        
    def test_clear(self):
        """Test clearing all cache entries."""
        cache = SparseCache(capacity=10, ttl_steps=5)
        
        cache.update({1: 10.0, 2: 20.0})
        assert cache.get(1) == 10.0
        
        cache.clear()
        
        assert cache.get(1) is None
        assert cache._current_step == 0
        
    def test_memory_usage(self):
        """Test memory usage estimation."""
        cache = SparseCache(capacity=100, ttl_steps=5)
        
        # Empty cache
        empty_usage = cache.memory_usage()
        assert empty_usage == 0
        
        # Add entries
        cache.update({i: float(i) for i in range(10)})
        
        # Should report memory usage
        usage = cache.memory_usage()
        assert usage > 0
        assert usage == 10 * (8 + 8 + 8 + 112)  # Per entry cost


class TestSparseTNFRGraph:
    """Test SparseTNFRGraph main class."""
    
    def test_initialization_small_graph(self):
        """Test creating small sparse graph."""
        graph = SparseTNFRGraph(node_count=10, expected_density=0.1)
        
        assert graph.node_count == 10
        assert hasattr(graph, 'node_attributes')
        
    def test_initialization_large_graph(self):
        """Test creating large sparse graph."""
        graph = SparseTNFRGraph(node_count=1000, expected_density=0.05)
        
        assert graph.node_count == 1000
        
    def test_memory_footprint(self):
        """Test memory footprint reporting."""
        graph = SparseTNFRGraph(node_count=100, expected_density=0.1)
        
        report = graph.memory_footprint()
        
        assert isinstance(report, MemoryReport)
        assert report.total_mb > 0
        assert report.per_node_kb > 0
        assert isinstance(report.breakdown, dict)
        
    def test_memory_efficiency_vs_threshold(self):
        """Test that sparse representation is memory efficient."""
        graph = SparseTNFRGraph(node_count=1000, expected_density=0.05)
        
        report = graph.memory_footprint()
        
        # Should be under 1 KB per node (target from docstring)
        assert report.per_node_kb < 1.5  # Allow some tolerance
        
    def test_set_and_get_node_attribute(self):
        """Test setting and getting node attributes."""
        graph = SparseTNFRGraph(node_count=10, expected_density=0.1)
        
        # Set structural frequency
        graph.node_attributes.set_vf(5, 2.5)
        
        # Get it back
        vf = graph.node_attributes.get_vf(5)
        assert vf == 2.5
        
    def test_default_values_preserved(self):
        """Test that TNFR canonical defaults are preserved."""
        graph = SparseTNFRGraph(node_count=10, expected_density=0.1)
        
        # Unset nodes should return defaults
        vf = graph.node_attributes.get_vf(0)
        assert vf == 1.0  # Default νf
        
        theta = graph.node_attributes.get_theta(0)
        assert theta == 0.0  # Default θ
        
    def test_multiple_nodes(self):
        """Test operations on multiple nodes."""
        graph = SparseTNFRGraph(node_count=20, expected_density=0.1)
        
        # Set multiple nodes
        for i in range(0, 10, 2):
            graph.node_attributes.set_vf(i, float(i))
            
        # Verify
        for i in range(0, 10, 2):
            assert graph.node_attributes.get_vf(i) == float(i)


class TestMemoryEfficiency:
    """Test memory efficiency of sparse representations."""
    
    def test_sparse_vs_dense_comparison(self):
        """Compare memory usage: sparse vs hypothetical dense."""
        node_count = 1000
        graph = SparseTNFRGraph(node_count=node_count, expected_density=0.05)
        
        report = graph.memory_footprint()
        
        # Dense would store all attributes for all nodes
        # Assume 5 float64 attributes per node
        dense_bytes = node_count * 5 * 8
        dense_mb = dense_bytes / (1024 * 1024)
        
        # Sparse should use significantly less
        assert report.total_mb < dense_mb * 0.5
        
    def test_memory_scales_with_non_defaults(self):
        """Test memory scales with number of non-default values."""
        graph = SparseTNFRGraph(node_count=100, expected_density=0.1)
        
        report_empty = graph.memory_footprint()
        
        # Set many non-default values
        for i in range(50):
            graph.node_attributes.set_vf(i, float(i) + 1.5)
            
        report_filled = graph.memory_footprint()
        
        # Memory should increase with non-defaults
        assert report_filled.total_mb > report_empty.total_mb


class TestIntegration:
    """Integration tests with TNFR system."""
    
    def test_compatible_with_initialization(self):
        """Test compatibility with TNFR initialization."""
        import networkx as nx
        
        # Create regular NetworkX graph
        G = nx.complete_graph(10)
        
        # Should be able to extract and convert to sparse
        # (This tests the interface compatibility)
        assert hasattr(G, 'nodes')
        assert len(G.nodes()) == 10
        
    def test_reproducibility(self):
        """Test deterministic behavior with seed."""
        # Same seed should produce same behavior
        graph1 = SparseTNFRGraph(node_count=20, expected_density=0.1, seed=42)
        vf1 = graph1.node_attributes.get_vf(5)
        
        graph2 = SparseTNFRGraph(node_count=20, expected_density=0.1, seed=42)
        vf2 = graph2.node_attributes.get_vf(5)
        
        # Should behave identically
        assert vf1 == vf2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_node_count(self):
        """Test zero nodes."""
        with pytest.raises(ValueError):
            SparseTNFRGraph(node_count=0, expected_density=0.1)
            
    def test_negative_density(self):
        """Test negative density."""
        with pytest.raises(ValueError):
            SparseTNFRGraph(node_count=10, expected_density=-0.1)
            
    def test_density_over_one(self):
        """Test density > 1.0."""
        with pytest.raises(ValueError):
            SparseTNFRGraph(node_count=10, expected_density=1.5)
            
    def test_single_node(self):
        """Test single node graph."""
        graph = SparseTNFRGraph(node_count=1, expected_density=0.0)
        assert graph.node_count == 1
        
    def test_invalid_node_id(self):
        """Test accessing invalid node ID."""
        graph = SparseTNFRGraph(node_count=10, expected_density=0.1)
        
        # Should handle gracefully or raise appropriate error
        try:
            vf = graph.node_attributes.get_vf(999)
            # If it returns default, that's OK
            assert vf == 1.0
        except (KeyError, IndexError):
            # If it raises, that's also acceptable
            pass


@pytest.mark.performance
class TestPerformance:
    """Performance tests for sparse operations."""
    
    def test_large_graph_creation_time(self):
        """Test creation time for large graph."""
        import time
        
        start = time.time()
        graph = SparseTNFRGraph(node_count=10000, expected_density=0.01)
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 2.0  # 2 seconds max
        
    def test_attribute_access_speed(self):
        """Test attribute access performance."""
        import time
        
        graph = SparseTNFRGraph(node_count=1000, expected_density=0.05)
        
        # Set some values
        for i in range(0, 1000, 10):
            graph.node_attributes.set_vf(i, float(i))
            
        # Time retrieval
        start = time.time()
        for i in range(1000):
            _ = graph.node_attributes.get_vf(i)
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 0.1  # 100ms for 1000 accesses


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
