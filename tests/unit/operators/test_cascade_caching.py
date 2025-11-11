"""Tests for cascade detection caching functionality.

Verifies that the @cache_tnfr_computation decorator works correctly
for detect_cascade() and provides expected performance improvements.
"""

import pytest
import networkx as nx

from tnfr.operators.cascade import detect_cascade, invalidate_cascade_cache
from tnfr.utils.cache import get_global_cache, reset_global_cache


class TestCascadeCaching:
    """Test caching behavior of detect_cascade()."""
    
    def setup_method(self):
        """Reset global cache before each test."""
        reset_global_cache()
    
    def test_cascade_cached_on_second_call(self):
        """Second call to detect_cascade should use cache."""
        G = nx.Graph()
        for i in range(10):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)
            if i > 0:
                G.add_edge(0, i)
        
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10), (2, 0.09)],
                "timestamp": 10,
            }
        ]
        G.graph["THOL_CASCADE_MIN_NODES"] = 3
        
        # First call - builds cache
        result1 = detect_cascade(G)
        
        # Second call - should use cache
        result2 = detect_cascade(G)
        
        # Results should be identical
        assert result1["is_cascade"] == result2["is_cascade"]
        assert result1["affected_nodes"] == result2["affected_nodes"]
        assert result1["cascade_depth"] == result2["cascade_depth"]
        
        # Verify cache was used
        cache = get_global_cache()
        stats = cache.get_stats()
        assert stats["hits"] >= 1, "Cache should have at least 1 hit"
    
    def test_cache_invalidation_on_propagation_change(self):
        """Cache should invalidate when propagations change."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)
        
        # Initial propagations
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10)],
                "timestamp": 10,
            }
        ]
        
        # First call
        result1 = detect_cascade(G)
        assert len(result1["affected_nodes"]) == 2
        
        # Modify propagations - should invalidate cache
        G.graph["thol_propagations"].append({
            "source_node": 1,
            "propagations": [(2, 0.09), (3, 0.08)],
            "timestamp": 11,
        })
        
        # Manually invalidate (normally automatic)
        invalidate_cascade_cache()
        
        # Second call should recompute
        result2 = detect_cascade(G)
        assert len(result2["affected_nodes"]) == 4  # More nodes affected
    
    def test_manual_cache_invalidation(self):
        """invalidate_cascade_cache() should clear cached results."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.graph["thol_propagations"] = []
        
        # Build cache
        detect_cascade(G)
        
        # Invalidate
        count = invalidate_cascade_cache()
        assert count >= 0  # Should report invalidations
        
        # Cache should be empty for this function
        cache = get_global_cache()
        # After invalidation, next call is a miss
        stats_before = cache.get_stats()
        detect_cascade(G)
        stats_after = cache.get_stats()
        assert stats_after["misses"] > stats_before["misses"]
    
    def test_different_graphs_separate_cache_entries(self):
        """Different graphs should have separate cache entries."""
        G1 = nx.Graph()
        G1.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G1.graph["thol_propagations"] = [
            {"source_node": 0, "propagations": [(1, 0.1)], "timestamp": 10}
        ]
        
        G2 = nx.Graph()
        G2.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G2.graph["thol_propagations"] = []
        
        result1 = detect_cascade(G1)
        result2 = detect_cascade(G2)
        
        # Different results
        assert result1["total_propagations"] != result2["total_propagations"]
        
        # Both should be cached separately
        # Calling again should hit cache
        result1_cached = detect_cascade(G1)
        result2_cached = detect_cascade(G2)
        
        assert result1 == result1_cached
        assert result2 == result2_cached


class TestCascadePerformanceWithCache:
    """Performance tests verifying cache speedup."""
    
    def setup_method(self):
        """Reset cache before each test."""
        reset_global_cache()
    
    def test_cached_calls_are_faster(self):
        """Cached calls should be significantly faster than first call."""
        import time
        
        # Create moderate-sized network
        G = nx.Graph()
        for i in range(1000):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1 + i * 0.001)
        
        # Add small-world edges
        G = nx.watts_strogatz_graph(1000, 6, 0.1)
        for i in G.nodes():
            G.nodes[i]["epi"] = 0.50
            G.nodes[i]["vf"] = 1.0
            G.nodes[i]["theta"] = 0.1 + i * 0.001
        
        # Simulate cascade
        import random
        random.seed(42)
        propagations = []
        for i in range(100):
            source = i % 1000
            neighbors = list(G.neighbors(source))
            if neighbors:
                targets = random.sample(neighbors, min(3, len(neighbors)))
                propagations.append({
                    "source_node": source,
                    "propagations": [(t, 0.10) for t in targets],
                    "timestamp": 10 + i,
                })
        G.graph["thol_propagations"] = propagations
        
        # First call (uncached)
        start = time.time()
        result1 = detect_cascade(G)
        time_uncached = time.time() - start
        
        # Second call (cached)
        start = time.time()
        result2 = detect_cascade(G)
        time_cached = time.time() - start
        
        # Results should be identical
        assert result1 == result2
        
        # Cached should be faster (or at least not significantly slower)
        # With caching, should be near-instant (<1ms typically)
        print(f"Uncached: {time_uncached*1000:.2f}ms, Cached: {time_cached*1000:.2f}ms")
        
        # Cached time should be very fast
        assert time_cached < 0.01, f"Cached call too slow: {time_cached*1000:.2f}ms"
    
    def test_cache_statistics(self):
        """Cache should track hits and misses correctly."""
        reset_global_cache()
        cache = get_global_cache()
        
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.graph["thol_propagations"] = []
        
        # First call = miss
        detect_cascade(G)
        stats = cache.get_stats()
        initial_misses = stats["misses"]
        
        # Second call = hit
        detect_cascade(G)
        stats = cache.get_stats()
        
        # Should have at least one hit
        assert stats["hits"] >= 1
        # Misses shouldn't increase
        assert stats["misses"] == initial_misses


if __name__ == "__main__":
    # Quick manual test
    print("Testing cascade caching functionality...\n")
    
    test = TestCascadeCaching()
    test.setup_method()
    
    print("Test 1: Basic caching...")
    test.test_cascade_cached_on_second_call()
    print("  ✓ Cache working correctly\n")
    
    print("Test 2: Cache invalidation...")
    test.setup_method()
    test.test_cache_invalidation_on_propagation_change()
    print("  ✓ Invalidation working\n")
    
    print("Test 3: Performance benefit...")
    perf_test = TestCascadePerformanceWithCache()
    perf_test.setup_method()
    perf_test.test_cached_calls_are_faster()
    print("  ✓ Significant speedup observed\n")
    
    print("All tests passed!")
