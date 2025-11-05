"""Example demonstrating TNFR intelligent caching system.

This example shows how to use the hierarchical caching system with
dependency-aware invalidation for TNFR computations.
"""

import time
import networkx as nx
from tnfr.caching import (
    TNFRHierarchicalCache,
    CacheLevel,
    cache_tnfr_computation,
    GraphChangeTracker,
    PersistentTNFRCache,
)


def example_basic_caching():
    """Demonstrate basic cache operations."""
    print("=" * 60)
    print("Example 1: Basic Hierarchical Cache")
    print("=" * 60)
    
    # Create cache with 128MB limit
    cache = TNFRHierarchicalCache(max_memory_mb=128)
    
    # Store values at different levels
    cache.set(
        "topology_hash",
        "abc123",
        CacheLevel.GRAPH_STRUCTURE,
        dependencies={'graph_topology'},
        computation_cost=10.0
    )
    
    cache.set(
        "node_epi_1",
        0.85,
        CacheLevel.NODE_PROPERTIES,
        dependencies={'node_epi_1'},
        computation_cost=1.0
    )
    
    cache.set(
        "si_node_1",
        0.92,
        CacheLevel.DERIVED_METRICS,
        dependencies={'node_epi_1', 'node_vf_1', 'graph_topology'},
        computation_cost=50.0
    )
    
    # Retrieve values
    print(f"Topology hash: {cache.get('topology_hash', CacheLevel.GRAPH_STRUCTURE)}")
    print(f"Node EPI: {cache.get('node_epi_1', CacheLevel.NODE_PROPERTIES)}")
    print(f"Si metric: {cache.get('si_node_1', CacheLevel.DERIVED_METRICS)}")
    
    # Show stats
    stats = cache.get_stats()
    print(f"\nCache stats: hits={stats['hits']}, misses={stats['misses']}, "
          f"hit_rate={stats['hit_rate']:.2%}")
    print(f"Memory used: {stats['memory_used_mb']:.2f} MB")
    
    # Invalidate by dependency
    print("\n--- Invalidating 'node_epi_1' ---")
    count = cache.invalidate_by_dependency('node_epi_1')
    print(f"Invalidated {count} entries")
    
    # Check what's still cached
    print(f"Topology hash: {cache.get('topology_hash', CacheLevel.GRAPH_STRUCTURE)}")
    print(f"Node EPI: {cache.get('node_epi_1', CacheLevel.NODE_PROPERTIES)}")
    print(f"Si metric: {cache.get('si_node_1', CacheLevel.DERIVED_METRICS)}")
    
    print()


def example_decorator_caching():
    """Demonstrate decorator-based caching."""
    print("=" * 60)
    print("Example 2: Decorator-Based Caching")
    print("=" * 60)
    
    call_count = {'expensive': 0, 'cheap': 0}
    
    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS,
        dependencies={'graph_topology', 'all_node_epi'},
        cost_estimator=lambda graph: len(graph.nodes()) * 10,
    )
    def compute_expensive_metric(graph):
        """Expensive computation that benefits from caching."""
        call_count['expensive'] += 1
        time.sleep(0.1)  # Simulate expensive computation
        return sum(data.get('epi', 0) for _, data in graph.nodes(data=True))
    
    @cache_tnfr_computation(
        level=CacheLevel.TEMPORARY,
        dependencies={'node_count'},
    )
    def compute_cheap_metric(graph):
        """Cheap computation."""
        call_count['cheap'] += 1
        return len(graph.nodes())
    
    # Create test graph
    G = nx.Graph()
    for i in range(5):
        G.add_node(f"n{i}", epi=0.5 + i * 0.1)
    
    # First calls - should compute
    print("First calls (computing)...")
    start = time.time()
    result1 = compute_expensive_metric(G)
    result2 = compute_cheap_metric(G)
    elapsed1 = time.time() - start
    print(f"  Expensive metric: {result1:.2f}")
    print(f"  Cheap metric: {result2}")
    print(f"  Time: {elapsed1:.3f}s")
    print(f"  Call counts: expensive={call_count['expensive']}, cheap={call_count['cheap']}")
    
    # Second calls - should use cache
    print("\nSecond calls (cached)...")
    start = time.time()
    result1 = compute_expensive_metric(G)
    result2 = compute_cheap_metric(G)
    elapsed2 = time.time() - start
    print(f"  Expensive metric: {result1:.2f}")
    print(f"  Cheap metric: {result2}")
    print(f"  Time: {elapsed2:.3f}s")
    print(f"  Call counts: expensive={call_count['expensive']}, cheap={call_count['cheap']}")
    print(f"  Speedup: {elapsed1/elapsed2:.1f}x")
    
    print()


def example_graph_change_tracking():
    """Demonstrate automatic cache invalidation on graph changes."""
    print("=" * 60)
    print("Example 3: Graph Change Tracking")
    print("=" * 60)
    
    # Create graph and cache
    G = nx.Graph()
    cache = TNFRHierarchicalCache()
    tracker = GraphChangeTracker(cache)
    
    # Add initial nodes
    G.add_nodes_from(['n1', 'n2', 'n3'])
    
    # Cache topology-dependent value
    cache.set(
        'adjacency',
        nx.to_numpy_array(G),
        CacheLevel.GRAPH_STRUCTURE,
        {'graph_topology'}
    )
    
    print("Initial cache state:")
    print(f"  Adjacency cached: {cache.get('adjacency', CacheLevel.GRAPH_STRUCTURE) is not None}")
    
    # Track changes
    tracker.track_graph_changes(G)
    
    # Modify topology
    print("\nAdding edge n1-n2...")
    G.add_edge('n1', 'n2')
    
    print(f"  Topology changes: {tracker.topology_changes}")
    print(f"  Adjacency cached: {cache.get('adjacency', CacheLevel.GRAPH_STRUCTURE) is not None}")
    
    # Cache node property
    cache.set(
        'node_metric_n1',
        0.75,
        CacheLevel.DERIVED_METRICS,
        {'node_epi_n1'}
    )
    
    print("\nCache node metric:")
    print(f"  Node metric cached: {cache.get('node_metric_n1', CacheLevel.DERIVED_METRICS)}")
    
    # Modify node property
    print("\nUpdating node property...")
    tracker.on_node_property_change('n1', 'epi', 0.75, 0.85)
    
    print(f"  Property changes: {tracker.property_changes}")
    print(f"  Node metric cached: {cache.get('node_metric_n1', CacheLevel.DERIVED_METRICS)}")
    
    print()


def example_persistent_cache():
    """Demonstrate persistent caching for expensive operations."""
    print("=" * 60)
    print("Example 4: Persistent Cache")
    print("=" * 60)
    
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        print(f"Cache directory: {cache_dir}")
        
        # Create persistent cache
        cache = PersistentTNFRCache(cache_dir=cache_dir)
        
        # Store expensive computation result
        print("\nStoring expensive result to disk...")
        cache.set_persistent(
            'large_graph_coherence',
            0.9123456789,
            CacheLevel.DERIVED_METRICS,
            {'graph_topology', 'all_node_vf'},
            computation_cost=1000.0,
            persist_to_disk=True
        )
        
        stats = cache.get_stats()
        print(f"  Disk files: {stats['disk_files']}")
        print(f"  Disk size: {stats['disk_size_mb']:.4f} MB")
        
        # Clear memory
        print("\nClearing memory cache...")
        cache._memory_cache.clear()
        
        # Retrieve from disk
        print("Loading from disk...")
        result = cache.get_persistent(
            'large_graph_coherence',
            CacheLevel.DERIVED_METRICS
        )
        print(f"  Result: {result}")
        
        # Cleanup old entries
        print("\nCleanup old cache files (age > 30 days)...")
        removed = cache.cleanup_old_entries(max_age_days=30)
        print(f"  Removed {removed} old files")
    
    print()


def example_cache_eviction():
    """Demonstrate intelligent cache eviction."""
    print("=" * 60)
    print("Example 5: Intelligent Cache Eviction")
    print("=" * 60)
    
    # Small cache to trigger eviction
    cache = TNFRHierarchicalCache(max_memory_mb=1)
    
    print("Adding entries with different costs...")
    
    # Low-cost entry
    cache.set('cheap_1', 'x' * 1000, CacheLevel.TEMPORARY, set(), computation_cost=1.0)
    
    # High-cost entry
    cache.set('expensive_1', 'x' * 1000, CacheLevel.TEMPORARY, set(), computation_cost=100.0)
    
    # Access expensive entry multiple times
    for _ in range(10):
        cache.get('expensive_1', CacheLevel.TEMPORARY)
    
    print(f"  Cheap entry access count: {cache._caches[CacheLevel.TEMPORARY]['cheap_1'].access_count}")
    print(f"  Expensive entry access count: {cache._caches[CacheLevel.TEMPORARY]['expensive_1'].access_count}")
    
    # Add many more entries to trigger eviction
    print("\nAdding more entries to trigger eviction...")
    for i in range(200):
        cache.set(f'filler_{i}', 'x' * 10000, CacheLevel.TEMPORARY, set())
    
    stats = cache.get_stats()
    print(f"  Evictions: {stats['evictions']}")
    print(f"  Memory used: {stats['memory_used_mb']:.2f} / {stats['memory_limit_mb']:.2f} MB")
    
    # Check which entries survived
    print("\nSurviving entries:")
    print(f"  Cheap entry: {cache.get('cheap_1', CacheLevel.TEMPORARY) is not None}")
    print(f"  Expensive entry: {cache.get('expensive_1', CacheLevel.TEMPORARY) is not None}")
    print("  (High-cost, frequently accessed entries prioritized for retention)")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 60)
    print("TNFR Intelligent Caching System - Examples")
    print("=" * 60)
    print()
    
    example_basic_caching()
    example_decorator_caching()
    example_graph_change_tracking()
    example_persistent_cache()
    example_cache_eviction()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
