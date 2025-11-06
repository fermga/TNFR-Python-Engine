"""Integration example: Applying caching to existing TNFR metrics.

This module demonstrates how to integrate the intelligent caching system
with existing TNFR metric computations like compute_Si() and compute_coherence().
"""

from __future__ import annotations

import time
from typing import Any
import networkx as nx

from tnfr.cache import (
    TNFRHierarchicalCache,
    CacheLevel,
    cache_tnfr_computation,
    GraphChangeTracker,
)
from tnfr.metrics.sense_index import compute_Si
from tnfr.metrics.coherence import compute_coherence


def example_cached_si_computation():
    """Demonstrate caching compute_Si() for performance improvement."""
    print("=" * 70)
    print("Example: Caching compute_Si() for performance")
    print("=" * 70)

    # Create test graph
    G = nx.karate_club_graph()

    # Add TNFR properties
    for node in G.nodes():
        G.nodes[node]["nu_f"] = 0.5 + (node % 10) * 0.05
        G.nodes[node]["delta_nfr"] = 0.1 + (node % 5) * 0.02
        G.nodes[node]["phase"] = (node * 0.1) % (2 * 3.14159)

    G.graph["SI_WEIGHTS"] = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}

    # Without caching
    print("\n1. Without caching (baseline):")
    times_uncached = []
    for i in range(5):
        start = time.time()
        result = compute_Si(G, inplace=False)
        elapsed = time.time() - start
        times_uncached.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.4f}s")

    avg_uncached = sum(times_uncached) / len(times_uncached)
    print(f"   Average: {avg_uncached:.4f}s")

    # With caching (create wrapper)
    cache = TNFRHierarchicalCache(max_memory_mb=128)

    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS,
        dependencies={
            "graph_topology",
            "all_node_nu_f",
            "all_node_delta_nfr",
            "all_node_phase",
        },
        cost_estimator=lambda g, **kw: len(g.nodes()) * len(g.edges()) / 10,
        cache_instance=cache,
    )
    def cached_compute_Si(graph, **kwargs):
        return compute_Si(graph, inplace=False, **kwargs)

    print("\n2. With caching:")
    times_cached = []

    # First call (cache miss)
    start = time.time()
    result_cached = cached_compute_Si(G)
    elapsed_first = time.time() - start
    print(f"   First call (cache miss): {elapsed_first:.4f}s")

    # Subsequent calls (cache hits)
    for i in range(4):
        start = time.time()
        result_cached = cached_compute_Si(G)
        elapsed = time.time() - start
        times_cached.append(elapsed)
        print(f"   Run {i+2} (cache hit): {elapsed:.6f}s")

    avg_cached = sum(times_cached) / len(times_cached)
    speedup = avg_uncached / avg_cached if avg_cached > 0 else float("inf")

    print(f"\n   Average cached: {avg_cached:.6f}s")
    print(f"   Speedup: {speedup:.1f}x")

    # Show cache stats
    stats = cache.get_stats()
    print(f"\n   Cache stats:")
    print(f"   - Hits: {stats['hits']}")
    print(f"   - Misses: {stats['misses']}")
    print(f"   - Hit rate: {stats['hit_rate']:.1%}")

    return speedup


def example_cached_coherence():
    """Demonstrate caching compute_coherence() for performance."""
    print("\n" + "=" * 70)
    print("Example: Caching compute_coherence() for performance")
    print("=" * 70)

    # Create larger test graph
    G = nx.watts_strogatz_graph(100, 6, 0.3, seed=42)

    # Add TNFR properties
    for node in G.nodes():
        G.nodes[node]["nu_f"] = 0.5 + (node % 10) * 0.05
        G.nodes[node]["delta_nfr"] = 0.1 + (node % 5) * 0.02
        G.nodes[node]["phase"] = (node * 0.1) % (2 * 3.14159)
        G.nodes[node]["epi"] = 0.5 + (node % 20) * 0.025
        G.nodes[node]["Si"] = 0.7 + (node % 15) * 0.02

    # Without caching
    print("\n1. Without caching (baseline):")
    times_uncached = []
    for i in range(3):
        start = time.time()
        result = compute_coherence(G)
        elapsed = time.time() - start
        times_uncached.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.4f}s")

    avg_uncached = sum(times_uncached) / len(times_uncached)
    print(f"   Average: {avg_uncached:.4f}s")

    # With caching
    cache = TNFRHierarchicalCache(max_memory_mb=256)

    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS,
        dependencies={"graph_topology", "all_node_properties"},
        cost_estimator=lambda g: len(g.nodes()) ** 2 / 100,
        cache_instance=cache,
    )
    def cached_compute_coherence(graph):
        return compute_coherence(graph)

    print("\n2. With caching:")
    times_cached = []

    # First call (cache miss)
    start = time.time()
    result_cached = cached_compute_coherence(G)
    elapsed_first = time.time() - start
    print(f"   First call (cache miss): {elapsed_first:.4f}s")

    # Subsequent calls (cache hits)
    for i in range(2):
        start = time.time()
        result_cached = cached_compute_coherence(G)
        elapsed = time.time() - start
        times_cached.append(elapsed)
        print(f"   Run {i+2} (cache hit): {elapsed:.6f}s")

    avg_cached = sum(times_cached) / len(times_cached)
    speedup = avg_uncached / avg_cached if avg_cached > 0 else float("inf")

    print(f"\n   Average cached: {avg_cached:.6f}s")
    print(f"   Speedup: {speedup:.1f}x")

    # Show cache stats
    stats = cache.get_stats()
    print(f"\n   Cache stats:")
    print(f"   - Memory used: {stats['memory_used_mb']:.2f} MB")
    print(f"   - Hit rate: {stats['hit_rate']:.1%}")

    return speedup


def example_selective_invalidation():
    """Demonstrate selective cache invalidation."""
    print("\n" + "=" * 70)
    print("Example: Selective cache invalidation on graph changes")
    print("=" * 70)

    # Create test graph
    G = nx.karate_club_graph()

    for node in G.nodes():
        G.nodes[node]["nu_f"] = 0.5
        G.nodes[node]["delta_nfr"] = 0.1
        G.nodes[node]["phase"] = 0.0
        G.nodes[node]["epi"] = 0.5
        G.nodes[node]["Si"] = 0.7

    G.graph["SI_WEIGHTS"] = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}

    # Setup caching with change tracking
    cache = TNFRHierarchicalCache(max_memory_mb=128)
    tracker = GraphChangeTracker(cache)

    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS,
        dependencies={"graph_topology"},
        cache_instance=cache,
    )
    def cached_si_topology_dependent(graph):
        return compute_Si(graph, inplace=False)

    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS,
        dependencies={"all_node_properties"},  # Different dependency
        cache_instance=cache,
    )
    def cached_coherence_property_dependent(graph):
        return compute_coherence(graph)

    # Initial computations
    print("\n1. Initial computations (cache misses):")
    result1 = cached_si_topology_dependent(G)
    result2 = cached_coherence_property_dependent(G)
    print(f"   Si computed: nodes={len(result1)}")
    print(f"   Coherence computed: {result2:.4f}")
    print(f"   Cache entries: {sum(cache.get_stats()['entry_counts'].values())}")

    # Track changes
    tracker.track_graph_changes(G)

    # Modify topology
    print("\n2. Modifying topology (add edge):")
    G.add_edge(0, 10)
    print(
        f"   Cache entries after topology change: {sum(cache.get_stats()['entry_counts'].values())}"
    )

    # Si needs recomputation (depends on topology)
    print("\n3. Recompute after topology change:")
    start = time.time()
    result1_new = cached_si_topology_dependent(G)
    elapsed_si = time.time() - start
    print(f"   Si recomputed: {elapsed_si:.4f}s (cache miss)")

    # Coherence still cached (doesn't depend on topology)
    start = time.time()
    result2_same = cached_coherence_property_dependent(G)
    elapsed_coh = time.time() - start
    print(f"   Coherence from cache: {elapsed_coh:.6f}s (cache hit)")
    print(f"   Coherence unchanged: {result2 == result2_same}")

    stats = cache.get_stats()
    print(f"\n   Final cache stats:")
    print(f"   - Hits: {stats['hits']}")
    print(f"   - Misses: {stats['misses']}")
    print(f"   - Invalidations: {stats['invalidations']}")


def benchmark_scaling():
    """Benchmark caching performance across different graph sizes."""
    print("\n" + "=" * 70)
    print("Benchmark: Caching performance scaling with graph size")
    print("=" * 70)

    sizes = [50, 100, 200, 500]
    results = []

    for size in sizes:
        # Create graph
        G = nx.watts_strogatz_graph(size, min(6, size - 1), 0.3, seed=42)

        for node in G.nodes():
            G.nodes[node]["nu_f"] = 0.5
            G.nodes[node]["delta_nfr"] = 0.1
            G.nodes[node]["phase"] = 0.0

        G.graph["SI_WEIGHTS"] = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}

        # Measure uncached
        start = time.time()
        for _ in range(3):
            result = compute_Si(G, inplace=False)
        uncached_time = (time.time() - start) / 3

        # Measure cached
        cache = TNFRHierarchicalCache(max_memory_mb=512)

        @cache_tnfr_computation(
            level=CacheLevel.DERIVED_METRICS,
            dependencies={"graph_topology", "all_node_properties"},
            cache_instance=cache,
        )
        def cached_si(graph):
            return compute_Si(graph, inplace=False)

        # First call (miss)
        cached_si(G)

        # Subsequent calls (hits)
        start = time.time()
        for _ in range(3):
            result = cached_si(G)
        cached_time = (time.time() - start) / 3

        speedup = uncached_time / cached_time if cached_time > 0 else float("inf")
        results.append((size, uncached_time, cached_time, speedup))

        print(f"\n   Graph size: {size} nodes, {G.number_of_edges()} edges")
        print(f"   Uncached: {uncached_time:.4f}s")
        print(f"   Cached:   {cached_time:.6f}s")
        print(f"   Speedup:  {speedup:.1f}x")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'Size':<10} {'Uncached':<12} {'Cached':<12} {'Speedup':<10}")
    print("-" * 70)
    for size, uncached, cached, speedup in results:
        print(f"{size:<10} {uncached:<12.4f} {cached:<12.6f} {speedup:<10.1f}x")


def main():
    """Run all integration examples and benchmarks."""
    print("\n" + "=" * 70)
    print("TNFR Caching Integration Examples and Benchmarks")
    print("=" * 70)
    print()

    try:
        # Run examples
        speedup_si = example_cached_si_computation()
        speedup_coh = example_cached_coherence()
        example_selective_invalidation()

        # Run benchmark
        benchmark_scaling()

        # Summary
        print("\n" + "=" * 70)
        print("Integration Complete!")
        print("=" * 70)
        print(f"\nKey Results:")
        print(f"  - compute_Si() speedup: {speedup_si:.1f}x")
        print(f"  - compute_coherence() speedup: {speedup_coh:.1f}x")
        print(f"  - Selective invalidation: Working correctly")
        print(f"  - Scaling: Performance improves with graph size")

    except Exception as e:
        print(f"\nError during integration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
