"""
Demonstrate cache telemetry and hit rate monitoring.

This example shows how to:
1. Access cache statistics from the global cache manager
2. Monitor field computation cache hit rates
3. Track edge cache performance
4. Optimize cache capacity based on telemetry

Target: >80% cache hit rate for optimal performance.
"""

import networkx as nx
import numpy as np

from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from tnfr.utils.cache import get_global_cache
from tnfr.validation.aggregator import run_structural_validation


def setup_test_graph(n=500, seed=42):
    """Create a test graph with TNFR attributes."""
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(n, m=3, seed=seed)
    
    for node in G.nodes():
        G.nodes[node]['EPI'] = np.random.randn(10)
        G.nodes[node]['nu_f'] = 1.0 + np.random.rand()
        G.nodes[node]['theta'] = np.random.uniform(0, 2 * np.pi)
        G.nodes[node]['DELTA_NFR'] = np.random.randn()
    
    return G


def demo_field_cache_telemetry():
    """Demonstrate field computation cache monitoring."""
    print("=" * 70)
    print("Field Computation Cache Telemetry Demo")
    print("=" * 70)
    
    G = setup_test_graph(500)
    cache = get_global_cache()
    
    print("\n1. Initial cache state (empty):")
    print(f"   Hits: {cache.hits}, Misses: {cache.misses}")
    print("   Hit rate: N/A (no operations yet)")
    
    print("\n2. First computation (cold cache - all misses):")
    _ = compute_structural_potential(G, alpha=2.0)
    _ = compute_phase_gradient(G)
    _ = compute_phase_curvature(G)
    _ = estimate_coherence_length(G)
    
    total = cache.hits + cache.misses
    hit_rate = (cache.hits / total * 100) if total > 0 else 0.0
    print(f"   Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"   Hit rate: {hit_rate:.1f}%")
    
    print("\n3. Second computation (hot cache - all hits):")
    _ = compute_structural_potential(G, alpha=2.0)
    _ = compute_phase_gradient(G)
    _ = compute_phase_curvature(G)
    _ = estimate_coherence_length(G)
    
    total = cache.hits + cache.misses
    hit_rate = (cache.hits / total * 100) if total > 0 else 0.0
    print(f"   Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"   Hit rate: {hit_rate:.1f}% ✓ (expected: ~50%)")
    
    print("\n4. Repeated computation (10x - testing cache):")
    for _ in range(10):
        _ = compute_structural_potential(G, alpha=2.0)
        _ = compute_phase_gradient(G)
        _ = compute_phase_curvature(G)
        _ = estimate_coherence_length(G)
    
    total = cache.hits + cache.misses
    hit_rate = (cache.hits / total * 100) if total > 0 else 0.0
    print(f"   Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"   Hit rate: {hit_rate:.1f}% ✓")
    print(f"   Evictions: {cache.evictions}")
    print(f"   Invalidations: {cache.invalidations}")
    
    if hit_rate >= 80:
        print(f"\n✅ Target achieved: {hit_rate:.1f}% >= 80%")
    else:
        print(f"\n⚠️  Below target: {hit_rate:.1f}% < 80%")
        print("   Consider increasing cache capacity")


def demo_validation_cache_performance():
    """Demonstrate validation pipeline cache behavior."""
    print("\n\n" + "=" * 70)
    print("Validation Pipeline Cache Performance")
    print("=" * 70)
    
    G = setup_test_graph(500)
    cache = get_global_cache()
    
    # Reset stats
    initial_hits = cache.hits
    initial_misses = cache.misses
    
    print("\n1. First validation (cold cache):")
    sequence = ["AL", "UM", "IL", "SHA"]
    
    _ = run_structural_validation(G, sequence=sequence)
    
    hits = cache.hits - initial_hits
    misses = cache.misses - initial_misses
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0.0
    print(f"   Hits: {hits}, Misses: {misses}")
    print(f"   Hit rate: {hit_rate:.1f}%")
    
    print("\n2. Repeated validation (10x - hot cache):")
    initial_hits = cache.hits
    initial_misses = cache.misses
    
    for i in range(10):
        _ = run_structural_validation(G, sequence=sequence)
    
    hits = cache.hits - initial_hits
    misses = cache.misses - initial_misses
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0.0
    print(f"   Hits: {hits}, Misses: {misses}")
    print(f"   Hit rate: {hit_rate:.1f}%")
    
    if hit_rate >= 80:
        print(f"\n✅ Excellent: {hit_rate:.1f}% hit rate")
        print("   Cache working perfectly for repeated validations")
    else:
        print(f"\n⚠️  Suboptimal: {hit_rate:.1f}% hit rate")


def demo_cache_capacity_tuning():
    """Demonstrate cache capacity optimization."""
    print("\n\n" + "=" * 70)
    print("Cache Capacity Tuning")
    print("=" * 70)
    
    print("\n1. Current cache statistics:")
    cache = get_global_cache()
    print(f"   Total hits: {cache.hits}")
    print(f"   Total misses: {cache.misses}")
    print(f"   Evictions: {cache.evictions}")
    print(f"   Invalidations: {cache.invalidations}")
    
    total = cache.hits + cache.misses
    if total > 0:
        hit_rate = (cache.hits / total) * 100
        print(f"   Overall hit rate: {hit_rate:.1f}%")
    
    print("\n2. Tuning recommendations:")
    print("   - If hit rate < 70%: Increase capacity (512, 1024)")
    print("   - If hit rate > 95%: Capacity may be excessive")
    print("   - Target: 80-90% for optimal memory/performance trade-off")
    
    print("\n3. How to configure:")
    print("   From code:")
    print("   ```python")
    print("   from tnfr.utils.cache import configure_graph_cache_limits")
    print("   config = configure_graph_cache_limits(")
    print("       G,")
    print("       default_capacity=512,  # Increase from 256")
    print("       overrides={'hierarchical_derived_metrics': 1024}")
    print("   )")
    print("   ```")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TNFR Cache Telemetry Demo" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    demo_field_cache_telemetry()
    demo_validation_cache_performance()
    demo_cache_capacity_tuning()
    
    print("\n\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Cache telemetry enables data-driven optimization")
    print("✓ Monitor hit rates to validate performance improvements")
    print("✓ Target: >80% hit rate for production workloads")
    print("✓ Adjust capacity based on observed eviction patterns")
    print("\nFor details, see: docs/OPTIMIZATION_PROGRESS.md")
    print("=" * 70)
