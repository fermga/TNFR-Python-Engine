"""
Comprehensive demonstration of the TNFR Cache-Aware FFT Engine ecosystem.

This script demonstrates the advanced cache optimizations and FFT arithmetic
that emerge naturally from the nodal equation ∂EPI/∂t = νf · ΔNFR(t).

Features demonstrated:
1. Cache-aware spectral convolution with basis reuse
2. Harmonic analysis with pattern compression
3. Spectral filtering with kernel caching
4. Multi-scale decomposition with hierarchical caching
5. Cross-spectral coherence with predictive prefetching
6. Performance analysis showing cache benefits

The demonstration shows how mathematical structure enables cache optimizations
that provide significant performance improvements while preserving TNFR physics.
"""

import numpy as np
import networkx as nx

# Import TNFR engines
try:
    from tnfr.dynamics.cache_aware_fft_engine import TNFRCacheAwareFFTEngine
    HAS_CACHE_FFT = True
except ImportError:
    HAS_CACHE_FFT = False


def create_test_graph(graph_type: str = "small_world", size: int = 20) -> nx.Graph:
    """Create test graph with TNFR node properties."""
    if graph_type == "small_world":
        G = nx.watts_strogatz_graph(size, 4, 0.3, seed=42)
    elif graph_type == "scale_free":
        G = nx.barabasi_albert_graph(size, 3, seed=42)
    elif graph_type == "regular":
        G = nx.random_regular_graph(6, size, seed=42)
    else:
        G = nx.erdos_renyi_graph(size, 0.3, seed=42)
    
    # Add TNFR properties
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['EPI'] = float(np.sin(i * 0.1) + 0.5)
        G.nodes[node]['nu_f'] = 1.0 + 0.1 * np.cos(i * 0.2)
        G.nodes[node]['phase'] = (i * 2 * np.pi / size) % (2 * np.pi)
        G.nodes[node]['ΔNFR'] = 0.1 * np.random.randn()
    
    return G


def demonstrate_spectral_convolution_caching():
    """Demonstrate cache benefits in spectral convolution."""
    print("\n=== Cache-Aware Spectral Convolution Demo ===")
    
    if not HAS_CACHE_FFT:
        print("Cache-aware FFT engine not available")
        return
    
    # Create test graphs
    graphs = [
        create_test_graph("small_world", 15),
        create_test_graph("small_world", 15),  # Same structure - should hit cache
        create_test_graph("scale_free", 15),
        create_test_graph("regular", 15)
    ]
    
    # Initialize engine
    engine = TNFRCacheAwareFFTEngine(enable_cache_optimization=True)
    
    results = []
    
    for i, G in enumerate(graphs):
        print(f"\n--- Processing Graph {i + 1} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) ---")
        
        # Perform spectral convolution
        result = engine.spectral_convolution_cached(
            G,
            signal1=None,  # Will use EPI values
            signal2=None,  # Will use nu_f values
            operation="multiply"
        )
        
        results.append(result)
        
        print(f"  Operation: {result.operation_type.value}")
        print(f"  Cache hits: {result.cache_hits}")
        print(f"  Cache misses: {result.cache_misses}")
        print(f"  Time saved: {result.optimization_time_saved:.4f}s")
        print(f"  Total time: {result.total_execution_time:.4f}s")
        print(f"  Spectral basis reused: {result.spectral_basis_reused}")
        print(f"  Kernel cache hits: {result.kernel_cache_hits}")
    
    # Show cumulative benefits
    total_cache_hits = sum(r.cache_hits for r in results)
    total_time_saved = sum(r.optimization_time_saved for r in results)
    
    print(f"\n=== Cumulative Cache Benefits ===")
    print(f"Total cache hits: {total_cache_hits}")
    print(f"Total time saved: {total_time_saved:.4f}s")
    print(f"Average cache hit rate: {total_cache_hits / len(results):.2f}")


def demonstrate_harmonic_analysis_optimization():
    """Demonstrate harmonic analysis with cache optimization."""
    print("\n=== Cache-Optimized Harmonic Analysis Demo ===")
    
    if not HAS_CACHE_FFT:
        print("Cache-aware FFT engine not available")
        return
    
    # Create test graph with harmonic structure
    G = create_test_graph("small_world", 24)
    
    # Add harmonic EPI patterns
    for i, node in enumerate(G.nodes()):
        # Superposition of harmonics
        fundamental = np.sin(2 * np.pi * i / 24)
        second_harmonic = 0.5 * np.sin(4 * np.pi * i / 24)
        third_harmonic = 0.25 * np.sin(6 * np.pi * i / 24)
        G.nodes[node]['EPI'] = fundamental + second_harmonic + third_harmonic + 1.0
    
    engine = TNFRCacheAwareFFTEngine(enable_cache_optimization=True)
    
    # Perform multiple harmonic analyses with different parameters
    harmonic_counts = [3, 5, 7, 5, 3]  # 5 and 3 repeated to show caching
    
    for i, num_harmonics in enumerate(harmonic_counts):
        print(f"\n--- Harmonic Analysis {i + 1} (harmonics={num_harmonics}) ---")
        
        result = engine.harmonic_analysis_cached(G, num_harmonics=num_harmonics)
        
        print(f"  Cache hits: {result.cache_hits}")
        print(f"  Time saved: {result.optimization_time_saved:.4f}s")
        print(f"  Total time: {result.total_execution_time:.4f}s")
        
        # Show harmonic analysis results
        harmonic_data = result.fft_result.output_data
        print(f"  Fundamental frequency: {harmonic_data['fundamental_frequency']:.4f}")
        print(f"  Total harmonic distortion: {harmonic_data['total_harmonic_distortion']:.4f}")
        print(f"  Dominant mode: {harmonic_data['dominant_mode_index']}")


def demonstrate_spectral_filtering_kernels():
    """Demonstrate spectral filtering with kernel caching."""
    print("\n=== Spectral Filtering with Kernel Caching Demo ===")
    
    if not HAS_CACHE_FFT:
        print("Cache-aware FFT engine not available")
        return
    
    # Create noisy graph signal
    G = create_test_graph("regular", 20)
    
    # Add noise to EPI values
    for node in G.nodes():
        noise = 0.2 * np.random.randn()
        G.nodes[node]['EPI'] = G.nodes[node]['EPI'] + noise
    
    engine = TNFRCacheAwareFFTEngine(enable_cache_optimization=True)
    
    # Apply different filters (some repeated to show kernel caching)
    filters = [
        ("lowpass", 0.3, 4),
        ("highpass", 0.1, 4),
        ("bandpass", 0.2, 4),
        ("lowpass", 0.3, 4),  # Repeated - should hit kernel cache
        ("notch", 0.25, 6),
        ("bandpass", 0.2, 4),  # Repeated - should hit kernel cache
    ]
    
    for i, (filter_type, cutoff, order) in enumerate(filters):
        print(f"\n--- Filter {i + 1}: {filter_type} (cutoff={cutoff}, order={order}) ---")
        
        result = engine.spectral_filtering_cached(
            G,
            filter_type=filter_type,
            cutoff_frequency=cutoff,
            filter_order=order
        )
        
        print(f"  Cache hits: {result.cache_hits}")
        print(f"  Kernel cache hits: {result.kernel_cache_hits}")
        print(f"  Time saved: {result.optimization_time_saved:.4f}s")
        print(f"  Total time: {result.total_execution_time:.4f}s")
        
        # Show filtering results
        filtered_data = result.fft_result.output_data
        print(f"  Attenuation: {filtered_data['attenuation_db']:.2f} dB")
        print(f"  Effective cutoff: {filtered_data['cutoff_frequency']:.3f}")


def demonstrate_multi_scale_analysis():
    """Demonstrate multi-scale analysis with hierarchical caching."""
    print("\n=== Multi-Scale Analysis with Hierarchical Caching Demo ===")
    
    if not HAS_CACHE_FFT:
        print("Cache-aware FFT engine not available")
        return
    
    # Create graph with multi-scale structure
    G = create_test_graph("small_world", 16)
    
    # Create multi-scale EPI pattern
    for i, node in enumerate(G.nodes()):
        # Different scales: fast oscillation + slow trend
        fast = 0.3 * np.sin(8 * np.pi * i / 16)
        slow = 0.7 * np.sin(np.pi * i / 16) 
        trend = 0.1 * i / 16
        G.nodes[node]['EPI'] = fast + slow + trend + 0.5
    
    engine = TNFRCacheAwareFFTEngine(enable_cache_optimization=True)
    
    # Perform multi-scale analysis
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    print(f"\n--- Multi-Scale Analysis (scales={scales}) ---")
    
    result = engine.multi_scale_analysis_cached(G, scales=scales, analysis_type="spectral")
    
    print(f"  Cache hits: {result.cache_hits}")
    print(f"  Cache misses: {result.cache_misses}")  
    print(f"  Time saved: {result.optimization_time_saved:.4f}s")
    print(f"  Total time: {result.total_execution_time:.4f}s")
    print(f"  Spectral basis reused: {result.spectral_basis_reused}")
    
    # Show multi-scale results
    multi_scale_data = result.fft_result
    print(f"\n  Scale Analysis Results:")
    for scale in scales:
        scale_result = multi_scale_data["scale_results"][scale]
        print(f"    Scale {scale}: energy={scale_result['energy']:.3f}, "
              f"dominant_freq={scale_result['dominant_freq']:.3f}")


def demonstrate_cross_spectral_coherence():
    """Demonstrate cross-spectral coherence analysis."""
    print("\n=== Cross-Spectral Coherence Analysis Demo ===")
    
    if not HAS_CACHE_FFT:
        print("Cache-aware FFT engine not available")
        return
    
    # Create two related graphs
    G1 = create_test_graph("small_world", 18)
    G2 = create_test_graph("small_world", 18)  # Similar structure
    
    # Make G2 partially correlated with G1
    correlation_strength = 0.7
    for node in G2.nodes():
        original_epi = G2.nodes[node]['EPI']
        correlated_epi = G1.nodes[node]['EPI']
        G2.nodes[node]['EPI'] = (
            correlation_strength * correlated_epi + 
            (1 - correlation_strength) * original_epi
        )
    
    engine = TNFRCacheAwareFFTEngine(enable_cache_optimization=True)
    
    # Define frequency bands for coherence analysis
    coherence_bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    print(f"\n--- Cross-Spectral Coherence Analysis ---")
    print(f"  Graph 1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"  Graph 2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    print(f"  Expected correlation: {correlation_strength:.1f}")
    
    result = engine.cross_spectral_coherence_cached(
        G1, G2, coherence_bands=coherence_bands
    )
    
    print(f"\n  Cache Performance:")
    print(f"    Cache hits: {result.cache_hits}")
    print(f"    Time saved: {result.optimization_time_saved:.4f}s")
    print(f"    Total time: {result.total_execution_time:.4f}s")
    print(f"    Spectral basis reused: {result.spectral_basis_reused}")
    
    # Show coherence results
    coherence_data = result.fft_result
    print(f"\n  Coherence Analysis Results:")
    print(f"    Mean coherence: {coherence_data['mean_coherence']:.3f}")
    print(f"    Frequency Band Coherences:")
    
    for band, band_result in coherence_data["coherence_results"].items():
        print(f"      {band[0]:.1f}-{band[1]:.1f}: {band_result['coherence']:.3f}")


def demonstrate_performance_summary():
    """Show comprehensive performance summary."""
    print("\n=== Comprehensive Performance Summary ===")
    
    if not HAS_CACHE_FFT:
        print("Cache-aware FFT engine not available")
        return
    
    # Create engine and run several operations
    engine = TNFRCacheAwareFFTEngine(enable_cache_optimization=True)
    
    # Test graphs
    graphs = [
        create_test_graph("small_world", 12),
        create_test_graph("scale_free", 12),
        create_test_graph("small_world", 12),  # Repeat for caching
    ]
    
    print(f"\n--- Running {len(graphs)} test operations ---")
    
    # Run various operations
    for i, G in enumerate(graphs):
        # Spectral convolution
        engine.spectral_convolution_cached(G, operation="multiply")
        
        # Harmonic analysis
        engine.harmonic_analysis_cached(G, num_harmonics=4)
        
        # Spectral filtering
        engine.spectral_filtering_cached(G, filter_type="lowpass", cutoff_frequency=0.3)
    
    # Get performance summary
    performance = engine.get_performance_summary()
    
    print(f"\n--- Performance Summary ---")
    
    # Cache-aware FFT engine stats
    fft_stats = performance["cache_aware_fft_engine"]
    print(f"  Cache-Aware FFT Engine:")
    print(f"    Total operations: {fft_stats['total_operations']}")
    print(f"    Total cache hits: {fft_stats['total_cache_hits']}")
    print(f"    Total time saved: {fft_stats['total_time_saved']:.4f}s")
    print(f"    Cache hit rate: {fft_stats['cache_hit_rate']:.2%}")
    
    # FFT cache coordinator stats
    if "fft_cache_coordinator" in performance:
        coord_stats = performance["fft_cache_coordinator"]
        print(f"\n  FFT Cache Coordinator:")
        print(f"    Spectral requests: {coord_stats.get('spectral_requests', 0)}")
        print(f"    Spectral hits: {coord_stats.get('spectral_hits', 0)}")
        print(f"    Kernel hits: {coord_stats.get('kernel_hits', 0)}")
        print(f"    Registered results: {coord_stats.get('registered_results', 0)}")
    
    # Cache optimizer stats
    if "cache_optimizer" in performance:
        opt_stats = performance["cache_optimizer"]
        print(f"\n  Cache Optimizer:")
        print(f"    Total optimizations: {opt_stats.get('total_optimizations', 0)}")
        print(f"    Total time saved: {opt_stats.get('total_time_saved', 0):.4f}s")
        print(f"    Total memory saved: {opt_stats.get('total_memory_saved_mb', 0):.2f} MB")
        print(f"    Patterns tracked: {opt_stats.get('patterns_tracked', 0)}")
    
    # Overall efficiency
    overall = performance["overall_efficiency"]
    print(f"\n  Overall Efficiency:")
    print(f"    Avg time saved per operation: {overall['average_time_saved_per_op']:.4f}s")
    print(f"    Cache effectiveness: {overall['cache_effectiveness']:.2%}")


def main():
    """Run comprehensive cache-aware FFT engine demonstration."""
    print("="*70)
    print("TNFR Cache-Aware FFT Engine Ecosystem Demonstration")
    print("="*70)
    print(f"Demonstrates cache optimizations emerging from ∂EPI/∂t = νf · ΔNFR(t)")
    
    if not HAS_CACHE_FFT:
        print("\nERROR: Cache-aware FFT engine not available")
        print("Please ensure all TNFR dynamics modules are installed")
        return
    
    # Run demonstrations
    demonstrate_spectral_convolution_caching()
    demonstrate_harmonic_analysis_optimization()
    demonstrate_spectral_filtering_kernels() 
    demonstrate_multi_scale_analysis()
    demonstrate_cross_spectral_coherence()
    demonstrate_performance_summary()
    
    print("\n" + "="*70)
    print("Cache-Aware FFT Engine Demonstration Complete!")
    print("="*70)
    print("\nKey Benefits Demonstrated:")
    print("• Spectral basis reuse across similar topologies")
    print("• FFT kernel caching for repeated filter operations") 
    print("• Cross-engine cache sharing via unified coordinator")
    print("• Predictive prefetching based on computation patterns")
    print("• Mathematical importance-based cache optimization")
    print("• Significant performance improvements while preserving TNFR physics")


if __name__ == "__main__":
    main()