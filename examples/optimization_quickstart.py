"""Quick start guide for TNFR optimization backends.

This example demonstrates how to leverage TNFR's optimization infrastructure
for high-performance structural network simulations.

The TNFR engine provides multiple computational backends that maintain semantic
fidelity while offering significant performance improvements through vectorization,
JIT compilation, and GPU acceleration.
"""

import networkx as nx
import numpy as np

from tnfr.backends import get_backend, set_backend, available_backends
from tnfr.initialization import init_node_attrs


def example_1_basic_backend_usage():
    """Example 1: Basic backend selection and usage."""
    print("=" * 70)
    print("Example 1: Basic Backend Usage")
    print("=" * 70)
    
    # Create a test network
    G = nx.erdos_renyi_graph(500, 0.1, seed=42)
    init_node_attrs(G)
    
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Available backends: {list(available_backends().keys())}")
    
    # Method 1: Get backend explicitly
    print("\n--- Method 1: Explicit backend ---")
    backend = get_backend("numpy")
    print(f"Using backend: {backend.name}")
    
    import time
    start = time.perf_counter()
    backend.compute_delta_nfr(G)
    dnfr_time = time.perf_counter() - start
    
    print(f"ΔNFR computation: {dnfr_time:.3f}s")
    
    # Verify results
    dnfr_values = [G.nodes[n].get("delta_nfr", 0) for n in G.nodes()]
    print(f"ΔNFR range: [{min(dnfr_values):.3f}, {max(dnfr_values):.3f}]")
    print(f"ΔNFR mean: {np.mean(dnfr_values):.3f}")
    
    # Method 2: Set default backend
    print("\n--- Method 2: Set default backend ---")
    set_backend("numpy")
    
    # Now all TNFR operations use the selected backend
    from tnfr.dynamics.dnfr import default_compute_delta_nfr
    
    G2 = G.copy()
    start = time.perf_counter()
    default_compute_delta_nfr(G2)
    dnfr_time2 = time.perf_counter() - start
    
    print(f"ΔNFR computation (via default): {dnfr_time2:.3f}s")


def example_2_profiling():
    """Example 2: Performance profiling and optimization."""
    print("\n" + "=" * 70)
    print("Example 2: Performance Profiling")
    print("=" * 70)
    
    G = nx.barabasi_albert_graph(1000, 5, seed=42)
    init_node_attrs(G)
    
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    backend = get_backend("numpy")
    
    # Enable detailed profiling
    profile = {}
    backend.compute_delta_nfr(G, profile=profile)
    
    print("\nDetailed timing breakdown:")
    print(f"  {'Stage':<30} {'Time (s)':<10} {'Percentage':<10}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10}")
    
    total_time = sum(v for k, v in profile.items() if k != "dnfr_path")
    for stage, duration in sorted(profile.items()):
        if stage != "dnfr_path":
            pct = (duration / total_time * 100) if total_time > 0 else 0
            print(f"  {stage:<30} {duration:<10.3f} {pct:<10.1f}%")
    
    print(f"\n  Execution path: {profile.get('dnfr_path', 'unknown')}")


def example_3_backend_comparison():
    """Example 3: Compare performance across backends."""
    print("\n" + "=" * 70)
    print("Example 3: Backend Comparison")
    print("=" * 70)
    
    G = nx.erdos_renyi_graph(1000, 0.1, seed=42)
    init_node_attrs(G)
    
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Density: {nx.density(G):.3f}\n")
    
    backends_to_test = ["numpy"]
    
    # Try to add optional backends
    all_backends = list(available_backends().keys())
    for optional in ["optimized", "optimized_numpy", "jax", "torch"]:
        if optional in all_backends and optional not in backends_to_test:
            backends_to_test.append(optional)
    
    print(f"Testing {len(backends_to_test)} backend(s): {', '.join(backends_to_test)}\n")
    
    results = {}
    
    for backend_name in backends_to_test:
        try:
            backend = get_backend(backend_name)
            print(f"Testing {backend_name} backend...")
            
            # Make a fresh copy for each backend
            G_test = G.copy()
            
            import time
            
            # Warmup run
            backend.compute_delta_nfr(G_test)
            
            # Timed runs
            times = []
            for _ in range(3):
                G_test = G.copy()
                start = time.perf_counter()
                backend.compute_delta_nfr(G_test)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            results[backend_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "info": {
                    "supports_gpu": backend.supports_gpu,
                    "supports_jit": backend.supports_jit,
                }
            }
            
            print(f"  Mean: {results[backend_name]['mean']:.3f}s")
            print(f"  Std:  {results[backend_name]['std']:.3f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[backend_name] = {"error": str(e)}
    
    # Print comparison
    print("\nPerformance Summary:")
    print(f"  {'Backend':<20} {'Time (s)':<12} {'Speedup':<10} {'Features'}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 20}")
    
    # Find baseline (slowest) time
    valid_times = [r["mean"] for r in results.values() if "mean" in r]
    if valid_times:
        baseline = max(valid_times)
        
        for backend_name in sorted(results.keys()):
            result = results[backend_name]
            if "mean" in result:
                speedup = baseline / result["mean"]
                features = []
                if result["info"]["supports_gpu"]:
                    features.append("GPU")
                if result["info"]["supports_jit"]:
                    features.append("JIT")
                features_str = ", ".join(features) if features else "CPU"
                
                print(f"  {backend_name:<20} {result['mean']:<12.3f} "
                      f"{speedup:<10.1f}x {features_str}")


def example_4_optimization_tuning():
    """Example 4: Fine-tuning optimization parameters."""
    print("\n" + "=" * 70)
    print("Example 4: Optimization Parameter Tuning")
    print("=" * 70)
    
    # Create a dense graph
    G = nx.erdos_renyi_graph(500, 0.3, seed=42)
    init_node_attrs(G)
    
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Density: {nx.density(G):.3f} (dense graph)")
    
    backend = get_backend("numpy")
    
    # Test 1: Default (auto-select strategy)
    print("\n--- Test 1: Default configuration ---")
    G1 = G.copy()
    profile1 = {}
    
    import time
    start = time.perf_counter()
    backend.compute_delta_nfr(G1, profile=profile1)
    time1 = time.perf_counter() - start
    
    print(f"Time: {time1:.3f}s")
    print(f"Path: {profile1.get('dnfr_path', 'unknown')}")
    
    # Test 2: Force dense strategy
    print("\n--- Test 2: Force dense matrix strategy ---")
    G2 = G.copy()
    G2.graph["dnfr_force_dense"] = True  # Force dense accumulation
    profile2 = {}
    
    start = time.perf_counter()
    backend.compute_delta_nfr(G2, profile=profile2)
    time2 = time.perf_counter() - start
    
    print(f"Time: {time2:.3f}s")
    print(f"Path: {profile2.get('dnfr_path', 'unknown')}")
    print(f"Speedup: {time1/time2:.2f}x")
    
    # Test 3: Chunked processing
    print("\n--- Test 3: Chunked processing ---")
    G3 = G.copy()
    G3.graph["DNFR_CHUNK_SIZE"] = 100  # Process in chunks of 100
    profile3 = {}
    
    start = time.perf_counter()
    backend.compute_delta_nfr(G3, profile=profile3)
    time3 = time.perf_counter() - start
    
    print(f"Time: {time3:.3f}s")
    
    # Test 4: Memory-constrained (limited cache)
    print("\n--- Test 4: Limited cache (memory-constrained) ---")
    G4 = G.copy()
    
    start = time.perf_counter()
    backend.compute_delta_nfr(G4, cache_size=2)  # Limit cache entries
    time4 = time.perf_counter() - start
    
    print(f"Time: {time4:.3f}s")


def example_5_si_computation():
    """Example 5: Optimized Si (Sense Index) computation."""
    print("\n" + "=" * 70)
    print("Example 5: Sense Index Computation")
    print("=" * 70)
    
    G = nx.barabasi_albert_graph(2000, 4, seed=42)
    init_node_attrs(G)
    
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    backend = get_backend("numpy")
    
    # Configure Si weights
    G.graph["SI_WEIGHTS"] = {
        "alpha": 0.4,   # Structural frequency weight
        "beta": 0.3,    # Phase alignment weight
        "gamma": 0.3,   # ΔNFR attenuation weight
    }
    
    import time
    
    # Compute ΔNFR first (required for Si)
    print("\nComputing ΔNFR...")
    backend.compute_delta_nfr(G)
    
    # Compute Si with profiling
    print("Computing Si...")
    profile = {}
    start = time.perf_counter()
    si_result = backend.compute_si(G, inplace=True, profile=profile)
    si_time = time.perf_counter() - start
    
    print(f"\nSi computation: {si_time:.3f}s")
    print(f"Execution path: {profile.get('path', 'unknown')}")
    
    # Analyze Si distribution
    si_values = [G.nodes[n].get("Si", 0) for n in G.nodes()]
    print(f"\nSi statistics:")
    print(f"  Mean: {np.mean(si_values):.3f}")
    print(f"  Std:  {np.std(si_values):.3f}")
    print(f"  Range: [{np.min(si_values):.3f}, {np.max(si_values):.3f}]")
    
    # Identify high/low Si nodes
    si_sorted = sorted([(n, G.nodes[n].get("Si", 0)) for n in G.nodes()],
                      key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 nodes by Si:")
    for i, (node, si_val) in enumerate(si_sorted[:5], 1):
        degree = G.degree(node)
        print(f"  {i}. Node {node}: Si={si_val:.3f}, degree={degree}")


def main():
    """Run all examples."""
    print("\nTNFR Optimization Backend Examples")
    print("=" * 70)
    
    # Run examples
    example_1_basic_backend_usage()
    example_2_profiling()
    example_3_backend_comparison()
    example_4_optimization_tuning()
    example_5_si_computation()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("\nFor more information, see:")
    print("  - docs/OPTIMIZATION_GUIDE.md")
    print("  - examples/backend_performance_comparison.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
