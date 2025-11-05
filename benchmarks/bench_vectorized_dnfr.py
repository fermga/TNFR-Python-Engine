"""Benchmark vectorized ΔNFR computation performance.

This script compares the performance of the standard NumPy backend vs the
optimized vectorized backend across different graph sizes and densities.
"""

import time
import networkx as nx
from tnfr.backends import get_backend


def benchmark_dnfr(n_nodes, density, n_iterations=10, seed=42):
    """Benchmark ΔNFR computation for a given graph size and density.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph
    density : float
        Edge probability (0.0 to 1.0)
    n_iterations : int
        Number of iterations to average
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Benchmark results with timing for both backends
    """
    # Create graph
    G = nx.erdos_renyi_graph(n_nodes, density, seed=seed)
    
    # Initialize node attributes
    for node in G.nodes():
        G.nodes[node]["phase"] = float(node) * 0.01
        G.nodes[node]["nu_f"] = 1.0 + float(node) * 0.001
        G.nodes[node]["EPI"] = 0.5 + float(node) * 0.0001
    
    # Set weights
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    
    # Warm up
    backend_std = get_backend("numpy")
    backend_std.compute_delta_nfr(G)
    
    backend_opt = get_backend("optimized_numpy")
    for node in G.nodes():
        G.nodes[node]["ΔNFR"] = 0.0
    backend_opt.compute_delta_nfr(G)
    
    # Benchmark standard backend
    times_std = []
    for _ in range(n_iterations):
        for node in G.nodes():
            G.nodes[node]["ΔNFR"] = 0.0
        
        start = time.perf_counter()
        backend_std.compute_delta_nfr(G)
        elapsed = time.perf_counter() - start
        times_std.append(elapsed)
    
    # Benchmark optimized backend
    times_opt = []
    for _ in range(n_iterations):
        for node in G.nodes():
            G.nodes[node]["ΔNFR"] = 0.0
        
        start = time.perf_counter()
        backend_opt.compute_delta_nfr(G)
        elapsed = time.perf_counter() - start
        times_opt.append(elapsed)
    
    avg_std = sum(times_std) / len(times_std)
    avg_opt = sum(times_opt) / len(times_opt)
    speedup = avg_std / avg_opt if avg_opt > 0 else 0
    
    return {
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "density": density,
        "time_std": avg_std,
        "time_opt": avg_opt,
        "speedup": speedup,
    }


def main():
    """Run benchmark suite."""
    print("=" * 80)
    print("TNFR Vectorized ΔNFR Computation Benchmark")
    print("=" * 80)
    print()
    
    # Test different graph sizes
    configs = [
        # (n_nodes, density, iterations)
        (50, 0.2, 20),
        (100, 0.2, 20),
        (200, 0.2, 10),
        (500, 0.2, 10),
        (1000, 0.2, 5),
        (2000, 0.1, 3),
    ]
    
    print(f"{'Nodes':>6} {'Edges':>7} {'Density':>8} {'Std (ms)':>10} {'Opt (ms)':>10} {'Speedup':>8}")
    print("-" * 80)
    
    results = []
    for n_nodes, density, n_iter in configs:
        result = benchmark_dnfr(n_nodes, density, n_iterations=n_iter)
        results.append(result)
        
        print(
            f"{result['n_nodes']:6d} "
            f"{result['n_edges']:7d} "
            f"{result['density']:8.2f} "
            f"{result['time_std']*1000:10.2f} "
            f"{result['time_opt']*1000:10.2f} "
            f"{result['speedup']:8.2f}x"
        )
    
    print()
    print("=" * 80)
    print("Summary:")
    print(f"  - Small graphs (<100 nodes) use standard implementation")
    print(f"  - Large graphs (≥100 nodes) use vectorized fused path")
    print(f"  - Average speedup for large graphs: "
          f"{sum(r['speedup'] for r in results[1:]) / len(results[1:]):.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
