"""Comprehensive performance comparison of TNFR computational backends.

This script benchmarks all available backends across different graph sizes and
topologies, measuring ΔNFR computation, Si computation, and coherence matrix
generation performance.

Run with:
    python examples/backend_performance_comparison.py

For detailed output:
    python examples/backend_performance_comparison.py --verbose

To save results:
    python examples/backend_performance_comparison.py --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Any, Callable

import networkx as nx
import numpy as np

from tnfr.backends import available_backends, get_backend
from tnfr.initialization import init_node_attrs


def create_test_graph(graph_type: str, n: int, **params) -> nx.Graph:
    """Create a test graph of specified type and size.
    
    Parameters
    ----------
    graph_type : str
        One of: "erdos_renyi", "barabasi_albert", "watts_strogatz", "complete"
    n : int
        Number of nodes
    **params
        Additional parameters specific to graph type
        
    Returns
    -------
    nx.Graph
        Initialized TNFR graph
    """
    if graph_type == "erdos_renyi":
        p = params.get("p", 0.1)
        G = nx.erdos_renyi_graph(n, p, seed=42)
    elif graph_type == "barabasi_albert":
        m = params.get("m", 3)
        G = nx.barabasi_albert_graph(n, m, seed=42)
    elif graph_type == "watts_strogatz":
        k = params.get("k", 4)
        p = params.get("p", 0.1)
        G = nx.watts_strogatz_graph(n, k, p, seed=42)
    elif graph_type == "complete":
        G = nx.complete_graph(n)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Initialize with TNFR attributes
    init_node_attrs(G)
    
    return G


def benchmark_operation(
    name: str,
    operation: Callable,
    warmup: int = 1,
    iterations: int = 3,
) -> dict[str, float]:
    """Benchmark a single operation with warmup and multiple iterations.
    
    Parameters
    ----------
    name : str
        Operation name for reporting
    operation : Callable
        Function to benchmark
    warmup : int, optional
        Number of warmup iterations (default: 1)
    iterations : int, optional
        Number of timed iterations (default: 3)
        
    Returns
    -------
    dict[str, float]
        Timing statistics (mean, std, min, max)
    """
    # Warmup
    for _ in range(warmup):
        try:
            operation()
        except Exception as e:
            return {"error": str(e)}
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            operation()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            return {"error": str(e)}
    
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "n": len(times),
    }


def benchmark_backend(
    backend_name: str,
    graph: nx.Graph,
    verbose: bool = False,
) -> dict[str, Any]:
    """Benchmark all operations for a single backend.
    
    Parameters
    ----------
    backend_name : str
        Backend to test
    graph : nx.Graph
        Test graph
    verbose : bool, optional
        Print progress messages
        
    Returns
    -------
    dict[str, Any]
        Benchmark results
    """
    if verbose:
        print(f"  Testing {backend_name} backend...")
    
    try:
        backend = get_backend(backend_name)
    except (ValueError, RuntimeError) as e:
        return {"available": False, "error": str(e)}
    
    results = {
        "available": True,
        "backend_info": {
            "name": backend.name,
            "supports_gpu": backend.supports_gpu,
            "supports_jit": backend.supports_jit,
        },
        "operations": {},
    }
    
    # ΔNFR computation
    G_copy = graph.copy()
    profile = {}
    
    def compute_dnfr():
        backend.compute_delta_nfr(G_copy, profile=profile)
    
    dnfr_stats = benchmark_operation("compute_delta_nfr", compute_dnfr)
    results["operations"]["delta_nfr"] = dnfr_stats
    
    if "error" not in dnfr_stats and profile:
        results["operations"]["delta_nfr"]["profile"] = dict(profile)
    
    # Si computation
    G_copy = graph.copy()
    profile = {}
    
    def compute_si():
        backend.compute_si(G_copy, inplace=True, profile=profile)
    
    si_stats = benchmark_operation("compute_si", compute_si)
    results["operations"]["si"] = si_stats
    
    if "error" not in si_stats and profile:
        results["operations"]["si"]["profile"] = dict(profile)
    
    return results


def run_benchmark_suite(
    graph_configs: list[dict],
    backends: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run comprehensive benchmark suite across multiple configurations.
    
    Parameters
    ----------
    graph_configs : list[dict]
        List of graph configurations, each with keys:
        - name: Configuration name
        - type: Graph type
        - n: Number of nodes
        - params: Graph-specific parameters
    backends : list[str] or None, optional
        Backends to test (None = all available)
    verbose : bool, optional
        Print progress messages
        
    Returns
    -------
    dict[str, Any]
        Complete benchmark results
    """
    if backends is None:
        backends = list(available_backends().keys())
    
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backends_tested": backends,
            "configurations": len(graph_configs),
        },
        "benchmarks": [],
    }
    
    for config in graph_configs:
        if verbose:
            print(f"\nBenchmarking: {config['name']}")
            print(f"  Graph: {config['type']}, n={config['n']}")
        
        # Create graph
        G = create_test_graph(
            config["type"],
            config["n"],
            **config.get("params", {})
        )
        
        graph_info = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        }
        
        benchmark_result = {
            "config": config,
            "graph_info": graph_info,
            "backend_results": {},
        }
        
        # Test each backend
        for backend_name in backends:
            backend_result = benchmark_backend(backend_name, G, verbose=verbose)
            benchmark_result["backend_results"][backend_name] = backend_result
        
        results["benchmarks"].append(benchmark_result)
    
    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print formatted summary of benchmark results.
    
    Parameters
    ----------
    results : dict[str, Any]
        Benchmark results from run_benchmark_suite
    """
    print("\n" + "=" * 80)
    print("TNFR Backend Performance Benchmark Summary")
    print("=" * 80)
    
    for benchmark in results["benchmarks"]:
        config = benchmark["config"]
        graph_info = benchmark["graph_info"]
        
        print(f"\n{config['name']}")
        print(f"  Graph: {graph_info['nodes']} nodes, {graph_info['edges']} edges")
        print(f"  Density: {graph_info['density']:.3f}, Avg degree: {graph_info['avg_degree']:.1f}")
        
        # Collect available backends and their ΔNFR times
        backend_times = {}
        for backend_name, backend_result in benchmark["backend_results"].items():
            if backend_result.get("available"):
                dnfr_result = backend_result["operations"].get("delta_nfr", {})
                if "mean" in dnfr_result:
                    backend_times[backend_name] = dnfr_result["mean"]
        
        if not backend_times:
            print("  No successful benchmark runs")
            continue
        
        # Find baseline (slowest) for speedup calculation
        baseline_time = max(backend_times.values())
        
        print("\n  ΔNFR Computation:")
        print(f"    {'Backend':<20} {'Time (s)':<12} {'Speedup':<10} {'Status'}")
        print(f"    {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 20}")
        
        for backend_name in sorted(backend_times.keys()):
            time_val = backend_times[backend_name]
            speedup = baseline_time / time_val
            backend_result = benchmark["backend_results"][backend_name]
            
            status = []
            if backend_result["backend_info"]["supports_gpu"]:
                status.append("GPU")
            if backend_result["backend_info"]["supports_jit"]:
                status.append("JIT")
            status_str = ", ".join(status) if status else "CPU"
            
            print(f"    {backend_name:<20} {time_val:<12.4f} {speedup:<10.1f}x {status_str}")
        
        # Si computation summary
        si_times = {}
        for backend_name, backend_result in benchmark["backend_results"].items():
            if backend_result.get("available"):
                si_result = backend_result["operations"].get("si", {})
                if "mean" in si_result:
                    si_times[backend_name] = si_result["mean"]
        
        if si_times:
            baseline_si = max(si_times.values())
            
            print("\n  Si Computation:")
            print(f"    {'Backend':<20} {'Time (s)':<12} {'Speedup':<10}")
            print(f"    {'-' * 20} {'-' * 12} {'-' * 10}")
            
            for backend_name in sorted(si_times.keys()):
                time_val = si_times[backend_name]
                speedup = baseline_si / time_val
                print(f"    {backend_name:<20} {time_val:<12.4f} {speedup:<10.1f}x")


def main():
    """Run backend performance benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark TNFR computational backends"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Backends to test (default: all available)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Graph sizes to test (default: 100 500 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    # Define benchmark configurations
    graph_configs = []
    
    for n in args.sizes:
        # Sparse random graph
        graph_configs.append({
            "name": f"Erdős-Rényi (n={n}, p=0.1)",
            "type": "erdos_renyi",
            "n": n,
            "params": {"p": 0.1},
        })
        
        # Scale-free network
        if n >= 10:
            m = min(5, n // 10)
            graph_configs.append({
                "name": f"Barabási-Albert (n={n}, m={m})",
                "type": "barabasi_albert",
                "n": n,
                "params": {"m": m},
            })
        
        # Small-world network
        if n >= 10:
            k = min(6, n // 5)
            graph_configs.append({
                "name": f"Watts-Strogatz (n={n}, k={k}, p=0.1)",
                "type": "watts_strogatz",
                "n": n,
                "params": {"k": k, "p": 0.1},
            })
    
    # Run benchmarks
    print("Starting TNFR backend performance benchmarks...")
    print(f"Testing backends: {args.backends or 'all available'}")
    print(f"Graph sizes: {args.sizes}")
    
    results = run_benchmark_suite(
        graph_configs,
        backends=args.backends,
        verbose=args.verbose,
    )
    
    # Print summary
    print_summary(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
