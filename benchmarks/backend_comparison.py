"""Benchmark comparing TNFR backend implementations.

This script benchmarks ΔNFR and Si computation across available backends
(NumPy, JAX, Torch) to demonstrate performance characteristics and validate
that vectorization provides meaningful speedup.

Usage:
    python benchmarks/backend_comparison.py --nodes 50 100 200 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import networkx as nx

from tnfr.backends import available_backends, get_backend
from tnfr.constants import get_aliases

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


@dataclass(slots=True)
class BackendBenchmark:
    """Container for backend benchmark results."""
    
    backend_name: str
    num_nodes: int
    edge_probability: float
    dnfr_times: list[float]
    si_times: list[float]
    
    @property
    def dnfr_median(self) -> float:
        """Median ΔNFR computation time."""
        return statistics.median(self.dnfr_times) if self.dnfr_times else 0.0
    
    @property
    def si_median(self) -> float:
        """Median Si computation time."""
        return statistics.median(self.si_times) if self.si_times else 0.0


def _build_graph(num_nodes: int, edge_probability: float, seed: int) -> nx.Graph:
    """Create a reproducible test graph with TNFR attributes."""
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        graph.nodes[node][ALIAS_THETA] = 0.0
        graph.nodes[node][ALIAS_EPI] = 0.5
        graph.nodes[node][ALIAS_VF] = 1.0
    return graph


def benchmark_backend(
    backend_name: str,
    num_nodes: int,
    edge_probability: float,
    repeats: int,
) -> BackendBenchmark:
    """Benchmark a single backend configuration."""
    
    try:
        backend = get_backend(backend_name)
    except (ValueError, RuntimeError) as exc:
        print(f"  Skipping {backend_name}: {exc}")
        return BackendBenchmark(backend_name, num_nodes, edge_probability, [], [])
    
    dnfr_times = []
    si_times = []
    
    for i in range(repeats):
        # Create fresh graph for each repeat
        G = _build_graph(num_nodes, edge_probability, seed=42 + i)
        
        # Benchmark ΔNFR computation
        start = time.perf_counter()
        backend.compute_delta_nfr(G)
        dnfr_time = time.perf_counter() - start
        dnfr_times.append(dnfr_time)
        
        # Ensure ΔNFR values exist for Si computation
        for node in G.nodes():
            if ALIAS_DNFR not in G.nodes[node]:
                G.nodes[node][ALIAS_DNFR] = 0.0
        
        # Benchmark Si computation
        start = time.perf_counter()
        backend.compute_si(G, inplace=True)
        si_time = time.perf_counter() - start
        si_times.append(si_time)
    
    return BackendBenchmark(
        backend_name,
        num_nodes,
        edge_probability,
        dnfr_times,
        si_times,
    )


def print_results(results: list[BackendBenchmark]) -> None:
    """Print formatted benchmark results."""
    
    if not results:
        print("No results to display.")
        return
    
    # Group by node count
    by_nodes: dict[int, list[BackendBenchmark]] = {}
    for result in results:
        by_nodes.setdefault(result.num_nodes, []).append(result)
    
    print("\n" + "=" * 80)
    print("TNFR Backend Performance Comparison")
    print("=" * 80)
    
    for num_nodes in sorted(by_nodes.keys()):
        group = by_nodes[num_nodes]
        
        print(f"\nNodes: {num_nodes}")
        print("-" * 80)
        print(f"{'Backend':<12} {'ΔNFR (ms)':<15} {'Si (ms)':<15} {'Total (ms)':<15}")
        print("-" * 80)
        
        for result in group:
            if not result.dnfr_times:
                print(f"{result.backend_name:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                continue
            
            dnfr_ms = result.dnfr_median * 1000
            si_ms = result.si_median * 1000
            total_ms = dnfr_ms + si_ms
            
            print(
                f"{result.backend_name:<12} "
                f"{dnfr_ms:>8.3f}        "
                f"{si_ms:>8.3f}        "
                f"{total_ms:>8.3f}"
            )
        
        # Calculate speedups relative to slowest backend
        valid_results = [r for r in group if r.dnfr_times]
        if len(valid_results) > 1:
            slowest = max(valid_results, key=lambda r: r.dnfr_median + r.si_median)
            print(f"\nSpeedups relative to {slowest.backend_name}:")
            for result in valid_results:
                if result is slowest:
                    continue
                speedup = (
                    (slowest.dnfr_median + slowest.si_median)
                    / (result.dnfr_median + result.si_median)
                )
                print(f"  {result.backend_name}: {speedup:.2f}x")
    
    print("\n" + "=" * 80)


def main() -> None:
    """Run backend comparison benchmark."""
    
    parser = argparse.ArgumentParser(
        description="Benchmark TNFR backend implementations",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Node counts to benchmark (default: 50 100)",
    )
    parser.add_argument(
        "--edge-probability",
        type=float,
        default=0.2,
        help="Edge probability for random graphs (default: 0.2)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions per configuration (default: 5)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Specific backends to test (default: all available)",
    )
    
    args = parser.parse_args()
    
    # Determine which backends to test
    available = available_backends()
    if args.backends:
        backends_to_test = args.backends
    else:
        backends_to_test = sorted(available.keys())
    
    print("Available backends:", ", ".join(sorted(available.keys())))
    print("Testing backends:", ", ".join(backends_to_test))
    print(f"Edge probability: {args.edge_probability}")
    print(f"Repeats per configuration: {args.repeats}")
    print()
    
    results = []
    
    for num_nodes in args.nodes:
        print(f"Benchmarking {num_nodes} nodes...")
        for backend_name in backends_to_test:
            result = benchmark_backend(
                backend_name,
                num_nodes,
                args.edge_probability,
                args.repeats,
            )
            results.append(result)
    
    print_results(results)


if __name__ == "__main__":
    main()
