"""Profile cache effectiveness in TNFR hot paths.

This benchmark focuses specifically on cache hit rates, eviction patterns,
and memory efficiency across the three primary hot paths:

1. **Laplacian Matrix** (via dnfr_laplacian and ΔNFR preparation)
2. **C(t) History** (coherence matrix tracking)
3. **Si Projections** (sense index computation buffers)

The profiler measures:
- Cache hit/miss rates per hot path
- Buffer reuse effectiveness
- Memory allocation patterns
- Invalidation frequency

Usage:
    python benchmarks/cache_hot_path_profiler.py --nodes 100 --steps 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import default_compute_delta_nfr, dnfr_laplacian
from tnfr.metrics.coherence import coherence_matrix
from tnfr.metrics.sense_index import compute_Si
from tnfr.metrics.cache_utils import configure_hot_path_caches
from tnfr.utils import json_dumps
from tnfr.utils.graph import get_graph

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


def _initialize_graph(
    *,
    node_count: int,
    edge_probability: float,
    seed: int,
) -> nx.Graph:
    """Create a test graph with initialized node attributes."""
    import random
    rng = random.Random(seed)
    G = nx.erdos_renyi_graph(node_count, edge_probability, seed=seed)
    
    # Initialize node attributes for TNFR
    for node in G.nodes():
        set_attr(G.nodes[node], ALIAS_THETA, rng.random() * 6.28318)
        set_attr(G.nodes[node], ALIAS_EPI, rng.random())
        set_attr(G.nodes[node], ALIAS_VF, rng.random())
        G.nodes[node]["nu_f"] = rng.random()
        G.nodes[node]["delta_nfr"] = rng.random() * 0.1
    
    return G


def _get_cache_snapshot(G: nx.Graph) -> dict[str, Any]:
    """Extract current cache metrics from graph managers."""
    graph_dict = get_graph(G)
    snapshot: dict[str, Any] = {
        "aggregate": {"hits": 0, "misses": 0, "evictions": 0, "hit_rate": 0.0},
        "by_cache": {},
    }

    cache_manager = graph_dict.get("_tnfr_cache_manager")
    if cache_manager is not None and hasattr(cache_manager, "aggregate_metrics"):
        try:
            aggregate = cache_manager.aggregate_metrics()
            snapshot["aggregate"] = {
                "hits": aggregate.hits,
                "misses": aggregate.misses,
                "evictions": aggregate.evictions,
                "hit_rate": (
                    aggregate.hits / (aggregate.hits + aggregate.misses)
                    if (aggregate.hits + aggregate.misses) > 0
                    else 0.0
                ),
            }
            
            if hasattr(cache_manager, "iter_metrics"):
                for name, stats in cache_manager.iter_metrics():
                    total = stats.hits + stats.misses
                    snapshot["by_cache"][name] = {
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "evictions": stats.evictions,
                        "hit_rate": stats.hits / total if total > 0 else 0.0,
                    }
        except Exception:
            pass

    edge_cache_manager = graph_dict.get("_edge_cache_manager")
    if edge_cache_manager is not None and hasattr(edge_cache_manager, "_manager"):
        try:
            edge_manager = edge_cache_manager._manager
            if hasattr(edge_manager, "aggregate_metrics"):
                edge_aggregate = edge_manager.aggregate_metrics()
                if snapshot["aggregate"]["hits"] == 0:
                    snapshot["aggregate"] = {
                        "hits": edge_aggregate.hits,
                        "misses": edge_aggregate.misses,
                        "evictions": edge_aggregate.evictions,
                        "hit_rate": (
                            edge_aggregate.hits / (edge_aggregate.hits + edge_aggregate.misses)
                            if (edge_aggregate.hits + edge_aggregate.misses) > 0
                            else 0.0
                        ),
                    }
        except Exception:
            pass

    return snapshot


def _compute_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute the difference in cache metrics between two snapshots."""
    delta: dict[str, Any] = {
        "aggregate": {},
        "by_cache": {},
    }
    
    # Aggregate delta
    for key in ["hits", "misses", "evictions"]:
        delta["aggregate"][key] = after["aggregate"].get(key, 0) - before["aggregate"].get(key, 0)
    
    total = delta["aggregate"]["hits"] + delta["aggregate"]["misses"]
    delta["aggregate"]["hit_rate"] = (
        delta["aggregate"]["hits"] / total if total > 0 else 0.0
    )
    
    # Per-cache delta
    all_caches = set(before["by_cache"].keys()) | set(after["by_cache"].keys())
    for cache_name in all_caches:
        before_cache = before["by_cache"].get(cache_name, {"hits": 0, "misses": 0, "evictions": 0})
        after_cache = after["by_cache"].get(cache_name, {"hits": 0, "misses": 0, "evictions": 0})
        
        cache_delta = {}
        for key in ["hits", "misses", "evictions"]:
            cache_delta[key] = after_cache.get(key, 0) - before_cache.get(key, 0)
        
        cache_total = cache_delta["hits"] + cache_delta["misses"]
        cache_delta["hit_rate"] = cache_delta["hits"] / cache_total if cache_total > 0 else 0.0
        
        delta["by_cache"][cache_name] = cache_delta
    
    return delta


def profile_hot_paths(
    *,
    node_count: int,
    edge_probability: float,
    steps: int,
    seed: int,
    buffer_cache_size: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Profile cache behavior across hot paths for multiple simulation steps.
    
    Parameters
    ----------
    node_count : int
        Number of nodes in test graph.
    edge_probability : float
        Edge probability for Erdős-Rényi graph.
    steps : int
        Number of simulation steps to profile.
    seed : int
        Random seed for reproducibility.
    buffer_cache_size : int
        Maximum entries for buffer caches.
    output_path : Path or None, optional
        Path to write JSON report. If None, only returns data.
    
    Returns
    -------
    dict[str, Any]
        Profiling report with cache metrics per hot path.
    """
    print(f"Initializing graph: {node_count} nodes, p={edge_probability}")
    G = _initialize_graph(
        node_count=node_count,
        edge_probability=edge_probability,
        seed=seed,
    )
    
    print(f"Configuring caches: buffer_max_entries={buffer_cache_size}")
    configure_hot_path_caches(G, buffer_max_entries=buffer_cache_size)
    
    # Warm up caches
    print("Warming up caches...")
    compute_Si(G, inplace=True)
    default_compute_delta_nfr(G)
    coherence_matrix(G)
    
    # Profile each hot path
    hot_paths = {
        "sense_index": lambda: compute_Si(G, inplace=True),
        "dnfr_laplacian": lambda: dnfr_laplacian(G),
        "coherence_matrix": lambda: coherence_matrix(G),
    }
    
    results: dict[str, Any] = {
        "configuration": {
            "node_count": node_count,
            "edge_count": G.number_of_edges(),
            "edge_probability": edge_probability,
            "steps": steps,
            "seed": seed,
            "buffer_cache_size": buffer_cache_size,
        },
        "hot_paths": {},
        "summary": {
            "total_hits": 0,
            "total_misses": 0,
            "total_evictions": 0,
            "overall_hit_rate": 0.0,
        },
    }
    
    for path_name, compute_fn in hot_paths.items():
        print(f"\nProfiling {path_name}...")
        step_metrics = []
        
        for step in range(steps):
            before = _get_cache_snapshot(G)
            
            start = perf_counter()
            compute_fn()
            elapsed = perf_counter() - start
            
            after = _get_cache_snapshot(G)
            delta = _compute_delta(before, after)
            
            step_metrics.append({
                "step": step,
                "elapsed_ms": elapsed * 1000,
                "cache_delta": delta,
            })
            
            if step % max(1, steps // 10) == 0:
                hit_rate = delta["aggregate"]["hit_rate"] * 100
                print(f"  Step {step}: {elapsed*1000:.2f}ms, hit_rate={hit_rate:.1f}%")
        
        # Aggregate metrics for this hot path
        total_hits = sum(m["cache_delta"]["aggregate"]["hits"] for m in step_metrics)
        total_misses = sum(m["cache_delta"]["aggregate"]["misses"] for m in step_metrics)
        total_evictions = sum(m["cache_delta"]["aggregate"]["evictions"] for m in step_metrics)
        avg_elapsed = sum(m["elapsed_ms"] for m in step_metrics) / len(step_metrics)
        
        results["hot_paths"][path_name] = {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_evictions": total_evictions,
            "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            "avg_elapsed_ms": avg_elapsed,
            "steps": step_metrics,
        }
        
        results["summary"]["total_hits"] += total_hits
        results["summary"]["total_misses"] += total_misses
        results["summary"]["total_evictions"] += total_evictions
    
    # Compute overall hit rate
    total = results["summary"]["total_hits"] + results["summary"]["total_misses"]
    results["summary"]["overall_hit_rate"] = (
        results["summary"]["total_hits"] / total if total > 0 else 0.0
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("CACHE PROFILING SUMMARY")
    print("=" * 70)
    print(f"Configuration: {node_count} nodes, {steps} steps")
    print(f"Buffer cache size: {buffer_cache_size}")
    print(f"\nOverall Statistics:")
    print(f"  Total Hits:      {results['summary']['total_hits']}")
    print(f"  Total Misses:    {results['summary']['total_misses']}")
    print(f"  Total Evictions: {results['summary']['total_evictions']}")
    print(f"  Overall Hit Rate: {results['summary']['overall_hit_rate']*100:.1f}%")
    print(f"\nPer Hot Path:")
    for path_name, metrics in results["hot_paths"].items():
        print(f"  {path_name}:")
        print(f"    Hit Rate: {metrics['hit_rate']*100:.1f}%")
        print(f"    Avg Time: {metrics['avg_elapsed_ms']:.2f}ms")
        print(f"    Evictions: {metrics['total_evictions']}")
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_dumps(results, indent=2, ensure_ascii=False))
        print(f"\nDetailed report written to: {output_path}")
    
    return results


def main() -> None:
    """Run cache profiling benchmark from command line."""
    parser = argparse.ArgumentParser(
        description="Profile TNFR cache effectiveness in hot paths"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=100,
        help="Number of nodes in test graph",
    )
    parser.add_argument(
        "--edge-probability",
        type=float,
        default=0.1,
        help="Edge probability for Erdős-Rényi graph",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of simulation steps to profile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--buffer-cache-size",
        type=int,
        default=128,
        help="Maximum entries for buffer caches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON report (default: print only)",
    )
    
    args = parser.parse_args()
    
    profile_hot_paths(
        node_count=args.nodes,
        edge_probability=args.edge_probability,
        steps=args.steps,
        seed=args.seed,
        buffer_cache_size=args.buffer_cache_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
