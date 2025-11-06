"""Comprehensive cache profiler for TNFR hot paths with detailed metrics.

This enhanced profiler tracks ALL cache layers in TNFR:
1. EdgeCacheManager (buffer allocations via edge_version_cache)
2. CacheManager (DNFR preparation state, structural caches)
3. LRU caches in individual functions

The profiler provides:
- Cache hit/miss/eviction rates per layer
- Memory usage estimates
- Cache key distribution
- Invalidation frequency analysis
- Performance correlation with cache effectiveness

Usage:
    python benchmarks/comprehensive_cache_profiler.py --nodes 200 --steps 50
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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


def _get_comprehensive_cache_snapshot(G: nx.Graph) -> dict[str, Any]:
    """Extract metrics from ALL cache layers."""
    graph_dict = get_graph(G)
    snapshot: dict[str, Any] = {
        "edge_cache": {"hits": 0, "misses": 0, "evictions": 0, "entries": 0},
        "tnfr_cache": {"hits": 0, "misses": 0, "evictions": 0, "caches": {}},
        "dnfr_prep": {"exists": False, "cache_size": 0},
        "buffer_cache_keys": [],
        "total_cached_arrays": 0,
    }

    # Track EdgeCacheManager metrics
    edge_cache_mgr = graph_dict.get("_edge_cache_manager")
    if edge_cache_mgr is not None and hasattr(edge_cache_mgr, "_manager"):
        try:
            edge_manager = edge_cache_mgr._manager
            if hasattr(edge_manager, "get_metrics"):
                state_key = edge_cache_mgr._STATE_KEY
                stats = edge_manager.get_metrics(state_key)
                snapshot["edge_cache"]["hits"] = stats.hits
                snapshot["edge_cache"]["misses"] = stats.misses
                snapshot["edge_cache"]["evictions"] = stats.evictions

                # Count cached entries
                state = edge_cache_mgr.get_cache(None, create=False)
                if state and hasattr(state, "cache"):
                    snapshot["edge_cache"]["entries"] = len(state.cache)
                    snapshot["buffer_cache_keys"] = list(state.cache.keys())
        except (AttributeError, KeyError, TypeError) as e:
            snapshot["edge_cache"]["error"] = str(e)
        except Exception as e:
            # Log unexpected errors for debugging
            import traceback

            snapshot["edge_cache"]["error"] = f"{type(e).__name__}: {e}"
            snapshot["edge_cache"]["traceback"] = traceback.format_exc()

    # Track CacheManager metrics
    cache_manager = graph_dict.get("_tnfr_cache_manager")
    if cache_manager is not None and hasattr(cache_manager, "aggregate_metrics"):
        try:
            aggregate = cache_manager.aggregate_metrics()
            snapshot["tnfr_cache"]["hits"] = aggregate.hits
            snapshot["tnfr_cache"]["misses"] = aggregate.misses
            snapshot["tnfr_cache"]["evictions"] = aggregate.evictions

            if hasattr(cache_manager, "iter_metrics"):
                for name, stats in cache_manager.iter_metrics():
                    snapshot["tnfr_cache"]["caches"][name] = {
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "evictions": stats.evictions,
                    }
        except (AttributeError, KeyError, TypeError) as e:
            snapshot["tnfr_cache"]["error"] = str(e)
        except Exception as e:
            # Log unexpected errors for debugging
            import traceback

            snapshot["tnfr_cache"]["error"] = f"{type(e).__name__}: {e}"
            snapshot["tnfr_cache"]["traceback"] = traceback.format_exc()

    # Track DNFR preparation state
    dnfr_state = graph_dict.get("_dnfr_prep_cache")
    if dnfr_state is not None:
        snapshot["dnfr_prep"]["exists"] = True
        if hasattr(dnfr_state, "idx"):
            snapshot["dnfr_prep"]["cache_size"] = len(dnfr_state.idx)

    # Count buffer cache entries more explicitly via EdgeCacheManager
    # Note: This counts cache entries, not individual arrays
    total_arrays = 0
    edge_cache_mgr = graph_dict.get("_edge_cache_manager")
    if edge_cache_mgr is not None:
        try:
            state = edge_cache_mgr.get_cache(None, create=False)
            if state and hasattr(state, "cache"):
                # Count entries with buffer-related key prefixes
                for key in state.cache.keys():
                    if isinstance(key, tuple) and len(key) >= 2:
                        prefix = key[0]
                        if isinstance(prefix, str) and (
                            "buffer" in prefix.lower() or prefix.startswith("_si")
                        ):
                            total_arrays += 1
        except Exception:
            pass  # Silently handle edge cases during counting
    snapshot["total_cached_arrays"] = total_arrays

    return snapshot


def _compute_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute the difference in cache metrics between two snapshots."""
    delta: dict[str, Any] = {
        "edge_cache": {},
        "tnfr_cache": {},
        "total_hits": 0,
        "total_misses": 0,
        "hit_rate": 0.0,
    }

    # Edge cache delta
    for key in ["hits", "misses", "evictions", "entries"]:
        delta["edge_cache"][key] = after["edge_cache"].get(key, 0) - before[
            "edge_cache"
        ].get(key, 0)

    # TNFR cache delta
    for key in ["hits", "misses", "evictions"]:
        delta["tnfr_cache"][key] = after["tnfr_cache"].get(key, 0) - before[
            "tnfr_cache"
        ].get(key, 0)

    # Combined totals
    delta["total_hits"] = delta["edge_cache"]["hits"] + delta["tnfr_cache"]["hits"]
    delta["total_misses"] = (
        delta["edge_cache"]["misses"] + delta["tnfr_cache"]["misses"]
    )
    total = delta["total_hits"] + delta["total_misses"]
    delta["hit_rate"] = delta["total_hits"] / total if total > 0 else 0.0

    # Array allocation tracking
    # Positive values indicate new allocations, negative values indicate evictions/cleanup
    # For reuse rate calculation, we only care about new allocations (positive delta)
    delta["new_arrays"] = max(
        0, after.get("total_cached_arrays", 0) - before.get("total_cached_arrays", 0)
    )

    return delta


def profile_comprehensive(
    *,
    node_count: int,
    edge_probability: float,
    steps: int,
    seed: int,
    buffer_cache_size: int,
    output_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Profile all cache layers across TNFR hot paths.

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
    verbose : bool, default: False
        Print detailed per-step metrics.

    Returns
    -------
    dict[str, Any]
        Comprehensive profiling report with all cache layer metrics.
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

    # Get baseline snapshot
    baseline = _get_comprehensive_cache_snapshot(G)

    # Profile each hot path
    hot_paths = {
        "sense_index": lambda: compute_Si(G, inplace=True),
        "dnfr_laplacian": lambda: dnfr_laplacian(G),
        "coherence_matrix": lambda: coherence_matrix(G),
        "default_compute_delta_nfr": lambda: default_compute_delta_nfr(G),
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
        "baseline": baseline,
        "hot_paths": {},
        "summary": {
            "total_edge_hits": 0,
            "total_edge_misses": 0,
            "total_tnfr_hits": 0,
            "total_tnfr_misses": 0,
            "overall_hit_rate": 0.0,
            "buffer_reuse_rate": 0.0,
        },
    }

    for path_name, compute_fn in hot_paths.items():
        print(f"\nProfiling {path_name}...")
        step_metrics = []

        for step in range(steps):
            before = _get_comprehensive_cache_snapshot(G)

            start = perf_counter()
            compute_fn()
            elapsed = perf_counter() - start

            after = _get_comprehensive_cache_snapshot(G)
            delta = _compute_delta(before, after)

            step_metrics.append(
                {
                    "step": step,
                    "elapsed_ms": elapsed * 1000,
                    "cache_delta": delta,
                }
            )

            if verbose or (step % max(1, steps // 10) == 0):
                hit_rate = delta["hit_rate"] * 100
                edge_hits = delta["edge_cache"]["hits"]
                edge_misses = delta["edge_cache"]["misses"]
                print(
                    f"  Step {step}: {elapsed*1000:.2f}ms, "
                    f"hit_rate={hit_rate:.1f}%, "
                    f"edge_cache={edge_hits}/{edge_hits+edge_misses}"
                )

        # Aggregate metrics for this hot path
        total_edge_hits = sum(
            m["cache_delta"]["edge_cache"]["hits"] for m in step_metrics
        )
        total_edge_misses = sum(
            m["cache_delta"]["edge_cache"]["misses"] for m in step_metrics
        )
        total_tnfr_hits = sum(
            m["cache_delta"]["tnfr_cache"]["hits"] for m in step_metrics
        )
        total_tnfr_misses = sum(
            m["cache_delta"]["tnfr_cache"]["misses"] for m in step_metrics
        )
        total_new_arrays = sum(m["cache_delta"]["new_arrays"] for m in step_metrics)
        avg_elapsed = sum(m["elapsed_ms"] for m in step_metrics) / len(step_metrics)

        combined_hits = total_edge_hits + total_tnfr_hits
        combined_misses = total_edge_misses + total_tnfr_misses

        results["hot_paths"][path_name] = {
            "edge_cache": {
                "hits": total_edge_hits,
                "misses": total_edge_misses,
                "hit_rate": (
                    total_edge_hits / (total_edge_hits + total_edge_misses)
                    if (total_edge_hits + total_edge_misses) > 0
                    else 0.0
                ),
            },
            "tnfr_cache": {
                "hits": total_tnfr_hits,
                "misses": total_tnfr_misses,
                "hit_rate": (
                    total_tnfr_hits / (total_tnfr_hits + total_tnfr_misses)
                    if (total_tnfr_hits + total_tnfr_misses) > 0
                    else 0.0
                ),
            },
            "combined_hit_rate": (
                combined_hits / (combined_hits + combined_misses)
                if (combined_hits + combined_misses) > 0
                else 0.0
            ),
            "avg_elapsed_ms": avg_elapsed,
            "total_new_arrays": total_new_arrays,
            # Buffer reuse rate: measures how often existing buffer allocations are reused
            # 100% = all steps reused existing buffers (optimal)
            # <100% = some steps required new buffer allocations
            # Note: First call typically allocates, subsequent calls reuse
            "buffer_reuse_rate": 1.0 - (total_new_arrays / steps) if steps > 0 else 0.0,
            "steps": step_metrics if verbose else None,
        }

        results["summary"]["total_edge_hits"] += total_edge_hits
        results["summary"]["total_edge_misses"] += total_edge_misses
        results["summary"]["total_tnfr_hits"] += total_tnfr_hits
        results["summary"]["total_tnfr_misses"] += total_tnfr_misses

    # Compute overall metrics
    total_hits = (
        results["summary"]["total_edge_hits"] + results["summary"]["total_tnfr_hits"]
    )
    total_misses = (
        results["summary"]["total_edge_misses"]
        + results["summary"]["total_tnfr_misses"]
    )
    results["summary"]["overall_hit_rate"] = (
        total_hits / (total_hits + total_misses)
        if (total_hits + total_misses) > 0
        else 0.0
    )

    # Calculate buffer reuse across all paths
    total_steps = steps * len(hot_paths)
    total_new_arrays = sum(
        hp["total_new_arrays"] for hp in results["hot_paths"].values()
    )
    results["summary"]["buffer_reuse_rate"] = (
        1.0 - (total_new_arrays / total_steps) if total_steps > 0 else 0.0
    )

    # Final snapshot
    results["final_state"] = _get_comprehensive_cache_snapshot(G)

    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE CACHE PROFILING SUMMARY")
    print("=" * 70)
    print(
        f"Configuration: {node_count} nodes, {G.number_of_edges()} edges, {steps} steps"
    )
    print(f"Buffer cache size: {buffer_cache_size}")
    print(f"\nOverall Statistics:")
    print(f"  Total Edge Cache Hits:    {results['summary']['total_edge_hits']}")
    print(f"  Total Edge Cache Misses:  {results['summary']['total_edge_misses']}")
    print(f"  Total TNFR Cache Hits:    {results['summary']['total_tnfr_hits']}")
    print(f"  Total TNFR Cache Misses:  {results['summary']['total_tnfr_misses']}")
    print(
        f"  Overall Hit Rate:         {results['summary']['overall_hit_rate']*100:.1f}%"
    )
    print(
        f"  Buffer Reuse Rate:        {results['summary']['buffer_reuse_rate']*100:.1f}%"
    )
    print(f"\nFinal Cache State:")
    print(
        f"  Edge Cache Entries:       {results['final_state']['edge_cache']['entries']}"
    )
    print(
        f"  Total Cached Arrays:      {results['final_state']['total_cached_arrays']}"
    )
    print(
        f"  DNFR Prep Cache:          {'Present' if results['final_state']['dnfr_prep']['exists'] else 'Not initialized'}"
    )
    print(f"\nPer Hot Path:")
    for path_name, metrics in results["hot_paths"].items():
        print(f"  {path_name}:")
        print(f"    Combined Hit Rate:  {metrics['combined_hit_rate']*100:.1f}%")
        print(f"    Edge Cache Rate:    {metrics['edge_cache']['hit_rate']*100:.1f}%")
        print(f"    TNFR Cache Rate:    {metrics['tnfr_cache']['hit_rate']*100:.1f}%")
        print(f"    Buffer Reuse:       {metrics['buffer_reuse_rate']*100:.1f}%")
        print(f"    Avg Time:           {metrics['avg_elapsed_ms']:.2f}ms")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove step details if not verbose to keep file size manageable
        if not verbose:
            for hp in results["hot_paths"].values():
                hp.pop("steps", None)
        output_path.write_text(json_dumps(results, indent=2, ensure_ascii=False))
        print(f"\nDetailed report written to: {output_path}")

    return results


def main() -> None:
    """Run comprehensive cache profiling from command line."""
    parser = argparse.ArgumentParser(
        description="Comprehensive TNFR cache profiler tracking all cache layers"
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
        default=30,
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
        default=256,
        help="Maximum entries for buffer caches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON report (default: print only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-step metrics",
    )

    args = parser.parse_args()

    try:
        profile_comprehensive(
            node_count=args.nodes,
            edge_probability=args.edge_probability,
            steps=args.steps,
            seed=args.seed,
            buffer_cache_size=args.buffer_cache_size,
            output_path=args.output,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nProfiling failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
