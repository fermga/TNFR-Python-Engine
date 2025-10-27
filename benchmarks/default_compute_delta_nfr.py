"""Benchmark for the default ΔNFR computation pipeline."""

from __future__ import annotations

import argparse
import cProfile
import statistics
import time
from pathlib import Path
from typing import Literal

import networkx as nx

from tnfr.constants import get_aliases
from tnfr.dynamics import default_compute_delta_nfr

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


def _build_graph(num_nodes: int, edge_probability: float, seed: int) -> nx.Graph:
    """Create a reproducible random graph with zeroed TNFR aliases."""
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        graph.nodes[node][ALIAS_THETA] = 0.0
        graph.nodes[node][ALIAS_EPI] = 0.0
        graph.nodes[node][ALIAS_VF] = 0.0
    return graph


def _profile(
    *,
    num_nodes: int,
    edge_probability: float,
    repeats: int,
    output: Path,
    fmt: Literal["pstats", "json"],
) -> None:
    """Record profiling information for ``default_compute_delta_nfr``."""

    def _runner() -> None:
        for rep in range(repeats):
            graph = _build_graph(num_nodes, edge_probability, seed=rep + 1)
            default_compute_delta_nfr(graph)

    profiler = cProfile.Profile()
    profiler.runcall(_runner)

    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pstats":
        profiler.dump_stats(str(output))
        print(f"Stored ΔNFR profile at {output}")
        return

    stats = profiler.getstats()
    payload = []
    for stat in stats:
        callcount = stat.callcount
        primitive = stat.primitive
        totaltime = stat.totaltime
        inlinetime = stat.inlinetime
        code = stat.code
        payload.append(
            {
                "function": getattr(code, "co_name", str(code)),
                "filename": getattr(code, "co_filename", None),
                "callcount": callcount,
                "primitive": primitive,
                "totaltime": totaltime,
                "inlinetime": inlinetime,
            }
        )

    payload.sort(key=lambda entry: entry["totaltime"], reverse=True)

    import json

    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Stored ΔNFR profile at {output}")


def run(num_nodes: int = 200, edge_probability: float = 0.1, repeats: int = 5) -> None:
    """Run the benchmark and print timing statistics."""
    durations = []
    for rep in range(repeats):
        graph = _build_graph(num_nodes, edge_probability, seed=rep + 1)
        start = time.perf_counter()
        default_compute_delta_nfr(graph)
        durations.append(time.perf_counter() - start)

    best = min(durations)
    worst = max(durations)
    median = statistics.median(durations)
    mean = sum(durations) / len(durations)

    print(
        "default_compute_delta_nfr "
        f"{repeats} runs on {num_nodes} nodes (p={edge_probability}):"
    )
    print(
        f"best={best:.6f}s median={median:.6f}s " f"mean={mean:.6f}s worst={worst:.6f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark and optionally profile default_compute_delta_nfr",
    )
    parser.add_argument("--nodes", type=int, default=200, help="Number of nodes")
    parser.add_argument(
        "--edge-probability",
        type=float,
        default=0.1,
        help="Edge probability for the Erdos-Renyi graph",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Benchmark repeats")
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional output path to save cProfile statistics",
    )
    parser.add_argument(
        "--profile-format",
        choices=("pstats", "json"),
        default="pstats",
        help="Format used when --profile is specified",
    )

    args = parser.parse_args()
    if args.profile:
        _profile(
            num_nodes=args.nodes,
            edge_probability=args.edge_probability,
            repeats=args.repeats,
            output=args.profile,
            fmt=args.profile_format,
        )
    else:
        run(num_nodes=args.nodes, edge_probability=args.edge_probability, repeats=args.repeats)
