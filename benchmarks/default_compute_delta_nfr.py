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

BenchmarkMode = Literal["auto", "force-dense", "python"]

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


def _apply_mode(graph: nx.Graph, mode: BenchmarkMode) -> None:
    """Configure ``graph`` so ΔNFR executes in the requested ``mode``."""

    if mode == "auto":
        graph.graph.pop("vectorized_dnfr", None)
        graph.graph.pop("dnfr_force_dense", None)
        return
    if mode == "force-dense":
        graph.graph["dnfr_force_dense"] = True
        graph.graph.pop("vectorized_dnfr", None)
        return
    if mode == "python":
        graph.graph["vectorized_dnfr"] = False
        graph.graph.pop("dnfr_force_dense", None)
        return
    raise ValueError(f"Unsupported benchmark mode: {mode}")


def _profile(
    *,
    num_nodes: int,
    edge_probability: float,
    repeats: int,
    output: Path,
    fmt: Literal["pstats", "json"],
    mode: BenchmarkMode,
) -> None:
    """Record profiling information for ``default_compute_delta_nfr``."""

    def _runner() -> None:
        for rep in range(repeats):
            graph = _build_graph(num_nodes, edge_probability, seed=rep + 1)
            _apply_mode(graph, mode)
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


def _summarise(durations: list[float]) -> tuple[float, float, float, float]:
    """Return (best, median, mean, worst) statistics for ``durations``."""

    if not durations:
        return (0.0, 0.0, 0.0, 0.0)
    best = min(durations)
    worst = max(durations)
    median = statistics.median(durations)
    mean = sum(durations) / len(durations)
    return best, median, mean, worst


def run(
    num_nodes: int = 200,
    edge_probability: float = 0.1,
    repeats: int = 5,
    *,
    mode: BenchmarkMode = "auto",
    compare_python: bool = True,
) -> None:
    """Run the benchmark and print timing statistics."""

    vectorized_durations: list[float] = []
    python_durations: list[float] = []
    densities: list[float] = []

    for rep in range(repeats):
        graph = _build_graph(num_nodes, edge_probability, seed=rep + 1)
        densities.append(nx.density(graph))
        _apply_mode(graph, mode)

        fallback_graph: nx.Graph | None = None
        if compare_python and mode != "python":
            fallback_graph = graph.copy()
            fallback_graph.graph["vectorized_dnfr"] = False

        start = time.perf_counter()
        default_compute_delta_nfr(graph)
        vectorized_durations.append(time.perf_counter() - start)

        if fallback_graph is not None:
            start = time.perf_counter()
            default_compute_delta_nfr(fallback_graph)
            python_durations.append(time.perf_counter() - start)

    v_best, v_median, v_mean, v_worst = _summarise(vectorized_durations)
    print(
        "default_compute_delta_nfr "
        f"mode={mode} {repeats} runs on {num_nodes} nodes (p={edge_probability}):"
    )
    if densities:
        print(
            f"density≈{sum(densities)/len(densities):.4f} "
            f"min={min(densities):.4f} max={max(densities):.4f}"
        )
    print(
        "vectorized: "
        f"best={v_best:.6f}s median={v_median:.6f}s "
        f"mean={v_mean:.6f}s worst={v_worst:.6f}s"
    )

    if python_durations:
        p_best, p_median, p_mean, p_worst = _summarise(python_durations)
        ratio = p_mean / v_mean if v_mean else float("inf")
        print(
            "python fallback: "
            f"best={p_best:.6f}s median={p_median:.6f}s "
            f"mean={p_mean:.6f}s worst={p_worst:.6f}s "
            f"ratio={ratio:.2f}x"
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
        "--mode",
        choices=("auto", "force-dense", "python"),
        default="auto",
        help="ΔNFR execution mode",
    )
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
    parser.add_argument(
        "--compare-python",
        dest="compare_python",
        action="store_true",
        help="Also time the pure-Python fallback",
    )
    parser.add_argument(
        "--no-compare-python",
        dest="compare_python",
        action="store_false",
        help="Skip timing the pure-Python fallback",
    )
    parser.set_defaults(compare_python=True)

    args = parser.parse_args()
    if args.profile:
        _profile(
            num_nodes=args.nodes,
            edge_probability=args.edge_probability,
            repeats=args.repeats,
            output=args.profile,
            fmt=args.profile_format,
            mode=args.mode,
        )
    else:
        run(
            num_nodes=args.nodes,
            edge_probability=args.edge_probability,
            repeats=args.repeats,
            mode=args.mode,
            compare_python=args.compare_python,
        )
