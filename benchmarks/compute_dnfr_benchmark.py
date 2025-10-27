"""Microbenchmark for `_compute_dnfr` vectorised vs. fallback execution."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import _compute_dnfr, _prepare_dnfr_data
from tnfr.numpy import get_numpy

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


@dataclass(slots=True)
class BenchmarkResult:
    """Container with summary statistics for a ΔNFR benchmark run."""

    num_nodes: int
    edge_probability: float
    density: float
    vectorized: tuple[float, float, float, float] | None
    fallback: tuple[float, float, float, float]


def _build_graph(num_nodes: int, edge_probability: float, seed: int) -> nx.Graph:
    """Create a reproducible Erdos-Renyi graph with the TNFR aliases initialised."""

    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        graph.nodes[node][ALIAS_THETA] = 0.0
        graph.nodes[node][ALIAS_EPI] = 0.0
        graph.nodes[node][ALIAS_VF] = 0.0
    return graph


def _summarise(samples: Iterable[float]) -> tuple[float, float, float, float]:
    """Return (best, median, mean, worst) statistics for ``samples``."""

    timings = list(samples)
    if not timings:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(timings),
        statistics.median(timings),
        sum(timings) / len(timings),
        max(timings),
    )


def _measure_run(graph: nx.Graph) -> float:
    """Compute ΔNFR on ``graph`` once and return the wall-clock duration."""

    data = _prepare_dnfr_data(graph)
    start = time.perf_counter()
    _compute_dnfr(graph, data)
    return time.perf_counter() - start


def benchmark_configuration(
    *,
    num_nodes: int,
    edge_probability: float,
    repeats: int,
    force_dense: bool,
    numpy_available: bool,
) -> BenchmarkResult:
    """Benchmark `_compute_dnfr` for the provided graph configuration."""

    vectorized_samples: list[float] = []
    fallback_samples: list[float] = []
    densities: list[float] = []

    for rep in range(repeats):
        base_graph = _build_graph(num_nodes, edge_probability, seed=rep + 1)
        densities.append(nx.density(base_graph))

        fallback_graph = base_graph.copy()
        fallback_graph.graph["vectorized_dnfr"] = False
        fallback_graph.graph.pop("dnfr_force_dense", None)
        fallback_samples.append(_measure_run(fallback_graph))

        if numpy_available:
            vector_graph = base_graph.copy()
            vector_graph.graph.pop("vectorized_dnfr", None)
            if force_dense:
                vector_graph.graph["dnfr_force_dense"] = True
            else:
                vector_graph.graph.pop("dnfr_force_dense", None)
            vectorized_samples.append(_measure_run(vector_graph))

    density = sum(densities) / len(densities) if densities else 0.0
    vectorized_stats = (
        _summarise(vectorized_samples) if vectorized_samples else None
    )
    fallback_stats = _summarise(fallback_samples)

    return BenchmarkResult(
        num_nodes=num_nodes,
        edge_probability=edge_probability,
        density=density,
        vectorized=vectorized_stats,
        fallback=fallback_stats,
    )


def run(
    nodes: list[int] | None = None,
    edge_probabilities: list[float] | None = None,
    repeats: int = 5,
    *,
    force_dense: bool = False,
) -> None:
    """Execute the benchmark matrix and print a comparison table."""

    numpy_available = get_numpy() is not None
    if nodes is None:
        nodes = [128, 256, 512]
    if edge_probabilities is None:
        edge_probabilities = [0.05, 0.1, 0.25]

    if not numpy_available:
        print(
            "NumPy is unavailable; only the pure-Python fallback timings will be reported."
        )

    headers = (
        "nodes",
        "p",
        "density",
        "vectorized best/median/mean/worst (s)",
        "fallback best/median/mean/worst (s)",
        "ratio (fallback ÷ vectorized)",
    )
    print(" | ".join(headers))
    print("-" * (len(" | ".join(headers))))

    for num_nodes in nodes:
        for probability in edge_probabilities:
            result = benchmark_configuration(
                num_nodes=num_nodes,
                edge_probability=probability,
                repeats=repeats,
                force_dense=force_dense,
                numpy_available=numpy_available,
            )

            vectorized_display = "n/a"
            ratio_display = "n/a"
            if result.vectorized:
                v_best, v_median, v_mean, v_worst = result.vectorized
                vectorized_display = (
                    f"{v_best:.6f} / {v_median:.6f} / {v_mean:.6f} / {v_worst:.6f}"
                )
                f_best, f_median, f_mean, f_worst = result.fallback
                ratio = f_mean / v_mean if v_mean else float("inf")
                ratio_display = f"{ratio:.2f}×"
            f_best, f_median, f_mean, f_worst = result.fallback
            fallback_display = (
                f"{f_best:.6f} / {f_median:.6f} / {f_mean:.6f} / {f_worst:.6f}"
            )
            print(
                f"{num_nodes} | {probability:.2f} | {result.density:.4f} | "
                f"{vectorized_display} | {fallback_display} | {ratio_display}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark `_compute_dnfr` vectorised vs. fallback execution",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="*",
        help="List of node counts to benchmark (defaults to 128 256 512)",
    )
    parser.add_argument(
        "--edge-probabilities",
        type=float,
        nargs="*",
        help="Edge probabilities for Erdos-Renyi graphs (defaults to 0.05 0.1 0.25)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of random graphs to benchmark per configuration",
    )
    parser.add_argument(
        "--force-dense",
        action="store_true",
        help="Force the dense NumPy accumulation path when vectorisation is enabled",
    )
    args = parser.parse_args()
    run(
        nodes=args.nodes,
        edge_probabilities=args.edge_probabilities,
        repeats=args.repeats,
        force_dense=args.force_dense,
    )
