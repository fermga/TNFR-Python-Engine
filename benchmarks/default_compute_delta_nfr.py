"""Benchmark for the default ΔNFR computation pipeline."""

import statistics
import time

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
    run()
