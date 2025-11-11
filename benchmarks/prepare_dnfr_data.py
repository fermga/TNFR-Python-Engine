"""Benchmark for _prepare_dnfr_data performance."""

import time
from typing import Callable

import networkx as nx

from tnfr.alias import collect_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import _prepare_dnfr_data
from tnfr.utils import cached_nodes_and_A

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


def _naive_prepare(G):
    nodes, _ = cached_nodes_and_A(G, cache_size=1)
    theta = collect_attr(G, nodes, ALIAS_THETA, 0.0)
    epi = collect_attr(G, nodes, ALIAS_EPI, 0.0)
    vf = collect_attr(G, nodes, ALIAS_VF, 0.0)
    return theta, epi, vf


def _measure(func: Callable[[], None], loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        func()
    return time.perf_counter() - start


def run():
    """Run the benchmark and print the elapsed times."""
    G = nx.gnp_random_graph(300, 0.1, seed=1)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_EPI, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)

    loops = 5

    vector_graph = G.copy()
    fallback_graph = G.copy()
    fallback_graph.graph["vectorized_dnfr"] = False

    # Warm up caches before timing.
    _prepare_dnfr_data(vector_graph)
    _prepare_dnfr_data(fallback_graph)

    vector_time = _measure(lambda: _prepare_dnfr_data(vector_graph), loops)
    fallback_time = _measure(lambda: _prepare_dnfr_data(fallback_graph), loops)
    naive_time = _measure(lambda: _naive_prepare(G), loops)

    vector_data = _prepare_dnfr_data(vector_graph)
    fallback_data = _prepare_dnfr_data(fallback_graph)

    theta, epi, vf = _naive_prepare(vector_graph)

    print(
        "vectorized: {0:.6f}s | fallback: {1:.6f}s | naive lists: {2:.6f}s".format(
            vector_time, fallback_time, naive_time
        )
    )
    print(
        "vectorized speedup vs fallback: {0:.2f}x".format(
            fallback_time / vector_time if vector_time else float("inf")
        )
    )

    if hasattr(vector_data["theta"], "__class__"):
        print(
            "vectorized types: theta={0}, epi={1}, vf={2}".format(
                type(vector_data["theta"]).__name__,
                type(vector_data["epi"]).__name__,
                type(vector_data["vf"]).__name__,
            )
        )
    if hasattr(fallback_data["theta"], "__class__"):
        print(
            "fallback types: theta={0}, epi={1}, vf={2}".format(
                type(fallback_data["theta"]).__name__,
                type(fallback_data["epi"]).__name__,
                type(fallback_data["vf"]).__name__,
            )
        )

    # Sanity check: the semantics match across collectors.
    import numpy as np

    np.testing.assert_allclose(vector_data["theta"], theta)
    np.testing.assert_allclose(vector_data["epi"], epi)
    np.testing.assert_allclose(vector_data["vf"], vf)
    np.testing.assert_allclose(fallback_data["theta"], theta)
    np.testing.assert_allclose(fallback_data["epi"], epi)
    np.testing.assert_allclose(fallback_data["vf"], vf)


if __name__ == "__main__":
    run()
