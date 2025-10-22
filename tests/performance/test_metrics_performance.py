"""Performance regression coverage for metric helpers."""

from __future__ import annotations

import math
import time

import networkx as nx
import pytest

np = pytest.importorskip("numpy")
import numpy.testing as npt

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.trig import neighbor_phase_mean
from tnfr.node import NodeNX

pytestmark = pytest.mark.slow

ALIAS_THETA = get_aliases("THETA")


def _seed_graph(
    num_nodes: int = 220, edge_probability: float = 0.25, *, seed: int = 21
) -> nx.Graph:
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        set_attr(graph.nodes[node], ALIAS_THETA, 0.1 * (node + 1))
    return graph


def _naive_neighbor_phase_mean(G: nx.Graph, node) -> float:
    wrapper = NodeNX(G, node)
    x = y = 0.0
    count = 0
    for neighbor in wrapper.neighbors():
        th = NodeNX.from_graph(wrapper.G, neighbor).theta
        x += math.cos(th)
        y += math.sin(th)
        count += 1
    if count == 0:
        return wrapper.theta
    return math.atan2(y, x)


def _measure(callback, loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        callback()
    return time.perf_counter() - start


def test_neighbor_phase_mean_vectorized_outperforms_naive_wrapper():
    graph_fast = _seed_graph()
    graph_slow = graph_fast.copy()

    def run_fast() -> None:
        for node in graph_fast.nodes:
            neighbor_phase_mean(graph_fast, node)

    def run_slow() -> None:
        for node in graph_slow.nodes:
            _naive_neighbor_phase_mean(graph_slow, node)

    # Warm caches to avoid measuring import overhead.
    run_fast()
    run_slow()

    loops = 5
    fast_time = _measure(run_fast, loops)
    slow_time = _measure(run_slow, loops)

    assert fast_time < slow_time
    assert fast_time <= slow_time * 0.5

    fast_angles = [neighbor_phase_mean(graph_fast, node) for node in graph_fast.nodes]
    slow_angles = [
        _naive_neighbor_phase_mean(graph_slow, node) for node in graph_slow.nodes
    ]
    npt.assert_allclose(fast_angles, slow_angles, rtol=1e-9, atol=1e-9)
