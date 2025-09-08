"""Benchmark for neighbor_phase_mean performance."""

import time
import math
import networkx as nx
import pytest

from tnfr.helpers.numeric import neighbor_phase_mean
from tnfr.constants import ALIAS_THETA
from tnfr.node import NodoNX


def _naive_neighbor_phase_mean(G, n):
    """Reference implementation using NodoNX wrappers for each neighbour."""
    node = NodoNX(G, n)
    x = y = 0.0
    count = 0
    for v in node.neighbors():
        th = NodoNX.from_graph(node.G, v).theta
        x += math.cos(th)
        y += math.sin(th)
        count += 1
    if count == 0:
        return node.theta
    return math.atan2(y, x)


@pytest.mark.slow
def test_neighbor_phase_mean_performance():
    """Optimised neighbour mean should be faster than naive version."""
    G = nx.gnp_random_graph(200, 0.2, seed=1)
    for n in G.nodes:
        G.nodes[n][ALIAS_THETA] = 0.0

    start = time.perf_counter()
    for _ in range(5):
        for n in G.nodes:
            neighbor_phase_mean(G, n)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(5):
        for n in G.nodes:
            _naive_neighbor_phase_mean(G, n)
    t_naive = time.perf_counter() - start

    assert t_opt <= t_naive
