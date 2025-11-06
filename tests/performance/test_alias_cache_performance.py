"""Performance regression coverage for alias caching utilities."""

from __future__ import annotations

import time

import networkx as nx
import numpy.testing as npt
import pytest

from tnfr.alias import (
    collect_attr,
    get_attr,
    multi_recompute_abs_max,
    set_attr,
    set_attr_with_max,
)
from tnfr.constants import get_aliases

pytestmark = pytest.mark.slow

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_MAP = {"_vfmax": ALIAS_VF}


def _seed_graph(
    num_nodes: int = 280, edge_probability: float = 0.2, *, seed: int = 33
) -> nx.Graph:
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        set_attr(graph.nodes[node], ALIAS_THETA, float(node % 7))
        set_attr(graph.nodes[node], ALIAS_VF, 0.0)
    return graph


def _measure(callback, loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        callback()
    return time.perf_counter() - start


def test_collect_attr_numpy_vectorization_is_significantly_faster():
    np = pytest.importorskip("numpy")

    graph_fast = _seed_graph(seed=41)
    graph_slow = graph_fast.copy()

    def run_fast() -> None:
        collect_attr(graph_fast, graph_fast.nodes, ALIAS_THETA, 0.0, np=np)

    def run_slow() -> None:
        [
            collect_attr(graph_slow, [node], ALIAS_THETA, 0.0)[0]
            for node in graph_slow.nodes
        ]

    run_fast()
    run_slow()

    loops = 6
    fast_time = _measure(run_fast, loops)
    slow_time = _measure(run_slow, loops)

    assert fast_time < slow_time
    assert fast_time <= slow_time * 0.75

    fast_values = collect_attr(graph_fast, graph_fast.nodes, ALIAS_THETA, 0.0, np=np)
    slow_values = np.array(
        [
            collect_attr(graph_fast, [node], ALIAS_THETA, 0.0)[0]
            for node in graph_fast.nodes
        ],
        dtype=float,
    )
    npt.assert_allclose(fast_values, slow_values, rtol=0.0, atol=0.0)


def test_set_attr_with_max_cache_beats_full_recompute():
    graph_cached = _seed_graph(seed=57)
    graph_naive = graph_cached.copy()

    for node in graph_cached.nodes:
        set_attr_with_max(graph_cached, node, ALIAS_VF, 0.0, cache="_vfmax")
        set_attr(graph_naive.nodes[node], ALIAS_VF, 0.0)
    multi_recompute_abs_max(graph_naive, ALIAS_MAP)

    nodes = list(graph_cached.nodes)
    values = [float(index) for index in range(len(nodes))]

    def run_cached() -> None:
        for node, value in zip(nodes, values):
            set_attr_with_max(graph_cached, node, ALIAS_VF, value, cache="_vfmax")

    def run_full() -> None:
        for node, value in zip(nodes, values):
            set_attr(graph_naive.nodes[node], ALIAS_VF, value)
            multi_recompute_abs_max(graph_naive, ALIAS_MAP)

    run_cached()
    run_full()

    loops = 3
    cached_time = _measure(run_cached, loops)
    full_time = _measure(run_full, loops)

    assert cached_time < full_time
    assert cached_time <= full_time * 0.7

    cached_max = float(graph_cached.graph.get("_vfmax", 0.0))
    cached_node = graph_cached.graph.get("_vfmax_node")
    expected_values = multi_recompute_abs_max(graph_cached, ALIAS_MAP)
    expected_max = float(expected_values["_vfmax"])
    expected_node = max(
        nodes,
        key=lambda node: abs(get_attr(graph_cached.nodes[node], ALIAS_VF, 0.0)),
    )
    assert cached_max == expected_max
    assert cached_node == expected_node
