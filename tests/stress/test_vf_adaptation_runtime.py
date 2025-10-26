"""Stress tests for νf adaptation over large, reproducible graphs."""

from __future__ import annotations

import math
import time
from collections import Counter

import pytest

pytest.importorskip("numpy")

from tnfr.alias import get_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import adapt_vf_by_coherence

from .test_dnfr_runtime import _seed_graph

ALIAS_VF = get_aliases("VF")
ALIAS_SI = get_aliases("SI")
ALIAS_DNFR = get_aliases("DNFR")

pytestmark = [pytest.mark.slow, pytest.mark.stress]


def _prepare_graph(
    *,
    node_count: int,
    edge_probability: float,
    seed: int,
    tau: int,
    mu: float,
):
    """Return a seeded graph with stable νf adaptation parameters."""

    graph = _seed_graph(
        num_nodes=node_count,
        edge_probability=edge_probability,
        seed=seed,
    )
    graph.graph["VF_ADAPT_TAU"] = tau
    graph.graph["VF_ADAPT_MU"] = mu

    for node, data in graph.nodes(data=True):
        base = seed + int(node)
        stability = 0.82 + 0.15 * ((math.sin(base * 0.017) + 1.0) / 2.0)
        set_attr(data, ALIAS_SI, stability)
        set_attr(data, ALIAS_DNFR, 1e-5 * math.cos(base * 0.013))
        data["stable_count"] = tau - 1

    return graph


@pytest.mark.timeout(30)
def test_adapt_vf_by_coherence_large_graph_consistent_across_workers() -> None:
    """Serial and parallel νf adaptation must stay deterministic on large graphs."""

    node_count = 2048
    edge_probability = 0.0065
    seed = 90210
    tau = 4
    mu = 0.35

    serial_graph = _prepare_graph(
        node_count=node_count,
        edge_probability=edge_probability,
        seed=seed,
        tau=tau,
        mu=mu,
    )
    parallel_graph = _prepare_graph(
        node_count=node_count,
        edge_probability=edge_probability,
        seed=seed,
        tau=tau,
        mu=mu,
    )

    serial_start = time.perf_counter()
    adapt_vf_by_coherence(serial_graph, n_jobs=None)
    serial_elapsed = time.perf_counter() - serial_start

    parallel_start = time.perf_counter()
    adapt_vf_by_coherence(parallel_graph, n_jobs=3)
    parallel_elapsed = time.perf_counter() - parallel_start

    assert serial_elapsed < 30.0
    assert parallel_elapsed < 30.0

    serial_sum = sum(
        float(get_attr(serial_graph.nodes[node], ALIAS_VF, 0.0))
        for node in serial_graph.nodes
    )
    parallel_sum = sum(
        float(get_attr(parallel_graph.nodes[node], ALIAS_VF, 0.0))
        for node in parallel_graph.nodes
    )

    expected_sum = 767.871955640765
    assert serial_sum == pytest.approx(expected_sum, rel=0.0, abs=1e-9)
    assert parallel_sum == pytest.approx(serial_sum, rel=0.0, abs=1e-9)

    serial_counts = Counter(
        int(serial_graph.nodes[node].get("stable_count", -1))
        for node in serial_graph.nodes
    )
    parallel_counts = Counter(
        int(parallel_graph.nodes[node].get("stable_count", -1))
        for node in parallel_graph.nodes
    )

    expected_counts = Counter({tau: node_count})
    assert serial_counts == expected_counts
    assert parallel_counts == expected_counts
    assert parallel_counts == serial_counts
