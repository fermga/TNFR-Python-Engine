"""Integration coverage for the pure-Python Si path on large graphs."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def _build_large_graph(node_count: int = 640) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    for idx in range(node_count):
        graph.add_edge(idx, (idx + 1) % node_count)
        graph.add_edge(idx, (idx + 7) % node_count)

    for node in graph.nodes:
        theta = (node % 48) * (math.pi / 24)
        vf = 0.4 + 0.015 * ((node * 5) % 37)
        dnfr = 0.2 + 0.01 * ((node * 3) % 29)
        set_attr(graph.nodes[node], ALIAS_THETA, theta)
        set_attr(graph.nodes[node], ALIAS_VF, vf)
        set_attr(graph.nodes[node], ALIAS_DNFR, dnfr)

    graph.graph["SI_CHUNK_SIZE"] = 32
    return graph


def test_parallel_si_matches_sequential_for_large_graph(monkeypatch):
    graph = _build_large_graph()

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    instances: list["_ImmediateExecutor"] = []

    class _ImmediateExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self.chunks: list[
                tuple[tuple[Any, tuple[Any, ...], float, float, float], ...]
            ] = []
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, chunk, **kwargs):
            self.chunks.append(chunk)
            return _ImmediateFuture(func(chunk, **kwargs))

    monkeypatch.setattr("tnfr.metrics.sense_index.ProcessPoolExecutor", _ImmediateExecutor)

    reference = compute_Si(graph, inplace=False)
    parallel = compute_Si(graph, inplace=False, n_jobs=4)

    nodes = sorted(reference)
    ref_values = [reference[n] for n in nodes]
    par_values = [parallel[n] for n in nodes]
    assert par_values == pytest.approx(ref_values, rel=1e-12, abs=1e-12)

    assert instances, "parallel path should instantiate the executor"
    chunk_lengths = [len(chunk) for chunk in instances[0].chunks]
    assert len(chunk_lengths) > 1
    assert all(0 < length <= graph.graph["SI_CHUNK_SIZE"] for length in chunk_lengths)
