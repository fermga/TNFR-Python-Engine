"""Backend selection must preserve structural pressure before capacity scaling."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("torch")
    from tnfr.backends.torch_backend import TorchBackend

    return TorchBackend()


def _graph(graph_type, size, frequency):
    graph = graph_type()
    graph.add_nodes_from(range(size))
    graph.add_weighted_edges_from([(0, 1, 0.5), (0, 2, 1.5), (1, 1, 2.0)])
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=2.0)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_EPI, float(node % 7))
        set_attr(graph.nodes[node], ALIAS_THETA, 0.0)
        set_attr(graph.nodes[node], ALIAS_VF, frequency)
        set_attr(graph.nodes[node], ALIAS_DNFR, 113.0)
    weights = dict(phase=0.0, epi=1.0, vf=0.0, topo=0.0)
    graph.graph["DNFR_WEIGHTS"] = weights.copy()
    graph.graph["_dnfr_weights"] = weights.copy()
    return graph


def _pressure(graph):
    return np.array([get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph])


def _edge_pressure(graph):
    """Small exact-dyadic fixtures use outgoing normalized edge differences."""
    expected = np.zeros(len(graph))
    for node, neighbors in graph.adj.items():
        weighted = []
        for neighbor, data in neighbors.items():
            weight = (
                sum(edge.get("weight", 1.0) for edge in data.values())
                if graph.is_multigraph() else data.get("weight", 1.0)
            )
            weighted.append((neighbor, weight))
        strength = sum(weight for _, weight in weighted)
        if strength:
            expected[node] = sum(
                weight * (
                    get_attr(graph.nodes[neighbor], ALIAS_EPI)
                    - get_attr(graph.nodes[node], ALIAS_EPI)
                ) for neighbor, weight in weighted
            ) / strength
    return expected


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("size", [999, 1000])
@pytest.mark.parametrize("frequency", [0.0, 2.0])
def test_graph_size_does_not_change_weighted_pressure(backend, graph_type, size, frequency):
    graph = _graph(graph_type, size, frequency)
    before = [
        tuple(get_attr(graph.nodes[n], alias) for alias in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA))
        for n in graph
    ]
    backend.compute_delta_nfr(graph)
    np.testing.assert_allclose(_pressure(graph), _edge_pressure(graph), atol=1e-12, rtol=0)
    after = [
        tuple(get_attr(graph.nodes[n], alias) for alias in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA))
        for n in graph
    ]
    assert after == before


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_all_pressure_channels_match_canonical_dispatch(backend, graph_type):
    graph = _graph(graph_type, 1000, 1.0)
    rng = np.random.default_rng(7)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, float(rng.uniform(-1.0, 1.0)))
        set_attr(graph.nodes[node], ALIAS_VF, float(rng.uniform(0.1, 2.0)))
    weights = dict(phase=0.25, epi=0.25, vf=0.25, topo=0.25)
    graph.graph.update(DNFR_WEIGHTS=weights.copy(), _dnfr_weights=weights.copy())
    reference = graph.copy()
    default_compute_delta_nfr(reference)
    backend.compute_delta_nfr(graph)
    np.testing.assert_array_equal(_pressure(graph), _pressure(reference))


def test_large_edgeless_graph_clears_existing_pressure_aliases(backend):
    graph = _graph(nx.Graph, 1000, 1.0)
    graph.remove_edges_from(list(graph.edges))
    backend.compute_delta_nfr(graph)
    np.testing.assert_array_equal(_pressure(graph), np.zeros(len(graph)))


def test_profile_reports_the_executing_kernel_device(backend):
    graph = _graph(nx.DiGraph, 1000, 2.0)
    profile = {}
    backend.compute_delta_nfr(graph, cache_size=2, profile=profile)
    assert profile["dnfr_backend"] == "torch"
    assert profile["dnfr_device"] == "cpu"
    assert profile["dnfr_implementation"] == "canonical"
    assert profile["dnfr_path"] != "torch_gpu"
    assert backend.supports_gpu is False
