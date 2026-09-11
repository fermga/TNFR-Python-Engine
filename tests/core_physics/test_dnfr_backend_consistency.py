"""The backend must preserve weighted structural pressure, independent of νf.

Deterministic initial data exercise the EPI-channel identity ΔNFR = -L_rw EPI.
Frequency belongs to dEPI/dt = νf ΔNFR, not to a second scaling of pressure.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.backends.optimized_numpy import OptimizedNumPyBackend
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import default_compute_delta_nfr


def _weighted_graph(graph_type, size, frequency):
    graph = graph_type()
    graph.add_nodes_from(range(size))
    # The mean weight is 1, but the weighted neighbor mean is not arithmetic.
    graph.add_weighted_edges_from([(0, 1, 0.5), (0, 2, 1.5)])
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=2.0)
    for node in graph:
        graph.nodes[node].update(EPI=float(node), theta=0.0, vf=frequency)
    graph.graph["_dnfr_weights"] = dict(phase=0.0, epi=1.0, vf=0.0, topo=0.0)
    graph.graph["DNFR_WEIGHTS"] = dict(phase=0.0, epi=1.0, vf=0.0, topo=0.0)
    return graph


def _expected_pressure(graph):
    adjacency = nx.to_numpy_array(graph, nodelist=list(graph), weight="weight")
    degree = adjacency.sum(axis=1)
    epi = np.array([graph.nodes[node]["EPI"] for node in graph])
    mean = np.divide(adjacency @ epi, degree, out=epi.copy(), where=degree > 0)
    return mean - epi


def _pressure(graph):
    return np.array([get_attr(graph.nodes[node], ALIAS_DNFR, 0.0) for node in graph])


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("frequency", [0.0, 2.0])
def test_canonical_balanced_and_parallel_weights(graph_type, frequency):
    graph = _weighted_graph(graph_type, 3, frequency)
    default_compute_delta_nfr(graph)
    np.testing.assert_allclose(_pressure(graph), _expected_pressure(graph), atol=1e-12)


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("size", [3, 100])
@pytest.mark.parametrize("frequency", [0.0, 2.0])
def test_optimized_backend_preserves_pressure(graph_type, size, frequency):
    graph = _weighted_graph(graph_type, size, frequency)
    epi_before = [graph.nodes[node]["EPI"] for node in graph]
    OptimizedNumPyBackend().compute_delta_nfr(graph)
    np.testing.assert_allclose(_pressure(graph), _expected_pressure(graph), atol=1e-12)
    assert [graph.nodes[node]["EPI"] for node in graph] == epi_before


def test_balanced_weight_mutation_is_observed():
    graph = _weighted_graph(nx.DiGraph, 3, 1.0)
    graph[0][1]["weight"] = graph[0][2]["weight"] = 1.0
    default_compute_delta_nfr(graph)
    before = _pressure(graph)
    graph[0][1]["weight"], graph[0][2]["weight"] = 0.5, 1.5
    default_compute_delta_nfr(graph)
    assert not np.array_equal(_pressure(graph), before)
    np.testing.assert_allclose(_pressure(graph), _expected_pressure(graph), atol=1e-12)
