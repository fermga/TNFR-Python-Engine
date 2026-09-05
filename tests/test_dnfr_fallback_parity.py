"""Execution paths must preserve the same nodal-pressure channels."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr


def _pressure(graph, *, vectorized, n_jobs=None):
    copy = graph.copy()
    copy.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(copy, n_jobs=n_jobs)
    return np.array([get_attr(copy.nodes[node], ALIAS_DNFR) for node in copy])


@pytest.mark.parametrize("channel", ["phase", "epi", "vf", "topo"])
@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_fallback_matches_fused_channels(graph_type, channel):
    """Weighted successors, loops, parallel arcs and sinks retain semantics."""
    graph = graph_type()
    graph.add_nodes_from(range(5))
    graph.add_weighted_edges_from([(0, 0, 1.0), (0, 1, 0.5), (0, 2, 1.5), (0, 1, 0.25), (1, 3, 0.0)])
    graph.graph["_dnfr_weights"] = {key: float(key == channel) for key in ("phase", "epi", "vf", "topo")}
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_EPI, node * 0.25)
        set_attr(graph.nodes[node], ALIAS_THETA, node * 0.2)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0 + node * 0.1)
    expected = _pressure(graph, vectorized=True)
    assert _pressure(graph, vectorized=False) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n_jobs", [None, 2])
@pytest.mark.parametrize("pure_python", [False, True])
def test_fallback_weighted_epi_matches_random_walk_laplacian(n_jobs, pure_python, monkeypatch):
    """EPI pressure is D^-1 W EPI - EPI; outgoing-isolated nodes have zero."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 0.5), (0, 2, 1.5)])
    graph.graph["_dnfr_weights"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_EPI, float(node))
        set_attr(graph.nodes[node], ALIAS_THETA, 0.0)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0)
    if pure_python:
        import tnfr.dynamics.dnfr as dnfr_module
        import tnfr.mathematics.unified_numerical as numerical_module

        monkeypatch.setattr(dnfr_module, "np", None)
        monkeypatch.setattr(numerical_module, "np", None)
        monkeypatch.setattr(numerical_module, "NUMPY_AVAILABLE", False)
    assert _pressure(graph, vectorized=False, n_jobs=n_jobs) == pytest.approx([1.75, 0.0, 0.0])
