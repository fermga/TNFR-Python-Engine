"""Read-only fields must implement the same neighborhood means on every graph."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from tnfr.physics.extended import compute_dnfr_flux, compute_phase_current
from tnfr.physics.telemetry import compute_structural_telemetry


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_fields_match_unique_neighbor_means(graph_type):
    """Successors, loops and parallel edges use the documented G.neighbors set."""
    graph = graph_type()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 0), (0, 1), (0, 1), (1, 2)])
    phases = {0: 0.1, 1: 2 * math.pi - 0.2, 2: 0.8, 3: 1.5}
    pressure = {0: 0.2, 1: 0.5, 2: 1.0, 3: 0.3}
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, phases[node])
        set_attr(graph.nodes[node], ALIAS_DNFR, pressure[node])

    expected = {key: {} for key in ("grad_phi", "curv_phi", "j_phi", "j_dnfr")}
    for node in graph:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            for field in expected.values():
                field[node] = 0.0
            continue
        differences = np.array([phases[other] - phases[node] for other in neighbors])
        wrapped = (differences + math.pi) % (2 * math.pi) - math.pi
        mean_phase = np.angle(np.mean(np.exp(1j * np.array([phases[n] for n in neighbors]))))
        expected["grad_phi"][node] = float(np.mean(np.abs(wrapped)))
        expected["curv_phi"][node] = (phases[node] - mean_phase + math.pi) % (2 * math.pi) - math.pi
        expected["j_phi"][node] = float(np.mean(np.sin(wrapped)))
        expected["j_dnfr"][node] = float(np.mean([pressure[n] - pressure[node] for n in neighbors]))

    standalone = {
        "grad_phi": compute_phase_gradient(graph),
        "curv_phi": compute_phase_curvature(graph),
        "j_phi": compute_phase_current(graph),
        "j_dnfr": compute_dnfr_flux(graph),
    }
    telemetry = compute_structural_telemetry(graph)
    for key in expected:
        assert standalone[key] == pytest.approx(expected[key], abs=1e-12), key
        assert telemetry[key] == pytest.approx(expected[key], abs=1e-12), key


def test_uniform_pressure_has_zero_directed_flux():
    """An equilibrium pressure field cannot acquire a sink from normalization."""
    graph = nx.DiGraph([(0, 1), (1, 2)])
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.5)
    assert compute_dnfr_flux(graph) == pytest.approx({node: 0.0 for node in graph})


def test_disconnected_potential_has_no_cross_component_source():
    """Unreachable sources contribute exactly zero, without a finite sentinel."""
    graph = nx.empty_graph(2)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.5)
    assert compute_structural_potential(graph) == {0: 0.0, 1: 0.0}


def test_landmark_potential_is_zero_at_equilibrium():
    """The landmark correction excludes self-interaction even for 0/0."""
    graph = nx.path_graph(12)
    assert compute_structural_potential(graph, landmark_ratio=0.5) == {
        node: 0.0 for node in graph
    }
