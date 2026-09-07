"""Boundary regressions for circular phase semantics outside operator modules."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.dynamics.emergent_centralization import (
    TNFREmergentCentralizationEngine,
)
from tnfr.dynamics.propagation import propagate_dissonance
from tnfr.mathematics import number_theory
from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork
from tnfr.multiscale.hierarchical import (
    HierarchicalTNFRNetwork,
    ScaleDefinition,
)
from tnfr.physics.integrity import _postcond_transition
from tnfr.riemann.delta_phi_max_type_signature import (
    _u3_scalar_verdict,
    _wrapped_abs_diff,
)
from tnfr.validation.invariants import Invariant5_ExplicitPhaseChecks


def test_phase_centralization_clusters_neighbors_across_wrap() -> None:
    graph = nx.star_graph(3)
    phases = (0.01, math.tau - 0.01, 0.02, math.tau - 0.02)
    for node, phase in enumerate(phases):
        graph.nodes[node].update(theta=phase, nu_f=1.0)

    coordination_nodes = (
        TNFREmergentCentralizationEngine()
        .analyze_phase_coordination_centralization(graph)
    )

    assert [item.node_id for item in coordination_nodes] == [0]
    assert coordination_nodes[0].connected_cluster == [0, 1, 2, 3]
    assert coordination_nodes[0].mathematical_signature[
        "average_phase_difference"
    ] == pytest.approx(0.02)


def test_cross_scale_synchrony_uses_shortest_arc_at_branch_cut() -> None:
    hierarchy = HierarchicalTNFRNetwork(
        [
            ScaleDefinition("micro", 1, 1.0),
            ScaleDefinition("macro", 1, 1.0),
        ],
        seed=7,
        parallel=False,
    )
    hierarchy.networks_by_scale["micro"].nodes[0]["phase"] = math.pi - 0.01
    hierarchy.networks_by_scale["macro"].nodes[0]["phase"] = math.pi + 0.01

    synchrony = hierarchy._compute_cross_scale_synchrony()

    assert synchrony == pytest.approx(1.0 - 0.02 / math.pi)


def test_dissonance_propagates_to_phase_neighbor_across_wrap() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0].update(theta=0.01, nu_f=1.0, delta_nfr=0.0)
    graph.nodes[1].update(
        theta=math.tau - 0.01,
        nu_f=1.0,
        delta_nfr=0.0,
    )

    affected = propagate_dissonance(graph, 0, 0.2)

    assert affected == {1}
    event = graph.nodes[1]["_oz_propagation"][0]
    assert event["phase_weight"] == pytest.approx(1.0 - 0.02 / (math.pi / 2.0))


def test_transition_postcondition_rejects_phase_only_full_turn() -> None:
    before = {"vf": 1.0, "theta": 0.0, "dnfr": 0.0}
    after = {"vf": 1.0, "theta": math.tau, "dnfr": 0.0}

    violation = _postcond_transition(nx.Graph(), 0, before, after)

    assert violation is not None
    assert "all unchanged" in violation


def test_phase_invariant_detects_antiphase_multiturn_representative() -> None:
    graph = nx.Graph([(0, 1)])
    graph.nodes[0]["theta"] = 4.0 * math.pi
    graph.nodes[1]["theta"] = math.pi

    violations = Invariant5_ExplicitPhaseChecks().validate(graph)
    edge_violations = [
        item
        for item in violations
        if item.description == "Large phase difference between coupled nodes"
    ]

    assert len(edge_violations) == 1
    assert edge_violations[0].actual_value == pytest.approx(math.pi)


def test_type_signature_u3_rejects_multiturn_antiphase() -> None:
    theta_i = 4.0 * math.pi
    theta_j = math.pi

    assert _wrapped_abs_diff(theta_i, theta_j) == pytest.approx(math.pi)
    assert not _u3_scalar_verdict(theta_i, theta_j, math.pi / 2.0)


def test_arithmetic_network_u3_rejects_multiturn_antiphase() -> None:
    network = ArithmeticTNFRNetwork(max_number=4)
    network.graph.nodes[2]["phi"] = 4.0 * math.pi
    network.graph.nodes[4]["phi"] = math.pi

    coupled = network.apply_coupling(delta_phi_max=math.pi / 2.0)

    assert coupled[(2, 4)] is False
    assert network._neighbor_contrib(2, delta_phi_max=math.pi / 2.0) == []


def test_arithmetic_phase_gradient_fallback_wraps_multiturn_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = ArithmeticTNFRNetwork(max_number=4)
    network.graph.nodes[2]["phi"] = 4.0 * math.pi
    network.graph.nodes[3]["phi"] = 0.0
    network.graph.nodes[4]["phi"] = math.pi
    monkeypatch.setattr(number_theory, "HAS_CENTRALIZED_FIELDS", False)

    gradient = network.compute_phase_gradient()

    assert gradient[2] == pytest.approx(math.pi)
    assert gradient[4] == pytest.approx(math.pi)
