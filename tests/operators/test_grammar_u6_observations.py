from __future__ import annotations

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.operators.grammar_u6 import (
    observe_structural_potential_confinement,
    structural_potential_change_terms,
    validate_structural_potential_confinement,
)
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.transient_u2 import potential_operator, potential_operator_from_graph


def _weighted_directed_cycle() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 0, 4.0)])
    for node, pressure in enumerate((1.0, -2.0, 3.0)):
        graph.nodes[node][ALIAS_DNFR[0]] = pressure
    return graph


def test_u6_observation_declares_reference_kernel_and_finite_coverage():
    graph = _weighted_directed_cycle()
    reference = compute_structural_potential(graph)
    graph.nodes[1][ALIAS_DNFR[0]] = -1.0
    observed = compute_structural_potential(graph)
    report = observe_structural_potential_confinement(
        graph,
        reference,
        observed,
        reference="before_operator_word",
        time_coverage="two_snapshot_finite_observation",
    )
    valid, drift, _ = validate_structural_potential_confinement(
        graph, reference, observed, strict=False
    )
    assert report.confined is valid
    assert report.mean_absolute_drift == pytest.approx(drift)
    assert report.kernel == (
        "canonical_directed_weighted_shortest_path_inverse_square"
    )
    assert report.reference == "before_operator_word"
    assert report.time_coverage == "two_snapshot_finite_observation"
    assert report.as_dict()["aggregation"] == "mean_absolute_nodewise_drift"


def test_u6_observation_preserves_isolates_and_explicit_reference():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_node(2)
    for node, pressure in enumerate((1.0, 2.0, -3.0)):
        graph.nodes[node][ALIAS_DNFR[0]] = pressure
    field = compute_structural_potential(graph)
    report = observe_structural_potential_confinement(
        graph,
        field,
        field,
        reference="equilibrium_zero_displacement",
    )
    assert report.confined
    assert report.mean_absolute_drift == 0.0
    assert report.reference == "equilibrium_zero_displacement"


def test_legacy_hop_kernel_is_not_the_canonical_directed_weighted_field():
    graph = _weighted_directed_cycle()
    canonical = compute_structural_potential(graph)
    legacy = potential_operator(nx.to_numpy_array(graph, weight="weight"))
    pressure = [graph.nodes[node][ALIAS_DNFR[0]] for node in graph]
    legacy_field = legacy @ pressure
    assert list(canonical.values()) != pytest.approx(legacy_field)


def test_u6_observation_is_available_from_operator_public_api():
    from tnfr.operators import observe_structural_potential_confinement

    assert callable(observe_structural_potential_confinement)


def test_topology_change_decomposition_matches_total_potential_change():
    before = _weighted_directed_cycle()
    after = before.copy()
    after[0][1]["weight"] = 1.0
    before_pressure = [before.nodes[node][ALIAS_DNFR[0]] for node in before]
    after_pressure = [after.nodes[node][ALIAS_DNFR[0]] for node in after]
    _, before_kernel = potential_operator_from_graph(before)
    _, after_kernel = potential_operator_from_graph(after)
    pressure, topology, total = structural_potential_change_terms(
        before_kernel, before_pressure, after_kernel, after_pressure
    )
    before_phi = compute_structural_potential(before)
    after_phi = compute_structural_potential(after)
    expected = [after_phi[node] - before_phi[node] for node in before]
    assert pressure == pytest.approx([0.0, 0.0, 0.0])
    assert topology != pytest.approx([0.0, 0.0, 0.0])
    assert total == pytest.approx(expected)


def test_topology_change_decomposition_requires_aligned_nodes():
    with pytest.raises(ValueError):
        structural_potential_change_terms(
            [[0.0, 1.0], [1.0, 0.0]], [1.0, 2.0], [[0.0]], [1.0]
        )
