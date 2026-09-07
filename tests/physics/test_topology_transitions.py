"""Tests for descriptive nodal-topology transition certificates."""

from __future__ import annotations

import copy
import math
import warnings

import networkx as nx
import pytest

from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.physics import (
    capture_nodal_topology_snapshot,
    detect_nodal_topology_transitions,
)
from tnfr.physics.fields import classify_nodal_topology


def _configured(graph: nx.Graph, *, offset: float = 0.0) -> nx.Graph:
    result = graph.copy()
    for node in result:
        value = float(node)
        result.nodes[node][ALIAS_EPI[0]] = 0.2 * value + offset
        result.nodes[node][ALIAS_VF[0]] = 0.8 + 0.03 * value
        result.nodes[node][ALIAS_THETA[0]] = 0.13 * value + 0.02 * offset
        pressure = ((-1.0) ** int(value)) * (0.05 + 0.01 * value) + offset
        result.nodes[node][ALIAS_DNFR[0]] = pressure
        result.nodes[node][ALIAS_DEPI[0]] = (0.8 + 0.03 * value) * pressure
    return result


def test_detects_annular_radial_and_multinodal_changes_without_interpolation():
    graphs = (
        _configured(nx.cycle_graph(10)),
        _configured(nx.star_graph(9), offset=0.02),
        _configured(nx.barbell_graph(5, 0), offset=0.04),
    )

    certificate = detect_nodal_topology_transitions(
        graphs, times=(0.0, 0.25, 1.0)
    )

    assert certificate.label_sequence == ("annular", "radial", "multinodal")
    assert certificate.transition_indices == (1, 2)
    assert len(certificate.steps) == 2
    assert certificate.transitions == certificate.steps
    assert certificate.predictive_status == "descriptive_finite_sequence_only"

    first = certificate.transitions[0]
    assert (first.before_time, first.after_time) == (0.0, 0.25)
    assert first.common_nodes == tuple(range(10))
    assert not first.node_support_changed
    assert first.node_alignment_status == "persistent_equal_node_ids"
    assert first.edge_support_changed
    assert first.before_topology == "annular"
    assert first.after_topology == "radial"
    assert first.centers_changed
    assert any(abs(value) > 0.0 for value in first.centrality_delta)
    assert any(abs(value) > 0.0 for value in first.phase_gradient_delta)
    assert first.coherence_delta == pytest.approx(
        certificate.snapshots[1].coherence - certificate.snapshots[0].coherence
    )
    assert first.coherence_length_delta == pytest.approx(
        certificate.snapshots[1].coherence_length
        - certificate.snapshots[0].coherence_length
    )


def test_state_changes_do_not_create_a_geometry_label_transition():
    before = _configured(nx.cycle_graph(8))
    after = _configured(nx.cycle_graph(8), offset=0.4)

    certificate = detect_nodal_topology_transitions((before, after))

    assert certificate.label_sequence == ("annular", "annular")
    assert certificate.transitions == ()
    assert certificate.transition_indices == ()
    assert not certificate.steps[0].topology_changed
    assert certificate.steps[0].centrality_delta == pytest.approx((0.0,) * 8)
    assert any(
        abs(value) > 0.0
        for value in certificate.steps[0].structural_potential_delta
    )
    assert certificate.steps[0].coherence_delta != 0.0


def test_pointwise_deltas_use_only_common_nodes_and_report_support_changes():
    before = _configured(nx.path_graph(4))
    after = _configured(nx.path_graph(5), offset=0.1)

    certificate = detect_nodal_topology_transitions((before, after))
    step = certificate.steps[0]

    assert step.common_nodes == (0, 1, 2, 3)
    assert step.added_nodes == (4,)
    assert step.removed_nodes == ()
    assert step.node_support_changed
    assert step.node_count_delta == 1
    assert step.added_edges == ((3, 4),)
    assert step.edge_support_changed
    assert len(step.structural_potential_delta) == 4


def test_exact_relabeling_is_aligned_without_spurious_structural_change():
    before = _configured(nx.path_graph(5))
    relabeling = {node: f"node-{node}" for node in before}
    after = nx.relabel_nodes(before, relabeling, copy=True)

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert dict(step.node_mapping) == relabeling
    assert step.node_alignment_status == "exact_state_isomorphism"
    assert step.node_alignment_candidate_count == 1
    assert not step.node_alignment_ambiguous
    assert step.mapping_dependent_deltas_available
    assert not step.node_support_changed
    assert not step.edge_support_changed
    assert not step.edge_weight_changed
    assert not step.centers_changed
    for values in (
        step.centrality_delta,
        step.structural_potential_delta,
        step.phase_gradient_delta,
        step.phase_curvature_delta,
        step.phase_current_delta,
        step.dnfr_flux_delta,
        step.sense_index_delta,
    ):
        assert values == pytest.approx((0.0,) * len(before), abs=1e-14)


def test_ambiguous_relabeling_suppresses_nonidentifiable_pointwise_deltas():
    before = _configured(nx.path_graph(3))
    after = nx.relabel_nodes(
        before, {node: f"node-{node}" for node in before}, copy=True
    )
    after.nodes["node-0"][ALIAS_EPI[0]] += 0.7
    after.nodes["node-0"][ALIAS_THETA[0]] += 0.4

    reordered = nx.Graph()
    reordered.add_nodes_from(
        (node, copy.deepcopy(after.nodes[node]))
        for node in reversed(tuple(after))
    )
    reordered.add_edges_from(
        (source, target, copy.deepcopy(data))
        for source, target, data in reversed(tuple(after.edges(data=True)))
    )

    first = detect_nodal_topology_transitions((before, after)).steps[0]
    second = detect_nodal_topology_transitions((before, reordered)).steps[0]

    for step in (first, second):
        assert step.node_alignment_status == "ambiguous_bare_topology_isomorphism"
        assert step.node_alignment_candidate_count == 2
        assert step.node_alignment_ambiguous
        assert not step.mapping_dependent_deltas_available
        assert step.node_mapping == ()
        assert step.common_nodes == ()
        assert not step.node_support_changed
        assert not step.edge_support_changed
        assert step.centers_changed is None
        assert step.edge_weight_changed is None
        for values in (
            step.centrality_delta,
            step.structural_potential_delta,
            step.phase_gradient_delta,
            step.phase_curvature_delta,
            step.phase_current_delta,
            step.dnfr_flux_delta,
            step.sense_index_delta,
            step.edge_conductance_delta,
        ):
            assert values == ()


def test_exact_state_alignment_includes_all_five_nodal_channels():
    before = _configured(nx.path_graph(3))
    after = nx.relabel_nodes(
        before, {node: f"node-{node}" for node in before}, copy=True
    )
    after.nodes["node-0"][ALIAS_EPI[0]] += 1.0

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert "exact_state" not in step.node_alignment_status
    assert step.node_alignment_status == "ambiguous_bare_topology_isomorphism"
    assert step.node_alignment_candidate_count == 2


def test_unrepresentable_pointwise_delta_is_rejected():
    before = nx.path_graph(2)
    after = nx.path_graph(2)
    for graph, pressure in ((before, -1.0e308), (after, 1.0e308)):
        for node in graph:
            graph.nodes[node].update(
                EPI=0.0,
                nu_f=0.0,
                theta=0.0,
                delta_nfr=pressure,
                dEPI_dt=0.0,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="pointwise field delta exceeds"):
            detect_nodal_topology_transitions((before, after))


def test_edge_rewiring_is_reported_even_when_edge_count_is_unchanged():
    before = _configured(nx.path_graph(4))
    rewired = nx.Graph([(0, 1), (0, 2), (2, 3)])
    after = _configured(rewired)

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert step.edge_count_delta == 0
    assert step.edge_support_changed
    assert step.removed_edges == ((1, 2),)
    assert step.added_edges == ((0, 2),)
    assert not step.edge_weight_changed


def test_edge_weight_change_is_separate_from_edge_support_change():
    before = _configured(nx.path_graph(3))
    after = before.copy()
    before.edges[0, 1]["weight"] = 0.5
    after.edges[0, 1]["weight"] = 1.25
    before.edges[0, 1]["length"] = 1.0
    after.edges[0, 1]["length"] = 1.0

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert not step.edge_support_changed
    assert step.edge_weight_changed
    assert not step.edge_length_changed
    assert step.edge_conductance_delta == pytest.approx((0.75, 0.0))


def test_edge_length_change_is_separate_from_conductance_and_support():
    before = _configured(nx.path_graph(3))
    after = before.copy()
    before.edges[0, 1].update(weight=3.0, length=1.0)
    after.edges[0, 1].update(weight=3.0, length=2.0)

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert not step.edge_support_changed
    assert not step.edge_weight_changed
    assert step.edge_length_changed
    assert step.edge_length_delta == pytest.approx((1.0, 0.0))


def test_removing_the_only_center_is_reported_as_a_center_change():
    before = _configured(nx.star_graph(4))
    after = _configured(nx.cycle_graph([1, 2, 3, 4, 5]))

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert step.before_centers == (0,)
    assert step.after_centers == ()
    assert step.centers_changed


def test_phase_curvature_delta_uses_signed_circular_difference():
    before = _configured(nx.path_graph(2))
    after = before.copy()
    epsilon = 0.01
    before.nodes[0][ALIAS_THETA[0]] = math.pi - epsilon
    before.nodes[1][ALIAS_THETA[0]] = 0.0
    after.nodes[0][ALIAS_THETA[0]] = -math.pi + epsilon
    after.nodes[1][ALIAS_THETA[0]] = 0.0

    step = detect_nodal_topology_transitions((before, after)).steps[0]

    assert step.phase_curvature_delta == pytest.approx(
        (2.0 * epsilon, -2.0 * epsilon), abs=1e-12
    )
    assert max(abs(value) for value in step.phase_curvature_delta) <= math.pi


def test_capture_is_reproducible_and_does_not_mutate_nodal_or_edge_state():
    graph = _configured(nx.star_graph(7))
    nodes_before = copy.deepcopy(dict(graph.nodes(data=True)))
    edges_before = tuple(graph.edges(data=True))

    first = capture_nodal_topology_snapshot(graph, index=3, time=1.25)
    second = capture_nodal_topology_snapshot(graph, index=3, time=1.25)

    assert first == second
    assert dict(graph.nodes(data=True)) == nodes_before
    assert tuple(graph.edges(data=True)) == edges_before
    assert first.topology == "radial"
    assert first.node_order == tuple(range(8))
    assert len(first.phase_curvature) == graph.number_of_nodes()
    assert math.isfinite(first.sense_index_mean)


@pytest.mark.parametrize(
    "alpha", [0.0, -1.0, 0.1, 1.0, 2.001, math.inf, math.nan, True, "invalid"]
)
def test_invalid_classifier_exponents_are_rejected(alpha):
    with pytest.raises(ValueError, match="alpha=2.0"):
        capture_nodal_topology_snapshot(nx.cycle_graph(4), alpha=alpha)


def test_canonical_classifier_and_transition_certificate_fix_inverse_square_kernel():
    graph = nx.star_graph(4)

    assert classify_nodal_topology(graph, alpha=2.0)["topology"] == "radial"
    with pytest.raises(ValueError, match="alpha=2.0"):
        classify_nodal_topology(graph, alpha=0.1)
    with pytest.raises(ValueError, match="alpha=2.0"):
        detect_nodal_topology_transitions((graph,), alpha=0.1)


def test_sequence_and_time_validation_are_explicit():
    graph = nx.cycle_graph(4)

    with pytest.raises(ValueError, match="at least one"):
        detect_nodal_topology_transitions(())
    with pytest.raises(ValueError, match="one value"):
        detect_nodal_topology_transitions((graph, graph), times=(0.0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_nodal_topology_transitions((graph, graph), times=(1.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        detect_nodal_topology_transitions((graph,), times=(math.nan,))
    with pytest.raises(ValueError, match="finite"):
        detect_nodal_topology_transitions((graph,), times=(None,))
    with pytest.raises(ValueError, match="finite"):
        capture_nodal_topology_snapshot(graph, time=True)
    with pytest.raises(ValueError, match="at least one node"):
        capture_nodal_topology_snapshot(nx.Graph())
    with pytest.raises(ValueError, match="non-negative integer"):
        capture_nodal_topology_snapshot(graph, index=-1)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -1.0, True, "1.0"])
def test_snapshot_rejects_invalid_effective_edge_weights(weight):
    graph = _configured(nx.path_graph(2))
    graph.edges[0, 1]["weight"] = weight

    with pytest.raises(ValueError, match="edge weights"):
        capture_nodal_topology_snapshot(graph)


def test_snapshot_rejects_boolean_nodal_channels_before_telemetry():
    graph = _configured(nx.path_graph(2))
    graph.nodes[0][ALIAS_EPI[0]] = True

    with pytest.raises(ValueError, match="EPI must contain finite real values"):
        capture_nodal_topology_snapshot(graph)


def test_snapshot_rejects_negative_structural_frequency():
    graph = _configured(nx.path_graph(2))
    graph.nodes[0][ALIAS_VF[0]] = -0.1

    with pytest.raises(ValueError, match="frequency must contain nonnegative"):
        capture_nodal_topology_snapshot(graph)


def test_snapshot_rejects_invalid_explicit_edge_length():
    graph = _configured(nx.path_graph(2))
    graph.edges[0, 1]["length"] = -1.0

    with pytest.raises(ValueError, match="edge lengths"):
        capture_nodal_topology_snapshot(graph)


def test_single_snapshot_has_no_invented_transition():
    certificate = detect_nodal_topology_transitions(
        (_configured(nx.star_graph(5)),), times=(2.0,)
    )

    assert certificate.label_sequence == ("radial",)
    assert certificate.steps == ()
    assert certificate.transitions == ()
    assert certificate.transition_indices == ()
