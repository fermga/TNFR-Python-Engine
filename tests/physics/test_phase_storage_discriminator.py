"""Admission and independent primitive controls for a supplied storage law."""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.relational_phase_storage import assess_saddle_storage_discriminator
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import relational_report_to_dict


def _anchor(*, order=tuple(range(10)), labels=tuple(range(10)), extra_edge=None):
    graph = nx.Graph()
    graph.add_nodes_from(labels[i] for i in order)
    graph.add_edges_from((labels[i], labels[(i + 1) % 5]) for i in range(5))
    graph.add_edges_from((labels[i], labels[i + 5]) for i in range(5))
    if extra_edge is not None:
        graph.add_edge(*(labels[i] for i in extra_edge))
    for i, node in enumerate(labels):
        graph.nodes[node].update(EPI=i / 8, theta=i / 16, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return source, labels[:5]


@pytest.fixture(scope="module")
def report():
    source, cycle = _anchor()
    return assess_saddle_storage_discriminator(
        source, cycle=cycle, initial_coordinate_radius=Q(1, 2**20)
    )


def test_distinct_law_keeps_anchor_preparation_and_full_cube_roles_separate(report):
    assert report.status == "certified" and report.discriminator_certified
    assert report.source.epi != report.prepared_state.epi
    assert report.prepared_state == report.preparation.prepared_state
    assert report.law != report.prepared_state.law
    assert report.prepared_state.law == "normalized_sine_reciprocal_exchange"
    assert len(report.initial_box) == 20
    for value, box in zip(
        report.prepared_state.epi + report.prepared_state.phase, report.initial_box
    ):
        assert box.contains(value - report.initial_coordinate_radius)
        assert box.contains(value + report.initial_coordinate_radius)
    assert report.initial_winding == 1
    assert report.nominal_sine_outer_gate_certified
    assert report.nominal_sine_inner_gate_certified
    assert report.alternative_zero_winding_passage_excluded
    assert report.alternative_winding_invariant_both_time_directions_certified
    assert not report.sine_full_cube_passage_certified
    assert len(report.initial_edge_turn_offsets) == len(report.edge_indices)
    assert len(report.initial_cycle_turn_offsets) == 5
    assert -sum(report.initial_cycle_turn_offsets) == report.initial_winding


def test_full_form_cost_retains_all_receiver_and_environmental_coordinates(report):
    center, delta = report.prepared_state.epi, report.initial_coordinate_radius
    # Choose a cube corner that breaks the signed-family preparation. All ten
    # actual support edges still contribute to its declared form energy.
    forms = tuple(
        value + (delta if i % 3 else -delta) for i, value in enumerate(center)
    )
    assert forms[2] != 0  # the reflection's fixed receiver is no longer zero
    total = sum(((forms[j] - forms[i]) ** 2 / 2 for i, j in report.edge_indices), Q(0))
    ring = set(report.cycle_indices)
    contact = sum(
        (
            (forms[j] - forms[i]) ** 2 / 2
            for i, j in report.edge_indices
            if (i in ring) != (j in ring)
        ),
        Q(0),
    )
    assert contact > 0
    assert report.initial_form_storage_bounds.contains(total)
    assert (
        report.alternative_initial_storage_bounds == report.sine_initial_storage_bounds
    )
    assert (
        report.energy_margin_lower_bound
        == 4 - report.alternative_initial_storage_bounds.hi
        > 0
    )


def test_whole_local_phase_guard_accounts_for_both_endpoints_and_cube_error(report):
    expected = 729 * (
        3 * report.epsilon
        + report.preparation.preparation_error_bound
        + report.initial_coordinate_radius
    )
    assert report.local_phase_radius_bound == expected
    assert report.local_scaled_duration == 3
    assert (
        report.local_agreement_margin_lower_bound
        == pi_interval().lo / 12 - 2 * expected
        > 0
    )
    assert report.whole_local_flow_agreement_certified
    assert report.initial_agreement_certified
    assert report.antipodal_storage == 4


def test_a_large_local_error_does_not_remove_an_independently_certified_global_obstruction(
    report,
):
    result = assess_saddle_storage_discriminator(
        report.source, cycle=report.cycle, initial_coordinate_radius=Q(1, 1000)
    )
    assert result.initial_agreement_certified
    assert not result.whole_local_flow_agreement_certified
    assert result.alternative_zero_winding_passage_excluded
    assert not result.discriminator_certified
    assert result.status == "unavailable" and result.reasons


def test_outside_agreement_chamber_does_not_silently_reuse_sine_storage(report):
    result = assess_saddle_storage_discriminator(
        report.source, cycle=report.cycle, initial_coordinate_radius=Q(1, 2)
    )
    assert not result.initial_agreement_certified
    assert result.alternative_initial_storage_bounds is None
    assert result.energy_margin_lower_bound is None
    assert not result.alternative_zero_winding_passage_excluded
    assert result.status == "unavailable"


def test_ambiguous_initial_branch_cannot_be_promoted_to_winding_invariance(report):
    result = assess_saddle_storage_discriminator(
        report.source, cycle=report.cycle, initial_coordinate_radius=Q(2)
    )
    assert result.initial_winding is None
    assert result.initial_edge_turn_offsets is None
    assert not result.alternative_winding_invariant_both_time_directions_certified


def test_source_cached_evidence_does_not_supply_the_generated_preparation(report):
    poisoned = replace(report.source, storage=I(0), form_rates=(I(999),) * 10)
    result = assess_saddle_storage_discriminator(
        poisoned,
        cycle=report.cycle,
        initial_coordinate_radius=report.initial_coordinate_radius,
    )
    assert result.prepared_state.epi == report.prepared_state.epi
    assert result.prepared_state.phase == report.prepared_state.phase
    assert (
        result.alternative_initial_storage_bounds
        == report.alternative_initial_storage_bounds
    )
    assert result.local_phase_radius_bound == report.local_phase_radius_bound


def test_tiny_recipe_preserves_unresolved_nominal_gate_status(report):
    result = assess_saddle_storage_discriminator(
        report.source,
        cycle=report.cycle,
        initial_coordinate_radius=0,
        epsilon=Q(1, 2**512),
    )
    assert result.alternative_zero_winding_passage_excluded
    assert not result.nominal_sine_outer_gate_certified
    assert not result.discriminator_certified
    assert "nominal_sine_corridor_gates_not_certified" in result.reasons


def test_arbitrary_labels_and_node_order_preserve_the_actual_common_preparation(report):
    labels = tuple(("node", i) for i in range(10))
    source, cycle = _anchor(order=(8, 2, 6, 4, 0, 9, 3, 5, 7, 1), labels=labels)
    result = assess_saddle_storage_discriminator(
        source, cycle=cycle, initial_coordinate_radius=report.initial_coordinate_radius
    )
    for i, label in enumerate(result.prepared_state.nodes):
        assert result.prepared_state.epi[i] == report.prepared_state.epi[label[1]]
        assert result.prepared_state.phase[i] == report.prepared_state.phase[label[1]]
    assert result.sine_initial_storage_bounds == report.sine_initial_storage_bounds
    assert result.initial_winding == report.initial_winding
    assert result.discriminator_certified


@pytest.mark.parametrize("radius", (True, False, -1, float("nan"), float("inf")))
def test_invalid_cube_radius_is_not_a_missing_or_successful_observation(report, radius):
    with pytest.raises((TypeError, ValueError)):
        assess_saddle_storage_discriminator(
            report.source, cycle=report.cycle, initial_coordinate_radius=radius
        )


@pytest.mark.parametrize("capacity", (True, Q(0), Q(1) + Q(1, 2**150)))
def test_full_source_capacity_is_readmitted(report, capacity):
    source = replace(report.source, capacity=(capacity,) + report.source.capacity[1:])
    with pytest.raises((TypeError, ValueError)):
        assess_saddle_storage_discriminator(
            source, cycle=report.cycle, initial_coordinate_radius=0
        )


def test_another_support_cannot_inherit_the_private_leaf_preparation():
    source, cycle = _anchor(extra_edge=(0, 2))
    with pytest.raises(ValueError):
        assess_saddle_storage_discriminator(
            source, cycle=cycle, initial_coordinate_radius=0
        )


def test_sdk_export_keeps_the_supplied_law_and_nested_label_admission(report):
    projection = relational_report_to_dict(report)
    assert projection["report_type"] == "SaddleStorageDiscriminator"
    assert projection["report"]["law"] == report.law
    assert projection["report"] == report.to_dict()["report"]

    @dataclass(frozen=True)
    class Opaque:
        index: int

    prepared = report.prepared_state
    altered = replace(
        report,
        prepared_state=replace(prepared, nodes=(Opaque(0),) + prepared.nodes[1:]),
    )
    with pytest.raises(TypeError):
        relational_report_to_dict(altered)
