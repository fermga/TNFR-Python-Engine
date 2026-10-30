"""Exact environmental direction maps and finite acute acquisition controls."""

from copy import copy
from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import (
    analyze_sine_conservative_source_geometry,
    certify_sine_conservative_winding_entry,
)


def _source(*, forms=(60, 30, 0, -30, -60), order=tuple(range(10)), label=lambda i: i):
    graph = nx.Graph()
    graph.add_nodes_from(label(i) for i in order)
    graph.add_edges_from((label(i), label((i + 1) % 5)) for i in range(5))
    graph.add_edges_from((label(i), label(5 + i)) for i in range(5))
    for i in range(10):
        graph.nodes[label(i)].update(EPI=0 if i < 5 else forms[i - 5], theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _entry(source, **kwargs):
    arguments = dict(
        cycle=(0, 1, 2, 3, 4),
        scaled_window=(Q(1, 8), Q(2, 15)),
        edge_turn_offsets=(0, 0, 0, 0, -1),
    )
    return certify_sine_conservative_winding_entry(source, **(arguments | kwargs))


def test_source_map_matches_independent_full_laplacian_rows():
    source = _source()
    report = analyze_sine_conservative_source_geometry(source, receiver=(2, 0, 4, 1, 3))
    assert report.receiver_indices == (2, 0, 4, 1, 3)
    assert report.environment_indices == (5, 6, 7, 8, 9)
    assert report.environment_form_contrasts == (60, 30, 0, -30, -60)
    expected = tuple(
        tuple(-Q(int(j == i + 5), 3) for j in range(5, 10))
        for i in report.receiver_indices
    )
    assert report.source_velocity_matrix == expected
    assert report.source_rank == 5
    assert report.relative_source_rank == 4
    neighbors = [set() for _ in source.nodes]
    for i, j in source.edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    velocity = tuple(
        sum((source.epi[i] - source.epi[j] for j in neighbors[i]), Q(0))
        / len(neighbors[i])
        for i in report.receiver_indices
    )
    assert report.actual_receiver_velocity == velocity == (0, -20, 20, -10, 10)
    assert report.relative_velocity == (-20, 20, -10, 10)
    assert report.reconstruction_residuals == (0,) * 5
    assert report.identical_velocity_row_groups == ()
    assert report.node_form_rate_bound == 1
    assert report.node_phase_acceleration_bound == 2


def test_declared_full_window_certifies_acute_winding_without_trajectory():
    source = _source()
    report = _entry(source)
    assert report.initial_total_storage == 4500
    assert report.initial_phase_velocity[:5] == (-20, -10, 0, 10, 20)
    assert report.acquisition_certified and report.acute_acquisition_certified
    assert report.certified_winding == 1
    assert report.status == "certified_finite_acquisition"
    assert report.acute_margin_lower_bound > Q(1, 5)
    assert report.branch_margin_lower_bound > report.acute_margin_lower_bound
    radius = 2 * Q(2, 15) ** 2
    for gap in report.cycle_raw_gap_bounds[:4]:
        assert gap == I(Q(5, 4) - radius, Q(4, 3) + radius)
    assert report.cycle_raw_gap_bounds[-1] == I(-Q(16, 3) - radius, -5 + radius)
    assert all(
        gap.abs_max < pi_interval().lo / 2 for gap in report.cycle_principal_gap_bounds
    )
    assert (
        len(report.whole_window_form_bounds)
        == len(report.whole_window_phase_bounds)
        == 10
    )
    assert report.node_form_remainder_bound == Q(2, 15)
    assert report.node_phase_remainder_bound == Q(4, 225)


def test_equal_energy_reflection_control_is_a_separate_full_law_obstruction():
    source = _source(forms=(0, -30, -60, -60, -30))
    source = replace(source, epi=tuple(value + 9 for value in source.epi))
    assert sum((x * d for x, d in zip(source.epi, source.degrees)), Q(0)) == 0
    permutation = tuple((-i) % 5 for i in range(5)) + tuple(
        5 + (-i) % 5 for i in range(5)
    )
    symmetry = source.assess_cycle_symmetry(
        permutation_indices=permutation, cycle=(0, 1, 2, 3, 4)
    )
    assert (
        symmetry.initial_form_storage == _entry(_source()).initial_total_storage == 4500
    )
    assert symmetry.trajectory_symmetry_certified
    assert symmetry.cycle_orientation_reversed
    assert symmetry.zero_winding_when_nonantipodal
    assert not _entry(source).acute_acquisition_certified
    # Matrix rank belongs to available preparations, not this particular one.
    geometry = analyze_sine_conservative_source_geometry(source, receiver=range(5))
    assert geometry.relative_source_rank == 4
    assert geometry.actual_receiver_velocity == (0, 10, 20, 20, 10)


def test_acquired_witness_later_leaves_acute_geometry_under_the_same_law():
    report = _entry(_source(), scaled_window=(Q(1, 6), Q(1, 5)))
    # This is a positive exclusion by lower gap bounds, not an inference
    # from failure of a sufficient acquisition flag. Winding can persist.
    for gap in report.cycle_principal_gap_bounds[:4]:
        assert gap == I(Q(5, 3) - Q(2, 25), 2 + Q(2, 25))
        assert gap.lo > pi_interval().hi / 2
        assert gap.hi < pi_interval().lo
    assert report.certified_winding == 1
    assert not report.acute_acquisition_certified


def test_nonacute_winding_success_and_acute_zero_period_remain_distinct():
    from tests.physics.test_sine_conservative_winding_entry import (
        _source as original_source,
    )

    original = certify_sine_conservative_winding_entry(
        original_source(),
        cycle=(5, 6, 7, 8, 9),
        scaled_window=(Q(1, 4), Q(1, 3)),
        edge_turn_offsets=(1, 0, 0, 0, 0),
    )
    assert original.acquisition_certified
    assert original.status == "certified_finite_acquisition"
    assert not original.acute_acquisition_certified
    assert original.acute_margin_lower_bound < 0
    flat = _entry(_source(forms=(0,) * 5), edge_turn_offsets=(0,) * 5)
    assert flat.acute_margin_lower_bound > 0
    assert flat.certified_winding == 0
    assert not flat.acquisition_certified and not flat.acute_acquisition_certified


def test_failed_sufficient_acute_bound_does_not_claim_excluded_dynamics():
    report = _entry(_source(), scaled_window=(Q(1, 8), 1))
    assert not report.acute_acquisition_certified
    assert report.acute_margin_lower_bound < 0
    assert report.status == "unavailable"
    assert report.certified_winding is None
    assert not hasattr(report, "acute_winding_excluded")


def test_four_private_leaves_suffice_for_all_relative_initial_directions():
    graph = nx.cycle_graph(5)
    graph.add_edges_from((i, i + 4) for i in range(1, 5))
    for i in graph:
        graph.nodes[i].update(EPI=0 if i < 5 else i - 4, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph, reference_model=_source().reference_model
    )
    report = analyze_sine_conservative_source_geometry(source, receiver=range(5))
    assert report.source_rank == report.relative_source_rank == 4
    assert report.source_velocity_matrix[0] == (0,) * 4
    assert report.actual_receiver_velocity == (0, -Q(1, 3), -Q(2, 3), -1, -Q(4, 3))


def test_uncontacted_environment_columns_and_identical_receiver_rows_are_retained():
    graph = nx.cycle_graph(5)
    graph.add_edges_from(((0, 5), (1, 6), (5, 7)))
    for i in graph:
        graph.nodes[i].update(EPI=0 if i < 5 else i, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph, reference_model=_source().reference_model
    )
    report = analyze_sine_conservative_source_geometry(source, receiver=range(5))
    assert report.environment == (5, 6, 7)
    assert all(row[-1] == 0 for row in report.source_velocity_matrix)
    assert report.identical_velocity_row_groups == ((2, 3, 4),)
    assert report.source_rank == report.relative_source_rank == 2


def test_empty_environment_and_single_receiver_are_explicit():
    source = _source(forms=(0,) * 5)
    whole = analyze_sine_conservative_source_geometry(source, receiver=source.nodes)
    assert whole.environment == whole.environment_form_contrasts == ()
    assert whole.source_velocity_matrix == ((),) * 10
    assert whole.source_rank == whole.relative_source_rank == 0
    assert whole.identical_velocity_row_groups == (tuple(range(10)),)
    single = analyze_sine_conservative_source_geometry(source, receiver=(2,))
    assert single.source_rank == 1
    assert single.relative_source_rank == 0
    assert single.relative_source_velocity_matrix == single.relative_velocity == ()


def test_common_origins_labels_and_order_preserve_geometry():
    source = _source()
    report = analyze_sine_conservative_source_geometry(source, receiver=range(5))
    shifted = replace(
        source,
        epi=tuple(value + Q(2, 3) for value in source.epi),
        phase=(Q(7, 8),) * 10,
    )
    actual = analyze_sine_conservative_source_geometry(shifted, receiver=range(5))
    assert actual.environment_form_contrasts == report.environment_form_contrasts
    assert actual.actual_receiver_velocity == report.actual_receiver_velocity
    mapped = _source(order=tuple(reversed(range(10))), label=lambda i: f"n{i}")
    renamed = analyze_sine_conservative_source_geometry(
        mapped, receiver=tuple(f"n{i}" for i in range(5))
    )
    assert renamed.actual_receiver_velocity == report.actual_receiver_velocity
    assert renamed.relative_source_rank == report.relative_source_rank
    assert renamed.source_velocity_matrix == tuple(
        tuple(reversed(row)) for row in report.source_velocity_matrix
    )


def test_cached_rows_are_ignored_and_consumed_primitives_are_readmitted():
    source = _source()
    forged = replace(
        source,
        form_gradient=(999,) * 10,
        form_rates=(I(999),) * 10,
        phase_rates=(I(999),) * 10,
        storage=I(0),
    )
    actual = analyze_sine_conservative_source_geometry(forged, receiver=range(5))
    expected = analyze_sine_conservative_source_geometry(source, receiver=range(5))
    assert actual.actual_receiver_velocity == expected.actual_receiver_velocity
    assert actual.source_velocity_matrix == expected.source_velocity_matrix
    assert (
        _entry(forged).acute_margin_lower_bound
        == _entry(source).acute_margin_lower_bound
    )


@pytest.mark.parametrize("receiver", [(), (0, 0), (0, "missing"), {0, 1}, "01234"])
def test_invalid_receiver_order_or_members_reject(receiver):
    with pytest.raises((TypeError, ValueError)):
        analyze_sine_conservative_source_geometry(_source(), receiver=receiver)


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True,) + (0,) * 9},
        {"epi": (1,) + (0,) * 9},
        {"phase": (False,) + (0,) * 9},
        {"phase": (1,) + (0,) * 9},
        {"capacity": (0,) + (1,) * 9},
        {"capacity": (True,) * 10},
        {"degrees": (1,) * 10},
        {"law": "native"},
        {"edges": ((0, 1),)},
    ],
)
def test_invalid_or_unsupported_source_rejects(changes):
    with pytest.raises((TypeError, ValueError)):
        analyze_sine_conservative_source_geometry(
            replace(_source(), **changes), receiver=range(5)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("epi_weight", False),
        ("phase_weight", True),
        ("storage_scale", True),
        ("epi_weight", Q(1, 2)),
    ],
)
def test_model_primitives_are_admitted_before_coefficient_equality(field, value):
    source = _source()
    model = copy(source.reference_model)
    object.__setattr__(model, field, value)
    with pytest.raises((TypeError, ValueError)):
        analyze_sine_conservative_source_geometry(
            replace(source, reference_model=model), receiver=range(5)
        )


def test_direct_projections_retain_geometry_and_additive_acute_evidence():
    source = _source()
    geometry = analyze_sine_conservative_source_geometry(source, receiver=range(5))
    payload = geometry.to_dict()
    assert payload["schema"] == "tnfr.sine-conservative-source-geometry.v1"
    assert payload["report"]["relative_source_rank"] == 4
    payload["report"]["receiver"].append("changed")
    assert geometry.receiver == tuple(range(5))
    entry = _entry(source).to_dict()
    assert entry["schema"] == "tnfr.sine-conservative-winding-entry.v1"
    assert entry["report"]["acute_acquisition_certified"] is True
    assert entry["report"]["acute_margin_lower_bound"] is not None
