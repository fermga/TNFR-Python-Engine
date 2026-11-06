"""Admission, correlated geometry and provenance of the supplied return-path law."""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from benchmarks import relational_return_geometry as baseline
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.phase_cycle_geometry import assess_return_path_storage_geometry
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

LEFT, RIGHT = tuple(range(5)), tuple(range(5, 10))


def _source(*, reverse=False, extra_edge=False):
    graph = nx.Graph()
    graph.add_nodes_from(reversed(range(11)) if reverse else range(11))
    graph.add_edges_from(
        [(offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)]
        + [(0, 10), (10, 5), (1, 6)]
    )
    if extra_edge:
        graph.add_edge(0, 2)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


@pytest.fixture(scope="module")
def source():
    return _source()


def _assess(source, **changes):
    arguments = dict(left_cycle=LEFT, right_cycle=RIGHT, mediator=10, epsilon=1)
    arguments.update(changes)
    return assess_return_path_storage_geometry(source, **arguments)


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_original_sine_bracket_and_affine_coordinates_are_reused(source):
    original = baseline.enclose_opposite_root(refinements=40)
    current = _assess(source, epsilon=0)
    root = current.root_turn_bracket
    assert max(original.lower, root.lower) < min(original.upper, root.upper)
    assert original.lower == Q(2619154484885, 26388279066624)
    assert original.upper == Q(436525747481, 4398046511104)
    assert baseline.opposite_nodal_turns(original.midpoint) == tuple(
        a + b * original.midpoint for a, b in current.nodal_turn_affine_coefficients
    )
    assert current.law != current.source.law


def test_root_signs_retain_fixed_computational_budget_without_midpoint_claim(report):
    root = report.root_turn_bracket
    assert Q(1, 12) < root.lower < root.upper < Q(1, 5)
    assert root.lower_residual.lo > 0 > root.upper_residual.hi
    assert root.upper - root.lower == Q(7, 60 * 2**root.refinements)
    assert (
        report.criticality_status
        == "implicit_root_with_exact_full_incidence_factorization"
    )
    assert (
        report.uniqueness_status
        == "unique_in_declared_acute_period_cell_modulo_common_phase"
    )
    assert (
        "implicit_root_defines_one_equilibrium_not_every_point_in_the_interval_box"
        in report.scope
    )


def test_affine_lifts_keep_shared_root_correlations_and_periods(report):
    # Integer offsets connect real nodal lifts to principal acute edge lifts
    # for every value of s, not merely the root's rounded representative.
    for edge, pair, offset in zip(
        report.geometry.edges,
        report.edge_turn_affine_coefficients,
        report.edge_integer_offsets,
    ):
        a, b = (report.nodal_turn_affine_coefficients[i] for i in edge)
        assert (b[0] - a[0] - offset, b[1] - a[1]) == pair
    assert report.named_cycle_periods == (1, -1, 0)
    assert report.nodal_root_residual_multipliers == (-1, 1, 0, 0, 0, 1, -1, 0, 0, 0, 0)
    assert all(value.contains(0) for value in report.nodal_current_residual_bounds)
    assert all(value.lo > 0 for value in report.edge_curvature_bounds)
    assert report.minimum_acute_margin_turns_bounds.lo > 0
    assert report.target_phase_storage_bounds.lo > 0


def test_geometry_changes_with_the_supplied_coefficient_at_fixed_support(source):
    sine, cubic = (_assess(source, epsilon=value) for value in (0, 1))
    assert sine.geometry == cubic.geometry
    assert sine.named_cycle_periods == cubic.named_cycle_periods
    assert sine.nodal_turn_affine_coefficients == cubic.nodal_turn_affine_coefficients
    assert sine.root_turn_bracket.upper < cubic.root_turn_bracket.lower
    assert (
        sine.minimum_acute_margin_turns_bounds.hi
        < cubic.minimum_acute_margin_turns_bounds.lo
    )


def test_node_order_is_only_a_coordinate_choice(source, report):
    reversed_report = _assess(_source(reverse=True))
    assert reversed_report.root_turn_bracket == report.root_turn_bracket
    mapping = dict(
        zip(
            reversed_report.source.nodes, reversed_report.nodal_turn_affine_coefficients
        )
    )
    assert (
        tuple(mapping[node] for node in source.nodes)
        == report.nodal_turn_affine_coefficients
    )
    assert reversed_report.named_cycle_periods == report.named_cycle_periods


@pytest.mark.parametrize("epsilon", [Q(1, 10**200), Q(10**200)])
def test_extreme_exact_coefficients_keep_normalized_root_and_actual_currents(
    source, epsilon
):
    report = _assess(source, epsilon=epsilon, refinements=8)
    assert report.epsilon == epsilon
    assert report.root_turn_bracket.lower_residual.lo > 0
    assert report.root_turn_bracket.upper_residual.hi < 0
    assert report.target_phase_storage_bounds.lo > 0
    if epsilon > 1:
        assert max(value.abs_max for value in report.edge_current_bounds) > epsilon / 10
        assert report.target_phase_storage_bounds.lo > epsilon


def test_represented_coefficient_is_retained_as_its_exact_binary_value(source):
    assert _assess(source, epsilon=0.1, refinements=8).epsilon == Q.from_float(0.1)


def test_source_snapshot_and_cached_fields_do_not_become_equilibrium_evidence(
    source, report
):
    forged = replace(
        source,
        epi=tuple(float(i) for i in range(11)),
        phase=(Q(1, 7),) * 11,
        form_rates=(I(999),) * 11,
        phase_rates=(I(999),) * 11,
        storage=I(999),
    )
    rebuilt = _assess(forged)
    assert rebuilt.source.epi == tuple(map(Q, range(11)))
    assert all(type(value) is Q for value in rebuilt.source.epi)
    assert rebuilt.target_epi == (0,) * 11
    assert rebuilt.root_turn_bracket == report.root_turn_bracket
    assert rebuilt.target_phase_storage_bounds == report.target_phase_storage_bounds
    assert rebuilt.nodal_current_residual_bounds == report.nodal_current_residual_bounds


@pytest.mark.parametrize("epsilon", [-1, True, float("inf"), float("nan"), "1"])
def test_invalid_coefficients_are_rejected_before_root_solving(source, epsilon):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, epsilon=epsilon)


@pytest.mark.parametrize("refinements", [0, 65, True, 2.0])
def test_refinement_policy_is_strictly_admitted(source, refinements):
    with pytest.raises(ValueError, match="refinements"):
        _assess(source, refinements=refinements)


@pytest.mark.parametrize(
    "changes",
    [
        {"left_cycle": LEFT[:-1]},
        {"left_cycle": (0, 2, 1, 3, 4)},
        {"right_cycle": LEFT},
        {"mediator": 0},
        {"mediator": 99},
    ],
)
def test_full_support_roles_are_required(source, changes):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, **changes)


def test_extra_edges_and_changed_law_or_capacity_do_not_inherit_certificate(source):
    with pytest.raises(ValueError, match="full support"):
        _assess(_source(extra_edge=True))
    with pytest.raises(ValueError):
        _assess(replace(source, law="native"))
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, capacity=(True,) + source.capacity[1:]))
    with pytest.raises(ValueError, match="unit capacity"):
        _assess(replace(source, capacity=(Q(2),) + source.capacity[1:]))
    with pytest.raises(ValueError, match="zero loss"):
        _assess(
            replace(
                source,
                reference_model=RelationalExchangeModel(
                    1, epi_weight=1, phase_weight=1, phase_domain="regular"
                ),
            )
        )


def test_export_retains_detached_exact_affine_and_normalized_root_evidence(report):
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.return-path-storage-geometry.v1"
    body = payload["report"]
    assert body["epsilon"] == {"numerator": 1, "denominator": 1}
    assert body["named_cycle_periods"][1] == {"numerator": -1, "denominator": 1}
    body["nodal_turn_affine_coefficients"][1][1]["numerator"] = 999
    assert report.nodal_turn_affine_coefficients[1][1] == 1


@pytest.mark.parametrize(
    "field", ["geometry", "left_cycle", "right_cycle", "mediator", "named_cycles"]
)
def test_export_rejects_opaque_dataclass_labels_in_every_group(report, field):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int = 9

    label = OpaqueLabel()
    if field == "geometry":
        value = replace(report.geometry, nodes=(label,) + report.geometry.nodes[1:])
    elif field == "mediator":
        value = label
    elif field == "named_cycles":
        value = ((label,) + report.named_cycles[0][1:],) + report.named_cycles[1:]
    else:
        value = (label,) + getattr(report, field)[1:]
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        replace(report, **{field: value}).to_dict()
