"""Admission and provenance for the explicitly supplied bridge storage family."""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_bridge_memory import assess_sine_bridge_memory
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_resonance import assess_bridge_storage_family

LEFT, RIGHT = tuple(range(6)), tuple(range(6, 12))


@pytest.fixture(scope="module")
def source():
    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _arguments(source):
    return dict(
        left_cycle=LEFT,
        right_cycle=RIGHT,
        target_phase_turns=tuple(Q(node % 6, 6) for node in source.nodes),
    )


def _assess(source, *, epsilon=Q(4, 9), **changes):
    arguments = _arguments(source)
    arguments.update(changes)
    return assess_bridge_storage_family(source, epsilon=epsilon, **arguments)


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_sine_limit_reuses_the_complete_existing_nodal_tangent(source):
    sine = assess_sine_bridge_memory(source, **_arguments(source))
    limit = _assess(source, epsilon=0)
    assert limit.full_tangent_generator == sine.full_tangent_generator
    assert limit.target_phase_turns == sine.channels.target_phase_turns
    assert limit.source == sine.channels.source
    assert limit.law != limit.source.law
    assert limit.epsilon == 0
    assert all(
        bound.contains(value)
        for bound, value in zip(
            sine.channels.second_jet_diagonal_bounds,
            (limit.second_jet[0][0], limit.second_jet[1][1]),
        )
    )


def test_storage_comparison_retains_declared_law_and_conditional_local_scope(report):
    assert report.law == "normalized_sine_cubic_reciprocal_exchange"
    assert report.source.law == "normalized_sine_reciprocal_exchange"
    assert report.criticality_status == "proved_by_equal_gap_odd_cancellation"
    assert (
        report.local_positivity_status
        == "positive_edge_curvatures_on_connected_support"
    )
    assert report.target_epi == (0,) * 12
    assert report.local_phase_radius_turns == Q(1, 24)
    assert report.local_curvature_lower_bounds.lo > 0
    assert report.local_excess_storage_threshold_bounds.lo > 0
    assert (
        "conditional_local_storage_barrier_not_global_capture_or_an_observed_state_test"
        in report.scope
    )


def test_unchanged_preparation_and_geometry_do_not_force_unchanged_response(source):
    reports = tuple(_assess(source, epsilon=value) for value in (0, Q(4, 9), 1))
    for report in reports[1:]:
        assert report.nodal_bridge_preparation == reports[0].nodal_bridge_preparation
        assert report.bridge_observation_rows == reports[0].bridge_observation_rows
        assert report.target_geometry == reports[0].target_geometry
        assert report.first_jet == reports[0].first_jet
    assert reports[0].channel_difference > 0
    assert reports[1].channel_difference == 0
    assert reports[2].channel_difference < 0
    assert reports[1].phase_hessian == reports[1].form_laplacian


def test_represented_epsilon_is_not_rounded_to_an_intended_rational(source):
    represented = _assess(source, epsilon=0.1)
    exact = _assess(source, epsilon=Q(1, 10))
    assert represented.epsilon == Q.from_float(0.1)
    assert represented.epsilon != exact.epsilon
    assert represented.phase_hessian != exact.phase_hessian
    tiny = _assess(source, epsilon=Q(1, 10**200))
    assert tiny.epsilon != 0
    assert tiny.phase_hessian != _assess(source, epsilon=0).phase_hessian


@pytest.mark.parametrize("epsilon", [-1, True, False, float("inf"), float("nan"), "1"])
def test_epsilon_admission_rejects_invalid_or_implicit_coefficients(source, epsilon):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, epsilon=epsilon)


def test_source_and_derived_caches_are_not_the_target_or_storage_evidence(source):
    forged = replace(
        source,
        epi=tuple(float(i) for i in range(12)),
        phase=(Q(1, 8),) * 12,
        form_rates=(I(999),) * 12,
        phase_rates=(I(999),) * 12,
        storage=I(999),
    )
    baseline, report = _assess(source), _assess(forged)
    assert report.source.epi == tuple(Q(i) for i in range(12))
    assert all(type(value) is Q for value in report.source.epi)
    assert report.source.phase == (Q(1, 8),) * 12
    assert report.target_epi == (0,) * 12
    for field in (
        "full_tangent_generator",
        "full_energy_metric",
        "first_jet",
        "second_jet",
        "target_storage",
        "local_excess_storage_threshold_bounds",
    ):
        assert getattr(report, field) == getattr(baseline, field)
    assert source.epi == (0,) * 12


@pytest.mark.parametrize(
    "changes",
    [
        {"left_cycle": LEFT[:-1]},
        {"left_cycle": (0, 2, 1, 3, 4, 5)},
        {"right_cycle": LEFT},
        {"target_phase_turns": (0,) * 12},
    ],
)
def test_shared_full_support_and_target_admission_are_preserved(source, changes):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, **changes)


@pytest.mark.parametrize("capacity", [True, 0, 2])
def test_source_capacity_is_readmitted_before_any_cached_comparison(source, capacity):
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, capacity=(capacity,) + source.capacity[1:]))


def test_loss_or_changed_reference_law_is_not_silently_reinterpreted(source):
    with pytest.raises(ValueError):
        _assess(replace(source, law="normalized_sine_cubic_reciprocal_exchange"))
    with pytest.raises(ValueError):
        _assess(
            replace(
                source,
                reference_model=RelationalExchangeModel(
                    1, epi_weight=1, phase_weight=1, phase_domain="regular"
                ),
            )
        )


def test_export_is_detached_and_preserves_exact_law_and_signed_difference(source):
    report = _assess(source, epsilon=1)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.bridge-storage-family.v1"
    body = payload["report"]
    assert body["law"] == report.law
    assert body["channel_difference"] == {"numerator": -5, "denominator": 18}
    assert body["epsilon"] == {"numerator": 1, "denominator": 1}
    body["full_tangent_generator"][0][12]["numerator"] = 999
    body["source"]["capacity"][0]["numerator"] = 999
    assert report.full_tangent_generator[0][12] != 999
    assert report.source.capacity[0] == 1


@pytest.mark.parametrize(
    "field", ["left_cycle", "right_cycle", "bridge", "target_geometry"]
)
def test_export_rejects_opaque_dataclass_labels_in_all_declared_groups(report, field):
    # Generic dataclass projection would serialize this object as a JSON map.
    # Node labels have the narrower shared scalar-or-tuple contract instead.
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int = 17

    label = OpaqueLabel()
    if field == "target_geometry":
        geometry = report.target_geometry.geometry
        changed = replace(
            report.target_geometry,
            geometry=replace(geometry, nodes=(label,) + geometry.nodes[1:]),
        )
    else:
        labels = getattr(report, field)
        changed = (label,) + labels[1:]
    forged = replace(report, **{field: changed})
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        forged.to_dict()
