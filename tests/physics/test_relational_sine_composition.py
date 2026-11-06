"""Static frame-aware composition controls under the declared sine law.

Frozen contact: P2 supports (0,1) and (2,3), capacities (1,2) and (3,4),
zero nominal form/phase, supplied bridge (1,2), event time 2, e=w=1/2,
beta=1, and relative component origins (1/4,1/4). Node residual radii are
zero unless explicitly changed. A separate positive integration fixture joins
two prepared C5 supports at ports 0 and 5, phases 5*j/4, unit capacities,
zero form and origins, with -1 turns on each closing cycle edge.

No trajectory, support-event execution, occurrence law or frozen producer is
evaluated. Capture and the optional supplied event-work budget are independent.
"""

import pickle
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_composition import assess_sine_pattern_composition
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

MODEL = RelationalExchangeModel(1, phase_domain="regular")
OFFSETS = (0, 0, 0)
ADMISSION_ERRORS = (TypeError, ValueError)


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    from tnfr.physics import relational_sine_forecast

    def forbidden(*args, **kwargs):
        pytest.fail("composition controls must remain static")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
        yield


def _component(nodes, capacities, *, forms=None, phases=None, error=Q(0), cycle=False):
    graph = nx.cycle_graph(nodes) if cycle else nx.path_graph(nodes)
    forms = (0,) * len(nodes) if forms is None else forms
    phases = (0,) * len(nodes) if phases is None else phases
    for node, capacity, form, phase in zip(graph, capacities, forms, phases):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    graph.graph.update(GAMMA={"type": "none"}, retained={"history": [1, 2]})
    source = bound_relational_sine_pattern(
        graph,
        reference_node=nodes[0],
        reference_model=MODEL,
        form_error_bounds=(error,) * len(nodes),
        phase_error_bounds=(error,) * len(nodes),
    )
    return graph, source


def _pair(*, error=Q(0)):
    return (
        _component((0, 1), (1, 2), error=error),
        _component((2, 3), (3, 4), error=error),
    )


def _compose(left, right, **changes):
    arguments = dict(
        bridge=(1, 2),
        observation_time=Q(2),
        edge_turn_offsets=OFFSETS,
        form_origin_difference=Q(1, 4),
        phase_origin_difference=Q(1, 4),
    )
    arguments.update(changes)
    return assess_sine_pattern_composition(left, right, **arguments)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


def test_positive_contact_recomputes_full_degrees_mobility_and_framed_mean():
    (_, left), (_, right) = _pair()
    report = _compose(left, right)
    assert report.status == "available" and report.unavailable_reasons == ()
    assert report.capture.admitted
    assert report.observation_time == report.capture.observation_time == Q(2)
    assert report.budget_status == "not_supplied"
    assert report.nodes == (0, 1, 2, 3)
    assert report.edges == ((0, 1), (1, 2), (2, 3))
    assert report.separate_degrees == (1, 1, 1, 1)
    assert report.joined_degrees == report.joined.degrees == (1, 2, 2, 1)
    assert report.separate_form_weights == (1, Q(1, 2), Q(1, 3), Q(1, 4))
    assert report.joined_form_weights == (1, 1, Q(2, 3), Q(1, 4))
    assert report.separate_mobility == (1, 2, 3, 4)
    assert report.joined_mobility == (1, 1, Q(3, 2), 4)
    assert report.joined.capacity == (1, 2, 3, 4)
    assert report.representative_weighted_form_mean_bounds == I(Q(11, 140))
    assert report.representative_weighted_phase_mean_bounds == I(Q(11, 140))
    assert report.capture.weighted_form_mean is None
    assert report.capture.weighted_phase_mean is None
    assert report.bridge_form_gap_bounds == report.bridge_phase_gap_bounds == I(Q(1, 4))
    assert report.bridge_form_storage_bounds == I(Q(1, 32))
    assert report.separate_storage_bounds == I(0)
    assert report.capture.boundary_storage_lower_bound == 1
    assert report.capture.storage_bounds == report.joined_storage_bounds
    with mp.workdps(90):
        energy = mp.mpf(1) / 32 + 1 - mp.cos(mp.mpf(1) / 4)
        _contains(report.bridge_storage_bounds, energy)
        _contains(report.joined_storage_bounds, energy)


@pytest.mark.parametrize(
    "missing", ["form_origin_difference", "phase_origin_difference"]
)
def test_missing_cross_component_frame_never_silently_aligns_sources(missing):
    (_, left), (_, right) = _pair()
    report = _compose(left, right, **{missing: None})
    assert report.status == "unavailable" and report.unavailable_reasons
    assert report.joined is report.capture is None
    assert report.bridge_storage_bounds is report.joined_storage_bounds is None
    assert report.budget_status == "not_supplied"
    with_allowance = _compose(left, right, work_allowance=1, **{missing: None})
    assert with_allowance.budget_status == "unresolved"


def test_misaligned_known_frame_is_available_but_does_not_certify_capture():
    (_, left), (_, right) = _pair()
    report = _compose(left, right, form_origin_difference=0, phase_origin_difference=2)
    assert report.status == "available" and report.joined is not None
    assert not report.capture.admitted
    assert (
        "whole_source_set_strict_acute_sector_not_certified"
        in report.capture.unresolved_conditions
    )
    assert report.bridge_phase_gap_bounds == I(2)
    assert report.bridge_storage_bounds.lo > 1


def test_bridge_work_budget_and_joint_capture_have_independent_verdicts():
    (_, left), (_, right) = _pair()
    rejected_work = _compose(left, right, work_allowance=0)
    assert rejected_work.capture.admitted
    assert rejected_work.budget_status == "exceeds_allowance"
    assert rejected_work.work_margin_bounds.hi < 0
    allowed_work = _compose(left, right, work_allowance=1)
    assert allowed_work.capture.admitted
    assert allowed_work.budget_status == "within_allowance"

    _, left = _component((0, 1), (1, 2), forms=(-1, 0))
    _, right = _component((2, 3), (3, 4), forms=(0, 1))
    assert left.certify_sector_capture(edge_turn_offsets=(0,)).admitted
    assert right.certify_sector_capture(edge_turn_offsets=(0,)).admitted
    joined = _compose(
        left,
        right,
        form_origin_difference=0,
        phase_origin_difference=0,
        work_allowance=0,
    )
    assert joined.component_storage_bounds == (I(Q(1, 2)), I(Q(1, 2)))
    assert joined.bridge_storage_bounds == I(0)
    assert joined.budget_status == "within_allowance"
    assert joined.joined_storage_bounds == I(1)
    assert not joined.capture.admitted
    assert joined.capture.energy_margin == 0


def test_original_node_residuals_are_retained_without_reference_error_duplication():
    radius = Q(1, 64)
    (_, left), (_, right) = _pair(error=radius)
    report = _compose(left, right)
    assert report.capture.admitted
    assert (
        report.joined.form_error_bounds
        == report.joined.phase_error_bounds
        == (radius,) * 4
    )
    assert (
        report.bridge_form_gap_bounds
        == report.bridge_phase_gap_bounds
        == I(Q(1, 4) - 2 * radius, Q(1, 4) + 2 * radius)
    )
    assert report.joined.edge_form_gap_bounds[0] == I(-2 * radius, 2 * radius)
    assert report.joined.edge_form_gap_bounds[2] == I(-2 * radius, 2 * radius)
    assert report.representative_weighted_form_mean_bounds == I(
        Q(11, 140) - radius, Q(11, 140) + radius
    )
    with mp.workdps(90):
        for signs in product((-1, 1), repeat=4):
            form = tuple(
                _mp(v + sign * radius)
                for v, sign in zip(report.joined.nominal_form, signs)
            )
            phase = tuple(
                _mp(v + (-1) ** i * sign * radius)
                for i, (v, sign) in enumerate(zip(report.joined.nominal_phase, signs))
            )
            energy = sum(
                (form[j] - form[i]) ** 2 / 2 + 1 - mp.cos(phase[j] - phase[i])
                for i, j in report.edges
            )
            _contains(report.joined_storage_bounds, energy)
            assert energy < _mp(report.capture.boundary_storage_lower_bound)

    form_only_left = replace(left, phase_error_bounds=(Q(0),) * 2)
    form_only_right = replace(right, phase_error_bounds=(Q(0),) * 2)
    uncertain_work = _compose(
        form_only_left,
        form_only_right,
        form_origin_difference=0,
        phase_origin_difference=0,
        work_allowance=radius**2,
    )
    assert uncertain_work.bridge_storage_bounds == I(0, 2 * radius**2)
    assert uncertain_work.budget_status == "unresolved"
    assert uncertain_work.capture.admitted


def test_exact_rational_primitive_admission_does_not_rematerialize_a_graph(monkeypatch):
    (_, left), (_, right) = _pair()
    left = replace(left, nominal_form=(Q(1, 3), Q(1, 3)), capacity=(Q(1, 3), Q(2)))

    def forbidden(*args, **kwargs):
        pytest.fail(
            "composition must rebuild exact report primitives without graph capture"
        )

    monkeypatch.setattr(
        "tnfr.physics.relational_sine_pattern._capture_sine_state", forbidden
    )
    report = _compose(left, right, form_origin_difference=Q(1, 3))
    assert report.joined.nominal_form == (Q(1, 3),) * 4
    assert report.joined.capacity[0] == Q(1, 3)
    assert report.joined_form_weights[0] == 3
    assert report.bridge_form_gap_bounds == I(0)
    assert report.representative_weighted_form_mean_bounds == I(Q(1, 3))


def test_common_frame_covariance_and_detached_source_graphs():
    (left_graph, left), (right_graph, right) = _pair(error=Q(1, 64))
    before = pickle.dumps((left_graph, right_graph, left, right))
    report = _compose(left, right)
    assert pickle.dumps((left_graph, right_graph, left, right)) == before
    shifted = tuple(
        replace(
            source,
            nominal_form=tuple(value + 7 for value in source.nominal_form),
            nominal_phase=tuple(value - 3 for value in source.nominal_phase),
        )
        for source in (left, right)
    )
    moved = _compose(*shifted)
    assert moved.joined.edge_form_gap_bounds == report.joined.edge_form_gap_bounds
    assert moved.joined.edge_phase_gap_bounds == report.joined.edge_phase_gap_bounds
    assert moved.joined.form_rate_bounds == report.joined.form_rate_bounds
    assert moved.joined.phase_rate_bounds == report.joined.phase_rate_bounds
    assert moved.bridge_storage_bounds == report.bridge_storage_bounds
    assert moved.joined_storage_bounds == report.joined_storage_bounds
    assert moved.capture.admitted == report.capture.admitted
    assert (
        moved.representative_weighted_form_mean_bounds
        == report.representative_weighted_form_mean_bounds + 7
    )
    assert (
        moved.representative_weighted_phase_mean_bounds
        == report.representative_weighted_phase_mean_bounds - 3
    )


def test_composition_rebuilds_consumed_fields_instead_of_cached_favorable_evidence():
    (_, left), (_, right) = _pair()
    expected = _compose(left, right)
    stale = replace(
        left,
        relative_form_bounds=(I(100),) * 2,
        edge_form_gap_bounds=(I(100),),
        storage_bounds=I(-100),
        phase_rate_bounds=(I(100),) * 2,
    )
    actual = _compose(stale, right)
    assert actual.joined == expected.joined
    assert actual.component_storage_bounds == expected.component_storage_bounds
    assert actual.joined_storage_bounds == expected.joined_storage_bounds
    assert actual.capture.admitted
    changed = replace(left, nominal_form=(Q(-2), Q(0)))
    rebuilt = _compose(changed, right)
    assert rebuilt.component_storage_bounds[0] == I(2)
    assert not rebuilt.capture.admitted
    uncertain = _compose(replace(left, form_error_bounds=(Q(2),) * 2), right)
    assert not uncertain.capture.admitted
    assert uncertain.joined_storage_bounds.hi > 8
    assert (
        "strict_total_storage_boundary_barrier_not_certified"
        in uncertain.capture.unresolved_conditions
    )


@pytest.mark.parametrize(
    "change",
    [
        {"nominal_form": (True, 0)},
        {"nominal_phase": (0, float("nan"))},
        {"form_error_bounds": (Q(-1), Q(0))},
        {"phase_error_bounds": (True, Q(0))},
        {"capacity": (Q(0), Q(2))},
        {"capacity": (True, Q(2))},
        {"degrees": (True, 1)},
        {"neighbors": ((0,), (0,))},
    ],
)
def test_invalid_primitive_source_cannot_reuse_favorable_derived_fields(change):
    (_, left), (_, right) = _pair()
    with pytest.raises(ADMISSION_ERRORS):
        _compose(replace(left, **change), right)


@pytest.mark.parametrize(
    "change",
    [
        {"observation_time": True},
        {"observation_time": -1},
        {"observation_time": float("inf")},
        {"form_origin_difference": True},
        {"phase_origin_difference": float("nan")},
        {"form_origin_difference": I(0, 1)},
        {"work_allowance": -1},
        {"work_allowance": True},
        {"bridge": (0, 1)},
        {"bridge": (0, 99)},
        {"bridge": (0,)},
        {"edge_turn_offsets": (0, 0)},
        {"edge_turn_offsets": (0, True, 0)},
    ],
)
def test_malformed_frame_event_or_bridge_is_rejected(change):
    (_, left), (_, right) = _pair()
    with pytest.raises(ADMISSION_ERRORS):
        _compose(left, right, **change)


def test_overlapping_nodes_and_different_complete_coefficients_are_rejected():
    (_, left), (_, right) = _pair()
    with pytest.raises(ADMISSION_ERRORS):
        _compose(left, left)
    different = RelationalExchangeModel(2, phase_domain="regular")
    with pytest.raises(ADMISSION_ERRORS):
        _compose(left, replace(right, reference_model=different))
    lossless = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    with pytest.raises(ADMISSION_ERRORS):
        _compose(
            replace(left, reference_model=lossless),
            replace(right, reference_model=lossless),
        )


def test_two_prepared_winding_cycles_use_the_existing_full_capture_handoff():
    phases = tuple(Q(5 * j, 4) for j in range(5))
    _, left = _component(tuple(range(5)), (1,) * 5, phases=phases, cycle=True)
    _, right = _component(tuple(range(5, 10)), (1,) * 5, phases=phases, cycle=True)
    edges = tuple(sorted((*left.edges, *right.edges, (0, 5))))
    offsets = tuple(-1 if edge in ((0, 4), (5, 9)) else 0 for edge in edges)
    report = _compose(
        left,
        right,
        bridge=(0, 5),
        edge_turn_offsets=offsets,
        form_origin_difference=0,
        phase_origin_difference=0,
        work_allowance=0,
    )
    assert report.status == "available" and report.capture.admitted
    assert report.capture.cycle_periods == (1, 1)
    assert report.bridge_storage_bounds == I(0)
    assert report.budget_status == "within_allowance"
    assert report.joined_degrees == (3, 2, 2, 2, 2, 3, 2, 2, 2, 2)
    assert len(report.capture.boundary_face_lower_bounds) == 22
    assert report.joined_storage_bounds.lo > 6
    assert report.capture.energy_margin > 0
