"""Independent joint reading of the original frozen phase-grouping sources."""

import json
import pickle
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import networkx as nx
import pytest

from tests.physics.test_relational_sine_replica import (
    PAIRS,
    _fine_field,
    _frozen_grouping_transition,
    _frozen_grouping_window,
)
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.physics import relational_sine_scale as owner
from tnfr.sdk import export_to_json, relational_report_to_dict

JOINT_PAIRS = ((0, 2), (1, 3), (4, 5), (6, 7), (8, 9))
JOINT_NEAREST = (2, 3, 0, 1, 5, 4, 7, 6, 9, 8)
RHO = Q(1, 2**20)
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    if bound.lo == bound.hi:
        assert abs(value - _mp(bound.lo)) < mp.mpf("1e-85")
    else:
        assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


@pytest.fixture(scope="module")
def frozen_sources():
    result = []
    for sign in (1, -1):
        graph = _frozen_grouping_transition(form_sign=sign)
        phase = _frozen_grouping_window(graph)
        result.append((graph, phase, phase.joint_observation()))
    return tuple(result)


def test_both_original_boxes_already_have_the_same_distinct_joint_pairing(
    frozen_sources,
):
    for _, phase, joint in frozen_sources:
        assert joint.phase_window is phase
        assert joint.comparison is phase.comparison
        assert joint.window == (Q(1, 128), Q(1, 64))
        assert phase.form_error_bounds == phase.phase_error_bounds == (RHO,) * 10
        assert joint.initial_observation.candidate_pairs == JOINT_PAIRS
        assert joint.initial_box_candidate_pairs == JOINT_PAIRS
        assert joint.candidate_pairs == JOINT_PAIRS
        assert joint.initial_box_nearest_partner_indices == JOINT_NEAREST
        assert joint.nearest_partner_indices == JOINT_NEAREST
        assert joint.whole_window_pairing_certified
        assert joint.same_pairing_as_initial_box is True
        assert len(joint.joint_distance_window_bounds) == 45
        assert all(margin > 0 for margin in joint.nearest_separation_margins)
        assert joint.support_admission_status == "rejected"
        # These new observable pairs are not the supplied replica partition:
        # each of (0,2) and (1,3) has a live internal edge on the same support.
        assert (0, 2) in joint.comparison.edges
        assert JOINT_PAIRS != PAIRS

    positive_phase, negative_phase = frozen_sources[0][1], frozen_sources[1][1]
    assert positive_phase.status == "certified_matching"
    assert positive_phase.candidate_pairs == PAIRS
    assert negative_phase.status == "certified_nonmutual"
    assert negative_phase.candidate_pairs is None
    assert negative_phase.nearest_partner_indices == (1, 2, 1, 2, 5, 4, 7, 6, 9, 8)


def test_initial_joint_margin_uses_form_and_every_competitor(frozen_sources):
    _, _, report = frozen_sources[0]
    observation = report.initial_observation
    distances = dict(
        zip(observation.distance_indices, observation.squared_joint_distance_bounds)
    )
    with mp.workdps(100):
        forms, phases = map(
            lambda values: list(map(_mp, values)),
            (observation.forms, observation.phases),
        )
        for (i, j), bound in distances.items():
            exact = (forms[i] - forms[j]) ** 2 + 2 * (1 - mp.cos(phases[i] - phases[j]))
            _contains(bound, exact)
        margin = mp.mpf(1) / 16 - 2 * (mp.cos(mp.mpf(1) / 8) - mp.cos(mp.mpf(1) / 4))
        assert margin > mp.mpf(1) / 64
        assert _mp(distances[0, 1].lo - distances[0, 2].hi) > _mp(Q(1, 64))
    for i, desired in enumerate(JOINT_NEAREST):
        nearest = distances[tuple(sorted((i, desired)))]
        for competitor in range(10):
            if competitor not in (i, desired):
                assert nearest.hi < distances[tuple(sorted((i, competitor)))].lo


def test_joint_rate_reads_both_full_rows_and_reverses_with_form(frozen_sources):
    with mp.workdps(100):
        expected_by_sign = []
        for graph, phase_window, projection in frozen_sources:
            comparison = projection.comparison
            state = mp.matrix(list(map(_mp, comparison.epi + comparison.phase)))
            field = _fine_field(graph, state)
            expected = []
            for (i, j), bound in zip(
                phase_window.distance_indices,
                projection.source_joint_distance_rate_bounds,
            ):
                rate = 2 * (state[i] - state[j]) * (field[i] - field[j])
                rate += (
                    2
                    * mp.sin(state[10 + i] - state[10 + j])
                    * (field[10 + i] - field[10 + j])
                )
                _contains(bound, rate)
                expected.append(rate)
            expected_by_sign.append(expected)
        assert any(abs(value) > mp.mpf("0.01") for value in expected_by_sign[0])
        for positive, negative in zip(*expected_by_sign):
            assert abs(positive + negative) < mp.mpf("1e-85")
        indices = frozen_sources[0][1].distance_indices
        for pair in ((0, 2), (1, 3)):
            for values in expected_by_sign:
                assert abs(values[indices.index(pair)]) < mp.mpf("1e-85")
    # A phase-rate-only calculation omits a nonzero form contribution.
    with mp.workdps(100):
        graph, _, projection = frozen_sources[0]
        c = projection.comparison
        state = mp.matrix(list(map(_mp, c.epi + c.phase)))
        field = _fine_field(graph, state)
        assert 2 * (state[0] - state[1]) * (field[0] - field[1]) < 0
        form_contribution = (
            2 * (c.epi[0] - c.epi[1]) * (c.form_rates[0] - c.form_rates[1])
        )
        assert form_contribution.hi < 0


def test_joint_tubes_cover_independent_full_coordinate_envelopes(frozen_sources):
    # These are corners of the admitted analytic tubes, not sampled ODE
    # solutions. Every form and phase constituent can vary independently.
    with mp.workdps(100):
        for _, phase, joint in frozen_sources:
            c = joint.comparison
            assert joint.form_remainder_bounds == tuple(
                radius + speed * phase.window[1]
                for radius, speed in zip(
                    phase.form_error_bounds, phase.form_speed_upper_bounds
                )
            )
            for time in phase.window:
                for form_sign, phase_sign in product((-1, 1), repeat=2):
                    forms = [
                        _mp(x) + form_sign * (-1) ** i * _mp(r)
                        for i, (x, r) in enumerate(
                            zip(c.epi, joint.form_remainder_bounds)
                        )
                    ]
                    phases = [
                        _mp(theta)
                        + _mp(numerator * time) / mp.pi
                        + phase_sign * (-1) ** i * _mp(r)
                        for i, (theta, numerator, r) in enumerate(
                            zip(
                                c.phase,
                                phase.phase_rate_numerators,
                                phase.phase_remainder_upper_bounds,
                            )
                        )
                    ]
                    for (i, j), bound in zip(
                        phase.distance_indices, joint.joint_distance_window_bounds
                    ):
                        distance = (forms[i] - forms[j]) ** 2 + 2 * (
                            1 - mp.cos(phases[i] - phases[j])
                        )
                        _contains(bound, distance)
            for form_sign, phase_sign in product((-1, 1), repeat=2):
                forms = [
                    _mp(x) + form_sign * (-1) ** i * _mp(RHO)
                    for i, x in enumerate(c.epi)
                ]
                phases = [
                    _mp(theta) + phase_sign * (-1) ** i * _mp(RHO)
                    for i, theta in enumerate(c.phase)
                ]
                for (i, j), bound in zip(
                    phase.distance_indices, joint.initial_box_joint_distance_bounds
                ):
                    distance = (forms[i] - forms[j]) ** 2 + 2 * (
                        1 - mp.cos(phases[i] - phases[j])
                    )
                    _contains(bound, distance)


def test_projection_keeps_the_original_phase_verdict_and_does_not_recapture(
    monkeypatch,
):
    graph = _frozen_grouping_transition()
    before = _snapshot(graph)
    original = owner.bound_relational_sine_exchange
    captures = []

    def counted(*args, **kwargs):
        captures.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    phase = _frozen_grouping_window(graph)
    phase_payload = phase.to_dict()
    joint = phase.joint_observation()
    assert captures == [graph]
    assert joint.phase_window is phase
    assert phase.to_dict() == phase_payload
    assert _snapshot(graph) == before
    assert phase.support_admission_status == "admitted"
    assert joint.support_admission_status == "rejected"


def test_unavailable_box_is_not_repaired_from_center_or_phase_matching():
    wide = _frozen_grouping_window(
        form_error_bounds=(Q(1, 8),) * 10,
        phase_error_bounds=(Q(1, 8),) * 10,
    ).joint_observation()
    assert wide.initial_observation.candidate_pairs == JOINT_PAIRS
    assert wide.initial_box_candidate_pairs is None
    assert wide.candidate_pairs is None
    assert not wide.whole_window_pairing_certified
    assert wide.same_pairing_as_initial_box is None
    assert wide.support_admission_status == "not_attempted"
    # Invalid source uncertainty remains the existing owner's error contract.
    with pytest.raises(ADMISSION_ERRORS):
        _frozen_grouping_window(form_error_bounds=(False,) * 10).joint_observation()


def test_general_beta_capacity_rows_and_resolved_nonmutual_observation():
    graph = nx.path_graph(4)
    for node, form, phase, capacity in zip(
        graph,
        (Q(-1, 4), Q(1, 8), Q(3, 4), Q(-1, 2)),
        (Q(1, 8), Q(-1, 4), Q(1, 2), Q(3, 8)),
        (0, Q(1, 2), Q(3, 2), 2),
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    beta = Q(3, 2)
    model = RelationalExchangeModel(beta, epi_weight=0, phase_domain="regular")
    phase = owner.assess_sine_pairing_window(
        graph,
        reference_model=model,
        form_error_bounds=(Q(1, 1000),) * 4,
        phase_error_bounds=(Q(1, 2000),) * 4,
        window_start=Q(1, 128),
        window_end=Q(1, 64),
    )
    report = phase.joint_observation()
    # A zero-capacity row preserves its independently supplied initial radius.
    assert phase.form_speed_upper_bounds[0] == 0
    assert report.form_remainder_bounds[0] == Q(1, 1000)
    assert all(radius > Q(1, 1000) for radius in report.form_remainder_bounds[1:])
    with mp.workdps(100):
        c = report.comparison
        state = mp.matrix(list(map(_mp, c.epi + c.phase)))
        field = _fine_field(graph, state, beta=beta)
        assert field[0] == field[4] == 0
        for k, (i, j) in enumerate(phase.distance_indices):
            gap, angle = state[j] - state[i], state[4 + j] - state[4 + i]
            distance = gap**2 + 2 * _mp(beta) * (1 - mp.cos(angle))
            rate = 2 * gap * (field[j] - field[i])
            rate += 2 * _mp(beta) * mp.sin(angle) * (field[4 + j] - field[4 + i])
            _contains(
                report.initial_observation.squared_joint_distance_bounds[k], distance
            )
            _contains(report.source_joint_distance_rate_bounds[k], rate)

    # Full numerical resolution can prove nonmutuality; it is not unavailability.
    frozen = nx.path_graph(3)
    for node, form in zip(frozen, (0, 1, 3)):
        frozen.nodes[node].update(EPI=form, theta=0, nu_f=0)
    nonmutual = owner.assess_sine_pairing_window(
        frozen,
        reference_model=model,
        form_error_bounds=(0,) * 3,
        phase_error_bounds=(0,) * 3,
        window_start=0,
        window_end=1,
    ).joint_observation()
    assert nonmutual.nearest_partner_indices == (1, 0, 1)
    assert nonmutual.initial_box_status == nonmutual.status == "certified_nonmutual"
    assert nonmutual.candidate_pairs is None
    assert nonmutual.same_pairing_as_initial_box is None
    assert not nonmutual.whole_window_pairing_certified
    assert nonmutual.support_admission_status == "not_attempted"
    assert nonmutual.form_remainder_bounds == (0,) * 3


def test_projection_exact_export_and_all_nested_labels(frozen_sources, tmp_path):
    report = frozen_sources[0][2]
    specialized = report.to_dict()
    assert specialized["schema"] == "tnfr.relational-sine-joint-pairing-projection.v1"
    assert relational_report_to_dict(report)["report"] == specialized["report"]
    target = tmp_path / "joint-projection.json"
    export_to_json(report, target)
    assert json.loads(target.read_text(encoding="utf-8")) == specialized

    @dataclass(frozen=True)
    class Opaque:
        value: str

    bad = ((Opaque("bad"), 2),)
    corrupted = (
        replace(report, candidate_pairs=bad),
        replace(report, initial_box_candidate_pairs=bad),
        replace(
            report,
            initial_observation=replace(
                report.initial_observation, candidate_pairs=bad
            ),
        ),
        replace(report, phase_window=replace(report.phase_window, candidate_pairs=bad)),
    )
    for value in corrupted:
        with pytest.raises(ADMISSION_ERRORS):
            value.to_dict()
