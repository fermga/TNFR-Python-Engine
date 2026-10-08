"""Synthetic joint-inverse controls; no reserved response or frozen replay."""

import itertools
import json
from fractions import Fraction as Q
from inspect import Parameter, signature

import mpmath as mp
import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics import relational_sine_two_pulse_inference as owner


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    assert _mp(interval.lo) <= value <= _mp(interval.hi)


def _arguments():
    # Artificial leading responses test the inverse algebra, not realization
    # by a trajectory. A separate unrelated full-flow control appears below.
    a1, a2, h, b, gain = Q(1, 5), Q(4, 5), Q(1, 2**22), Q(91, 64), Q(7, 5)
    with mp.workdps(100):
        changes = tuple(
            -2
            * _mp(gain)
            * _mp(h)
            * mp.sin(3 * _mp(a) / 2)
            * mp.cos(_mp(b - a / 2))
            / (1023 * mp.pi)
            for a in (a1, a2)
        )
        baseline = mp.mpf(5) / 9
        readings = (baseline, baseline + changes[0], baseline + sum(changes))
        centers = tuple(Q(mp.nstr(value, 90)) for value in readings)
    radius = Q(1, 10**88)
    return dict(
        bulk_angle_bounds=(Q(11, 8), Q(3, 2)),
        receiver_short_angle_bounds=(Q(2, 3), Q(1)),
        form_radius=Q(1, 2**42),
        phase_radius=Q(1, 2**42),
        phase_increments=(a1, a2),
        probe_duration=h,
        recorded_reading_bounds=tuple(
            (value - radius, value + radius) for value in centers
        ),
        readout_error_bound=Q(1, 2**70),
        readout_gain_bounds=(Q(1), Q(2)),
    )


@pytest.fixture(scope="module")
def report():
    return owner.infer_sine_two_pulse_geometry_gain(**_arguments())


def test_nine_mandatory_primitive_keywords_exclude_hidden_truth():
    parameters = signature(owner.infer_sine_two_pulse_geometry_gain).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 9
    assert all(
        value.kind == Parameter.KEYWORD_ONLY and value.default == Parameter.empty
        for value in parameters.values()
    )
    with pytest.raises(TypeError):
        owner.infer_sine_two_pulse_geometry_gain(
            **_arguments(), true_bulk_angle=Q(91, 64)
        )


def _mp_matrix(amplitudes, duration):
    return mp.matrix(
        [
            [
                2
                * _mp(duration)
                * mp.sin(3 * _mp(a) / 2)
                * part(_mp(a) / 2)
                / (1023 * mp.pi)
                for part in (mp.cos, mp.sin)
            ]
            for a in amplitudes
        ]
    )


def test_factored_determinant_and_inverse_enclose_independent_matrix_algebra(report):
    with mp.workdps(100):
        matrix = _mp_matrix(report.phase_increments, report.probe_duration)
        determinant = mp.det(matrix)
        inverse = matrix**-1
        _contains(report.determinant_bounds, determinant)
        assert report.determinant_bounds.lo > 0
        for i in range(2):
            for j in range(2):
                _contains(report.response_matrix_bounds[i][j], matrix[i, j])
                _contains(report.inverse_matrix_bounds[i][j], inverse[i, j])
            assert sum(abs(inverse[i, j]) for j in range(2)) <= _mp(
                report.inverse_row_sum_upper_bounds[i]
            )
        assert report.rank_certified and not report.rank_deficient
        assert report.actual_phase_jumps == (Q(1, 5), Q(3, 5))
        assert report.total_duration == 2 * report.probe_duration


def _graph_and_source(b, c):
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    degree = np.array([graph.degree[i] for i in range(18)], dtype=float)
    short, bulk = 4 * np.pi - 8 * b, (2 * np.pi - c) / 8
    delta = (short - c) / 2
    theta = np.array(
        [
            0,
            *(short + (j - 1) * b for j in range(1, 9)),
            delta,
            *(delta + c + (j - 1) * bulk for j in range(1, 9)),
        ]
    )
    theta -= degree @ theta / sum(degree)
    q = np.zeros(18)
    q[4], q[5] = 1, -1
    laplacian = nx.laplacian_matrix(graph, nodelist=range(18)).toarray()

    def full_flow(_, state):
        x, phase = state[:18], state[18:]
        gradient, currents = laplacian @ x, np.zeros(18)
        for i, j in graph.edges:
            current = np.sin(phase[j] - phase[i])
            currents[i] += current
            currents[j] -= current
        gamma = 1 / (1023 * np.pi)
        return np.r_[(-gradient + gamma * currents) / degree, gamma * gradient / degree]

    return graph, degree, theta, q, full_flow


def test_actual_edge_sums_give_each_local_leading_coefficient(report):
    _, degree, theta, q, field = _graph_and_source(91 / 64, 9 / 10)
    for index, amplitude in enumerate(report.phase_increments):
        state = np.r_[np.zeros(18), theta + float(amplitude) * q]
        rate = field(0, state)
        expected = -float(report.response_scale_bounds[index].midpoint) * np.cos(
            91 / 64 - float(amplitude) / 2
        )
        assert float(report.probe_duration) * (q @ rate[:18]) == pytest.approx(
            expected, rel=1e-12, abs=1e-23
        )
        assert np.array_equal(rate[18:], np.zeros(18))
    assert degree @ q == 0 and degree @ q**2 == 4


def test_synthetic_leading_observation_covers_geometry_and_held_gain(report):
    assert report.status == "bounded_candidate" and report.inverse_enclosure_available
    assert report.source_admitted and report.finite_response_certified
    assert report.nominal_bulk_angle_outer_bounds.contains(Q(91, 64))
    assert report.readout_gain_outer_bounds.contains(Q(7, 5))
    assert report.nominal_bulk_angle_outer_bounds.width < Q(1, 1024)
    assert report.readout_gain_outer_bounds.width < Q(1, 1024)
    assert (
        report.actual_long_arc_mean_outer_bounds
        == report.nominal_bulk_angle_outer_bounds
        + I(-report.phase_radius / 8, report.phase_radius / 8)
    )
    assert (
        not report.whole_window_acute_certified
    )  # Larger pulses need no false acute claim.
    assert (
        "bounded_candidate_gives_necessary_marginals_not_existence_or_point_identification"
        in report.scope
    )


def test_shared_middle_error_corners_are_retained_without_independent_middle_copies(
    report,
):
    with mp.workdps(100):
        inverse = _mp_matrix(report.phase_increments, report.probe_duration) ** -1
        bounds = tuple(
            radius + report.readout_error_bound for radius in report.reading_radii
        )
        for i in range(2):
            left, right = inverse[i, 0], inverse[i, 1]
            expected_coefficients = (left, right - left, -right)
            for interval, value in zip(
                report.transformed_reading_coefficients[i], expected_coefficients
            ):
                _contains(interval, value)
            for signs in itertools.product((-1, 1), repeat=3):
                actual = sum(
                    coefficient * sign * _mp(bound)
                    for coefficient, sign, bound in zip(
                        expected_coefficients, signs, bounds
                    )
                )
                assert abs(actual) <= _mp(report.transformed_observation_error_radii[i])
            assert abs(sum(expected_coefficients)) < mp.mpf("1e-80")
            # These inverse rows have opposite signs: sharing documents the
            # joint error but does not tighten their individual sensor radii.
            assert left * right < 0
            assert mp.almosteq(
                abs(left) + abs(left - right) + abs(right), 2 * (abs(left) + abs(right))
            )


def test_huge_common_offset_cancels_before_interval_coefficient_arithmetic(report):
    arguments = _arguments()
    offset = Q(10**400) + Q(2, 7)
    arguments["recorded_reading_bounds"] = tuple(
        (lo + offset, hi + offset) for lo, hi in arguments["recorded_reading_bounds"]
    )
    shifted = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert shifted.reading_midpoints == tuple(
        value + offset for value in report.reading_midpoints
    )
    for field in (
        "recorded_increment_midpoints",
        "reading_radii",
        "transformed_centers",
        "transformed_observation_error_radii",
        "raw_transformed_bounds",
        "nominal_bulk_angle_outer_bounds",
        "readout_gain_outer_bounds",
    ):
        assert getattr(shifted, field) == getattr(report, field)


def test_nonreserved_successive_full_flow_retains_first_window_memory():
    b, c = 91 / 64, 9 / 10
    _, degree, theta, q, flow = _graph_and_source(b, c)
    x = np.array([((3 * i) % 7 - 3) / 2**30 for i in range(18)])
    eta = np.array([((5 * i) % 11 - 5) / 2**28 for i in range(18)])
    x -= degree @ x / 40
    eta -= degree @ eta / 40
    assert degree @ x**2 < (1 / 2**20) ** 2
    assert degree @ eta**2 < (1 / 2**20) ** 2
    a1, a2, h = 1 / 5, 4 / 5, 1 / 128
    original = np.r_[x, theta + eta]
    first = original.copy()
    first[18:] += a1 * q
    first_solution = solve_ivp(
        flow, (0, h), first, method="DOP853", rtol=1e-12, atol=1e-14
    )
    assert first_solution.success
    reached = first_solution.y[:, -1]
    second = reached.copy()
    second[18:] += (a2 - a1) * q
    second_solution = solve_ivp(
        flow, (0, h), second, method="DOP853", rtol=1e-12, atol=1e-14
    )
    assert second_solution.success
    final = second_solution.y[:, -1]
    assert np.linalg.norm(reached[:18] - x) > 1e-7
    assert np.linalg.norm(reached[18:] - first[18:]) > 1e-13
    gain, offset = 7 / 5, 5 / 9
    readings = tuple(
        gain * (q @ state[:18]) + offset for state in (original, reached, final)
    )
    arguments = _arguments()
    arguments.update(
        form_radius=Q(1, 2**20),
        phase_radius=Q(1, 2**20),
        probe_duration=Q(1, 128),
        readout_error_bound=Q(1, 10**12),
        recorded_reading_bounds=tuple(
            (Q.from_float(value), Q.from_float(value)) for value in readings
        ),
    )
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.status == "bounded_candidate"
    assert result.nominal_bulk_angle_outer_bounds.contains(Q(91, 64))
    assert result.readout_gain_outer_bounds.contains(Q(7, 5))
    actual_angle = b - (eta[1] - eta[0]) / 8
    assert (
        float(result.actual_long_arc_mean_outer_bounds.lo)
        <= actual_angle
        <= float(result.actual_long_arc_mean_outer_bounds.hi)
    )
    for amplitude, before, after in ((a1, original, reached), (a2, reached, final)):
        ideal = (
            -2
            * h
            * np.sin(3 * amplitude / 2)
            * np.cos(b - amplitude / 2)
            / (1023 * np.pi)
        )
        actual = q @ (after[:18] - before[:18])
        assert abs(actual - ideal) < float(result.finite_remainder_upper_bound)
    # A reset would be a different preparation; the second window actually
    # starts with the previously reached nonzero form and drifting phase.
    reset = original.copy()
    reset[18:] += a2 * q
    reset_solution = solve_ivp(
        flow, (0, h), reset, method="DOP853", rtol=1e-12, atol=1e-14
    )
    assert reset_solution.success
    reset_end = reset_solution.y[:, -1]
    assert (
        abs(q @ ((final[:18] - reached[:18]) - (reset_end[:18] - original[:18]))) > 1e-9
    )


def test_rational_design_budget_without_any_reserved_readout():
    x = y = Q(1, 2**40)
    h, noise, g = Q(1, 2**21), Q(1, 2**60), Q(1, 3069)
    total, amplitude = 2 * h, Q(3, 4)
    qmax = (x + g * total * (4 + 4 * amplitude + 2 * y)) / (
        1 - 4 * g * g * total * total
    )
    error = 2 * h * qmax + 2 * g * h * y + 4 * g * g * total * h * qmax
    # Determinant/inverse feasibility uses constants only, never observations.
    gamma = 1 / (1023 * pi_interval())
    scales = tuple(2 * gamma * h * sin(I(Q(3, 2) * a)) for a in (Q(1, 4), Q(3, 4)))
    factor = sin(I(Q(1, 4)))
    from tnfr.mathematics._rational_interval import cos

    rows = (
        (
            sin(I(Q(3, 8))) / (scales[0] * factor),
            -sin(I(Q(1, 8))) / (scales[1] * factor),
        ),
        (
            -cos(I(Q(3, 8))) / (scales[0] * factor),
            cos(I(Q(1, 8))) / (scales[1] * factor),
        ),
    )
    row_bounds = tuple(sum(value.abs_max for value in row) for row in rows)
    assert sum(bound**2 for bound in row_bounds) < (36000 / h) ** 2
    radius = 72000 * (error + 2 * noise) / h
    assert 2 * radius < Q(1, 1024)
    assert 2 * radius / (1 - 2 * radius) + y / 4 < Q(1, 1024)


def test_equal_pulses_are_unavailable_not_a_full_response_exclusion():
    arguments = _arguments()
    arguments["phase_increments"] = (Q(1, 4), Q(1, 4))
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.status == "unavailable" and result.source_admitted
    assert result.rank_deficient and not result.rank_certified
    assert result.actual_phase_jumps == (Q(1, 4), Q(0))
    assert result.determinant_bounds == I(0)
    assert result.inverse_matrix_bounds is None
    assert (
        result.nominal_bulk_angle_outer_bounds
        is result.readout_gain_outer_bounds
        is None
    )
    assert result.incompatibility_reasons == ()


@pytest.mark.parametrize(
    "amplitudes", ((Q(1, 4), Q(1, 4) + Q(1, 10**100)), (Q(1, 10**60), Q(2, 10**60)))
)
def test_positive_but_unresolved_determinants_remain_unavailable(amplitudes):
    arguments = _arguments()
    arguments["phase_increments"] = amplitudes
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.phase_increments == amplitudes
    assert not result.rank_deficient and not result.rank_certified
    assert result.status == "unavailable" and not result.incompatibility_reasons


def test_positive_subgrid_gain_prior_cannot_become_false_incompatibility():
    arguments = _arguments()
    arguments.update(
        readout_gain_bounds=(Q(1, 10**200), Q(2, 10**200)),
        recorded_reading_bounds=((Q(0), Q(0)),) * 3,
        readout_error_bound=Q(0),
    )
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.rank_certified and result.source_admitted
    assert result.status == "unavailable"
    assert (
        result.nominal_bulk_angle_outer_bounds
        is result.readout_gain_outer_bounds
        is None
    )
    assert "projection_unavailable" in result.unavailable_reasons[0]
    assert not result.incompatibility_reasons


def test_huge_exact_gain_is_not_coerced_to_binary64():
    arguments = _arguments()
    factor = Q(10**400)
    arguments["readout_gain_bounds"] = (factor, 2 * factor)
    arguments["recorded_reading_bounds"] = tuple(
        (lo * factor, hi * factor) for lo, hi in arguments["recorded_reading_bounds"]
    )
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.status == "bounded_candidate"
    assert result.readout_gain_outer_bounds.contains(Q(7, 5) * factor)
    assert result.nominal_bulk_angle_outer_bounds.contains(Q(91, 64))


def test_source_boundary_is_strict_even_with_resolved_rank(report):
    arguments = _arguments()
    arguments["phase_radius"] = report.nominal_acute_margin
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.source_acute_margin == 0
    assert result.rank_certified and not result.source_admitted
    assert result.status == "unavailable"
    assert result.finite_remainder_upper_bound is result.raw_transformed_bounds is None


def test_zero_readings_can_exclude_positive_gain_under_combined_premises():
    arguments = _arguments()
    arguments["recorded_reading_bounds"] = ((Q(0), Q(0)),) * 3
    result = owner.infer_sine_two_pulse_geometry_gain(**arguments)
    assert result.status == "incompatible"
    assert result.source_admitted and result.rank_certified
    assert (
        not result.inverse_enclosure_available
        and result.transformed_coordinate_bounds is None
    )
    assert (
        result.nominal_bulk_angle_outer_bounds
        is result.readout_gain_outer_bounds
        is None
    )
    assert not result.unavailable_reasons


def test_interval_boundary_equality_is_not_strict_exclusion():
    assert owner._intersection(I(1, 2), I(2, 3)) == I(2)
    assert owner._intersection(I(1, 2), I(3, 4)) is None


def test_serialization_retains_primitives_and_joint_error_map(report):
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.sine-two-pulse-inference.v1"
    saved = payload["report"]
    assert saved["phase_increments"] == [
        {"numerator": 1, "denominator": 5},
        {"numerator": 4, "denominator": 5},
    ]
    assert len(saved["recorded_reading_bounds"]) == 3
    assert [len(row) for row in saved["transformed_reading_coefficients"]] == [3, 3]
    lower = saved["readout_gain_outer_bounds"]["lo"]
    assert (
        Q(lower["numerator"], lower["denominator"])
        == report.readout_gain_outer_bounds.lo
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("form_radius", -1),
        ("phase_radius", -1),
        ("readout_error_bound", -1),
        ("probe_duration", 0),
        ("probe_duration", Q(501, 1000)),
        ("phase_increments", (0, Q(1, 2))),
        ("phase_increments", (1, Q(1, 2))),
        ("phase_increments", (Q(1, 2), 2)),
        ("phase_increments", (Q(1, 2),)),
        ("phase_increments", (True, 1)),
        ("readout_gain_bounds", (0, 1)),
        ("bulk_angle_bounds", (1, Q(3, 2))),
        ("receiver_short_angle_bounds", (Q(1, 2), 1)),
        ("recorded_reading_bounds", ((0, 0),) * 2),
        ("recorded_reading_bounds", ((0, 0),) * 4),
        ("recorded_reading_bounds", ((1, 0),) * 3),
        ("recorded_reading_bounds", (I(0),) * 3),
        ("recorded_reading_bounds", ((True, 1),) * 3),
        ("form_radius", True),
        ("phase_radius", np.bool_(True)),
        ("readout_error_bound", float("nan")),
        ("probe_duration", float("inf")),
        ("form_radius", mp.mpf("1e-4000")),
    ),
)
def test_invalid_primitives_reject_before_geometry(monkeypatch, field, value):
    arguments = _arguments()
    arguments[field] = value
    monkeypatch.setattr(
        owner,
        "_inference_geometry",
        lambda: pytest.fail("invalid input reached geometry"),
    )
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_two_pulse_geometry_gain(**arguments)


def test_no_single_inverse_target_or_forward_producer_is_needed(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import relational_sine_two_port_compatibility as compatibility
    from tnfr.physics import relational_sine_two_port_inference as single
    from tnfr.physics import relational_sine_two_port_readout as forward

    def forbidden(*args, **kwargs):
        pytest.fail("joint inverse called a target, inverse or trajectory producer")

    monkeypatch.setattr(compatibility, "assess_sine_two_port_compatibility", forbidden)
    monkeypatch.setattr(single, "infer_sine_two_port_geometry", forbidden)
    monkeypatch.setattr(forward, "bound_sine_two_port_readout", forbidden)
    monkeypatch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
    assert (
        owner.infer_sine_two_pulse_geometry_gain(**_arguments()).status
        == "bounded_candidate"
    )
