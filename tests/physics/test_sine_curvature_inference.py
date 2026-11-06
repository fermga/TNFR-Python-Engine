"""Independent finite-curvature controls, never a reserved-response replay."""

import itertools
import json
from fractions import Fraction as Q
from inspect import Parameter, signature

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import relational_sine_curvature_inference as owner


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _arguments(horizon=Q(1, 2**24)):
    # Synthetic leading/quadratic constraints, not an acquired trajectory.
    b, gain, rate = Q(71, 50), Q(9, 7), Q(7, 5)
    a1, a2 = Q(1, 4), Q(3, 4)
    with mp.workdps(120):
        gamma = 1 / (1023 * mp.pi)
        h = _mp(horizon)
        changes = tuple(
            -2
            * _mp(gain * rate)
            * gamma
            * h
            * mp.sin(3 * _mp(a) / 2)
            * mp.cos(_mp(b - a / 2))
            for a in (a1, a2)
        )
        curvature = gamma * (
            2 * mp.sin(_mp(b + a1))
            - mp.mpf(3) / 2 * mp.sin(_mp(b - 2 * a1))
            - mp.sin(_mp(b)) / 2
        )
        acceleration = _mp(gain * rate * rate) * curvature
        base = mp.mpf(7) / 11
        first = base + changes[0] + acceleration * h * h / 2
        values = (
            base,
            base + changes[0] / 2 + acceleration * h * h / 8,
            first,
            first + changes[1],
        )
        centers = tuple(Q(mp.nstr(value, 110)) for value in values)
    radius = Q(1, 10**108)
    noise = Q(1, 2**90) * (horizon / Q(1, 2**24)) ** 2
    return dict(
        bulk_angle_bounds=(Q(11, 8), Q(3, 2)),
        receiver_short_angle_bounds=(Q(2, 3), Q(1)),
        form_radius=Q(1, 2**48),
        phase_radius=Q(1, 2**48),
        phase_increments=(a1, a2),
        probe_duration=horizon,
        recorded_reading_bounds=tuple(
            (value - radius, value + radius) for value in centers
        ),
        readout_error_bound=noise,
        readout_gain_bounds=(Q(1), Q(2)),
        clock_rate_bounds=(Q(1, 2), Q(2)),
    )


@pytest.fixture(scope="module")
def report():
    return owner.infer_sine_geometry_gain_clock_curvature(**_arguments())


def _graph():
    edges = tuple(
        sorted(
            {tuple(sorted((o + j, o + (j + 1) % 9))) for o in (0, 9) for j in range(9)}
            | {(0, 9), (1, 10)}
        )
    )
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    return edges, degrees


def _mp_phase(b, c, degrees):
    short, bulk = 4 * mp.pi - 8 * b, (2 * mp.pi - c) / 8
    shift = (short - c) / 2
    raw = [mp.mpf(0)] + [short + (j - 1) * b for j in range(1, 9)]
    raw += [shift] + [shift + c + (j - 1) * bulk for j in range(1, 9)]
    mean = sum(d * value for d, value in zip(degrees, raw)) / 40
    return [value - mean for value in raw]


def test_ten_mandatory_inputs_have_four_declared_reading_times(report):
    parameters = signature(owner.infer_sine_geometry_gain_clock_curvature).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 10
    assert all(
        value.kind == Parameter.KEYWORD_ONLY and value.default == Parameter.empty
        for value in parameters.values()
    )
    h = report.probe_duration
    assert report.observation_times == (0, h / 2, h, 2 * h)
    assert report.clock_envelope.recorded_reading_bounds == tuple(
        report.recorded_reading_bounds[i] for i in (0, 2, 3)
    )
    with pytest.raises(TypeError):
        owner.infer_sine_geometry_gain_clock_curvature(
            **_arguments(), clock_envelope=report.clock_envelope
        )


def test_synthetic_constraints_cover_and_refine_gain_and_clock(report):
    assert report.status == "bounded_candidate"
    assert report.curvature_refinement_available and report.inverse_enclosure_available
    assert (
        report.curvature_coefficient_positive
        and report.finite_curvature_bound_certified
    )
    for bound, truth in (
        (report.nominal_bulk_angle_outer_bounds, Q(71, 50)),
        (report.effective_gain_outer_bounds, Q(9, 5)),
        (report.readout_gain_outer_bounds, Q(9, 7)),
        (report.clock_rate_outer_bounds, Q(7, 5)),
    ):
        assert bound.contains(truth)
    child = report.clock_envelope
    assert (
        report.nominal_bulk_angle_outer_bounds == child.nominal_bulk_angle_outer_bounds
    )
    assert (
        report.actual_long_arc_mean_outer_bounds
        == child.actual_long_arc_mean_outer_bounds
    )
    assert (
        report.actual_long_arc_mean_outer_bounds
        == report.nominal_bulk_angle_outer_bounds
        + I(-report.phase_radius / 8, report.phase_radius / 8)
    )
    assert report.clock_rate_outer_bounds.width < child.clock_rate_outer_bounds.width
    assert (
        report.readout_gain_outer_bounds.width < child.readout_gain_outer_bounds.width
    )
    assert report.clock_rate_outer_bounds.width < Q(1, 80)
    assert report.readout_gain_outer_bounds.width < Q(1, 16)


def test_all_sixteen_reading_error_corners_share_the_same_four_coordinates(report):
    assert report.curvature_reading_coefficients == (1, -2, 1, 0)
    expected_radius = (
        report.reading_radii[0]
        + 2 * report.reading_radii[1]
        + report.reading_radii[2]
        + 4 * report.readout_error_bound
    )
    assert report.curvature_observation_radius == expected_radius
    base = report.clock_envelope.constraint_envelope
    for embedded, original in zip(
        report.embedded_inverse_reading_coefficients,
        base.transformed_reading_coefficients,
    ):
        assert embedded == (original[0], I(0), original[1], original[2])
    delta = report.readout_error_bound
    for signs in itertools.product((-1, 1), repeat=4):
        errors = tuple(delta * sign for sign in signs)
        curvature_error = errors[0] - 2 * errors[1] + errors[2]
        assert abs(curvature_error) <= 4 * delta
        assert report.sensor_corrected_curvature_bounds.contains(
            report.curvature_midpoint + curvature_error
        )
        # The same endpoint errors enter both inverse rows. The half-time
        # error is absent from those rows; the final error is absent from C.
        for embedded, row in zip(
            report.embedded_inverse_reading_coefficients, base.inverse_matrix_bounds
        ):
            actual = sum(
                (coefficient * error for coefficient, error in zip(embedded, errors)),
                I(0),
            )
            increments = (errors[2] - errors[0], errors[3] - errors[2])
            direct = -row[0] * increments[0] - row[1] * increments[1]
            assert max(actual.lo, direct.lo) <= min(actual.hi, direct.hi)
    assert 4 * delta == max(
        abs(delta * (signs[0] - 2 * signs[1] + signs[2]))
        for signs in itertools.product((-1, 1), repeat=4)
    )


def test_offset_cancels_before_second_difference_interval_rounding(report):
    arguments = _arguments()
    offset = Q(10**400) + Q(2, 7)
    arguments["recorded_reading_bounds"] = tuple(
        (lo + offset, hi + offset) for lo, hi in arguments["recorded_reading_bounds"]
    )
    changed = owner.infer_sine_geometry_gain_clock_curvature(**arguments)
    assert changed.reading_midpoints == tuple(
        value + offset for value in report.reading_midpoints
    )
    for field in (
        "curvature_midpoint",
        "curvature_observation_radius",
        "normalized_curvature_observation_bounds",
        "clock_rate_constraint_bounds",
        "clock_rate_outer_bounds",
        "readout_gain_outer_bounds",
    ):
        assert getattr(changed, field) == getattr(report, field)


@pytest.mark.parametrize(
    "b,a", ((Q(11, 8), Q(1, 4)), (Q(71, 50), Q(1, 5)), (Q(3, 2), Q(3, 4)))
)
def test_independent_full_graph_curvature_matches_stable_trigonometric_coefficient(
    b, a
):
    edges, degrees = _graph()
    with mp.workdps(90):
        phase = _mp_phase(_mp(b), mp.mpf(7) / 8, degrees)
        phase[4] += _mp(a)
        phase[5] -= _mp(a)
        currents = [mp.mpf(0) for _ in degrees]
        for i, j in edges:
            value = mp.sin(phase[j] - phase[i])
            currents[i] += value
            currents[j] -= value
        forcing = [value / d for value, d in zip(currents, degrees)]
        gradient = [mp.mpf(0) for _ in degrees]
        for i, j in edges:
            value = forcing[i] - forcing[j]
            gradient[i] += value
            gradient[j] -= value
        gamma = 1 / (1023 * mp.pi)
        actual = -gamma * (gradient[4] / degrees[4] - gradient[5] / degrees[5])
        stable = gamma * (
            2 * mp.sin(_mp(a) / 2) ** 2 * (1 + 3 * mp.cos(_mp(a))) * mp.sin(_mp(b))
            + mp.sin(_mp(a)) * (2 + 3 * mp.cos(_mp(a))) * mp.cos(_mp(b))
        )
        assert abs(actual - stable) < mp.mpf("1e-85")
        assert actual > 0


def test_initial_error_and_third_derivative_bounds_cover_both_full_rows(report):
    edges, degrees = _graph()
    with mp.workdps(90):
        gamma = 1 / (1023 * mp.pi)
        phase = _mp_phase(mp.mpf(71) / 50, mp.mpf(7) / 8, degrees)
        phase[4] += mp.mpf(1) / 4
        phase[5] -= mp.mpf(1) / 4
        raw_x = [Q((3 * i) % 7 - 3, 2**54) for i in range(18)]
        raw_v = [Q((5 * i) % 11 - 5, 2**54) for i in range(18)]
        centered = []
        for raw in (raw_x, raw_v):
            mean = sum(d * value for d, value in zip(degrees, raw)) / 40
            centered.append(tuple(value - mean for value in raw))
        assert (
            sum(d * value**2 for d, value in zip(degrees, centered[0]))
            < report.form_radius**2
        )
        assert (
            sum(d * value**2 for d, value in zip(degrees, centered[1]))
            < report.phase_radius**2
        )
        x = [_mp(value) + mp.mpf(1) / 11 for value in centered[0]]
        theta = [
            value + _mp(error) - mp.mpf(2) / 13
            for value, error in zip(phase, centered[1])
        ]

        def laplacian(values):
            result = [mp.mpf(0) for _ in degrees]
            for i, j in edges:
                delta = values[i] - values[j]
                result[i] += delta
                result[j] -= delta
            return [value / d for value, d in zip(result, degrees)]

        def sine_derivative(vector=None, second=False):
            result = [mp.mpf(0) for _ in degrees]
            for i, j in edges:
                angle = theta[j] - theta[i]
                value = (
                    mp.sin(angle)
                    if vector is None
                    else (
                        -mp.sin(angle) * (vector[j] - vector[i]) ** 2
                        if second
                        else mp.cos(angle) * (vector[j] - vector[i])
                    )
                )
                result[i] += value
                result[j] -= value
            return [value / d for value, d in zip(result, degrees)]

        ax = laplacian(x)
        f = sine_derivative()
        x1 = [-a + gamma * b for a, b in zip(ax, f)]
        theta1 = [gamma * value for value in ax]
        x2 = [-a + gamma * b for a, b in zip(laplacian(x1), sine_derivative(theta1))]
        theta2 = [gamma * value for value in laplacian(x1)]
        x3 = [
            -a + gamma * (b + c)
            for a, b, c in zip(
                laplacian(x2),
                sine_derivative(theta1, second=True),
                sine_derivative(theta2),
            )
        ]
        nominal = gamma * (
            2 * mp.sin(mp.mpf(71) / 50 + mp.mpf(1) / 4)
            - mp.mpf(3) / 2 * mp.sin(mp.mpf(71) / 50 - mp.mpf(1) / 2)
            - mp.sin(mp.mpf(71) / 50) / 2
        )
        assert abs(x2[4] - x2[5] - nominal) <= _mp(
            report.initial_curvature_error_candidate
        )
        assert abs(x3[4] - x3[5]) <= _mp(report.third_derivative_bound_candidate)
        assert (
            sum(d * value**2 for d, value in zip(degrees, x3))
            <= _mp(report.third_derivative_bound_candidate) ** 2
        )
        assert any(abs(value) > 0 for value in theta1)
        assert any(abs(value) > 0 for value in theta2)


def test_rational_informative_budget_without_any_reading():
    h, x, y, delta, g = Q(1, 2**24), Q(1, 2**48), Q(1, 2**48), Q(1, 2**90), Q(1, 3069)
    upper_h = 2 * h
    total = 2 * upper_h
    qmax = (x + g * total * (4 + 3 + 2 * y)) / (1 - 4 * g * g * total * total)
    error = (
        2 * upper_h * qmax + 2 * g * upper_h * y + 4 * g * g * total * upper_h * qmax
    )
    radius = (72000 * error + 144000 * delta) / upper_h
    assert radius < Q(1, 12500)
    wj, wb = 4 * radius, 2 * radius / (Q(1, 4) - 2 * radius)
    assert wj < Q(1, 3000) and wb < Q(1, 1500)
    sine_bound = min(Q(7), 5 + 2 * y + 8 * g * upper_h * qmax)
    eps2 = 4 * (1 + g * g) * x + 4 * g * y
    m3 = (
        8 * (1 + 2 * g * g) * qmax
        + 4 * g * (1 + g * g) * sine_bound
        + 8 * g**3 * qmax * qmax
    )
    error_normalized = 2 * eps2 + 2 * h * m3
    assert error_normalized < Q(1, 10**9)
    lower = Q(1, 18000)
    amplitude, angle = I(Q(1, 4)), I(Q(11, 8), Q(3, 2))
    sine_coefficient = 2 * sin(I(Q(1, 8))) ** 2 * (1 + 3 * cos(amplitude))
    cosine_coefficient = sin(amplitude) * (2 + 3 * cos(amplitude))
    coefficient = (sine_coefficient * sin(angle) + cosine_coefficient * cos(angle)) / (
        1023 * pi_interval()
    )
    assert coefficient.lo > lower
    # |dK/db| <= gamma*(P+R), with P<=2a^2 and R<=5a.
    assert sine_coefficient.hi <= Q(1, 8)
    assert cosine_coefficient.hi <= Q(5, 4)
    assert Q(1, 8) + Q(5, 4) == Q(11, 8)
    wk = 11 * g * wb / 8
    beta = wk / lower
    factor = 2 * wj + beta + 2 * wj * beta
    data_error = 64 * delta / (h * h)
    clock_width = (
        2 * data_error / lower
        + (2 + error_normalized / lower) * factor
        + 2 * error_normalized / lower
    )
    assert clock_width < Q(1, 80)
    assert 2 * wj + 4 * clock_width < Q(1, 16)


def test_tiny_duration_squared_never_becomes_a_zero_interval_denominator():
    arguments = _arguments(Q(1, 2**70))
    assert I(arguments["probe_duration"] ** 2).contains(0)
    result = owner.infer_sine_geometry_gain_clock_curvature(**arguments)
    assert result.status == "bounded_candidate"
    assert result.clock_rate_outer_bounds.contains(Q(7, 5))
    assert result.readout_gain_outer_bounds.contains(Q(9, 7))


def test_positive_quotient_retains_subgrid_denominators_and_signed_numerators():
    tiny = Q(1, 2**300)
    assert I(tiny).contains(0)
    assert owner._positive_quotient_bounds((-tiny, 2 * tiny), (tiny, 2 * tiny)) == I(
        -1, 2
    )


@pytest.mark.parametrize("shift", (-1, 1))
def test_half_time_observation_can_exclude_an_unchanged_coarse_clock_report(
    report, shift
):
    arguments = _arguments()
    readings = list(arguments["recorded_reading_bounds"])
    readings[1] = tuple(value + shift for value in readings[1])
    arguments["recorded_reading_bounds"] = tuple(readings)
    result = owner.infer_sine_geometry_gain_clock_curvature(**arguments)
    assert result.clock_envelope.to_dict() == report.clock_envelope.to_dict()
    assert result.status == "incompatible"
    assert result.clock_rate_outer_bounds is result.readout_gain_outer_bounds is None
    assert result.incompatibility_reasons


def test_unresolved_curvature_keeps_coarse_result_but_no_refined_outputs(monkeypatch):
    monkeypatch.setattr(owner, "sin", lambda _: I(0))
    result = owner.infer_sine_geometry_gain_clock_curvature(**_arguments())
    assert result.clock_envelope.status == "bounded_candidate"
    assert result.status == "unavailable" and not result.curvature_coefficient_positive
    assert (
        result.nominal_bulk_angle_outer_bounds
        is result.clock_rate_outer_bounds
        is result.readout_gain_outer_bounds
        is None
    )
    assert "positive_curvature_coefficient" in result.unavailable_reasons[0]


@pytest.mark.parametrize(
    "override",
    (
        {"phase_increments": (Q(1, 4), Q(1, 4))},
        {"phase_radius": Q(1)},
    ),
)
def test_source_or_leading_rank_failure_does_not_become_curvature_identification(
    override,
):
    result = owner.infer_sine_geometry_gain_clock_curvature(
        **{**_arguments(), **override}
    )
    assert (
        result.status == "unavailable" and not result.base_inverse_enclosure_available
    )
    assert result.curvature_coefficient_bounds is result.clock_rate_outer_bounds is None
    if "phase_radius" in override:
        assert result.normalized_curvature_error_upper_bound is None


@pytest.mark.parametrize(
    "readings",
    (
        ((0, 0),) * 3,
        ((0, 0),) * 5,
        {(0, 0), (1, 1)},
        ((0, 0), (True, 1), (0, 0), (0, 0)),
        ((0, 0), (np.bool_(False), 1), (0, 0), (0, 0)),
        ((0, 0), (0, float("inf")), (0, 0), (0, 0)),
        ((0, 0), I(0), (0, 0), (0, 0)),
        ((0, 0), (1, 0), (0, 0), (0, 0)),
    ),
)
def test_all_four_pairs_are_admitted_before_the_three_reading_child(
    monkeypatch, readings
):
    monkeypatch.setattr(
        owner,
        "infer_sine_geometry_gain_clock",
        lambda **kwargs: pytest.fail("invalid fourth-channel evidence reached child"),
    )
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_geometry_gain_clock_curvature(
            **{**_arguments(), "recorded_reading_bounds": readings}
        )


def test_json_retains_finite_observation_and_coarse_refinement_distinction(report):
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.sine-curvature-inference.v1"
    body = payload["report"]
    assert len(body["recorded_reading_bounds"]) == 4
    assert len(body["clock_envelope"]["recorded_reading_bounds"]) == 3
    assert body["curvature_reading_coefficients"] == [
        {"numerator": value, "denominator": 1} for value in (1, -2, 1, 0)
    ]
    assert body["curvature_refinement_available"]


def test_fresh_full_state_four_reading_flow_preserves_every_intermediate_state():
    # Unreserved numerical implementation crosscheck, not a validated source.
    b, c, gain, rate = Q(73, 50), Q(11, 12), Q(7, 5), Q(6, 5)
    h, a1, a2 = Q(1, 2**12), Q(1, 5), Q(4, 5)
    edges, degrees = _graph()
    degree = np.array(degrees)
    laplacian = np.diag(degree).astype(float)
    for i, j in edges:
        laplacian[i, j] = laplacian[j, i] = -1
    normalized = laplacian / degree[:, None]
    with mp.workdps(80):
        nominal = np.array(tuple(map(float, _mp_phase(_mp(b), _mp(c), degrees))))
    errors = []
    for raw in (
        np.array([(i + 1) / 2**48 for i in range(18)]),
        np.array([(2 * i + 1) / 2**48 for i in range(18)]),
    ):
        raw -= degree @ raw / 40
        assert np.all(raw != 0)
        assert degree @ raw**2 < float(Q(1, 2**80))
        errors.append(raw)
    original = np.r_[errors[0], nominal + errors[1]]
    q = np.zeros(18)
    q[4], q[5] = 1, -1
    gamma = 1 / (1023 * np.pi)

    def flow(_, state):
        currents = np.zeros(18)
        for i, j in edges:
            value = np.sin(state[18 + j] - state[18 + i])
            currents[i] += value
            currents[j] -= value
        gradient = normalized @ state[:18]
        return np.r_[-gradient + gamma * currents / degree, gamma * gradient]

    state = original.copy()
    state[18:] += float(a1) * q
    reached = [original]
    starts = (0, float(rate * h / 2), float(rate * h))
    ends = (float(rate * h / 2), float(rate * h), float(2 * rate * h))
    for index, (start, end) in enumerate(zip(starts, ends)):
        if index == 2:
            state = state.copy()
            state[18:] += float(a2 - a1) * q
        solution = solve_ivp(
            flow, (start, end), state, method="DOP853", rtol=1e-12, atol=1e-16
        )
        assert solution.success
        state = solution.y[:, -1]
        reached.append(state)
    readings = tuple(Q.from_float(float(gain) * (q @ state[:18])) for state in reached)
    arguments = _arguments()
    arguments.update(
        form_radius=Q(1, 2**40),
        phase_radius=Q(1, 2**40),
        phase_increments=(a1, a2),
        probe_duration=h,
        recorded_reading_bounds=tuple((value, value) for value in readings),
        readout_error_bound=Q(1, 10**17),
    )
    result = owner.infer_sine_geometry_gain_clock_curvature(**arguments)
    assert result.status == "bounded_candidate"
    actual = Q.from_float(float(b) - (errors[1][1] - errors[1][0]) / 8)
    assert result.nominal_bulk_angle_outer_bounds.contains(b)
    assert result.actual_long_arc_mean_outer_bounds.contains(actual)
    assert result.effective_gain_outer_bounds.contains(gain * rate)
    assert result.readout_gain_outer_bounds.contains(gain)
    assert result.clock_rate_outer_bounds.contains(rate)


def test_no_frozen_producer_or_shared_integrator_is_used(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        relational_sine_two_port_capture,
        relational_sine_two_port_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("finite-curvature inference attempted response evaluation")

    monkeypatch.setattr(
        relational_sine_two_port_capture, "assess_sine_two_port_capture", forbidden
    )
    monkeypatch.setattr(
        relational_sine_two_port_readout, "bound_sine_two_port_readout", forbidden
    )
    monkeypatch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
    assert (
        owner.infer_sine_geometry_gain_clock_curvature(**_arguments()).status
        == "bounded_candidate"
    )
