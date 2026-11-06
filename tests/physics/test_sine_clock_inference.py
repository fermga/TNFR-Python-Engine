"""Independent clock-envelope controls, without frozen response replay."""

import json
from fractions import Fraction as Q
from inspect import Parameter, signature

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_clock_inference as owner


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _arguments():
    # Artificial leading observations exercise necessary algebra only.
    # Neither a chosen midpoint nor these values assert full-flow realization.
    a1, a2, h, b, effective = Q(1, 5), Q(4, 5), Q(1, 2**24), Q(91, 64), Q(3, 2)
    with mp.workdps(100):
        changes = tuple(
            -2
            * _mp(effective * h)
            * mp.sin(3 * _mp(a) / 2)
            * mp.cos(_mp(b - a / 2))
            / (1023 * mp.pi)
            for a in (a1, a2)
        )
        baseline = mp.mpf(7) / 11
        centers = tuple(
            Q(mp.nstr(value, 90))
            for value in (baseline, baseline + changes[0], baseline + sum(changes))
        )
    radius = Q(1, 10**88)
    return dict(
        bulk_angle_bounds=(Q(11, 8), Q(3, 2)),
        receiver_short_angle_bounds=(Q(2, 3), Q(1)),
        form_radius=Q(1, 2**44),
        phase_radius=Q(1, 2**44),
        phase_increments=(a1, a2),
        probe_duration=h,
        recorded_reading_bounds=tuple((c - radius, c + radius) for c in centers),
        readout_error_bound=Q(1, 2**80),
        readout_gain_bounds=(Q(1), Q(2)),
        clock_rate_bounds=(Q(1, 2), Q(2)),
    )


@pytest.fixture(scope="module")
def report():
    return owner.infer_sine_geometry_gain_clock(**_arguments())


def test_ten_mandatory_primitives_do_not_accept_a_cached_envelope():
    parameters = signature(owner.infer_sine_geometry_gain_clock).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 10
    assert all(
        value.kind == Parameter.KEYWORD_ONLY and value.default == Parameter.empty
        for value in parameters.values()
    )
    with pytest.raises(TypeError):
        owner.infer_sine_geometry_gain_clock(**_arguments(), constraint_envelope=None)


def test_auxiliary_envelope_and_product_marginals_cover_synthetic_truth(report):
    envelope = report.constraint_envelope
    assert report.status == "bounded_candidate" and report.inverse_enclosure_available
    assert report.probe_duration == Q(1, 2**24)
    assert report.structural_probe_duration_bounds == (Q(1, 2**25), Q(1, 2**23))
    assert envelope.probe_duration == report.structural_probe_duration_bounds[1]
    assert (
        report.auxiliary_gain_prior_bounds
        == envelope.readout_gain_bounds
        == (Q(1, 4), Q(2))
    )
    assert report.effective_gain_prior_bounds == (Q(1, 2), Q(4))
    assert report.nominal_bulk_angle_outer_bounds.contains(Q(91, 64))
    assert report.effective_gain_outer_bounds.contains(Q(3, 2))
    assert (
        report.actual_long_arc_mean_outer_bounds
        == report.nominal_bulk_angle_outer_bounds
        + I(-report.phase_radius / 8, report.phase_radius / 8)
    )
    for gain, rate in ((Q(1), Q(3, 2)), (Q(3, 2), Q(1)), (Q(2), Q(3, 4))):
        assert gain * rate == Q(3, 2)
        assert report.readout_gain_outer_bounds.contains(gain)
        assert report.clock_rate_outer_bounds.contains(rate)
    assert report.readout_gain_outer_bounds.width > Q(9, 10)
    assert report.clock_rate_outer_bounds.width > Q(7, 10)
    assert report.effective_gain_relation == "J=G*rho"
    assert not report.whole_window_acute_certified


def test_sensor_noise_and_three_readings_are_not_rescaled(report):
    envelope = report.constraint_envelope
    assert envelope.recorded_reading_bounds == report.recorded_reading_bounds
    assert envelope.readout_error_bound == report.readout_error_bound
    for row, radius in zip(
        envelope.transformed_reading_coefficients,
        envelope.transformed_observation_error_radii,
    ):
        assert radius == sum(
            (r + report.readout_error_bound) * coefficient.abs_max
            for r, coefficient in zip(envelope.reading_radii, row)
        )


def test_uniform_error_uses_complete_monotone_error_over_clock(report):
    x, y, h = report.form_radius, report.phase_radius, report.probe_duration
    a2, g = report.phase_increments[1], Q(1, 3069)

    def error(rate):
        total = 2 * rate * h
        qmax = (x + g * total * (4 + 4 * a2 + 2 * y)) / (1 - 4 * g * g * total * total)
        return (
            2 * rate * h * qmax
            + 2 * g * rate * h * y
            + 4 * g * g * total * rate * h * qmax
        )

    upper = report.clock_rate_bounds[1]
    expected = error(upper) / upper
    assert report.finite_remainder_over_clock_upper_bound == expected
    assert report.constraint_envelope.finite_remainder_upper_bound == error(upper)
    # Rational interior controls retain the common clock in every term.
    for rate in (Q(1, 2), Q(3, 4), Q(1), Q(5, 4), Q(3, 2), Q(2)):
        assert error(rate) / rate <= expected
        for gain in (Q(1), Q(3, 2), Q(2)):
            auxiliary = gain * rate / upper
            assert gain * error(rate) <= auxiliary * error(upper)


def test_rational_conditioning_budget_requires_no_chosen_reading():
    x = y = Q(1, 2**40)
    observed, clock_upper, auxiliary_min = Q(1, 2**22), Q(2), Q(1, 4)
    horizon = observed * clock_upper
    assert horizon == Q(1, 2**21)
    g, total, a2, noise = Q(1, 3069), 2 * horizon, Q(3, 4), Q(1, 2**60)
    qmax = (x + g * total * (4 + 4 * a2 + 2 * y)) / (1 - 4 * g * g * total * total)
    error = (
        2 * horizon * qmax + 2 * g * horizon * y + 4 * g * g * total * horizon * qmax
    )
    # The fixed amplitudes 1/4,3/4 give the independent elementary row
    # bounds S1<41205/(4h), S2<34170/h; retain their joint Euclidean norm.
    row_bounds = (Q(41205, 4) / horizon, Q(34170) / horizon)
    assert sum(bound**2 for bound in row_bounds) < (36000 / horizon) ** 2
    radius = 72000 * error / horizon + 144000 * noise / horizon
    assert radius < Q(1, 3072)
    assert 2 * radius / (auxiliary_min - 2 * radius) + y / 4 < Q(1, 256)
    assert 2 * clock_upper * radius < Q(1, 512)


@pytest.mark.parametrize("rate", (Q(1), Q(3, 2)))
def test_point_clock_prior_reuses_the_exact_known_clock_envelope(rate):
    arguments = _arguments()
    arguments["clock_rate_bounds"] = (rate, rate)
    result = owner.infer_sine_geometry_gain_clock(**arguments)
    known = {
        key: value for key, value in arguments.items() if key != "clock_rate_bounds"
    }
    known["probe_duration"] *= rate
    envelope = owner.infer_sine_two_pulse_geometry_gain(**known)
    assert result.constraint_envelope.to_dict() == envelope.to_dict()
    assert (
        result.nominal_bulk_angle_outer_bounds
        == envelope.nominal_bulk_angle_outer_bounds
    )
    assert result.readout_gain_outer_bounds == envelope.readout_gain_outer_bounds
    assert result.clock_rate_outer_bounds == I(rate)


def test_observed_unit_change_preserves_the_complete_constraint(report):
    arguments = _arguments()
    arguments["probe_duration"] *= 2
    arguments["clock_rate_bounds"] = tuple(
        value / 2 for value in arguments["clock_rate_bounds"]
    )
    changed = owner.infer_sine_geometry_gain_clock(**arguments)
    assert (
        changed.structural_probe_duration_bounds
        == report.structural_probe_duration_bounds
    )
    assert changed.constraint_envelope.to_dict() == report.constraint_envelope.to_dict()
    assert changed.readout_gain_outer_bounds == report.readout_gain_outer_bounds
    assert (
        changed.actual_long_arc_mean_outer_bounds
        == report.actual_long_arc_mean_outer_bounds
    )
    assert (
        changed.finite_remainder_over_clock_upper_bound
        == 2 * report.finite_remainder_over_clock_upper_bound
    )


def test_subgrid_positive_clock_never_becomes_an_interval_zero_denominator():
    arguments = _arguments()
    rate = Q(1, 2**300)
    arguments["clock_rate_bounds"] = (rate, rate)
    arguments["probe_duration"] /= rate
    result = owner.infer_sine_geometry_gain_clock(**arguments)
    assert result.status == "bounded_candidate"
    assert result.clock_rate_bounds == (rate, rate)
    assert result.clock_rate_outer_bounds.contains(rate)
    assert result.clock_rate_outer_bounds.lo == 0  # Honest absolute-grid rounding.
    assert result.effective_gain_outer_bounds.contains(Q(3, 2) * rate)
    assert result.readout_gain_outer_bounds.contains(Q(3, 2))
    assert result.readout_gain_outer_bounds.width < Q(1, 1024)


def test_unavailable_and_incompatible_child_results_keep_missing_projections():
    for overrides, expected in (
        ({"phase_increments": (Q(1, 4), Q(1, 4))}, "unavailable"),
        ({"phase_radius": Q(1)}, "unavailable"),
        ({"recorded_reading_bounds": ((0, 0),) * 3}, "incompatible"),
        ({"phase_increments": (Q(1, 4), Q(1, 4) + Q(1, 10**100))}, "unavailable"),
    ):
        result = owner.infer_sine_geometry_gain_clock(**{**_arguments(), **overrides})
        assert result.status == expected and not result.inverse_enclosure_available
        assert (
            result.effective_gain_outer_bounds
            is result.readout_gain_outer_bounds
            is result.clock_rate_outer_bounds
            is None
        )
        if "phase_radius" in overrides:
            assert not result.source_admitted
            assert result.finite_remainder_over_clock_upper_bound is None


def test_full_graph_curvature_distinguishes_equal_leading_gain_clock_products():
    # Independent full 18-node graph sums, without any response producer.
    edges = sorted(
        {tuple(sorted((o + j, o + (j + 1) % 9))) for o in (0, 9) for j in range(9)}
        | {(0, 9), (1, 10)}
    )
    degree = tuple(sum(i in edge for edge in edges) for i in range(18))
    with mp.workdps(85):
        b, c, amplitude = mp.pi / 2 - mp.mpf(1) / 8, mp.mpf(3) / 4, mp.mpf(1) / 4
        short, bulk = 4 * mp.pi - 8 * b, (2 * mp.pi - c) / 8
        shift = (short - c) / 2
        theta = [mp.mpf(0)] + [short + (j - 1) * b for j in range(1, 9)]
        theta += [shift] + [shift + c + (j - 1) * bulk for j in range(1, 9)]
        theta[4] += amplitude
        theta[5] -= amplitude
        currents = [mp.mpf(0) for _ in degree]
        for i, j in edges:
            value = mp.sin(theta[j] - theta[i])
            currents[i] += value
            currents[j] -= value
        forcing = [value / d for value, d in zip(currents, degree)]
        gradient = [mp.mpf(0) for _ in degree]
        for i, j in edges:
            value = forcing[i] - forcing[j]
            gradient[i] += value
            gradient[j] -= value
        af = [value / d for value, d in zip(gradient, degree)]
        factor = 2 * mp.sin(mp.mpf(3) / 8) * mp.sin(mp.mpf(1) / 4)
        assert abs(forcing[4] - forcing[5] + factor) < mp.mpf("1e-80")
        assert abs(af[4] - af[5] + 3 * factor / 2) < mp.mpf("1e-80")
        gamma = 1 / (1023 * mp.pi)
        derivatives = tuple(
            (
                _mp(gain * rate) * gamma * (forcing[4] - forcing[5]),
                -_mp(gain * rate * rate) * gamma * (af[4] - af[5]),
            )
            for gain, rate in ((Q(3, 2), Q(1)), (Q(1), Q(3, 2)))
        )
        assert derivatives[0][0] == derivatives[1][0] < 0
        assert abs(derivatives[1][1] - 3 * derivatives[0][1] / 2) < mp.mpf("1e-80")
        assert derivatives[1][1] > derivatives[0][1] > 0
        # For x(0)=0 the initial phase rate vanishes, but its acceleration
        # gamma^2*A*f is nonzero and also scales with rho^2.
        assert abs(gamma * gamma * (af[4] - af[5])) > 0


@pytest.mark.parametrize("rate", (Q(3, 5), Q(7, 5)))
def test_fresh_full_state_flow_covers_geometry_gain_and_clock_without_reset(rate):
    # This independent numerical implementation control is not a validated
    # response, retained research observation or frozen protocol replay.
    b, c, gain = Q(71, 50), Q(7, 8), Q(9, 7)
    observed = Q(1, 2**20)
    a1, a2 = Q(1, 5), Q(4, 5)
    edges = sorted(
        {tuple(sorted((o + j, o + (j + 1) % 9))) for o in (0, 9) for j in range(9)}
        | {(0, 9), (1, 10)}
    )
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    degree = np.array(degrees)
    laplacian = np.diag(degree).astype(float)
    for i, j in edges:
        laplacian[i, j] = laplacian[j, i] = -1
    normalized = laplacian / degree[:, None]
    gamma = 1 / (1023 * np.pi)
    dipole = np.zeros(18)
    dipole[4], dipole[5] = 1, -1
    short, bulk = 4 * np.pi - 8 * float(b), (2 * np.pi - float(c)) / 8
    shift = (short - float(c)) / 2
    nominal = np.array(
        [0]
        + [short + (j - 1) * float(b) for j in range(1, 9)]
        + [shift]
        + [shift + float(c) + (j - 1) * bulk for j in range(1, 9)]
    )
    nominal -= degree @ nominal / 40
    residuals = []
    for raw in (
        tuple(Q(i + 1, 2**55) for i in range(18)),
        tuple(Q(2 * i + 1, 2**55) for i in range(18)),
    ):
        mean = sum(d * value for d, value in zip(degrees, raw)) / 40
        centered = tuple(value - mean for value in raw)
        assert all(centered)
        assert sum(d * value for d, value in zip(degrees, centered)) == 0
        assert sum(d * value**2 for d, value in zip(degrees, centered)) < Q(1, 2**80)
        residuals.append(centered)
    form = float(Q(1, 11)) + np.array(tuple(map(float, residuals[0])))
    phase = nominal + float(-Q(2, 13)) + np.array(tuple(map(float, residuals[1])))
    materialized_form = form - degree @ form / 40
    materialized_phase_error = phase - nominal
    materialized_phase_error -= degree @ materialized_phase_error / 40
    assert degree @ materialized_form**2 < float(Q(1, 2**80))
    assert degree @ materialized_phase_error**2 < float(Q(1, 2**80))

    def complete_flow(_, state):
        x, theta = state[:18], state[18:]
        current = np.zeros(18)
        for i, j in edges:
            value = np.sin(theta[j] - theta[i])
            current[i] += value
            current[j] -= value
        gradient = normalized @ x
        return np.r_[-gradient + gamma * current / degree, gamma * gradient]

    original = np.r_[form, phase]
    first_initial = original.copy()
    first_initial[18:] += float(a1) * dipole
    structural_horizon = float(rate * observed)
    first = solve_ivp(
        complete_flow,
        (0, structural_horizon),
        first_initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-15,
    )
    assert first.success
    carried = first.y[:, -1]
    assert np.linalg.norm(carried[:18] - original[:18]) > 1e-12
    assert np.linalg.norm(complete_flow(structural_horizon, carried)[18:]) > 1e-15
    second_initial = carried.copy()
    second_initial[18:] += float(a2 - a1) * dipole
    second = solve_ivp(
        complete_flow,
        (structural_horizon, 2 * structural_horizon),
        second_initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-15,
    )
    assert second.success
    reached = (original, carried, second.y[:, -1])
    readings = tuple(
        Q.from_float(float(gain) * (dipole @ state[:18]) + float(-Q(3, 11)))
        for state in reached
    )
    arguments = _arguments()
    arguments.update(
        form_radius=Q(1, 2**40),
        phase_radius=Q(1, 2**40),
        phase_increments=(a1, a2),
        probe_duration=observed,
        recorded_reading_bounds=tuple((value, value) for value in readings),
        readout_error_bound=Q(1, 10**14),
    )
    result = owner.infer_sine_geometry_gain_clock(**arguments)
    actual_mean = b - (residuals[1][1] - residuals[1][0]) / 8
    assert result.status == "bounded_candidate"
    assert result.nominal_bulk_angle_outer_bounds.contains(b)
    assert result.actual_long_arc_mean_outer_bounds.contains(actual_mean)
    assert result.readout_gain_outer_bounds.contains(gain)
    assert result.clock_rate_outer_bounds.contains(rate)
    assert result.effective_gain_outer_bounds.contains(gain * rate)
    assert result.actual_long_arc_mean_outer_bounds.width < Q(1, 32)
    assert result.effective_gain_outer_bounds.width < Q(1, 16)


def test_json_projection_preserves_parent_units_and_auxiliary_semantics(report):
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.sine-clock-inference.v1"
    body = payload["report"]
    assert body["probe_duration"] == {"numerator": 1, "denominator": 2**24}
    assert body["constraint_envelope"]["probe_duration"] == {
        "numerator": 1,
        "denominator": 2**23,
    }
    assert body["effective_gain_relation"] == "J=G*rho"
    assert (
        "auxiliary_upper_duration_is_not_the_actual_unknown_trajectory_duration"
        in body["scope"]
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("clock_rate_bounds", (0, 1)),
        ("clock_rate_bounds", (2, 1)),
        ("clock_rate_bounds", (True, 2)),
        ("clock_rate_bounds", (np.bool_(True), 2)),
        ("clock_rate_bounds", (1, float("inf"))),
        ("clock_rate_bounds", (1,)),
        ("clock_rate_bounds", {1, 2}),
        ("clock_rate_bounds", I(1, 2)),
        ("probe_duration", 0),
        ("probe_duration", True),
        ("probe_duration", Q(1, 4) + Q(1, 10**100)),
        ("probe_duration", mp.mpf("1e-4000")),
        ("readout_gain_bounds", (0, 1)),
        ("readout_gain_bounds", (1, float("nan"))),
    ),
)
def test_invalid_new_primitives_are_rejected_before_auxiliary_inference(
    monkeypatch, field, value
):
    monkeypatch.setattr(
        owner,
        "infer_sine_two_pulse_geometry_gain",
        lambda **kwargs: pytest.fail("invalid primitive reached inverse"),
    )
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_geometry_gain_clock(**{**_arguments(), field: value})


def test_no_forward_or_frozen_producer_is_needed(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        relational_sine_two_port_capture,
        relational_sine_two_port_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("clock constraint attempted a trajectory")

    monkeypatch.setattr(
        relational_sine_two_port_capture, "assess_sine_two_port_capture", forbidden
    )
    monkeypatch.setattr(
        relational_sine_two_port_readout, "bound_sine_two_port_readout", forbidden
    )
    monkeypatch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
    assert (
        owner.infer_sine_geometry_gain_clock(**_arguments()).status
        == "bounded_candidate"
    )
