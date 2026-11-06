"""Independent clock-exposure controls; no frozen response is evaluated."""

import itertools
import json
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.physics import test_sine_curvature_inference as controls
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_clock_drift_inference as owner
from tnfr.utils.io import json_loads


def _arguments():
    # Existing independent leading/quadratic constraints, not acquired data.
    return {**controls._arguments(), "clock_rate_derivative_bound": Q(1, 2**22)}


@pytest.fixture(scope="module")
def report():
    return owner.infer_sine_geometry_gain_clock_drift(**_arguments())


def test_eleven_primitive_inputs_and_mean_rate_statistic(report):
    parameters = signature(owner.infer_sine_geometry_gain_clock_drift).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 11
    assert all(
        item.kind == Parameter.KEYWORD_ONLY and item.default == Parameter.empty
        for item in parameters.values()
    )
    assert report.status == "bounded_candidate"
    assert report.effective_gain_relation == "J=G*rho_bar_1"
    assert not hasattr(report, "clock_rate_outer_bounds")
    assert report.first_window_mean_clock_rate_outer_bounds.contains(Q(7, 5))
    assert report.effective_gain_outer_bounds.contains(Q(9, 5))
    assert report.readout_gain_outer_bounds.contains(Q(9, 7))
    assert report.nominal_bulk_angle_outer_bounds.contains(Q(71, 50))
    assert report.actual_long_arc_mean_outer_bounds == (
        report.nominal_bulk_angle_outer_bounds
        + I(-report.phase_radius / 8, report.phase_radius / 8)
    )
    for extra in ("clock_profile", "reference_envelope"):
        with pytest.raises(TypeError):
            owner.infer_sine_geometry_gain_clock_drift(
                **_arguments(), **{extra: report}
            )


def test_one_reference_call_uses_only_exact_expanded_primitive_pairs(
    monkeypatch, report
):
    calls = []
    real = owner.infer_sine_geometry_gain_clock_curvature

    def record(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(owner, "infer_sine_geometry_gain_clock_curvature", record)
    result = owner.infer_sine_geometry_gain_clock_drift(**_arguments())
    assert result == report and len(calls) == 1
    child_inputs = calls[0]
    expected = {
        key: value
        for key, value in _arguments().items()
        if key != "clock_rate_derivative_bound"
    }
    expected["recorded_reading_bounds"] = report.comparison_reading_bounds
    assert child_inputs == expected
    assert report.recorded_reading_bounds == _arguments()["recorded_reading_bounds"]
    assert (
        report.reference_envelope.recorded_reading_bounds
        == report.comparison_reading_bounds
    )
    assert report.reference_envelope.readout_error_bound == report.readout_error_bound


@pytest.mark.parametrize("point_prior", (False, True))
def test_zero_drift_or_singleton_prior_reduces_exactly_to_constant_clock(point_prior):
    arguments = _arguments()
    if point_prior:
        arguments["clock_rate_bounds"] = (Q(7, 5), Q(7, 5))
        arguments["clock_rate_derivative_bound"] = Q(10**400)
    else:
        arguments["clock_rate_derivative_bound"] = Q(0)
    result = owner.infer_sine_geometry_gain_clock_drift(**arguments)
    original = owner.infer_sine_geometry_gain_clock_curvature(
        **{
            key: value
            for key, value in arguments.items()
            if key != "clock_rate_derivative_bound"
        }
    )
    assert result.reference_envelope == original
    assert result.exposure_discrepancy_bounds == (0,) * 4
    assert result.recorded_discrepancy_upper_bounds == (0,) * 4
    assert result.comparison_reading_bounds == result.recorded_reading_bounds
    assert (
        result.first_window_mean_clock_rate_outer_bounds
        == original.clock_rate_outer_bounds
    )


def test_global_contraction_and_four_asymmetric_bands(report):
    h, g = report.probe_duration, Q(1, 3069)
    total = 2 * report.clock_rate_bounds[1] * h
    norm = report.form_radius + 7 * g * total
    speed = 2 * norm + 7 * g
    derivative = report.clock_rate_derivative_bound
    assert report.structural_duration_upper_bound == total
    assert (
        report.whole_window_form_norm_candidate
        == report.whole_window_form_norm_upper_bound
        == norm
    )
    assert (
        report.structural_readout_speed_candidate
        == report.structural_readout_speed_upper_bound
        == speed
    )
    assert report.exposure_discrepancy_bounds == (
        0,
        derivative * h**2 / 8,
        0,
        derivative * h**2,
    )
    for index, ((lo, hi), expanded, exposure) in enumerate(
        zip(
            report.recorded_reading_bounds,
            report.comparison_reading_bounds,
            report.exposure_discrepancy_bounds,
        )
    ):
        allowance = report.readout_gain_bounds[1] * speed * exposure
        assert report.recorded_discrepancy_candidates[index] == allowance
        assert expanded == (lo - allowance, hi + allowance)
    assert (
        report.clock_drift_transfer_certified
        and report.finite_curvature_bound_certified
    )


def test_large_drift_is_capped_by_the_pointwise_rate_range():
    arguments = {**_arguments(), "clock_rate_derivative_bound": Q(10**400)}
    result = owner.infer_sine_geometry_gain_clock_drift(**arguments)
    h = result.probe_duration
    width = result.clock_rate_bounds[1] - result.clock_rate_bounds[0]
    assert result.exposure_discrepancy_bounds == (0, width * h / 4, 0, width * h)
    assert all(isinstance(value, Q) for value in result.recorded_discrepancy_candidates)


def test_subgrid_positive_drift_and_huge_common_offset_are_preserved():
    arguments = {**_arguments(), "clock_rate_derivative_bound": Q(1, 2**400)}
    shift = Q(10**400)
    before = owner.infer_sine_geometry_gain_clock_drift(**arguments)
    arguments["recorded_reading_bounds"] = tuple(
        (lo + shift, hi + shift) for lo, hi in arguments["recorded_reading_bounds"]
    )
    after = owner.infer_sine_geometry_gain_clock_drift(**arguments)
    assert I(before.clock_rate_derivative_bound).contains(0)
    assert before.exposure_discrepancy_bounds[1] > 0
    assert (
        before.recorded_discrepancy_upper_bounds
        == after.recorded_discrepancy_upper_bounds
    )
    assert (
        after.reference_envelope.curvature_midpoint
        == before.reference_envelope.curvature_midpoint
    )
    assert (
        after.first_window_mean_clock_rate_outer_bounds
        == before.first_window_mean_clock_rate_outer_bounds
    )
    assert after.readout_gain_outer_bounds == before.readout_gain_outer_bounds


def test_all_reading_error_corners_keep_sensor_and_transfer_budgets_separate(report):
    child = report.reference_envelope
    half, end = report.recorded_discrepancy_candidates[1::2]
    raw_radii = tuple((hi - lo) / 2 for lo, hi in report.recorded_reading_bounds)
    assert child.curvature_observation_radius == (
        raw_radii[0]
        + 2 * raw_radii[1]
        + raw_radii[2]
        + 4 * report.readout_error_bound
        + 2 * half
    )
    base = child.clock_envelope.constraint_envelope
    for row, radius in zip(
        base.transformed_reading_coefficients, base.transformed_observation_error_radii
    ):
        original = sum(
            (raw_radii[i] + report.readout_error_bound) * coefficient.abs_max
            for i, coefficient in zip((0, 2, 3), row)
        )
        assert radius == original + end * row[2].abs_max
    for signs in itertools.product((-1, 1), repeat=4):
        errors = tuple(sign * report.readout_error_bound for sign in signs)
        curvature_error = errors[0] - 2 * errors[1] + errors[2]
        assert abs(curvature_error) <= 4 * report.readout_error_bound
        for row in child.embedded_inverse_reading_coefficients:
            # Exact endpoint support avoids adding outward interval arithmetic
            # slack to the mathematical sensor-error combination being checked.
            lower = sum(
                min(coefficient.lo * error, coefficient.hi * error)
                for coefficient, error in zip(row, errors)
            )
            upper = sum(
                max(coefficient.lo * error, coefficient.hi * error)
                for coefficient, error in zip(row, errors)
            )
            assert max(abs(lower), abs(upper)) <= sum(
                coefficient.abs_max * report.readout_error_bound for coefficient in row
            )


@pytest.mark.parametrize("slope", (Q(-3, 16), Q(3, 16)))
def test_linear_clocks_attain_both_derivative_exposure_bounds(slope):
    h, origin = Q(1, 8), Q(6, 5)
    mean = origin + slope * h / 2

    def exposure(time):
        return origin * time + slope * time**2 / 2

    assert exposure(h) == mean * h
    assert abs(exposure(h / 2) - mean * h / 2) == abs(slope) * h**2 / 8
    assert abs(exposure(2 * h) - 2 * mean * h) == abs(slope) * h**2
    assert origin != mean != origin + slope * h


def test_distinct_smooth_clock_profiles_have_identical_segment_exposures():
    # rho_+/-=r +/- eps*cos(4*pi*s/H) have different pointwise values even
    # at all four readings, but exactly equal exposure at each event/reading.
    h, mean, amplitude = Q(1, 32), Q(5, 4), Q(1, 20)
    rate_lower, rate_upper = mean - amplitude, mean + amplitude
    derivative_bound = 4 * Q(22, 7) * amplitude / h
    assert rate_lower > 0
    with mp.workdps(90):
        hh, rr, aa = map(controls._mp, (h, mean, amplitude))
        omega = 4 * mp.pi / hh
        assert aa * omega < controls._mp(derivative_bound)
        for time in (Q(0), h / 2, h, 2 * h):
            s = controls._mp(time)
            perturbation = aa * mp.sin(omega * s) / omega
            assert abs(perturbation) < mp.mpf("1e-85")
            plus, minus = rr + aa * mp.cos(omega * s), rr - aa * mp.cos(omega * s)
            assert abs(plus - minus) > aa
            assert abs((rr * s + perturbation) - (rr * s - perturbation)) < mp.mpf(
                "1e-85"
            )
        assert controls._mp(rate_lower) <= rr <= controls._mp(rate_upper)


def test_rational_informative_drift_budget_needs_no_recorded_response():
    h, x, y, noise, drift, g = (
        Q(1, 2**24),
        Q(1, 2**48),
        Q(1, 2**48),
        Q(1, 2**90),
        Q(1, 2**22),
        Q(1, 3069),
    )
    upper_h = 2 * h
    total = 2 * upper_h
    qmax = (x + g * total * (7 + 2 * y)) / (1 - 4 * g * g * total * total)
    finite_error = (
        2 * upper_h * qmax + 2 * g * upper_h * y + 4 * g * g * total * upper_h * qmax
    )
    global_form = x + 7 * g * total
    speed = 2 * global_form + 7 * g
    half, end = 2 * speed * drift * h**2 / 8, 2 * speed * drift * h**2
    radius = (72000 * finite_error + 144000 * noise + 36000 * end) / upper_h
    effective_width = 4 * radius
    angle_width = 2 * radius / (Q(1, 4) - 2 * radius)
    assert effective_width < Q(1, 3000) and angle_width < Q(1, 1500)
    sine_bound = min(Q(7), 5 + 2 * y + 8 * g * upper_h * qmax)
    eps2 = 4 * (1 + g * g) * x + 4 * g * y
    m3 = (
        8 * (1 + 2 * g * g) * qmax
        + 4 * g * (1 + g * g) * sine_bound
        + 8 * g**3 * qmax * qmax
    )
    error = 2 * eps2 + 2 * h * m3
    assert error < Q(1, 10**9)
    data_error = (64 * noise + 16 * half) / h**2
    assert data_error < Q(1, 2**28)
    minimum = Q(1, 18000)
    relative_coefficient_width = 11 * g * angle_width / (8 * minimum)
    factor = (
        2 * effective_width
        + relative_coefficient_width
        + 2 * effective_width * relative_coefficient_width
    )
    rate_width = (
        2 * data_error / minimum + (2 + error / minimum) * factor + 2 * error / minimum
    )
    assert rate_width < Q(1, 80)
    assert 2 * effective_width + 4 * rate_width < Q(1, 16)


@pytest.mark.parametrize("slope", (Q(-1, 16), Q(1, 16)))
def test_fresh_complete_variable_clock_history_matches_reference_at_first_event(slope):
    # Independent unreserved full-state integration checks the transfer. The
    # theorem supplies the bound; this numerical solver is not validated data.
    b, c, gain = Q(29, 20), Q(5, 6), Q(4, 3)
    h, a1, a2, initial_rate = Q(1, 2**12), Q(1, 5), Q(4, 5), Q(6, 5)
    mean = initial_rate + slope * h / 2
    edges, degrees = controls._graph()
    degree = np.array(degrees)
    laplacian = np.diag(degree).astype(float)
    for i, j in edges:
        laplacian[i, j] = laplacian[j, i] = -1
    normalized = laplacian / degree[:, None]
    with mp.workdps(90):
        nominal = np.array(
            tuple(
                map(
                    float, controls._mp_phase(controls._mp(b), controls._mp(c), degrees)
                )
            )
        )
    residuals = []
    for raw in (
        np.array([(i + 2) / 2**54 for i in range(18)]),
        np.array([(3 * i + 1) / 2**54 for i in range(18)]),
    ):
        raw -= degree @ raw / 40
        assert np.all(raw != 0) and degree @ raw**2 < float(Q(1, 2**80))
        residuals.append(raw)
    initial = np.r_[residuals[0] + 1 / 17, nominal + residuals[1] - 2 / 19]
    q = np.zeros(18)
    q[4], q[5] = 1, -1
    gamma = 1 / (1023 * np.pi)

    def structural(state):
        currents = np.zeros(18)
        for i, j in edges:
            value = np.sin(state[18 + j] - state[18 + i])
            currents[i] += value
            currents[j] -= value
        gradient = normalized @ state[:18]
        return np.r_[-gradient + gamma * currents / degree, gamma * gradient]

    def history(variable):
        state = initial.copy()
        state[18:] += float(a1) * q
        result = [initial.copy()]
        for index, (start, end) in enumerate(zip((0, h / 2, h), (h / 2, h, 2 * h))):
            if index == 2:
                state = state.copy()
                state[18:] += float(a2 - a1) * q

            def flow(time, values):
                rate = (
                    float(initial_rate) + float(slope) * time
                    if variable
                    else float(mean)
                )
                return rate * structural(values)

            solution = solve_ivp(
                flow,
                (float(start), float(end)),
                state,
                method="DOP853",
                rtol=1e-12,
                atol=1e-16,
            )
            assert solution.success
            state = solution.y[:, -1]
            result.append(state)
        return tuple(result)

    actual, reference = history(True), history(False)
    assert np.allclose(actual[2], reference[2], rtol=0, atol=2e-14)
    assert np.linalg.norm(actual[1] - reference[1]) > 1e-14
    assert np.linalg.norm(actual[3] - reference[3]) > 1e-14
    readings = tuple(
        Q.from_float(float(gain) * float(q @ state[:18]) + 7 / 13) for state in actual
    )
    arguments = {
        **_arguments(),
        "form_radius": Q(1, 2**40),
        "phase_radius": Q(1, 2**40),
        "phase_increments": (a1, a2),
        "probe_duration": h,
        "recorded_reading_bounds": tuple((value, value) for value in readings),
        "readout_error_bound": Q(1, 10**15),
        "clock_rate_derivative_bound": abs(slope),
    }
    result = owner.infer_sine_geometry_gain_clock_drift(**arguments)
    assert result.status == "bounded_candidate"
    original_mean = Q.from_float(float(b) - (residuals[1][1] - residuals[1][0]) / 8)
    for bound, truth in (
        (result.nominal_bulk_angle_outer_bounds, b),
        (result.actual_long_arc_mean_outer_bounds, original_mean),
        (result.readout_gain_outer_bounds, gain),
        (result.effective_gain_outer_bounds, gain * mean),
        (result.first_window_mean_clock_rate_outer_bounds, mean),
    ):
        assert bound.contains(truth)
    for index in (1, 3):
        measured_difference = float(gain) * abs(
            q @ (actual[index][:18] - reference[index][:18])
        )
        assert (
            measured_difference
            <= float(result.recorded_discrepancy_upper_bounds[index]) + 1e-15
        )
    assert mean != initial_rate and mean != initial_rate + slope * h


@pytest.mark.parametrize(
    "override", ({"phase_radius": Q(1)}, {"phase_increments": (Q(1, 4), Q(1, 4))})
)
def test_source_or_rank_unavailability_keeps_reference_scope(override):
    result = owner.infer_sine_geometry_gain_clock_drift(**{**_arguments(), **override})
    assert result.status == result.reference_envelope.status == "unavailable"
    assert result.unavailable_reasons == result.reference_envelope.unavailable_reasons
    assert result.first_window_mean_clock_rate_outer_bounds is None
    if "phase_radius" in override:
        assert not result.clock_drift_transfer_certified
        assert result.whole_window_form_norm_upper_bound is None
        assert result.structural_readout_speed_upper_bound is None
        assert result.recorded_discrepancy_upper_bounds is None
    else:
        assert result.clock_drift_transfer_certified and result.rank_deficient


def test_strict_constraint_exclusion_passes_through_without_missing_as_zero():
    arguments = _arguments()
    arguments["recorded_reading_bounds"] = (
        (Q(0), Q(0)),
        (Q(1), Q(1)),
        (Q(0), Q(0)),
        (Q(0), Q(0)),
    )
    result = owner.infer_sine_geometry_gain_clock_drift(**arguments)
    assert result.status == "incompatible"
    assert (
        result.incompatibility_reasons
        == result.reference_envelope.incompatibility_reasons
    )
    assert result.first_window_mean_clock_rate_outer_bounds is None
    assert result.readout_gain_outer_bounds is None


@pytest.mark.parametrize(
    "override",
    (
        {"clock_rate_derivative_bound": True},
        {"clock_rate_derivative_bound": np.bool_(False)},
        {"clock_rate_derivative_bound": float("nan")},
        {"clock_rate_derivative_bound": float("inf")},
        {"clock_rate_derivative_bound": Q(-1, 10**400)},
        {"clock_rate_derivative_bound": Decimal("1e-400")},
        {"clock_rate_bounds": (0, 1)},
        {"clock_rate_bounds": (2, 1)},
        {"clock_rate_bounds": (True, 2)},
        {"readout_gain_bounds": (0, 1)},
        {"readout_gain_bounds": (1, float("inf"))},
        {"probe_duration": 0},
        {"probe_duration": True},
        {"probe_duration": Q(1)},
        {"form_radius": Q(-1)},
        {"form_radius": True},
        {"phase_radius": Q(-1)},
        {"readout_error_bound": Q(-1)},
        {"phase_increments": (0, 1)},
        {"phase_increments": (Q(3, 4), Q(1, 4))},
        {"bulk_angle_bounds": (1, Q(3, 2))},
        {"receiver_short_angle_bounds": (0, 1)},
        {"recorded_reading_bounds": ((0, 0),) * 3},
        {"recorded_reading_bounds": ((0, 0), (True, 1), (0, 0), (0, 0))},
        {"recorded_reading_bounds": ((0, 0), (1, 0), (0, 0), (0, 0))},
        {"recorded_reading_bounds": ((0, 0), I(0), (0, 0), (0, 0))},
    ),
)
def test_invalid_consumed_primitive_rejects_before_reference(monkeypatch, override):
    monkeypatch.setattr(
        owner,
        "infer_sine_geometry_gain_clock_curvature",
        lambda **kwargs: pytest.fail("invalid primitive reached reference calculation"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.infer_sine_geometry_gain_clock_drift(**{**_arguments(), **override})


def test_json_retains_original_readings_and_mean_reference_semantics(report):
    payload = json_loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.sine-clock-drift-inference.v1"
    body = payload["report"]
    assert body["clock_rate_derivative_bound"] == {"numerator": 1, "denominator": 2**22}
    assert (
        body["readout_error_bound"] == body["reference_envelope"]["readout_error_bound"]
    )
    assert body["recorded_reading_bounds"] != body["comparison_reading_bounds"]
    assert (
        body["first_window_mean_clock_rate_outer_bounds"]
        == body["reference_envelope"]["clock_rate_outer_bounds"]
    )
    assert "clock_rate_outer_bounds" not in body


def test_no_frozen_producer_or_shared_integrator_is_used(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        relational_sine_two_port_capture,
        relational_sine_two_port_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("clock-drift inference attempted response evaluation")

    monkeypatch.setattr(
        relational_sine_two_port_capture, "assess_sine_two_port_capture", forbidden
    )
    monkeypatch.setattr(
        relational_sine_two_port_readout, "bound_sine_two_port_readout", forbidden
    )
    monkeypatch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
    assert owner.infer_sine_geometry_gain_clock_drift(
        **_arguments()
    ).inverse_enclosure_available
