"""Independent moment, error and fresh-flow controls for finite apertures."""

import itertools
import json
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature
from math import comb

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.physics import test_sine_curvature_inference as controls
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_aperture_inference as owner
from tnfr.utils.io import json_loads


def _average_power(lo, hi, degree):
    return (hi ** (degree + 1) - lo ** (degree + 1)) / ((degree + 1) * (hi - lo))


def _arguments():
    # Synthetic quadratic first window and affine second window. These are
    # constraints for implementation controls, not an acquired full flow.
    args = controls._arguments()
    points = tuple((lo + hi) / 2 for lo, hi in args.pop("recorded_reading_bounds"))
    coefficients = (
        points[0],
        -3 * points[0] + 4 * points[1] - points[2],
        2 * points[0] - 4 * points[1] + 2 * points[2],
    )
    averages = tuple(
        sum(
            c * _average_power(Q(j, 3), Q(j + 1, 3), k)
            for k, c in enumerate(coefficients)
        )
        for j in range(3)
    ) + ((points[2] + points[3]) / 2,)
    radius = Q(1, 2**90)
    return {
        **args,
        "averaged_reading_bounds": tuple((c - radius, c + radius) for c in averages),
        "clock_rate_derivative_bound": Q(1, 2**22),
    }


@pytest.fixture(scope="module")
def report():
    return owner.infer_sine_geometry_gain_clock_aperture(**_arguments())


def test_eleven_inputs_fixed_apertures_and_mean_statistic(report):
    parameters = signature(owner.infer_sine_geometry_gain_clock_aperture).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 11
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    h = report.probe_duration
    assert report.aperture_windows == (
        (0, h / 3),
        (h / 3, 2 * h / 3),
        (2 * h / 3, h),
        (h, 2 * h),
    )
    assert report.virtual_observation_times == (0, h / 2, h, 2 * h)
    assert report.status == "bounded_candidate"
    assert report.effective_gain_relation == "J=G*rho_bar_1"
    assert not hasattr(report, "clock_rate_outer_bounds")
    for bound, value in (
        (report.nominal_bulk_angle_outer_bounds, Q(71, 50)),
        (report.readout_gain_outer_bounds, Q(9, 7)),
        (report.first_window_mean_clock_rate_outer_bounds, Q(7, 5)),
        (report.effective_gain_outer_bounds, Q(9, 5)),
    ):
        assert bound.contains(value)
    assert (
        report.actual_long_arc_mean_outer_bounds
        == report.nominal_bulk_angle_outer_bounds
        + I(-report.phase_radius / 8, report.phase_radius / 8)
    )
    with pytest.raises(TypeError):
        owner.infer_sine_geometry_gain_clock_aperture(
            **_arguments(), clock_profile=lambda t: 1
        )


def test_exact_moment_matrix_reconstructs_quadratics_and_preserves_offset(report):
    matrix = report.reconstruction_matrix
    for degree in range(3):
        averages = tuple(_average_power(Q(j, 3), Q(j + 1, 3), degree) for j in range(3))
        for row, point in zip(matrix[:3], (Q(0), Q(1, 2), Q(1))):
            assert (
                sum(w * average for w, average in zip(row, averages)) == point**degree
            )
    assert all(sum(row) == 1 for row in matrix)
    assert tuple(matrix[0][j] - 2 * matrix[1][j] + matrix[2][j] for j in range(4)) == (
        Q(9, 4),
        Q(-9, 2),
        Q(9, 4),
        Q(0),
    )
    assert matrix[3] == tuple(2 * (j == 3) - matrix[2][j] for j in range(4))


def _power_polynomial(endpoint, degree):
    return tuple(
        Q(comb(degree, k)) * endpoint ** (degree - k) * (-1) ** k
        for k in range(degree + 1)
    )


def _kernel_polynomial(row, point, middle):
    # Peano kernel L[(t-u)_+^2/2] from the defining average functional.
    result = [Q(0)] * 4
    for j, weight in enumerate(row[:3]):
        lo, hi = Q(j, 3), Q(j + 1, 3)
        if middle < hi:
            for k, value in enumerate(_power_polynomial(hi, 3)):
                result[k] += weight * value / (6 * (hi - lo))
            if middle < lo:
                for k, value in enumerate(_power_polynomial(lo, 3)):
                    result[k] -= weight * value / (6 * (hi - lo))
    if middle < point:
        for k, value in enumerate(_power_polynomial(point, 2)):
            result[k] -= value / 2
    return tuple(result)


def _bernstein_coefficients(polynomial, lo, hi):
    degree = len(polynomial) - 1
    local = tuple(
        sum(
            polynomial[j] * comb(j, k) * lo ** (j - k) * (hi - lo) ** k
            for j in range(k, degree + 1)
        )
        for k in range(degree + 1)
    )
    return tuple(
        sum(local[k] * Q(comb(i, k), comb(degree, k)) for k in range(i + 1))
        for i in range(degree + 1)
    )


def test_exact_peano_absolute_integrals_certify_reconstruction_constants(report):
    breaks = (Q(0), Q(1, 3), Q(1, 2), Q(2, 3), Q(1))
    integrals = []
    for index, (row, point) in enumerate(
        zip(report.reconstruction_matrix[:3], (Q(0), Q(1, 2), Q(1)))
    ):
        absolute_integral = Q(0)
        for lo, hi in zip(breaks, breaks[1:]):
            polynomial = _kernel_polynomial(row, point, (lo + hi) / 2)
            sign = -1 if index == 2 or (index == 1 and lo >= Q(1, 2)) else 1
            assert all(
                sign * value >= 0
                for value in _bernstein_coefficients(polynomial, lo, hi)
            )
            absolute_integral += sign * sum(
                value * (hi ** (k + 1) - lo ** (k + 1)) / (k + 1)
                for k, value in enumerate(polynomial)
            )
        integrals.append(absolute_integral)
    assert integrals == [Q(1, 108), Q(1, 2304), Q(1, 108)]
    averages = tuple(_average_power(Q(j, 3), Q(j + 1, 3), 3) for j in range(3))
    curvature = sum(
        w * a for w, a in zip(report.original_curvature_average_coefficients, averages)
    )
    assert curvature == Q(6, 8)  # Sharp M3*H^3/8 for f(t)=t^3.


def test_last_window_endpoint_bound_does_not_taylor_across_phase_kink(report):
    # f=t^2 before H=1; f=1+10(t-1)+(t-1)^2 afterwards is continuous
    # with a derivative jump. The two separate reconstruction rules remain valid.
    first = tuple(_average_power(Q(j, 3), Q(j + 1, 3), 2) for j in range(3))
    last_average = 1 + Q(10, 2) + Q(1, 3)
    virtual = tuple(
        sum(w * a for w, a in zip(row, first + (last_average,)))
        for row in report.reconstruction_matrix
    )
    assert virtual[:3] == (0, Q(1, 4), 1)
    assert virtual[3] - 12 == Q(-1, 3)  # M2/6=2/6 is attained.
    assert virtual[3] != 4  # Extending the first quadratic ignores the event.


def test_one_child_receives_noiseless_virtual_bands_and_no_hidden_inputs(
    monkeypatch, report
):
    calls = []
    real = owner.infer_sine_geometry_gain_clock_curvature

    def record(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(owner, "infer_sine_geometry_gain_clock_curvature", record)
    result = owner.infer_sine_geometry_gain_clock_aperture(**_arguments())
    assert result == report and len(calls) == 1
    expected = {
        k: v
        for k, v in _arguments().items()
        if k not in ("averaged_reading_bounds", "clock_rate_derivative_bound")
    }
    expected.update(
        recorded_reading_bounds=report.point_reference_reading_bounds,
        readout_error_bound=Q(0),
    )
    assert calls[0] == expected
    assert (
        report.readout_error_bound > 0
        and report.reference_envelope.readout_error_bound == 0
    )


def test_radii_keep_numerical_sensor_clock_and_reconstruction_distinct(report):
    g, h, rate = Q(1, 3069), report.probe_duration, report.clock_rate_bounds[1]
    q0 = report.form_radius + 14 * g * rate * h
    assert report.whole_window_form_norm_upper_bound == q0
    assert report.structural_readout_speed_upper_bound == 2 * q0 + 7 * g
    assert report.second_derivative_bound_upper_bound == 4 * (1 + g * g) * q0 + 14 * g
    assert (
        report.third_derivative_bound_upper_bound
        == 8 * (1 + 2 * g * g) * q0 + 28 * g * (1 + g * g) + 8 * g**3 * q0**2
    )
    assert report.projected_sensor_error_radii == tuple(
        report.readout_error_bound * value
        for value in (Q(10, 3), Q(7, 6), Q(10, 3), Q(16, 3))
    )
    for i, row in enumerate(report.reconstruction_matrix):
        assert report.projected_numerical_radii[i] == sum(
            abs(w) * r for w, r in zip(row, report.averaged_reading_radii)
        )
        assert report.projected_clock_discrepancy_upper_bounds[i] == sum(
            abs(w) * r
            for w, r in zip(row, report.averaged_clock_discrepancy_upper_bounds)
        )
        total = sum(
            values[i]
            for values in (
                report.projected_numerical_radii,
                report.projected_sensor_error_radii,
                report.projected_clock_discrepancy_upper_bounds,
                report.reconstruction_error_upper_bounds,
            )
        )
        center = report.virtual_reading_midpoints[i]
        assert report.point_reference_reading_bounds[i] == (
            center - total,
            center + total,
        )


def test_sixteen_original_sensor_error_corners_and_composed_coefficients(report):
    matrix = report.reconstruction_matrix
    embedded = report.reference_envelope.embedded_inverse_reading_coefficients
    assert embedded is not None
    for signs in itertools.product((-1, 1), repeat=4):
        errors = tuple(sign * report.readout_error_bound for sign in signs)
        virtual = tuple(sum(w * e for w, e in zip(row, errors)) for row in matrix)
        assert all(
            abs(error) <= radius
            for error, radius in zip(virtual, report.projected_sensor_error_radii)
        )
        curvature = virtual[0] - 2 * virtual[1] + virtual[2]
        assert curvature == sum(
            w * e
            for w, e in zip(report.original_curvature_average_coefficients, errors)
        )
        assert abs(curvature) <= 9 * report.readout_error_bound
        for point_row, average_row in zip(
            embedded, report.embedded_inverse_average_coefficients
        ):
            # Each endpoint choice belongs to the coefficient rectangle;
            # it need not be jointly realized by one geometric parameter.
            coefficients = tuple(value.lo for value in point_row)
            direct = sum(c * e for c, e in zip(coefficients, virtual))
            composed = sum((w * e for w, e in zip(average_row, errors)), I(0))
            assert composed.contains(direct)
    assert (
        report.projected_sensor_error_radii[0]
        + 2 * report.projected_sensor_error_radii[1]
        + report.projected_sensor_error_radii[2]
        == 9 * report.readout_error_bound
    )


@pytest.mark.parametrize("mode", ("zero_drift", "singleton_rate", "range_cap"))
def test_average_exposure_zero_and_range_caps(mode):
    arguments = _arguments()
    if mode == "zero_drift":
        arguments["clock_rate_derivative_bound"] = 0
    else:
        arguments["clock_rate_derivative_bound"] = Q(10**400)
    if mode == "singleton_rate":
        arguments["clock_rate_bounds"] = (Q(7, 5), Q(7, 5))
    result = owner.infer_sine_geometry_gain_clock_aperture(**arguments)
    h = result.probe_duration
    width = result.clock_rate_bounds[1] - result.clock_rate_bounds[0]
    expected = (
        (0,) * 4
        if mode != "range_cap"
        else tuple(
            width * h * value for value in (Q(7, 54), Q(13, 54), Q(7, 54), Q(1, 2))
        )
    )
    assert result.averaged_exposure_discrepancy_bounds == expected
    assert any(value > 0 for value in result.reconstruction_error_upper_bounds)


def test_linear_clock_average_exposure_is_enclosed_exactly(report):
    h, slope = report.probe_duration, report.clock_rate_derivative_bound
    first_expected = tuple(
        slope * (h * _average_power(lo, hi, 1) - _average_power(lo, hi, 2)) / 2
        for lo, hi in report.aperture_windows[:3]
    )
    assert first_expected == report.averaged_exposure_discrepancy_bounds[:3]
    # The linear clock attains the Lipschitz envelope on the last aperture too.
    assert report.averaged_exposure_discrepancy_bounds[3] == 5 * slope * h * h / 12


def test_huge_offset_and_subgrid_drift_survive_exact_moment_arithmetic():
    args = {**_arguments(), "clock_rate_derivative_bound": Q(1, 2**400)}
    before = owner.infer_sine_geometry_gain_clock_aperture(**args)
    shift = Q(10**400)
    args["averaged_reading_bounds"] = tuple(
        (lo + shift, hi + shift) for lo, hi in args["averaged_reading_bounds"]
    )
    after = owner.infer_sine_geometry_gain_clock_aperture(**args)
    assert before.averaged_exposure_discrepancy_bounds[0] > 0
    assert I(before.clock_rate_derivative_bound).contains(0)
    assert after.virtual_reading_midpoints == tuple(
        c + shift for c in before.virtual_reading_midpoints
    )
    assert (
        after.reference_envelope.curvature_midpoint
        == before.reference_envelope.curvature_midpoint
    )
    assert after.readout_gain_outer_bounds == before.readout_gain_outer_bounds
    assert (
        after.first_window_mean_clock_rate_outer_bounds
        == before.first_window_mean_clock_rate_outer_bounds
    )


def test_positive_subgrid_horizon_retains_apertures_and_abstains_on_inverse_rank():
    arguments = {**_arguments(), "probe_duration": Q(1, 2**400)}
    result = owner.infer_sine_geometry_gain_clock_aperture(**arguments)
    assert result.probe_duration > 0 and I(result.probe_duration).contains(0)
    assert all(lo < hi for lo, hi in result.aperture_windows)
    assert all(value > 0 for value in result.reconstruction_error_upper_bounds)
    assert result.status == "unavailable" and not result.rank_certified
    assert result.aperture_reconstruction_certified
    assert result.first_window_mean_clock_rate_outer_bounds is None


def test_response_free_rational_conditioning_budget():
    h, x, y, delta, drift, g = (
        Q(1, 2**24),
        Q(1, 2**48),
        Q(1, 2**48),
        Q(1, 2**90),
        Q(1, 2**22),
        Q(1, 3069),
    )
    hstar, total = 2 * h, 4 * h
    q0 = x + 7 * g * total
    speed = 2 * q0 + 7 * g
    m2 = 4 * (1 + g * g) * q0 + 14 * g
    m3 = 8 * (1 + 2 * g * g) * q0 + 28 * g * (1 + g * g) + 8 * g**3 * q0**2
    endpoint, midpoint, last = (
        2 * m3 * hstar**3 / 108,
        2 * m3 * hstar**3 / 2304,
        2 * m2 * hstar**2 / 6,
    )
    clock = 2 * speed * drift * h * h
    r_h = 20 * delta / 3 + 91 * clock / 324 + endpoint
    r_half = 7 * delta / 3 + 11 * clock / 81 + midpoint
    r_last = r_h + 4 * delta + 5 * clock / 6 + last
    q = (x + g * total * (7 + 2 * y)) / (1 - 4 * g * g * total**2)
    error = 2 * hstar * q + 2 * g * hstar * y + 4 * g * g * total * hstar * q
    radius = 36000 * (2 * error + r_h + r_last) / hstar
    assert radius < Q(1, 11700)
    w_j, w_b = 4 * radius, 2 * radius / (Q(1, 4) - 2 * radius)
    assert w_j < Q(1, 2900) and w_b < Q(1, 1450)
    special_sine = 5 + 2 * y + 8 * g * hstar * q
    special_m3 = (
        8 * (1 + 2 * g * g) * q + 4 * g * (1 + g * g) * special_sine + 8 * g**3 * q**2
    )
    eps2 = 4 * (1 + g * g) * x + 4 * g * y
    d = 2 * eps2 + 2 * h * special_m3
    assert d < Q(1, 800000000)
    data = 8 * (2 * r_h + 2 * r_half) / h**2
    assert data < Q(1, 100000000)
    minimum = Q(1, 18000)
    relative = 11 * g * w_b / (8 * minimum)
    factor = 2 * w_j + relative + 2 * w_j * relative
    w_rate = 2 * data / minimum + (2 + d / minimum) * factor + 2 * d / minimum
    assert w_rate < Q(1, 64)
    assert 2 * w_j + 4 * w_rate < Q(1, 16)
    assert w_b + y / 4 < Q(1, 1024) and w_j < Q(1, 2048)


@pytest.mark.parametrize("slope", (Q(-1, 16), Q(1, 16)))
def test_fresh_full_state_boxcar_flow_retains_history_and_mean_rate(slope):
    # Independent unreserved numerical crosscheck, not validated evidence.
    b, c, gain, h = Q(29, 20), Q(5, 6), Q(4, 3), Q(1, 2**12)
    a1, a2, initial_rate = Q(1, 5), Q(4, 5), Q(6, 5)
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

    state = initial.copy()
    state[18:] += float(a1) * q
    averages = []
    boundaries = (Q(0), h / 3, 2 * h / 3, h, 2 * h)
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if index == 3:
            state = state.copy()
            state[18:] += float(a2 - a1) * q

        def flow(time, extended):
            rate = float(initial_rate) + float(slope) * time
            return np.r_[
                rate * structural(extended[:36]), float(gain) * (q @ extended[:18])
            ]

        result = solve_ivp(
            flow,
            (float(start), float(end)),
            np.r_[state, 0.0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-17,
        )
        assert result.success
        state = result.y[:36, -1]
        averages.append(Q.from_float(result.y[36, -1] / float(end - start)) + Q(7, 13))
    args = {
        **_arguments(),
        "form_radius": Q(1, 2**40),
        "phase_radius": Q(1, 2**40),
        "phase_increments": (a1, a2),
        "probe_duration": h,
        "averaged_reading_bounds": tuple((a, a) for a in averages),
        "readout_error_bound": Q(1, 10**15),
        "clock_rate_derivative_bound": abs(slope),
    }
    result = owner.infer_sine_geometry_gain_clock_aperture(**args)
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
    assert mean != initial_rate and mean != initial_rate + slope * h


@pytest.mark.parametrize(
    "override", ({"phase_radius": Q(1)}, {"phase_increments": (Q(1, 4), Q(1, 4))})
)
def test_source_or_rank_failure_retains_candidates_but_not_inferred_values(override):
    result = owner.infer_sine_geometry_gain_clock_aperture(
        **{**_arguments(), **override}
    )
    assert result.status == result.reference_envelope.status == "unavailable"
    assert result.first_window_mean_clock_rate_outer_bounds is None
    assert result.unavailable_reasons == result.reference_envelope.unavailable_reasons
    if "phase_radius" in override:
        assert not result.aperture_reconstruction_certified
        assert result.reconstruction_error_upper_bounds is None
        assert result.averaged_clock_discrepancy_upper_bounds is None
        assert result.projected_clock_discrepancy_upper_bounds is None
        assert result.second_derivative_bound_upper_bound is None
        assert result.third_derivative_bound_upper_bound is None
    else:
        assert result.aperture_reconstruction_certified and result.rank_deficient


def test_strict_necessary_exclusion_is_not_a_source_rejection():
    args = {**_arguments(), "averaged_reading_bounds": ((0, 0), (1, 1), (0, 0), (0, 0))}
    result = owner.infer_sine_geometry_gain_clock_aperture(**args)
    assert result.status == "incompatible" and result.source_admitted
    assert result.readout_gain_outer_bounds is None
    assert (
        result.incompatibility_reasons
        == result.reference_envelope.incompatibility_reasons
    )


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
        {"averaged_reading_bounds": ((0, 0),) * 3},
        {"averaged_reading_bounds": ((0, 0), (True, 1), (0, 0), (0, 0))},
        {"averaged_reading_bounds": ((0, 0), (1, 0), (0, 0), (0, 0))},
        {"averaged_reading_bounds": ((0, 0), I(0), (0, 0), (0, 0))},
    ),
)
def test_invalid_consumed_primitives_reject_before_child(monkeypatch, override):
    monkeypatch.setattr(
        owner,
        "infer_sine_geometry_gain_clock_curvature",
        lambda **kwargs: pytest.fail("invalid primitive reached child"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.infer_sine_geometry_gain_clock_aperture(**{**_arguments(), **override})


def test_json_keeps_averages_sensor_and_virtual_reference_separate(report):
    payload = json_loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.sine-aperture-inference.v1"
    body = payload["report"]
    assert body["readout_error_bound"] == {"numerator": 1, "denominator": 2**90}
    assert body["reference_envelope"]["readout_error_bound"] == {
        "numerator": 0,
        "denominator": 1,
    }
    assert body["averaged_reading_bounds"] != body["point_reference_reading_bounds"]
    assert (
        body["first_window_mean_clock_rate_outer_bounds"]
        == body["reference_envelope"]["clock_rate_outer_bounds"]
    )


def test_no_response_producer_or_integrator_is_called(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        relational_sine_two_port_capture,
        relational_sine_two_port_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("aperture inverse attempted response evaluation")

    monkeypatch.setattr(
        relational_sine_two_port_capture, "assess_sine_two_port_capture", forbidden
    )
    monkeypatch.setattr(
        relational_sine_two_port_readout, "bound_sine_two_port_readout", forbidden
    )
    monkeypatch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
    assert owner.infer_sine_geometry_gain_clock_aperture(
        **_arguments()
    ).inverse_enclosure_available
