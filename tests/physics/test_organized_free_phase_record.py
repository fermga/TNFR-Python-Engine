"""Independent static controls for finite records of the free phase probe.

No response, source, sampled record or trajectory is evaluated. The only
prospective calculation below evaluates the budget frozen at fd5dc157.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics._organized_free_phase_probe import _derive_organized_free_phase_probe
from tnfr.physics._organized_free_phase_record import _bound_organized_free_phase_record

FROZEN_BUDGET = dict(
    form_error_bound=Q(0),
    phase_error_bound=Q(1, 10**6),
    sample_intervals=10000,
    quadrature_subdivisions=100,
    raw_form_error_bound=Q(1, 10**11),
    exposure_time_error_bound=Q(1, 10**18),
    kinetic_relative_error_bound=Q(1, 10**18),
    initial_phase_readout_error_bound=Q(1, 10**10),
    numerical_coefficient_error_bound=Q(1, 10**8),
    harmonic=9,
)


@pytest.fixture(scope="module", autouse=True)
def no_response_source_or_trajectory_evaluation():
    with forbid_organized_response_evaluation():
        yield


@pytest.fixture(scope="module")
def frozen_budget():
    return _bound_organized_free_phase_record(**FROZEN_BUDGET)


def test_energy_residual_identity_retains_both_endpoint_terms():
    v, h, delta, gamma, current = sp.symbols("v h delta gamma J", real=True)
    n = sp.Symbol("n", integer=True, positive=True)
    residual = (v**2 - h**2) / (2 * gamma**2)

    def derivative(expression):
        return (
            sp.diff(expression, v) * (-v - gamma**2 * current)
            - sp.diff(expression, h) * h
            + sp.diff(expression, delta) * v
        )

    assert sp.simplify(derivative(residual) + 2 * residual + v * current) == 0
    integrand = residual * (n * v * sp.cos(n * delta) - 2 * sp.sin(n * delta))
    boundary = residual * sp.sin(n * delta)
    assert sp.simplify(integrand - derivative(boundary)) == current * v * sp.sin(
        n * delta
    )
    assert sp.factor(residual) == (v - h) * (v + h) / (2 * gamma**2)
    # A full reconstructed increment does not erase a phase-readout offset.
    # Noisy initial readings can also make the initial residual nonzero.
    initial_phase = sp.pi / (2 * n)
    endpoint_difference = 2 * sp.sin(n * (initial_phase + 2 * sp.pi)) - sp.sin(
        n * initial_phase
    )
    assert sp.simplify(endpoint_difference) == 1


def test_calibration_affects_reconstruction_and_residual_normalization(frozen_budget):
    v, vhat, gamma, gammahat, h = sp.symbols("v vhat gamma gammahat h", positive=True)
    residual = (v**2 - h**2) / (2 * gamma**2)
    observed = (vhat**2 - h**2) / (2 * gammahat**2)
    decomposition = (vhat**2 - v**2) / (2 * gammahat**2) + (
        gamma**2 / gammahat**2 - 1
    ) * residual
    assert sp.simplify(observed - residual - decomposition) == 0
    # Raw-form normalization cancels gamma only when the same gamma and
    # correlated initial raw reference are used in both squared terms.
    raw, raw_initial, decay = sp.symbols("raw raw_initial decay", real=True)
    assert (
        sp.simplify(residual.subs({v: gamma * raw, h: gamma * raw_initial * decay}))
        == (raw**2 - raw_initial**2 * decay**2) / 2
    )

    arguments = dict(FROZEN_BUDGET, kinetic_relative_error_bound=Q(1, 1000))
    calibrated = _bound_organized_free_phase_record(**arguments)
    assert calibrated.sample_velocity_error_upper_bound >= Q(12, 1000)
    assert calibrated.phase_reconstruction_error_upper_bound > (
        frozen_budget.phase_reconstruction_error_upper_bound
    )
    assert calibrated.kinetic_gamma_squared_lower_bound < (
        frozen_budget.kinetic_gamma_squared_lower_bound
    )
    assert calibrated.normalization_relative_error_upper_bound == Q(1000, 999) ** 2 - 1
    assert calibrated.energy_residual_error_upper_bound > (
        calibrated.true_scaled_energy_residual_upper_bound
        * calibrated.normalization_relative_error_upper_bound
    )
    assert calibrated.coefficient_error_upper_bound > Q(1, 4374)


def test_fine_current_derivative_cancellation_and_second_derivative_constant(
    frozen_budget,
):
    # The nine actual same-slot gaps have a zero mean, as do their rates.
    q = sp.Matrix([sp.Rational(k - 4, 1000) for k in range(9)])
    rate = sp.Matrix([sp.Rational((3 * k) % 9 - 3, 700) for k in range(9)])
    assert sum(q) == sum(rate) == 0
    derivative_at_mean, curvature = sp.symbols("jprime curvature", real=True)
    fine_derivative = (
        sum((derivative_at_mean + curvature * q[k]) * rate[k] for k in range(9)) / 9
    )
    assert sp.expand(fine_derivative) == curvature * q.dot(rate) / 9
    assert q.dot(rate) ** 2 <= q.dot(q) * rate.dot(rate)
    shifted_rate = rate + sp.ones(9, 1) / 1000
    assert sp.diff(
        sum((derivative_at_mean + curvature * q[k]) * shifted_rate[k] for k in range(9))
        / 9,
        derivative_at_mean,
    ) == sp.Rational(1, 1000)

    # For a gap q_A-q_B, the two difference-map norms supply factor 2.
    # q<=2 sqrt(F), gamma*z<=sqrt(2F), ||Lfine||<5, and a=gamma^2/1000.
    gamma, fine, j2 = sp.symbols("gamma fine j2", positive=True)
    a = gamma**2 / 1000
    bound = 2 * j2 / 9 * 2 * sp.sqrt(fine) * 5 * sp.sqrt(2 * fine) / (2 * a)
    assert sp.simplify(gamma**2 * bound) == 10000 * sp.sqrt(2) * fine * j2 / 9
    assert 10000 * sp.sqrt(2) / 9 < sp.Rational(5000, 3)
    report = frozen_budget
    g2 = report.probe_bound.gamma_bounds.hi**2
    j0, j1, j2 = report.probe_bound.current_derivative_bounds
    assert report.decay_residual_second_derivative_upper_bound == (
        g2 * (2 * j0 + 12 * j1)
        + Q(5000, 3) * j2 * report.probe_bound.fine_energy_upper_bound
    )


def test_piecewise_linear_interpolation_bound_is_sharp_for_independent_quadratic():
    t, width, curvature, slope, offset = sp.symbols(
        "t width curvature slope offset", real=True
    )
    residual = offset + slope * t + curvature * t**2 / 2
    left, right = residual.subs(t, -width / 2), residual.subs(t, width / 2)
    interpolant = (left + right) / 2 + t * (right - left) / width
    assert sp.expand(interpolant - residual) == sp.expand(
        curvature * (width**2 / 8 - t**2 / 2)
    )
    assert sp.expand((interpolant - residual).subs(t, 0)) == curvature * width**2 / 8
    noise = sp.Symbol("noise", nonnegative=True)
    assert sp.expand((interpolant + noise - residual).subs(t, 0)) == (
        noise + curvature * width**2 / 8
    )
    # Adjacent independent nodal errors also contribute to the PL slope.
    assert sp.simplify(((right + noise) - (left - noise)) / width) == (
        slope + 2 * noise / width
    )


def test_reconstructed_increment_stop_has_transport_error_without_phase_reset():
    # Independent constant signals make the stopping calculation exact; these
    # are controls of the observation argument, not solutions of the TNFR law.
    speed, error = Q(8), Q(1, 10)
    actual_stop = 2 * sp.pi / speed
    reconstructed_stop = 2 * sp.pi / (speed + error)
    assert sp.simplify(actual_stop - reconstructed_stop) == sp.pi / 324
    assert pi_interval().hi / 324 < error / Q(11, 3)
    actual_phase_gap = sp.simplify(2 * sp.pi - speed * reconstructed_stop)
    assert actual_phase_gap == 2 * sp.pi / 81
    assert 2 * pi_interval().hi / 81 < error
    # Adding an uncertain initial readout changes registration but not the
    # observed increment or its stop; no actual phase is reset here.
    origin = sp.Symbol("origin", real=True)
    reconstructed_phase = origin + (speed + error) * reconstructed_stop
    assert sp.simplify(reconstructed_phase - origin) == 2 * sp.pi


def test_panelwise_residual_and_integrand_derivatives_need_no_record():
    h, w, slope, delta, gamma = sp.symbols("h w slope delta gamma", real=True)
    n = sp.Symbol("n", integer=True, positive=True)
    speed = h + w

    def derivative(expression):
        return (
            -h * sp.diff(expression, h)
            + slope * sp.diff(expression, w)
            + speed * sp.diff(expression, delta)
        )

    residual = (speed**2 - h**2) / (2 * gamma**2)
    first = (-h * w + speed * slope) / gamma**2
    second = (h * (w - 2 * slope) + slope**2) / gamma**2
    assert sp.simplify(derivative(residual) - first) == 0
    assert sp.simplify(derivative(first) - second) == 0
    factor = n * speed * sp.cos(n * delta) - 2 * sp.sin(n * delta)
    rate = -h + slope
    first_factor = (n * rate - 2 * n * speed) * sp.cos(n * delta) - (
        n**2 * speed**2 * sp.sin(n * delta)
    )
    second_factor = (n * h - 2 * n * rate - n**3 * speed**3) * sp.cos(n * delta) + (
        -3 * n**2 * speed * rate + 2 * n**2 * speed**2
    ) * sp.sin(n * delta)
    assert sp.simplify(derivative(factor) - first_factor) == 0
    assert sp.simplify(derivative(first_factor) - second_factor) == 0
    assert (
        sp.simplify(
            derivative(derivative(residual * factor))
            - (second * factor + 2 * first * first_factor + residual * second_factor)
        )
        == 0
    )


def test_midpoint_remainder_and_interpolation_knot_negative_control():
    t, width = sp.symbols("t width", positive=True)
    curvature, slope, offset = sp.symbols("curvature slope offset", real=True)
    polynomial = curvature * t**2 / 2 + slope * t + offset
    integral = sp.integrate(polynomial, (t, 0, width))
    midpoint = width * polynomial.subs(t, width / 2)
    assert sp.simplify(integral - midpoint) == curvature * width**3 / 24
    # A cell crossing a PL knot cannot use its a.e. zero second derivative.
    assert sp.integrate(t, (t, 0, width)) * 2 == width**2
    assert width * 2 * abs(Q(0)) == 0


def test_frozen_finite_budget_resolves_the_declared_arithmetic_control(frozen_budget):
    report = frozen_budget
    assert report.sample_count == 10001
    assert report.quadrature_evaluation_count_upper_bound == 1000000
    assert report.sample_step == Q(1, 10000)
    assert report.quadrature_step == Q(1, 1000000)
    assert Q(0) < report.coefficient_error_upper_bound < Q(219, 10**7) < Q(1, 4374)
    assert report.reconstructed_velocity_lower_bound > 0
    assert report.full_turn_reconstruction_margin > 0
    assert report.endpoint_time_error_upper_bound == (
        report.velocity_reconstruction_error_upper_bound / Q(11, 3)
    )
    assert report.phase_reconstruction_error_upper_bound == (
        report.initial_phase_readout_error_bound
        + report.velocity_reconstruction_error_upper_bound
    )
    assert report.quadrature_numerator_error_upper_bound == (
        report.integrand_second_derivative_upper_bound * report.quadrature_step**2 / 24
    )
    assert report.coefficient_error_upper_bound == (
        report.fine_harmonic_error_upper_bound
        + (
            report.observation_numerator_error_upper_bound
            + report.quadrature_numerator_error_upper_bound
        )
        / pi_interval().lo
        + report.numerical_coefficient_error_bound
    )
    assert report.conditional_on_preparation_and_error_bounds
    assert (
        "no_law_selection_sensor_capability_or_physical_admission_verdict"
        in report.scope
    )
    with pytest.raises(FrozenInstanceError):
        report.sample_intervals = 1


def test_family_bound_is_uniform_and_readout_origin_does_not_change_turn(frozen_budget):
    report = frozen_budget
    for deformation in (Q(0), Q(1, 7), Q(1)):
        own = _derive_organized_free_phase_probe(
            deformation=deformation,
            form_error_bound=report.form_error_bound,
            phase_error_bound=report.phase_error_bound,
        )
        assert all(
            local <= common
            for local, common in zip(
                own.current_derivative_bounds,
                report.probe_bound.current_derivative_bounds,
            )
        )
        assert own.harmonic_defect_upper_bound <= report.fine_harmonic_error_upper_bound
    shifted = _bound_organized_free_phase_record(
        **dict(FROZEN_BUDGET, initial_phase_readout_error_bound=Q(1, 10))
    )
    assert (
        shifted.endpoint_time_error_upper_bound
        == report.endpoint_time_error_upper_bound
    )
    assert (
        shifted.full_turn_reconstruction_margin
        == report.full_turn_reconstruction_margin
    )
    assert shifted.coefficient_error_upper_bound > report.coefficient_error_upper_bound


def test_fixed_width_fraction_components_normalize_before_budget_arithmetic(
    frozen_budget,
):
    arguments = {
        field: (
            Q(np.int64(value.numerator), np.int64(value.denominator))
            if isinstance(value, Q)
            else value
        )
        for field, value in FROZEN_BUDGET.items()
    }
    report = _bound_organized_free_phase_record(**arguments)
    for field, value in FROZEN_BUDGET.items():
        if isinstance(value, Q):
            normalized = getattr(report, field)
            assert type(normalized.numerator) is type(normalized.denominator) is int
            assert normalized == value
    assert report == frozen_budget


def test_carrier_ambiguity_scale_does_not_disappear_under_normalization():
    # This bounds two possible laws without constructing either response.
    law_gap_bound = (Q(1) + Q(1, 2)) / 2187
    assert law_gap_bound == Q(1, 1458)
    gamma = 1 / (1023 * pi_interval())
    gronwall_exponent = gamma.hi**2 * Q(245, 243)
    assert 0 < gronwall_exponent < Q(1, 2)
    # exp(z)<=1/(1-z) for 0<=z<1, from the nonnegative Taylor series.
    raw_difference_bound = gamma.hi * law_gap_bound / (1 - gronwall_exponent)
    assert (
        gamma.hi * law_gap_bound < raw_difference_bound < 2 * gamma.hi * law_gap_bound
    )
    first, second = sp.symbols("first second", real=True)
    middle = (first + second) / 2
    assert sp.expand(middle - first) == sp.expand(second - middle)
    assert sp.expand(2 * (middle - first)) == second - first


@pytest.mark.parametrize(
    "field,value",
    (
        ("form_error_bound", True),
        ("phase_error_bound", float("inf")),
        ("raw_form_error_bound", False),
        ("raw_form_error_bound", -1),
        ("raw_form_error_bound", float("nan")),
        ("exposure_time_error_bound", float("inf")),
        ("kinetic_relative_error_bound", True),
        ("kinetic_relative_error_bound", Q(1)),
        ("kinetic_relative_error_bound", Q(11, 10)),
        ("initial_phase_readout_error_bound", -1),
        ("initial_phase_readout_error_bound", float("nan")),
        ("numerical_coefficient_error_bound", False),
        ("numerical_coefficient_error_bound", float("inf")),
        ("sample_intervals", True),
        ("sample_intervals", 0),
        ("sample_intervals", Q(2)),
        ("quadrature_subdivisions", -1),
        ("quadrature_subdivisions", 2.0),
        ("harmonic", 0),
        ("harmonic", False),
        ("harmonic", Q(9)),
    ),
)
def test_invalid_primitive_budgets_reject(field, value):
    with pytest.raises((TypeError, ValueError)):
        _bound_organized_free_phase_record(**dict(FROZEN_BUDGET, **{field: value}))


def test_failure_of_sufficient_stop_or_fine_margin_rejects_without_model_verdict():
    with pytest.raises(ValueError, match="reconstructed-turn"):
        _bound_organized_free_phase_record(
            **dict(FROZEN_BUDGET, exposure_time_error_bound=Q(1, 10))
        )
    with pytest.raises(ValueError, match="F0_upper"):
        _bound_organized_free_phase_record(
            **dict(FROZEN_BUDGET, phase_error_bound=Q(1, 100))
        )
    with pytest.raises(TypeError):
        _bound_organized_free_phase_record(
            **dict(
                FROZEN_BUDGET,
                raw_form_error_bound=_bound_organized_free_phase_record(
                    **FROZEN_BUDGET
                ),
            )
        )
