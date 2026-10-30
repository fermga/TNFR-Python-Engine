"""Static periodic-envelope bounds and a scoped majorant obstruction.

All lower bounds below concern explicitly chosen upper-bound budgets, never
the actual reference defect, trajectory error, or nonlinear maintenance.
"""

from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def test_first_integral_gives_uniform_derivative_bounds_without_profile_sampling():
    modulus, excursion = sp.symbols("m s", real=True)
    # s=1-(F/b)^2 lies in [0,m], with 0<m<1. The first integral gives
    # F'^2/(q_*^2 b^2)=s*(m-s), not a sampled derivative estimate.
    squared_speed = excursion * (modulus - excursion)
    assert (
        sp.expand(modulus**2 / 4 - squared_speed - (excursion - modulus / 2) ** 2) == 0
    )
    squared_acceleration = (1 - excursion) * (2 * excursion - modulus) ** 2
    acceleration_slack = (
        4 * excursion * (modulus - excursion)
        + excursion * (2 * excursion - modulus) ** 2
    )
    assert sp.expand(modulus**2 - squared_acceleration - acceleration_slack) == 0
    position = sp.Symbol("X", real=True)
    profile = sp.Function("F")(position)
    height, width = sp.symbols("b q_star", positive=True)
    acceleration = width**2 * ((2 - modulus) * profile - 2 * profile**3 / height**2)
    third_derivative = sp.diff(acceleration, position)
    expected = (
        width**2
        * ((2 - modulus) - 6 * profile**2 / height**2)
        * sp.diff(profile, position)
    )
    assert sp.expand(third_derivative - expected) == 0
    third_coefficient = -4 - modulus + 6 * excursion
    assert (
        sp.expand(
            (4 + modulus) ** 2
            - third_coefficient**2
            - 12 * excursion * (4 + modulus - 3 * excursion)
        )
        == 0
    )
    # Its final factor is >=4-2m>0. Together these identities yield
    # |F'|<=q_* b m/2, |F''|<=q_*^2 b m,
    # |F'''|<=q_*^3 b m(4+m)/2 throughout the continuous profile.
    assert sp.expand(4 + modulus - 3 * excursion - (4 - 2 * modulus)) == 3 * (
        modulus - excursion
    )
    assert squared_speed.subs({modulus: 0, excursion: 0}) == 0


def test_periodicity_removes_bookkeeping_scale_from_the_reference_and_bounds():
    epsilon, kappa, rho, size, period, dispersion, nonlinear = sp.symbols(
        "eps kappa rho N K d g", positive=True
    )
    modulus = sp.Symbol("m", real=True)
    physical_height, inverse_width = epsilon * rho, epsilon * kappa
    derivative_bounds = (
        rho,
        kappa * rho * modulus / 2,
        kappa**2 * rho * modulus,
        kappa**3 * rho * modulus * (4 + modulus) / 2,
    )
    physical_bounds = (
        physical_height,
        inverse_width * physical_height * modulus / 2,
        inverse_width**2 * physical_height * modulus,
        inverse_width**3 * physical_height * modulus * (4 + modulus) / 2,
    )
    for order, (unscaled, physical) in enumerate(
        zip(derivative_bounds, physical_bounds)
    ):
        assert sp.expand(epsilon ** (order + 1) * unscaled - physical) == 0
    imposed_kappa = 2 * period / (epsilon * size)
    imposed_rho = imposed_kappa * sp.sqrt(2 * dispersion / nonlinear)
    fixed_height = sp.simplify(physical_height.subs(rho, imposed_rho))
    fixed_width = sp.simplify(inverse_width.subs(kappa, imposed_kappa))
    assert fixed_height == 2 * period * sp.sqrt(2 * dispersion / nonlinear) / size
    assert fixed_width == 2 * period / size
    assert sp.diff(fixed_height, epsilon) == 0
    assert sp.diff(fixed_width, epsilon) == 0
    physical_slow_frequency = epsilon**2 * dispersion * kappa**2 * (2 - modulus)
    assert (
        sp.simplify(
            physical_slow_frequency - dispersion * inverse_width**2 * (2 - modulus)
        )
        == 0
    )


def test_complex_edge_amplitude_derivative_retains_translation_and_slow_phase():
    time, site, carrier, velocity, slow_frequency = sp.symbols(
        "t j k v omega", real=True
    )
    profile = sp.Function("F")
    position = site + velocity * time
    phase = sp.exp(sp.I * carrier * site - sp.I * slow_frequency * time)
    edge = profile(position) - sp.exp(sp.I * carrier) * profile(position + 1)
    amplitude = phase * edge
    translation_derivative = sp.diff(edge, time)
    expected_derivative = (
        phase * translation_derivative - sp.I * slow_frequency * amplitude
    )
    assert sp.simplify(sp.diff(amplitude, time) - expected_derivative) == 0
    height = sp.Symbol("b", positive=True)
    constant_amplitude = phase * height * (1 - sp.exp(sp.I * carrier))
    assert (
        sp.simplify(
            sp.diff(constant_amplitude, time)
            + sp.I * slow_frequency * constant_amplitude
        )
        == 0
    )
    assert (
        sp.diff(constant_amplitude, time).subs(
            {carrier: sp.pi / 4, slow_frequency: 1, time: 0}
        )
        != 0
    )
    value, next_value, first, next_first = sp.symbols("F F_next Fprime Fprime_next")
    incidence_symbol = 1 - sp.exp(sp.I * carrier)
    assert (
        sp.expand(
            value
            - sp.exp(sp.I * carrier) * next_value
            - incidence_symbol * value
            + sp.exp(sp.I * carrier) * (next_value - value)
        )
        == 0
    )
    assert (
        sp.expand(
            first
            - sp.exp(sp.I * carrier) * next_first
            - incidence_symbol * first
            + sp.exp(sp.I * carrier) * (next_first - first)
        )
        == 0
    )
    # These two exact decompositions yield C0=D_k*b+B1 and
    # C1=|v|*(D_k*B1+B2)+omega*C0 under the derivative bounds above.


def test_moving_curvature_remainder_keeps_harmonics_slow_motion_and_sine_error():
    leading, harmonic, fast_rate, slow_rate, harmonic_rate = sp.symbols(
        "a0 ah afprime asprime ahprime", real=True
    )
    total = leading + harmonic
    total_rate = fast_rate + slow_rate + harmonic_rate
    actual_coefficient = -sp.sin(total) * total_rate / 2
    principal_coefficient = -leading * fast_rate / 2
    split = (
        -(
            (sp.sin(total) - total) * total_rate
            + harmonic * total_rate
            + leading * (slow_rate + harmonic_rate)
        )
        / 2
    )
    assert sp.expand(actual_coefficient - principal_coefficient - split) == 0
    assert sp.diff(split, slow_rate) != 0
    assert sp.diff(split, harmonic_rate) != 0
    assert sp.simplify(split.subs({harmonic: 0, slow_rate: 0, harmonic_rate: 0})) != 0
    # The norm bound therefore needs (Ch+G^3/6)*a_* and
    # C0*(C1+Ch1), each with its factor 1/2.


def test_square_root_majorant_has_the_published_positive_feedback_allowance():
    growth, coercivity, derivative_budget, radius, residual_budget, correction = (
        sp.symbols("c delta a_star rho R_star k_star", positive=True)
    )
    alpha = growth / coercivity
    beta = (1 + correction) / sp.sqrt(coercivity)
    time = sp.Symbol("T", positive=True)
    # This is a selected constant-budget comparison with zero initial error.
    majorant = beta * residual_budget * (sp.exp(alpha * time) - 1) / alpha
    assert (
        sp.simplify(sp.diff(majorant, time) - alpha * majorant - beta * residual_budget)
        == 0
    )
    quadratic_term = beta * residual_budget * alpha * time**2 / 2
    curvature_floor = derivative_budget * radius / 6
    ratio_floor = sp.simplify(quadratic_term.subs(growth, curvature_floor) / radius)
    assert (
        sp.simplify(
            ratio_floor
            - (1 + correction)
            * derivative_budget
            * residual_budget
            * time**2
            / (12 * coercivity ** sp.Rational(3, 2))
        )
        == 0
    )
    exponent = sp.Symbol("z", nonnegative=True)
    exponential_remainder = sp.exp(exponent) - 1 - exponent - exponent**2 / 2
    assert exponential_remainder.subs(exponent, 0) == 0
    assert sp.diff(exponential_remainder, exponent).subs(exponent, 0) == 0
    assert sp.diff(exponential_remainder, exponent, 2).subs(exponent, 0) == 0
    assert sp.diff(exponential_remainder, exponent, 3) == sp.exp(exponent)
    # Thus exp(z)>=1+z+z^2/2 for z>=0. For admitted delta<=1,
    # this gives Q/rho>=a_* R_* T^2/12; it is not an actual error lower bound.


def test_inherited_uniform_budget_obstruction_is_independent_of_height_selection():
    height, nonlinear, modulus, sigma, sine_budget, carrier, edge_factor, size = (
        sp.symbols("b g m sigma C lambda Dk N", positive=True)
    )
    phase_time = 2 / (nonlinear * height**2 * (2 - modulus))
    assert (
        sp.simplify(
            phase_time
            - 1 / (nonlinear * height**2)
            - modulus / (nonlinear * height**2 * (2 - modulus))
        )
        == 0
    )
    # These are floors of the inherited *upper-bound expressions*. They
    # are not lower bounds on the true defect or moving reference speed.
    derivative_floor = carrier * edge_factor * height
    residual_floor = 2 * sp.sqrt(size) * (2 * sigma * height + sine_budget * height**5)
    time_floor = 1 / (nonlinear * height**2)
    comparison_floor = sp.simplify(
        derivative_floor * residual_floor * time_floor**2 / 12
    )
    expected = (
        sp.sqrt(size)
        * carrier
        * edge_factor
        * (2 * sigma / height**2 + sine_budget * height**2)
        / (6 * nonlinear**2)
    )
    assert sp.simplify(comparison_floor - expected) == 0
    amplitude_slack = (sp.sqrt(2 * sigma) / height - sp.sqrt(sine_budget) * height) ** 2
    assert (
        sp.simplify(
            2 * sigma / height**2
            + sine_budget * height**2
            - 2 * sp.sqrt(2 * sigma * sine_budget)
            - amplitude_slack
        )
        == 0
    )
    # The minimizer of this algebraic expression is only a proof device;
    # no profile parameter is selected and no response is evaluated.
    assert (
        sp.simplify(
            amplitude_slack.subs(height, (2 * sigma / sine_budget) ** sp.Rational(1, 4))
        )
        == 0
    )


def test_actual_cycle_and_loss_constants_force_the_inherited_majorant_above_radius():
    size = 1024
    frequency = 1 - sp.sqrt(2) / 2
    edge_factor = sp.sqrt(2 * frequency)
    nonlinear = frequency**2 / 8
    sine_budget = edge_factor**5 / 120
    assert frequency * edge_factor > Q(3, 16)
    assert nonlinear < Q(1, 72)
    assert sine_budget > Q(1, 512)
    pi = pi_interval()
    assert pi.lo / (size**2 - 1) > Q(3, size**2)
    lower_bound = (
        sp.sqrt(size)
        * sp.Rational(3, 16)
        * sp.sqrt(2 * sp.Rational(3, size**2) * sp.Rational(1, 512))
        / (3 * sp.Rational(1, 72) ** 2)
    )
    assert sp.simplify(lower_bound - sp.Rational(81, 128) * sp.sqrt(3)) == 0
    assert lower_bound**2 > 1
    # Bootstrap requires Q<rho*sqrt(delta)<=rho. Its chosen uniform
    # majorant already exceeds rho under these constants for every b>0.
    assert lower_bound > 1


def test_site_scale_residual_is_the_existing_two_row_budget_without_epsilon():
    eps, height, first, second, third, d, g, sigma, edge = sp.symbols(
        "eps b B1 B2 B3 d g sigma D", positive=True
    )
    a0, a1 = sp.symbols("A0 A1", nonnegative=True)
    b0, b1, b2, b3 = height / eps, first / eps**2, second / eps**3, third / eps**4
    slow = d * b2 + g * b0**3
    gap = eps * (edge * b0 + eps * b1) + 2 * eps**3 * a0 * b0**3
    old = (
        eps**4 * (b3 / 6 + 3 * a1 * b0**2 * b1 + b1 * (edge * b0 + eps * b1) ** 2 / 2)
        + eps**5 * a0 * (3 * b0**2 * slow + 3 * b0 * b1**2 + 3 * b0**2 * b2 / 2)
        + eps**3 * a0 * b0**3 * gap**2
        + gap**5 / 120
        + 2 * sigma * (eps * b0 + eps**3 * a0 * b0**3)
    )
    physical_gap = edge * height + first + 2 * a0 * height**3
    physical = (
        third / 6
        + 3 * a1 * height**2 * first
        + first * (edge * height + first) ** 2 / 2
        + a0
        * (
            3 * height**2 * (d * second + g * height**3)
            + 3 * height * first**2
            + 3 * height**2 * second / 2
        )
        + a0 * height**3 * physical_gap**2
        + physical_gap**5 / 120
        + 2 * sigma * (height + a0 * height**3)
    )
    assert sp.expand(old - physical) == 0
    assert sp.diff(sp.expand(old), eps) == 0
    assert sp.diff(physical, sigma) == 2 * (height + a0 * height**3)
    # The budget retains damping, even though its true signed defect can
    # cancel against other terms. Its positive summands are allowances.


def test_periodic_fourier_bounds_preserve_aliasing_and_derivative_information():
    size = 8
    coefficients = {
        0: Q(1, 3),
        1: Q(1, 4),
        -1: Q(1, 4),
        9: Q(-1, 8),
        -9: Q(-1, 8),
        7: Q(1, 16),
        -7: Q(1, 16),
    }
    sampled = [Q(0) for _ in range(size)]
    differentiated = [sp.Integer(0) for _ in range(size)]
    for mode, value in coefficients.items():
        sampled[mode % size] += value
        differentiated[mode % size] += sp.I * 2 * sp.pi * mode * value / size
    assert sum(abs(value) for value in sampled) <= sum(
        abs(value) for value in coefficients.values()
    )
    assert sampled[1] == Q(3, 16)
    assert sp.simplify(differentiated[1] - sp.I * 2 * sp.pi * sampled[1] / size) != 0
    assert differentiated[0] == 0
    shifted = sum(
        value * sp.exp(sp.I * sp.pi * mode / size)
        for mode, value in coefficients.items()
        if mode % size == 1
    )
    assert sp.simplify(shifted - Q(5, 16) * sp.exp(sp.I * sp.pi / size)) == 0
    assert sp.simplify(shifted - sampled[1] * sp.exp(sp.I * sp.pi / size)) != 0
    # Continuous differentiation precedes aliasing. Differentiating only
    # the representative sampled mode would discard high-mode derivatives.
    rms_derivative_squared = sum(
        (2 * sp.pi * mode / size) ** 2 * value**2
        for mode, value in coefficients.items()
    )
    nonzero_absolute_sum = sum(
        abs(value) for mode, value in coefficients.items() if mode
    )
    assert (
        sp.simplify(size**2 * rms_derivative_squared / 12 - nonzero_absolute_sum**2) > 0
    )
    n = sp.Symbol("n", integer=True, positive=True)
    assert sp.summation(2 / n**2, (n, 1, sp.oo)) == sp.pi**2 / 3
    assert (
        sp.simplify((size / (2 * sp.pi)) ** 2 * sp.pi**2 / 3 - size**2 / sp.Integer(12))
        == 0
    )


def test_full_mode_mask_has_the_wiener_schur_constants_including_metric_derivative():
    # Independent static matrices, not a prepared envelope or response.
    size = 8
    incidence = np.eye(size) - np.roll(np.eye(size), -1, axis=1)
    laplacian = incidence.T @ incidence
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    positive = eigenvalues > 1e-12
    inverse_sqrt = (
        eigenvectors[:, positive] / np.sqrt(eigenvalues[positive])
    ) @ eigenvectors[:, positive].T
    edge_map = incidence @ inverse_sqrt
    amplitudes = incidence @ np.array([1, 2j, -2, 1j, 3, -1j, 0, 2], dtype=complex)
    derivative = incidence @ np.array([2j, 1, -1j, 3, 0, 2, -1, 1j], dtype=complex)
    carrier_frequency = 1 - np.cos(np.pi / 4)
    cutoff = carrier_frequency / 2
    modes, frequencies, columns = [], [], []
    for mode in range(1, size):
        vector = np.exp(2j * np.pi * mode * np.arange(size) / size) / np.sqrt(size)
        for sign in (-1, 1):
            modes.append(mode)
            frequencies.append(sign * (1 - np.cos(2 * np.pi * mode / size)))
            columns.append(np.concatenate((vector, -1j * sign * vector)) / np.sqrt(2))
    basis = np.column_stack(columns)
    skew = np.block(
        [
            [np.zeros_like(laplacian), -laplacian / 2],
            [laplacian / 2, np.zeros_like(laplacian)],
        ]
    )
    np.testing.assert_allclose(
        skew @ basis, basis * (1j * np.array(frequencies)), atol=2e-15
    )
    h = np.zeros((2 * size, 2 * size), dtype=complex)
    hdot = np.zeros_like(h)
    h[size:, size:] = (
        1j * carrier_frequency * edge_map.T @ np.diag(amplitudes**2) @ edge_map / 4
    )
    hdot[size:, size:] = (
        1j
        * carrier_frequency
        * edge_map.T
        @ np.diag(amplitudes * derivative)
        @ edge_map
        / 2
    )
    congruence = basis.T @ h @ basis
    derivative_congruence = basis.T @ hdot @ basis
    squared_coefficients = np.fft.fft(amplitudes**2) / size
    expected_magnitude = np.array(
        [
            [
                carrier_frequency * abs(squared_coefficients[(-q - r) % size]) / 8
                for r in modes
            ]
            for q in modes
        ]
    )
    np.testing.assert_allclose(abs(congruence), expected_magnitude, atol=4e-15)
    divisors = 2 * carrier_frequency + np.array(frequencies)[:, None] + frequencies
    near_mask = abs(divisors) <= cutoff
    near = np.where(near_mask, congruence, 0)
    correction, correction_derivative = np.zeros_like(near), np.zeros_like(near)
    far_mask = ~near_mask
    correction[far_mask] = -2 * congruence[far_mask] / (1j * divisors[far_mask])
    correction_derivative[far_mask] = (
        -2 * derivative_congruence[far_mask] / (1j * divisors[far_mask])
    )
    w0 = sum(abs(np.fft.fft(amplitudes) / size))
    w1 = sum(abs(np.fft.fft(derivative) / size))
    assert np.linalg.norm(near, 2) <= carrier_frequency * w0**2 / 4
    assert np.linalg.norm(correction, 2) <= w0**2
    assert np.linalg.norm(correction_derivative, 2) <= 2 * w0 * w1
    assert np.linalg.norm(correction_derivative, 2) > 0
    assert near.shape == (2 * (size - 1),) * 2
    np.testing.assert_allclose(near + np.where(far_mask, congruence, 0), congruence)


def test_full_state_allowance_fails_even_with_zero_growth_and_exact_initial_state():
    height, nonlinear, modulus, sigma, fifth, size, spectral, harmonics = sp.symbols(
        "b g m sigma C N ell1 A0", positive=True
    )
    horizon = 2 / (nonlinear * height**2 * (2 - modulus))
    residual_allowance = 2 * sp.sqrt(size) * (2 * sigma * height + fifth * height**5)
    reference_norm_upper = sp.sqrt(size) * height * (1 + harmonics * height**2)
    ratio = horizon * residual_allowance / sp.sqrt(spectral) / reference_norm_upper
    expected = (
        4
        * (2 * sigma / height**2 + fifth * height**2)
        / (nonlinear * (2 - modulus) * sp.sqrt(spectral) * (1 + harmonics * height**2))
    )
    assert sp.simplify(ratio - expected) == 0
    fixed_size = 1024
    # Admitted acute chart implies b<4; A0<1/16, sqrt(ell1)<8/N.
    frequency = 1 - sp.sqrt(2) / 2
    harmonic_sum = (
        (2 - sp.sqrt(2)) / 32 + sp.Rational(1, 96) + (3 * sp.sqrt(2) - 2) / 672
    )
    assert harmonic_sum < sp.Rational(1, 16)
    assert sp.sqrt(2 * frequency) > sp.Rational(1, 2)
    assert pi_interval().hi < 4
    floor = (
        4
        / (sp.Rational(1, 72) * 2 * sp.Rational(8, fixed_size) * 2)
        * 2
        * sp.sqrt(2 * sp.Rational(3, fixed_size**2) * sp.Rational(1, 512))
    )
    assert sp.simplify(floor - 9 * sp.sqrt(3) / 8) == 0
    assert floor > 1
