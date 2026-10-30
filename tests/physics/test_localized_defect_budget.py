"""Static exclusion of the declared canonical coefficient-norm budget.

These controls concern a prescribed upper-bound calculation. They neither
bound actual correction errors from below nor evaluate a prepared response.
"""

from fractions import Fraction as Q

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def test_profile_means_follow_from_first_integral_quadrature():
    angle = sp.Symbol("u", real=True)
    height, modulus, inverse_width, size = sp.symbols("b m q_star N", positive=True)
    lower, upper = height**2 * (1 - modulus), height**2
    squared_profile = lower + (upper - lower) * sp.sin(angle) ** 2
    # Y=F^2 transforms the half-period first-integral quadrature. On
    # 0<u<pi/2, dY/sqrt((upper-Y)(Y-lower))=2du.
    square_root_product = (upper - lower) * sp.sin(angle) * sp.cos(angle)
    assert sp.simplify(sp.diff(squared_profile, angle) - 2 * square_root_product) == 0
    first_integral = sp.integrate(2, (angle, 0, sp.pi / 2))
    third_integral = sp.integrate(2 * squared_profile, (angle, 0, sp.pi / 2))
    first_mean = height * first_integral / (inverse_width * size)
    third_mean = height * third_integral / (inverse_width * size)
    assert first_mean == sp.pi * height / (inverse_width * size)
    assert sp.simplify(third_mean / first_mean - (2 - modulus) * height**2 / 2) == 0
    # This independently matches the zero Fourier coefficient of the ODE;
    # a periodic F'' has zero mean, including its turning points.
    assert (
        sp.simplify(
            inverse_width**2 * ((2 - modulus) * first_mean - 2 * third_mean / height**2)
        )
        == 0
    )
    scale = sp.Symbol("scale", positive=True)
    assert (
        sp.simplify(
            first_mean.subs(height, inverse_width * scale) - sp.pi * scale / size
        )
        == 0
    )


def test_nonlinear_phase_force_has_the_joint_reflection_parity_needed_by_the_budget():
    center, forward, backward = sp.symbols(
        "theta center_next center_previous", real=True
    )
    nonlinear = (
        center
        - forward
        - sp.sin(center - forward)
        + center
        - backward
        - sp.sin(center - backward)
    ) / 2
    reflected = nonlinear.subs(
        {center: -center, forward: -backward, backward: -forward}, simultaneous=True
    )
    assert sp.simplify(reflected + nonlinear) == 0
    # For an even real profile multiplying odd carrier sines, joint
    # reflection changes (n,l) to (-n,-l). Reality and oddness then force
    # each nonlinear node coefficient to be purely imaginary.
    real, imaginary = sp.symbols("R_real R_imag", real=True)
    coefficient = real + sp.I * imaginary
    assert sp.expand(sp.conjugate(coefficient) + coefficient) == 2 * real
    assert sp.solve(sp.Eq(sp.conjugate(coefficient), -coefficient), real) == [0]


def test_dominant_resolvent_retains_a_loss_component_in_both_numerator_rows():
    sigma, eigenvalue, frequency, form, phase = sp.symbols(
        "sigma lambda Omega X Theta", positive=True
    )
    imaginary_residual = sp.Symbol("R_imag", real=True)
    real_residual = sigma * eigenvalue * form / 2
    phase_residual = (frequency * phase - eigenvalue * form) / 2
    residual = sp.Matrix([real_residual + sp.I * imaginary_residual, phase_residual])
    inverse_numerator = (
        sp.Matrix(
            [
                [sp.I * frequency, -eigenvalue],
                [eigenvalue, sp.I * frequency + sigma * eigenvalue],
            ]
        )
        * residual
    )
    protected_form_component = sp.im(inverse_numerator[0])
    protected_phase_component = sp.re(inverse_numerator[1])
    assert (
        sp.simplify(
            protected_form_component - sigma * eigenvalue * frequency * form / 2
        )
        == 0
    )
    assert (
        sp.simplify(
            protected_phase_component - sigma * eigenvalue * frequency * phase / 2
        )
        == 0
    )
    squared_numerator = sum(
        sp.expand_complex(value * sp.conjugate(value)) for value in inverse_numerator
    )
    protected_square = (
        sigma**2 * eigenvalue**2 * frequency**2 * (form**2 + phase**2) / 4
    )
    remaining_squares = (
        sp.re(inverse_numerator[0]) ** 2 + sp.im(inverse_numerator[1]) ** 2
    )
    assert sp.expand(squared_numerator - protected_square - remaining_squares) == 0
    mean, correction = sp.symbols("a epsilon", positive=True)
    substituted = (form**2 + phase**2).subs(
        {form: mean * (1 - correction), phase: mean * (1 + correction)}
    )
    assert sp.expand(substituted - 2 * mean**2 - 2 * mean**2 * correction**2) == 0


def test_both_branches_of_the_declared_minimum_retain_the_same_linear_budget_floor():
    sigma, eigenvalue, mean, slow_frequency, time = sp.symbols(
        "sigma lambda a omega_s t", positive=True
    )
    final_time = 1 / slow_frequency
    # X>=a/2 implies ||r||>=sigma*lambda*a/4. The separately proved
    # resolvent numerator and |det|<=3lambda*omega_s supply the second floor.
    contractive_floor = sigma * eigenvalue * mean * time / 4
    resolvent_floor = sigma * eigenvalue * mean / (3 * sp.sqrt(2) * slow_frequency)
    selected_floor = sigma * eigenvalue * mean * time / 5
    assert sp.simplify(contractive_floor - selected_floor).is_positive
    assert sp.simplify(
        resolvent_floor - selected_floor.subs(time, final_time)
    ).is_positive
    frequency = eigenvalue - slow_frequency
    determinant = eigenvalue**2 - frequency**2 + sp.I * sigma * eigenvalue * frequency
    assert (
        sp.expand(sp.re(determinant) - slow_frequency * (eigenvalue + frequency)) == 0
    )
    assert sp.im(determinant) == sigma * eigenvalue * frequency
    # sigma<omega_s, 0<Omega<lambda<1 imply the elementary triangle bound
    # |det|<2lambda*omega_s+lambda*omega_s=3lambda*omega_s.
    assert (
        sp.expand(
            3 * eigenvalue * slow_frequency
            - (2 * eigenvalue * slow_frequency + eigenvalue * slow_frequency)
        )
        == 0
    )


def test_weighted_source_moment_is_a_radius_independent_necessary_condition():
    time, final_time, radius, source_slope, phase_gap = sp.symbols(
        "s T rho C G_app", positive=True
    )
    correction = sp.Symbol("k", nonnegative=True)
    # For 0<=k<1 and min(1,G)>=G_app, equation (63)'s exponent is at
    # least G_app*rho/sqrt(2); the prefactor is at least one.
    assert (
        sp.simplify(
            (correction + sp.Rational(1, 2)) / (1 - correction)
            - sp.Rational(1, 2)
            - 3 * correction / (2 * (1 - correction))
        )
        == 0
    )
    exponent_floor = phase_gap * radius / sp.sqrt(2)
    moment = sp.integrate(
        (final_time - time) * source_slope * time, (time, 0, final_time)
    )
    assert moment == source_slope * final_time**3 / 6
    assert (
        sp.simplify(
            exponent_floor * moment / radius
            - phase_gap * source_slope * final_time**3 / (6 * sp.sqrt(2))
        )
        == 0
    )
    exponent = sp.Symbol("z", nonnegative=True)
    assert (sp.exp(exponent) - exponent).subs(exponent, 0) == 1
    assert sp.diff(sp.exp(exponent) - exponent, exponent) == sp.exp(exponent) - 1
    # Therefore exp(z)>=z. Any chosen source budget >=C*s makes this
    # lower bound apply to Q/rho, not to the actual trajectory error.


def test_canonical_parameter_margins_follow_from_exact_constants_and_integrand_bounds():
    size = 1024
    eigenvalue = 1 - sp.sqrt(2) / 2
    dispersion = sp.sqrt(2) / 4
    nonlinear = eigenvalue**2 / 8
    edge_factor = sp.sqrt(2 * eigenvalue)
    scale = sp.sqrt(2 * dispersion / nonlinear)
    assert sp.Rational(1, 4) < eigenvalue < sp.Rational(1, 3)
    assert sp.Rational(1, 3) < dispersion < sp.Rational(3, 8)
    assert 8 < scale < 9
    assert edge_factor > sp.Rational(3, 4)
    pi = pi_interval()
    assert 3 < pi.lo < pi.hi < Q(7, 2)
    assert pi.lo / (size**2 - 1) > Q(3, size**2)
    # For m=1/2, the integral has pi/2<K<=pi/sqrt(2), and
    # omega_s=6*d*K^2/N^2. No elliptic integral or profile is evaluated.
    slow_lower = Q(6) * Q(1, 3) * Q(3, 2) ** 2 / size**2
    slow_upper = Q(6) * Q(3, 8) * Q(7, 2) ** 2 / (2 * size**2)
    assert slow_lower == Q(9, 2 * size**2)
    assert slow_upper < Q(15, size**2)
    sigma_upper = Q(4, size**2 - 1)
    assert sigma_upper < slow_lower
    assert slow_upper < Q(1, 8)  # omega_s<lambda/2, so Omega>lambda/2.
    # b=(2K/N)*scale <72/N<1/8; a=pi*scale/N>24/N.
    height_upper = Q(72, size)
    assert height_upper < Q(1, 8)
    mean_lower = Q(3) * 8 / size
    assert mean_lower == Q(24, size)
    # X=a*(1-lambda*3*b^2/64)>a/2, Theta>=a.
    correction_upper = Q(1, 3) * 3 * height_upper**2 / 64
    assert correction_upper < Q(1, 2)
    # The declared original gap budget is also below one: q_*<8/N,
    # B1=q_*b/4, Dk<1 and A0<1 are sufficient.
    harmonics = (
        -(2 - sp.sqrt(2)) / 32,
        sp.Rational(1, 96),
        -(3 * sp.sqrt(2) - 2) / 672,
    )
    assert sum(abs(value) for value in harmonics) < 1
    gap_upper = height_upper + Q(8, size) * height_upper / 4 + 2 * height_upper**3
    assert gap_upper < 1
    assert Q(3, 4) * mean_lower == Q(18, size)


def test_conjugate_pair_and_weighted_moment_fix_the_strict_obstruction_constant():
    size, phase_gap, edge_factor, sigma, eigenvalue, mean, time = sp.symbols(
        "N G_app Dk sigma lambda a T", positive=True
    )
    modal_floor_slope = sigma * eigenvalue * mean / 5
    phase_norm_floor_slope = sp.sqrt(2 * size) * edge_factor * modal_floor_slope
    source_slope = phase_gap**2 * phase_norm_floor_slope / 2
    lower_bound = sp.simplify(phase_gap * source_slope * time**3 / (6 * sp.sqrt(2)))
    assert (
        sp.simplify(
            lower_bound
            - sp.sqrt(size)
            * phase_gap**3
            * edge_factor
            * sigma
            * eigenvalue
            * mean
            * time**3
            / 60
        )
        == 0
    )
    primitive_floor = lower_bound.subs(
        {
            phase_gap: 18 / size,
            edge_factor: sp.Rational(3, 4),
            sigma: 3 / size**2,
            eigenvalue: sp.Rational(1, 4),
            mean: 24 / size,
            time: size**2 / 15,
        }
    )
    assert sp.simplify(primitive_floor - sp.Rational(243, 625) * sp.sqrt(size)) == 0
    assert primitive_floor.subs(size, 1024) == sp.Rational(7776, 625)
    assert primitive_floor.subs(size, 1024) > 12


def test_positive_coefficient_envelope_is_not_a_lower_bound_on_actual_transport():
    time = sp.Symbol("t", real=True)
    # Exact mean-mode forcing with frequency one: the whole oscillatory
    # integral vanishes after a full turn, although its norm envelope does not.
    exact_kernel = (sp.exp(sp.I * time) - 1) / sp.I
    assert sp.simplify(sp.diff(exact_kernel, time) - sp.exp(sp.I * time)) == 0
    assert exact_kernel.subs(time, 2 * sp.pi) == 0
    declared_envelope = sp.Min(2 * sp.pi, 2)
    assert declared_envelope == 2
    assert declared_envelope > abs(exact_kernel.subs(time, 2 * sp.pi))
