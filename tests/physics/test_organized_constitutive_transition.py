"""Static joint-parameter controls for the weak-contact constitutive transition.

These verify coefficients, uniform analytic bounds and the critical Schur
structure. They evaluate no stationary root, source, flow or frozen response.
"""

from fractions import Fraction as Q

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import cos, pi_interval
from tnfr.physics._sine_class_constitutive_comparison import (
    _current_derivative_bound,
    _current_fourier_coefficients,
    _edge_law,
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_organized_response_evaluation():
        yield


def test_constitutive_transition_exact_weighted_leading_coefficients():
    """The C5 inverse and sine curvature give the same critical scale."""
    phi, cosine = (1 + sp.sqrt(5)) / 2, (sp.sqrt(5) - 1) / 4
    first_eigenvalue = cosine * (2 - 2 * cosine)
    harmonic_coefficient = sp.simplify(25 / (2187 * first_eigenvalue))
    assert sp.simplify(harmonic_coefficient - (25 + 15 * sp.sqrt(5)) / 2187) == 0
    sine_coefficient = 5 * phi**10
    kappa = sp.simplify(sine_coefficient / (2 * harmonic_coefficient))
    assert sp.simplify(kappa - 2187 * (105 + 47 * sp.sqrt(5)) / 40) == 0
    assert (sine_coefficient - 600).is_positive
    assert (12000 - kappa).is_positive
    assert (kappa - 11000).is_positive

    origin, mu = sp.symbols("origin mu", real=True)
    leading = sp.sin(5 * origin) - mu * sp.sin(10 * origin) / 2
    assert sp.diff(leading, origin).subs(origin, 0) == 5 - 5 * mu
    assert sp.diff(leading, origin, 3).subs(origin, 0) == 500 * mu - 125
    assert sp.diff(leading, origin, 3).subs({origin: 0, mu: 1}) == 375
    assert sp.diff(leading, origin, mu).subs(origin, 0) == -5

    # The common law owner preserves the C9 tangent for every coefficient;
    # this identity does not erase the C5 harmonic coefficient above.
    deformation, internal_cosine, internal_sine = sp.symbols(
        "deformation internal_cosine internal_sine", real=True
    )
    potential, derivatives = _edge_law(internal_cosine, internal_sine, deformation)
    # For alpha=2*pi/9, cos(3*alpha)=-1/2 gives this exact cubic.
    cubic = 8 * internal_cosine**3 - 6 * internal_cosine + 1
    for difference in (
        potential - (1 - internal_cosine),
        derivatives[0] - internal_sine,
        derivatives[1] - internal_cosine,
    ):
        assert sp.rem(difference, cubic, internal_cosine) == 0


def test_constitutive_transition_weighted_degree_and_parity_boundary():
    """The even eighteenth harmonic prevents transferring sine eta parity."""
    origin = sp.symbols("origin", real=True)
    coefficients = _current_fourier_coefficients(Q(1))
    sine = sp.sin(origin)
    harmonic = sum(
        sp.Rational(value) * sp.sin(frequency * origin)
        for frequency, value in coefficients
        if frequency != 1
    )
    assert sp.simplify(sine.subs(origin, origin + sp.pi) + sine) == 0
    assert (
        sp.simplify(
            harmonic.subs(origin, origin + sp.pi)
            + harmonic
            + sp.sin(18 * origin) / 2187
        )
        == 0
    )
    assert sp.simplify(harmonic.subs(origin, origin + sp.pi) + harmonic) != 0

    # With lambda=kappa*eta^3, a sine spoke has weight 1 and frequency 1;
    # a harmonic spoke has weight 4 and frequency 9 or 18. All have degree<=5n.
    # Rim deformation has no explicit origin frequency and costs weight 3.
    building_blocks = tuple(
        (1 if frequency == 1 else 4, frequency) for frequency, _ in coefficients
    ) + ((3, 0),)
    assert all(frequency <= 5 * weight for weight, frequency in building_blocks)
    assert all(frequency % 5 for frequency, _ in coefficients)
    # Pure-sine orders below five cannot carry a nonzero fivefold odd mode;
    # the first direct deformed-spoke term is also killed by the character sum.
    assert all(
        not any(frequency % 5 == 0 for frequency in range(1, order + 1))
        for order in range(1, 5)
    )


def test_constitutive_transition_joint_complex_shape_admission():
    """Bound both current laws on the weighted complex disk with real majorants."""
    radius, deformation_radius, ball = Q(1, 100), Q(1, 25), Q(1, 200)
    shape, difference = Q(3, 100), Q(1, 25)
    first_shape, first_difference = Q(47, 2000), Q(69, 2500)
    inverse, edge_inverse = Q(259, 100), Q(389, 100)
    sine_bound, cosine_bound = Q(24, 25), Q(31, 100)
    assert 40000 * radius**3 == deformation_radius
    assert first_shape + ball < shape
    assert first_difference + 2 * ball < difference

    def cosh_remainder(value):
        return value**2 / (2 * (1 - value**2 / 12))

    def sinh_upper(value):
        return value / (1 - value**2 / 6)

    def sinh_remainder(value):
        return value**3 / (6 * (1 - value**2 / 20))

    def harmonic_current(value):
        return (1 / (1 - 9 * value) + 1 / (2 * (1 - 18 * value))) / 2187

    def harmonic_derivative(value):
        return (1 / (1 - 9 * value) + 1 / (1 - 18 * value)) / 243

    assert 18 * difference < 1
    contraction = (
        2
        * edge_inverse
        * (
            sine_bound * sinh_upper(difference)
            + cosine_bound * cosh_remainder(difference)
        )
        + inverse * radius / (1 - shape)
        + 2 * edge_inverse * deformation_radius * harmonic_derivative(difference)
        + inverse * radius * deformation_radius * harmonic_derivative(shape)
    )
    center_residual = (
        edge_inverse
        * (
            sine_bound * cosh_remainder(first_difference)
            + cosine_bound * sinh_remainder(first_difference)
        )
        + inverse * radius * first_shape / (1 - first_shape)
        + edge_inverse
        * deformation_radius
        * first_difference
        * harmonic_derivative(first_difference)
        + inverse * radius * deformation_radius * harmonic_current(first_shape)
    )
    assert contraction < Q(7, 20)
    assert center_residual < Q(11, 5000)
    assert center_residual + contraction * ball < ball
    torque = (
        5 * radius * (1 / (1 - shape) + deformation_radius * harmonic_current(shape))
    )
    assert torque < Q(3, 50)


def test_constitutive_transition_weighted_series_and_pitchfork_margins():
    """Derive derivative tails from degree and Cauchy bounds, including mixed mu."""
    eta, radius, torque_bound = Q(1, 10**12), Q(1, 100), Q(3, 50)
    ratio = eta / radius
    assert 2 * ratio < 1
    limits, tails = (Q(73, 100), Q(22), Q(650), Q(19500)), []
    for derivative in range(4):
        # For n>=6 and derivative<=3, successive n^(derivative+1)
        # increase by less than2. Sum the resulting geometric majorant.
        assert Q(7, 6) ** (derivative + 1) < 2
        tail = (
            2
            * torque_bound
            * 5**derivative
            * 6 ** (derivative + 1)
            * eta
            / radius**6
            / (1 - 2 * ratio)
        )
        assert tail < limits[derivative]
        tails.append(tail)
    amplitude_lower, kappa_upper, kappa_distance = Q(600), Q(12000), Q(20000)
    normalized_first = tails[1] / amplitude_lower
    normalized_mixed = kappa_upper * tails[1] / (amplitude_lower * kappa_distance)
    normalized_third = tails[3] / amplitude_lower
    assert normalized_first < Q(1, 20)
    assert normalized_mixed < Q(1, 20)
    assert normalized_third < 40
    assert Q(11, 10) * kappa_upper < kappa_distance
    assert 5 * Q(1, 10) > normalized_first
    assert -5 + normalized_mixed < 0
    critical_mu_error = normalized_first / 5
    assert critical_mu_error < Q(1, 100)
    assert -125 + 500 * (1 - critical_mu_error) - normalized_third > 330
    origin_radius, parameter_radius = Q(1, 20), Q(1, 100)
    cosine_lower = 1 - (10 * origin_radius) ** 2 / 2
    third_lower = -125 + 500 * Q(9, 10) * cosine_lower - normalized_third
    assert third_lower > 200
    # The exact integral formula for B_s averages f''' with total weight 1/6.
    first, second = sp.symbols("first second", nonnegative=True)
    integral_weight = sp.integrate(first**2 / 2, (first, 0, 1), (second, 0, 1))
    assert integral_weight == sp.Rational(1, 6)
    shape_slope_lower = third_lower / 6
    boundary = (
        shape_slope_lower * origin_radius**2 - (5 + normalized_mixed) * parameter_radius
    )
    assert boundary > 0


def test_constitutive_transition_uniform_normal_and_critical_schur_structure():
    """Positive fine/shape blocks leave precisely one relative critical mode."""
    eta, fine_radius, deformation = Q(1, 10**12), Q(1, 100), Q(1)
    internal_ceiling, rim_ceiling = Q(18, 25), Q(13, 10)
    assert 2 * Q(22, 7) / 9 + 2 * fine_radius < internal_ceiling
    assert 2 * Q(22, 7) / 5 + 12 * eta + 2 * fine_radius < rim_ceiling

    def cosine_lower(value):
        return 1 - value**2 / 2 + value**4 / 24 - value**6 / 720

    derivative_defect = sum(
        abs(value) * frequency
        for frequency, value in _current_fourier_coefficients(deformation)
        if frequency != 1
    )
    assert cosine_lower(internal_ceiling) - derivative_defect > Q(7, 10)
    assert cosine_lower(rim_ceiling) - derivative_defect > Q(1, 4)
    internal_gap = 2 - 2 * cos(2 * pi_interval() / 9)
    assert internal_gap.lo > Q(2, 5)
    assert Q(7, 10) * internal_gap.lo - 6 * eta * _current_derivative_bound(
        deformation, 1
    ) > Q(1, 4)
    assert Q(1, 4) - eta * _current_derivative_bound(deformation, 1) > Q(1, 5)

    # This is an algebraic consequence of the proved critical Schur zero,
    # not a supplied state asserted to be the unknown critical equilibrium.
    incidence = sp.zeros(5)
    for index in range(5):
        incidence[index, index], incidence[index, (index + 1) % 5] = -1, 1
    zero_mean = sp.eye(5)[:, :4]
    for index in range(4):
        zero_mean[4, index] = -1
    shape_block = zero_mean.T * incidence.T * incidence * zero_mean
    assert all(shape_block[:size, :size].det() > 0 for size in range(1, 5))
    coupling = sp.Matrix(sp.symbols("coupling0:4", real=True))
    critical_diagonal = (coupling.T * shape_block.inv() * coupling)[0]
    relative_hessian = shape_block.row_join(coupling).col_join(
        coupling.T.row_join(sp.Matrix([[critical_diagonal]]))
    )
    transform = sp.eye(5)
    transform[:4, 4] = -shape_block.inv() * coupling
    reduced = sp.simplify(transform.T * relative_hessian * transform)
    assert reduced == sp.diag(shape_block, 0)
    assert 8 * 6 + shape_block.rank() == 52
    # Compress the pencil derivative at s=0 onto its sole stiffness null
    # direction. This does not assume that mass and stiffness diagonalize
    # together: the mass term is quadratic in s and drops out of this derivative.
    null_direction = (-shape_block.inv() * coupling).col_join(sp.ones(1, 1))
    assert sp.simplify(relative_hessian * null_direction) == sp.zeros(5, 1)
    damping = sp.symbols("damping", positive=True)
    derivative_compression = (
        null_direction.T * (damping * sp.eye(5)) * null_direction
    )[0]
    assert (
        sp.simplify(
            derivative_compression
            - damping * (1 + (null_direction[:4, :].T * null_direction[:4, :])[0])
        )
        == 0
    )
    # A coupled positive mass need not share stiffness eigenvectors. Its
    # full determinant still has a simple zero, but removing damping does not.
    spectral, stiffness_scale = sp.symbols("spectral stiffness_scale", positive=True)
    mass = sp.eye(3) + sp.ones(3)
    stiffness = sp.diag(2, 3, 0)
    assert mass * stiffness != stiffness * mass
    pencil = (
        spectral**2 * mass
        + spectral * damping * sp.eye(3)
        + stiffness_scale * stiffness
    )
    determinant = sp.expand(pencil.det())
    assert determinant.coeff(spectral, 0) == 0
    assert determinant.coeff(spectral, 1) == 6 * damping * stiffness_scale**2
    undamped = sp.expand(determinant.subs(damping, 0))
    assert undamped.coeff(spectral, 1) == 0
    assert undamped.coeff(spectral, 2) == 12 * stiffness_scale**2
