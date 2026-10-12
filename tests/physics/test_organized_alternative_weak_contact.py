"""Static own-law controls for the fixed weak harmonic contact comparison.

The analytic stationary construction is checked through its identities and
uniform remainder bounds. No root, source, trajectory or frozen response runs.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics._sine_class_constitutive_comparison import (
    _current_derivative_bound,
    _current_fourier_coefficients,
    _edge_law,
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_organized_response_evaluation():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _current(mp, angle, order=0):
    return sum(
        _mp(mp, coefficient)
        * frequency**order
        * mp.sin(frequency * angle + order * mp.pi / 2)
        for frequency, coefficient in ((1, Q(1)), (9, Q(1, 2187)), (18, Q(-1, 4374)))
    )


def _potential(mp, angle):
    return 1 - mp.cos(angle) + (1 - mp.cos(9 * angle)) ** 2 / (6 * 9**4)


def _weighted_edges(eta):
    return (
        tuple(
            (9 * child + slot, 9 * child + (slot + 1) % 9, 1)
            for child in range(6)
            for slot in range(9)
        )
        + tuple(
            (9 * child + slot, 9 * ((child + 1) % 5) + slot, 1)
            for child in range(5)
            for slot in range(9)
        )
        + tuple(
            (9 * child + slot, 45 + slot, eta)
            for child in range(5)
            for slot in range(9)
        )
    )


def test_alternative_weak_contact_shared_edge_law_and_real_derivative_bounds(mp):
    coefficients = ((1, Q(1)), (9, Q(1, 2187)), (18, Q(-1, 4374)))
    assert _current_fourier_coefficients(1) == coefficients
    expected_bounds = tuple(
        sum(abs(value) * frequency**order for frequency, value in coefficients)
        for order in range(4)
    )
    assert expected_bounds == (Q(1459, 1458), Q(245, 243), Q(10, 9), Q(8, 3))
    for order, bound in enumerate(expected_bounds):
        assert _current_derivative_bound(1, order) == bound
    pi = pi_interval()
    arguments = (
        (I(Q(2, 7)), mp.mpf(2) / 7),
        (2 * pi / 9, 2 * mp.pi / 9),
        (2 * pi / 5, 2 * mp.pi / 5),
    )
    for angle, expected_angle in arguments:
        potential, derivatives = _edge_law(cos(angle), sin(angle), Q(1))
        assert (
            _mp(mp, potential.lo)
            <= _potential(mp, expected_angle)
            <= _mp(mp, potential.hi)
        )
        for order, derivative in enumerate(derivatives):
            value = _current(mp, expected_angle, order)
            assert _mp(mp, derivative.lo) <= value <= _mp(mp, derivative.hi)
            assert abs(value) <= _mp(mp, expected_bounds[order])
        assert abs(
            mp.diff(lambda x: _potential(mp, x), expected_angle)
            - _current(mp, expected_angle)
        ) < mp.mpf("1e-65")
    # Internal C9 matching does not give a matched collective C5 tangent.
    assert abs(_current(mp, 2 * mp.pi / 9, 1) - mp.cos(2 * mp.pi / 9)) < mp.mpf("1e-65")
    assert _current(mp, 2 * mp.pi / 5, 1) - mp.cos(2 * mp.pi / 5) > mp.mpf(1) / 1000


def test_alternative_weak_contact_mean_zero_inverse_and_fivefold_torque():
    """Fivefold character orthogonality retains the 1+9 harmonic coupling."""
    stiffness = sp.symbols("stiffness", positive=True)
    incidence = sp.zeros(5)
    for index in range(5):
        incidence[index, index], incidence[index, (index + 1) % 5] = -1, 1
    cycle = incidence.T * incidence
    first_row = tuple(map(sp.Rational, (Q(2, 5), 0, Q(-1, 5), Q(-1, 5), 0)))
    inverse = sp.Matrix(
        5, 5, lambda left, right: first_row[(right - left) % 5] / stiffness
    )
    projector = sp.eye(5) - sp.ones(5) / 5
    assert stiffness * cycle * inverse == projector
    assert inverse * sp.ones(5, 1) == sp.zeros(5, 1)
    cosine = (sp.sqrt(5) - 1) / 4
    own_stiffness = cosine + sp.sqrt(5) / 486
    first_eigenvalue = own_stiffness * (2 - 2 * cosine)
    assert (own_stiffness - sp.Rational(3, 10)).is_positive
    assert (sp.Rational(8, 25) - own_stiffness).is_positive
    assert (first_eigenvalue - sp.Rational(2, 5)).is_positive

    # A sum over the five host origins annihilates every spatial character
    # except multiples of five. Apply that rule to f'(t+b)^T(-G f(t+b)).
    lambda_one, lambda_two = sp.symbols("lambda_one lambda_two", positive=True)
    eigenvalues = {1: lambda_one, 2: lambda_two}
    fourier = {}
    for frequency, amplitude in (
        (1, sp.Integer(1)),
        (9, sp.Rational(1, 2187)),
        (18, -sp.Rational(1, 4374)),
    ):
        fourier[frequency] = amplitude / (2 * sp.I)
        fourier[-frequency] = -amplitude / (2 * sp.I)
    torque = {}
    for left_frequency, left_value in fourier.items():
        for right_frequency, right_value in fourier.items():
            frequency = left_frequency + right_frequency
            if frequency % 5:
                continue
            spatial_mode = min(right_frequency % 5, (-right_frequency) % 5)
            value = (
                -5
                * sp.I
                * left_frequency
                * left_value
                * right_value
                / eigenvalues[spatial_mode]
            )
            torque[frequency] = torque.get(frequency, 0) + value
    torque = {
        frequency: sp.simplify(value)
        for frequency, value in torque.items()
        if sp.simplify(value) != 0
    }
    coefficient = 25 / (2187 * lambda_one)
    assert torque == {10: sp.I * coefficient / 2, -10: -sp.I * coefficient / 2}
    # This is -K sin(10t), with curvature -10K at zero. The eighteenth
    # harmonic is retained in the calculation; its self contributions cancel.
    assert (
        sp.simplify(
            sum(sp.I * frequency * value for frequency, value in torque.items())
            + 10 * coefficient
        )
        == 0
    )


def test_alternative_weak_contact_uniform_contraction_torque_and_root_windows():
    """Real Taylor bounds distinguish all stationary windows at the fixed eta."""
    eta, inverse, radius = Q(1, 100000), Q(5, 2), Q(6)
    sqrt_five = Q(161, 72)
    assert sqrt_five**2 > 5
    j0, j1 = Q(1001, 1000), Q(101, 100)
    j2, j3 = _current_derivative_bound(1, 2), _current_derivative_bound(1, 3)
    assert _current_derivative_bound(1, 0) < j0
    assert _current_derivative_bound(1, 1) < j1
    contraction = inverse * (8 * j2 * radius + j1) * eta
    image = inverse * (4 * j2 * radius**2 * eta**2 + sqrt_five * j0 * eta)
    assert contraction < Q(1, 500)
    assert image < radius * eta
    shape_remainder = inverse * (4 * j2 * radius**2 + j1 * radius)
    assert shape_remainder < 416
    differentiated_radius = inverse * sqrt_five * j1 / (1 - contraction)
    assert differentiated_radius < radius
    differentiated_remainder = inverse * (
        8 * j2 * radius**2 + j2 * radius + j1 * radius
    )
    assert differentiated_remainder < 832
    torque_remainder = sqrt_five * j1 * 416 + j2 * radius**2 / 2
    derivative_remainder = (
        sqrt_five * (j2 * 416 + j1 * 832) + j3 * radius**2 / 2 + j2 * radius**2
    )
    assert torque_remainder < 960
    assert derivative_remainder < 3001

    # Own rim stiffness a<8/25 and C5 gap<7/5 give this lower signal.
    coefficient_lower = Q(25, 2187) / (Q(8, 25) * Q(7, 5))
    assert coefficient_lower > Q(1, 40)
    assert 960 * eta < coefficient_lower / 2
    assert Q(4, 5) ** 2 < Q(3, 4)
    assert 3001 * eta < 10 * coefficient_lower * Q(4, 5)
    # Windows k*pi/10 +-pi/60 are disjoint. Boundary sine magnitudes
    # are 1/2, and cosine magnitudes throughout them exceed 4/5.
    assert 2 * Q(1, 60) < Q(1, 10)
    assert 960 * eta < Q(1, 80)
    assert 3001 * eta < Q(1, 5)


def test_alternative_weak_contact_full_tube_normal_curvature():
    """The whole fine segment is coercive, not just the carrier Hessian."""
    eta, fine_radius = Q(1, 100000), Q(1, 100)
    pi_upper = Q(22, 7)
    internal_ceiling, rim_ceiling = Q(18, 25), Q(13, 10)
    assert 2 * pi_upper / 9 + 2 * fine_radius < internal_ceiling
    assert 2 * pi_upper / 5 + 12 * eta + 2 * fine_radius < rim_ceiling

    def cosine_lower(angle):
        return 1 - angle**2 / 2 + angle**4 / 24 - angle**6 / 720

    # Both harmonic derivative terms remain in this global lower estimate.
    harmonic_defect = sum(
        abs(coefficient) * frequency
        for frequency, coefficient in _current_fourier_coefficients(1)
        if frequency != 1
    )
    assert cosine_lower(internal_ceiling) - harmonic_defect > Q(7, 10)
    assert cosine_lower(rim_ceiling) - harmonic_defect > Q(1, 4)
    internal_gap = 2 - 2 * cos(2 * pi_interval() / 9)
    assert internal_gap.lo > Q(2, 5)
    spoke_star = sp.zeros(6)
    for child in range(5):
        spoke_star[child, child] += 1
        spoke_star[5, 5] += 1
        spoke_star[child, 5] -= 1
        spoke_star[5, child] -= 1
    assert spoke_star.eigenvals() == {0: 1, 1: 4, 6: 1}
    normal_lower = Q(7, 10) * Q(2, 5) - 6 * eta * _current_derivative_bound(1, 1)
    assert normal_lower > Q(1, 4)
    # The same domain also keeps the four host-shape directions positive.
    assert Q(1, 4) - eta * _current_derivative_bound(1, 1) > Q(1, 5)


def test_alternative_weak_contact_complete_weighted_loss_and_normal_hessian(mp):
    """One supplied carrier state checks all rows; it is not a found equilibrium."""
    eta, gamma, alpha, tolerance = (
        mp.mpf(1) / 100000,
        1 / (1023 * mp.pi),
        2 * mp.pi / 9,
        mp.mpf("1e-60"),
    )
    origins = tuple(
        2 * mp.pi * (child - 2) / 5 + mp.mpf((child - 2) ** 2 - 2) / 100000
        for child in range(5)
    ) + (mp.pi / 7,)
    phase = mp.matrix(
        [alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)]
    )
    raw = tuple(mp.mpf((7 * index) % 17 - 8) / 19 for index in range(54))
    form = mp.matrix([value - sum(raw) / 54 for value in raw])
    current, sine_current, form_gradient = (mp.zeros(54, 1) for _ in range(3))
    laplacian, hessian = mp.zeros(54), mp.zeros(54)
    edges = _weighted_edges(eta)
    assert len(edges) == 144
    for left, right, weight in edges:
        angle = phase[right] - phase[left]
        for vector, exchange in (
            (current, weight * _current(mp, angle)),
            (sine_current, weight * mp.sin(angle)),
            (form_gradient, weight * (form[left] - form[right])),
        ):
            vector[left] += exchange
            vector[right] -= exchange
        for matrix, coefficient in (
            (laplacian, weight),
            (hessian, weight * _current(mp, angle, 1)),
        ):
            matrix[left, left] += coefficient
            matrix[right, right] += coefficient
            matrix[left, right] -= coefficient
            matrix[right, left] -= coefficient
    for child in range(6):
        child_current = current[9 * child : 9 * child + 9]
        average = sum(child_current) / 9
        assert max(abs(value - average) for value in child_current) < tolerance
    assert max(abs(value) for value in current) > mp.mpf("1e-10")
    assert max(abs(value) for value in laplacian * form - form_gradient) < tolerance
    form_rate = (-form_gradient + gamma * current) / 4
    phase_rate = gamma * form_gradient / 4
    assert abs(sum(form_rate)) < tolerance
    assert abs(sum(phase_rate)) < tolerance

    def storage_on_tangent(parameter, sine=False):
        forms, phases = form + parameter * form_rate, phase + parameter * phase_rate
        return sum(
            weight
            * (
                (forms[right] - forms[left]) ** 2 / 2
                + (
                    1 - mp.cos(phases[right] - phases[left])
                    if sine
                    else _potential(mp, phases[right] - phases[left])
                )
            )
            for left, right, weight in edges
        )

    loss = -(form_gradient.T * form_gradient)[0] / 4
    assert abs(mp.diff(storage_on_tangent, 0) - loss) < tolerance
    wrong_defect = gamma * ((current - sine_current).T * form_gradient)[0] / 4
    assert abs(wrong_defect) > 1000 * tolerance
    assert (
        abs(
            mp.diff(lambda t: storage_on_tangent(t, sine=True), 0) - loss - wrong_defect
        )
        < tolerance
    )

    wheel = mp.zeros(6)
    for child in range(5):
        for neighbor, weight in (((child + 1) % 5, 1), (5, eta)):
            curvature = weight * _current(mp, origins[neighbor] - origins[child], 1)
            wheel[child, child] += curvature
            wheel[neighbor, neighbor] += curvature
            wheel[child, neighbor] -= curvature
            wheel[neighbor, child] -= curvature
    tensor = mp.zeros(54)
    for child in range(6):
        for slot in range(9):
            index = 9 * child + slot
            for neighbor in range(6):
                tensor[index, 9 * neighbor + slot] += wheel[child, neighbor]
            tensor[index, index] += 2 * mp.cos(alpha)
            for neighbor in ((slot - 1) % 9, (slot + 1) % 9):
                tensor[index, 9 * child + neighbor] -= mp.cos(alpha)
    assert max(abs(value) for value in hessian - tensor) < tolerance
    assert max(abs(value) for value in hessian * mp.ones(54, 1)) < tolerance
    normal_lower = cos(2 * pi_interval() / 9) * (
        2 - 2 * cos(2 * pi_interval() / 9)
    ) - 6 * Q(1, 100000) * _current_derivative_bound(1, 1)
    assert normal_lower.lo > Q(1, 5)
    for mode in range(1, 9):
        shift = 2 * mp.cos(alpha) * (1 - mp.cos(2 * mp.pi * mode / 9))
        factor = mp.cholesky(wheel + shift * mp.eye(6))
        assert all(factor[index, index] > 0 for index in range(6))
