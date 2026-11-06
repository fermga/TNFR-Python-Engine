"""Static own-law controls for the fixed weak harmonic contact comparison.

The stationary construction and actual-source capture proof are checked through
their identities and uniform bounds. No root, source, trajectory or response runs.
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


def test_alternative_weak_contact_source_budget_and_own_curvature():
    """Carry retained own-law inequalities through the entire observation word."""
    eta, gamma, duration = Q(1, 100000), Q(1, 3000), Q(1, 5000)
    recovered_error, zeta = Q(1, 10**70), Q(1, 10**6)
    j0, j1 = (_current_derivative_bound(1, order) for order in (0, 1))
    form_bound = Q(1, 100) + gamma * j0 * duration
    form_drift = (2 * form_bound + gamma * j0) * duration
    phase_drift = 2 * gamma * form_bound * duration
    word_increase = 360 * form_bound * form_drift + 180 * j0 * phase_drift
    new_work = 45 * eta * ((form_bound + recovered_error) ** 2 / 2 + 2 + Q(2, 19683))
    donor_excess = 2 * (1 + j1) * recovered_error**2
    # These two primitive majorants come from the retained own-law outer
    # proof. They are combined here, without re-evaluating its source producer.
    reached_excess, old_boundary = Q(9588, 10**6), Q(1451, 100000)
    full_excess = reached_excess + word_increase + donor_excess + new_work
    assert word_increase < Q(1, 50000)
    assert new_work < Q(901, 10**6)
    assert full_excess < Q(1051, 100000) < Q(3, 250)
    assert old_boundary - full_excess > Q(1, 250)

    def cosine_lower(value):
        return 1 - value**2 / 2 + value**4 / 24 - value**6 / 720

    def cosine_upper(value):
        return cosine_lower(value) + value**8 / 40320

    gap = Q(29, 20)
    remaining_gap = (2 * Q(157, 50) - gap) / 4
    sine_face = 5 * Q(309, 1000) - cosine_upper(gap) - 4 * cosine_upper(remaining_gap)
    # Every alternative edge correction is nonnegative; subtracting the
    # entire five-edge correction ceiling at the ground remains sufficient.
    own_face = sine_face - 5 * Q(2, 19683)
    assert own_face > Q(1, 750)
    assert cosine_lower(gap) - Q(2, 243) > Q(1, 10)
    assert Q(1, 10) - eta * j1 > Q(9, 100)
    pi = pi_interval()
    internal_face = Q(1, 2) + 8 * (1 - cos(5 * pi / 24)) - 9 * (1 - cos(2 * pi / 9))
    assert internal_face.lo > Q(1, 25) > Q(3, 250)
    fine_coercivity = (Q(1, 2) - Q(2, 243)) * Q(2, 5) - 6 * eta * j1
    assert fine_coercivity > Q(19, 100)
    assert 10 * j1 < 11
    assert Q(2, 5) - 2 * 11**2 * gamma**2 / Q(19, 100) > Q(19, 200)

    phase_fine, scaled_form_fine = 2 * (zeta + phase_drift), 2 * gamma * form_bound
    fine_source = 27 * ((phase_fine + scaled_form_fine) ** 2 + scaled_form_fine**2)
    assert fine_source < Q(1, 10**8)
    assert 2 * Q(1, 750) / Q(1, 10) < Q(1, 6) ** 2
    assert 5 * form_bound**2 < Q(1, 40) ** 2
    assert Q(1, 6) + 6 * eta + gamma / 40 < Q(1, 5)


def test_alternative_weak_contact_fine_and_mean_rows_retain_actual_remainder(mp):
    """Independent noncarrier edge sums check fine decay and both mean channels."""
    eta, gamma, alpha, tolerance = (
        mp.mpf(1) / 100000,
        1 / (1023 * mp.pi),
        2 * mp.pi / 9,
        mp.mpf("1e-60"),
    )
    origins = tuple(2 * mp.pi * (child - 2) / 5 for child in range(5)) + (mp.pi / 7,)
    forms = tuple(mp.mpf(2 * child - 5) / 10000 for child in range(6))
    q, z = mp.zeros(54, 1), mp.zeros(54, 1)
    for child in range(6):
        raw_phase = tuple(
            mp.mpf((3 * child + 5 * slot) % 11 - 5) / 20000 for slot in range(9)
        )
        raw_form = tuple(
            mp.mpf((7 * child + 3 * slot) % 13 - 6) / 40000 for slot in range(9)
        )
        for slot in range(9):
            q[9 * child + slot] = raw_phase[slot] - sum(raw_phase) / 9
            z[9 * child + slot] = raw_form[slot] - sum(raw_form) / 9
    carrier = mp.matrix(
        [alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)]
    )
    phase = carrier + q
    form = mp.matrix([value for value in forms for _ in range(9)]) + z

    def means(values):
        return mp.matrix(
            [sum(values[9 * child : 9 * child + 9]) / 9 for child in range(6)]
        )

    def fine(values):
        averaged = means(values)
        return values - mp.matrix([value for value in averaged for _ in range(9)])

    current, carrier_current, gradient = (
        mp.zeros(54, 1),
        mp.zeros(54, 1),
        mp.zeros(54, 1),
    )
    laplacian, storage = mp.zeros(54), mp.mpf(0)
    for left, right, weight in _weighted_edges(eta):
        angle = phase[right] - phase[left]
        for vector, exchange in (
            (current, weight * _current(mp, angle)),
            (carrier_current, weight * _current(mp, carrier[right] - carrier[left])),
            (gradient, weight * (form[left] - form[right])),
        ):
            vector[left] += exchange
            vector[right] -= exchange
        laplacian[left, left] += weight
        laplacian[right, right] += weight
        laplacian[left, right] -= weight
        laplacian[right, left] -= weight
        storage += weight * (
            (form[right] - form[left]) ** 2 / 2 + _potential(mp, angle)
        )
    ground = 54 * _potential(mp, alpha) + 45 * _potential(mp, 2 * mp.pi / 5)
    assert 0 < storage - ground < mp.mpf(3) / 250
    assert max(abs(value) for value in fine(carrier_current)) < tolerance
    fine_gradient = -fine(current)
    assert (q.T * fine_gradient)[0] > mp.mpf(19) / 100 * (q.T * q)[0]
    assert (fine_gradient.T * fine_gradient)[0] < 121 * (q.T * q)[0]
    form_rate = (-gradient + gamma * current) / 4
    phase_rate = gamma * gradient / 4
    assert (
        max(abs(value) for value in fine(phase_rate) - gamma * laplacian * z / 4)
        < tolerance
    )
    assert (
        max(
            abs(value)
            for value in fine(form_rate) + (laplacian * z + gamma * fine_gradient) / 4
        )
        < tolerance
    )

    def fine_energy_on_tangent(parameter):
        actual_q = fine(phase + parameter * phase_rate - carrier)
        actual_z = fine(form + parameter * form_rate)
        mixed = actual_q + gamma * actual_z
        return ((mixed.T * mixed)[0] + gamma**2 * (actual_z.T * actual_z)[0]) / 2

    fine_energy = fine_energy_on_tangent(0)
    bracket = (
        (q.T * fine_gradient)[0]
        + (z.T * laplacian * z)[0]
        + 2 * gamma * (z.T * fine_gradient)[0]
    )
    assert abs(mp.diff(fine_energy_on_tangent, 0) + gamma**2 * bracket / 4) < tolerance
    assert mp.diff(fine_energy_on_tangent, 0) <= -19 * gamma**2 * fine_energy / 800

    wheel_current, remainder = mp.zeros(6, 1), mp.zeros(6, 1)
    wheel_laplacian = mp.zeros(6)
    wheel_edges = tuple((child, (child + 1) % 5, 1) for child in range(5)) + tuple(
        (child, 5, eta) for child in range(5)
    )
    for left, right, weight in wheel_edges:
        angle = origins[right] - origins[left]
        exchange = weight * _current(mp, angle)
        correction = (
            weight
            * sum(
                _current(mp, angle + q[9 * right + slot] - q[9 * left + slot])
                - _current(mp, angle)
                for slot in range(9)
            )
            / 9
        )
        wheel_current[left] += exchange
        wheel_current[right] -= exchange
        remainder[left] += correction
        remainder[right] -= correction
        wheel_laplacian[left, left] += weight
        wheel_laplacian[right, right] += weight
        wheel_laplacian[left, right] -= weight
        wheel_laplacian[right, left] -= weight
    assert (
        max(abs(value) for value in means(current) - wheel_current - remainder)
        < tolerance
    )
    assert (
        max(
            abs(value)
            for value in means(form_rate)
            - (
                -wheel_laplacian * mp.matrix(forms)
                + gamma * (wheel_current + remainder)
            )
            / 4
        )
        < tolerance
    )
    assert (
        max(
            abs(value)
            for value in means(phase_rate)
            - gamma * wheel_laplacian * mp.matrix(forms) / 4
        )
        < tolerance
    )
    assert abs(remainder[5]) > 1000 * tolerance
    assert sum(abs(value) for value in remainder) <= mp.mpf(80) * fine_energy / 27
    assert abs(remainder[5]) <= mp.mpf(40) * eta * fine_energy / 27

    chi = gamma * (sum(forms[:5]) / 5 - forms[5])
    mean_phase_rate, mean_form_rate = means(phase_rate), means(form_rate)
    t_rate = sum(mean_phase_rate[:5]) / 5 - mean_phase_rate[5]
    chi_rate = gamma * (sum(mean_form_rate[:5]) / 5 - mean_form_rate[5])
    assert abs(t_rate - 3 * eta * chi / 2) < tolerance
    assert (
        abs(t_rate + chi_rate + 3 * gamma**2 * (wheel_current[5] + remainder[5]) / 10)
        < tolerance
    )
    # Discarding the nonzero fine torque changes the corrected-origin row,
    # even though every fine phase and form array has zero child mean.
    assert (
        abs(t_rate + chi_rate + 3 * gamma**2 * wheel_current[5] / 10) > 1000 * tolerance
    )


def test_alternative_weak_contact_tracking_impulse_and_ten_source_arcs():
    """Derive own-law tracking constants and disjoint limit windows analytically."""
    eta, gamma = Q(1, 100000), Q(1, 3000)
    j1, j2 = (_current_derivative_bound(1, order) for order in (1, 2))
    host_coercivity, shape_gradient_norm = Q(9, 100), Q(5)
    assert 5 * j1**2 < (26 * host_coercivity) ** 2
    assert (26 * Q(3, 2)) ** 2 < 2 * 28**2
    assert 4 * 5 * j1**2 < 5**2
    assert (
        1 - 2 * shape_gradient_norm**2 * gamma**2 / host_coercivity
        > host_coercivity / 2
    )
    torque_upper = Q(25, 2187) / Q(2, 5) + 960 * eta
    assert torque_upper < Q(1, 20)
    mean_gain, mean_damping = Q(3, 10), Q(3, 2)
    torque_shape, donor_fine, host_fine = Q(5), 4 * j2 / 3, 8 * j2 / 3
    weight, decay, fine_decay = Q(20), Q(1, 200), Q(19, 800)
    assert host_coercivity / 16 - weight * mean_gain * torque_shape * eta**2 > decay
    assert (weight * mean_damping - 28) * eta > weight * decay * gamma**2
    assert host_fine + weight * mean_gain * donor_fine * eta**2 < 3
    plateau = weight * mean_gain * Q(1, 20) * eta**3 / decay
    assert plateau == 60 * eta**3
    convolution = Q(3) / (fine_decay - decay)
    fine_source, shape_source = Q(1, 10**8), Q(1, 5)
    transient = shape_source + convolution * fine_source
    assert transient < Q(21, 100)
    chi_bound = (
        mean_gain
        * gamma**2
        * (
            Q(1, 20) * eta
            + torque_shape * (transient + plateau)
            + donor_fine * fine_source
        )
        / mean_damping
    )
    assert chi_bound < gamma**2 / 4 < Q(1, 10**6)
    impulse = (
        mean_gain
        * eta
        * (torque_shape * transient / decay + donor_fine * fine_source / fine_decay)
    )
    assert impulse < Q(1, 1000)

    inner, outer, chi_margin = Q(1, 10), Q(1, 5), Q(1, 10**6)
    assert 10 * (inner - chi_margin) > Q(99, 100)
    assert 10 * (outer + chi_margin) < Q(201, 100)

    def sine_lower(value):
        return value - value**3 / 6 + value**5 / 120 - value**7 / 5040

    # Concavity on this positive subinterval of (0, pi) reduces its minimum
    # to these endpoints; the lower Taylor polynomials certify both.
    assert sine_lower(Q(99, 100)) > Q(4, 5)
    assert sine_lower(Q(201, 100)) > Q(4, 5)
    restoring = (Q(1, 40) * Q(4, 5) - 960 * eta) * eta**2
    permanent_error = torque_shape * eta * plateau
    assert restoring > eta**2 / 100 > permanent_error
    assert impulse < outer - inner
    pi_lower, pi_upper = Q(157, 50), Q(22, 7)
    assert outer + chi_margin < pi_lower / 12
    assert 2 * inner < pi_lower / 5

    # Every primitive prototype is covered by a sine basin, while its own
    # harmonic-law root window is separated from the corresponding sine root.
    sine_labels = range(0, 20, 4)
    for label in range(1, 20, 2):
        circular_distance = min(
            min((label - other) % 20, (other - label) % 20) for other in sine_labels
        )
        assert circular_distance == 1
        assert circular_distance * pi_upper / 10 + inner < Q(3, 5)
        assert circular_distance * pi_lower / 10 - pi_upper / 60 > 0
