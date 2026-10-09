"""Independent amplitude-hierarchy and holomorphic remainder algebra.

These controls distinguish ordinary amplitude coefficients from time jets.
They run no acquired-source assessment or complete nonlinear response.
"""

from fractions import Fraction as Q
from math import factorial, pi

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_cubic_response as cubic


@pytest.fixture(scope="module")
def report():
    # The coefficient budget was fixed before its first retained evaluation.
    # The production owner caches only the two pure normalized coefficients;
    # source/error/policy consequences are freshly reconstructed here.
    return cubic.bound_sine_class_cubic_response(
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1),
        total_duration=Q(2),
        endpoint_radius=Q(1, 10**32),
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        first_probe_work_allowance=Q(2, 10**6),
        second_probe_work_allowance=Q(2, 10**6),
    )


@pytest.fixture(scope="module")
def geometry():
    cycles = tuple(
        tuple((9 * c + j, 9 * c + (j + 1) % 9) for j in range(9)) for c in range(3)
    )
    edges = sum(cycles, ()) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    reflection = tuple(9 * c + 8 - j for c in range(3) for j in range(9))
    return edges, degrees, reflection


def _amplitude_rates(geometry, state, gamma, cosines, sines):
    """Direct edge expansion through amplitude degree three, not a flow."""
    edges, degrees, _ = geometry
    rates = [[Q(0)] * 27 for _ in range(6)]
    for edge_index, (i, j) in enumerate(edges):
        cosine, sine = cosines[edge_index], sines[edge_index]
        differences = tuple(row[j] - row[i] for row in state)
        y1, y2, y3 = differences[1::2]
        current = (
            cosine * y1,
            cosine * y2 - sine * y1**2 / 2,
            cosine * y3 - sine * y1 * y2 - cosine * y1**3 / 6,
        )
        for k in range(3):
            form_current = differences[2 * k] + gamma * current[k]
            phase_current = -gamma * differences[2 * k]
            for row, value in ((2 * k, form_current), (2 * k + 1, phase_current)):
                rates[row][i] += value / degrees[i]
                rates[row][j] -= value / degrees[j]
    return tuple(map(tuple, rates))


def _algebraic_edge_weights():
    # Arbitrary exact weights with the target's reflection parity; these are
    # algebraic controls, not replacement trigonometric model parameters.
    cosines = (Q(3, 4),) * 9 + (Q(1, 5),) * 9 + (Q(3, 4),) * 9 + (Q(1),) * 2
    sines = (Q(2, 3),) * 9 + (Q(4, 5),) * 9 + (Q(2, 3),) * 9 + (Q(0),) * 2
    return cosines, sines


def _original_time_coefficients(geometry, initial, gamma, cosines, sines, order):
    """Exact time series of the unscaled amplitude expansion on each edge."""
    edges, degrees, _ = geometry
    series = [initial]
    for n in range(order):
        rates = [[Q(0)] * 27 for _ in range(6)]
        for edge_index, (i, j) in enumerate(edges):
            differences = tuple(
                tuple(row[j] - row[i] for row in coefficient) for coefficient in series
            )
            quadratic = sum(
                differences[p][1] * differences[n - p][1] for p in range(n + 1)
            )
            recoupling = sum(
                differences[p][1] * differences[n - p][3] for p in range(n + 1)
            )
            third = sum(
                differences[p][1] * differences[q][1] * differences[n - p - q][1]
                for p in range(n + 1)
                for q in range(n - p + 1)
            )
            sine, cosine = sines[edge_index], cosines[edge_index]
            sine_coefficients = (
                cosine * differences[n][1],
                cosine * differences[n][3] - sine * quadratic / 2,
                cosine * differences[n][5] - sine * recoupling - cosine * third / 6,
            )
            for level in range(3):
                dx = differences[n][2 * level]
                for row, current in (
                    (2 * level, dx + gamma * sine_coefficients[level]),
                    (2 * level + 1, -gamma * dx),
                ):
                    rates[row][i] += current / degrees[i]
                    rates[row][j] -= current / degrees[j]
        series.append(tuple(tuple(value / (n + 1) for value in row) for row in rates))
    return tuple(series)


def test_scaled_recurrence_encloses_independent_original_coordinate_series(geometry):
    # All three levels are initially nonzero, so this checks the carried
    # quadratic state and its eta-weighted cubic feedback, not only a fresh
    # impulse at a symmetric equilibrium. The parameters are exact algebraic
    # fixtures, not a replacement constitutive model or a reserved source.
    _, degrees, _ = geometry
    gamma = Q(1, 32)
    cosines, sines = _algebraic_edge_weights()
    parameters = cubic._CubicParameters(
        (1, 2, 1),
        I(gamma),
        I(gamma**2),
        degrees,
        tuple(map(I, sines)),
        tuple(map(I, cosines)),
    )
    initial = tuple(
        tuple(Q(((11 * i + 7 * row) % 19) - 9, 2 ** (row + 5)) for i in range(27))
        for row in range(6)
    )
    scales = (Q(1), gamma, gamma**3, gamma**4, gamma**4, gamma**5)
    original = tuple(
        tuple(scale * value for value in row) for row, scale in zip(initial, scales)
    )
    expected = _original_time_coefficients(
        geometry, original, gamma, cosines, sines, order=3
    )
    assert expected[1] == _amplitude_rates(geometry, original, gamma, cosines, sines)
    levels = tuple(initial[2 * k] + initial[2 * k + 1] for k in range(3))
    retained = cubic._time_coefficients(levels, parameters, order=3)
    for level in range(3):
        for n in range(4):
            for row_index in (2 * level, 2 * level + 1):
                offset = 27 * (row_index % 2)
                values = expected[n][row_index]
                for node, value in enumerate(values):
                    enclosure = retained[level][n][offset + node]
                    assert enclosure.contains(value / scales[row_index])
                    assert enclosure.width < Q(1, 10**30)
                if n:
                    assert sum(d * value for d, value in zip(degrees, values)) == 0


def test_form_event_carries_every_other_amplitude_coordinate():
    levels = tuple(
        tuple(I(Q(54 * level + node - 50, 128)) for node in range(54))
        for level in range(3)
    )
    jump = Q(-3, 64)
    after = cubic._event_levels(levels, jump)
    assert after[0][4] == levels[0][4] + jump
    assert tuple(after[0][i] for i in range(54) if i != 4) == tuple(
        levels[0][i] for i in range(54) if i != 4
    )
    assert after[1:] == levels[1:]
    assert cubic._event_levels(levels, Q(0)) == levels


def test_hierarchy_preserves_weighted_charge_and_alternating_reflection(geometry):
    _, degrees, reflection = geometry
    state = tuple(
        tuple(
            Q((c + 1) * (j - 4) ** (1 if row in (2, 3) else 2), 11 + row)
            for c in range(3)
            for j in range(9)
        )
        for row in range(6)
    )
    rates = _amplitude_rates(geometry, state, Q(1, 37), *_algebraic_edge_weights())
    for index, row in enumerate(rates):
        assert any(row)
        assert sum(d * value for d, value in zip(degrees, row)) == 0
        parity = -1 if index in (2, 3) else 1
        assert tuple(row[reflection[i]] for i in range(27)) == tuple(
            parity * value for value in row
        )
    assert all(rates[index][node] == 0 for index in (2, 3) for node in (4, 13, 22))


def test_quadratic_odd_phase_feeds_nonzero_central_cubic_form(geometry):
    zero = (Q(0),) * 27
    y1 = tuple(Q(1, 7) if i in (12, 14) else Q(0) for i in range(27))
    y2 = tuple(Q(i - 13, 11) if i in (12, 14) else Q(0) for i in range(27))
    gamma = Q(1, 37)
    weights = _algebraic_edge_weights()
    full = _amplitude_rates(geometry, (zero, y1, zero, y2, zero, zero), gamma, *weights)
    discarded = _amplitude_rates(
        geometry, (zero, y1, zero, zero, zero, zero), gamma, *weights
    )
    assert y2[13] == 0
    expected_feedback = -gamma * Q(4, 5) / (2 * 7 * 11)
    assert full[4][13] - discarded[4][13] == expected_feedback != 0
    assert full[4][12] == full[4][14]


def test_cauchy_bootstrap_and_odd_tail_have_the_declared_rational_budget():
    # A rational upper bound for cosh(2) establishes both complex sine
    # estimates without interpreting a complex proof extension as EPI.
    partial = sum(Q(4**n, factorial(2 * n)) for n in range(5))
    first_tail = Q(4**5, factorial(10))
    assert partial + first_tail / (1 - Q(1, 33)) < 4
    gamma, time = Q(1, 3000), Q(2)
    d = 1 - 2 * gamma**2 * time**2
    assert 16 * gamma**2 * time**2 < 1
    assert Q(1, 2) / (1 - 8 * gamma**2 * time**2) < 1

    def tail(amplitude):
        radius = 1 / (2 * gamma * amplitude)
        assert radius > 1
        boundary_remainder = 8 * gamma * time / d
        geometric = boundary_remainder * radius**-5 / (1 - radius**-2)
        compact = (
            256
            * gamma**6
            * amplitude**5
            * time
            / (d * (1 - 4 * gamma**2 * amplitude**2))
        )
        assert geometric == compact
        return compact

    a = b = Q(1, 2000)
    all_six_histories = 2 * (tail(a) + tail(b) + tail(a + b))
    assert all_six_histories < Q(16, 10**34)
    source = 8 * Q(1, 10**32) / (1 - 2 * gamma * time)
    assert source > all_six_histories
    # Source uncertainty is additional; ideal-source amplitude parity cannot
    # erase it. No coefficient or nonlinear response is evaluated here.
    assert source + all_six_histories < Q(83, 10**33)


def test_time_majorants_are_coefficientwise_supersolutions():
    v1, v2, v3, eta, ell = Q(2, 7), Q(3, 11), Q(5, 13), Q(1, 17), Q(201, 100)
    p2 = (v2, 2 * v1**2)
    p3 = (v3, 4 * eta * v1 * v2 + Q(4, 3) * v1**3, 4 * eta * v1**3)
    # Factor the common e^(2Lt) or e^(3Lt) from V'-L*V-forcing.
    residual2 = (p2[1] + ell * p2[0] - 2 * v1**2, ell * p2[1])
    residual3 = (
        p3[1] + 2 * ell * p3[0] - 4 * eta * v1 * p2[0] - Q(4, 3) * v1**3,
        2 * p3[2] + 2 * ell * p3[1] - 4 * eta * v1 * p2[1],
        2 * ell * p3[2],
    )
    assert residual2 == tuple(ell * value for value in p2)
    assert residual3 == tuple(2 * ell * value for value in p3)
    assert min(residual2 + residual3) > 0


def test_time_tail_uses_shifted_exponential_starts_for_carried_cubic_state():
    v1, v2, v3, eta, ell, h = (
        Q(2, 7),
        Q(3, 11),
        Q(5, 13),
        Q(1, 17),
        Q(201, 100),
        Q(1),
    )
    polynomial = (v3, 4 * eta * v1 * v2 + Q(4, 3) * v1**3, 4 * eta * v1**3)
    rate, order = 3 * ell, 64

    def exponential_tail(start):
        z = rate * h
        assert 0 < z < start + 1
        return z**start / factorial(start) / (1 - z / (start + 1))

    # Independently multiply the positive exponential by its quadratic
    # prefactor, then collect the omitted powers of the FULL time series.
    omitted_prefix = sum(
        sum(
            coefficient * rate ** (n - j) / factorial(n - j)
            for j, coefficient in enumerate(polynomial)
        )
        * h**n
        for n in range(order + 1, order + 22)
    )
    bound = sum(
        coefficient * h**j * exponential_tail(order + 1 - j)
        for j, coefficient in enumerate(polynomial)
    )
    assert bound > omitted_prefix > 0
    # Starting every exponential remainder at65 would omit part of the
    # actual tail from t and t^2, even though all coefficients are positive.
    wrong_unshifted = sum(
        coefficient * h**j * exponential_tail(order + 1)
        for j, coefficient in enumerate(polynomial)
    )
    assert wrong_unshifted < omitted_prefix


def test_production_time_tail_matches_all_three_independent_majorants():
    norms = (Q(2, 7), Q(3, 11), Q(5, 13))
    # Use retained interval norms explicitly: non-dyadic source coordinates
    # are enclosed, rather than silently treated as exactly materialized.
    levels = tuple((I(-value, value),) * 54 for value in norms)
    v1, v2, v3 = (level[0].abs_max for level in levels)
    h, eta, ell = Q(3, 4), Q(1, 17), Q(201, 100)

    def tail(multiple, start):
        z = multiple * ell * h
        return z**start / factorial(start) / (1 - z / (start + 1))

    expected = (
        v1 * tail(1, 65),
        v2 * tail(2, 65) + 2 * v1**2 * h * tail(2, 64),
        v3 * tail(3, 65)
        + (4 * eta * v1 * v2 + Q(4, 3) * v1**3) * h * tail(3, 64)
        + 4 * eta * v1**3 * h**2 * tail(3, 63),
    )
    retained_norms, retained_tails = cubic._time_tails(levels, h, eta)
    assert retained_norms == (v1, v2, v3)
    assert retained_tails == expected
    assert min(retained_tails) > 0


def _floating_variational_coefficient(geometry, mediator_class):
    """Auxiliary floating three-level flow, never the nonlinear sine flow.

    Independent dense incidence matrices and numerical integration check the
    polynomial calculation. This is not a rigorous time-tail certificate.
    """
    edges, degrees, _ = geometry
    incidence = np.zeros((27, len(edges)))
    for column, (left, right) in enumerate(edges):
        incidence[left, column], incidence[right, column] = -1, 1
    divergence = incidence / np.asarray(degrees)[:, None]
    laplacian = divergence @ incidence.T
    theta = np.array(
        [2 * pi * k * (j - 4) / 9 for k in (1, mediator_class, 1) for j in range(9)]
    )
    sine, cosine = np.sin(incidence.T @ theta), np.cos(incidence.T @ theta)
    phase_laplacian = (divergence * cosine) @ incidence.T
    gamma = 1 / (1023 * pi)
    eta = gamma**2

    def rhs(_time, flat):
        state = flat.reshape(3, 2, 27)
        rates = np.empty_like(state)
        for level in range(3):
            rates[level, 0] = (
                -laplacian @ state[level, 0] - eta * phase_laplacian @ state[level, 1]
            )
            rates[level, 1] = laplacian @ state[level, 0]
        first = incidence.T @ state[0, 1]
        second = incidence.T @ state[1, 1]
        rates[1, 0] += divergence @ (sine * first**2 / 2)
        rates[2, 0] += divergence @ (
            eta * sine * first * second + cosine * first**3 / 6
        )
        return rates.ravel()

    def advance(initial, jump):
        state = initial.copy()
        state[4] += jump
        result = solve_ivp(
            rhs,
            (0, 1),
            state,
            method="DOP853",
            rtol=1e-12,
            atol=np.repeat((1e-18, 1e-23, 1e-28), 54),
        )
        assert result.success and result.t[-1] == 1
        return result.y[:, -1]

    zero = np.zeros(162)
    prefix = advance(zero, 1 / 2000)
    first = advance(prefix, 0)
    both = advance(prefix, 1 / 2000)
    second = advance(zero, 1 / 2000)
    receiver_cubic = 108 + 22
    return gamma**4 * (
        both[receiver_cubic] - first[receiver_cubic] - second[receiver_cubic]
    )


def test_complete_cubic_matches_independent_triangular_integration(geometry, report):
    independent = tuple(
        _floating_variational_coefficient(geometry, mediator_class)
        for mediator_class in (1, 2)
    )
    retained = tuple(
        float((lower + upper) / 2) for lower, upper in report.scaled_class_cubic_bounds
    )
    assert min(abs(value) for value in independent) > 1e-28
    assert np.allclose(independent, retained, rtol=1e-8, atol=0)
    difference = independent[0] - independent[1]
    lower, upper = report.complete_cubic_contrast_bounds
    assert difference < -Q(6, 10**30)
    assert np.isclose(difference, float((lower + upper) / 2), rtol=1e-6, atol=0)


def test_actual_source_transfer_keeps_the_new_signed_bound_below_noise(report):
    # Rebuild the finite full-law consequence from primitive coefficient and
    # analytical remainder bounds; do not infer it from a cached verdict.
    gamma = report.gamma_bounds.hi
    source = 8 * Q(1, 10**32) / (1 - 4 * gamma)
    assert report.source_contrast_error_upper_bound == source
    g, duration = Q(1, 3000), Q(2)

    def tail(amplitude):
        return (
            256
            * g**6
            * amplitude**5
            * duration
            / ((1 - 2 * g**2 * duration**2) * (1 - 4 * g**2 * amplitude**2))
        )

    higher = 2 * (tail(Q(1, 1000)) + 2 * tail(Q(1, 2000)))
    assert report.higher_amplitude_contrast_error_upper_bound == higher
    lower, upper = report.complete_cubic_contrast_bounds
    true_bounds = (lower - higher - source, upper + higher + source)
    assert report.decision.true_bounds == true_bounds
    assert -8 * report.readout_error_bound < true_bounds[0] < true_bounds[1] < 0
    assert true_bounds[1] < -Q(69, 10**31)
    # This proves an actual conditional sign and a scalar error-cancellation
    # budget. It does not prove overlap of the eight raw endpoint records.
    assert report.decision.true_sign
    assert report.decision.scalar_cancellation
