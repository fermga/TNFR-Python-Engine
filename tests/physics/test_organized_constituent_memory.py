"""Static controls for memory inherited from the unchanged C9 x C5 law.

Independent edge sums, exact polynomial coefficients and pointwise derivatives
check the conditional identities. They are not trajectories, acquired hosts,
reserved responses, measured derivatives or an autonomous covariance closure.
"""

from fractions import Fraction as Q
import hashlib
from math import factorial
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
    exact_square_matrix_power,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin


_INNER = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9))
_OUTER = tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
_EDGES = _INNER + _OUTER
_CYCLE = tuple((c, (c + 1) % 5) for c in range(5))


@pytest.fixture(scope="module", autouse=True)
def preserve_frozen_evidence_and_producers():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
            "docs/assets/organized_robust_current/attempt.json",
            "docs/assets/organized_robust_current/result.json",
            "benchmarks/organized_robust_current.py",
            "src/tnfr/physics/_organized_robust_current.py",
            "docs/assets/organized_finite_window_design/method-sources-v1.json",
            "docs/assets/organized_recovery_observation/method-sources-v1.json",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_organized_response_evaluation():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _action(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _transpose(matrix):
    return tuple(zip(*matrix))


def _laplacian(size, edges, weights=None):
    weights = (Q(1),) * len(edges) if weights is None else weights
    matrix = [[Q(0)] * size for _ in range(size)]
    for (i, j), weight in zip(edges, weights):
        matrix[i][i] += weight
        matrix[j][j] += weight
        matrix[i][j] -= weight
        matrix[j][i] -= weight
    return tuple(map(tuple, matrix))


def _means(vector):
    return tuple(sum(vector[9 * c : 9 * c + 9]) / 9 for c in range(5))


def _repeat(vector):
    return tuple(value for value in vector for _ in range(9))


def _hidden(vector):
    return tuple(a - b for a, b in zip(vector, _repeat(_means(vector))))


def _source(context, theta, edges=_EDGES):
    result = [context.mpf(0)] * len(theta)
    for i, j in edges:
        current = context.sin(theta[j] - theta[i])
        result[i] += current
        result[j] -= current
    return tuple(result)


def _source_derivative(context, theta, direction, edges=_EDGES):
    result = [context.mpf(0)] * len(theta)
    for i, j in edges:
        current = context.cos(theta[j] - theta[i]) * (direction[j] - direction[i])
        result[i] += current
        result[j] -= current
    return tuple(result)


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _close(context, left, right):
    assert abs(left - right) < context.mpf("1e-80")


def _carrier(context, phases):
    return tuple(
        2 * context.pi * (j - 4) / 9 + phases[c] for c in range(5) for j in range(9)
    )


def _assert_winding_one(context, phases):
    gaps = tuple(
        context.atan2(
            context.sin(phases[(i + 1) % len(phases)] - phase),
            context.cos(phases[(i + 1) % len(phases)] - phase),
        )
        for i, phase in enumerate(phases)
    )
    assert all(0 < gap < context.pi for gap in gaps)
    _close(context, sum(gaps), 2 * context.pi)


@pytest.fixture(scope="module")
def context():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


@pytest.fixture(scope="module")
def matrices():
    lap = _laplacian(45, _EDGES)
    cycle = _laplacian(5, _CYCLE)
    repeat = tuple(tuple(Q(i // 9 == c) for c in range(5)) for i in range(45))
    projection = tuple(
        tuple(Q(i // 9 == j // 9, 9) for j in range(45)) for i in range(45)
    )
    complement = tuple(
        tuple(Q(i == j) - projection[i][j] for j in range(45)) for i in range(45)
    )
    return lap, cycle, repeat, projection, complement


def test_exact_support_projection_and_neutral_hidden_coordinates(matrices):
    lap, cycle, repeat, projection, complement = matrices
    assert len(set(tuple(sorted(edge)) for edge in _EDGES)) == 90
    assert all(lap[i][i] == 4 for i in range(45))
    assert exact_matrix_product(_transpose(repeat), repeat) == tuple(
        tuple(Q(9 * (i == j)) for j in range(5)) for i in range(5)
    )
    assert exact_matrix_product(projection, projection) == projection
    assert exact_matrix_product(complement, complement) == complement
    assert exact_matrix_product(lap, repeat) == exact_matrix_product(repeat, cycle)
    assert exact_matrix_product(lap, projection) == exact_matrix_product(
        projection, lap
    )
    assert exact_matrix_product(complement, repeat) == ((Q(0),) * 5,) * 45
    # The spectral bound used in hidden transport follows from the actual
    # degree-four support, not from the C5 observation matrix.
    assert exact_symmetric_semidefinite(
        tuple(tuple(Q(8 * (i == j)) - lap[i][j] for j in range(45)) for i in range(45))
    )


def test_projected_rows_and_signed_hidden_work_balance(context, matrices):
    lap, cycle, _, _, _ = matrices
    gamma = 1 / (1023 * context.pi)
    form = tuple(Q(value, 10000) for value in (2, -1, 3, -5, 1))
    phases = tuple(context.mpf(value) / 13 for value in (2, -1, 3, -4, 0))
    u = _hidden(tuple(Q((i * 7) % 11 - 5, 10**8) for i in range(45)))
    z = _hidden(tuple(Q((i * 3) % 7 - 3, 10**8) for i in range(45)))
    forcing = tuple(Q(value, 17) for value in (1, -2, 3, -1, -1))
    x = tuple(a + b for a, b in zip(_repeat(form), u))
    theta = tuple(a + _mp(context, b) for a, b in zip(_carrier(context, phases), z))
    q, lu = _action(lap, x), _action(lap, u)
    source, coarse_source = _source(context, theta), _source(context, phases, _CYCLE)
    x_prime = tuple(
        (-_mp(context, a) + gamma * s) / 4 + _mp(context, f)
        for a, s, f in zip(q, source, _repeat(forcing))
    )
    theta_prime = tuple(gamma * _mp(context, a) / 4 for a in q)
    form_prime, phase_prime = _means(x_prime), _means(theta_prime)
    residual = tuple(gamma * (a - b) / 4 for a, b in zip(_means(source), coarse_source))
    for actual, a, s in zip(_hidden(x_prime), lu, _hidden(source)):
        _close(context, actual, -_mp(context, a) / 4 + gamma * s / 4)
    for actual, a in zip(_hidden(theta_prime), lu):
        _close(context, actual, gamma * _mp(context, a) / 4)
    assert _hidden(_repeat(forcing)) == (Q(0),) * 45
    coarse_q = _action(cycle, form)
    for actual, a, s, f, r in zip(
        form_prime, coarse_q, coarse_source, forcing, residual
    ):
        _close(
            context, actual, (-_mp(context, a) + gamma * s) / 4 + _mp(context, f) + r
        )
    fine_rate = sum(_mp(context, a) * b for a, b in zip(q, x_prime)) - _dot(
        source, theta_prime
    )
    fine_work = _dot(q, _repeat(forcing))
    assert fine_work == 9 * _dot(coarse_q, forcing)
    _close(context, fine_rate, -_mp(context, _dot(q, q)) / 4 + _mp(context, fine_work))
    collective_rate = sum(
        _mp(context, a) * b for a, b in zip(coarse_q, form_prime)
    ) - _dot(coarse_source, phase_prime)
    _close(
        context,
        fine_rate - 9 * collective_rate,
        -_mp(context, _dot(lu, lu)) / 4
        - 9 * sum(_mp(context, a) * r for a, r in zip(coarse_q, residual)),
    )


@pytest.mark.parametrize(
    "weights", ((Q(1),) * 5, (Q(3, 5), Q(-2, 7), Q(0), Q(5, 8), Q(-4, 9)))
)
def test_weighted_tangent_blocks_vanish_without_positive_edge_weights(
    matrices, weights
):
    _, _, repeat, projection, complement = matrices
    hessian = _laplacian(
        45,
        _EDGES,
        (Q(2, 3),) * 45 + tuple(weight for weight in weights for _ in range(9)),
    )
    coarse_hessian = _laplacian(5, _CYCLE, weights)
    assert exact_matrix_product(hessian, repeat) == exact_matrix_product(
        repeat, coarse_hessian
    )
    assert exact_matrix_product(hessian, projection) == exact_matrix_product(
        projection, hessian
    )
    assert (
        exact_matrix_product(complement, exact_matrix_product(hessian, repeat))
        == ((Q(0),) * 5,) * 45
    )
    assert (
        exact_matrix_product(
            _transpose(repeat), exact_matrix_product(hessian, complement)
        )
        == ((Q(0),) * 45,) * 5
    )


@pytest.mark.parametrize("time_label", (Q(0), Q(1, 3), Q(2, 7)))
def test_actual_carrier_and_time_dependent_tangent_coefficients(context, time_label):
    # These three supplied chart points do not solve or sample a trajectory.
    t = _mp(context, time_label)
    phases = tuple(c * t + (c - 2) * t**2 for c in range(5))
    theta = _carrier(context, phases)
    coarse_source = _source(context, phases, _CYCLE)
    for fine, coarse in zip(_source(context, theta), _repeat(coarse_source)):
        _close(context, fine, coarse)
    hidden = _hidden(tuple(Q((i * 3) % 7 - 3, 19) for i in range(45)))
    direction = tuple(_mp(context, a) for a in hidden)
    derivative = _source_derivative(context, theta, direction)
    for mean in _means(derivative):
        _close(context, mean, 0)
    collective_direction = tuple(context.mpf(c - 2) for c in range(5))
    derivative = _source_derivative(context, theta, _repeat(collective_direction))
    for value in _hidden(derivative):
        _close(context, value, 0)


@pytest.mark.parametrize("gamma", (Q(1, 7), Q(2, 9)))
def test_volterra_heat_kernel_coefficients_recover_both_hidden_rows(matrices, gamma):
    lap = matrices[0]
    initial_u = _hidden(tuple(Q((i * 7) % 13 - 6, 31) for i in range(45)))
    initial_z = _hidden(tuple(Q((i * 5) % 11 - 5, 23) for i in range(45)))
    history = tuple(
        _hidden(tuple(Q((i * (k + 2)) % 7 - 3, 17 + k) for i in range(45)))
        for k in range(3)
    )
    zero = (Q(0),) * 45
    u_coefficients, z_coefficients = [initial_u], [initial_z]
    for n in range(6):
        lu = _action(lap, u_coefficients[n])
        source = history[n] if n < len(history) else zero
        u_coefficients.append(
            tuple((-a + gamma * b) / (4 * (n + 1)) for a, b in zip(lu, source))
        )
        z_coefficients.append(tuple(gamma * a / (4 * (n + 1)) for a in lu))

    def heat_coefficient(vector, degree):
        result = vector
        for n in range(1, degree + 1):
            result = tuple(-a / (4 * n) for a in _action(lap, result))
        return result

    # Compare formal coefficients only. h(t) is an independently supplied
    # polynomial hidden source; no sine response or exponential is evaluated.
    assert z_coefficients[0] == initial_z
    for n in range(1, 7):
        expected_u = list(heat_coefficient(initial_u, n))
        expected_z = [-gamma * a for a in heat_coefficient(initial_u, n)]
        for j, source in enumerate(history):
            m = n - j - 1
            if m < 0:
                continue
            integral_factor = Q(factorial(m) * factorial(j), factorial(n))
            heat_source = heat_coefficient(source, m)
            for i, value in enumerate(heat_source):
                expected_u[i] += gamma * integral_factor * value / 4
                if m:
                    expected_z[i] -= gamma**2 * integral_factor * value / 4
        assert tuple(expected_u) == u_coefficients[n]
        assert tuple(expected_z) == z_coefficients[n]


@pytest.mark.parametrize("outer_winding", (0, 1, -1))
def test_covariance_feedback_and_cubic_remainder_at_supplied_states(
    context, outer_winding
):
    gamma = 1 / (1023 * context.pi)
    epsilon = Q(1, 10**7)
    z = (epsilon, epsilon, -2 * epsilon) + (Q(0),) * 42
    phases = tuple(outer_winding * 2 * context.pi * (c - 2) / 5 for c in range(5))
    zz = tuple(tuple(a * b for b in z) for a in z)
    remainder, quadratic = [context.mpf(0)] * 5, [context.mpf(0)] * 5
    for c, d in _CYCLE:
        differences = tuple(z[9 * d + j] - z[9 * c + j] for j in range(9))
        variance = sum(value**2 for value in differences) / 9
        covariance_variance = (
            sum(
                zz[9 * d + j][9 * d + j]
                + zz[9 * c + j][9 * c + j]
                - 2 * zz[9 * d + j][9 * c + j]
                for j in range(9)
            )
            / 9
        )
        assert covariance_variance == variance
        assert sum(differences) == 0
        angle = phases[d] - phases[c]
        edge_remainder = sum(
            context.sin(angle + _mp(context, value)) for value in differences
        ) / 9 - context.sin(angle)
        edge_quadratic = -context.sin(angle) * _mp(context, variance) / 2
        cubic_bound = _mp(context, sum(abs(value) ** 3 for value in differences) / 54)
        assert abs(edge_remainder - edge_quadratic) <= cubic_bound + context.mpf(
            "1e-80"
        )
        for target, sign in ((c, 1), (d, -1)):
            remainder[target] += sign * gamma * edge_remainder / 4
            quadratic[target] += sign * gamma * edge_quadratic / 4
    norm = context.sqrt(_mp(context, _dot(z, z)))
    assert (
        sum(abs(a - b) for a, b in zip(remainder, quadratic))
        <= 2 * gamma * norm**3 / 27
    )
    assert sum(abs(a) for a in quadratic) <= gamma * norm**2 / 9
    if outer_winding == 0:
        assert quadratic == [0] * 5
        assert abs(remainder[0]) > context.mpf("1e-30")


def test_same_collective_state_storage_loss_and_residual_have_different_future_jets(
    context, matrices
):
    lap, cycle, _, _, _ = matrices
    epsilon = Q(1, 2**40)
    z = (epsilon, -epsilon) + (Q(0),) * 43
    phases = tuple(2 * context.pi * (c - 2) / 5 for c in range(5))
    theta = tuple(a + _mp(context, b) for a, b in zip(_carrier(context, phases), z))
    gamma = 1 / (1023 * context.pi)
    source = _source(context, theta)
    coarse_source = _source(context, phases, _CYCLE)
    residual = tuple(gamma * (a - b) / 4 for a, b in zip(_means(source), coarse_source))
    forcing = (Q(0),) * 5
    correction_scale = (
        gamma
        * context.sin(2 * context.pi / 5)
        * (1 - context.cos(_mp(context, epsilon)))
        / 18
    )
    for actual, coefficient in zip(residual, (0, 1, 0, 0, -1)):
        _close(context, actual, coefficient * correction_scale)
    _assert_winding_one(context, phases)
    for c in range(5):
        _assert_winding_one(context, theta[9 * c : 9 * c + 9])
        _close(context, _means(theta)[c], phases[c])
    for j in range(9):
        _assert_winding_one(context, theta[j::9])
    records = []
    potential = sum(1 - context.cos(theta[j] - theta[i]) for i, j in _EDGES)
    for sign in (1, -1):
        u = tuple(sign * a for a in z)
        q = _action(lap, u)
        theta_prime = tuple(gamma * _mp(context, a) / 4 for a in q)
        source_prime = _source_derivative(context, theta, theta_prime)
        residual_prime = tuple(gamma * a / 4 for a in _means(source_prime))
        form_prime = tuple(_mp(context, f) + r for f, r in zip(forcing, residual))
        form_second = tuple(
            -a / 4 + b for a, b in zip(_action(cycle, form_prime), residual_prime)
        )
        assert _dot(u, q) / 2 == 5 * epsilon**2
        assert _dot(q, q) / 4 == 14 * epsilon**2
        assert _dot(q, _repeat(forcing)) == 0
        records.append(
            (
                _means(u),
                residual,
                _mp(context, _dot(u, q) / 2) + potential,
                _dot(q, q) / 4,
                residual_prime,
                form_second,
            )
        )
        assert _means(u) == _means(z) == (Q(0),) * 5
        assert max(abs(value) for value in u) < Q(1, 100)
        assert max(abs(value) for value in z) < Q(1, 10**6)
        assert _mp(context, _dot(u, q) / 2) + potential > 0
    assert records[0][:4] == records[1][:4]
    expected = (
        gamma**2
        * _mp(context, epsilon)
        * context.sin(2 * context.pi / 5)
        * context.sin(_mp(context, epsilon))
        / 12
    )
    _close(context, records[0][4][1], expected)
    _close(context, records[1][4][1], -expected)
    _close(context, records[0][5][1] - records[1][5][1], 2 * expected)
    assert expected > 0
    pi = pi_interval()
    assert (2 * pi / 5).hi < Q(7, 5)
    assert (2 * pi / 9 + 2 * epsilon).hi < pi.lo
    assert (2 * pi / 9 - 2 * epsilon).lo > 0
    assert 2 * epsilon < pi.lo / 2
    assert sin(I(epsilon)).lo > 0
    assert sin(2 * pi / 5).lo > 0
    # This proves membership in the declared inference class only; it says
    # nothing about whether either state occurs in an acquired source image.


def test_even_full_hidden_covariance_does_not_close_the_cubic_feedback(context):
    epsilon = Q(1, 2**40)
    z = (epsilon, epsilon, -2 * epsilon) + (Q(0),) * 42
    phases = tuple(2 * context.pi * (c - 2) / 5 for c in range(5))
    gamma = 1 / (1023 * context.pi)
    results, covariances = [], []
    for sign in (1, -1):
        signed_z = tuple(sign * value for value in z)
        state = (Q(0),) * 45 + signed_z
        covariances.append(tuple(tuple(a * b for b in state) for a in state))
        theta = tuple(
            a + _mp(context, b) for a, b in zip(_carrier(context, phases), signed_z)
        )
        assert _means(signed_z) == (Q(0),) * 5
        for c in range(5):
            _assert_winding_one(context, theta[9 * c : 9 * c + 9])
            _close(context, _means(theta)[c], phases[c])
        for j in range(9):
            _assert_winding_one(context, theta[j::9])
        results.append(
            gamma
            * (_means(_source(context, theta))[0] - _source(context, phases, _CYCLE)[0])
            / 4
        )
    assert covariances[0] == covariances[1]
    e = _mp(context, epsilon)
    expected = (
        -gamma
        * context.cos(2 * context.pi / 5)
        * context.sin(e)
        * (1 - context.cos(e))
        / 9
    )
    _close(context, results[0], expected)
    _close(context, results[1], -expected)
    assert results[0] < 0 < results[1]
    assert sin(I(epsilon)).lo > 0
    assert (1 - cos(I(epsilon))).lo > 0
    assert cos(2 * pi_interval() / 5).lo > 0
    assert 2 * epsilon < Q(1, 10**6)
    # The two states have equal quadratic feedback, not necessarily equal
    # nonlinear storage. The preceding u+- witness owns the latter comparison.


@pytest.mark.parametrize("ratio", (Q(1, 100), Q(1, 10), Q(2, 7)))
def test_positive_response_matrix_changes_the_unchanged_uniform_input_map(
    matrices, ratio
):
    cycle = matrices[1]
    square = exact_square_matrix_power(cycle, 2)
    metric = tuple(
        tuple(Q(i == j) + ratio * square[i][j] for j in range(5)) for i in range(5)
    )
    inverse = exact_matrix_inverse(metric)
    forcing = (Q(1), Q(-1), Q(0), Q(0), Q(0))
    supplied_law_response = _action(inverse, forcing)
    assert sum(forcing) == sum(supplied_law_response) == 0
    assert _action(metric, supplied_law_response) == forcing
    assert supplied_law_response != forcing
    assert _action(square, forcing) != (Q(0),) * 5
    assert _dot(forcing, _action(metric, forcing)) > _dot(forcing, forcing)
    # At an exact carrier, the fine mean derivative responds directly to f.
    # Coordinate/input changes could repair the map, but are different models.
