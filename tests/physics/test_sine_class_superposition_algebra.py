"""Independent finite jets and budgets for two matched donor form probes.

These controls evaluate static full-state rows, exact polynomial coefficients
and rational bounds. No acquired source, trajectory or reserved response runs.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath
import pytest


@pytest.fixture(scope="module")
def graph():
    cycles = tuple(
        tuple((9 * part + j, 9 * part + (j + 1) % 9) for j in range(9))
        for part in range(3)
    )
    edges = sum(cycles, ()) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    reflection = tuple(9 * (i // 9) + 8 - i % 9 for i in range(27))
    return edges, degrees, reflection


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 70
    return context


def _laplacian(graph, values):
    edges, degrees, _ = graph
    result = [0] * 27
    for i, j in edges:
        result[i] += (values[i] - values[j]) / degrees[i]
        result[j] += (values[j] - values[i]) / degrees[j]
    return tuple(result)


def _rows(graph, x, theta, gamma, mp):
    edges, degrees, _ = graph
    ax = _laplacian(graph, x)
    f = [mp.mpf(0)] * 27
    for i, j in edges:
        value = mp.sin(theta[j] - theta[i])
        f[i] += value / degrees[i]
        f[j] -= value / degrees[j]
    return tuple(-v + gamma * w for v, w in zip(ax, f)), tuple(gamma * v for v in ax)


def _acceleration(graph, x, theta, gamma, mp):
    edges, degrees, _ = graph
    dx, dy = _rows(graph, x, theta, gamma, mp)
    adx = _laplacian(graph, dx)
    derivative = [mp.mpf(0)] * 27
    for i, j in edges:
        value = mp.cos(theta[j] - theta[i]) * (dy[j] - dy[i])
        derivative[i] += value / degrees[i]
        derivative[j] -= value / degrees[j]
    return tuple(-v + gamma * w for v, w in zip(adx, derivative)), tuple(
        gamma * v for v in adx
    )


def _target(k, mp):
    return tuple(
        2 * mp.pi * (k if 9 <= i < 18 else 1) * (i % 9 - 4) / 9 for i in range(27)
    )


@pytest.mark.parametrize("k", [1, 2])
def test_full54_sign_reflection_equivariance_with_arbitrary_residuals(graph, mp, k):
    _, _, reflection = graph
    target = _target(k, mp)
    x = tuple(mp.mpf((7 * i + 2) % 19 - 9) / 103 for i in range(27))
    y = tuple(mp.mpf((5 * i + 3) % 17 - 8) / 211 for i in range(27))
    gamma = mp.mpf(1) / 13
    original = _rows(graph, x, tuple(a + b for a, b in zip(target, y)), gamma, mp)
    transformed = _rows(
        graph,
        tuple(-x[i] for i in reflection),
        tuple(target[j] - y[i] for j, i in enumerate(reflection)),
        gamma,
        mp,
    )
    assert max(
        abs(transformed[channel][j] + original[channel][i])
        for channel in range(2)
        for j, i in enumerate(reflection)
    ) < mp.mpf("1e-65")
    assert reflection[4] == 4 and reflection[22] == 22
    # Equivariance transforms arbitrary errors; it does not make them symmetric.
    assert any(x[j] != -x[i] for j, i in enumerate(reflection))


def test_exact_postevent_mixed_acceleration_keeps_nonzero_baseline_gap(graph, mp):
    gamma, b = mp.mpf(1) / 17, mp.mpf(2) / 101
    target = _target(2, mp)
    x0 = tuple(mp.mpf((i + 3) % 11) / 107 for i in range(27))
    xa = tuple(value + mp.mpf((3 * i + 1) % 7) / 503 for i, value in enumerate(x0))
    theta0 = tuple(value + mp.mpf((i + 2) % 5) / 100 for i, value in enumerate(target))
    thetaa = tuple(
        value + mp.mpf((2 * i + 1) % 11) / 71 for i, value in enumerate(theta0)
    )

    def kicked(values):
        return tuple(value + (b if i == 4 else 0) for i, value in enumerate(values))

    histories = ((kicked(xa), thetaa), (xa, thetaa), (kicked(x0), theta0), (x0, theta0))
    signs = (1, -1, -1, 1)
    first = [_rows(graph, *state, gamma, mp) for state in histories]
    second = [_acceleration(graph, *state, gamma, mp) for state in histories]
    assert max(
        abs(sum(sign * state[channel][i] for sign, state in zip(signs, histories)))
        for channel in range(2)
        for i in range(27)
    ) < mp.mpf("1e-65")
    assert max(
        abs(sum(sign * rows[channel][i] for sign, rows in zip(signs, first)))
        for channel in range(2)
        for i in range(27)
    ) < mp.mpf("1e-65")
    d0, da = theta0[13] - theta0[22], thetaa[13] - thetaa[22]
    actual = sum(sign * rows[0][22] for sign, rows in zip(signs, second))
    assert d0 != 0 and da != d0
    assert abs(actual - gamma**2 * b * (mp.cos(d0) - mp.cos(da)) / 12) < mp.mpf("1e-65")
    assert abs(actual - gamma**2 * b * (1 - mp.cos(da)) / 12) > mp.mpf("1e-10")


def _multiply(left, right):
    result = [Q(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return tuple(result)


def _cube(values):
    return _multiply(_multiply(values, values), values)


def test_full_edge_tensors_and_integrated_first_cubic_mixed_coefficient(graph):
    edges, degrees, reflection = graph
    impulse = tuple(Q(i == 4) for i in range(27))
    phase_slope = _laplacian(graph, impulse)
    quadratic, cubic = [Q(0)] * 27, [Q(0)] * 27
    for index, (i, j) in enumerate(edges):
        # Independent rational coefficients test the graph/tensor identity;
        # no rational value is substituted for the actual class cosine.
        sine = (Q(2, 7), Q(3, 11), Q(2, 7))[index // 9] if index < 27 else Q(0)
        cosine = (Q(3, 4), Q(1, 5), Q(3, 4))[index // 9] if index < 27 else Q(1)
        difference = phase_slope[j] - phase_slope[i]
        quadratic[i] -= sine * difference**2 / degrees[i]
        quadratic[j] += sine * difference**2 / degrees[j]
        cubic[i] -= cosine * difference**3 / (6 * degrees[i])
        cubic[j] += cosine * difference**3 / (6 * degrees[j])
    assert tuple(quadratic[i] for i in reflection) == tuple(-v for v in quadratic)
    assert quadratic[22] == 0
    assert tuple(cubic[i] for i in reflection) == tuple(cubic)
    assert cubic[22] == Q(1, 1152)
    a, b, s, u = Q(2, 7), Q(-1, 11), Q(3, 13), Q(2, 17)
    both, first, second = _cube((a * s, a + b)), _cube((a * s, a)), _cube((Q(0), b))
    integral = sum(
        (v - va - vb) * u ** (order + 1) / (order + 1)
        for order, (v, va, vb) in enumerate(zip(both, first, second))
    )
    independent = cubic[22] * integral
    i1 = s**2 * u**2 / 2 + 2 * s * u**3 / 3 + u**4 / 4
    i2 = s * u**3 / 3 + u**4 / 4
    assert independent == (a**2 * b * i1 + a * b**2 * i2) / 384
    scale = Q(5, 3)
    scaled = sum(
        (v - va - vb) * (scale * u) ** (order + 1) / (order + 1)
        for order, (v, va, vb) in enumerate(
            zip(_cube((a * scale * s, a + b)), _cube((a * scale * s, a)), second)
        )
    )
    assert scaled == scale**4 * integral
    # Fixed s has a generally nonzero u^2 onset, not a u^4-only onset.
    assert (both[1] - first[1] - second[1]) / 2 == 3 * a**2 * b * s**2 / 2


def _budget():
    g, gl, a, b, t, eps, delta = (
        Q(1, 3000),
        Q(1, 3216),
        Q(1, 2000),
        Q(1, 2000),
        Q(1, 10000),
        Q(1, 10**32),
        Q(1, 10**30),
    )
    s, d, ell = t / 2, 1 - 2 * g**2 * t**2, 1 - 3 * t
    ct = Q(8, 3) * g**4 * t**4 / (d**3 * ell) * (1 + Q(2, 3) * g**2 * t**2 / ell)
    return g, gl, a, b, s, t, eps, delta, ct, ell


def test_rational_projection_budget_and_positive_actual_event_curvature():
    g, gl, a, b, s, t, eps, delta, ct, ell = _budget()
    # Integrating the two even-forcing monomials independently recovers Ct.
    d = 1 - 2 * g**2 * t**2
    direct_cubic = Q(32, 3) * g**4 / d**3 * t**4 / 4 / ell
    odd_feedback = Q(32, 3) * g**6 / (d**3 * ell) * t**6 / 6 / ell
    assert ct == direct_cubic + odd_feedback
    assert ct * ((a + b) ** 3 + a**3 + b**3) + 4 * eps / ell < 4 * delta
    assert ct * (a + b) ** 3 + 2 * eps / ell < delta
    q = a / (1 - 2 * g**2 * s**2)
    baseline = 2 * eps / (1 - 3 * s)
    remainder = 4 * g * q * (1 + 2 * g**2 * s) * s**2 + baseline
    qmin, qmax = gl * a * s / 4 - remainder, g * a * s / 4 + remainder
    assert 0 < qmin < qmax < 1
    assert gl**2 * b * (qmin**2 / 3 - baseline**2 / 2) / 12 > 0
    factor = (
        a**2 * b * (s**2 * s**2 / 2 + 2 * s * s**3 / 3 + s**4 / 4)
        + a * b**2 * (s * s**3 / 3 + s**4 / 4)
    ) / 384
    assert 0 < g**4 * factor < Q(1, 2**128)


def test_sequential_event_work_uses_carried_state_and_split_identity_budget(graph):
    edges, degrees, _ = graph
    x = tuple(Q((5 * i + 1) % 13 - 6, 107) for i in range(27))
    a, b = Q(2, 101), Q(-3, 103)
    carried = tuple(
        value + (a if i == 4 else 0) + Q((3 * i + 1) % 5, 1009)
        for i, value in enumerate(x)
    )

    def energy(values):
        return sum((values[i] - values[j]) ** 2 / 2 for i, j in edges)

    def work(values, amplitude):
        jumped = tuple(
            value + (amplitude if i == 4 else 0) for i, value in enumerate(values)
        )
        laplacian = sum(
            values[4] - values[j if i == 4 else i] for i, j in edges if 4 in (i, j)
        )
        assert (
            energy(jumped) - energy(values)
            == amplitude * laplacian + Q(3, 2) * amplitude**2
        )
        assert (
            sum(
                d * (after - before)
                for d, after, before in zip(degrees, jumped, values)
            )
            == 3 * amplitude
        )
        return energy(jumped) - energy(values)

    work(x, a)
    assert work(carried, b) != work(x, b)
    g, _, a, b, s, t, eps, _, _, _ = _budget()
    qa = (a + eps + 2 * g * s * eps) / (1 - 2 * g**2 * s**2)
    excess = 22 * eps**2 + 6 * a * eps + Q(3, 2) * a**2 + 6 * b * qa + Q(3, 2) * b**2
    fullq = (a + b + eps + 2 * g * t * eps) / (1 - 2 * g**2 * t**2)
    phase = eps + 2 * g * t * fullq
    assert excess < Q(1, 388800)
    assert 27 * (fullq**2 + phase**2) < Q(1, 144)


def test_four_endpoint_noise_overlap_is_a_constructive_record_set_statement():
    _, _, a, b, _, _, eps, delta, ct, ell = _budget()
    amplitudes = (a + b, a, b, Q(0))
    radii = tuple(ct * amplitude**3 + 2 * eps / ell for amplitude in amplitudes)
    assert max(radii) < delta
    base, first, second = Q(1, 7), Q(1, 11), Q(-2, 13)
    tangent = (base + first + second, base + first, base + second, base)
    signs = (1, -1, -1, 1)
    assert sum(sign * value for sign, value in zip(signs, tangent)) == 0
    # Arbitrary extrema of the admitted full-minus-tangent error box, not
    # invented TNFR trajectories: each has the same admissible endpoint record.
    for corner in product((-1, 1), repeat=4):
        complete = tuple(
            value + direction * radius
            for value, direction, radius in zip(tangent, corner, radii)
        )
        sensor_error = tuple(
            record - actual for record, actual in zip(tangent, complete)
        )
        assert max(map(abs, sensor_error)) <= delta
        recorded = tuple(value + error for value, error in zip(complete, sensor_error))
        assert recorded == tangent
        assert sum(sign * value for sign, value in zip(signs, recorded)) == 0
