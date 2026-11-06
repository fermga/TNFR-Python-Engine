"""Independent coefficient and error algebra for a finite nonlinear protocol.

The only auxiliary numerical calculation is an analytic heat coefficient,
using a symmetric eigensystem and quadrature. No complete sine trajectory,
acquisition assessment, retained producer or archived worker is executed.
"""

from fractions import Fraction as Q
from math import factorial

import numpy as np
import pytest

from tnfr.physics.relational_sine_class_nonlinear_protocol import (
    _ideal_heat_remainder,
    bound_sine_class_nonlinear_protocol,
)


@pytest.fixture(scope="module")
def report():
    return bound_sine_class_nonlinear_protocol(
        mediator_class=2,
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
def graph():
    edges = tuple(
        (9 * component + local, 9 * component + (local + 1) % 9)
        for component in range(3)
        for local in range(9)
    ) + ((4, 13), (13, 22))
    degrees = tuple(sum(node in edge for edge in edges) for node in range(27))
    laplacian = np.zeros((27, 27))
    for i, j in edges:
        laplacian[i, i] += 1
        laplacian[j, j] += 1
        laplacian[i, j] -= 1
        laplacian[j, i] -= 1
    return edges, degrees, laplacian


@pytest.fixture(scope="module")
def spectral_heat(graph):
    _, degrees, laplacian = graph
    root_degree = np.sqrt(np.array(degrees))
    symmetric = laplacian / root_degree[:, None] / root_degree[None, :]
    eigenvalues, vectors = np.linalg.eigh(symmetric)

    def heat(time):
        symmetric_heat = (vectors * np.exp(-time * eigenvalues)) @ vectors.T
        return symmetric_heat * root_degree[None, :] / root_degree[:, None]

    return heat


def _mixed_cubic(graph, first, second, cosines):
    edges, degrees, _ = graph
    forcing = [0] * 27
    for index, (i, j) in enumerate(edges):
        p, q = first[j] - first[i], second[j] - second[i]
        cosine = cosines[1 if 9 <= index < 18 else 0] if index < 27 else 1
        current = cosine * p * q * (p + q) / 2
        forcing[i] -= current / degrees[i]
        forcing[j] += current / degrees[j]
    return tuple(forcing)


def test_heat_representation_uses_actual_nonregular_degrees(graph, spectral_heat):
    _, degrees, laplacian = graph
    h = spectral_heat(0.7)
    assert np.max(abs(h.sum(axis=1) - 1)) < 5e-14
    assert h.min() > -5e-15
    assert (
        np.max(abs(np.array(degrees)[:, None] * h - np.array(degrees)[None, :] * h.T))
        < 5e-14
    )
    # The nonsymmetric normalized generator needs the degree similarity.
    assert abs(h[4, 13] - h[13, 4]) > 1e-3
    normalized = laplacian / np.array(degrees)[:, None]
    n, t = 8, 0.7
    term = np.eye(27)
    polynomial = term.copy()
    for order in range(1, n + 1):
        term = -t * normalized @ term / order
        polynomial += term
    assert np.linalg.norm(h - polynomial, ord=np.inf) < (2 * t) ** (n + 1) / factorial(
        n + 1
    )


def test_cubic_edge_operator_and_heat_tail_are_normalized_without_edge_count(graph):
    a, b, eta = Q(2, 7), Q(-3, 11), Q(1, 17)
    first = tuple(a * Q((i * 3) % 5 - 2, 2) for i in range(27))
    second = tuple(b * Q((i * 5) % 7 - 3, 3) for i in range(27))
    approximate_first = tuple(
        value + abs(a) * eta * (1 if i % 2 else -1) for i, value in enumerate(first)
    )
    approximate_second = tuple(
        value + abs(b) * eta * (1 if i % 3 else -1) for i, value in enumerate(second)
    )
    cosines = (Q(3, 4), Q(1, 5))
    exact = _mixed_cubic(graph, first, second, cosines)
    approximate = _mixed_cubic(graph, approximate_first, approximate_second, cosines)
    unit_bound = 4 * (abs(a) ** 2 * abs(b) + abs(a) * abs(b) ** 2)
    assert max(map(abs, exact)) <= unit_bound
    assert max(abs(x - y) for x, y in zip(exact, approximate)) <= unit_bound * (
        (1 + eta) ** 3 - 1
    )
    row = tuple(Q(i + 1, sum(range(1, 28))) for i in range(27))
    approximate_row = tuple(
        value + (eta if i == 22 else 0) for i, value in enumerate(row)
    )
    propagated = sum(x * y for x, y in zip(row, exact))
    approximate_propagated = sum(x * y for x, y in zip(approximate_row, approximate))
    assert abs(propagated - approximate_propagated) <= unit_bound * ((1 + eta) ** 4 - 1)


@pytest.mark.parametrize("amplitude,time", [(Q(1, 1000), Q(2)), (Q(3, 101), Q(7, 3))])
def test_gamma_six_remainder_is_the_integral_of_distinct_error_channels(
    amplitude, time
):
    g = Q(1, 3000)
    d, ell = 1 - 2 * g**2 * time**2, 1 - 2 * g * time
    shifted_cubic = 8 * g**6 * amplitude**3 / d**3 * time**3 / 3
    odd_feedback = 8 * g**6 * amplitude**3 / (d**3 * ell) * time**3 / 3
    fifth_sine = Q(4, 15) * g**6 * amplitude**5 / d**5 * time
    # 4 g^2 integral_0^T (T-t) [c1*t+c3*t^3] dt.
    c1 = Q(4, 3) * g**4 * amplitude**3 / (d**3 * ell)
    c3 = Q(8, 3) * g**6 * amplitude**3 / (d**3 * ell**2)
    tangent_feedback = 4 * g**2 * (c1 * time**3 / 6 + c3 * time**5 / 20)
    assembled = shifted_cubic + odd_feedback + fifth_sine + tangent_feedback
    compact = (
        g**6 * amplitude**3 * time**3 / d**3 * (Q(8, 3) + Q(32, 9) / ell)
        + Q(4, 15) * g**6 * amplitude**5 * time / d**5
        + Q(8, 15) * g**8 * amplitude**3 * time**5 / (d**3 * ell**2)
    )
    assert assembled == compact
    assert assembled == _ideal_heat_remainder(amplitude, time, g)
    assert d > 0 and ell > 0


def test_long_horizon_source_work_identity_and_eight_error_separation_budget():
    g, t, s, a, b, eps, delta = (
        Q(1, 3000),
        Q(2),
        Q(1),
        Q(1, 2000),
        Q(1, 2000),
        Q(1, 10**32),
        Q(1, 10**30),
    )
    ell = 1 - 2 * g * t
    assert ell > 0
    assert 4 * eps / ell < Q(41, 10**33)
    before_second = (a + eps + 2 * g * s * eps) / (1 - 2 * g**2 * s**2)
    first_work = Q(3, 2) * a**2 + 6 * a * eps
    second_work = Q(3, 2) * b**2 + 6 * b * before_second
    assert first_work < Q(2, 10**6) and second_work < Q(2, 10**6)
    assert 22 * eps**2 + first_work + second_work < Q(1, 388800)
    form = (a + b + eps + 2 * g * t * eps) / (1 - 2 * g**2 * t**2)
    phase = eps + 2 * g * t * form
    assert 27 * (form**2 + phase**2) < Q(1, 144)
    # A noiseless signed interval separated by 8 delta has disjoint mixed
    # recorded intervals from the linear comparator's [-4delta,4delta].
    true_upper = -9 * delta
    recorded_upper = true_upper + 4 * delta
    comparator_lower = -4 * delta
    assert recorded_upper < comparator_lower
    assert -8 * delta + 4 * delta == comparator_lower


def test_independent_spectral_quadrature_checks_all_rational_heat_channels(
    graph, spectral_heat, report
):
    edges, degrees, _ = graph
    abscissae, weights = np.polynomial.legendre.leggauss(48)
    impulse = np.eye(27)[:, 4]
    channels = np.zeros(3)
    a = b = 1 / 2000
    for abscissa, weight in zip(abscissae, weights):
        time = 1.5 + abscissa / 2
        first = a * (impulse - spectral_heat(time)[:, 4])
        second = b * (impulse - spectral_heat(time - 1)[:, 4])
        row = spectral_heat(2 - time)[22, :]
        forcing = np.zeros((3, 27))
        for index, (i, j) in enumerate(edges):
            p, q = first[j] - first[i], second[j] - second[i]
            current = p * q * (p + q) / 2
            channel = 2 if index >= 27 else 1 if 9 <= index < 18 else 0
            forcing[channel, i] -= current / degrees[i]
            forcing[channel, j] += current / degrees[j]
        channels += weight / 2 * (forcing @ row)
    # This checks the algebra through a different numerical representation;
    # the rigorous enclosure remains the rational polynomial and its tail.
    np.testing.assert_allclose(
        channels,
        [float(value) for value in report.heat_cubic_channel_coefficients],
        rtol=1e-10,
        atol=1e-30,
    )
    assert tuple(np.sign(channels)) == (1, -1, -1)
    weighted_coefficient = np.dot(
        channels, [np.cos(2 * np.pi / 9), np.cos(4 * np.pi / 9), 1]
    )
    midpoint = sum(report.heat_polynomial_coefficient_bounds, Q(0)) / 2
    assert np.isclose(weighted_coefficient, float(midpoint), rtol=1e-10, atol=1e-30)


def test_finite_bound_retains_source_errors_and_disjoins_the_four_record_sets(report):
    lower, upper = report.gamma_fourth_scaled_heat_bounds
    remainder = report.ideal_mixed_remainder_upper_bound
    source = report.source_mixed_error_upper_bound
    assert source == 4 * report.endpoint_radius / (1 - 2 * report.gamma_bounds.hi * 2)
    assert report.true_mixed_bounds == (
        lower - remainder - source,
        upper + remainder + source,
    )
    assert report.true_mixed_bounds[1] < -8 * report.readout_error_bound
    assert report.recorded_mixed_bounds[1] < -4 * report.readout_error_bound
    assert report.four_record_sets_disjoint
    assert report.all_identities_certified and report.all_work_within_allowances
    assert all(
        history.tangent_endpoint_discrepancy_upper_bound is None
        for history in report.histories
    )
