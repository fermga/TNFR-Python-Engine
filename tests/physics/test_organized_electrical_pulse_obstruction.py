"""Exact kinematic controls for the precontact electrical pulse obstruction.

The five-child observation support is C5, before the sixth child is attached.
These static paths check energy inequalities, not realizable observation words,
source transport, supplied current controls or any frozen response producer.
"""

from fractions import Fraction as Q

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_product,
)


def _action(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _quadratic(matrix, vector):
    return sum(a * b for a, b in zip(vector, _action(matrix, vector)))


def _edge_matrix(size, edges):
    matrix = [[Q(0) for _ in range(size)] for _ in range(size)]
    for left, right, weight in edges:
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(tuple(row) for row in matrix)


def _squared_integral(coefficients):
    """Integrate a polynomial square over [0,1] by exact monomials."""
    return sum(
        left * right / Q(i + j + 1)
        for i, left in enumerate(coefficients)
        for j, right in enumerate(coefficients)
    )


def _evaluate(coefficients, value):
    return sum(coefficient * value**i for i, coefficient in enumerate(coefficients))


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def precontact():
    size = 5
    ring = tuple((node, (node + 1) % size, Q(1)) for node in range(size))
    laplacian = _edge_matrix(size, ring)
    inverse = exact_matrix_inverse(
        tuple(tuple(value + Q(1, size) for value in row) for row in laplacian)
    )
    centered_inverse = tuple(
        tuple(value - Q(1, size) for value in row) for row in inverse
    )
    projection = tuple(
        tuple(Q(int(i == j)) - Q(1, size) for j in range(size)) for i in range(size)
    )
    return laplacian, centered_inverse, projection


def test_five_child_reference_has_passive_pentagram_capacitance(precontact):
    laplacian, capacitance, projection = precontact
    assert len(laplacian) == 5  # The post-contact six-island wheel is different.
    assert exact_square_matrix_product(laplacian, capacitance) == projection
    assert exact_square_matrix_product(capacitance, laplacian) == projection
    assert all(sum(row) == 0 for row in capacitance)
    pentagram = tuple((node, (node + 2) % 5, Q(1, 5)) for node in range(5))
    assert capacitance == _edge_matrix(5, pentagram)
    for node in range(5):
        assert capacitance[node][node] == Q(2, 5)
        assert capacitance[node][(node + 1) % 5] == 0
        assert capacitance[node][(node + 2) % 5] == -Q(1, 5)
    all_pairs = tuple((i, j, Q(1, 5)) for i in range(5) for j in range(i + 1, 5))
    assert projection == _edge_matrix(5, all_pairs)


def test_actual_marked_event_directions_have_positive_relative_energy(precontact):
    _, capacitance, conductance = precontact
    epsilon = Q(1, 10)
    squared_amplitudes = Q(0)
    for child in range(5):
        direction = tuple(Q(int(node == child)) - Q(1, 5) for node in range(5))
        assert sum(direction) == 0
        assert sum(value**2 for value in direction) == Q(4, 5)
        for multiplier in (1, -2, 1):
            amplitude = multiplier * epsilon
            jump = tuple(amplitude * value for value in direction)
            assert _quadratic(capacitance, jump) == Q(2, 5) * amplitude**2 > 0
            assert _quadratic(conductance, jump) == Q(4, 5) * amplitude**2 > 0
            squared_amplitudes += amplitude**2
    assert squared_amplitudes == Q(3, 10)
    # A common phase shift is a gauge direction, excluded from the obstruction.
    common = (epsilon,) * 5
    assert _quadratic(capacitance, common) == 0
    assert _quadratic(conductance, common) == 0


def test_affine_phase_path_attains_integral_and_peak_lower_bounds(precontact):
    _, capacitance, conductance = precontact
    a0, duration = Q(7, 13), Q(11, 17)
    jump = (Q(2, 25),) + (Q(-1, 50),) * 4
    derivative = (Q(1),)
    velocity_factor = _squared_integral(derivative)
    voltage = tuple(a0 * value / duration for value in jump)
    charge = _action(capacitance, voltage)
    assert sum(charge) == 0
    assert any(charge)  # Equality does not supply a zero-charge endpoint pulse.
    energy = _quadratic(capacitance, voltage) / 2
    heat = duration * _quadratic(conductance, voltage) * velocity_factor
    assert duration * energy == a0**2 * _quadratic(capacitance, jump) / (2 * duration)
    assert energy == a0**2 * _quadratic(capacitance, jump) / (2 * duration**2)
    assert heat == a0**2 * _quadratic(conductance, jump) / duration


def test_cubic_rest_endpoint_path_has_strictly_larger_energy(precontact):
    _, capacitance, conductance = precontact
    a0, duration = Q(7, 13), Q(11, 17)
    jump = (Q(2, 25),) + (Q(-1, 50),) * 4
    profile = (Q(0), Q(0), Q(3), Q(-2))
    derivative = tuple(i * value for i, value in enumerate(profile) if i)
    assert _evaluate(profile, Q(0)) == 0
    assert _evaluate(profile, Q(1)) == 1
    for endpoint in (Q(0), Q(1)):
        voltage = tuple(
            a0 * value * _evaluate(derivative, endpoint) / duration for value in jump
        )
        assert _action(capacitance, voltage) == (Q(0),) * 5
    factor = _squared_integral(derivative)
    assert factor == Q(6, 5) > 1
    capacitive_lower = a0**2 * _quadratic(capacitance, jump) / (2 * duration)
    joule_lower = a0**2 * _quadratic(conductance, jump) / duration
    # Direct polynomial integration retains the same strict factor for both.
    assert factor * capacitive_lower > capacitive_lower > 0
    assert factor * joule_lower > joule_lower > 0
    # 6*r*(1-r) peaks at r=1/2; its squared peak is 9/4.
    peak_factor = _evaluate(derivative, Q(1, 2)) ** 2
    assert peak_factor == Q(9, 4)
    assert peak_factor * capacitive_lower / duration > capacitive_lower / duration


def test_shortening_a_fixed_phase_path_increases_heat_and_peak_storage(precontact):
    _, capacitance, conductance = precontact
    jump = (Q(2, 25),) + (Q(-1, 50),) * 4
    derivative = (Q(0), Q(6), Q(-6))
    durations = (Q(3, 7), Q(1, 7))
    integrated, peaks, heat = [], [], []
    for duration in durations:
        integrated.append(
            _quadratic(capacitance, jump)
            * _squared_integral(derivative)
            / (2 * duration)
        )
        voltage = tuple(
            value * _evaluate(derivative, Q(1, 2)) / duration for value in jump
        )
        peaks.append(_quadratic(capacitance, voltage) / 2)
        heat.append(
            _quadratic(conductance, jump) * _squared_integral(derivative) / duration
        )
    assert integrated[1] == 3 * integrated[0] > 0
    assert heat[1] == 3 * heat[0] > 0
    assert peaks[1] == 9 * peaks[0] > 0
