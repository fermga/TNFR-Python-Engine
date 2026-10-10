"""Exact algebra controls for a supplied collective matrix-response law.

Rational gamma values test identities valid for every positive gamma. They
are neither replacements for the model's fixed gamma nor physical bounds.
No trajectory, formation certificate or reserved response is evaluated.
"""

from fractions import Fraction as Q
import hashlib
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_power,
    exact_square_matrix_product,
    exact_symmetric_semidefinite,
)


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
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_sine_regeneration():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


_EDGES = tuple((c, (c + 1) % 5) for c in range(5))
_IDENTITY = tuple(tuple(Q(i == j) for j in range(5)) for i in range(5))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _action(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _edge_form_gradient(form):
    gradient = [Q(0)] * 5
    for left, right in _EDGES:
        difference = form[left] - form[right]
        gradient[left] += difference
        gradient[right] -= difference
    return tuple(gradient)


def _quarter_turn_sine_source(phases):
    """Evaluate the edge sine sum exactly for integer multiples of pi/2."""
    values = (Q(0), Q(1), Q(0), Q(-1))
    source = [Q(0)] * 5
    for left, right in _EDGES:
        current = values[(phases[right] - phases[left]) % 4]
        source[left] += current
        source[right] -= current
    return tuple(source)


@pytest.fixture(scope="module")
def cycle_laplacian():
    matrix = [[Q(0)] * 5 for _ in range(5)]
    for left, right in _EDGES:
        matrix[left][left] += 1
        matrix[right][right] += 1
        matrix[left][right] -= 1
        matrix[right][left] -= 1
    return tuple(map(tuple, matrix))


def _response_matrices(laplacian, ratio):
    square = exact_square_matrix_power(laplacian, 2)
    metric = tuple(
        tuple(base + ratio * value for base, value in zip(unit, row))
        for unit, row in zip(_IDENTITY, square)
    )
    return metric, exact_matrix_inverse(metric)


@pytest.mark.parametrize("ratio", (Q(0), Q(1, 100), Q(2, 7), Q(5)))
def test_response_is_positive_symmetric_and_preserves_common_mode(
    cycle_laplacian, ratio
):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    assert exact_symmetric_semidefinite(metric, strict=True)
    assert exact_symmetric_semidefinite(response, strict=True)
    assert exact_square_matrix_product(metric, response) == _IDENTITY
    assert exact_square_matrix_product(response, metric) == _IDENTITY
    storage_metric = exact_square_matrix_product(cycle_laplacian, metric)
    assert storage_metric == exact_square_matrix_product(metric, cycle_laplacian)
    assert exact_symmetric_semidefinite(storage_metric)
    assert not exact_symmetric_semidefinite(storage_metric, strict=True)
    common = (Q(1),) * 5
    assert _action(metric, common) == _action(response, common) == common
    assert _action(storage_metric, common) == (Q(0),) * 5
    if ratio == 0:
        assert metric == response == _IDENTITY
        assert storage_metric == cycle_laplacian


@pytest.mark.parametrize("ratio", (Q(0), Q(1, 100), Q(2, 7), Q(5)))
@pytest.mark.parametrize("gamma", (Q(1, 1023), Q(3, 7)))
def test_complete_rows_balance_storage_and_transform_the_driven_input(
    cycle_laplacian, ratio, gamma
):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    forcing = tuple(Q(value, 19) for value in (-2, 5, 1, -3, -1))
    source = _quarter_turn_sine_source((0, 1, -1, 2, 0))
    gradient = _edge_form_gradient(form)
    assert gradient == _action(cycle_laplacian, form)
    unforced = tuple((-q + gamma * s) / 4 for q, s in zip(gradient, source))
    form_rate = _action(response, tuple(a + b for a, b in zip(unforced, forcing)))
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    assert sum(form) == sum(forcing) == sum(form_rate) == sum(phase_rate) == 0

    # Differentiate U.L.U/2 + r*(L U).L.(L U)/2 directly. Its gradient is
    # L U + r*L^3 U; the potential gradient is the negative edge sine sum.
    cubic_gradient = _edge_form_gradient(_edge_form_gradient(gradient))
    storage_gradient = tuple(
        q + ratio * cube for q, cube in zip(gradient, cubic_gradient)
    )
    derivative = _dot(storage_gradient, form_rate) - _dot(source, phase_rate)
    loss = _dot(gradient, gradient) / 4
    work = _dot(gradient, forcing)
    assert loss > 0 and work != 0
    assert derivative == -loss + work
    assert _action(metric, form_rate) == tuple(a + b for a, b in zip(unforced, forcing))

    wrong_input_rate = tuple(
        a + b for a, b in zip(_action(response, unforced), forcing)
    )
    wrong_derivative = _dot(storage_gradient, wrong_input_rate) - _dot(
        source, phase_rate
    )
    if ratio:
        assert wrong_input_rate != form_rate
        assert wrong_derivative != -loss + work
    else:
        assert form_rate == tuple(a + b for a, b in zip(unforced, forcing))
        assert wrong_input_rate == form_rate


@pytest.mark.parametrize("ratio", (Q(0), Q(1, 100), Q(2, 7)))
def test_charge_chart_transforms_phase_row_storage_and_work(cycle_laplacian, ratio):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    gamma = Q(3, 7)
    form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    forcing = tuple(Q(value, 19) for value in (-2, 5, 1, -3, -1))
    source = _quarter_turn_sine_source((0, 1, -1, 2, 0))
    gradient = _edge_form_gradient(form)
    charge = _action(metric, form)
    recovered = _action(response, charge)
    assert recovered == form
    charge_rate = tuple(
        (-q + gamma * s) / 4 + f for q, s, f in zip(gradient, source, forcing)
    )
    form_rate = _action(response, charge_rate)
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    transformed_phase_rate = tuple(
        gamma * q / 4 for q in _edge_form_gradient(recovered)
    )
    wrong_phase_rate = tuple(gamma * q / 4 for q in _edge_form_gradient(charge))
    assert _action(metric, form_rate) == charge_rate
    assert transformed_phase_rate == phase_rate

    original_form_storage = _dot(form, _edge_form_gradient(charge)) / 2
    transformed_form_storage = _dot(charge, _edge_form_gradient(recovered)) / 2
    assert original_form_storage == transformed_form_storage
    # The changed chart has gradient L M Z. Its work conjugate stays L U.
    transformed_derivative = _dot(gradient, charge_rate) - _dot(
        source, transformed_phase_rate
    )
    assert transformed_derivative == -_dot(gradient, gradient) / 4 + _dot(
        gradient, forcing
    )
    if ratio:
        assert wrong_phase_rate != phase_rate
        assert _dot(source, wrong_phase_rate) != _dot(source, phase_rate)
    else:
        assert charge == form and wrong_phase_rate == phase_rate


def test_positive_definite_response_is_not_a_nonnegative_averaging_map(cycle_laplacian):
    metric, response = _response_matrices(cycle_laplacian, Q(1, 100))
    # Construct from edge incidence, invert, then compare the claimed row;
    # this is not a lookup table used to define the tested matrix itself.
    assert response[0] == tuple(Q(value, 2305) for value in (2181, 81, -19, -19, 81))
    assert exact_square_matrix_product(metric, response) == _IDENTITY
    assert exact_symmetric_semidefinite(response, strict=True)
    nonnegative = (Q(0), Q(0), Q(1), Q(0), Q(0))
    result = _action(response, nonnegative)
    assert result[0] == Q(-19, 2305) < 0
    assert _dot(nonnegative, result) > 0
    assert sum(result) == sum(nonnegative)


@pytest.mark.parametrize("ratio", (Q(1, 10**9), Q(1, 100), Q(2, 7), Q(5)))
@pytest.mark.parametrize("gamma", (Q(1, 1023), Q(3, 7)))
def test_old_storage_can_increase_while_own_storage_decreases(
    cycle_laplacian, ratio, gamma
):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    # Phi=(0,pi/2,0,0,0) supplies this exact current from the actual edge law.
    source = _quarter_turn_sine_source((0, 1, 0, 0, 0))
    assert source == (Q(1), Q(-2), Q(1), Q(0), Q(0))
    form = tuple(-gamma * ratio * value / 2 for value in _edge_form_gradient(source))
    gradient = _edge_form_gradient(form)
    form_rate = _action(
        response, tuple((-q + gamma * s) / 4 for q, s in zip(gradient, source))
    )
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    old_derivative = _dot(gradient, form_rate) - _dot(source, phase_rate)
    own_derivative = _dot(_action(metric, gradient), form_rate) - _dot(
        source, phase_rate
    )
    # Algebra gives these identities for every positive r and gamma, while
    # SPD of M makes the first expression strictly positive for q!=0.
    assert any(gradient)
    assert old_derivative == _dot(gradient, _action(response, gradient)) / 4 > 0
    assert own_derivative == -_dot(gradient, gradient) / 4 < 0
    extra_storage = ratio * _dot(gradient, _edge_form_gradient(gradient)) / 2
    assert extra_storage > 0
