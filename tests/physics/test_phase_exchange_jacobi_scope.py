"""Power cancellation does not certify a Poisson phase/form exchange.

The full-state skew tensor below reproduces the supplied lossless part of
the centered-storage comparison, including its form-mean motion. These
exact controls test that tensor's Jacobi identity. They neither select a
phase law nor exclude a different justified joint geometric structure.
"""

from itertools import combinations

import pytest

from tests.physics._internal_mode_fixture import (
    _metric_differential,
    _nonrepeated_phase_geometry,
)


def _exchange_tensor(s, mobility, coefficient):
    n = mobility.rows
    zero = s.zeros(n)
    return zero.row_join(-coefficient * mobility).col_join(
        (coefficient * mobility).row_join(zero)
    )


def _jacobi(s, tensor, coordinates, i, j, k):
    """Evaluate {z_i,{z_j,z_k}} plus its two cyclic permutations."""
    return s.simplify(
        sum(
            tensor[i, ell] * s.diff(tensor[j, k], coordinate)
            + tensor[j, ell] * s.diff(tensor[k, i], coordinate)
            + tensor[k, ell] * s.diff(tensor[i, j], coordinate)
            for ell, coordinate in enumerate(coordinates)
        )
    )


def test_diagonal_phase_dependent_exchange_has_explicit_jacobi_condition():
    s = pytest.importorskip("sympy")
    x = s.symbols("x0:3", real=True)
    theta = s.symbols("theta0:3", real=True)
    coefficient = s.Symbol("c", positive=True)
    values = tuple(s.Function(f"m{i}")(*theta) for i in range(3))
    tensor = _exchange_tensor(s, s.diag(*values), coefficient)
    coordinates = (*x, *theta)

    for i, j, k in combinations(range(6), 3):
        if i < 3 and j < 3 <= k:
            ell = k - 3
            expected = coefficient**2 * (
                int(j == ell) * values[i] * s.diff(values[j], theta[i])
                - int(i == ell) * values[j] * s.diff(values[i], theta[j])
            )
        else:
            expected = 0
        assert s.simplify(_jacobi(s, tensor, coordinates, i, j, k) - expected) == 0

    # On a positive diagonal-mobility domain these coordinate triples
    # vanish iff every off-diagonal derivative partial_i m_j vanishes.
    # All other independent coordinate triples have already vanished.
    for i in range(3):
        for j in range(3):
            if i != j:
                assert (
                    s.simplify(
                        _jacobi(s, tensor, coordinates, i, j, 3 + j)
                        / (coefficient**2 * values[i])
                        - s.diff(values[j], theta[i])
                    )
                    == 0
                )


def test_constant_positive_mobility_is_a_poisson_control():
    s = pytest.importorskip("sympy")
    coordinates = s.symbols("x0:3 theta0:3", real=True)
    coefficient = s.Symbol("c", positive=True)
    values = s.symbols("m0:3", positive=True)
    tensor = _exchange_tensor(s, s.diag(*values), coefficient)

    assert tensor + tensor.T == s.zeros(6)
    assert s.simplify(tensor.det()) == coefficient**6 * s.prod(values) ** 2
    assert all(
        _jacobi(s, tensor, coordinates, *triple) == 0
        for triple in combinations(range(6), 3)
    )


def test_retained_phase_metric_has_nonzero_exact_jacobi_obstruction():
    s, _, rows, phases = _nonrepeated_phase_geometry()
    metric, derivative, _, _ = _metric_differential(s, rows, phases)
    coefficient = s.Symbol("c", positive=True)
    assert metric[0, 0] == metric[1, 1] == 3 * (1 + s.sqrt(3))
    assert s.simplify(derivative[0, 1] - (3 - 9 * s.sqrt(3) / s.pi)) == 0

    # Jacobi(x_1,x_0,theta_0)=c^2*m_1*partial_1 m_0.
    obstruction = (
        -(coefficient**2) * derivative[0, 1] / (metric[1, 1] * metric[0, 0] ** 2)
    )
    expected = (
        coefficient**2 * (3 * s.sqrt(3) - s.pi) / (9 * s.pi * (1 + s.sqrt(3)) ** 3)
    )
    assert s.simplify(obstruction - expected) == 0
    # The elementary bound pi<4<sqrt(27)=3*sqrt(3) proves strict
    # positivity without a tolerance, finite difference or trajectory.
    assert s.pi < 4
    assert 4**2 < 27


def test_full_exchange_retains_centered_work_and_unprojected_mean_source():
    s = pytest.importorskip("sympy")
    n = 3
    x = s.Matrix(s.symbols("x0:3", real=True))
    g = s.Matrix(s.symbols("g0:3", real=True))
    diagonal = s.symbols("h0:3", positive=True)
    metric = s.diag(*diagonal)
    mobility = metric.inv()
    beta, weight = s.symbols("beta w", positive=True)
    projection = s.eye(n) - s.ones(n) / n
    centered = projection * x
    tensor = _exchange_tensor(s, mobility, weight / beta)
    # The full x gradient of ||Pi*x||^2/2 is y=Pi*x. The phase
    # gradient is -beta*H*g on the regular canonical metric domain.
    storage_gradient = centered.col_join(-beta * metric * g)
    exchange = (tensor * storage_gradient).applyfunc(s.simplify)

    assert exchange[:n, :] == weight * g
    assert (exchange[n:, :] - (weight / beta) * mobility * centered).applyfunc(
        s.simplify
    ) == s.zeros(n, 1)
    assert s.simplify(storage_gradient.dot(exchange)) == 0
    assert s.simplify(sum(exchange[:n, :]) / n - weight * sum(g) / n) == 0
    assert (projection * exchange[:n, :] - weight * projection * g).applyfunc(
        s.simplify
    ) == s.zeros(n, 1)
    # An independently projected cross tensor has a different Jacobi
    # problem. No result about that tensor is inferred from this one.
