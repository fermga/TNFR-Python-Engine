"""Independent polynomial and sharp-error controls for rational Gauss rules."""

from fractions import Fraction as Q
from math import factorial

import pytest
import sympy as sp

from tnfr.mathematics._gauss_legendre import (
    composite_gauss_remainder,
    gauss_legendre_rule,
)
from tnfr.mathematics._rational_interval import I, sqrt


@pytest.fixture(scope="module")
def rules():
    return {order: gauss_legendre_rule(order) for order in range(1, 17)}


def _rodrigues_polynomial(order):
    argument = sp.Symbol("x")
    polynomial = sp.Poly(
        sp.diff((argument**2 - 1) ** order, argument, order)
        / (2**order * factorial(order)),
        argument,
    )
    coefficients = tuple(Q(coefficient) for coefficient in polynomial.all_coeffs())

    def evaluate(value):
        result = Q(0)
        for coefficient in coefficients:
            result = result * value + coefficient
        return result

    return evaluate


def test_every_admitted_rule_has_all_simple_roots_and_positive_weights(rules):
    for order, (nodes, weights) in rules.items():
        assert len(nodes) == len(weights) == order
        polynomial = _rodrigues_polynomial(order)
        for node, weight in zip(nodes, weights):
            assert -1 < node.lo <= node.hi < 1
            assert weight.lo > 0
            assert polynomial(node.lo) * polynomial(node.hi) <= 0
        assert all(left.hi < right.lo for left, right in zip(nodes, nodes[1:]))
        assert nodes == tuple(-node for node in reversed(nodes))
        assert weights == tuple(reversed(weights))
        # These are n disjoint real root brackets for a degree-n polynomial.
        # Counting them excludes missing or multiply enclosed roots.


def test_all_polynomial_moments_through_the_exactness_degree_are_enclosed(rules):
    for order, (nodes, weights) in rules.items():
        for degree in range(2 * order):
            observed = sum(
                (weight * node**degree for node, weight in zip(nodes, weights)),
                I(0),
            )
            exact = Q(0) if degree % 2 else Q(2, degree + 1)
            assert observed.contains(exact)
            assert observed.width < Q(1, 10**24)


def test_low_order_rules_have_independent_closed_form_controls(rules):
    assert rules[1] == ((I(0),), (I(2),))
    nodes, weights = rules[2]
    assert all((node**2).contains(Q(1, 3)) for node in nodes)
    assert all(weight.contains(1) for weight in weights)
    nodes, weights = rules[3]
    assert nodes[1] == I(0)
    assert weights[1].contains(Q(8, 9))
    assert (nodes[0] ** 2).contains(Q(3, 5))
    nodes, weights = rules[4]
    inner = sqrt((3 - 2 * sqrt(I(Q(6, 5)))) / 7)
    outer = sqrt((3 + 2 * sqrt(I(Q(6, 5)))) / 7)
    expected_nodes = (-outer, -inner, inner, outer)
    outer_weight = (18 - sqrt(I(30))) / 36
    inner_weight = (18 + sqrt(I(30))) / 36
    expected_weights = (outer_weight, inner_weight, inner_weight, outer_weight)
    for actual, expected in zip(nodes + weights, expected_nodes + expected_weights):
        assert actual.lo <= expected.hi and expected.lo <= actual.hi


@pytest.mark.parametrize("order", [1, 2, 4, 8, 12, 16])
def test_degree_two_n_polynomial_attains_the_gauss_remainder_constant(rules, order):
    nodes, weights = rules[order]
    integral = Q(2, 2 * order + 1)
    quadrature = sum(
        (weight * node ** (2 * order) for node, weight in zip(nodes, weights)),
        I(0),
    )
    error = integral - quadrature
    remainder = composite_gauss_remainder(
        order=order,
        panels=1,
        interval_length=2,
        derivative_bound=factorial(2 * order),
    )
    assert error.lo > 0
    assert error.contains(remainder)


def test_composite_rule_and_vector_norm_have_the_same_positive_kernel_constant():
    order, panels, length = 2, 3, Q(3)
    nodes, weights = gauss_legendre_rule(order)
    half_step = length / (2 * panels)
    quadrature = I(0)
    for panel in range(panels):
        center = (2 * panel + 1) * half_step
        quadrature += half_step * sum(
            (
                weight * (center + half_step * node) ** 4
                for node, weight in zip(nodes, weights)
            ),
            I(0),
        )
    scalar_error = length**5 / 5 - quadrature
    vector_error = sqrt((3 * scalar_error) ** 2 + (4 * scalar_error) ** 2)
    bound = composite_gauss_remainder(
        order=order,
        panels=panels,
        interval_length=length,
        derivative_bound=5 * factorial(4),
    )
    assert scalar_error.lo > 0
    assert vector_error.contains(bound)
    refined = composite_gauss_remainder(
        order=order,
        panels=2 * panels,
        interval_length=length,
        derivative_bound=5 * factorial(4),
    )
    assert refined == bound / 2 ** (2 * order)


def test_exact_scalar_normalization_precedes_remainder_powers():
    np = pytest.importorskip("numpy")
    length = Q(np.int64(2**62), np.int64(3))
    derivative = Q(np.int64(7), np.int64(11))
    result = composite_gauss_remainder(
        order=12, panels=64, interval_length=length, derivative_bound=derivative
    )
    expected = composite_gauss_remainder(
        order=12,
        panels=64,
        interval_length=Q(2**62, 3),
        derivative_bound=Q(7, 11),
    )
    assert result == expected > 0
    assert type(result.numerator) is type(result.denominator) is int
    tiny = Q(1, 10**500)
    assert (
        composite_gauss_remainder(
            order=16, panels=1, interval_length=tiny, derivative_bound=tiny
        )
        > 0
    )


@pytest.mark.parametrize("bad", [True, False, 1.0, Q(1), "4", None])
def test_order_admission_precedes_cached_rule_access(rules, bad):
    with pytest.raises(TypeError):
        gauss_legendre_rule(bad)
    with pytest.raises(TypeError):
        composite_gauss_remainder(
            order=bad, panels=1, interval_length=1, derivative_bound=1
        )


@pytest.mark.parametrize("bad", [0, -1, 17])
def test_orders_outside_the_certified_implementation_are_rejected(bad):
    with pytest.raises(ValueError):
        gauss_legendre_rule(bad)


@pytest.mark.parametrize("bad", [True, 1.0, Q(1), "1", None])
def test_panel_counts_require_ordinary_integers(bad):
    with pytest.raises(TypeError):
        composite_gauss_remainder(
            order=4, panels=bad, interval_length=1, derivative_bound=1
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_panel_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        composite_gauss_remainder(
            order=4, panels=bad, interval_length=1, derivative_bound=1
        )


@pytest.mark.parametrize("name", ["interval_length", "derivative_bound"])
@pytest.mark.parametrize("bad", [True, 1.0, float("nan"), float("inf"), "1", None])
def test_quadrature_bounds_do_not_implicitly_admit_inexact_scalars(name, bad):
    arguments = dict(order=4, panels=1, interval_length=1, derivative_bound=1)
    arguments[name] = bad
    with pytest.raises(TypeError):
        composite_gauss_remainder(**arguments)


@pytest.mark.parametrize("name", ["interval_length", "derivative_bound"])
def test_negative_bounds_reject_and_zero_bounds_remain_zero(name):
    arguments = dict(order=4, panels=1, interval_length=1, derivative_bound=1)
    arguments[name] = -1
    with pytest.raises(ValueError):
        composite_gauss_remainder(**arguments)
    arguments[name] = 0
    assert composite_gauss_remainder(**arguments) == 0
