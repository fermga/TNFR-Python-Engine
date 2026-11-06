"""Small Gauss--Legendre rules with certified rational nodes and weights.

The rule is mathematical quadrature data, independent of any response. Exact
sign bisection isolates all roots before the shared outward arithmetic is
used for the weights. No floating root estimate or response-dependent search
is consumed. The remainder helper bounds quadrature error only: evaluation,
node/weight materialization and subsequent reductions remain the caller's
responsibility.
"""

from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache
from math import factorial

from .._exact_time import exact_or_represented_real
from ._rational_interval import INTERVAL_BITS, I

_MAX_ORDER = 16


def _order(value):
    if type(value) is not int:
        raise TypeError("quadrature order must be an ordinary Python integer")
    if not 1 <= value <= _MAX_ORDER:
        raise ValueError(f"quadrature order must lie in [1, {_MAX_ORDER}]")
    return value


def _nonnegative_exact(value, name):
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError(f"{name} requires an exact Fraction or integer")
    value = exact_or_represented_real(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@lru_cache(maxsize=_MAX_ORDER)
def _integer_coefficients(order):
    """Ascending integer coefficients of 2**order * P_order."""
    coefficients = [0] * (order + 1)
    for index in range(order // 2 + 1):
        degree = order - 2 * index
        coefficients[degree] = (
            (-1) ** index
            * factorial(2 * order - 2 * index)
            // (factorial(index) * factorial(order - index) * factorial(degree))
        )
    return tuple(coefficients)


def _polynomial_sign(order, argument):
    """Sign of P_order at an exact rational, using integer arithmetic."""
    numerator, denominator = argument.numerator, argument.denominator
    coefficients = _integer_coefficients(order)
    value, denominator_power = coefficients[-1], 1
    for coefficient in reversed(coefficients[:-1]):
        denominator_power *= denominator
        value = value * numerator + coefficient * denominator_power
    return (value > 0) - (value < 0)


@lru_cache(maxsize=_MAX_ORDER)
def _nodes(order):
    if order == 1:
        return (I(0),)
    # Previous roots supply deterministic separator candidates. We do not
    # rely on approximate interlacing: all resulting signs are verified.
    separators = (Q(-1),) + tuple(node.midpoint for node in _nodes(order - 1)) + (Q(1),)
    signs = tuple(_polynomial_sign(order, point) for point in separators)
    if any(left * right != -1 for left, right in zip(signs, signs[1:])):
        raise ArithmeticError("Legendre root separation was not certified")
    target_width = Q(1, 1 << (INTERVAL_BITS + 2))
    roots = []
    for left, right, left_sign in zip(separators, separators[1:], signs):
        while right - left > target_width:
            midpoint = (left + right) / 2
            sign = _polynomial_sign(order, midpoint)
            if not sign:
                left = right = midpoint
                break
            if sign == left_sign:
                left = midpoint
            else:
                right = midpoint
        roots.append(I(left, right))
    # n disjoint sign brackets give n distinct real roots; degree n leaves
    # no other roots and makes each enclosed root simple.
    if any(left.hi >= right.lo for left, right in zip(roots, roots[1:])):
        raise ArithmeticError("outward Legendre root enclosures overlap")
    return tuple(roots)


def _legendre_derivative(order, argument):
    """Enclose P_order' by differentiating the three-term recurrence."""
    previous, current = I(1), argument
    previous_derivative, current_derivative = I(0), I(1)
    for degree in range(2, order + 1):
        following = (
            (2 * degree - 1) * argument * current - (degree - 1) * previous
        ) / degree
        following_derivative = (
            (2 * degree - 1) * (current + argument * current_derivative)
            - (degree - 1) * previous_derivative
        ) / degree
        previous, current = current, following
        previous_derivative, current_derivative = (
            current_derivative,
            following_derivative,
        )
    return current_derivative


@lru_cache(maxsize=_MAX_ORDER)
def _rule(order):
    nodes = _nodes(order)
    weights = []
    for node in nodes:
        derivative = _legendre_derivative(order, node)
        denominator = (1 - node**2) * derivative**2
        if denominator.lo <= 0:
            raise ArithmeticError("Legendre weight denominator was not separated")
        weights.append(2 / denominator)
    return nodes, tuple(weights)


def gauss_legendre_rule(order):
    """Return immutable outward node and weight tuples for [-1, 1].

    Orders 1 through 16 are supported. Admission precedes cache access, so
    booleans or integer-like aliases cannot reuse a cached ordinary integer.
    The weights are positive and the exact underlying rule integrates every
    polynomial of degree at most 2*order-1 exactly. Returned interval sums
    enclose that identity; their positive widths are retained.
    """
    return _rule(_order(order))


def composite_gauss_remainder(*, order, panels, interval_length, derivative_bound):
    """Bound the norm error of equal-panel Gauss quadrature.

    The caller supplies a bound on the norm of the 2*order derivative on the
    whole interval. The positive Peano kernel gives the scalar Gauss constant
    also for a finite-dimensional vector norm. For interval length T and P
    panels this returns M*T**(2*n+1)*(n!)**4 /
    (P**(2*n)*(2*n+1)*((2*n)!)**3). It contains no arithmetic-error estimate.
    """
    order = _order(order)
    if type(panels) is not int:
        raise TypeError("quadrature panels must be an ordinary Python integer")
    if panels <= 0:
        raise ValueError("quadrature panels must be positive")
    length = _nonnegative_exact(interval_length, "interval_length")
    bound = _nonnegative_exact(derivative_bound, "derivative_bound")
    constant = Q(
        factorial(order) ** 4,
        (2 * order + 1) * factorial(2 * order) ** 3,
    )
    return bound * length ** (2 * order + 1) * constant / panels ** (2 * order)
