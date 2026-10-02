"""Exact reference checks for outward interval arithmetic and elementary bounds."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial

import pytest

from tnfr.mathematics._rational_interval import (
    INTERVAL_BITS,
    I,
    atan,
    atan_ratio,
    cos,
    pi_interval,
    sin,
)


def _trig_reference(angle, *, sine=False):
    # Direct Taylor degree121 with a symmetric Lagrange remainder. It has
    # no period reduction and does not share the implementation polynomial.
    partial = sum(
        (
            Q((-1) ** k) * angle ** (2 * k + int(sine)) / factorial(2 * k + int(sine))
            for k in range(61)
        ),
        Q(0),
    )
    remainder = abs(angle) ** 122 / factorial(122)
    return partial - remainder, partial + remainder


def _small_atan_reference(value):
    assert abs(value) <= Q(3, 4)
    partial = sum(
        (Q((-1) ** k) * value ** (2 * k + 1) / (2 * k + 1) for k in range(256)), Q(0)
    )
    omitted = value**513 / 513
    return min(partial, partial + omitted), max(partial, partial + omitted)


def _atan_reference(value):
    if value < 0:
        lo, hi = _atan_reference(-value)
        return -hi, -lo
    if value <= Q(3, 4):
        return _small_atan_reference(value)
    # atan(3/4)+atan(1/7)=pi/4 follows directly from tangent addition.
    # This supplies a pi/4 reference independent of the Machin pi owner.
    first = _small_atan_reference(Q(3, 4))
    second = _small_atan_reference(Q(1, 7))
    quarter_pi = (first[0] + second[0], first[1] + second[1])
    if value <= 1:
        lo, hi = _small_atan_reference((value - 1) / (value + 1))
        return quarter_pi[0] + lo, quarter_pi[1] + hi
    lo, hi = _atan_reference(1 / value)
    return 2 * quarter_pi[0] - hi, 2 * quarter_pi[1] - lo


def test_construction_rounds_outward_and_is_immutable():
    value = Q(1, 3)
    enclosure = I(value)
    assert enclosure.lo < value < enclosure.hi
    assert enclosure.width == Q(1, 1 << INTERVAL_BITS)
    assert enclosure.midpoint == (enclosure.lo + enclosure.hi) / 2
    assert enclosure.radius * 2 == enclosure.width
    assert (1 << INTERVAL_BITS) % enclosure.lo.denominator == 0
    assert (1 << INTERVAL_BITS) % enclosure.hi.denominator == 0
    with pytest.raises(FrozenInstanceError):
        enclosure.lo = Q(0)


@pytest.mark.parametrize("value", (True, 1.0, "1", None))
def test_no_implicit_inexact_scalar_admission(value):
    with pytest.raises(TypeError):
        I(value)
    with pytest.raises(TypeError):
        I(0) + value


def test_invalid_order_and_zero_denominators_reject():
    with pytest.raises(ValueError):
        I(2, 1)
    for denominator in (I(-1, 1), I(0), I(-1, 0), I(0, 1)):
        with pytest.raises(ZeroDivisionError):
            I(1) / denominator
        with pytest.raises(ZeroDivisionError):
            denominator**-1
    with pytest.raises(TypeError):
        I(1) ** True


@pytest.mark.parametrize(
    "left,right",
    (
        (I(-3, -1), I(-2, 4)),
        (I(-3, 2), I(1, 4)),
        (I(1, 3), I(-4, -2)),
        (I(-3, 2), I(-4, 2)),
    ),
)
def test_arithmetic_encloses_all_exact_endpoint_and_midpoint_combinations(left, right):
    for x in (left.lo, left.midpoint, left.hi):
        for y in (right.lo, right.midpoint, right.hi):
            assert (left + right).contains(x + y)
            assert (left - right).contains(x - y)
            assert (left * right).contains(x * y)
            if not right.contains(0):
                assert (left / right).contains(x / y)
    assert (2 - left).contains(2 - left.midpoint)
    assert (2 * left).contains(2 * left.midpoint)
    assert (2 + left).contains(2 + left.midpoint)
    if not left.contains(0):
        assert (2 / left).contains(2 / left.midpoint)


def test_even_powers_retain_interior_zero_and_odd_powers_retain_sign():
    value = I(-3, 2)
    assert value**0 == I(1)
    assert value**2 == I(0, 9)
    assert value**3 == I(-27, 8)
    assert I(-3, -2) ** -2 == I(Q(1, 9), Q(1, 4))
    assert abs(value) == I(0, 3)
    assert abs(I(-3, -2)) == I(2, 3)
    assert value.hull(I(1, 4)) == I(-3, 4)
    assert I(0, 2).subset_of(value)
    assert value.contains(I(-1, 1))
    assert not value.contains(I(-4, 1))


@pytest.mark.parametrize(
    "angle", (Q(-7), Q(-3), Q(-1, 2), Q(0), Q(1, 1000), Q(1, 16), Q(1, 2), Q(3), Q(7))
)
@pytest.mark.parametrize("sine", (False, True))
def test_trigonometry_encloses_independent_direct_taylor_reference(angle, sine):
    result = (sin if sine else cos)(I(angle))
    lower, upper = _trig_reference(angle, sine=sine)
    assert result.lo <= lower <= upper <= result.hi
    assert result.width < Q(1, 10**35)


def test_trigonometric_interval_range_and_pi_uncertainty():
    value = I(-2, 3)
    for point in (Q(-2), Q(-1), Q(0), Q(1), Q(2), Q(3)):
        for function, sine in ((sin, True), (cos, False)):
            lower, upper = _trig_reference(point, sine=sine)
            assert function(value).lo <= lower <= upper <= function(value).hi
    assert cos(I(0)) == I(1)
    assert sin(I(0)).contains(0)
    assert cos(pi_interval() / 2).contains(0)
    assert sin(pi_interval() / 2).contains(1)


@pytest.mark.parametrize(
    "value",
    (Q(-5), Q(-1), Q(-1, 2), Q(0), Q(1, 10000), Q(1, 2), Q(3, 4), Q(1), Q(3, 2), Q(10)),
)
def test_arctangent_encloses_independent_rational_reference(value):
    result = atan(I(value))
    lower, upper = _atan_reference(value)
    assert result.lo <= lower <= upper <= result.hi
    assert result.width < Q(1, 10**34)


@pytest.mark.parametrize(
    "value", (Q(-10), Q(-1), Q(-1, 2), Q(0), Q(1, 1000), Q(1, 2), Q(1), Q(10))
)
def test_atan_ratio_encloses_reference_and_is_even(value):
    result = atan_ratio(I(value))
    assert result == atan_ratio(I(-value))
    if value:
        reference = _atan_reference(value)
        lower, upper = sorted(bound / value for bound in reference)
        assert result.lo <= lower <= upper <= result.hi
    else:
        assert result == I(1)


def test_atan_ratio_handles_zero_crossing_and_subgrid_input_without_division():
    crossing = atan_ratio(I(-2, 1))
    assert crossing.hi == 1
    for value in (Q(-2), Q(-1), Q(0), Q(1)):
        assert crossing.contains(atan_ratio(I(value)))
    tiny = Q(1, 10**90)
    assert atan_ratio(I(tiny)).contains(1)
    assert atan_ratio(I(tiny)).contains(1 - tiny * tiny / 3)
    assert atan(I(tiny)).contains(tiny)


def test_atan_range_uses_monotonicity_across_multiple_reductions():
    bounds = atan(I(-10, 5))
    for value in (Q(-10), Q(-2), Q(-1, 2), Q(0), Q(1), Q(5)):
        assert bounds.contains(atan(I(value)))
