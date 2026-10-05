"""Independent normalized-derivative and analytic-tail interval controls."""

from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial

import pytest

import tnfr.mathematics._interval_taylor as owner
from tnfr.mathematics._interval_taylor import Jet, arg, atan_ratio, cos, sin, sinc
from tnfr.mathematics._rational_interval import I


def _linear(constant, order=16):
    return Jet((I.coerce(constant), I(1)) + (I(0),) * (order - 1))


def _assert_encloses(coefficients, expected):
    assert len(coefficients) == len(expected)
    for actual, value in zip(coefficients, expected):
        assert actual.contains(value)


def _series_derivative_bounds(kind, point, degree):
    """Independent exact sums with a signed alternating next-term bound.

    The much later cutoff makes all remaining term magnitudes decreasing;
    unlike the implementation this keeps exact Fractions until the end and
    uses an alternating tail instead of an absolute geometric majorant.
    """
    count = 180 if kind == "atan_ratio" else 50
    terms = []
    for index in range((degree + 1) // 2, count + 1):
        denominator = 2 * index + 1
        if kind == "sinc":
            denominator = factorial(denominator)
        terms.append(
            Q((-1) ** index * comb(2 * index, degree), denominator)
            * point ** (2 * index - degree)
        )
    partial = sum(terms[:-1], Q(0))
    other = partial + terms[-1]
    return min(partial, other), max(partial, other)


def test_normalized_sine_and_cosine_coefficients_at_zero_are_exact_series():
    argument = _linear(0)
    expected_sin = [
        Q(0) if degree % 2 == 0 else Q((-1) ** ((degree - 1) // 2), factorial(degree))
        for degree in range(17)
    ]
    expected_cos = [
        Q(0) if degree % 2 else Q((-1) ** (degree // 2), factorial(degree))
        for degree in range(17)
    ]
    _assert_encloses(sin(argument).coeffs, expected_sin)
    _assert_encloses(cos(argument).coeffs, expected_cos)
    assert all(value.width < Q(1, 10**34) for value in sin(argument).coeffs)


def test_reciprocal_of_exponential_jet_has_the_opposite_exponent():
    argument = Jet(tuple(I(Q(1, factorial(degree))) for degree in range(17)))
    inverse = 1 / argument
    _assert_encloses(
        inverse.coeffs, [Q((-1) ** degree, factorial(degree)) for degree in range(17)]
    )
    _assert_encloses((argument * inverse).coeffs, [1] + [0] * 16)


def test_scalar_arithmetic_and_polynomial_powers_preserve_normalization():
    argument = _linear(Q(1, 4), 4)
    result = ((2 * argument + Q(1, 2)) ** 2 - 1) / 4
    _assert_encloses(result.coeffs, [0, 1, 1, 0, 0])
    _assert_encloses((1 - argument).coeffs, [Q(3, 4), -1, 0, 0, 0])
    _assert_encloses(
        (argument / (1 + argument)).coeffs,
        [Q(1, 5), Q(16, 25), -Q(64, 125), Q(256, 625), -Q(1024, 3125)],
    )


@pytest.mark.parametrize(
    "function,denominator", [(sinc, factorial), (atan_ratio, lambda value: value)]
)
def test_removable_zero_retains_every_analytic_derivative(function, denominator):
    coefficients = function(_linear(0)).coeffs
    expected = [
        Q(0) if degree % 2 else Q((-1) ** (degree // 2), denominator(degree + 1))
        for degree in range(17)
    ]
    _assert_encloses(coefficients, expected)
    assert all(coefficients[degree] == I(0) for degree in range(1, 17, 2))


@pytest.mark.parametrize(
    "kind,function,point",
    [
        ("sinc", sinc, Q(1)),
        ("sinc", sinc, Q(-1)),
        ("atan_ratio", atan_ratio, Q(1, 2)),
        ("atan_ratio", atan_ratio, Q(-1, 2)),
    ],
)
def test_analytic_derivative_enclosures_at_admitted_boundaries(kind, function, point):
    coefficients = function(_linear(point)).coeffs
    for degree in (0, 1, 2, 7, 16):
        lower, upper = _series_derivative_bounds(kind, point, degree)
        assert coefficients[degree].lo <= lower <= upper <= coefficients[degree].hi
        assert coefficients[degree].width < Q(1, 10**28)


@pytest.mark.parametrize(
    "kind,function,limit",
    [
        ("sinc", sinc, Q(1)),
        ("atan_ratio", atan_ratio, Q(1, 2)),
    ],
)
def test_interval_expansion_point_encloses_derivatives_across_zero(
    kind, function, limit
):
    coefficients = function(_linear(I(-limit, limit))).coeffs
    for point in (-limit, -limit / 3, Q(0), limit / 3, limit):
        for degree in (0, 1, 2, 7, 16):
            lower, upper = _series_derivative_bounds(kind, point, degree)
            assert coefficients[degree].lo <= lower <= upper <= coefficients[degree].hi


@pytest.mark.parametrize("function", [sinc, atan_ratio])
def test_nonlinear_argument_composes_higher_derivatives(function):
    # For u=t+t^2, [t^j]u^m=binom(m,j-m). This independent formula checks
    # derivative composition beyond a linear argument, including degree 16.
    argument = Jet((I(0), I(1), I(1)) + (I(0),) * 14)
    expected = []
    for degree in range(17):
        value = Q(0)
        for power in range(0, degree + 1, 2):
            if power <= degree <= 2 * power:
                denominator = factorial(power + 1) if function is sinc else power + 1
                value += Q(
                    (-1) ** (power // 2) * comb(power, degree - power), denominator
                )
        expected.append(value)
    _assert_encloses(function(argument).coeffs, expected)


def test_sinc_domain_checks_the_entire_constant_interval():
    function, limit = sinc, Q(1)
    with pytest.raises(ValueError, match="constant interval"):
        function(_linear(I(-limit, limit + Q(1, 2**100))))
    with pytest.raises(ValueError, match="constant interval"):
        function(_linear(-limit - Q(1, 2**100)))


@pytest.mark.parametrize("bound", [I(-1, 1), I(0, 1), I(-1, 0)])
def test_atan_ratio_wide_box_crossing_zero_is_explicitly_unsupported(bound):
    with pytest.raises(ValueError, match="constant interval.*must avoid zero"):
        atan_ratio(_linear(bound))


@lru_cache(maxsize=None)
def _atan_bounds(point):
    """Exact alternating sums, independent of the interval jet recurrence."""
    if point < 0:
        lower, upper = _atan_bounds(-point)
        return -upper, -lower
    if point >= 1:
        a, b = _atan_bounds(Q(1, 5)), _atan_bounds(Q(1, 239))
        quarter_pi = (4 * a[0] - b[1], 4 * a[1] - b[0])
        if point == 1:
            return quarter_pi
        lower, upper = _atan_bounds(1 / point)
        return 2 * quarter_pi[0] - upper, 2 * quarter_pi[1] - lower
    count = 180
    terms = [Q((-1) ** k, 2 * k + 1) * point ** (2 * k + 1) for k in range(count + 1)]
    partial = sum(terms[:-1], Q(0))
    other = partial + terms[-1]
    return min(partial, other), max(partial, other)


def _large_ratio_derivatives(point, order):
    """Closed partial-fraction derivatives, using exact complex rational pairs.

    For k>=1, atan^(k)(x)/k! = (-1)^(k-1)*Im((x-i)^(-k))/k.
    Leibniz's formula with 1/x gives the ratio derivatives without the
    implementation's formal division or arctangent differential recurrence.
    """
    inverse = point / (1 + point**2), 1 / (1 + point**2)
    real, imag = Q(1), Q(0)
    angular = [None]
    for degree in range(1, order + 1):
        real, imag = (
            real * inverse[0] - imag * inverse[1],
            real * inverse[1] + imag * inverse[0],
        )
        angular.append(Q((-1) ** (degree - 1), degree) * imag)
    angle = _atan_bounds(point)
    result = []
    for degree in range(order + 1):
        factor = Q((-1) ** degree) / point ** (degree + 1)
        base = tuple(factor * endpoint for endpoint in angle)
        exact = sum(
            (
                Q((-1) ** (degree - index))
                * angular[index]
                / point ** (degree - index + 1)
                for index in range(1, degree + 1)
            ),
            Q(0),
        )
        result.append((min(base) + exact, max(base) + exact))
    return result


@pytest.mark.parametrize("point", [Q(3, 4), Q(-3, 4), Q(1), Q(-1), Q(2), Q(-2)])
def test_large_atan_ratio_jet_encloses_independent_partial_fraction_derivatives(point):
    coefficients = atan_ratio(_linear(point)).coeffs
    expected = _large_ratio_derivatives(point, 16)
    for coefficient, (lower, upper) in zip(coefficients, expected):
        assert coefficient.lo <= lower <= upper <= coefficient.hi
        assert coefficient.width < Q(1, 10**28)


def test_large_atan_ratio_interval_jet_encloses_the_entire_expansion_box():
    coefficients = atan_ratio(_linear(I(Q(3, 4), Q(5, 4)))).coeffs
    for point in (Q(3, 4), Q(1), Q(5, 4)):
        for coefficient, (lower, upper) in zip(
            coefficients, _large_ratio_derivatives(point, 16)
        ):
            assert coefficient.lo <= lower <= upper <= coefficient.hi


def test_large_atan_ratio_nonlinear_argument_uses_all_normalized_derivatives():
    argument = Jet((I(Q(3, 4)), I(1), I(1)) + (I(0),) * 6)
    derivatives = _large_ratio_derivatives(Q(3, 4), 8)
    for degree, coefficient in enumerate(atan_ratio(argument).coeffs):
        terms = [
            tuple(
                comb(power, degree - power) * endpoint
                for endpoint in derivatives[power]
            )
            for power in range(degree + 1)
            if power <= degree <= 2 * power
        ]
        lower, upper = (sum((term[side] for term in terms), Q(0)) for side in (0, 1))
        assert coefficient.lo <= lower <= upper <= coefficient.hi


@pytest.mark.parametrize(
    "constant", [I(0), I(Q(1, 4)), I(Q(1, 2)), I(Q(-1, 2), Q(1, 2))]
)
def test_zero_safe_atan_ratio_branch_remains_bit_identical(constant):
    argument = _linear(constant)
    assert atan_ratio(argument) == owner._analytic_composition(argument, "atan_ratio")


def test_large_atan_ratio_zero_order_and_constant_jets():
    assert len(atan_ratio(Jet.constant(2, 0)).coeffs) == 1
    assert all(value == I(0) for value in atan_ratio(Jet.constant(2, 16)).coeffs[1:])


def test_orders_types_and_denominators_do_not_invent_admission():
    with pytest.raises(ValueError, match="same order"):
        Jet.constant(1, 2) + Jet.constant(1, 3)
    with pytest.raises(ValueError, match="between"):
        Jet.constant(1, 17)
    with pytest.raises(ValueError, match="between"):
        Jet(())
    with pytest.raises(TypeError, match="integer"):
        Jet.constant(1, True)
    for invalid in (0.0, True):
        with pytest.raises(TypeError, match="exact Fraction or integer"):
            Jet.constant(invalid, 1)
        with pytest.raises(TypeError, match="exact Fraction or integer"):
            _linear(0) * invalid
    with pytest.raises(ZeroDivisionError, match="contains zero"):
        1 / _linear(I(-1, 1))


def test_derivative_family_is_reused_across_growing_flow_jet_orders():
    owner._derivative_family.cache_clear()
    owner._normalized_derivatives.cache_clear()
    for degree in range(1, 10):
        sinc(_linear(Q(3, 8), degree))
    info = owner._derivative_family.cache_info()
    assert info.misses == 1
    assert info.hits == 8


def test_order_zero_functions_and_exact_constant_series_are_supported():
    for function in (sin, cos, sinc, atan_ratio):
        assert len(function(Jet.constant(Q(1, 4), 0)).coeffs) == 1
        assert all(
            value == I(0) for value in function(Jet.constant(Q(1, 4), 16)).coeffs[1:]
        )


def _argument_derivatives(real, imaginary, dx, dy, order):
    """Exact coefficients from Im(log(z+v*t)), independent of the recurrence.

    The local expansion has degree-n coefficient
    Im((-1)**(n+1) * (v/z)**n / n). Complex arithmetic uses rational pairs.
    """
    radius = real**2 + imaginary**2
    ratio = (dx * real + dy * imaginary) / radius, (dy * real - dx * imaginary) / radius
    power = Q(1), Q(0)
    result = []
    for degree in range(1, order + 1):
        power = (
            power[0] * ratio[0] - power[1] * ratio[1],
            power[0] * ratio[1] + power[1] * ratio[0],
        )
        result.append(Q((-1) ** (degree + 1), degree) * power[1])
    return result


@pytest.mark.parametrize("real,imaginary", ((-2, 1), (-2, -1), (2, 0), (0, 2), (0, -2)))
def test_arg_jet_encloses_independent_complex_log_coefficients(real, imaginary):
    real, imaginary = Q(real), Q(imaginary)
    x = Jet((I(real), I(1)) + (I(0),) * 15)
    y = Jet((I(imaginary), I(2)) + (I(0),) * 15)
    result = arg(x, y)
    reference = _argument_derivatives(real, imaginary, Q(1), Q(2), 16)
    _assert_encloses(result.coeffs[1:], reference)
    assert all(value.width < Q(1, 10**27) for value in result.coeffs)
    quarter = _atan_bounds(Q(1))
    if real == 0:
        lower, upper = (2 * endpoint for endpoint in quarter)
        if imaginary < 0:
            lower, upper = -upper, -lower
    else:
        lower, upper = _atan_bounds(imaginary / real)
        if real < 0 and imaginary > 0:
            lower, upper = lower + 4 * quarter[0], upper + 4 * quarter[1]
        elif real < 0:
            lower, upper = lower - 4 * quarter[1], upper - 4 * quarter[0]
    assert result.coeffs[0].lo <= lower <= upper <= result.coeffs[0].hi


def test_arg_jet_encloses_nonlinear_composition_without_scalar_error_derivatives():
    # z=(-2+i)+(1+2i)*(t+t²); binomial coefficients compose the independently
    # calculated complex-log coefficients, exercising every order through16.
    x = Jet((I(-2), I(1), I(1)) + (I(0),) * 14)
    y = Jet((I(1), I(2), I(2)) + (I(0),) * 14)
    coefficients = _argument_derivatives(Q(-2), Q(1), Q(1), Q(2), 16)
    expected = [
        sum(
            (
                comb(power, degree - power) * coefficients[power - 1]
                for power in range(1, degree + 1)
                if power <= degree <= 2 * power
            ),
            Q(0),
        )
        for degree in range(1, 17)
    ]
    _assert_encloses(arg(x, y).coeffs[1:], expected)


def test_arg_jet_handles_a_real_box_crossing_zero_without_inventing_a_pole():
    real, imaginary = I(-3, 3), I(1, 2)
    x = Jet((real, I(1)) + (I(0),) * 7)
    y = Jet((imaginary, I(2)) + (I(0),) * 7)
    result = arg(x, y)
    for a in (real.lo, real.midpoint, real.hi):
        for b in (imaginary.lo, imaginary.midpoint, imaginary.hi):
            expected = _argument_derivatives(a, b, Q(1), Q(2), 8)
            _assert_encloses(result.coeffs[1:], expected)


def test_arg_jet_requires_regular_constant_box_and_matching_jet_orders():
    for real, imaginary in ((I(-2, 1), I(-1, 1)), (I(-1), I(0)), (I(0), I(0))):
        with pytest.raises(ValueError, match="nonpositive real ray"):
            arg(Jet.constant(real, 2), Jet.constant(imaginary, 2))
    with pytest.raises(ValueError, match="same order"):
        arg(Jet.constant(1, 2), Jet.constant(1, 3))
    for invalid in (I(1), 1, True, 1.0):
        with pytest.raises(TypeError, match="require a Jet"):
            arg(invalid, Jet.constant(1, 2))
        with pytest.raises(TypeError, match="require a Jet"):
            arg(Jet.constant(1, 2), invalid)


def test_arg_jet_zero_order_and_constant_series_preserve_the_positive_axis():
    assert arg(Jet.constant(2, 0), Jet.constant(0, 0)).coeffs == (I(0),)
    assert arg(Jet.constant(2, 16), Jet.constant(0, 16)).coeffs == (I(0),) * 17
    assert all(
        value == I(0)
        for value in arg(Jet.constant(-2, 16), Jet.constant(1, 16)).coeffs[1:]
    )
