"""Normalized Taylor jets with outward rational interval coefficients.

The coefficient at index ``j`` encloses a derivative divided by ``j!``.
Operations concern formal truncated series; a caller validating an ODE must
independently enclose its trajectory and its Taylor remainder. In particular,
an elementary function's scalar approximation error is never silently used
as a bound on its derivative coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial

from ._rational_interval import I
from ._rational_interval import arg as interval_arg
from ._rational_interval import atan as interval_atan
from ._rational_interval import cos as interval_cos
from ._rational_interval import sin as interval_sin

__all__ = ("Jet", "MAX_ORDER", "sin", "cos", "sinc", "atan_ratio", "arg")

MAX_ORDER = 16
_ZERO = I(0)
_SINC_TERMS = 32
_ATAN_RATIO_TERMS = 128


def _order(value):
    if type(value) is not int:
        raise TypeError("Taylor jet order must be an integer")
    if not 0 <= value <= MAX_ORDER:
        raise ValueError(f"Taylor jet order must lie between 0 and {MAX_ORDER}")
    return value


def _zero(value):
    return value.lo == 0 and value.hi == 0


@dataclass(frozen=True, slots=True)
class Jet:
    """Immutable formal series with zero through sixteen normalized derivatives.

    Scalars are exact integers, Fractions or existing rational intervals.
    Jet operands must have the same order; silently extending a truncated
    derivative list would incorrectly invent zero higher derivatives.
    """

    coeffs: tuple[I, ...]

    def __post_init__(self):
        coefficients = tuple(I.coerce(value) for value in self.coeffs)
        _order(len(coefficients) - 1)
        object.__setattr__(self, "coeffs", coefficients)

    @classmethod
    def constant(cls, value, order):
        """Make a constant series, retaining the admitted scalar uncertainty."""
        order = _order(order)
        return cls((I.coerce(value),) + (_ZERO,) * order)

    @property
    def order(self):
        return len(self.coeffs) - 1

    def _same_order(self, other):
        if self.order != other.order:
            raise ValueError("Taylor jet operands must have the same order")

    def __neg__(self):
        return Jet(tuple(-value for value in self.coeffs))

    def __pos__(self):
        return self

    def __add__(self, other):
        if isinstance(other, Jet):
            self._same_order(other)
            return Jet(
                tuple(left + right for left, right in zip(self.coeffs, other.coeffs))
            )
        return Jet((self.coeffs[0] + I.coerce(other),) + self.coeffs[1:])

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Jet):
            self._same_order(other)
            return Jet(
                tuple(left - right for left, right in zip(self.coeffs, other.coeffs))
            )
        return Jet((self.coeffs[0] - I.coerce(other),) + self.coeffs[1:])

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, Jet):
            factor = I.coerce(other)
            return Jet(
                tuple(
                    _ZERO if _zero(value) or _zero(factor) else value * factor
                    for value in self.coeffs
                )
            )
        self._same_order(other)
        coefficients = []
        for degree in range(self.order + 1):
            total = _ZERO
            for index in range(degree + 1):
                left, right = self.coeffs[index], other.coeffs[degree - index]
                if not _zero(left) and not _zero(right):
                    total = total + left * right
            coefficients.append(total)
        return Jet(tuple(coefficients))

    __rmul__ = __mul__

    def reciprocal(self):
        """Invert a formal series whose entire constant interval avoids zero."""
        inverse = self.coeffs[0].reciprocal()
        coefficients = [inverse]
        for degree in range(1, self.order + 1):
            total = _ZERO
            for index in range(1, degree + 1):
                left, right = self.coeffs[index], coefficients[degree - index]
                if not _zero(left) and not _zero(right):
                    total = total + left * right
            coefficients.append(-inverse * total if not _zero(total) else _ZERO)
        return Jet(tuple(coefficients))

    def __truediv__(self, other):
        if isinstance(other, Jet):
            self._same_order(other)
            return self * other.reciprocal()
        return self * I.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    def __pow__(self, exponent):
        if type(exponent) is not int:
            raise TypeError("Taylor jet powers require an integer exponent")
        if exponent < 0:
            return self.reciprocal() ** (-exponent)
        result = Jet.constant(1, self.order)
        factor = self
        while exponent:
            if exponent & 1:
                result = result * factor
            exponent >>= 1
            if exponent:
                factor = factor * factor
        return result


def _jet(value):
    if not isinstance(value, Jet):
        raise TypeError("Taylor elementary functions require a Jet")
    return value


@lru_cache(maxsize=256)
def _sincos_coefficients(coefficients):
    if len(coefficients) <= 1:
        return (interval_sin(coefficients[0]),), (interval_cos(coefficients[0]),)
    # Flow jets request ascending orders. Reuse the cached prefix while
    # retaining the original summation and outward-rounding order.
    sine, cosine = _sincos_coefficients(coefficients[:-1])
    degree = len(coefficients) - 1
    sine_total = cosine_total = _ZERO
    for index in range(1, degree + 1):
        if not _zero(coefficients[index]):
            factor = coefficients[index] * index
            sine_total = sine_total + factor * cosine[degree - index]
            cosine_total = cosine_total - factor * sine[degree - index]
    return sine + (sine_total / degree,), cosine + (cosine_total / degree,)


def sin(value):
    """Enclose the sine jet through the coupled sine/cosine recurrence."""
    return Jet(_sincos_coefficients(_jet(value).coeffs)[0])


def cos(value):
    """Enclose the cosine jet through the coupled sine/cosine recurrence."""
    return Jet(_sincos_coefficients(_jet(value).coeffs)[1])


def arg(real, imaginary):
    """Enclose the principal argument jet on a regular constant rectangle.

    Both arguments must be same-order jets. The interval Arg owner admits
    the entire constant rectangle against the nonpositive-real branch cut.
    Higher normalized derivatives follow analytically from
    ``Arg(x+i*y)'=(x*y'-y*x')/(x*x+y*y)`` and formal integration. They are
    not obtained by differentiating an approximate scalar angle or by
    extending a bounded atan-ratio power series beyond its domain.

    The squared-radius constant uses interval squares to preserve a known
    positive denominator even when one coordinate crosses zero. Unresolved
    denominator rounding still rejects. This local derivative enclosure
    does not certify that a future trajectory remains in the regular domain.
    """
    real, imaginary = _jet(real), _jet(imaginary)
    real._same_order(imaginary)
    coefficients = [interval_arg(real.coeffs[0], imaginary.coeffs[0])]
    if not real.order:
        return Jet(tuple(coefficients))
    x, y = Jet(real.coeffs[:-1]), Jet(imaginary.coeffs[:-1])
    dx = Jet(tuple(index * real.coeffs[index] for index in range(1, real.order + 1)))
    dy = Jet(
        tuple(index * imaginary.coeffs[index] for index in range(1, real.order + 1))
    )
    radius_squared = x**2 + y**2
    radius_squared = Jet(
        (real.coeffs[0] ** 2 + imaginary.coeffs[0] ** 2,) + radius_squared.coeffs[1:]
    )
    derivative = (x * dy - y * dx) / radius_squared
    coefficients.extend(
        coefficient / index for index, coefficient in enumerate(derivative.coeffs, 1)
    )
    return Jet(tuple(coefficients))


@lru_cache(maxsize=2)
def _derivative_polynomials(kind):
    terms = _SINC_TERMS if kind == "sinc" else _ATAN_RATIO_TERMS
    polynomials = []
    for degree in range(MAX_ORDER + 1):
        coefficients = []
        for index in range((degree + 1) // 2, terms):
            denominator = factorial(2 * index + 1) if kind == "sinc" else 2 * index + 1
            coefficients.append(Q((-1) ** index * comb(2 * index, degree), denominator))
        polynomials.append(tuple(coefficients))
    return tuple(polynomials)


def _derivative_tail(kind, maximum, degree):
    """Bound the normalized derivative of every omitted series term.

    The first omitted power is 2*K. The absolute subsequent-term ratio is
    at most rho below; its decreasing rational factors give a geometric
    majorant. For atan-ratio a beneficial (2*k+1)/(2*k+3) factor is dropped.
    These bounds concern every |u| <= maximum, including intervals crossing
    zero, and apply individually to all requested derivatives.
    """
    terms = _SINC_TERMS if kind == "sinc" else _ATAN_RATIO_TERMS
    power = 2 * terms
    if kind == "sinc":
        first = maximum ** (power - degree) / (
            factorial(degree) * (power + 1) * factorial(power - degree)
        )
        ratio = maximum * maximum / ((power + 2 - degree) * (power + 1 - degree))
    else:
        first = Q(comb(power, degree), power + 1) * maximum ** (power - degree)
        ratio = (
            maximum
            * maximum
            * Q(
                (power + 2) * (power + 1),
                (power + 2 - degree) * (power + 1 - degree),
            )
        )
    if ratio >= 1:
        raise ValueError("analytic derivative-tail geometric bound is unresolved")
    return first / (1 - ratio)


@lru_cache(maxsize=512)
def _derivative_family(kind, constant):
    maximum = constant.abs_max
    limit = Q(1) if kind == "sinc" else Q(1, 2)
    if maximum > limit:
        raise ValueError(
            f"{kind} jet requires its constant interval within [-{limit}, {limit}]"
        )
    if _zero(constant):
        return tuple(
            (
                _ZERO
                if degree % 2
                else I(
                    Q(
                        (-1) ** (degree // 2),
                        factorial(degree + 1) if kind == "sinc" else degree + 1,
                    )
                )
            )
            for degree in range(MAX_ORDER + 1)
        )
    square = constant**2
    derivatives = []
    for degree, polynomial in enumerate(_derivative_polynomials(kind)):
        value = _ZERO
        for coefficient in reversed(polynomial):
            value = value * square + coefficient
        if degree % 2:
            value = value * constant
        tail = _derivative_tail(kind, maximum, degree)
        derivatives.append(value + I(-tail, tail))
    return tuple(derivatives)


@lru_cache(maxsize=4096)
def _normalized_derivatives(kind, constant, order):
    # A fixed expansion point is reused as flow jets grow in order. Computing
    # the family once avoids repeating each rational series at every order.
    return _derivative_family(kind, constant)[: order + 1]


def _analytic_composition(value, kind):
    value = _jet(value)
    derivatives = _normalized_derivatives(kind, value.coeffs[0], value.order)
    delta = Jet((_ZERO,) + value.coeffs[1:])
    # Delta has exactly zero constant coefficient. Terms above the jet order
    # therefore contribute nothing to any retained coefficient, even when
    # the expansion point itself is an interval.
    result = Jet.constant(derivatives[-1], value.order)
    for derivative in reversed(derivatives[:-1]):
        result = result * delta + derivative
    return result


def sinc(value):
    """Enclose sin(u)/u and its jets, continuously one at zero; |u[0]| <= 1."""
    return _analytic_composition(value, "sinc")


def atan_ratio(value):
    """Enclose the continuous atan(u)/u jet on an admitted expansion box.

    The zero-safe derivative series is unchanged for ``abs(u[0]) <= 1/2``.
    Outside that box the entire constant interval must avoid zero. There,
    the principal arctangent jet follows from ``atan(u)'=u'/(1+u**2)``;
    formal division by ``u`` gives the same analytic function. A wide box
    crossing zero remains unsupported rather than creating a singular jet.
    """
    value = _jet(value)
    if value.coeffs[0].abs_max <= Q(1, 2):
        return _analytic_composition(value, "atan_ratio")
    if value.coeffs[0].contains(0):
        raise ValueError(
            "atan_ratio jet constant interval outside [-1/2, 1/2] must avoid zero"
        )
    coefficients = [interval_atan(value.coeffs[0])]
    if value.order:
        derivative = Jet(
            tuple(index * value.coeffs[index] for index in range(1, value.order + 1))
        )
        truncated = Jet(value.coeffs[:-1])
        rate = derivative / (1 + truncated**2)
        coefficients.extend(
            coefficient / index for index, coefficient in enumerate(rate.coeffs, 1)
        )
    return Jet(tuple(coefficients)) / value
