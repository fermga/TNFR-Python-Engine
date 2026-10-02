"""Small outward rational intervals for conditional continuous enclosures.

Every endpoint is rounded outwards to a fixed dyadic grid. Elementary bounds
use the existing rational Machin pi/cosine owners and alternating arctangent
series. These are numerical enclosures, not assertions about any model or
trajectory. Wide or unresolved bounds must be handled by their caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from ._phase_midpoint import _pi_bounds
from ._phase_resultant_chamber import certified_cosine_bounds

__all__ = (
    "I",
    "RationalInterval",
    "INTERVAL_BITS",
    "INTERVAL_METHOD",
    "pi_interval",
    "sin",
    "cos",
    "atan",
    "atan_ratio",
)

INTERVAL_BITS = 128
INTERVAL_METHOD = "rational_outward_dyadic128_machin_trigonometry_v1"
_SCALE = 1 << INTERVAL_BITS
_ATAN_TERMS = 64


def _exact(value):
    if isinstance(value, Q):
        return value
    if type(value) is int:
        return Q(value)
    raise TypeError("interval endpoints require exact Fraction or integer values")


def _down(value):
    return Q((value.numerator * _SCALE) // value.denominator, _SCALE)


def _up(value):
    return Q(-((-value.numerator * _SCALE) // value.denominator), _SCALE)


@dataclass(frozen=True, slots=True, init=False)
class I:  # noqa: E742 - conventional private interval-arithmetic notation
    """Immutable bounded interval with outward 128-bit dyadic endpoints.

    Construction retains the supplied real interval by outward rounding;
    even a single non-dyadic rational can therefore have positive width.
    Floats and booleans require an explicit admission outside this kernel.
    """

    lo: Q
    hi: Q

    def __init__(self, lo, hi=None):
        lo = _exact(lo)
        hi = lo if hi is None else _exact(hi)
        if lo > hi:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lo", _down(lo))
        object.__setattr__(self, "hi", _up(hi))

    @classmethod
    def point(cls, value):
        """Enclose an exact rational point, retaining rounding uncertainty."""
        return cls(value)

    @classmethod
    def coerce(cls, value):
        """Admit an existing interval or an exact rational scalar."""
        return value if isinstance(value, cls) else cls(value)

    @property
    def midpoint(self):
        return (self.lo + self.hi) / 2

    @property
    def radius(self):
        return (self.hi - self.lo) / 2

    @property
    def width(self):
        return self.hi - self.lo

    @property
    def abs_max(self):
        return max(abs(self.lo), abs(self.hi))

    def contains(self, value):
        """Whether the whole supplied rational point/interval is enclosed."""
        if isinstance(value, I):
            return self.lo <= value.lo and value.hi <= self.hi
        value = _exact(value)
        return self.lo <= value <= self.hi

    def __contains__(self, value):
        return self.contains(value)

    def subset_of(self, other):
        return I.coerce(other).contains(self)

    def hull(self, other):
        other = I.coerce(other)
        return I(min(self.lo, other.lo), max(self.hi, other.hi))

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __pos__(self):
        return self

    def __abs__(self):
        return I(
            0 if self.contains(0) else min(abs(self.lo), abs(self.hi)), self.abs_max
        )

    def __add__(self, other):
        other = I.coerce(other)
        return I(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __sub__(self, other):
        other = I.coerce(other)
        return I(self.lo - other.hi, self.hi - other.lo)

    def __rsub__(self, other):
        return I.coerce(other) - self

    def __mul__(self, other):
        other = I.coerce(other)
        corners = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return I(min(corners), max(corners))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.contains(0):
            raise ZeroDivisionError("interval denominator contains zero")
        return I(1 / self.hi, 1 / self.lo)

    def __truediv__(self, other):
        return self * I.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return I.coerce(other) / self

    def __pow__(self, exponent):
        if type(exponent) is not int:
            raise TypeError("interval powers require an integer exponent")
        if exponent < 0:
            return (self ** (-exponent)).reciprocal()
        if not exponent:
            return I(1)
        endpoints = (self.lo**exponent, self.hi**exponent)
        lower = 0 if exponent % 2 == 0 and self.contains(0) else min(endpoints)
        return I(lower, max(endpoints))


RationalInterval = I


@lru_cache(maxsize=1)
def pi_interval():
    """Enclose mathematical pi using the shared Machin-series owner."""
    return I(*_pi_bounds())


def cos(value):
    """Enclose cosine with midpoint bounds and its global unit Lipschitz bound."""
    value = I.coerce(value)
    lower, upper = certified_cosine_bounds(value.midpoint, terms=32, bits=128)
    return I(max(Q(-1), lower - value.radius), min(Q(1), upper + value.radius))


def sin(value):
    """Enclose sine through cos(pi/2-x), including mathematical pi uncertainty."""
    return cos(pi_interval() / 2 - I.coerce(value))


def _small_atan_ratio(value):
    """Enclose atan(t)/t for exact 0<=t<=1/2, continuously one at zero.

    The first 64 terms end with a negative term. The omitted alternating
    tail is positive and at most t**128/129. Interval arithmetic encloses
    all rounding in the partial sum separately from this analytic tail.
    """
    if value == 0:
        return I(1)
    square = I(value) ** 2
    term = partial = I(1)
    for index in range(1, _ATAN_TERMS):
        term = -term * square
        partial = partial + term / (2 * index + 1)
    remainder = (square**_ATAN_TERMS).hi / (2 * _ATAN_TERMS + 1)
    return partial + I(0, remainder)


@lru_cache(maxsize=4096)
def _atan_point(value):
    if value < 0:
        return -_atan_point(-value)
    if value <= Q(1, 2):
        return I(value) * _small_atan_ratio(value)
    if value <= 1:
        # The transformed argument belongs to [-1/3, 0]. Keeping it exact
        # avoids introducing a branch decision from a rounded denominator.
        reduced = (value - 1) / (value + 1)
        return pi_interval() / 4 + _atan_point(reduced)
    return pi_interval() / 2 - _atan_point(1 / value)


def atan(value):
    """Enclose the principal arctangent using monotonicity and rational series."""
    value = I.coerce(value)
    return I(_atan_point(value.lo).lo, _atan_point(value.hi).hi)


@lru_cache(maxsize=4096)
def _atan_ratio_point(value):
    if value <= Q(1, 2):
        return _small_atan_ratio(value)
    return _atan_point(value) / I(value)


def atan_ratio(value):
    """Enclose the continuous even function atan(t)/t, equal to one at zero.

    The integral representation integral_0^1 1/(1+t*t*s*s) ds proves it is
    decreasing in abs(t). This handles intervals crossing zero without a
    division by an interval containing zero or a cancellation near zero.
    """
    absolute = abs(I.coerce(value))
    return I(_atan_ratio_point(absolute.hi).lo, _atan_ratio_point(absolute.lo).hi)
