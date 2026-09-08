"""Exact helpers for represented physical time and transcendental bounds.

TNFR time controls are materialized as binary64 values at the API boundary.
This module retains the exact rational value of that representation, rejects a
positive source that underflows to zero, and normalizes both signed zeros to
``+0.0``.  It also supplies rational logarithm enclosures and a binary64 upper
bound on the exponential.  The helpers are internal so operator schedules and
physics certificates can share one numerical contract without introducing an
operators/physics import cycle.
"""

from __future__ import annotations

from fractions import Fraction
import math
from numbers import Real
from typing import Any, Iterable

__all__ = (
    "atanh_log_bounds",
    "exact_log_bounds",
    "exp_unit_bounds",
    "exp_upper_float",
    "finite_represented_real",
    "fraction_lower_float",
    "fraction_upper_float",
    "fraction_upper_signed_float",
    "materialize_nonnegative_time_sequence",
    "nonnegative_represented_time",
    "represented_fraction",
    "represented_fraction_as_float",
)


# For log, the normalized atanh ratio is at most 1/3, so 34 terms leave a
# tail below 2**-114.  For exp on [0, 1], the same count leaves a tail below
# 36 / (35 * 35!) < 2**-132.
_TRANSCENDENTAL_ENCLOSURE_TERMS = 34


def finite_represented_real(value: Any, label: str) -> tuple[float, Fraction]:
    """Return one finite binary64 scalar and its exact represented rational."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite real scalar, not a boolean")
    try:
        source_nonzero = bool(value != 0)
        represented = float(value)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be representable as binary64") from exc
    if not math.isfinite(represented):
        raise ValueError(f"{label} must be finite")
    if source_nonzero and represented == 0.0:
        raise ValueError(f"{label} is nonzero but underflows to represented zero")
    if represented == 0.0:
        represented = 0.0
    return represented, Fraction.from_float(represented)


def nonnegative_represented_time(
    value: Any, label: str
) -> tuple[float, Fraction]:
    """Return a canonical nonnegative represented time and exact rational."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a nonnegative finite real, not a boolean")
    try:
        source_negative = bool(value < 0)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} must be an ordered real scalar") from exc
    if source_negative:
        raise ValueError(f"{label} must be nonnegative")
    represented, exact = finite_represented_real(value, label)
    if represented < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return represented, exact


def materialize_nonnegative_time_sequence(
    values: Iterable[Any], label: str = "flow_durations"
) -> tuple[tuple[float, ...], tuple[Fraction, ...]]:
    """Materialize a replayable sequence under the shared time contract."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an iterable of nonnegative times")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(
            f"{label} must be an iterable of nonnegative times"
        ) from exc
    materialized = tuple(
        nonnegative_represented_time(value, f"{label}[{index}]")
        for index, value in enumerate(raw)
    )
    return (
        tuple(item[0] for item in materialized),
        tuple(item[1] for item in materialized),
    )


def represented_fraction(value: float, label: str) -> Fraction:
    """Require an exact float payload and return its represented rational."""

    if type(value) is not float:
        raise TypeError(f"{label} must be a binary64 float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must use canonical positive zero")
    return Fraction.from_float(value)


def represented_fraction_as_float(value: Fraction, label: str) -> float:
    """Round an exact rational to its nearest finite binary64 representation."""

    if type(value) is not Fraction:
        raise TypeError(f"{label} must be an exact Fraction")
    try:
        represented = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite represented time") from exc
    if not math.isfinite(represented):
        raise ValueError(f"{label} exceeds finite represented time")
    return 0.0 if represented == 0.0 else represented


def fraction_upper_float(value: Fraction) -> float:
    """Return the smallest binary64 value known not to be below ``value``."""

    if value < 0:
        raise ValueError("an upper-rounded fraction must be nonnegative")
    try:
        rounded = float(value)
    except OverflowError:
        return float("inf")
    if math.isinf(rounded):
        return rounded
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, float("inf"))
    return 0.0 if rounded == 0.0 else rounded


def fraction_upper_signed_float(value: Fraction) -> float:
    """Return the smallest nearby binary64 value not below ``value``."""

    try:
        rounded = float(value)
    except OverflowError:
        return (
            float("inf")
            if value > 0
            else math.nextafter(float("-inf"), float("inf"))
        )
    if math.isinf(rounded):
        return (
            rounded
            if rounded > 0
            else math.nextafter(rounded, float("inf"))
        )
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, float("inf"))
    return 0.0 if rounded == 0.0 else rounded


def fraction_lower_float(value: Fraction) -> float:
    """Return the largest nearby binary64 not above a nonnegative value."""

    if value < 0:
        raise ValueError("a lower-rounded fraction must be nonnegative")
    try:
        rounded = float(value)
    except OverflowError:
        return math.nextafter(float("inf"), 0.0)
    if math.isinf(rounded):
        return math.nextafter(rounded, 0.0)
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, 0.0)
    return 0.0 if rounded == 0.0 else rounded


def atanh_log_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Enclose ``log(value)`` for ``1 <= value <= 2`` rationally."""

    if not Fraction(1) <= value <= Fraction(2):
        raise ValueError("log-series input must lie in [1, 2]")
    ratio = (value - 1) / (value + 1)
    ratio_squared = ratio * ratio
    term = ratio
    partial = Fraction(0)
    term_count = _TRANSCENDENTAL_ENCLOSURE_TERMS
    for index in range(term_count):
        partial += term / (2 * index + 1)
        term *= ratio_squared
    lower = 2 * partial
    if term == 0:
        return lower, lower
    remainder = 2 * term / (
        (2 * term_count + 1) * (1 - ratio_squared)
    )
    return lower, lower + remainder


def exact_log_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Return exact rational lower and upper bounds on a positive logarithm."""

    if value <= 0:
        raise ValueError("logarithm input must be positive")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = (
        Fraction(1 << exponent)
        if exponent >= 0
        else Fraction(1, 1 << -exponent)
    )
    if value < power:
        exponent -= 1
        power /= 2
    elif value >= 2 * power:
        exponent += 1
        power *= 2
    mantissa = value / power
    mantissa_lower, mantissa_upper = atanh_log_bounds(mantissa)
    log_two_lower, log_two_upper = atanh_log_bounds(Fraction(2))
    if exponent >= 0:
        return (
            exponent * log_two_lower + mantissa_lower,
            exponent * log_two_upper + mantissa_upper,
        )
    return (
        exponent * log_two_upper + mantissa_lower,
        exponent * log_two_lower + mantissa_upper,
    )


def exp_unit_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Enclose ``exp(value)`` on ``[0, 1]`` by a rational Taylor sum."""

    if not Fraction(0) <= value <= Fraction(1):
        raise ValueError("exponential-series input must lie in [0, 1]")
    term = Fraction(1)
    partial = Fraction(1)
    term_count = _TRANSCENDENTAL_ENCLOSURE_TERMS
    for index in range(1, term_count + 1):
        term = term * value / index
        partial += term
    first_omitted = term * value / (term_count + 1)
    if first_omitted == 0:
        return partial, partial
    remainder = first_omitted / (1 - value / (term_count + 2))
    return partial, partial + remainder


def exp_upper_float(exponent: Fraction) -> float:
    """Return a binary64 upper bound on ``exp(exponent)``."""

    bounded_exponent = fraction_upper_signed_float(exponent)
    if bounded_exponent == float("inf"):
        return bounded_exponent
    exponent = Fraction.from_float(bounded_exponent)
    if exponent == 0:
        return 1.0
    if exponent > 0:
        if exponent >= 1024:
            return float("inf")
        integer = exponent.numerator // exponent.denominator
        remainder = exponent - integer
        _, e_upper = exp_unit_bounds(Fraction(1))
        _, remainder_upper = exp_unit_bounds(remainder)
        return fraction_upper_float(e_upper**integer * remainder_upper)

    magnitude = -exponent
    if magnitude >= 1075:
        return math.nextafter(0.0, float("inf"))
    integer = magnitude.numerator // magnitude.denominator
    remainder = magnitude - integer
    e_lower, _ = exp_unit_bounds(Fraction(1))
    remainder_lower, _ = exp_unit_bounds(remainder)
    return fraction_upper_float(
        Fraction(1) / (e_lower**integer * remainder_lower)
    )
