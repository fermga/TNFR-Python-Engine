"""Private exact helpers for represented metric vectors.

Metric comparisons in the stability certificates refer to the values actually
represented by binary64, not to an approximate floating-point tolerance. This
module centralizes that conversion and the exact ray normalization used when
composing certificates.

The permissive proportionality helper deliberately preserves the historical
prevalidated-caller contract: empty inputs still fail while unequal lengths are
compared over ``zip``. Callers that accept untrusted vectors must validate
shape, finiteness, non-emptiness and equal length before invoking it.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any


def binary64_fraction_vector(values: Any) -> tuple[Fraction, ...]:
    """Return exact rationals for the materialized binary64 input values."""

    return tuple(Fraction.from_float(float(value)) for value in values)


def finite_binary64_fraction_vector_or_none(
    values: Any,
) -> tuple[Fraction, ...] | None:
    """Return exact finite binary64 values, or ``None`` for invalid input."""

    try:
        materialized = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in materialized):
        return None
    return tuple(Fraction.from_float(value) for value in materialized)


def binary64_vectors_exactly_proportional(left: Any, right: Any) -> bool:
    """Compare two prevalidated binary64 vectors as exact metric rays.

    Conversion exceptions, empty-vector failure and unequal-length ``zip``
    behavior intentionally match the original certificate-local helpers.
    """

    left_exact = binary64_fraction_vector(left)
    right_exact = binary64_fraction_vector(right)
    return all(
        lhs * right_exact[0] == rhs * left_exact[0]
        for lhs, rhs in zip(left_exact, right_exact)
    )


def exact_vectors_proportional_if_aligned(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> bool:
    """Compare validated exact vectors, rejecting empty or misaligned inputs."""

    if not left or not right or len(left) != len(right):
        return False
    return all(
        lhs * right[0] == rhs * left[0]
        for lhs, rhs in zip(left, right)
    )


def normalized_positive_binary64_metric(
    values: Any,
) -> tuple[Fraction, ...] | None:
    """Return the exact ray of a finite positive binary64 metric, if valid."""

    exact = finite_binary64_fraction_vector_or_none(values)
    if exact is None:
        return None
    if not exact or any(value <= 0 for value in exact):
        return None
    total = sum(exact, Fraction(0))
    if total <= 0:
        return None
    return tuple(value / total for value in exact)


def normalized_positive_fraction_metric(
    values: tuple[Fraction, ...] | None,
) -> tuple[Fraction, ...] | None:
    """Normalize a strictly typed positive rational metric, if valid."""

    if values is None or not values or any(
        type(value) is not Fraction or value <= 0 for value in values
    ):
        return None
    total = sum(values, Fraction(0))
    if total <= 0:
        return None
    return tuple(value / total for value in values)
