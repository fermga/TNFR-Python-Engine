"""Dependency-light numeric contracts for noncanonical diagnostic scores."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any

from ..errors import TNFRValueError


def finite_real(value: Any, *, label: str) -> float:
    """Return one finite, non-boolean real diagnostic input."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} must be a finite real scalar"
        ) from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    return result


def nonnegative_magnitude(value: Any, *, label: str) -> float:
    """Return one finite unbounded magnitude in [0, +inf)."""

    result = finite_real(value, label=label)
    if result < 0.0:
        raise TNFRValueError(f"{label} must be nonnegative")
    return result


def sum_nonnegative_magnitudes(
    values: Iterable[Any], *, label: str
) -> float:
    """Return a finite compensated sum of unbounded magnitudes."""

    materialized = tuple(
        nonnegative_magnitude(value, label=f"{label} component")
        for value in values
    )
    try:
        total = math.fsum(sorted(materialized))
    except OverflowError as exc:
        raise TNFRValueError(f"{label} must remain finite") from exc
    return nonnegative_magnitude(total, label=label)


def unit_score(value: Any, *, label: str) -> float:
    """Return one finite bounded diagnostic score in [0, 1]."""

    result = finite_real(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise TNFRValueError(f"{label} must lie in [0, 1]")
    return result


def mean_unit_score(values: Iterable[Any], *, label: str) -> float:
    """Return the compensated mean of a nonempty collection of unit scores."""

    materialized = tuple(
        unit_score(value, label=f"{label} component") for value in values
    )
    if not materialized:
        raise TNFRValueError(f"{label} requires at least one component")
    return unit_score(
        math.fsum(materialized) / len(materialized),
        label=label,
    )


__all__ = [
    "finite_real",
    "mean_unit_score",
    "nonnegative_magnitude",
    "sum_nonnegative_magnitudes",
    "unit_score",
]
