"""Numeric helper functions and compensated summation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

__all__ = (
    "clamp",
    "clamp01",
    "within_range",
    "similarity_abs",
    "kahan_sum_nd",
    "angle_diff",
)


def clamp(x: float, a: float, b: float) -> float:
    """Return ``x`` clamped to the ``[a, b]`` interval."""
    return max(a, min(b, x))


def clamp01(x: float) -> float:
    """Clamp ``x`` to the ``[0,1]`` interval."""
    return clamp(float(x), 0.0, 1.0)


def within_range(val: float, lower: float, upper: float, tol: float = 1e-9) -> bool:
    """Return ``True`` if ``val`` lies in ``[lower, upper]`` within ``tol``.

    The comparison uses absolute differences instead of :func:`math.isclose`.
    """

    v = float(val)
    return lower <= v <= upper or abs(v - lower) <= tol or abs(v - upper) <= tol


def _norm01(x: float, lo: float, hi: float) -> float:
    """Normalize ``x`` to the unit interval given bounds.

    ``lo`` and ``hi`` delimit the original value range. When ``hi`` is not
    greater than ``lo`` the function returns ``0.0`` to avoid division by
    zero. The result is clamped to ``[0,1]``.
    """

    if hi <= lo:
        return 0.0
    return clamp01((float(x) - float(lo)) / (float(hi) - float(lo)))


def similarity_abs(a: float, b: float, lo: float, hi: float) -> float:
    """Return absolute similarity of ``a`` and ``b`` over ``[lo, hi]``.

    It computes ``1`` minus the normalized absolute difference between
    ``a`` and ``b``. Values are scaled using :func:`_norm01` so the result
    falls within ``[0,1]``.
    """

    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, hi - lo)


def kahan_sum_nd(
    values: Iterable[Sequence[float]], dims: int
) -> tuple[float, ...]:
    """Return compensated sums of ``values`` with ``dims`` components.

    Each component of the tuples in ``values`` is summed independently using the
    Kahan–Babuška (Neumaier) algorithm to reduce floating point error.
    """
    if dims < 1:
        raise ValueError("dims must be >= 1")
    totals = [0.0] * dims
    comps = [0.0] * dims
    for vs in values:
        for i in range(dims):
            v = vs[i]
            t = totals[i] + v
            if abs(totals[i]) >= abs(v):
                comps[i] += (totals[i] - t) + v
            else:
                comps[i] += (v - t) + totals[i]
            totals[i] = t
    return tuple(float(totals[i] + comps[i]) for i in range(dims))


def angle_diff(a: float, b: float) -> float:
    """Return the minimal difference between two angles in radians."""
    return (float(a) - float(b) + math.pi) % math.tau - math.pi
