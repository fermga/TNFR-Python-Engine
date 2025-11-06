"""Numeric helper functions and compensated summation utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = (
    "clamp",
    "clamp01",
    "within_range",
    "similarity_abs",
    "kahan_sum_nd",
    "angle_diff",
    "angle_diff_array",
)


def clamp(x: float, a: float, b: float) -> float:
    """Return ``x`` clamped to the ``[a, b]`` interval."""

    return max(a, min(b, x))


def clamp01(x: float) -> float:
    """Clamp ``x`` to the ``[0,1]`` interval."""

    return clamp(float(x), 0.0, 1.0)


def within_range(val: float, lower: float, upper: float, tol: float = 1e-9) -> bool:
    """Return ``True`` if ``val`` lies in ``[lower, upper]`` within ``tol``."""

    v = float(val)
    return lower <= v <= upper or abs(v - lower) <= tol or abs(v - upper) <= tol


def _norm01(x: float, lo: float, hi: float) -> float:
    """Normalize ``x`` to the unit interval given bounds."""

    if hi <= lo:
        return 0.0
    return clamp01((float(x) - float(lo)) / (float(hi) - float(lo)))


def similarity_abs(a: float, b: float, lo: float, hi: float) -> float:
    """Return absolute similarity of ``a`` and ``b`` over ``[lo, hi]``."""

    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, hi - lo)


def kahan_sum_nd(values: Iterable[Sequence[float]], dims: int) -> tuple[float, ...]:
    """Return compensated sums of ``values`` with ``dims`` components."""

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


def angle_diff_array(
    a: Sequence[float] | "np.ndarray",  # noqa: F821
    b: Sequence[float] | "np.ndarray",  # noqa: F821
    *,
    np: Any,
    out: "np.ndarray | None" = None,  # noqa: F821
    where: "np.ndarray | None" = None,  # noqa: F821
) -> "np.ndarray":  # noqa: F821
    """Vectorised :func:`angle_diff` compatible with NumPy arrays."""

    if np is None:
        raise TypeError("angle_diff_array requires a NumPy module")

    kwargs = {"where": where} if where is not None else {}
    minuend = np.asarray(a, dtype=float)
    subtrahend = np.asarray(b, dtype=float)
    if out is None:
        out = np.empty_like(minuend, dtype=float)
        if where is not None:
            out.fill(0.0)
    else:
        if getattr(out, "shape", None) != minuend.shape:
            raise ValueError("out must match the broadcasted shape of inputs")

    np.subtract(minuend, subtrahend, out=out, **kwargs)
    np.add(out, math.pi, out=out, **kwargs)
    if where is not None:
        mask = np.asarray(where, dtype=bool)
        if mask.shape != out.shape:
            raise ValueError("where mask must match the broadcasted shape of inputs")
        selected = out[mask]
        if selected.size:
            out[mask] = np.remainder(selected, math.tau)
    else:
        np.remainder(out, math.tau, out=out)
    np.subtract(out, math.pi, out=out, **kwargs)
    return out
