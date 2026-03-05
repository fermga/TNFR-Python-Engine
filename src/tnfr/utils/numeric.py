"""Numeric helper functions and compensated summation utilities.

DEPRECATED: Use src/tnfr/mathematics/unified_numerical.py instead.
"""

from __future__ import annotations

import math
# import warnings
from collections.abc import Iterable, Sequence
from typing import Any

from ..errors import TNFRValueError

# Lazy import to avoid circular dependency
# from ..mathematics.unified_numerical import ...

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
    from ..mathematics.unified_numerical import clamp_value
    return float(clamp_value(x, a, b))


def clamp01(x: float) -> float:
    """Clamp ``x`` to the ``[0,1]`` interval."""
    from ..mathematics.unified_numerical import clamp_value
    return float(clamp_value(x, 0.0, 1.0))


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
    from ..mathematics.unified_numerical import kahan_sum_nd as unified_kahan_sum_nd
    return unified_kahan_sum_nd(values, dims)


def angle_diff(a: float, b: float) -> float:
    """Return the minimal difference between two angles in radians."""
    from ..mathematics.unified_numerical import compute_phase_difference
    return float(compute_phase_difference(a, b))


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
            raise TNFRValueError(
                "out must match the broadcasted shape of inputs",
                context={"out_shape": getattr(out, "shape", None), "expected": minuend.shape},
                suggestion="Ensure output array has correct shape."
            )

    np.subtract(minuend, subtrahend, out=out, **kwargs)
    np.add(out, math.pi, out=out, **kwargs)
    if where is not None:
        mask = np.asarray(where, dtype=bool)
        if mask.shape != out.shape:
            raise TNFRValueError(
                "where mask must match the broadcasted shape of inputs",
                context={"mask_shape": mask.shape, "expected": out.shape},
                suggestion="Ensure mask array has correct shape."
            )
        selected = out[mask]
        if selected.size:
            out[mask] = np.remainder(selected, math.tau)
    else:
        np.remainder(out, math.tau, out=out)
    np.subtract(out, math.pi, out=out, **kwargs)
    return out
