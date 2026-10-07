"""Numeric helper functions and compensated summation utilities.

The scalar wrappers here (``clamp``, ``clamp01``, ``within_range``,
``kahan_sum_nd``, …) are thin compatibility shims that delegate to the canonical
:mod:`tnfr.mathematics.unified_numerical`. The phase-difference helpers also
delegate to its signed circular difference and are re-exported publicly via
:mod:`tnfr.utils` — prefer ``from tnfr.utils import angle_diff``.
"""

from __future__ import annotations

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
    """Vectorised :func:`angle_diff`, retaining its signed atan2 branch.

    Only entries selected by ``where`` are evaluated. Output and mask domains
    are checked before writing, and unselected output entries are preserved.
    Selected raw elements use the shared phase reader's admission before
    binary64 conversion; an invalid selected element leaves ``out`` unchanged.
    Narrow output types must preserve nonzero results; conversion loss also
    rejects before any selected output entry is written.
    """
    from ..mathematics.unified_numerical import compute_phase_difference

    if np is None:
        raise TypeError("angle_diff_array requires a NumPy module")

    # Preserve raw elements until the shared reader admits the selected pairs.
    # Typed arrays keep their fast path; object conversion prevents a mixed
    # Python sequence from coercing Boolean, text or complex phase values.
    operands = (
        value if isinstance(value, np.ndarray) else np.asarray(value, dtype=object)
        for value in (a, b)
    )
    minuend, subtrahend = np.broadcast_arrays(*operands)
    if out is None:
        out = np.empty_like(minuend, dtype=float)
        if where is not None:
            out.fill(0.0)
    else:
        if getattr(out, "shape", None) != minuend.shape:
            raise TNFRValueError(
                "out must match the broadcasted shape of inputs",
                context={
                    "out_shape": getattr(out, "shape", None),
                    "expected": minuend.shape,
                },
                suggestion="Ensure output array has correct shape.",
            )

    if not np.can_cast(np.dtype(float), out.dtype, casting="same_kind"):
        raise TNFRValueError("out must accept floating-point phase differences")
    if where is not None:
        mask = np.asarray(where, dtype=bool)
        if mask.shape != out.shape:
            raise TNFRValueError(
                "where mask must match the broadcasted shape of inputs",
                context={"mask_shape": mask.shape, "expected": out.shape},
                suggestion="Ensure mask array has correct shape.",
            )
        selected = compute_phase_difference(minuend[mask], subtrahend[mask])
    else:
        mask = None
        selected = compute_phase_difference(minuend, subtrahend)
    selected = np.asarray(selected, dtype=float)
    with np.errstate(under="ignore"):
        represented = np.asarray(selected, dtype=out.dtype)
    if np.any((selected != 0.0) & (represented == 0.0)):
        raise TNFRValueError("Phase differences underflow in the output dtype")
    if mask is None:
        np.copyto(out, represented)
    else:
        out[mask] = represented
    return out
