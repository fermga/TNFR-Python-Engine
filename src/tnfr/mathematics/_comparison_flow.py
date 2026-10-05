"""Rational upper bounds for a bounded-dimensional Metzler comparison flow.

For two solutions remaining in the same convex smooth-domain tube, let
``M[i,i] >= sup J[i,i]`` and ``M[i,j] >= sup abs(J[i,j])`` for ``i != j``.
Componentwise absolute separation then satisfies the upper-Dini comparison
inequality ``D+ abs(error) <= M abs(error)``. This kernel encloses the resulting
``exp(h*M) radius``; the caller must establish the tube and Jacobian premises.
An endpoint enclosure error for the center solution may be added afterwards.

Shifting the diagonal makes the exponential series nonnegative. Outward
dyadic arithmetic bounds every term, and an infinity-norm geometric tail
bounds all omitted terms. The negative scalar exponential uses the existing
exact-time Taylor owner. Neither empirical exponentials nor matrix libraries
participate in the bound. The unit-scaled work limits are numerical policies.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from fractions import Fraction as Q

from .._exact_time import exp_unit_bounds
from ._rational_interval import I

__all__ = (
    "comparison_flow_upper",
    "COMPARISON_FLOW_METHOD",
    "MAX_COMPARISON_DIMENSION",
)

COMPARISON_FLOW_METHOD = "metzler_shift_nonnegative_series32_dyadic128_norm_tail_v1"
_TERMS = 32
MAX_COMPARISON_DIMENSION = 24


def _exact(value):
    if isinstance(value, Q):
        return value
    if type(value) is int:
        return Q(value)
    raise TypeError("comparison flow requires exact Fraction or integer values")


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered iterable")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered iterable") from exc


def comparison_flow_upper(matrix, radii, duration) -> tuple[Q, ...]:
    """Bound ``exp(duration*matrix) radii`` componentwise from above.

    The matrix must be square and Metzler (nonnegative off-diagonals), with
    signed diagonal entries. Its dimension is admitted from one to twenty-four;
    this work cap is numerical policy, not a mathematical restriction of the
    comparison theorem. Radii must match that dimension, and radii and
    duration must be nonnegative.
    Values must already be exact rationals or integers; floats and booleans
    are rejected. For ``alpha=max(0,-min(diagonal))`` and ``N=M+alpha*Id``,
    the admitted numerical domain is ``h*max_row_sum(N)<=1`` and
    ``alpha*h<=1``. Unsupported domains raise rather than claiming a bound.

    All returned fractions are nonnegative outward 128-bit dyadic bounds.
    If ``v_k=(h*N)**k*r/k!`` and ``x=h*||N||_inf``, the omitted vector tail
    after degree p is componentwise at most
    ``||v_p||_inf*x/(p+1)/(1-x/(p+2))``. The computed nonnegative upper
    terms also bound ``v_p``, so their rounding cannot invalidate this tail.
    """
    rows = _ordered(matrix, "matrix")
    rows = tuple(_ordered(row, "matrix row") for row in rows)
    dimension = len(rows)
    if not 1 <= dimension <= MAX_COMPARISON_DIMENSION:
        raise ValueError(
            f"comparison matrix dimension must lie between 1 and {MAX_COMPARISON_DIMENSION}"
        )
    if any(len(row) != dimension for row in rows):
        raise ValueError("comparison matrix must be square")
    rows = tuple(tuple(_exact(value) for value in row) for row in rows)
    radius = tuple(_exact(value) for value in _ordered(radii, "radii"))
    if len(radius) != dimension:
        raise ValueError("comparison radii must match the matrix dimension")
    h = _exact(duration)
    if h < 0 or any(value < 0 for value in radius):
        raise ValueError("comparison duration and radii must be nonnegative")
    if any(
        rows[i][j] < 0 for i in range(dimension) for j in range(dimension) if i != j
    ):
        raise ValueError("comparison matrix must have nonnegative off-diagonals")

    alpha = max(Q(0), -min(rows[i][i] for i in range(dimension)))
    shifted = tuple(
        tuple(value + (alpha if i == j else 0) for j, value in enumerate(row))
        for i, row in enumerate(rows)
    )
    norm_step = h * max(sum(row, Q(0)) for row in shifted)
    if norm_step > 1 or alpha * h > 1:
        raise ValueError("comparison flow requires h*norm(N)<=1 and alpha*h<=1")
    if h == 0 or not any(radius):
        return tuple(I(value).hi for value in radius)

    term = tuple(I(value).hi for value in radius)
    partial = term
    scaled = tuple(tuple(h * value for value in row) for row in shifted)
    for order in range(1, _TERMS + 1):
        term = tuple(
            I(sum((value * term[j] for j, value in enumerate(row)), Q(0)) / order).hi
            for row in scaled
        )
        partial = tuple(I(left + right).hi for left, right in zip(partial, term))
    tail = max(term) * norm_step / (_TERMS + 1) / (1 - norm_step / (_TERMS + 2))
    positive_exp_lower, _ = exp_unit_bounds(alpha * h)
    decay_upper = I(1 / positive_exp_lower).hi
    return tuple(I((value + tail) * decay_upper).hi for value in partial)
