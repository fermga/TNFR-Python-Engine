"""Exact signed contractions of locally admitted shared interval rows.

This private kernel consumes normalized I values, not physical primitives,
external reports or an alternate interval representation. Consumers must
complete source admission and bound their row/table sizes before encoding,
caching or arithmetic. Encoded tuples are immutable internal data; their
presence alone authenticates neither a model nor any supplied observation.

All integer products and sums are exact. Only the final result uses the
existing outward I projection. Work and integer storage are linear in row
length; integer bit complexity depends on admitted coefficient magnitudes.
"""

from __future__ import annotations

from fractions import Fraction as Q

from ._rational_interval import INTERVAL_BITS, I

_GRID = 1 << INTERVAL_BITS
_PRODUCT_GRID = 4 * _GRID**2


def _encode_admitted_row(values):
    """Encode owned I coefficients as (center, radius, absolute bound) integers.

    For endpoints L/S,U/S, store C=L+U, R=U-L and M=abs(C)+R, all on
    the common 2*S grid. No float conversion or rounding occurs here. The
    representation checks below protect that grid; they do not replace
    the caller's prior admission of authoritative primitive values.
    """
    encoded = []
    for value in values:
        if not isinstance(value, I):
            raise TypeError("dyadic encoding requires locally admitted I values")
        if not isinstance(value.lo, Q) or not isinstance(value.hi, Q):
            raise TypeError("dyadic encoding requires exact shared interval endpoints")
        if value.lo > value.hi:
            raise ValueError("dyadic encoding requires ordered interval endpoints")
        if _GRID % value.lo.denominator or _GRID % value.hi.denominator:
            raise ValueError("dyadic encoding requires the shared interval grid")
        lower = value.lo.numerator * (_GRID // value.lo.denominator)
        upper = value.hi.numerator * (_GRID // value.hi.denominator)
        center, radius = lower + upper, upper - lower
        encoded.append((center, radius, abs(center) + radius))
    return tuple(encoded)


def _dot_encoded_rows(left, right, *, numerator=1, denominator=1):
    """Enclose a rationally scaled dot product of two owned encoded rows.

    Rows must originate from _encode_admitted_row or an exact internal
    transformation preserving its 2*S-grid convention. They are not a
    public admission format. Length and rational-scale checks precede all
    arithmetic. Empty equal rows represent zero; denominator is positive.

    For each pair, the product center is C_l*C_r and an error radius is
    abs(C_l)*R_r + R_l*(abs(C_r)+R_r). Keeping signed centers until the end
    retains cancellation while separately bounding all input uncertainty.
    Shared dependencies between coefficients are conservatively enclosed,
    never presumed independent for cancellation of their errors.
    """
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError("dyadic dot scale requires ordinary integer components")
    if denominator <= 0:
        raise ValueError("dyadic dot denominator must be positive")
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        raise TypeError("dyadic dot requires immutable encoded rows")
    if len(left) != len(right):
        raise ValueError("dyadic dot rows must have the same length")
    center = radius = 0
    for (value, error, _), (weight, width, magnitude) in zip(left, right):
        center += value * weight
        radius += abs(value) * width + error * magnitude
    lower, upper = numerator * (center - radius), numerator * (center + radius)
    if numerator < 0:
        lower, upper = upper, lower
    scale = _PRODUCT_GRID * denominator
    return I(Q(lower, scale), Q(upper, scale))
