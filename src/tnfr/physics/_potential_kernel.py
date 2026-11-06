"""Shared distance-power row reduction for structural potential observers."""

import math
from fractions import Fraction

from ..mathematics.unified_numerical import np
from ._helpers import compensated_sum


def potential_row_sum(pairs, alpha, *, dtype=float):
    """Sum pressure/distance**alpha over admitted positive finite distances.

    Ordinary rows retain the selected floating dtype and compensated sum.
    Exceptional canonical inverse-square rows use the exact rational values
    of the represented pressure and distances before final float conversion.
    This prevents intermediate square overflow/underflow or signed cancellation
    from discarding a representable potential. It does not change path sums
    into exact rational shortest paths or certify landmark approximations.
    """
    pairs = tuple((source, distance) for source, distance in pairs if source != 0)
    if not pairs:
        return 0.0
    pressure = np.asarray([pair[0] for pair in pairs], dtype=dtype)
    distances = np.asarray([pair[1] for pair in pairs], dtype=dtype)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        powers = distances**alpha
        contributions = pressure / powers
    unsafe = (
        np.any(~np.isfinite(powers))
        or np.any(powers == 0)
        or np.any(np.abs(powers) < np.finfo(dtype).tiny)
        or np.any(~np.isfinite(contributions))
        or np.any((pressure != 0) & (contributions == 0))
    )
    if not unsafe:
        try:
            result = compensated_sum(contributions, dtype=dtype)
            if math.isfinite(result):
                return result
        except (OverflowError, ValueError):
            pass
    if alpha != 2.0:
        raise ValueError(
            "potential distance-power arithmetic exceeds the selected finite range"
        )
    exact = sum(
        (
            Fraction.from_float(float(source))
            / Fraction.from_float(float(distance)) ** 2
            for source, distance in pairs
        ),
        Fraction(),
    )
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf
