"""Exact dyadic weighted reductions shared by pressure and transport."""

from __future__ import annotations


def exact_weighted_sum_ratio(weights, values, *, center=0.0, normalize=False):
    """Return an exact integer ratio for a weighted sum of float differences.

    Compute ``sum(w * (value-center))``, or its weighted mean when
    ``normalize=True``. Callers validate finite values and positive effective
    weights before this exceptional arithmetic path. The result is never
    rounded here; a caller may apply another coefficient before conversion.

    Every binary float denominator is a power of two, so the maximum
    denominator is their common multiple. Integer alignment, multiplication
    and summation preserve every bit without intermediate Fraction reductions.
    Empty input represents zero. Normalized nonempty input needs positive
    total weight, as ensured by the pressure caller.
    """
    value_ratios = [float(value).as_integer_ratio() for value in values]
    weight_ratios = [float(weight).as_integer_ratio() for weight in weights]
    if len(value_ratios) != len(weight_ratios):
        raise ValueError("Weighted values and weights must have the same length")
    if not value_ratios:
        return 0, 1
    center_num, center_den = float(center).as_integer_ratio()
    value_den = max(center_den, *(den for _, den in value_ratios))
    weight_den = max(den for _, den in weight_ratios)
    center_num *= value_den // center_den
    integer_weights = [num * (weight_den // den) for num, den in weight_ratios]
    numerator = sum(
        weight * (num * (value_den // den) - center_num)
        for (num, den), weight in zip(value_ratios, integer_weights)
    )
    denominator = value_den * (sum(integer_weights) if normalize else weight_den)
    return numerator, denominator
