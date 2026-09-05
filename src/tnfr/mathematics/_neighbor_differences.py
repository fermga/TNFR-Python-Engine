"""Stable linear neighborhood pressure without averaging a common offset."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import Any, Sequence

from .unified_numerical import np
from ._exact_weighted import exact_weighted_sum_ratio
from ._weight_normalization import normalize_weights


def _exact_mean_difference(center, pairs, coefficient):
    """Round the final pressure after exact range-limited or mixed-sign reduction."""
    numerator, denominator = exact_weighted_sum_ratio(
        [weight for _, weight in pairs], [value for value, _ in pairs],
        center=center, normalize=True,
    )
    coefficient_num, coefficient_den = float(coefficient).as_integer_ratio()
    try:
        result = float(Fraction(numerator * coefficient_num,
                                denominator * coefficient_den))
    except OverflowError as exc:
        raise ValueError("Linear pressure exceeds finite floating-point range") from exc
    return result


def _require_finite_pressure(values):
    """Reject nonfinite assembled pressure before any node receives a write."""
    finite = np.all(np.isfinite(values)) if np is not None else all(map(math.isfinite, values))
    if not finite:
        raise ValueError("Assembled pressure exceeds finite floating-point range")


def mean_neighbor_difference(
    center: float, neighbors: Sequence[float], weights: Sequence[float] | None = None,
    *, coefficient: float = 1.0,
) -> float:
    """Return a weighted mean of differences, with an exact extreme fallback.

    Only positive-weight pairs participate. Subtracting before reduction keeps
    representable small differences under a large common offset. Exact rational
    arithmetic handles mixed signs, intermediate overflow and probability/product
    underflow; it does not require the unweighted component to be representable.
    """
    if coefficient == 0.0:
        return 0.0
    if not math.isfinite(coefficient):
        raise ValueError("Linear pressure requires a finite channel coefficient")
    if weights is None:
        weights = [1.0] * len(neighbors)
    if len(weights) != len(neighbors):
        raise ValueError("Neighbor values and weights must have the same length")
    if any(not math.isfinite(w) or w < 0.0 for w in weights):
        raise ValueError("Pressure requires finite nonnegative effective weights")
    pairs = [(float(value), float(w)) for value, w in zip(neighbors, weights) if w > 0.0]
    if not pairs:
        return 0.0
    if not math.isfinite(center) or any(not math.isfinite(value) for value, _ in pairs):
        raise ValueError("Active linear pressure requires finite nodal values")
    # Mixed signs can cancel rounded products, even when every intermediate
    # value is finite. Compensating already-rounded probabilities alone cannot
    # recover the residual, so retain the original float values and weights.
    if any(value > center for value, _ in pairs) and any(value < center for value, _ in pairs):
        return _exact_mean_difference(center, pairs, coefficient)
    weight_scale = max(w for _, w in pairs)
    scaled_weights = [w / weight_scale for _, w in pairs]
    total = math.fsum(scaled_weights)
    probabilities = [w / total for w in scaled_weights]
    differences = [value - center for value, _ in pairs]
    contributions = [p * d for p, d in zip(probabilities, differences)]
    range_loss = any(
        d != 0.0 and (p < sys.float_info.min or abs(c) < sys.float_info.min)
        for p, d, c in zip(probabilities, differences, contributions)
    )
    if not range_loss and all(math.isfinite(value) for value in contributions):
        try:
            result = coefficient * math.fsum(contributions)
        except OverflowError:
            result = float("inf")
        if math.isfinite(result):
            return result
    return _exact_mean_difference(center, pairs, coefficient)


def edge_mean_differences(
    values: Any, source: Any, target: Any, weights: Any = None,
    *, coefficient: float = 1.0,
) -> Any:
    """Reduce outgoing edges without an all-pair matrix.

    One-sign ordinary rows take O(V+E) work and storage. Mixed-sign and range-
    limited rows use grouped arbitrary-precision rational arithmetic. Arbitrary
    edge order can require sorting selected edges; no row scans the whole graph.
    """
    values = np.asarray(values, dtype=float)
    if coefficient == 0.0:
        return np.zeros(len(values), dtype=float)
    if not math.isfinite(coefficient):
        raise ValueError("Linear pressure requires a finite channel coefficient")
    source = np.asarray(source, dtype=np.intp)
    target = np.asarray(target, dtype=np.intp)
    weights = np.ones(len(source), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != source.shape or target.shape != source.shape:
        raise ValueError("Edge arrays and weights must have the same shape")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Pressure requires finite nonnegative effective weights")
    active = weights > 0.0
    source, target, weights = source[active], target[active], weights[active]
    if not len(source):
        return np.zeros(len(values), dtype=float)
    if not np.all(np.isfinite(values[source])) or not np.all(np.isfinite(values[target])):
        raise ValueError("Active linear pressure requires finite nodal values")
    probability, _, _ = normalize_weights(weights, source=source, node_count=len(values))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        differences = values[target] - values[source]
        contributions = probability * differences
        result = coefficient * np.bincount(source, weights=contributions, minlength=len(values))
    # A probability/product can round to zero although its eventual pressure
    # contribution is representable; recompute only affected rows exactly.
    range_loss = (differences != 0.0) & (
        (probability < sys.float_info.min) | (np.abs(contributions) < sys.float_info.min)
    )
    bad_rows = np.unique(np.concatenate((
        source[~np.isfinite(contributions) | range_loss],
        np.flatnonzero(~np.isfinite(result)),
    )))
    positive = np.zeros(len(values), dtype=bool)
    negative = np.zeros(len(values), dtype=bool)
    positive[source[differences > 0.0]] = True
    negative[source[differences < 0.0]] = True
    exact_rows = positive & negative
    exact_rows[bad_rows] = True
    selected = np.flatnonzero(exact_rows[source])
    if len(selected):
        selected_source = source[selected]
        # Canonical graph edge arrays are already grouped. Direct callers may
        # supply shuffled/interleaved rows, which are grouped once here.
        if np.any(selected_source[1:] < selected_source[:-1]):
            selected = selected[np.argsort(selected_source, kind="stable")]
            selected_source = source[selected]
        boundaries = np.flatnonzero(selected_source[1:] != selected_source[:-1]) + 1
        for edges in np.split(selected, boundaries):
            row = source[edges[0]]
            pairs = list(zip(values[target[edges]].tolist(), weights[edges].tolist()))
            result[row] = _exact_mean_difference(float(values[row]), pairs, coefficient)
    return result
