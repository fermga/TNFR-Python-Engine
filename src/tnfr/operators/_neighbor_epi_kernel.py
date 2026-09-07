"""Pure scalar kernel shared by local neighbour-EPI operators.

Reception (EN) and Resonance (RA) both use the arithmetic mean of the
runtime neighbour set, independently of transport edge weights, followed by
one scalar blend.  Keeping those floating-point operations here gives the
runtime and the realization certificates one numerical definition without
invoking grammar, history, metrics, or graph mutation.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable

__all__ = [
    "neighbor_epi_blend_value",
    "neighbor_epi_represented_affine_row",
    "neighbor_epi_unweighted_mean",
]


def neighbor_epi_unweighted_mean(values: Iterable[Any]) -> float:
    """Return the unweighted binary64 mean used by EN and RA.

    The caller owns the nonempty-neighbour precondition.  ``fmean`` performs
    the same conversion to floating point for graph-backed and object-backed
    nodes, so both runtime paths have one numerical definition.
    """

    return float(fmean(values))


def neighbor_epi_blend_value(
    current_epi: Any,
    neighbor_mean: float,
    mix_factor: float,
) -> float:
    """Evaluate and scalarize the unclipped local neighbour-EPI blend.

    ``current_epi`` may be the engine's BEPI object.  Arithmetic is performed
    in that representation before ``float`` applies its scalar projection.
    The uniform real-scalar BEPI embedding retains its sign and therefore
    follows the same affine formula as a raw scalar.  General nonuniform or
    complex BEPI payloads can follow a different scalar projection.
    """

    return float(
        (1.0 - mix_factor) * current_epi + mix_factor * neighbor_mean
    )


def neighbor_epi_represented_affine_row(
    dimension: int,
    target_index: int,
    neighbor_indices: Iterable[int],
    mix_factor: float,
) -> tuple[float, ...]:
    """Build the binary64 coefficient row for the shared blend formula.

    The row is a declared real-affine representation of the two-stage
    binary64 mean/blend calculation.  Floating-point evaluation of the runtime
    kernel is not asserted to equal a matrix product for every input.
    """

    neighbors = tuple(neighbor_indices)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not 0 <= target_index < dimension:
        raise ValueError("target_index is outside the matrix dimension")
    if not neighbors:
        raise ValueError("a local neighbour-EPI blend requires at least one neighbor")
    if any(not 0 <= index < dimension for index in neighbors):
        raise ValueError("neighbor index is outside the matrix dimension")
    if len(set(neighbors)) != len(neighbors):
        raise ValueError("runtime neighbor indices must be unique")

    row = [0.0] * dimension
    row[target_index] = 1.0 - mix_factor
    neighbor_coefficient = mix_factor / len(neighbors)
    for index in neighbors:
        row[index] += neighbor_coefficient
    return tuple(row)
