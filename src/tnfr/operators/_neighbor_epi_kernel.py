"""Pure scalar kernel shared by local neighbour-EPI operators.

Reception (EN) and Resonance (RA) both use the arithmetic mean of the
runtime neighbour set, independently of transport edge weights, followed by
one scalar blend.  They also choose the semantic EPI kind from the dominant
labelled neighbour, while retaining their distinct historical policy for an
unlabelled dominant magnitude.  Keeping those operations here gives runtime
and realization certificates one numerical definition without invoking
grammar, history, metrics, or graph mutation.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable

__all__ = [
    "dominant_neighbor_epi_kind",
    "neighbor_epi_blend_value",
    "neighbor_epi_proposed_kind",
    "neighbor_epi_represented_affine_row",
    "neighbor_epi_unweighted_mean",
    "reception_proposed_epi_kind",
]


def dominant_neighbor_epi_kind(
    neighbor_value_kinds: Iterable[tuple[float, str]],
    fallback_kind: str,
    *,
    unlabeled_magnitude_dominates: bool = False,
) -> tuple[str, float]:
    """Return the kind attached to the largest strict EPI magnitude.

    Equal magnitudes retain the first neighbour in runtime iteration order.
    Zero-magnitude or empty inputs use ``fallback_kind`` and report zero.
    ``unlabeled_magnitude_dominates`` retains the magnitude attached to an
    empty kind so RA can apply its historical fallback-identity gate; EN keeps
    an established target kind when no neighbour supplies a semantic label.
    """

    best_kind = ""
    best_abs = 0.0
    for value, kind in neighbor_value_kinds:
        magnitude = abs(value)
        if magnitude > best_abs:
            best_abs = magnitude
            best_kind = kind
    if not best_kind:
        return (
            fallback_kind,
            best_abs if unlabeled_magnitude_dominates else 0.0,
        )
    return best_kind, best_abs


def neighbor_epi_proposed_kind(
    current_kind: str,
    neighbor_value_kinds: Iterable[tuple[float, str]],
    proposed_target_epi: float,
    *,
    fallback_kind: str,
    unlabeled_magnitude_dominates: bool = False,
) -> str:
    """Resolve the semantic kind accompanying a neighbour-EPI blend."""

    dominant, best_abs = dominant_neighbor_epi_kind(
        neighbor_value_kinds,
        fallback_kind,
        unlabeled_magnitude_dominates=unlabeled_magnitude_dominates,
    )
    proposed = dominant if best_abs > abs(proposed_target_epi) else current_kind
    return proposed or fallback_kind


def reception_proposed_epi_kind(
    current_kind: str,
    neighbor_value_kinds: Iterable[tuple[float, str]],
    *,
    unclipped_target_epi: float,
    fallback_kind: str = "EN",
) -> str:
    """Resolve EN identity from the blend proposal before boundary projection.

    Reception historically chooses its semantic source before structural
    clipping accepts the numeric EPI value. Soft clipping can move that value
    across a neighbour-magnitude comparison, so every EN runtime and
    realization path must pass the shared *unclipped* blend explicitly.
    """

    return neighbor_epi_proposed_kind(
        current_kind,
        neighbor_value_kinds,
        unclipped_target_epi,
        fallback_kind=fallback_kind,
    )


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
    follows the same affine formula as a raw scalar. Canonical runtime callers
    reject nonuniform or complex BEPI payloads before invoking this scalar
    kernel; their magnitude projection is outside the affine domain.
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
