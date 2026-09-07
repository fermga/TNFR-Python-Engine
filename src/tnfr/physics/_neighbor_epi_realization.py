"""Shared numerical helpers for local neighbour-EPI realization bridges."""

from __future__ import annotations

from fractions import Fraction
import math
from numbers import Real
from typing import Any

from ..alias import get_attr
from ..constants import DEFAULTS
from ..constants.aliases import ALIAS_EPI
from ..mathematics.unified_numerical import np
from ..operators._neighbor_epi_kernel import neighbor_epi_represented_affine_row
from ..types import ZERO_BEPI_STORAGE, real_scalar_epi
from ._helpers import finite_real_scalar

__all__ = [
    "exact_binary64_matrix",
    "exact_ideal_neighbor_blend_map",
    "exact_matrix_vector",
    "fraction_float_or_infinity",
    "optional_flow_duration",
    "readonly_float_array",
    "represented_neighbor_blend_map",
    "require_explicit_epi",
    "resolve_epi_bounds",
    "uses_scalar_epi_embedding",
    "validate_certificate_tolerance",
]


def readonly_float_array(value: Any) -> Any:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def exact_matrix_vector(
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (coefficient * entry for coefficient, entry in zip(row, vector)),
            Fraction(0),
        )
        for row in matrix
    )


def exact_ideal_neighbor_blend_map(
    dimension: int,
    target_index: int,
    neighbor_indices: tuple[int, ...],
    mix: Fraction,
) -> tuple[tuple[Fraction, ...], ...]:
    matrix = [
        [
            Fraction(1) if row == column else Fraction(0)
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    matrix[target_index] = [Fraction(0) for _ in range(dimension)]
    matrix[target_index][target_index] = Fraction(1) - mix
    neighbor_coefficient = mix / len(neighbor_indices)
    for index in neighbor_indices:
        matrix[target_index][index] += neighbor_coefficient
    return tuple(tuple(row) for row in matrix)


def represented_neighbor_blend_map(
    dimension: int,
    target_index: int,
    neighbor_indices: tuple[int, ...],
    mix: float,
) -> Any:
    matrix = np.eye(dimension, dtype=float)
    matrix[target_index] = np.asarray(
        neighbor_epi_represented_affine_row(
            dimension, target_index, neighbor_indices, mix
        ),
        dtype=float,
    )
    return matrix


def exact_binary64_matrix(matrix: Any) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in np.asarray(matrix, dtype=float)
    )


def fraction_float_or_infinity(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return float("-inf") if value < 0 else float("inf")


def resolve_epi_bounds(G: Any) -> tuple[float, float, str]:
    try:
        lower = float(G.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
        upper = float(G.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("EPI bounds must be finite real scalars") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("EPI bounds must be finite and ordered")
    mode = str(G.graph.get("CLIP_MODE", "hard"))
    if mode not in ("hard", "soft"):
        mode = "hard"
    return lower, upper, mode


def validate_certificate_tolerance(value: Any) -> float:
    tolerance = finite_real_scalar(value, "tolerance")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie in the open interval (0, 1)")
    return tolerance


def optional_flow_duration(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("recovery_flow_duration must be a finite nonnegative real")
    try:
        source_is_negative = bool(value < 0)
        source_is_nonzero = bool(value != 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "recovery_flow_duration must be a finite nonnegative real"
        ) from exc
    if source_is_negative:
        raise ValueError("recovery_flow_duration must be nonnegative")
    duration = finite_real_scalar(value, "recovery_flow_duration")
    if source_is_nonzero and duration == 0.0:
        raise ValueError(
            "recovery_flow_duration is below nonzero floating-point range"
        )
    return duration


def require_explicit_epi(
    G: Any, nodes: tuple[Any, ...], *, operator: str
) -> None:
    missing = tuple(
        node
        for node in nodes
        if not any(alias in G.nodes[node] for alias in ALIAS_EPI)
    )
    if missing:
        raise ValueError(f"{operator} realization requires explicit EPI on every node")


def uses_scalar_epi_embedding(G: Any, nodes: tuple[Any, ...], state: Any) -> bool:
    """Whether every EPI payload is the real scalar embedding read by flow."""

    for index, node in enumerate(nodes):
        raw = get_attr(
            G.nodes[node],
            ALIAS_EPI,
            ZERO_BEPI_STORAGE,
            strict=True,
            conv=lambda value: value,
        )
        scalar = real_scalar_epi(raw)
        if scalar is None or scalar != float(state[index]):
            return False
    return True
