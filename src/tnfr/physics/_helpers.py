"""Internal shared utilities for TNFR physics module.

Centralises small helper functions used across multiple physics submodules
to eliminate duplication and ensure a single source of truth.

This module is PRIVATE (leading underscore) — it is not exported via
``__init__.py`` and should only be imported by sibling modules inside
``tnfr.physics``.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable

from ..mathematics.unified_numerical import kahan_sum_nd, np

# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
except ImportError:
    ALIAS_THETA = ["phase", "theta"]
    ALIAS_DNFR = ["delta_nfr", "dnfr"]

# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------


def finite_real_scalar(value: Any, name: str) -> float:
    """Return one finite real scalar while rejecting logical values.

    Python and NumPy booleans are integer-like, so an unchecked ``float``
    conversion silently turns state labels into physical zero/one values.  All
    physics readers that require a scalar state channel should use this helper
    before applying channel-specific sign constraints.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def finite_real_series(
    values: Any,
    name: str,
    *,
    nonnegative: bool = False,
    nonempty: bool = False,
) -> np.ndarray:
    """Return a strict finite one-dimensional real series.

    Validation precedes float coercion so logical and textual samples cannot
    silently become physical zero/one values or parsed numbers. A detached
    float64 array is returned after every element has passed the scalar
    contract.
    """

    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a numeric one-dimensional series")
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a numeric one-dimensional series"
        ) from exc
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series")
    if nonempty and raw.size == 0:
        raise ValueError(f"{name} must not be empty")

    normalized: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"{name} must contain only finite real numeric values; "
                f"item {index} is boolean"
            )
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError(
                f"{name} must contain only finite real numeric values; "
                f"item {index} is textual"
            )
        try:
            normalized.append(finite_real_scalar(value, f"{name}[{index}]"))
        except ValueError as exc:
            raise ValueError(
                f"{name} must contain only finite real numeric values; "
                f"item {index} is invalid"
            ) from exc

    result = np.asarray(normalized, dtype=float)
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must contain nonnegative magnitudes")
    return result


# ---------------------------------------------------------------------------
# Phase / angle helpers
# ---------------------------------------------------------------------------


def wrap_angle(angle: float) -> float:
    """Map *angle* to the half-open interval [-π, π)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def get_phase(G: Any, node: Any) -> float:
    """Retrieve phase value φ for *node* (radians in [0, 2π))."""
    node_data = G.nodes[node]
    for alias in ALIAS_THETA:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


def get_dnfr(G: Any, node: Any) -> float:
    """Retrieve ΔNFR value for *node*."""
    node_data = G.nodes[node]
    for alias in ALIAS_DNFR:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


def compensated_sum(values: Iterable[float], *, dtype: type = float) -> float:
    """Accumulate signed field contributions, retaining an extended dtype.

    longdouble aliases float64 on some platforms, including Windows. In that
    case use fsum; otherwise the shared compensated accumulator keeps NumPy
    scalar precision until its final public float conversion.
    """
    if np.finfo(dtype).eps < np.finfo(float).eps:
        return kahan_sum_nd(((value,) for value in values), dims=1)[0]
    return math.fsum(values)


def neighborhood_arrays(
    G: Any, nodes: list[Any], *, dtype: type = np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return neighbor indices, center indices and unique-neighbor counts.

    The local-field contracts average over ``G.neighbors(node)``. This means
    successors on a directed graph, one contribution per parallel neighbor,
    and one contribution from a self-loop. NetworkX's degree counts do not
    have these semantics: they include incoming arcs and count loops twice.
    Build numerators and denominators together so every vectorized read-out
    agrees with the scalar neighborhood definition.
    """
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    neighbor_indices: list[int] = []
    center_indices: list[int] = []
    counts = np.zeros(len(nodes), dtype=dtype)
    for center_index, node in enumerate(nodes):
        for neighbor in G.neighbors(node):
            neighbor_indices.append(node_to_idx[neighbor])
            center_indices.append(center_index)
            counts[center_index] += 1
    return (
        np.asarray(neighbor_indices, dtype=np.intp),
        np.asarray(center_indices, dtype=np.intp),
        counts,
    )


# ---------------------------------------------------------------------------
# Safe division
# ---------------------------------------------------------------------------


def safe_div(
    numerator: np.ndarray,
    denominator: np.ndarray | float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Element-wise division guarded against division by zero.

    Uses the ``a / (b + eps)`` strategy which is simple, differentiable,
    and sufficient for TNFR telemetry computations.
    """
    return numerator / (denominator + eps)


def safe_div_mask(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fallback: float = 0.0,
) -> np.ndarray:
    """Element-wise division using a mask for zero denominators.

    Returns *fallback* where |denominator| < 1e-12.  Useful when the
    eps-offset strategy would bias results.
    """
    result = np.full_like(numerator, fallback, dtype=float)
    mask = np.abs(denominator) > 1e-12
    result[mask] = numerator[mask] / denominator[mask]
    return result
