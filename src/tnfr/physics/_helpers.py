"""Internal shared utilities for TNFR physics module.

Centralises small helper functions used across multiple physics submodules
to eliminate duplication and ensure a single source of truth.

This module is PRIVATE (leading underscore) — it is not exported via
``__init__.py`` and should only be imported by sibling modules inside
``tnfr.physics``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..mathematics.unified_numerical import kahan_sum_nd, np

# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
except ImportError:
    ALIAS_THETA = ["phase", "theta"]
    ALIAS_DNFR = ["delta_nfr", "dnfr"]

# ---------------------------------------------------------------------------
# Phase / angle helpers
# ---------------------------------------------------------------------------


def wrap_angle(angle: float) -> float:
    """Map *angle* to the interval [-π, π]."""
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
