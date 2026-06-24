"""Internal shared utilities for TNFR physics module.

Centralises small helper functions used across multiple physics submodules
to eliminate duplication and ensure a single source of truth.

This module is PRIVATE (leading underscore) — it is not exported via
``__init__.py`` and should only be imported by sibling modules inside
``tnfr.physics``.
"""

from __future__ import annotations

import math
from typing import Any

from ..mathematics.unified_numerical import np

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
