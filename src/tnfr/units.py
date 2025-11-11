"""Structural unit conversion helpers.

The TNFR engine tracks structural dynamics using the ``Hz_str`` unit.  A
single configurable scale factor ``k`` bridges this canonical structural
frequency with the conventional ``Hz`` base unit.  The factor is resolved
according to the following invariants:

* ``k`` is always read from the graph configuration via :func:`get_param` so
  per-graph overrides take precedence over the package defaults.
* The fallback value comes from :data:`tnfr.constants.DEFAULTS`, ensuring the
  canonical 1 Hz_str↔Hz relationship is preserved when callers do not provide
  explicit overrides.
* ``k`` must remain strictly positive.  Invalid overrides raise
  :class:`ValueError` to prevent incoherent conversions.

All helpers defined here operate purely on ``GraphLike`` instances and only
depend on :mod:`tnfr.constants` for configuration access, keeping the
conversion logic transparent and side-effect free.
"""

from __future__ import annotations

from typing import Final

from .constants import get_param
from .types import GraphLike

__all__ = ("get_hz_bridge", "hz_str_to_hz", "hz_to_hz_str")

HZ_STR_BRIDGE_KEY: Final[str] = "HZ_STR_BRIDGE"


def _coerce_bridge_factor(raw: object) -> float:
    """Return ``raw`` coerced to a strictly positive floating point factor."""

    try:
        factor = float(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TypeError("HZ_STR_BRIDGE must be a real number convertible to float") from exc

    if factor <= 0.0:
        raise ValueError("HZ_STR_BRIDGE must be strictly positive")

    return factor


def get_hz_bridge(G: GraphLike) -> float:
    """Return the ``Hz_str``→``Hz`` bridge factor for ``G``.

    The helper always consults ``G.graph`` via :func:`get_param` so per-graph
    overrides remain authoritative.
    """

    return _coerce_bridge_factor(get_param(G, HZ_STR_BRIDGE_KEY))


def hz_str_to_hz(value: float, G: GraphLike) -> float:
    """Convert ``value`` expressed in ``Hz_str`` into ``Hz`` using ``G``."""

    return float(value) * get_hz_bridge(G)


def hz_to_hz_str(value: float, G: GraphLike) -> float:
    """Convert ``value`` expressed in ``Hz`` into ``Hz_str`` using ``G``."""

    return float(value) / get_hz_bridge(G)
