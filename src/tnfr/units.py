"""Structural unit conversion helpers.

The TNFR engine tracks structural dynamics using the ``Hz_str`` unit.  A
single configurable scale factor ``k`` bridges this canonical structural
frequency with the conventional ``Hz`` base unit.  The factor is resolved
according to the following invariants:

* ``k`` is always read from the graph configuration via :func:`get_param` so
  per-graph overrides take precedence over the package defaults.
* The fallback value comes from :data:`tnfr.constants.DEFAULTS`. The default
  factor one is a numerical convention, not an independently calibrated
  physical clock or a derivation of equality between the two units.
* ``k`` must remain strictly positive.  Invalid overrides raise
  :class:`ValueError` to prevent incoherent conversions.

All helpers defined here operate purely on ``GraphLike`` instances and only
reuse the shared represented-real boundary and configuration access, keeping
conversion logic explicit and side-effect free.
"""

from __future__ import annotations

import math
from typing import Final

from ._exact_time import finite_represented_real
from .constants import get_param
from .errors import TNFRValueError
from .types import GraphLike

__all__ = ("get_hz_bridge", "hz_str_to_hz", "hz_to_hz_str")

HZ_STR_BRIDGE_KEY: Final[str] = "HZ_STR_BRIDGE"


def _finite_real(raw: object, name: str) -> float:
    """Preserve the unit error type around shared represented-real validation."""
    try:
        return finite_represented_real(raw, name)[0]
    except ValueError as exc:
        raise TNFRValueError(str(exc)) from exc


def _coerce_bridge_factor(raw: object) -> float:
    """Return a finite strictly positive bridge factor."""
    factor = _finite_real(raw, HZ_STR_BRIDGE_KEY)

    if factor <= 0.0:
        raise TNFRValueError(
            "HZ_STR_BRIDGE must be strictly positive",
            context={"HZ_STR_BRIDGE": factor},
        )

    return factor


def get_hz_bridge(G: GraphLike) -> float:
    """Return the ``Hz_str``→``Hz`` bridge factor for ``G``.

    The helper always consults ``G.graph`` via :func:`get_param` so per-graph
    overrides remain authoritative.
    """

    return _coerce_bridge_factor(get_param(G, HZ_STR_BRIDGE_KEY))


def hz_str_to_hz(value: float, G: GraphLike) -> float:
    """Convert a finite signed value, preserving nonzero representability.

    This scalar conversion does not admit a positive-capacity physical model.
    That model must separately validate capacity and its calibrated clock.
    """

    source = _finite_real(value, "value")
    return _checked_conversion(source, source * get_hz_bridge(G))


def hz_to_hz_str(value: float, G: GraphLike) -> float:
    """Convert ``value`` expressed in ``Hz`` into ``Hz_str`` using ``G``."""

    source = _finite_real(value, "value")
    return _checked_conversion(source, source / get_hz_bridge(G))


def _checked_conversion(source: float, converted: float) -> float:
    if not math.isfinite(converted) or (source != 0 and converted == 0):
        raise TNFRValueError("unit conversion overflowed or erased a nonzero value")
    return converted
