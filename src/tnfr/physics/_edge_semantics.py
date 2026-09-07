"""Central edge-channel semantics for TNFR graph read-outs.

``weight`` is the established transport conductance used by the EPI channel.
Structural-potential path geometry can instead declare an independent
``length``.  For compatibility, an edge without ``length`` still uses its
``weight`` as the legacy path length; an edge carrying neither attribute has
unit conductance and unit length.

Keeping this compatibility fallback in one module makes the ambiguity visible
and lets new graphs separate coupling strength from metric distance.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Callable, Mapping

EDGE_CONDUCTANCE_ATTRIBUTE = "weight"
EDGE_LENGTH_ATTRIBUTE = "length"

__all__ = [
    "EDGE_CONDUCTANCE_ATTRIBUTE",
    "EDGE_LENGTH_ATTRIBUTE",
    "effective_edge_length",
    "has_explicit_edge_lengths",
    "has_nonpositive_edge_length",
    "structural_path_weight",
]


def effective_edge_length(attributes: Mapping[str, Any]) -> float:
    """Return one finite nonnegative structural path length.

    Explicit ``length`` wins. ``weight`` is accepted only as the historical
    fallback so existing weighted-potential callers retain their result.
    """
    raw = attributes.get(
        EDGE_LENGTH_ATTRIBUTE,
        attributes.get(EDGE_CONDUCTANCE_ATTRIBUTE, 1.0),
    )
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ValueError("Structural edge length must be a finite nonnegative real")
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "Structural edge length must be a finite nonnegative real"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Structural edge length must be a finite nonnegative real")
    return value


def has_explicit_edge_lengths(graph: Any) -> bool:
    """Whether any edge declares the independent ``length`` channel."""
    return any(
        EDGE_LENGTH_ATTRIBUTE in data
        for _, _, data in graph.edges(data=True)
    )


def has_nonpositive_edge_length(graph: Any) -> bool:
    """Whether the graph contains a zero structural path length."""
    return any(
        effective_edge_length(data) <= 0.0
        for _, _, data in graph.edges(data=True)
    )


def structural_path_weight(graph: Any) -> Callable[[Any, Any, Any], float]:
    """Return a NetworkX weight callback for the structural length channel.

    NetworkX passes a mapping of edge keys to attribute mappings for a
    multigraph. Parallel structural lengths combine by the shortest edge, as
    required by shortest-path geometry.
    """
    multiple = bool(graph.is_multigraph())

    def read(_source: Any, _target: Any, data: Any) -> float:
        if multiple:
            lengths = tuple(
                effective_edge_length(attributes)
                for attributes in data.values()
            )
            return min(lengths) if lengths else 1.0
        return effective_edge_length(data)

    return read
