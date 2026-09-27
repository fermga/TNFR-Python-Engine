"""Radius-local canonical and lightweight coherence helpers.

The canonical radius-local read-out includes the centre node and every node in
the graph ball, then applies the same constitutive kernel as global C(t):

    C_local = 1 / (1 + mean(|DeltaNFR|) + mean(|dEPI/dt|))

The historical immediate-neighbour helper remains available for compatibility.
"""

from __future__ import annotations

from operator import index as integer_index
from typing import Any

from .common import _coherence_on_nodes


def _radius_nodes(G: Any, node: Any, radius: int) -> tuple[Any, ...]:
    """Return the validated center-inclusive outgoing graph ball once per node."""
    if isinstance(radius, bool) or type(radius).__name__ == "bool_":
        raise TypeError("radius must be a nonnegative integer")
    try:
        resolved_radius = integer_index(radius)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("radius must be a nonnegative integer") from exc
    if resolved_radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if node not in G:
        raise KeyError(node)

    seen = {node}
    ordered = [node]
    frontier = [node]
    for _depth in range(resolved_radius):
        next_frontier: list[Any] = []
        for current in frontier:
            for neighbor in G.neighbors(current):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                ordered.append(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(ordered)


def compute_radius_structural_coherence(
    G: Any,
    node: Any,
    radius: int = 1,
) -> float:
    """Compute canonical structural C(t) on the radius-``radius`` graph ball.

    The centre node is always included. A radius-zero or isolated-node read-out
    therefore evaluates the node itself instead of inventing a zero or perfect
    neighborhood value. Stored aliases use the global observation's strict
    scalar admission and stable mean reduction; malformed evidence cannot
    fall through to a later alias or become zero.
    """

    nodes = _radius_nodes(G, node, radius)
    return _coherence_on_nodes(G, nodes)[0]


def compute_local_coherence_fallback(G: Any, node: Any) -> float:
    """Compute the historical immediate-neighbour structural proxy.

    This compatibility helper excludes the centre node and returns ``0.0`` for
    an isolate. New radius-aware operator telemetry should use
    :func:`compute_radius_structural_coherence`. Only the historical support
    choice is retained: consumed values share strict global scalar admission.
    """

    return _coherence_on_nodes(G, tuple(G.neighbors(node)))[0]


__all__ = [
    "compute_local_coherence_fallback",
    "compute_radius_structural_coherence",
]
