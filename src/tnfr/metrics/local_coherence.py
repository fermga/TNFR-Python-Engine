"""Radius-local canonical and lightweight coherence helpers.

The canonical radius-local read-out includes the centre node and every node in
the graph ball, then applies the same constitutive kernel as global C(t):

    C_local = 1 / (1 + mean(|DeltaNFR|) + mean(|dEPI/dt|))

The historical immediate-neighbour helper remains available for compatibility.
"""

from __future__ import annotations

import math
from operator import index as integer_index
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DEPI, ALIAS_DNFR


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _radius_nodes(G: Any, node: Any, radius: int) -> tuple[Any, ...]:
    seen = {node}
    ordered = [node]
    frontier = [node]
    for _depth in range(radius):
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


def _mean_absolute(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    result = scale * (
        math.fsum(abs(value) / scale for value in values) / len(values)
    )
    if not math.isfinite(result):
        raise ValueError("local mean magnitude exceeds finite range")
    return result


def compute_radius_structural_coherence(
    G: Any,
    node: Any,
    radius: int = 1,
) -> float:
    """Compute canonical structural C(t) on the radius-``radius`` graph ball.

    The centre node is always included. A radius-zero or isolated-node read-out
    therefore evaluates the node itself instead of inventing a zero or perfect
    neighborhood value.
    """

    if isinstance(radius, bool):
        raise TypeError("radius must be a nonnegative integer")
    try:
        resolved_radius = integer_index(radius)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("radius must be a nonnegative integer") from exc
    if resolved_radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if node not in G:
        raise KeyError(node)

    nodes = _radius_nodes(G, node, resolved_radius)
    pressures = tuple(
        _finite_real(
            get_attr(G.nodes[candidate], ALIAS_DNFR, 0.0),
            label=f"DeltaNFR state for {candidate!r}",
        )
        for candidate in nodes
    )
    rates = tuple(
        _finite_real(
            get_attr(G.nodes[candidate], ALIAS_DEPI, 0.0),
            label=f"dEPI state for {candidate!r}",
        )
        for candidate in nodes
    )
    from .common import structural_coherence

    return float(
        structural_coherence(_mean_absolute(pressures), _mean_absolute(rates))
    )


def compute_local_coherence_fallback(G: Any, node: Any) -> float:
    """Compute the historical immediate-neighbour structural proxy.

    This compatibility helper excludes the centre node and returns ``0.0`` for
    an isolate. New radius-aware operator telemetry should use
    :func:`compute_radius_structural_coherence`.
    """

    neighbors = list(G.neighbors(node))
    if not neighbors:
        return 0.0

    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    dnfr_vals = [
        abs(_as_float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0))) for n in neighbors
    ]
    depi_vals = [
        abs(_as_float(get_attr(G.nodes[n], ALIAS_DEPI, 0.0))) for n in neighbors
    ]

    dnfr_mean = sum(dnfr_vals) / len(dnfr_vals) if dnfr_vals else 0.0
    depi_mean = sum(depi_vals) / len(depi_vals) if depi_vals else 0.0
    from .common import structural_coherence

    return structural_coherence(dnfr_mean, depi_mean)


__all__ = [
    "compute_local_coherence_fallback",
    "compute_radius_structural_coherence",
]
