"""Exact scalar-EPI domain checks shared by affine graph operators."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA
from ..errors import TNFRValueError
from ..types import Glyph, real_scalar_epi
from ._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors

__all__ = [
    "AFFINE_EPI_GLYPHS",
    "require_real_scalar_epi",
    "validate_affine_epi_graph_input",
]


AFFINE_EPI_GLYPHS = frozenset(
    {Glyph.AL, Glyph.EN, Glyph.RA, Glyph.VAL, Glyph.NUL}
)
"""Glyphs whose scalar stage consumes and writes a real EPI coordinate.

Their unclipped local arithmetic is affine. Boundary clipping and identity
gates can make the complete accepted runtime map piecewise or partial.
"""


def require_real_scalar_epi(value: Any, *, operator: str, label: str) -> float:
    """Return an exact signed scalar EPI embedding or reject the operation.

    Raw reals, live ``BEPIElement`` objects and canonical serialized BEPI
    mappings share this boundary. Rich nonuniform or complex BEPI elements have
    a useful magnitude projection for read-only diagnostics, but that projection
    is not a real affine state and must never be written back by a scalar glyph.
    """

    if isinstance(value, bool):
        raise TNFRValueError(
            f"{operator} {label} requires a finite real scalar EPI, not boolean",
            context={"operator": operator, "failed_condition": "scalar_epi"},
        )
    try:
        scalar = real_scalar_epi(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{operator} {label} requires a raw scalar or uniform-real BEPI",
            context={
                "operator": operator,
                "failed_condition": "scalar_epi",
                "value": repr(value),
            },
        ) from exc
    if scalar is None or not math.isfinite(scalar):
        raise TNFRValueError(
            f"{operator} {label} requires a finite uniform-real BEPI embedding",
            context={
                "operator": operator,
                "failed_condition": "scalar_epi",
                "value": repr(value),
            },
        )
    return scalar


def _raw_node_value(
    node_data: Mapping[str, Any], aliases: tuple[str, ...], default: Any
) -> Any:
    """Read the authoritative alias without applying a lossy projection."""

    return get_attr(
        node_data,  # type: ignore[arg-type]
        aliases,
        default,
        strict=True,
        conv=lambda value: value,
    )


def _resonance_neighbors(graph: Any, node: Any) -> tuple[Any, ...]:
    """Return the same compatible subset consumed by the RA runtime."""

    def phase(candidate: Any) -> Any:
        return _raw_node_value(graph.nodes[candidate], ALIAS_THETA, None)

    try:
        selection = resolve_u3_phase_neighbors(
            graph.graph,
            phase(node),
            graph.neighbors(node),
            phase_getter=phase,
            operator_code="RA",
        )
    except U3PhaseGateError as exc:
        raise TNFRValueError(
            f"Resonance phase gate rejected EPI-domain preflight: {exc}",
            context={
                "operator": "Resonance",
                "failed_condition": exc.failed_condition,
            },
        ) from exc
    return selection.neighbors


def validate_affine_epi_graph_input(
    graph: Any,
    node: Any,
    glyph: Glyph | None,
    *,
    operator: str | None = None,
) -> None:
    """Preflight every EPI value a scalar-writing local glyph would consume.

    The check is deliberately read-only and runs before construction of a
    ``NodeNX`` adapter or any public-operator metadata. Reception consumes every
    graph neighbor; Resonance consumes only its U3-compatible subset.
    """

    if glyph not in AFFINE_EPI_GLYPHS:
        return
    name = operator or glyph.value
    target = _raw_node_value(graph.nodes[node], ALIAS_EPI, 0.0)
    require_real_scalar_epi(target, operator=name, label="target EPI")

    if glyph is Glyph.EN:
        neighbors = tuple(graph.neighbors(node))
    elif glyph is Glyph.RA:
        neighbors = _resonance_neighbors(graph, node)
    else:
        neighbors = ()
    for neighbor in neighbors:
        raw = _raw_node_value(graph.nodes[neighbor], ALIAS_EPI, 0.0)
        require_real_scalar_epi(
            raw,
            operator=name,
            label=f"neighbor EPI for {neighbor!r}",
        )
