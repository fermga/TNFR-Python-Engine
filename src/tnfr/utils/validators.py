"""Validation utilities for TNFR graph structures."""

from __future__ import annotations

import numbers
import sys

from collections.abc import Mapping
from typing import Callable, Sequence

from .._compat import TypeAlias

from ..alias import get_attr
from ..config.constants import GLYPHS_CANONICAL_SET
from ..constants import get_aliases, get_param
from ..helpers.numeric import within_range
from ..types import (
    EPIValue,
    NodeId,
    StructuralFrequency,
    TNFRGraph,
)
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")

ValidatorFunc: TypeAlias = Callable[[TNFRGraph], None]
"""Callable signature expected for validation routines."""

NodeData = Mapping[str, object]
"""Read-only node attribute mapping used by validators."""

AliasSequence = Sequence[str]
"""Sequence of accepted attribute aliases."""

__all__ = ("validate_window", "run_validators")


def validate_window(window: int, *, positive: bool = False) -> int:
    """Validate ``window`` as an ``int`` and return it.

    Non-integer values raise :class:`TypeError`. When ``positive`` is ``True``
    the value must be strictly greater than zero; otherwise it may be zero.
    Negative values always raise :class:`ValueError`.
    """

    if isinstance(window, bool) or not isinstance(window, numbers.Integral):
        raise TypeError("'window' must be an integer")
    if window < 0 or (positive and window == 0):
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"'window'={window} must be {kind}")
    return int(window)


def _require_attr(data: NodeData, alias: AliasSequence, node: NodeId, name: str) -> float:
    """Return attribute value or raise if missing."""

    mapping: dict[str, object]
    if isinstance(data, dict):
        mapping = data
    else:
        mapping = dict(data)
    val = get_attr(mapping, alias, None)
    if val is None:
        raise ValueError(f"Missing {name} attribute in node {node}")
    return float(val)


def _validate_sigma(graph: TNFRGraph) -> None:
    from ..sense import sigma_vector_from_graph

    sv = sigma_vector_from_graph(graph)
    if sv.get("mag", 0.0) > 1.0 + sys.float_info.epsilon:
        raise ValueError("σ norm exceeds 1")


GRAPH_VALIDATORS: tuple[ValidatorFunc, ...] = (_validate_sigma,)
"""Ordered collection of graph-level validators."""


def _check_epi_vf(
    epi: EPIValue,
    vf: StructuralFrequency,
    epi_min: float,
    epi_max: float,
    vf_min: float,
    vf_max: float,
    node: NodeId,
) -> None:
    _check_range(epi, epi_min, epi_max, "EPI", node)
    _check_range(vf, vf_min, vf_max, "VF", node)


def _out_of_range_msg(name: str, node: NodeId, val: float) -> str:
    return f"{name} out of range in node {node}: {val}"


def _check_range(
    val: float,
    lower: float,
    upper: float,
    name: str,
    node: NodeId,
    tol: float = 1e-9,
) -> None:
    if not within_range(val, lower, upper, tol):
        raise ValueError(_out_of_range_msg(name, node, val))


def _check_glyph(glyph: str | None, node: NodeId) -> None:
    if glyph and glyph not in GLYPHS_CANONICAL_SET:
        raise KeyError(f"Invalid glyph {glyph} in node {node}")


def run_validators(graph: TNFRGraph) -> None:
    """Run all invariant validators on ``G`` with a single node pass."""
    from ..glyph_history import last_glyph

    epi_min = float(get_param(graph, "EPI_MIN"))
    epi_max = float(get_param(graph, "EPI_MAX"))
    vf_min = float(get_param(graph, "VF_MIN"))
    vf_max = float(get_param(graph, "VF_MAX"))

    for node, data in graph.nodes(data=True):
        epi = EPIValue(_require_attr(data, ALIAS_EPI, node, "EPI"))
        vf = StructuralFrequency(_require_attr(data, ALIAS_VF, node, "VF"))
        _check_epi_vf(epi, vf, epi_min, epi_max, vf_min, vf_max, node)
        _check_glyph(last_glyph(data), node)

    for validator in GRAPH_VALIDATORS:
        validator(graph)
