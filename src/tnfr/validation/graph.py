"""Graph-level validation helpers enforcing TNFR invariants."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from ..alias import get_attr
from ..glyph_runtime import last_glyph
from ..config.constants import GLYPHS_CANONICAL_SET
from ..constants import get_aliases, get_param
from ..utils import within_range
from ..types import (
    EPIValue,
    NodeAttrMap,
    NodeId,
    StructuralFrequency,
    TNFRGraph,
    ValidatorFunc,
)
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")

NodeData = NodeAttrMap
"""Read-only node attribute mapping used by validators."""

AliasSequence = Sequence[str]
"""Sequence of accepted attribute aliases."""

__all__ = ("run_validators", "GRAPH_VALIDATORS")


def _require_attr(
    data: NodeData, alias: AliasSequence, node: NodeId, name: str
) -> float:
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
    """Run all invariant validators on ``graph`` with a single node pass."""

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
