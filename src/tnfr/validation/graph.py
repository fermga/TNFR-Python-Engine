"""Graph-level validation helpers enforcing TNFR invariants."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from ..alias import get_attr
from ..glyph_runtime import last_glyph
from ..config.constants import GLYPHS_CANONICAL_SET
from ..constants import get_param
from ..constants.aliases import ALIAS_EPI, ALIAS_VF
from ..utils import within_range
from ..types import (
    EPIValue,
    NodeAttrMap,
    NodeId,
    StructuralFrequency,
    TNFRGraph,
    ValidatorFunc,
    ensure_bepi,
)

NodeData = NodeAttrMap
"""Read-only node attribute mapping used by validators."""

AliasSequence = Sequence[str]
"""Sequence of accepted attribute aliases."""

__all__ = ("run_validators", "GRAPH_VALIDATORS")


def _materialize_node_mapping(data: NodeData) -> dict[str, object]:
    if isinstance(data, dict):
        return data
    return dict(data)


def _require_attr(data: NodeData, alias: AliasSequence, node: NodeId, name: str) -> float:
    """Return scalar attribute value or raise if missing."""

    mapping = _materialize_node_mapping(data)
    val = get_attr(mapping, alias, None)
    if val is None:
        raise ValueError(f"Missing {name} attribute in node {node}")
    return float(val)


def _require_epi(data: NodeData, node: NodeId) -> EPIValue:
    """Return a validated BEPI element stored in ``data``."""

    mapping = _materialize_node_mapping(data)
    value = get_attr(mapping, ALIAS_EPI, None, conv=lambda obj: obj)
    if value is None:
        raise ValueError(f"Missing EPI attribute in node {node}")
    try:
        return ensure_bepi(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid EPI payload in node {node}: {exc}") from exc


def _validate_sigma(graph: TNFRGraph) -> None:
    from ..sense import sigma_vector_from_graph

    sv = sigma_vector_from_graph(graph)
    if sv.get("mag", 0.0) > 1.0 + sys.float_info.epsilon:
        raise ValueError("σ norm exceeds 1")


GRAPH_VALIDATORS: tuple[ValidatorFunc, ...] = (_validate_sigma,)
"""Ordered collection of graph-level validators."""


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _check_epi(
    epi: EPIValue,
    epi_min: float,
    epi_max: float,
    node: NodeId,
) -> None:
    continuous_max = _max_abs(epi.f_continuous)
    discrete_max = _max_abs(epi.a_discrete)
    _check_range(continuous_max, epi_min, epi_max, "EPI continuous", node)
    _check_range(discrete_max, epi_min, epi_max, "EPI discrete", node)

    spacings = np.diff(epi.x_grid)
    if np.any(spacings <= 0.0):
        raise ValueError(f"EPI grid must be strictly increasing for node {node}")
    if not np.allclose(spacings, spacings[0], rtol=1e-9, atol=1e-12):
        raise ValueError(f"EPI grid must be uniform for node {node}")


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
        epi = _require_epi(data, node)
        vf = StructuralFrequency(_require_attr(data, ALIAS_VF, node, "VF"))
        _check_epi(epi, epi_min, epi_max, node)
        _check_range(vf, vf_min, vf_max, "VF", node)
        _check_glyph(last_glyph(data), node)

    for validator in GRAPH_VALIDATORS:
        validator(graph)
