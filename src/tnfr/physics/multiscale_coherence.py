"""Assumption-explicit U5 parent/child coherence assessment.

The sequence grammar can verify declared recursion and stabilizer coverage, but
it cannot infer the trajectory-level target

    C_parent >= alpha * sum(C_child).

This module evaluates that target only for a concrete hierarchy and an explicit
non-negative alpha. Every coherence value uses the canonical TNFR kernel
1 / (1 + |DeltaNFR| + |dEPI|), with both stored inputs explicitly supplied.
Absent observations cannot establish zero pressure/rate or a passing target.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from ..metrics.common import structural_coherence
from ..types import NodeId, TNFRGraph
from ._helpers import finite_real_scalar


@dataclass(frozen=True, slots=True)
class U5CoherenceAssessment:
    """Stored-input result for one declared parent/child U5 target."""

    parent: NodeId
    children: tuple[NodeId, ...]
    alpha: float
    tolerance: float
    parent_coherence: float
    child_coherences: tuple[float, ...]
    child_coherence_sum: float
    required_parent_coherence: float
    residual: float
    satisfies_target: bool


def _finite_nonnegative(value: Any, *, label: str) -> float:
    result = finite_real_scalar(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _node_coherence(graph: TNFRGraph, node: NodeId) -> float:
    if node not in graph:
        raise KeyError(f"hierarchy node {node!r} is not present in the graph")
    data = graph.nodes[node]
    missing = object()
    channels = []
    for aliases, label in ((ALIAS_DNFR, "DeltaNFR"), (ALIAS_DEPI, "dEPI")):
        raw = get_attr(data, aliases, missing, strict=True, conv=lambda value: value)
        if raw is missing:
            raise ValueError(f"node {node!r} is missing required stored {label}")
        channels.append(abs(finite_real_scalar(raw, f"node {node!r} {label}")))
    return finite_real_scalar(
        structural_coherence(*channels), f"node {node!r} coherence"
    )


def assess_u5_parent_child_coherence(
    graph: TNFRGraph,
    parent: NodeId,
    *,
    alpha: float,
    children: Iterable[NodeId] | None = None,
    tolerance: float = 0.0,
) -> U5CoherenceAssessment:
    """Evaluate C_parent >= alpha * sum(C_child) for one hierarchy.

    alpha has no graph-independent TNFR default, so callers must state it.
    tolerance is an explicit additive comparison margin and is recorded in the
    returned certificate. If children is omitted, the function reads the
    parent's sub_nodes.
    Each child must declare parent_node == parent. Every selected node must
    explicitly supply finite stored pressure and dEPI inputs. Missing channels
    raise ValueError rather than implying zero; this reader does not derive a
    rate from capacity/pressure or authenticate temporal provenance. Scalar
    materialization and output arithmetic must remain representable. The result
    is a stored-state diagnostic; it does not replace U5 sequence coverage.
    """

    alpha_value = _finite_nonnegative(alpha, label="alpha")
    tolerance_value = _finite_nonnegative(tolerance, label="tolerance")
    if parent not in graph:
        raise KeyError(f"parent node {parent!r} is not present in the graph")

    if children is None:
        raw_children = graph.nodes[parent].get("sub_nodes", ())
        if isinstance(raw_children, (str, bytes)) or not isinstance(
            raw_children, Iterable
        ):
            raise TypeError("parent sub_nodes must be an iterable of node identifiers")
        child_nodes = tuple(raw_children)
    else:
        if isinstance(children, (str, bytes)):
            raise TypeError("children must be an iterable of node identifiers")
        child_nodes = tuple(children)

    if not child_nodes:
        raise ValueError("U5 assessment requires at least one child")
    if len(set(child_nodes)) != len(child_nodes):
        raise ValueError("U5 assessment children must be unique")
    if parent in child_nodes:
        raise ValueError("a U5 parent cannot be its own child")

    for child in child_nodes:
        if child not in graph:
            raise KeyError(f"child node {child!r} is not present in the graph")
        declared_parent = graph.nodes[child].get("parent_node")
        if declared_parent != parent:
            raise ValueError(
                f"child node {child!r} declares parent {declared_parent!r}, "
                f"expected {parent!r}"
            )

    parent_coherence = _node_coherence(graph, parent)
    child_coherences = tuple(_node_coherence(graph, child) for child in child_nodes)
    child_sum = finite_real_scalar(math.fsum(child_coherences), "child coherence sum")
    required = finite_real_scalar(alpha_value * child_sum, "required parent coherence")
    if alpha_value != 0.0 and child_sum != 0.0 and required == 0.0:
        raise ValueError("required parent coherence is nonzero but underflows to zero")
    residual = finite_real_scalar(parent_coherence - required, "U5 coherence residual")
    comparison_bound = finite_real_scalar(
        parent_coherence + tolerance_value, "tolerated parent coherence"
    )

    return U5CoherenceAssessment(
        parent=parent,
        children=child_nodes,
        alpha=alpha_value,
        tolerance=tolerance_value,
        parent_coherence=parent_coherence,
        child_coherences=child_coherences,
        child_coherence_sum=child_sum,
        required_parent_coherence=required,
        residual=residual,
        satisfies_target=comparison_bound >= required,
    )


__all__ = ["U5CoherenceAssessment", "assess_u5_parent_child_coherence"]
