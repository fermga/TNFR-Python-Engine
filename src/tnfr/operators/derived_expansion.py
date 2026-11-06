"""An explicit single-derived-member family composed with actual scalar VAL.

Inheritance ratios and pre/post ordering are supplied constitutive choices.
Ordinary Expansion is unchanged. Birth supplies no edge, autonomous clock,
collective identity, conservation law or inverse synthesis operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._exact_time import bounded_represented_real, finite_represented_real
from ..constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._birth_materialization import (
    materialize_isolated_member,
    require_fresh_member_identity,
)
from .expansion import Expansion
from .network_stage import GraphTransactionSnapshot
from .retained_organization import (
    RetainedMemberState,
    RetainedOrganization,
    RetainedScaleDispatch,
    _member_payload,
    capture_retained_organization,
    execute_retained_scale_stage,
)

__all__ = [
    "DerivedExpansionPolicy",
    "DerivedExpansionDispatch",
    "execute_derived_expansion_stage",
]


def _admit_policy(policy: DerivedExpansionPolicy) -> tuple[str, float, float]:
    if type(policy) is not DerivedExpansionPolicy:
        raise TypeError("derived Expansion requires its exact policy type")
    order = policy.source_order
    if type(order) is not str or order not in ("pre_val", "post_val"):
        raise ValueError("source_order must be pre_val or post_val")
    ratios = []
    for key in ("form_ratio", "capacity_ratio"):
        value = bounded_represented_real(
            getattr(policy, key), key, minimum=0, maximum=1
        )[0]
        if value == 0.0:
            raise ValueError(f"{key} must be strictly positive")
        ratios.append(value)
    return order, ratios[0], ratios[1]


@dataclass(frozen=True, slots=True, kw_only=True)
class DerivedExpansionPolicy:
    """Required source boundary and ratios for this bounded derived-copy family.

    Both ratios lie in (0,1]. These restrictions define the selected family,
    not a necessary or uniquely canonical law of proliferation.
    """

    source_order: str
    form_ratio: float
    capacity_ratio: float

    def __post_init__(self) -> None:
        _order, form, capacity = _admit_policy(self)
        object.__setattr__(self, "form_ratio", form)
        object.__setattr__(self, "capacity_ratio", capacity)


def _product(value: float, ratio: float, label: str) -> float:
    result = value * ratio
    if not math.isfinite(result) or (value != 0.0 and result == 0.0):
        raise ValueError(f"{label} must remain finite without nonzero underflow")
    return result


def _signature(value: Any, graph: Any) -> Any:
    return structural_proof_signature(value, opaque_references=(graph,))


def _same(left: Any, right: Any, graph: Any) -> bool:
    return proof_stamps_are_identical(_signature(left, graph), _signature(right, graph))


def _core_boundary(graph: Any, nodes: tuple[Any, ...]) -> Any:
    """Retain complete old state across the birth commit after scalar VAL."""
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return _signature(
        (
            tuple((node, graph.nodes[node]) for node in nodes),
            tuple(
                (
                    node,
                    tuple(
                        graph.predecessors(node)
                        if graph.is_directed()
                        else graph.neighbors(node)
                    ),
                    tuple(graph.neighbors(node)),
                )
                for node in nodes
            ),
            edges,
            graph.graph,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        graph,
    )


@dataclass(frozen=True, slots=True)
class DerivedExpansionDispatch:
    """Consistent source/successor summary, not authenticated execution evidence."""

    policy: DerivedExpansionPolicy
    parent: Any
    child: Any
    inheritance_source: RetainedMemberState
    scale_result: RetainedScaleDispatch
    after: RetainedOrganization

    def __post_init__(self) -> None:
        order, form_ratio, capacity_ratio = _admit_policy(self.policy)
        if type(self.scale_result) is not RetainedScaleDispatch:
            raise TypeError("derived Expansion requires its actual scale result")
        self.scale_result.__post_init__()
        scale = self.scale_result
        if scale.stage_result.operator != Expansion.name:
            raise ValueError("derived Expansion requires an Expansion stage")
        graph = scale.before._graph_owner
        if (
            type(self.after) is not RetainedOrganization
            or not self.after._proof_fields_are_intact()
        ):
            raise ValueError(
                "derived Expansion requires an intact successor observation"
            )
        source = scale.before if order == "pre_val" else scale.after
        if (
            self.after._graph_owner is not graph
            or not _same(scale.before.members, (self.parent,), graph)
            or not _same(self.after.members, (self.parent, self.child), graph)
            or self.parent == self.child
            or type(self.inheritance_source) is not RetainedMemberState
            or not _same(
                _member_payload(self.inheritance_source),
                _member_payload(source.member_states[0]),
                graph,
            )
        ):
            raise ValueError("derived Expansion source or member association changed")
        retained_parent, child = self.after.member_states
        if (
            not _same(
                _member_payload(retained_parent),
                _member_payload(scale.after.member_states[0]),
                graph,
            )
            or not _same(
                self.after.member_metadata[0], scale.after.member_metadata[0], graph
            )
            or self.after.directed != scale.after.directed
            or self.after.multigraph != scale.after.multigraph
            or not _same(
                self.after.incoming_ports[0], scale.after.incoming_ports[0], graph
            )
            or not _same(
                self.after.outgoing_ports[0], scale.after.outgoing_ports[0], graph
            )
            or self.after.incoming_ports[1]
            or self.after.outgoing_ports[1]
            or not _same(self.after.incident_edges, scale.after.incident_edges, graph)
        ):
            raise ValueError(
                "derived Expansion must retain its core and isolate the child"
            )
        expected = (
            _product(self.inheritance_source.epi, form_ratio, "child form"),
            _product(
                self.inheritance_source.capacity, capacity_ratio, "child capacity"
            ),
            self.inheritance_source.phase,
            0.0,
            self.inheritance_source.epi_kind,
        )
        if not _same(
            (child.epi, child.capacity, child.phase, child.pressure, child.epi_kind),
            expected,
            graph,
        ):
            raise ValueError("derived child coordinates contradict the declared map")
        if (
            child.grammar.history
            or child.grammar.u2_debt
            or child.grammar.prior_coherence
            or child.grammar.operator_step
        ):
            raise ValueError("derived child must retain its own fresh operator context")
        if not _same(
            child.temporal_context,
            (("epi_history", _signature([child.epi], graph)),),
            graph,
        ):
            raise ValueError(
                "derived child must retain only its own initial EPI sample"
            )


def execute_derived_expansion_stage(
    graph: Any,
    source: RetainedOrganization,
    child: Any,
    policy: DerivedExpansionPolicy,
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> DerivedExpansionDispatch:
    """Apply actual parent VAL and create one declared isolated derived member.

    Source must be an intact current retained singleton. All ordinary VAL
    admission still applies. Both inherited coordinates use the required
    source order; no caller state replaces the actual pre/post stage capture.
    The graph-owned transaction encloses the scale, birth and final checks.
    Later coupling requires a separate actual UM request under its own gates.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        order, form_ratio, capacity_ratio = _admit_policy(policy)
        policy = DerivedExpansionPolicy(
            source_order=order, form_ratio=form_ratio, capacity_ratio=capacity_ratio
        )
        if type(source) is not RetainedOrganization or len(source.members) != 1:
            raise ValueError("derived Expansion requires a retained singleton parent")
        require_fresh_member_identity(graph, child)
        minimum = finite_represented_real(graph.graph.get("EPI_MIN", -1.0), "EPI_MIN")[
            0
        ]
        maximum = finite_represented_real(graph.graph.get("EPI_MAX", 1.0), "EPI_MAX")[0]
        if minimum > maximum:
            raise ValueError("EPI_MIN must not exceed EPI_MAX")
        scale = execute_retained_scale_stage(
            graph, source, Expansion(), execution_kwargs=execution_kwargs
        )
        parent = source.members[0]
        inherited = (scale.before if order == "pre_val" else scale.after).member_states[
            0
        ]
        epi = _product(inherited.epi, form_ratio, "child form")
        capacity = _product(inherited.capacity, capacity_ratio, "child capacity")
        if not minimum <= epi <= maximum:
            raise ValueError("derived child form lies outside configured EPI bounds")
        attributes = {
            EPI_PRIMARY: epi,
            VF_PRIMARY: capacity,
            THETA_PRIMARY: inherited.phase,
            DNFR_PRIMARY: 0.0,
            "epi_kind": inherited.epi_kind,
            "derived_from": parent,
            "derivation_operator": Expansion.glyph.value,
            "derivation_source_order": order,
            "epi_history": [epi],
            "glyph_history": [],
        }
        old_nodes = tuple(graph.nodes)
        boundary = _core_boundary(graph, old_nodes)
        materialize_isolated_member(graph, child, attributes)
        if not _same(
            tuple(graph.nodes), (*old_nodes, child), graph
        ) or not proof_stamps_are_identical(boundary, _core_boundary(graph, old_nodes)):
            raise RuntimeError(
                "derived member materialization changed the retained core"
            )
        after = capture_retained_organization(graph, (parent, child))
        if not proof_stamps_are_identical(boundary, _core_boundary(graph, old_nodes)):
            raise RuntimeError("derived member observation changed the retained core")
        return DerivedExpansionDispatch(policy, parent, child, inherited, scale, after)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
