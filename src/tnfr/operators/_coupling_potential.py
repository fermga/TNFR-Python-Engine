"""Admit one supplied potential UM pair without fabricating realized support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..errors import TNFRValueError
from ._epi_domain import require_real_scalar_epi
from ._phase_gate import (
    U3PhaseNeighborSet,
    resolve_u3_phase_neighbors,
    select_u3_phase_neighbors,
)
from .factor_contracts import _resolve_coupling_flags


def _resolve_potential_scope(scope: Any, partner: Any) -> str:
    """Admit an explicit request scope without reading or mutating a graph."""
    if type(scope) is not str:
        raise TypeError("potential_scope must be an exact string")
    if scope not in {"isolated", "disjoint_components"}:
        raise ValueError("potential_scope must be 'isolated' or 'disjoint_components'")
    if scope != "isolated" and partner is None:
        raise ValueError("disjoint_components requires potential_partner")
    return scope


def _raw(snapshot: Any, node: Any, aliases: tuple[str, ...], default: Any) -> Any:
    return get_attr(
        snapshot.nodes[node], aliases, default, strict=True, conv=lambda value: value
    )


def _component(snapshot: Any, start: Any) -> set[Any]:
    """Read one whole undirected support component, including zero weights."""
    reached = {start}
    pending = [start]
    while pending:
        for neighbor in snapshot.neighbors(pending.pop()):
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _potential_coupling_boundary(
    snapshot: Any, node: Any, partner: Any, selection: U3PhaseNeighborSet
) -> tuple[tuple[Any, Any], ...]:
    """Read all old relations incident to the declared phase write set."""
    participants = (
        (node, *selection.neighbors)
        if _resolve_coupling_flags(snapshot.graph).bidirectional
        else (node,)
    )
    seen: set[frozenset[Any]] = set()
    relations: list[tuple[Any, Any]] = []
    for left in dict.fromkeys(participants):
        for right in snapshot.neighbors(left):
            key = frozenset((left, right))
            if key not in seen:
                seen.add(key)
                relations.append((left, right))
    key = frozenset((node, partner))
    if key not in seen:
        relations.append((node, partner))
    return tuple(relations)


def _validate_coupling_boundary(
    snapshot: Any,
    relations: tuple[tuple[Any, Any], ...],
    phase_limit: float,
    *,
    phase_overrides: Mapping[Any, float] | None = None,
) -> None:
    """Require every declared relation, without extending the consensus set."""

    def phase(endpoint: Any) -> Any:
        if phase_overrides is not None and endpoint in phase_overrides:
            return phase_overrides[endpoint]
        return _raw(snapshot, endpoint, ALIAS_THETA, None)

    for left, right in relations:
        select_u3_phase_neighbors(
            phase(left),
            (right,),
            phase_getter=phase,
            phase_limit=phase_limit,
            require_compatible=True,
        )


def resolve_potential_coupling(
    snapshot: Any,
    node: Any,
    potential_partner: Any,
    *,
    potential_scope: str = "isolated",
) -> U3PhaseNeighborSet:
    """Admit a supplied partner and an augmented initial U3 participant set.

    In the default scope only the partner requires degree zero, including
    incoming directed edges. Existing compatible outgoing neighbors retain
    their ordinary order; the mandatory partner follows them. This supplied
    access is not an existing
    edge, authenticated parent/child lineage or an autonomous selector.
    Phase, scalar form and nonnegative capacity belong to the pair domain;
    the current missing-form/capacity defaults are retained. Public activity
    thresholds, grammar, the final affinity and commit are separate checks.

    Domain errors use ``TNFRValueError``; the shared phase gate retains its
    ``U3PhaseGateError``. This reader performs no mutation or random selection.
    ``disjoint_components`` instead requires two distinct whole components
    of an undirected non-multigraph, each with at least two nodes. All old
    relations incident to any proposed phase writer must initially pass U3;
    their extra endpoints do not become numerical consensus participants.
    """
    potential_scope = _resolve_potential_scope(potential_scope, potential_partner)
    if node not in snapshot or potential_partner not in snapshot:
        raise TNFRValueError(
            "Coupling potential pair requires two existing node identities",
            context={"operator": "Coupling", "failed_condition": "potential_pair"},
        )
    if node == potential_partner:
        raise TNFRValueError(
            "Coupling potential partner must differ from its target",
            context={"operator": "Coupling", "failed_condition": "potential_pair"},
        )
    if potential_scope == "disjoint_components":
        if snapshot.is_directed() or snapshot.is_multigraph():
            raise TNFRValueError(
                "disjoint_components requires an undirected non-multigraph",
                context={
                    "operator": "Coupling",
                    "failed_condition": "potential_support",
                },
            )
        first = _component(snapshot, node)
        second = _component(snapshot, potential_partner)
        if len(first) < 2 or len(second) < 2 or not first.isdisjoint(second):
            raise TNFRValueError(
                "Coupling requires distinct components with at least two nodes each",
                context={
                    "operator": "Coupling",
                    "failed_condition": "potential_support",
                },
            )
    elif (
        snapshot.has_edge(node, potential_partner)
        or snapshot.degree(potential_partner) != 0
    ):
        raise TNFRValueError(
            "Coupling potential partner must be totally isolated",
            context={"operator": "Coupling", "failed_condition": "potential_support"},
        )
    if not _resolve_coupling_flags(snapshot.graph).functional_links:
        raise TNFRValueError(
            "Coupling potential pair requires enabled UM_FUNCTIONAL_LINKS",
            context={"operator": "Coupling", "failed_condition": "potential_links"},
        )

    def raw(endpoint: Any, aliases: tuple[str, ...], default: Any) -> Any:
        return _raw(snapshot, endpoint, aliases, default)

    for endpoint in (node, potential_partner):
        require_real_scalar_epi(
            raw(endpoint, ALIAS_EPI, 0.0),
            operator="Coupling",
            label="potential endpoint EPI",
        )
        try:
            capacity, _ = finite_represented_real(
                raw(endpoint, ALIAS_VF, 0.0), "potential endpoint structural frequency"
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise TNFRValueError(
                str(exc),
                context={
                    "operator": "Coupling",
                    "failed_condition": "potential_capacity",
                },
            ) from exc
        if capacity < 0.0:
            raise TNFRValueError(
                "Coupling potential endpoint structural frequency must be nonnegative",
                context={
                    "operator": "Coupling",
                    "failed_condition": "potential_capacity",
                },
            )

    # The requested partner must pass independently. An existing compatible
    # neighbor cannot hide an incompatible supplied relation.
    pair = resolve_u3_phase_neighbors(
        snapshot.graph,
        raw(node, ALIAS_THETA, None),
        (potential_partner,),
        phase_getter=lambda endpoint: raw(endpoint, ALIAS_THETA, None),
        operator_code="UM",
    )
    existing_neighbors = tuple(snapshot.neighbors(node))
    selection = pair
    if existing_neighbors:
        existing = resolve_u3_phase_neighbors(
            snapshot.graph,
            pair.target_phase,
            existing_neighbors,
            phase_getter=lambda endpoint: raw(endpoint, ALIAS_THETA, None),
            operator_code="UM",
            require_compatible=False,
        )
        selection = U3PhaseNeighborSet(
            canonical_graph_limit=pair.canonical_graph_limit,
            effective_limit=pair.effective_limit,
            target_phase=pair.target_phase,
            neighbors=(*existing.neighbors, potential_partner),
            phases=(*existing.phases, *pair.phases),
        )
    if potential_scope == "disjoint_components":
        _validate_coupling_boundary(
            snapshot,
            _potential_coupling_boundary(snapshot, node, potential_partner, selection),
            selection.effective_limit,
        )
    return selection


__all__ = ["resolve_potential_coupling"]
