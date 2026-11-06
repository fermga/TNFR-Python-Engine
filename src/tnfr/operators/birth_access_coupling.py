"""Live direct birth associations as an explicit potential Coupling policy.

The caller supplies a parent and invocation occasion. Access selects its unique
isolated associated child before numerical UM admission; ambiguity abstains.
Consistent current metadata is not authentication of a historical birth, and a
passive report is never accepted as permission for a later action.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config.parsing import parse_bool
from ..constants import get_param
from ._argument_validation import finite_real, validate_common_execution_arguments
from .coupling import Coupling
from .network_stage import (
    GraphTransactionSnapshot,
    NetworkStageResult,
    execute_coupling_stage,
)

__all__ = [
    "BirthAssociationAccess",
    "BirthAccessCouplingDispatch",
    "observe_birth_association_access",
    "execute_birth_access_coupling",
]

_POLICY = "unique_isolated_birth_child"
_OPERATOR = "Birth-association Coupling"
_MISSING = object()
_OPTIONS = frozenset(
    {
        "window",
        "validate_preconditions",
        "collect_metrics",
        "validate_nodal_equation",
        "validate_postconditions",
        "dt",
    }
)


@dataclass(frozen=True, slots=True)
class BirthAssociationAccess:
    """Current association and isolation, without numerical UM eligibility.

    Tuple order records storage order and never breaks ties. The record is
    neither a birth certificate nor an execution input. All reported children
    were independently checked before the isolation partition was formed.
    """

    parent: Any
    associated_children: tuple[Any, ...]
    isolated_children: tuple[Any, ...]
    connected_children: tuple[Any, ...]
    policy: str = field(default=_POLICY, init=False)

    @property
    def status(
        self,
    ) -> Literal["no_associated_child", "no_isolated_child", "ambiguous", "selected"]:
        if not self.associated_children:
            return "no_associated_child"
        if not self.isolated_children:
            return "no_isolated_child"
        if len(self.isolated_children) > 1:
            return "ambiguous"
        return "selected"

    @property
    def selected_partner(self) -> Any | None:
        return self.isolated_children[0] if self.status == "selected" else None


@dataclass(frozen=True, slots=True)
class BirthAccessCouplingDispatch:
    """Outcome of one supplied occasion, retaining the access selection origin."""

    access: BirthAssociationAccess
    stage_result: NetworkStageResult | None

    @property
    def executed(self) -> bool:
        return self.stage_result is not None


def _read_access(graph: Any, parent: Any) -> BirthAssociationAccess:
    if parent not in graph:
        raise KeyError(parent)
    raw = graph.nodes[parent].get("sub_nodes", [])
    if type(raw) is not list:
        raise TypeError("birth access sub_nodes must be an exact list")
    children = tuple(raw)
    if len(set(children)) != len(children):
        raise ValueError("birth access child associations must be unique")

    hierarchy = graph.graph.get("hierarchy", _MISSING)
    if hierarchy is not _MISSING:
        if type(hierarchy) is not dict:
            raise TypeError("birth access hierarchy must be an exact dictionary")
        if parent in hierarchy:
            indexed = hierarchy[parent]
            if type(indexed) is not list:
                raise TypeError("birth access hierarchy children must be an exact list")
            if tuple(indexed) != children:
                raise ValueError("birth access hierarchy disagrees with sub_nodes")

    # Check every reference before support filtering. Inaccessible siblings must
    # not conceal malformed association state behind a unique usable child.
    for child in children:
        if child == parent:
            raise ValueError("birth access child must differ from its parent")
        if child not in graph:
            raise ValueError("birth access child association is missing from the graph")
        owner = graph.nodes[child].get("parent_node", _MISSING)
        if owner is _MISSING or owner != parent:
            raise ValueError("birth access child has inconsistent parent_node")

    isolated = []
    connected = []
    for child in children:
        (isolated if graph.degree(child) == 0 else connected).append(child)
    return BirthAssociationAccess(parent, children, tuple(isolated), tuple(connected))


def observe_birth_association_access(graph: Any, parent: Any) -> BirthAssociationAccess:
    """Read direct associations without requiring UM readiness or changing state.

    Missing hierarchy indexing is allowed; a present index must agree. Historical
    sub_epis and unrelated nodes do not supply or authenticate access. Candidate
    discovery is local, although transaction ownership may inspect the full graph.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        access = _read_access(graph, parent)
        transaction.restore(graph)
        return access
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


def _materialize_options(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError("execution_kwargs must be a mapping or None")
    options = {} if raw is None else dict(raw)
    if any(type(key) is not str or key not in _OPTIONS for key in options):
        raise ValueError("unsupported birth-access Coupling execution option")
    return options


def _admit_execution_options(graph: Any, options: dict[str, Any]) -> dict[str, Any]:
    from . import _validated_execution_window

    options = dict(options)
    options.setdefault("validate_preconditions", True)
    validate_common_execution_arguments(graph.graph, options, operator=_OPERATOR)
    if not options["validate_preconditions"]:
        raise ValueError("birth-access Coupling requires validate_preconditions=True")
    options["window"] = _validated_execution_window(graph, options.get("window"))
    if "dt" in options:
        options["dt"] = finite_real(
            options["dt"],
            operator=_OPERATOR,
            label="dt",
            lower=math.nextafter(0.0, math.inf),
        )
    canon = get_param(graph, "GRAMMAR_CANON")
    if not isinstance(canon, Mapping):
        raise TypeError("GRAMMAR_CANON must be a mapping")
    if not parse_bool(canon.get("enabled", False)) or (
        graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise"
    ):
        raise ValueError(
            "birth-access Coupling requires enabled grammar and raise mode"
        )
    return options


def execute_birth_access_coupling(
    graph: Any,
    parent: Any,
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> BirthAccessCouplingDispatch:
    """Derive one isolated child and execute the actual atomic UM stage once.

    Empty or ambiguous access abstains without dispatching an empty stage.
    Unique access does not imply numerical eligibility: all UM failures propagate
    and restore this invocation, preserving any earlier completed birth. Common
    request and grammar policy are admitted even when access abstains; unused UM
    numerical fields are not. Optional activity gates retain their graph policy.

    The inner stage keeps its explicit-request provenance; this outer result owns
    the state-derived access policy. No clock advance or one-use birth receipt is
    supplied. A later declared disconnection can make a child accessible again.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        # Materializing a caller mapping must not silently change the source from
        # which access is derived. Existing graph-owned entry state is restored.
        transaction.restore(graph)
        options = _admit_execution_options(graph, options)
        access = _read_access(graph, parent)
        if access.status != "selected":
            transaction.restore(graph)
            return BirthAccessCouplingDispatch(access, None)
        stage = execute_coupling_stage(
            graph,
            Coupling(),
            (parent,),
            potential_partner=access.selected_partner,
            transaction_snapshot=transaction,
            **options,
        )
        return BirthAccessCouplingDispatch(access, stage)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
