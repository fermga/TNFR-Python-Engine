"""Set-valued THOL birth eligibility and an explicit finite dispatch policy.

The relation reads all current nodes, retaining grammar, configured gates,
history and public birth-proposal evidence separately. It does not select a
unique parent or derive a decision to execute. The optional dispatcher is the
declared policy 'execute every eligible parent once in one simultaneous stage'.
It reuses public THOL planning/commit and never synthesizes pressure or history.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from ..rng import validate_graph_seed
from ..types import Glyph
from ._argument_validation import finite_real, nonnegative_integer, strict_bool
from ._thol_config import resolve_thol_bifurcation_threshold
from .grammar_debt import require_replayable_history
from .grammar_dynamics import validate_candidate
from .network_stage import (
    GraphTransactionSnapshot, NetworkStageResult, TWO_PHASE_JACOBI,
    _detached_stage_graph, _merge_and_validate_self_organization_stage,
    execute_self_organization_stage,
)
from .nodal_equation import (
    StructuralAccelerationObservation, observe_structural_acceleration,
)
from .preconditions import OperatorPreconditionError
from .self_organization import SelfOrganization

__all__ = [
    "SelfOrganizationCandidate", "SelfOrganizationEligibility",
    "EligibleSelfOrganizationDispatch", "observe_self_organization_eligibility",
    "execute_eligible_self_organization_stage",
]

_OPERATOR = "Self-organization eligibility"
_OPTIONS = frozenset({
    "tau", "window", "validate_preconditions", "collect_metrics",
    "validate_nodal_equation", "dt",
})
_DOMAIN_ERRORS = (ValueError, TypeError, KeyError, OverflowError, OperatorPreconditionError)


@dataclass(frozen=True, slots=True)
class SelfOrganizationCandidate:
    """Independent checks of supplied state, not an execution certificate."""

    node: Any
    acceleration: StructuralAccelerationObservation | None
    acceleration_error: str | None
    grammar_allowed: bool | None
    grammar_violations: tuple[tuple[str, str, str], ...]
    grammar_error: str | None
    optional_gate_enabled: bool
    application_preconditions_passed: bool
    application_error: str | None
    proposal_valid: bool
    proposal_error: str | None
    threshold_crossed: bool | None
    birth_proposed: bool | None
    depth_limit_reached: bool | None

    @property
    def eligible(self) -> bool:
        return bool(
            self.acceleration is not None and self.acceleration.available
            and self.grammar_allowed and self.application_preconditions_passed
            and self.proposal_valid and self.threshold_crossed and self.birth_proposed
        )


@dataclass(frozen=True, slots=True)
class SelfOrganizationEligibility:
    """All-node relation in snapshot rank; tuple order is not a tie-break.

    Joint viability concerns detached merged support only. Monitors and later
    execution checks can still fail. This supplied-state observation carries
    no causal seal, future guarantee or general relabeling theorem.
    """

    candidates: tuple[SelfOrganizationCandidate, ...]
    tau: float
    execution_options: tuple[tuple[str, Any], ...]
    joint_stage_viable: bool
    joint_error: str | None

    @property
    def eligible_nodes(self) -> tuple[Any, ...]:
        return tuple(candidate.node for candidate in self.candidates if candidate.eligible)


@dataclass(frozen=True, slots=True)
class EligibleSelfOrganizationDispatch:
    """Outcome of one explicit invocation; not autonomous or future behavior."""

    eligibility: SelfOrganizationEligibility
    stage_result: NetworkStageResult | None
    parent_children: tuple[tuple[Any, Any], ...]
    policy: str = "all_eligible_once"


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _edge_support(graph: Any) -> frozenset[Any]:
    """Capture topology, including orientation and parallel keys, not attributes."""
    edges = graph.edges(keys=True) if graph.is_multigraph() else (
        (left, right, None) for left, right in graph.edges
    )
    return frozenset(
        ((left, right) if graph.is_directed() else frozenset((left, right)), key)
        for left, right, key in edges
    )


def _materialize_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Freeze supported caller values once, inside the outer transaction."""
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("execution_kwargs must be a mapping or None")
    values = dict(options)
    if any(type(key) is not str or key not in _OPTIONS for key in values):
        raise ValueError("unsupported Self-organization eligibility option")
    result = {}
    for key, value in values.items():
        if key in ("validate_preconditions", "collect_metrics", "validate_nodal_equation"):
            result[key] = strict_bool(value, operator=_OPERATOR, label=key)
        elif key == "window":
            result[key] = (
                None if value is None else
                nonnegative_integer(value, operator=_OPERATOR, label=key)
            )
        elif key == "tau" and value is None:
            result[key] = None
        else:
            result[key] = finite_real(
                value, operator=_OPERATOR, label=key,
                lower=0.0 if key == "tau" else math.nextafter(0.0, math.inf),
            )
    return result


def _observe(snapshot: Any, options: dict[str, Any]) -> SelfOrganizationEligibility:
    from . import _validated_execution_window
    from ._argument_validation import validate_common_execution_arguments

    options = dict(options)
    options["window"] = _validated_execution_window(snapshot, options.get("window"))
    validate_graph_seed(snapshot)
    validate_common_execution_arguments(snapshot.graph, options, operator=_OPERATOR)
    tau = resolve_thol_bifurcation_threshold(snapshot.graph, options.get("tau"))
    options["tau"] = tau
    gate_enabled = options.get("validate_preconditions", True) and snapshot.graph.get(
        "VALIDATE_OPERATOR_PRECONDITIONS", False,
    )
    operator = SelfOrganization()
    candidates = []
    eligible_proposals = []
    for node in tuple(snapshot.nodes):
        acceleration = None
        acceleration_error = None
        try:
            acceleration = observe_structural_acceleration(snapshot, node)
        except _DOMAIN_ERRORS as exc:
            acceleration_error = _error(exc)
        grammar_allowed = None
        grammar_violations = ()
        grammar_error = None
        try:
            require_replayable_history(snapshot.nodes[node].get("glyph_history"))
            # Runtime grammar's own recent window is distinct from trace retention.
            grammar = validate_candidate(snapshot, node, Glyph.THOL)
            grammar_allowed = grammar.allowed
            grammar_violations = tuple(
                (item.rule, item.severity, item.message) for item in grammar.violations
            )
        except _DOMAIN_ERRORS as exc:
            grammar_error = _error(exc)
        application_error = None
        try:
            operator._validate_hard_invariants(snapshot, node)
            operator._validate_application_preconditions(snapshot, node, **options)
        except _DOMAIN_ERRORS as exc:
            application_error = _error(exc)
        proposal = None
        proposal_error = None
        try:
            # Inspect independently even when grammar or the optional gate refuses.
            proposal = operator._prepare_execution(snapshot, node, options)
        except _DOMAIN_ERRORS as exc:
            proposal_error = _error(exc)
        candidate = SelfOrganizationCandidate(
            node, acceleration, acceleration_error,
            grammar_allowed, grammar_violations, grammar_error,
            bool(gate_enabled), application_error is None, application_error,
            proposal is not None, proposal_error,
            abs(acceleration.value) > tau
            if acceleration is not None and acceleration.available else None,
            proposal.bifurcation is not None if proposal is not None else None,
            proposal.depth_limit_reached is not None if proposal is not None else None,
        )
        candidates.append(candidate)
        if candidate.eligible:
            eligible_proposals.append((node, proposal))
    joint_error = None
    if eligible_proposals:
        try:
            _merge_and_validate_self_organization_stage(
                snapshot, operator, tuple(eligible_proposals),
            )
        except _DOMAIN_ERRORS as exc:
            joint_error = _error(exc)
    return SelfOrganizationEligibility(
        tuple(candidates), tau, tuple(sorted(options.items())),
        joint_error is None, joint_error,
    )


def observe_self_organization_eligibility(
    graph: Any, *, execution_kwargs: Mapping[str, Any] | None = None,
) -> SelfOrganizationEligibility:
    """Observe all current nodes without committing graph-owned changes.

    Supported options are tau, trace window, precondition/metric/nodal-check
    flags and positive dt. No target subset, callback, operator override or
    future sequence context is accepted. Input materialization is captured and
    restored before reading state; successful reads are restored as well.
    Graph transactions cannot undo external I/O or effects outside their owner.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        transaction.restore(graph)
        result = _observe(_detached_stage_graph(graph), options)
        transaction.restore(graph)
        return result
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


def execute_eligible_self_organization_stage(
    graph: Any, *, execution_kwargs: Mapping[str, Any] | None = None,
) -> EligibleSelfOrganizationDispatch:
    """Explicitly execute every currently eligible parent once, atomically.

    Eligibility is freshly observed inside this invocation; no caller report
    can authorize a later action. Empty eligibility skips the stage entirely.
    One exact built-in two-phase stage owns merging and commit. No pressure
    refresh, coupling, physical time advance or repeated policy is implied.
    Success verifies one new isolated child per selected parent and preserves
    the original node/edge support (not every old attribute); failure
    restores graph-owned state using the same outer transaction. This is a
    supplied policy, not a derivation of spontaneous generation.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        transaction.restore(graph)
        eligibility = _observe(_detached_stage_graph(graph), options)
        transaction.restore(graph)
        targets = eligibility.eligible_nodes
        if not targets:
            return EligibleSelfOrganizationDispatch(eligibility, None, ())
        if not eligibility.joint_stage_viable:
            raise ValueError(f"joint THOL birth proposal rejected: {eligibility.joint_error}")
        initial_nodes = frozenset(graph.nodes)
        initial_edges = _edge_support(graph)
        initial_children = {
            node: tuple(graph.nodes[node].get("sub_nodes", ())) for node in targets
        }
        stage = execute_self_organization_stage(
            graph, SelfOrganization(), targets, transaction_snapshot=transaction,
            **dict(eligibility.execution_options),
        )
        if stage.schedule != TWO_PHASE_JACOBI or stage.nodes_processed != len(targets):
            raise RuntimeError("eligible THOL dispatch requires the built-in two-phase stage")
        births = []
        for node in targets:
            children = tuple(graph.nodes[node].get("sub_nodes", ()))
            previous = initial_children[node]
            if len(children) != len(previous) + 1 or children[:-1] != previous:
                raise RuntimeError("eligible THOL parent did not append exactly one child")
            child = children[-1]
            if (
                child in initial_nodes or child not in graph
                or graph.nodes[child].get("parent_node") != node
                or graph.degree(child) != 0
            ):
                raise RuntimeError("eligible THOL child does not match isolated birth")
            births.append((node, child))
        if set(graph.nodes) != initial_nodes | {child for _, child in births}:
            raise RuntimeError("eligible THOL stage changed unexpected node support")
        if _edge_support(graph) != initial_edges:
            raise RuntimeError("eligible THOL stage changed original edge support")
        return EligibleSelfOrganizationDispatch(eligibility, stage, tuple(births))
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
