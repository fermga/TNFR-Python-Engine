"""Actual Mutation on explicitly selected members of a retained organization.

This consumer binds the current phase-only ZHIR map to actual own temporal and
grammar evidence. It changes no form law, topology, occurrence timing or notion
of collective identity. Full executable state remains in the graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..config.operator_names import BIFURCATION_WINDOW
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_VF
from ..mathematics.unified_numerical import normalize_phase
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._mutation_stage_kernel import mutation_phase_changed
from .grammar_debt import advance_debt, advance_prior_coherence
from .grammar_types import glyph_function_name
from .mutation import Mutation
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
    execute_pointwise_stage,
)
from .retained_organization import (
    _OPTIONS,
    RetainedOrganization,
    _member_payload,
    _protected_state,
    _readmit_retained_organization,
    _signature,
    capture_retained_organization,
)

__all__ = ["RetainedMutationDispatch", "execute_retained_mutation_stage"]

_MUTATION_OPTIONS = _OPTIONS | {"tau", "validate_postconditions"}
_CANONICAL_MUTATION = (
    Mutation.__call__,
    Mutation._execute,
    Mutation.name,
    Mutation.glyph,
)
_HELD_ALIASES = (ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_VF, ALIAS_DNFR)


def _same(left: Any, right: Any) -> bool:
    return proof_stamps_are_identical(
        structural_proof_signature(left), structural_proof_signature(right)
    )


def _targets(record: RetainedOrganization, targets: tuple[Any, ...]) -> None:
    if type(targets) is not tuple or not targets:
        raise TypeError("retained Mutation targets must be a nonempty exact tuple")
    if len(set(targets)) != len(targets):
        raise ValueError("retained Mutation targets must be unique")
    if any(node not in record.members for node in targets):
        raise ValueError("retained Mutation targets must belong to the organization")


def _held_target_state(graph: Any, targets: tuple[Any, ...]) -> Any:
    return _signature(
        graph,
        tuple(
            (
                node,
                tuple(
                    tuple(
                        (key, graph.nodes[node][key])
                        for key in aliases
                        if key in graph.nodes[node]
                    )
                    for aliases in _HELD_ALIASES
                ),
            )
            for node in targets
        ),
    )


def _boundary(record: RetainedOrganization) -> tuple[Any, ...]:
    return (
        record.members,
        record.directed,
        record.multigraph,
        record.incoming_ports,
        record.outgoing_ports,
        record.incident_edges,
    )


@dataclass(frozen=True, slots=True)
class RetainedMutationDispatch:
    """Consistent retained boundaries and actual ordered Mutation observations.

    This summary does not authenticate invocation or certify collective identity.
    Timestamped and legacy trigger provenance stay distinct. A stage's compatible
    acceleration zero need not establish an available three-sample observation.
    Constructor checks do not reconstruct sampled history from opaque retained
    signatures. Actual temporal origin comes from fresh canonical execution;
    manually combining compatible records cannot authenticate that origin.
    """

    before: RetainedOrganization
    after: RetainedOrganization
    targets: tuple[Any, ...]
    stage_result: NetworkStageResult

    def __post_init__(self) -> None:
        for record in (self.before, self.after):
            if (
                type(record) is not RetainedOrganization
                or not record._proof_fields_are_intact()
            ):
                raise ValueError("retained Mutation requires intact observations")
        _targets(self.before, self.targets)
        if self.before._graph_owner is not self.after._graph_owner or not _same(
            _boundary(self.before), _boundary(self.after)
        ):
            raise ValueError(
                "retained Mutation must preserve member identities and ports"
            )
        stage = self.stage_result
        if type(stage) is not NetworkStageResult:
            raise TypeError("retained Mutation requires its canonical stage result")
        stage.__post_init__()
        if (
            stage.operator != _CANONICAL_MUTATION[2]
            or stage.glyph != _CANONICAL_MUTATION[3].value
            or stage.schedule != TWO_PHASE_JACOBI
            or stage.nodes_processed != len(self.targets)
        ):
            raise ValueError("retained Mutation requires an exact two-phase ZHIR stage")
        selected = set(self.targets)
        positions = {node: index for index, node in enumerate(self.before.members)}
        for index, (before, after) in enumerate(
            zip(self.before.member_states, self.after.member_states, strict=True)
        ):
            if not _same(
                (before.epi, before.epi_kind, before.capacity, before.pressure),
                (after.epi, after.epi_kind, after.capacity, after.pressure),
            ):
                raise ValueError(
                    "Mutation must preserve non-phase structural coordinates"
                )
            if before.node not in selected and not _same(
                (_member_payload(before), self.before.member_metadata[index]),
                (_member_payload(after), self.after.member_metadata[index]),
            ):
                raise ValueError(
                    "Mutation must preserve every non-target member's state"
                )
        for index, (node, observed) in enumerate(
            zip(self.targets, stage.mutation_decision_observations, strict=True)
        ):
            position = positions[node]
            before = self.before.member_states[position]
            after = self.after.member_states[position]
            certificate = observed.trigger_certificate
            if (
                observed.target_index != index
                or not _same(observed.node, node)
                or not observed._proof_fields_are_intact()
                or observed.theta_before != float(normalize_phase(before.phase))
                or observed.theta_after != after.phase
                or not mutation_phase_changed(before.phase, after.phase)
                or certificate.current_epi != before.epi
                or certificate.nu_f != before.capacity
                or certificate.delta_nfr != before.pressure
                or (
                    "" if observed.epi_kind_before is None else observed.epi_kind_before
                )
                != before.epi_kind
                or certificate.source not in dict(before.temporal_context)
            ):
                raise ValueError(
                    "Mutation decision does not match its retained target boundary"
                )
            prior = before.grammar
            current = after.grammar
            expected_history = (*prior.history, _CANONICAL_MUTATION[3].value)
            recent_history = tuple(
                glyph_function_name(item)
                for item in prior.history[-BIFURCATION_WINDOW:]
            )
            if (
                observed.operator_step != prior.operator_step + 1
                or current.operator_step != observed.operator_step
                or current.u2_debt
                != advance_debt(prior.u2_debt, _CANONICAL_MUTATION[3])
                or current.prior_coherence
                != advance_prior_coherence(
                    prior.prior_coherence, _CANONICAL_MUTATION[3]
                )
                or len(current.history) > len(expected_history)
                or (
                    current.history
                    and current.history != expected_history[-len(current.history) :]
                )
                or observed.recent_history != recent_history
            ):
                raise ValueError(
                    "Mutation decision must use and advance its own grammar context"
                )


def _canonical_mutation() -> Mutation:
    call, execute, name, glyph = _CANONICAL_MUTATION
    operator = Mutation()
    if (
        type(operator) is not Mutation
        or Mutation.__call__ is not call
        or Mutation._execute is not execute
        or operator.name != name
        or operator.glyph is not glyph
        or any(
            key in vars(operator) for key in ("__call__", "_execute", "name", "glyph")
        )
    ):
        raise TypeError("retained Mutation requires unmodified canonical bindings")
    return operator


def execute_retained_mutation_stage(
    graph: Any,
    record: RetainedOrganization,
    targets: tuple[Any, ...],
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> RetainedMutationDispatch:
    """Apply actual ZHIR once to an explicit retained target subset, atomically.

    Each target must supply its own admitted temporal evidence and grammar.
    Runtime gates remain authoritative: legacy unit-step evidence is not given
    timestamp provenance. No history, context, callback or fallback is supplied.
    Full support, non-target state and raw target non-phase coordinates remain
    intact. Actual target lifecycle metadata and graph diagnostics may change.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        operator = _canonical_mutation()
        if execution_kwargs is not None and not isinstance(execution_kwargs, Mapping):
            raise TypeError("execution_kwargs must be a mapping or None")
        options = {} if execution_kwargs is None else dict(execution_kwargs)
        if any(type(key) is not str or key not in _MUTATION_OPTIONS for key in options):
            raise ValueError("unsupported retained Mutation execution option")
        transaction.restore(graph)
        before = _readmit_retained_organization(graph, record)
        if before is None:
            raise ValueError(
                "retained organization is foreign, stale, invalid or unsealed"
            )
        _targets(before, targets)
        protected = _protected_state(graph, targets, preserve_member_phase=False)
        held = _held_target_state(graph, targets)
        stage = execute_pointwise_stage(
            graph,
            operator,
            targets,
            transaction_snapshot=transaction,
            require_exact_operator=True,
            **options,
        )
        after = capture_retained_organization(graph, before.members)
        if not proof_stamps_are_identical(
            protected, _protected_state(graph, targets, preserve_member_phase=False)
        ) or not proof_stamps_are_identical(held, _held_target_state(graph, targets)):
            raise RuntimeError("retained Mutation changed protected state or ports")
        return RetainedMutationDispatch(before, after, targets, stage)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
