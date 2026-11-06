"""Atomic opt-in Mutation driven by an actually executed operator schedule.

This supplied hybrid trigger is separate from continuous-history Mutation.
Strict requests commit source and all requested phase actions together. The
conditional variant retains a completed source when no candidate is eligible.
Returned observations are historical reports, never reusable permissions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .._exact_time import represented_fraction
from ..alias import get_attr
from ..config.operator_names import BIFURCATION_WINDOW
from ..constants.aliases import ALIAS_THETA
from ..types import Glyph
from ..utils._structural_signature import proof_stamps_are_identical
from ._argument_validation import validate_common_execution_arguments
from ._operator_growth_trigger import _operator_growth_source_context
from .event_runtime import OperatorEventExecutionResult
from .event_timing import OperatorEventSchedule, PhysicalFlowPartition
from .grammar_debt import advance_debt, advance_prior_coherence, read_grammar_state
from .grammar_dynamics import _check_violations
from .grammar_types import glyph_function_name
from .network_stage import (
    GraphTransactionSnapshot,
    NetworkStageResult,
    execute_pointwise_stage,
)
from .operator_growth import OperatorEPIGrowthObservation
from .retained_mutation import (
    _MUTATION_OPTIONS,
    _canonical_mutation,
    _held_target_state,
)
from .retained_organization import _protected_state, _signature

__all__ = (
    "OperatorGrowthMutationDispatch",
    "OperatorGrowthMutationCandidate",
    "EligibleOperatorGrowthMutationDispatch",
    "execute_operator_growth_mutation",
    "execute_eligible_operator_growth_mutation",
)

_WATERMARK = "_operator_growth_mutation_end"
_TEMPORAL_KEYS = ("epi_time_history", "epi_history", "_epi_history")


@dataclass(frozen=True, slots=True)
class OperatorGrowthMutationDispatch:
    """Past source and accepted phase stage; construction grants no permission."""

    source_result: OperatorEventExecutionResult
    growth: OperatorEPIGrowthObservation
    targets: tuple[Any, ...]
    stage_result: NetworkStageResult


@dataclass(frozen=True, slots=True)
class OperatorGrowthMutationCandidate:
    """One own-state selection observation, not a phase-action permission.

    Ordinary predicate failures are retained together. Action factors and the
    actual circular phase effect are admitted later by the selected stage.
    """

    node: Any
    exact_rate: Fraction
    exact_xi: Fraction
    nu_f: float
    minimum_nu_f: float
    previous_consumed_end: Fraction | None
    grammar_violations: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def selected(self) -> bool:
        """Whether all declared source/state/context predicates passed."""

        return not self.reasons


@dataclass(frozen=True, slots=True)
class EligibleOperatorGrowthMutationDispatch:
    """Completed source with selected ZHIR actions or an explicit abstention.

    No stage means no ZHIR occurred; the source still executed. Construction
    does not authenticate execution, select a source word or install a clock.
    """

    source_result: OperatorEventExecutionResult
    growth: OperatorEPIGrowthObservation
    candidates: tuple[Any, ...]
    decisions: tuple[OperatorGrowthMutationCandidate, ...]
    selected_targets: tuple[Any, ...]
    stage_result: NetworkStageResult | None

    @property
    def executed(self) -> bool:
        """Whether the source was followed by an actual committed ZHIR stage."""

        return self.stage_result is not None


def _watermarks(graph: Any, targets: tuple[Any, ...]) -> tuple[Fraction | None, ...]:
    values = []
    for node in targets:
        data = graph.nodes[node]
        if _WATERMARK not in data:
            values.append(None)
            continue
        value = data[_WATERMARK]
        if type(value) is not Fraction:
            raise TypeError(f"{_WATERMARK} must be an exact Fraction")
        # Normalize rational components before arithmetic (including fixed-width
        # integer components); a malformed present receipt never means absence.
        values.append(Fraction(int(value.numerator), int(value.denominator)))
    return tuple(values)


def _temporal_state(graph: Any, targets: tuple[Any, ...]) -> Any:
    return _signature(
        graph,
        tuple(
            (
                node,
                tuple(
                    (key, graph.nodes[node][key])
                    for key in _TEMPORAL_KEYS
                    if key in graph.nodes[node]
                ),
            )
            for node in targets
        ),
    )


def _prepare_growth_request(graph, schedule, targets, mutation_kwargs, transaction):
    """Admit the shared strict/conditional request without running a source."""

    from . import _validated_execution_window

    operator = _canonical_mutation()
    if type(schedule) is not OperatorEventSchedule:
        raise TypeError("schedule must be an OperatorEventSchedule")
    schedule.__post_init__()
    if schedule.exact_total_flow_duration <= 0:
        raise ValueError("operator growth Mutation needs a positive window")
    if type(targets) is not tuple or not targets:
        raise TypeError("Mutation targets must be a nonempty exact tuple")
    if len(set(targets)) != len(targets):
        raise ValueError("Mutation targets must be unique")
    if any(node not in graph for node in targets):
        raise ValueError("Mutation targets must belong to the source graph")
    if mutation_kwargs is not None and not isinstance(mutation_kwargs, Mapping):
        raise TypeError("mutation_kwargs must be a mapping or None")
    options = {} if mutation_kwargs is None else dict(mutation_kwargs)
    if any(type(key) is not str or key not in _MUTATION_OPTIONS for key in options):
        raise ValueError("unsupported operator growth Mutation option")
    # Match retained-stage admission: materializing a supplied mapping must
    # not become an unrecorded state transformation before the real source.
    transaction.restore(graph)
    history_window = _validated_execution_window(graph, options.get("window"))
    options["window"] = history_window
    before_watermarks = _watermarks(graph, targets)
    return operator, options, before_watermarks


def _execute_growth_stage(graph, source, operator, targets, options, before_watermarks):
    """One protected canonical phase commit shared by both public consumers."""

    current_watermarks = _watermarks(graph, targets)
    physical_intervals = tuple(
        source.response_interval_for(graph, node) for node in targets
    )
    if any(
        item is not None and item.previously_consumed for item in physical_intervals
    ):
        raise ValueError("mutation_interval_already_consumed")
    if current_watermarks != before_watermarks:
        raise RuntimeError("Mutation consumption memory changed before stage")
    if any(
        end is not None and source.observation.exact_start_time < end
        for end in current_watermarks
    ):
        raise ValueError("operator growth Mutation window overlaps consumed time")
    history_window = options["window"]
    protected = _protected_state(graph, targets, preserve_member_phase=False)
    held = _held_target_state(graph, targets)
    temporal = _temporal_state(graph, targets)
    grammar = tuple(read_grammar_state(graph.nodes[node]) for node in targets)
    clock = represented_fraction(graph.graph["_t"], "runtime clock")
    stage = execute_pointwise_stage(
        graph,
        operator,
        targets,
        require_exact_operator=True,
        _operator_growth_context=source,
        **options,
    )
    if not proof_stamps_are_identical(
        protected,
        _protected_state(graph, targets, preserve_member_phase=False),
    ) or not proof_stamps_are_identical(held, _held_target_state(graph, targets)):
        raise RuntimeError("hybrid Mutation changed protected state or ports")
    if represented_fraction(graph.graph["_t"], "runtime clock") != clock:
        raise RuntimeError("hybrid Mutation changed the source clock")
    if _watermarks(graph, targets) != before_watermarks:
        raise RuntimeError("Mutation consumption memory changed during stage")
    if not proof_stamps_are_identical(temporal, _temporal_state(graph, targets)):
        raise RuntimeError("hybrid Mutation changed target temporal evidence")
    for node, prior, decision in zip(
        targets, grammar, stage.mutation_decision_observations, strict=True
    ):
        if (
            not decision._proof_fields_are_intact()
            or decision.node != node
            or get_attr(graph.nodes[node], ALIAS_THETA) != decision.theta_after
        ):
            raise RuntimeError("hybrid Mutation lost its phase decision")
        current = read_grammar_state(graph.nodes[node])
        unbounded = (*prior.history, Glyph.ZHIR.value)
        expected = unbounded[-history_window:] if history_window else ()
        if (
            current.operator_step != prior.operator_step + 1
            or decision.operator_step != current.operator_step
            or current.u2_debt != advance_debt(prior.u2_debt, Glyph.ZHIR)
            or current.prior_coherence
            != advance_prior_coherence(prior.prior_coherence, Glyph.ZHIR)
            or current.history != expected
            or decision.recent_history
            != tuple(
                glyph_function_name(glyph)
                for glyph in prior.history[-BIFURCATION_WINDOW:]
            )
        ):
            raise RuntimeError("hybrid Mutation lost its grammar transition")
        certificate = decision.trigger_certificate
        expected_gate = {
            "_zhir_gate_depi_dt": certificate.observed_depi_dt,
            "_zhir_gate_xi": certificate.xi,
            "_zhir_gate_history_key": certificate.source,
            "_zhir_gate_sample_interval": certificate.evidence.sample_interval,
            "_zhir_gate_exact_depi_dt": certificate.exact_observed_rate,
            "_zhir_gate_exact_xi": certificate.exact_xi,
            "_zhir_operator_step": decision.operator_step,
        }
        actual_gate = {key: graph.nodes[node].get(key) for key in expected_gate}
        if not proof_stamps_are_identical(
            _signature(graph, expected_gate), _signature(graph, actual_gate)
        ):
            raise RuntimeError("hybrid Mutation lost its trigger telemetry")
    source.readmit_source()
    from ..dynamics._mutation_occurrence import commit_native_mutation_interval

    for node, admission in zip(targets, physical_intervals, strict=True):
        if admission is not None:
            commit_native_mutation_interval(graph.nodes[node], admission)
    for node in targets:
        graph.nodes[node][_WATERMARK] = source.observation.exact_end_time
    return stage


def execute_operator_growth_mutation(
    graph: Any,
    schedule: OperatorEventSchedule,
    targets: tuple[Any, ...],
    *,
    context: Mapping[str, Any] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    mutation_kwargs: Mapping[str, Any] | None = None,
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = (),
) -> OperatorGrowthMutationDispatch:
    """Execute a real source and an immediate canonical ZHIR stage atomically.

    The source acts on the schedule's complete initial population; Mutation acts
    only on ``targets``. Each target must have exact positive net growth above
    the configured threshold, active capacity and its own admitted grammar.
    The full source window uses (s,t] in the declared runtime clock. Per-target
    windows must not overlap earlier successful windows of this variant.

    No source result or observation can be supplied as activation evidence.
    Ordinary Mutation, retained-member Mutation and native occurrence remain
    unchanged. External callback side effects and manual graph/ledger rewrites
    are outside the graph-owned transaction and nonreuse guarantee.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        operator, options, before_watermarks = _prepare_growth_request(
            graph, schedule, targets, mutation_kwargs, transaction
        )
        if any(
            end is not None and schedule.exact_start_time < end
            for end in before_watermarks
        ):
            raise ValueError("operator growth Mutation window overlaps consumed time")
        with _operator_growth_source_context(
            graph,
            schedule,
            targets,
            context=context,
            method=method,
            n_jobs=n_jobs,
            physical_flow_partitions=physical_flow_partitions,
        ) as source:
            if _watermarks(graph, targets) != before_watermarks:
                raise RuntimeError("Mutation consumption memory changed during source")
            stage = _execute_growth_stage(
                graph, source, operator, targets, options, before_watermarks
            )
            return OperatorGrowthMutationDispatch(
                source.result, source.observation, targets, stage
            )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


def _select_growth_candidates(
    source, candidates, before_watermarks, *, native_mutated=()
):
    """Observe all predicates once without a fallback or an action attempt."""

    decisions = []
    with source.probe_candidates() as (snapshot, inputs):
        for node, values, previous in zip(
            candidates, inputs, before_watermarks, strict=True
        ):
            # Reuse the same default context window as actual incremental
            # grammar. The execution history-retention window is separate.
            allowed, violations = _check_violations(snapshot, node, "ZHIR")
            reasons = []
            if values.nu_f <= 0.0:
                reasons.append("inactive_structural_capacity")
            if values.nu_f < values.minimum_nu_f:
                reasons.append("below_configured_capacity_floor")
            if not values.exact_rate > values.exact_xi:
                reasons.append("observed_threshold_not_crossed")
            if not allowed:
                reasons.append("grammar_rejected")
            if previous is not None and source.observation.exact_start_time < previous:
                reasons.append("window_overlaps_consumed_time")
            if any(node is actual for actual in native_mutated):
                reasons.append("native_mutation_precedence")
            else:
                physical = source.response_interval_for(snapshot, node)
                if physical is not None and physical.previously_consumed:
                    reasons.append("mutation_interval_already_consumed")
            decisions.append(
                OperatorGrowthMutationCandidate(
                    node=node,
                    exact_rate=values.exact_rate,
                    exact_xi=values.exact_xi,
                    nu_f=values.nu_f,
                    minimum_nu_f=values.minimum_nu_f,
                    previous_consumed_end=previous,
                    grammar_violations=tuple(
                        violation.rule
                        for violation in violations
                        if violation.severity == "error"
                    ),
                    reasons=tuple(reasons),
                )
            )
    return tuple(decisions)


def execute_eligible_operator_growth_mutation(
    graph: Any,
    schedule: OperatorEventSchedule,
    candidates: tuple[Any, ...] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    mutation_kwargs: Mapping[str, Any] | None = None,
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = (),
) -> EligibleOperatorGrowthMutationDispatch:
    """Execute one real source and select its eligible Mutation targets.

    ``None`` considers every initial node in graph order. An explicit candidate
    set must be a nonempty unique tuple of existing nodes. Selection uses each
    candidate's own exact growth, capacity/floor, grammar and consumed window.
    It neither chooses a source word nor installs an autonomous invocation clock.

    Normal noncrossing, inactive/below-floor capacity, grammar rejection or
    overlap yields an explicit candidate abstention. If none are selected, the
    completed source stays committed and no empty stage, ZHIR occurrence or
    receipt is created. Malformed consumed inputs and errors in an actually
    selected phase stage roll back the entire graph-owned source and stage.
    Unselected phase-action parameters are not certified by this predicate.
    External callback effects remain outside graph-owned rollback.
    """

    transaction = GraphTransactionSnapshot(graph)
    try:
        candidates = tuple(graph.nodes) if candidates is None else candidates
        operator, options, before_watermarks = _prepare_growth_request(
            graph, schedule, candidates, mutation_kwargs, transaction
        )
        # These request flags are consumed even if no action is eventually
        # selected. A malformed request is not a state-dependent abstention.
        validate_common_execution_arguments(
            graph.graph, options, operator=operator.name
        )
        with _operator_growth_source_context(
            graph,
            schedule,
            candidates,
            context=context,
            method=method,
            n_jobs=n_jobs,
            physical_flow_partitions=physical_flow_partitions,
        ) as source:
            if _watermarks(graph, candidates) != before_watermarks:
                raise RuntimeError("Mutation consumption memory changed during source")
            decisions = _select_growth_candidates(source, candidates, before_watermarks)
            selected = tuple(item.node for item in decisions if item.selected)
            stage = None
            if selected:
                retained_watermarks = tuple(
                    item.previous_consumed_end for item in decisions if item.selected
                )
                stage = _execute_growth_stage(
                    graph, source, operator, selected, options, retained_watermarks
                )
            else:
                # No empty execute_pointwise_stage: even an empty stage would
                # add schedule bookkeeping for an action that did not occur.
                source.validate_live_before_commit(graph)
            return EligibleOperatorGrowthMutationDispatch(
                source.result,
                source.observation,
                candidates,
                decisions,
                selected,
                stage,
            )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
