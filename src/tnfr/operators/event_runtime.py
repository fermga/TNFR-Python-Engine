"""Atomic runtime binding for finite canonical operator-event schedules.

A schedule alternates configured nodal-flow intervals with zero-duration
canonical operator jumps. Exact rational offsets remain the scheduling
authority; the existing binary64 runtime clock must represent every positive
boundary without collapse. This module coordinates existing integrator and
network-stage contracts. It does not certify numerical accuracy, infer an
operator duration, or adapt grammar U2/U4.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import TYPE_CHECKING, Any

import networkx as nx

from ..dynamics.integrators import (
    DefaultIntegrator as _CanonicalDefaultIntegrator,
)
from ..dynamics.integrators import (
    prepare_integration_params as _canonical_prepare_integration_params,
)
from ..errors import TNFRValueError
from ..types import Glyph
from .event_timing import (
    OperatorEventSchedule,
    ScheduledOperatorEvent,
    StructuralFlowInterval,
    diagnose_operator_event_runtime_clock,
)
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
)

if TYPE_CHECKING:
    from ..physics.runtime_flow_stability import (
        NodalFlowIntervalCertificate,
        NodalFlowStateSnapshot,
    )


_HYBRID_EVENT_LOG = "hybrid_event_log"
_FLOW_PROVENANCE = (
    "tnfr.operators.event_runtime.execute_operator_event_schedule"
)
_NODAL_FLOW_INPUTS = "live_nu_f_and_delta_nfr_at_each_interval_start"
_CANONICAL_DEFAULT_INTEGRATE = _CanonicalDefaultIntegrator.integrate
_FLOW_SCOPE = (
    "configured nodal EPI integrator over each declared positive interval, "
    "with current stored fields held according to that integrator and pressure "
    "refresh confined to explicit operator-stage callbacks; solver accuracy "
    "and adaptive U2/U4 behavior are not certified"
)


@dataclass(frozen=True, slots=True)
class _FlowRuntimeMetadata:
    """Trusted metadata read from the actual interval execution path."""

    integrator_name: str
    integrator_provenance_certified: bool
    resolved_method: str | None
    resolved_substeps: int | None
    gamma_is_none: bool | None
    extended_dynamics_requested: bool


@dataclass(frozen=True, slots=True)
class ExecutedOperatorEvent:
    """One scheduled zero-duration jump committed by the canonical stage."""

    event_index: int
    cycle_index: int
    word_position: int
    operator_name: str
    glyph: Glyph
    event_time: float
    event_offset: Fraction
    exact_event_time: Fraction
    stage_schedule: str
    nodes_processed: int
    zero_duration: bool = field(default=True, init=False)
    history_channel: str = field(default=_HYBRID_EVENT_LOG, init=False)
    feeds_epi_time_history: bool = field(default=False, init=False)

    @classmethod
    def from_stage(
        cls,
        event: ScheduledOperatorEvent,
        result: NetworkStageResult,
    ) -> "ExecutedOperatorEvent":
        """Bind one immutable schedule event to its accepted stage result."""

        return cls(
            event_index=event.event_index,
            cycle_index=event.cycle_index,
            word_position=event.word_position,
            operator_name=event.operator_name,
            glyph=event.glyph,
            event_time=event.event_time,
            event_offset=event.event_offset,
            exact_event_time=event.exact_event_time,
            stage_schedule=result.schedule,
            nodes_processed=result.nodes_processed,
        )

    def as_record(self) -> dict[str, Any]:
        """Return a detached append-only graph telemetry record."""

        return {
            "event_index": self.event_index,
            "cycle_index": self.cycle_index,
            "word_position": self.word_position,
            "operator_name": self.operator_name,
            "glyph": self.glyph.value,
            "event_time": self.event_time,
            "event_offset": self.event_offset,
            "exact_event_time": self.exact_event_time,
            "stage_schedule": self.stage_schedule,
            "nodes_processed": self.nodes_processed,
            "zero_duration": self.zero_duration,
            "history_channel": self.history_channel,
            "feeds_epi_time_history": self.feeds_epi_time_history,
        }


@dataclass(frozen=True, slots=True)
class ExecutedNodalFlowInterval:
    """Runtime-bound evidence for one executed positive flow interval.

    The wrapper supplies integrator provenance from the execution site. The
    nested certificate remains a standalone endpoint result and therefore
    deliberately leaves its own ``integrator_provenance_certified`` flag
    false. Solver accuracy and future or repeated schedules are outside both
    scopes.
    """

    interval: StructuralFlowInterval
    certificate: NodalFlowIntervalCertificate | None
    abstention_reason: str | None
    integrator_name: str
    integrator_provenance_certified: bool
    resolved_method: str | None
    resolved_substeps: int | None
    gamma_is_none: bool | None
    clipping_applied: bool | None
    extended_dynamics_requested: bool
    solver_accuracy_certified: bool = field(default=False, init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=False,
        init=False,
    )

    @property
    def runtime_bound_binary64_interval_identified(self) -> bool:
        """Whether the observed endpoint has trusted built-in provenance."""

        return bool(
            self.integrator_provenance_certified
            and self.certificate is not None
            and self.certificate.binary64_runtime_interval_identified
        )

    @property
    def runtime_bound_exact_affine_map_identified(self) -> bool:
        """Whether this execution realizes the certified rational Euler map."""

        return bool(
            self.integrator_provenance_certified
            and self.certificate is not None
            and self.certificate.explicit_euler_map_identified
        )

    @property
    def runtime_bound_global_disagreement_contraction_certified(self) -> bool:
        """Whether that identified map contracts global disagreement."""

        return bool(
            self.runtime_bound_exact_affine_map_identified
            and self.certificate is not None
            and self.certificate.global_disagreement_contraction_certified
        )


@dataclass(frozen=True, slots=True)
class OperatorEventExecutionResult:
    """Immutable evidence that one complete finite schedule committed.

    pressure_refresh_callback_invocations counts explicit stage callbacks that
    returned successfully during this schedule.
    """

    schedule: OperatorEventSchedule
    target_nodes: tuple[Any, ...]
    flow_interval_indices: tuple[int, ...]
    positive_flow_interval_indices: tuple[int, ...]
    events: tuple[ExecutedOperatorEvent, ...]
    final_time: float
    integrator_name: str | None
    pressure_refresh_callback_invocations: int
    flow_certification_requested: bool = False
    flow_interval_evidence: tuple[ExecutedNodalFlowInterval, ...] = ()
    runtime_clock_checked: bool = field(default=True, init=False)
    flow_provenance: str = field(default=_FLOW_PROVENANCE, init=False)
    nodal_flow_inputs: str = field(default=_NODAL_FLOW_INPUTS, init=False)
    whole_schedule_graph_state_atomic: bool = field(default=True, init=False)
    operator_jumps_have_zero_duration: bool = field(default=True, init=False)
    solver_accuracy_certified: bool = field(default=False, init=False)
    adaptive_u2_u4_policy: bool = field(default=False, init=False)
    external_side_effects_rolled_back: bool = field(default=False, init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=False,
        init=False,
    )
    flow_scope: str = field(default=_FLOW_SCOPE, init=False)

    def _all_positive_flow_intervals(self, attribute: str) -> bool | None:
        if not self.flow_certification_requested:
            return None
        if len(self.flow_interval_evidence) != len(
            self.positive_flow_interval_indices
        ):
            return False
        return all(
            bool(getattr(evidence, attribute))
            for evidence in self.flow_interval_evidence
        )

    @property
    def all_positive_flow_intervals_binary64_identified(self) -> bool | None:
        """Aggregate trusted binary64 realization, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_binary64_interval_identified"
        )

    @property
    def all_positive_flow_intervals_exact_affine(self) -> bool | None:
        """Aggregate exact affine identification, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_exact_affine_map_identified"
        )

    @property
    def all_positive_flow_intervals_contracting(self) -> bool | None:
        """Aggregate exact contraction result, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_global_disagreement_contraction_certified"
        )


def _require_graph(graph: Any) -> nx.Graph:
    """Return a supported NetworkX graph before any runtime inspection."""

    if not isinstance(
        graph,
        (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
    ):
        raise TypeError("graph must be a NetworkX graph")
    return graph


def _require_runtime_clock(
    graph: nx.Graph,
    expected: float,
    *,
    boundary: str,
) -> float:
    """Require one exact binary64 runtime boundary without resetting it."""

    if "_t" not in graph.graph:
        raise TNFRValueError(
            "Operator-event execution requires an existing binary64 runtime clock.",
            context={"boundary": boundary, "reason": "missing_graph_time"},
        )
    current = graph.graph["_t"]
    if type(current) is not float or not math.isfinite(current):
        raise TNFRValueError(
            "Operator-event execution requires a finite binary64 runtime clock.",
            context={
                "boundary": boundary,
                "runtime_time": repr(current),
                "runtime_time_type": type(current).__name__,
            },
        )
    if current != expected:
        raise TNFRValueError(
            "The live runtime clock does not match the scheduled boundary.",
            context={
                "boundary": boundary,
                "runtime_time": current,
                "scheduled_time": expected,
            },
        )
    return current


def _validate_schedule_clock(schedule: OperatorEventSchedule) -> None:
    """Reject schedules the current binary64 runtime cannot represent."""

    diagnostic = diagnose_operator_event_runtime_clock(schedule)
    if diagnostic.clock_binding_ready:
        return
    raise TNFRValueError(
        "The operator-event schedule cannot bind to the current runtime clock.",
        context={
            "collapsed_positive_interval_indices": (
                diagnostic.collapsed_positive_interval_indices
            ),
            "nonadditive_positive_interval_indices": (
                diagnostic.nonadditive_positive_interval_indices
            ),
            "zhir_event_indices_without_positive_preflow": (
                diagnostic.zhir_event_indices_without_positive_preflow
            ),
            "zhir_event_indices_with_collapsed_preflow": (
                diagnostic.zhir_event_indices_with_collapsed_preflow
            ),
        },
        suggestion=(
            "Choose a representable time origin and positive pre-flow for every "
            "Mutation event."
        ),
    )


def _validate_hybrid_log(graph: nx.Graph) -> None:
    """Require the graph event channel to be appendable before any write."""

    if _HYBRID_EVENT_LOG in graph.graph and not isinstance(
        graph.graph[_HYBRID_EVENT_LOG], list
    ):
        raise TNFRValueError(
            "hybrid_event_log must be a list when already configured.",
            context={
                "history_channel": _HYBRID_EVENT_LOG,
                "value_type": type(graph.graph[_HYBRID_EVENT_LOG]).__name__,
            },
        )


def _prepare_word(
    schedule: OperatorEventSchedule,
    context: Mapping[str, Any] | None,
) -> tuple[tuple[Any, ...], Any | None]:
    """Validate one canonical word without stale Mutation preflight."""

    if not schedule.operator_names:
        return (), None

    from ..validation import validate_sequence
    from .grammar_execution import ValidatedSequence
    from .registry import get_operator_class

    operators = tuple(
        get_operator_class(name)() for name in schedule.operator_names
    )
    validation_context = None if context is None else dict(context)
    outcome = validate_sequence(
        list(schedule.operator_names),
        context=validation_context,
    )
    if not outcome.passed:
        raise TNFRValueError(
            "Invalid operator-event sequence: "
            + outcome.summary.get("message", "validation failed"),
            context={
                "sequence": schedule.operator_names,
                "outcome": outcome.summary,
            },
        )
    return operators, ValidatedSequence(
        operators,
        context=validation_context,
    )


def _flow_runtime_metadata(
    graph: nx.Graph,
    interval: StructuralFlowInterval,
    integrator: Any,
    *,
    method: str | None,
) -> _FlowRuntimeMetadata:
    """Derive execution metadata without invoking the integrator twice."""

    from ..gamma import _get_gamma_spec

    integrator_name = type(integrator).__qualname__
    bound_integrate = getattr(integrator, "integrate", None)
    instance_attributes = getattr(integrator, "__dict__", {})
    provenance = bool(
        type(integrator) is _CanonicalDefaultIntegrator
        and "integrate" not in instance_attributes
        and getattr(bound_integrate, "__self__", None) is integrator
        and getattr(bound_integrate, "__func__", None)
        is _CANONICAL_DEFAULT_INTEGRATE
        and _CanonicalDefaultIntegrator.integrate
        is _CANONICAL_DEFAULT_INTEGRATE
    )
    extended_requested = bool(
        graph.graph.get("use_extended_dynamics", False)
    )
    if not provenance:
        return _FlowRuntimeMetadata(
            integrator_name=integrator_name,
            integrator_provenance_certified=False,
            resolved_method=None,
            resolved_substeps=None,
            gamma_is_none=None,
            extended_dynamics_requested=extended_requested,
        )

    _, substeps, _, resolved_method = _canonical_prepare_integration_params(
        graph,
        interval.duration,
        interval.start_time,
        method,
    )
    gamma_spec = _get_gamma_spec(graph)
    gamma_is_none = bool(
        isinstance(gamma_spec, Mapping)
        and gamma_spec.get("type", "none") == "none"
    )
    return _FlowRuntimeMetadata(
        integrator_name=integrator_name,
        integrator_provenance_certified=True,
        resolved_method=resolved_method,
        resolved_substeps=substeps,
        gamma_is_none=gamma_is_none,
        extended_dynamics_requested=extended_requested,
    )


def _capture_interval_endpoint(
    graph: nx.Graph,
) -> tuple[NodalFlowStateSnapshot | None, bool]:
    """Capture a detached endpoint while treating unsupported state as abstention."""

    from ..physics.runtime_flow_stability import capture_nodal_flow_state

    try:
        return capture_nodal_flow_state(graph), True
    except (TypeError, ValueError, nx.NetworkXException):
        return None, False


def _clipping_intervened(
    left: NodalFlowStateSnapshot,
    right: NodalFlowStateSnapshot,
    interval: StructuralFlowInterval,
    metadata: _FlowRuntimeMetadata,
) -> bool | None:
    """Detect one-step clipping by comparison with the unclipped Euler update."""

    if not (
        metadata.integrator_provenance_certified
        and metadata.resolved_method == "euler"
        and metadata.resolved_substeps == 1
        and metadata.gamma_is_none is True
        and left.nodes == right.nodes
    ):
        return None

    from ..mathematics.unified_numerical import np

    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            rate = np.multiply(
                np.asarray(left.nu_f, dtype=float),
                np.asarray(left.delta_nfr, dtype=float),
            )
            increment = np.multiply(interval.duration, rate)
            replay = np.add(
                np.asarray(left.epi, dtype=float),
                increment,
            )
    except (FloatingPointError, TypeError, ValueError, OverflowError):
        return None
    if not bool(np.all(np.isfinite(replay))):
        return None
    return any(
        float(expected) != observed
        for expected, observed in zip(replay, right.epi)
    )


def _execute_flow_interval(
    graph: nx.Graph,
    interval: StructuralFlowInterval,
    integrator: Any,
    *,
    method: str | None,
    n_jobs: int | None,
    include_flow_certificate: bool,
) -> ExecutedNodalFlowInterval | None:
    """Advance one positive interval and optionally bind endpoint evidence."""

    from ..dynamics.runtime import _record_mutation_flow_boundary

    _require_runtime_clock(
        graph,
        interval.start_time,
        boundary=f"interval[{interval.index}].start",
    )
    if interval.exact_duration == 0:
        return None

    metadata = (
        _flow_runtime_metadata(
            graph,
            interval,
            integrator,
            method=method,
        )
        if include_flow_certificate
        else None
    )
    _record_mutation_flow_boundary(graph)
    if include_flow_certificate:
        left, left_captured = _capture_interval_endpoint(graph)
    else:
        left, left_captured = None, False

    integrator.integrate(
        graph,
        dt=interval.duration,
        t=interval.start_time,
        method=method,
        n_jobs=n_jobs,
    )

    _require_runtime_clock(
        graph,
        interval.end_time,
        boundary=f"interval[{interval.index}].end",
    )
    if include_flow_certificate:
        right, right_captured = _capture_interval_endpoint(graph)
    else:
        right, right_captured = None, False
    _record_mutation_flow_boundary(graph)

    if not include_flow_certificate:
        return None
    if metadata is None:
        raise RuntimeError("flow certification metadata was not prepared")
    if not left_captured or not right_captured:
        if not left_captured and not right_captured:
            reason = "left_and_right_state_capture_failed"
        elif not left_captured:
            reason = "left_state_capture_failed"
        else:
            reason = "right_state_capture_failed"
        return ExecutedNodalFlowInterval(
            interval=interval,
            certificate=None,
            abstention_reason=reason,
            integrator_name=metadata.integrator_name,
            integrator_provenance_certified=(
                metadata.integrator_provenance_certified
            ),
            resolved_method=metadata.resolved_method,
            resolved_substeps=metadata.resolved_substeps,
            gamma_is_none=metadata.gamma_is_none,
            clipping_applied=None,
            extended_dynamics_requested=(
                metadata.extended_dynamics_requested
            ),
        )
    if left is None or right is None:
        raise RuntimeError("captured flow endpoint is unexpectedly absent")

    clipping_applied = _clipping_intervened(
        left,
        right,
        interval,
        metadata,
    )
    from ..physics.runtime_flow_stability import (
        certify_observed_nodal_flow_interval,
    )

    certificate = certify_observed_nodal_flow_interval(
        left,
        right,
        duration=interval.duration,
        integrator_name=(
            "DefaultIntegrator"
            if metadata.integrator_provenance_certified
            else metadata.integrator_name
        ),
        method=metadata.resolved_method,
        substeps=metadata.resolved_substeps,
        gamma_is_none=metadata.gamma_is_none,
        clipping_applied=clipping_applied,
        extended_dynamics_requested=(
            metadata.extended_dynamics_requested
        ),
    )
    return ExecutedNodalFlowInterval(
        interval=interval,
        certificate=certificate,
        abstention_reason=None,
        integrator_name=metadata.integrator_name,
        integrator_provenance_certified=(
            metadata.integrator_provenance_certified
        ),
        resolved_method=metadata.resolved_method,
        resolved_substeps=metadata.resolved_substeps,
        gamma_is_none=metadata.gamma_is_none,
        clipping_applied=clipping_applied,
        extended_dynamics_requested=metadata.extended_dynamics_requested,
    )


def _require_exact_stage(
    event: ScheduledOperatorEvent,
    result: NetworkStageResult,
    *,
    target_count: int,
) -> None:
    """Reject a replacement, override or partial target realization."""

    if (
        result.operator != event.operator_name
        or result.glyph != event.glyph.value
        or result.schedule != TWO_PHASE_JACOBI
        or result.nodes_processed != target_count
    ):
        raise TNFRValueError(
            "The scheduled operator event was not realized by its exact "
            "canonical all-target stage.",
            context={
                "event_index": event.event_index,
                "operator_name": event.operator_name,
                "glyph": event.glyph.value,
                "observed_operator": result.operator,
                "observed_glyph": result.glyph,
                "observed_schedule": result.schedule,
                "observed_nodes_processed": result.nodes_processed,
                "expected_nodes_processed": target_count,
            },
        )


def execute_operator_event_schedule(
    graph: nx.Graph,
    schedule: OperatorEventSchedule,
    *,
    context: Mapping[str, Any] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    suppress_birth_warnings: bool = False,
    include_flow_certificates: bool = False,
) -> OperatorEventExecutionResult:
    """Execute one finite flow/jump schedule inside a whole-schedule rollback.

    The live graph time must already equal schedule.start_time. Every positive
    flow is delegated once to the graph's configured nodal EPI integrator with
    the schedule duration; the resulting binary64 clock must equal the
    scheduled endpoint. Operator jumps use the same fixed initial target tuple
    and canonical dispatcher as ordinary network words. Any failure restores
    graph-owned state, topology, histories, event telemetry and runtime caches.
    A secondary rollback fault is attached to the primary failure instead of
    replacing it. Warnings and effects emitted by external integrators,
    callbacks, monitors or resources remain outside that graph transaction.

    Event timestamps identify zero-duration jumps in hybrid_event_log. Boundary
    sampling may retain the same physical coordinate as the right endpoint of
    one flow and the left endpoint of the next, but a same-time EPI jump resets
    that node's history and can never become an epi_time_history secant.
    ``include_flow_certificates`` captures detached endpoints around each
    positive interval and returns runtime-bound evidence without writing it to
    graph metadata or claiming solver accuracy or repeated stability.
    """

    graph = _require_graph(graph)
    if type(schedule) is not OperatorEventSchedule:
        raise TypeError("schedule must be an OperatorEventSchedule")
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping or None")
    if method is not None and type(method) is not str:
        raise TypeError("method must be a string or None")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral)
    ):
        raise TypeError("n_jobs must be an integer or None")
    if type(suppress_birth_warnings) is not bool:
        raise TypeError("suppress_birth_warnings must be a bool")
    if type(include_flow_certificates) is not bool:
        raise TypeError("include_flow_certificates must be a bool")
    schedule.__post_init__()
    _validate_schedule_clock(schedule)
    _require_runtime_clock(
        graph,
        schedule.start_time,
        boundary="schedule.start",
    )
    _validate_hybrid_log(graph)
    operators, execution_word = _prepare_word(schedule, context)

    from ..dynamics.runtime import (
        _record_mutation_flow_boundary,
        _resolve_integrator_instance,
    )
    from .word_execution import execute_network_operator_stage

    targets = tuple(graph.nodes())
    configured_compute_delta_nfr = graph.graph.get("compute_delta_nfr")
    pressure_refresh_callback_invocations = 0
    if callable(configured_compute_delta_nfr):

        def compute_delta_nfr(live_graph: nx.Graph) -> None:
            nonlocal pressure_refresh_callback_invocations
            configured_compute_delta_nfr(live_graph)
            pressure_refresh_callback_invocations += 1

    else:
        compute_delta_nfr = None
    transaction = GraphTransactionSnapshot(graph)
    events_committed: list[ExecutedOperatorEvent] = []
    flow_interval_evidence: list[ExecutedNodalFlowInterval] = []
    positive_intervals = tuple(
        interval.index
        for interval in schedule.intervals
        if interval.exact_duration > 0
    )
    integrator = None

    try:
        if positive_intervals:
            integrator = _resolve_integrator_instance(graph)

        with warnings.catch_warnings():
            if suppress_birth_warnings:
                warnings.filterwarnings(
                    "ignore",
                    message=r".*has no sources.*",
                )
            for interval in schedule.intervals:
                if interval.exact_duration > 0:
                    if integrator is None:
                        raise RuntimeError(
                            "positive flow interval has no configured integrator"
                        )
                    evidence = _execute_flow_interval(
                        graph,
                        interval,
                        integrator,
                        method=method,
                        n_jobs=n_jobs,
                        include_flow_certificate=(
                            include_flow_certificates
                        ),
                    )
                    if evidence is not None:
                        flow_interval_evidence.append(evidence)
                else:
                    _require_runtime_clock(
                        graph,
                        interval.start_time,
                        boundary=f"interval[{interval.index}].start",
                    )
                    _require_runtime_clock(
                        graph,
                        interval.end_time,
                        boundary=f"interval[{interval.index}].end",
                    )

                if interval.index >= schedule.event_count:
                    continue

                event = schedule.events[interval.index]
                operator = operators[event.word_position]
                sequence_step = (
                    None
                    if execution_word is None
                    else execution_word.step(event.word_position)
                )
                result = execute_network_operator_stage(
                    graph,
                    operator,
                    targets,
                    sequence_context=sequence_step,
                    compute_delta_nfr=compute_delta_nfr,
                )
                _require_exact_stage(
                    event,
                    result,
                    target_count=len(targets),
                )
                _require_runtime_clock(
                    graph,
                    event.event_time,
                    boundary=f"event[{event.event_index}]",
                )
                _record_mutation_flow_boundary(graph)
                committed = ExecutedOperatorEvent.from_stage(event, result)
                sink = graph.graph.setdefault(_HYBRID_EVENT_LOG, [])
                if not isinstance(sink, list):
                    raise TNFRValueError(
                        "hybrid_event_log changed type during event execution.",
                        context={
                            "event_index": event.event_index,
                            "value_type": type(sink).__name__,
                        },
                    )
                sink.append(committed.as_record())
                events_committed.append(committed)

        _require_runtime_clock(
            graph,
            schedule.end_time,
            boundary="schedule.end",
        )
        return OperatorEventExecutionResult(
            schedule=schedule,
            target_nodes=targets,
            flow_interval_indices=tuple(
                interval.index for interval in schedule.intervals
            ),
            positive_flow_interval_indices=positive_intervals,
            events=tuple(events_committed),
            final_time=schedule.end_time,
            integrator_name=(
                None if integrator is None else type(integrator).__qualname__
            ),
            pressure_refresh_callback_invocations=(
                pressure_refresh_callback_invocations
            ),
            flow_certification_requested=include_flow_certificates,
            flow_interval_evidence=tuple(flow_interval_evidence),
        )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


__all__ = (
    "ExecutedNodalFlowInterval",
    "ExecutedOperatorEvent",
    "OperatorEventExecutionResult",
    "execute_operator_event_schedule",
)
