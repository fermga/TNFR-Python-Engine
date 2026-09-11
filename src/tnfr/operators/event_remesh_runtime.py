"""Atomic composition of one operator-event schedule and delayed REMESH.

The bridge owns one canonical pre-REMESH history sample. It preserves the
runtime indexing convention and passes one frozen positive diagonal metric to
the delayed-map evidence. Runtime flow accuracy and repeated history-updated
stability remain outside this execution contract.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from numbers import Real
from typing import Any

import networkx as nx

from .._remesh_contract import (
    materialize_delayed_remesh_configuration,
)
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..errors import TNFRValueError
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._delayed_remesh_kernel import (
    _materialize_indexed_history,
    _materialize_remesh_metric,
    _runtime_mapping_values_for_nodes,
)
from ._epi_domain import require_real_scalar_epi
from .event_runtime import (
    OperatorEventExecutionResult,
    _invoke_restricted_pressure_refresh,
    _require_runtime_clock,
    _schedule_preparation_state_signature,
    _selected_mapping_entries,
    execute_operator_event_schedule,
)
from .event_timing import OperatorEventSchedule, PhysicalFlowPartition
from .network_stage import (
    GraphTransactionSnapshot,
    _networkx_runtime_layout,
    _runtime_class_mro,
    _runtime_class_namespace,
    _runtime_mapping_items,
)
from .remesh import (
    DelayedRemeshResult,
    DelayedRemeshStabilityEvidence,
    _RemeshGraphSurfaceState,
    _contract_values_equal,
    _materialize_network_remesh_configuration,
    _raw_string_entry,
    _remesh_configuration_input_signature,
    _remesh_configuration_input_signatures_are_identical,
    _require_same_epi_time_histories,
    _require_same_graph_surface,
    _snapshot_alias_channels,
    _snapshot_edge_state,
    _snapshot_epi_time_histories,
    _snapshot_graph_surface,
    apply_network_remesh,
)

__all__ = [
    "EventRemeshCycleResult",
    "RemeshHistoryTransitionObservation",
    "WeightedEPIObservation",
    "execute_event_remesh_cycle",
]


_SCOPE = (
    "One atomic graph-owned cycle: execute one finite canonical operator-event "
    "schedule, append exactly one full-support pre-REMESH EPI snapshot using "
    "the runtime history convention, then invoke the separate delayed REMESH "
    "map on fixed ordered node support; edge support may change. One frozen "
    "positive diagonal metric measures all cycle-level weighted EPI "
    "observations and REMESH stability evidence; REMESH metadata retains its "
    "legacy unweighted summaries. Capacity changes and pressure refreshes are "
    "reported separately. An applied map records its same-time right endpoint "
    "in epi_time_history without adding a second delayed-history sample. The "
    "optional post-REMESH pressure callback runs only after an applied map. "
    "Phase, clock, event log, pressure-hook identity and deterministic REMESH "
    "configuration remain fixed after the schedule. External callback or "
    "integrator side effects are not rolled back. No cycle-level field "
    "composes the retained represented flow/glyph gain with delayed REMESH, "
    "or certifies solver accuracy, a global binary64 runtime gain, or "
    "stability under repeated cycles with evolving history."
)
_HISTORY_CONVENTION = (
    "append_pre_remesh_snapshot_then_read_delay_at_history[-(tau+1)]"
)
_MISSING = object()
_HISTORY_TRANSITION_PROOF_VERSION = "remesh_history_transition_v1"
_EVENT_REMESH_CYCLE_PROOF_VERSION = "event_remesh_cycle_v1"


def _require_canonical_nodes_surface(graph: nx.Graph) -> None:
    """Reject a graph subclass whose ``nodes`` read can execute user code."""

    networkx_bases = {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}
    for owner in _runtime_class_mro(type(graph)):
        if owner in networkx_bases:
            return
        if "nodes" in _runtime_class_namespace(owner):
            raise TNFRValueError(
                "Event/REMESH input materialization changed graph state: "
                "custom nodes accessors are outside the atomic cycle domain."
            )
    raise TNFRValueError("unsupported NetworkX graph runtime type")


def _exact_epi_vector(values: tuple[float, ...]) -> tuple[Fraction, ...]:
    """Return the exact rational values represented by one binary64 vector."""

    return tuple(Fraction.from_float(value) for value in values)


def _ordered_identity_is(left: Any, right: Any) -> bool:
    """Match two frozen support sequences without caller equality methods."""

    return bool(
        type(left) is tuple
        and type(right) is tuple
        and len(left) == len(right)
        and all(
            observed is expected
            for observed, expected in zip(left, right, strict=True)
        )
    )


def _exact_history(
    history: tuple[tuple[float, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Canonicalize an ordered delayed history by exact represented value."""

    return tuple(_exact_epi_vector(snapshot) for snapshot in history)


def _exact_vector_is_valid(
    value: Any,
    *,
    length: int,
    optional: bool = False,
) -> bool:
    if optional and value is None:
        return True
    return bool(
        type(value) is tuple
        and len(value) == length
        and all(type(item) is Fraction for item in value)
    )


_HISTORY_TRANSITION_FIELD_NAMES = (
    "nodes",
    "incoming_exact_history",
    "outgoing_exact_history",
    "appended_exact_pre_remesh_epi",
    "selected_local_delayed_epi",
    "selected_global_delayed_epi",
    "tau_local",
    "tau_global",
    "history_maxlen",
    "incoming_history_present",
    "incoming_history_is_canonical_deque",
    "history_container_rebuilt",
    "oldest_snapshot_evicted",
    "incoming_history_truncated_during_rebuild",
    "history_rebuild_truncation_count",
)


def _history_transition_fields(
    transition: "RemeshHistoryTransitionObservation",
) -> dict[str, Any]:
    return {
        item.name: getattr(transition, item.name)
        for item in fields(RemeshHistoryTransitionObservation)
        if item.name != "_proof_stamp"
    }


def _history_transition_stamp(**values: Any) -> tuple[Any, ...]:
    return (
        _HISTORY_TRANSITION_PROOF_VERSION,
        tuple(_proof_value(values[name]) for name in _HISTORY_TRANSITION_FIELD_NAMES),
    )


@dataclass(frozen=True, slots=True)
class RemeshHistoryTransitionObservation:
    """Sealed exact observation of one canonical delayed-history append."""

    nodes: tuple[Hashable, ...]
    incoming_exact_history: tuple[tuple[Fraction, ...], ...]
    outgoing_exact_history: tuple[tuple[Fraction, ...], ...]
    appended_exact_pre_remesh_epi: tuple[Fraction, ...]
    selected_local_delayed_epi: tuple[Fraction, ...] | None
    selected_global_delayed_epi: tuple[Fraction, ...] | None
    tau_local: int
    tau_global: int
    history_maxlen: int
    incoming_history_present: bool
    incoming_history_is_canonical_deque: bool
    history_container_rebuilt: bool
    oldest_snapshot_evicted: bool
    incoming_history_truncated_during_rebuild: bool
    history_rebuild_truncation_count: int
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        values = _history_transition_fields(self)
        if not proof_stamps_are_identical(
            self._proof_stamp,
            _history_transition_stamp(**values),
        ):
            raise ValueError(
                "REMESH history-transition proof fields are inconsistent"
            )
        _validate_history_transition_fields(**values)
        if not proof_stamps_are_identical(
            self._proof_stamp,
            _history_transition_stamp(**values),
        ):
            raise ValueError(
                "REMESH history-transition proof fields changed during validation"
            )

    def _proof_fields_are_intact(self) -> bool:
        """Fail closed after ordinary replacement or field mutation."""

        try:
            self.__post_init__()
        except BaseException:
            return False
        return True

    @property
    def canonical_history_transition_certified(self) -> bool:
        """Whether exact append, truncation and lag bindings remain intact."""

        return self._proof_fields_are_intact()


def _validate_history_transition_fields(**values: Any) -> None:
    nodes = values["nodes"]
    incoming = values["incoming_exact_history"]
    outgoing = values["outgoing_exact_history"]
    appended = values["appended_exact_pre_remesh_epi"]
    if type(nodes) is not tuple or not nodes or len(frozenset(nodes)) != len(nodes):
        raise ValueError("history-transition nodes must be a nonempty unique tuple")
    size = len(nodes)
    for label, history in (("incoming", incoming), ("outgoing", outgoing)):
        if type(history) is not tuple or any(
            not _exact_vector_is_valid(snapshot, length=size)
            for snapshot in history
        ):
            raise TypeError(
                f"{label} history must contain exact Fraction vectors"
            )
    if not _exact_vector_is_valid(appended, length=size):
        raise TypeError(
            "appended pre-REMESH EPI must be an exact Fraction vector"
        )
    for label in (
        "selected_local_delayed_epi",
        "selected_global_delayed_epi",
    ):
        if not _exact_vector_is_valid(
            values[label],
            length=size,
            optional=True,
        ):
            raise TypeError(
                f"{label} must be an exact Fraction vector or None"
            )

    tau_local = values["tau_local"]
    tau_global = values["tau_global"]
    history_maxlen = values["history_maxlen"]
    if type(tau_local) is not int or tau_local <= 0:
        raise ValueError("tau_local must be a positive integer")
    if type(tau_global) is not int or tau_global <= 0:
        raise ValueError("tau_global must be a positive integer")
    if type(history_maxlen) is not int or history_maxlen <= max(
        tau_local, tau_global
    ):
        raise ValueError(
            "history_maxlen must retain every declared delayed index"
        )

    boolean_names = (
        "incoming_history_present",
        "incoming_history_is_canonical_deque",
        "history_container_rebuilt",
        "oldest_snapshot_evicted",
        "incoming_history_truncated_during_rebuild",
    )
    if any(type(values[name]) is not bool for name in boolean_names):
        raise TypeError("history-transition facts must be strict booleans")
    truncation_count = values["history_rebuild_truncation_count"]
    if type(truncation_count) is not int or truncation_count < 0:
        raise ValueError(
            "history rebuild truncation count must be nonnegative"
        )

    present = values["incoming_history_present"]
    canonical = values["incoming_history_is_canonical_deque"]
    rebuilt = values["history_container_rebuilt"]
    if not present and incoming:
        raise ValueError("absent incoming history cannot contain snapshots")
    if canonical and (
        not present or rebuilt or len(incoming) > history_maxlen
    ):
        raise ValueError("canonical incoming deque facts are inconsistent")
    if rebuilt == canonical:
        raise ValueError(
            "history rebuild must be the complement of canonical input"
        )

    expected_truncation = (
        max(0, len(incoming) - history_maxlen) if rebuilt else 0
    )
    if truncation_count != expected_truncation:
        raise ValueError(
            "history rebuild truncation count is inconsistent"
        )
    if values["incoming_history_truncated_during_rebuild"] != bool(
        expected_truncation
    ):
        raise ValueError("history rebuild truncation flag is inconsistent")

    retained_before_append = incoming[-history_maxlen:]
    expected_eviction = len(retained_before_append) == history_maxlen
    if values["oldest_snapshot_evicted"] != expected_eviction:
        raise ValueError("history append eviction fact is inconsistent")
    expected_outgoing = (
        retained_before_append + (appended,)
    )[-history_maxlen:]
    if outgoing != expected_outgoing:
        raise ValueError(
            "outgoing history violates canonical deque append semantics"
        )

    expected_local = (
        outgoing[-(tau_local + 1)] if len(outgoing) > tau_local else None
    )
    expected_global = (
        outgoing[-(tau_global + 1)] if len(outgoing) > tau_global else None
    )
    if values["selected_local_delayed_epi"] != expected_local:
        raise ValueError(
            "local delayed vector is not bound to outgoing history"
        )
    if values["selected_global_delayed_epi"] != expected_global:
        raise ValueError(
            "global delayed vector is not bound to outgoing history"
        )


def _build_history_transition(
    *,
    nodes: tuple[Hashable, ...],
    history_before: tuple[
        bool,
        Any,
        tuple[tuple[float, ...], ...],
    ],
    appended_history: tuple[
        bool,
        Any,
        tuple[tuple[float, ...], ...],
    ],
    appended_epi: tuple[float, ...],
    tau_local: int,
    tau_global: int,
    history_maxlen: int,
    history_container_rebuilt: bool,
    oldest_snapshot_evicted: bool,
) -> RemeshHistoryTransitionObservation:
    incoming = _exact_history(history_before[2])
    outgoing = _exact_history(appended_history[2])
    fields_by_name = dict(
        nodes=nodes,
        incoming_exact_history=incoming,
        outgoing_exact_history=outgoing,
        appended_exact_pre_remesh_epi=_exact_epi_vector(appended_epi),
        selected_local_delayed_epi=(
            outgoing[-(tau_local + 1)]
            if len(outgoing) > tau_local
            else None
        ),
        selected_global_delayed_epi=(
            outgoing[-(tau_global + 1)]
            if len(outgoing) > tau_global
            else None
        ),
        tau_local=tau_local,
        tau_global=tau_global,
        history_maxlen=history_maxlen,
        incoming_history_present=history_before[0],
        incoming_history_is_canonical_deque=bool(
            type(history_before[1]) is deque
            and history_before[1].maxlen == history_maxlen
        ),
        history_container_rebuilt=history_container_rebuilt,
        oldest_snapshot_evicted=oldest_snapshot_evicted,
        incoming_history_truncated_during_rebuild=bool(
            history_container_rebuilt
            and len(incoming) > history_maxlen
        ),
        history_rebuild_truncation_count=(
            max(0, len(incoming) - history_maxlen)
            if history_container_rebuilt
            else 0
        ),
    )
    return RemeshHistoryTransitionObservation(
        **fields_by_name,
        _proof_stamp=_history_transition_stamp(**fields_by_name),
    )


@dataclass(frozen=True, slots=True)
class WeightedEPIObservation:
    """Exact represented-input EPI observation in one diagonal metric."""

    nodes: tuple[Hashable, ...]
    epi_values: tuple[float, ...]
    metric_weights: tuple[float, ...]
    exact_weighted_mean: Fraction
    weighted_mean: float
    exact_disagreement_energy: Fraction
    disagreement_energy: float | None


def _proof_value(
    value: Any,
    *,
    seen: dict[int, int] | None = None,
) -> Any:
    """Seal values, reusing exact child seals for closed TNFR records."""

    del seen
    kind = type(value)
    if kind in (OperatorEventExecutionResult, RemeshHistoryTransitionObservation):
        try:
            stamp = object.__getattribute__(value, "_proof_stamp")
        except BaseException:
            stamp = None
        if type(stamp) is not tuple:
            stamp = None
        return (
            "tnfr-nested-proof-stamp-v1",
            kind.__module__,
            kind.__qualname__,
            stamp,
        )
    return structural_proof_signature(value)


def _nested_proof_records_are_intact(
    value: Any,
    *,
    seen: set[int] | None = None,
) -> bool:
    """Validate only the closed set of proof-bearing TNFR record roots."""

    del seen
    if type(value) is OperatorEventExecutionResult:
        try:
            return bool(
                OperatorEventExecutionResult._proof_fields_are_intact(value)
            )
        except BaseException:
            return False
    if type(value) is DelayedRemeshResult:
        # DelayedRemeshResult has no independent stamp. Its complete immutable
        # structure is already included in the enclosing cycle proof stamp.
        return True
    return True


_CYCLE_FIXED_PROOF_FIELDS = {
    "history_convention": _HISTORY_CONVENTION,
    "schedule_left_history_unchanged": True,
    "whole_cycle_graph_state_atomic": True,
    "common_metric_is_frozen": True,
    "schedule_endpoint_clock_preserved": True,
    "hybrid_event_log_preserved": True,
    "pressure_hook_identity_preserved": True,
    "remesh_configuration_frozen": True,
    "scope": _SCOPE,
}


def _cycle_result_fields(
    result: "EventRemeshCycleResult",
) -> dict[str, Any]:
    return {
        item.name: getattr(result, item.name)
        for item in fields(EventRemeshCycleResult)
        if item.name != "_proof_stamp"
    }


def _cycle_result_stamp(values: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _EVENT_REMESH_CYCLE_PROOF_VERSION,
        tuple(
            (name, _proof_value(values[name]))
            for name in _EVENT_REMESH_CYCLE_FIELD_NAMES
        ),
    )


def _validate_weighted_observation(
    observation: WeightedEPIObservation,
    *,
    nodes: tuple[Hashable, ...],
    weights: tuple[float, ...],
    label: str,
) -> None:
    if type(observation) is not WeightedEPIObservation:
        raise TypeError(f"{label} must be a WeightedEPIObservation")
    if (
        not _ordered_identity_is(observation.nodes, nodes)
        or observation.metric_weights != weights
    ):
        raise ValueError(f"{label} changed node order or metric")
    if (
        type(observation.epi_values) is not tuple
        or len(observation.epi_values) != len(nodes)
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in observation.epi_values
        )
    ):
        raise ValueError(f"{label} must contain finite binary64 EPI values")
    values_q = _exact_epi_vector(observation.epi_values)
    weights_q = _exact_epi_vector(weights)
    total_weight = sum(weights_q, Fraction(0))
    mean = sum(
        (
            weight * value
            for weight, value in zip(weights_q, values_q, strict=True)
        ),
        Fraction(0),
    ) / total_weight
    energy = sum(
        (
            weight * (value - mean) ** 2
            for weight, value in zip(weights_q, values_q, strict=True)
        ),
        Fraction(0),
    ) / 2
    if (
        observation.exact_weighted_mean != mean
        or observation.weighted_mean
        != _diagnostic_float(mean, f"{label} weighted EPI mean")
        or observation.exact_disagreement_energy != energy
        or observation.disagreement_energy
        != _optional_diagnostic_float(energy)
    ):
        raise ValueError(f"{label} exact diagnostics are inconsistent")


def _validate_event_remesh_cycle_result(
    result: "EventRemeshCycleResult",
) -> None:
    nodes = result.target_nodes
    weights = result.metric_weights
    if type(nodes) is not tuple or not nodes or len(frozenset(nodes)) != len(nodes):
        raise ValueError("cycle target_nodes must be a nonempty unique tuple")
    if (
        type(weights) is not tuple
        or len(weights) != len(nodes)
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value <= 0.0
            for value in weights
        )
    ):
        raise ValueError("cycle metric_weights must be finite and positive")

    if type(result.history_transition) is not RemeshHistoryTransitionObservation:
        raise TypeError(
            "history_transition must be a RemeshHistoryTransitionObservation"
        )
    transition = result.history_transition
    if (
        not transition._proof_fields_are_intact()
        or not _ordered_identity_is(transition.nodes, nodes)
    ):
        raise ValueError("cycle history-transition proof is not intact")

    if type(result.event_execution) is not OperatorEventExecutionResult:
        raise TypeError(
            "event_execution must be an OperatorEventExecutionResult"
        )
    if not _ordered_identity_is(result.event_execution.target_nodes, nodes):
        raise ValueError("event execution target order changed")
    schedule = result.event_execution.schedule
    try:
        schedule.__post_init__()
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("event schedule is not canonical") from exc
    if (
        result.event_execution.final_time != schedule.end_time
        or result.event_execution.flow_interval_indices
        != tuple(range(len(schedule.intervals)))
        or result.event_execution.positive_flow_interval_indices
        != tuple(
            interval.index
            for interval in schedule.intervals
            if interval.exact_duration > 0
        )
        or len(result.event_execution.events) != len(schedule.events)
    ):
        raise ValueError("event execution disagrees with its schedule")
    for executed, scheduled in zip(
        result.event_execution.events,
        schedule.events,
        strict=True,
    ):
        if (
            executed.event_index != scheduled.event_index
            or executed.cycle_index != scheduled.cycle_index
            or executed.word_position != scheduled.word_position
            or executed.operator_name != scheduled.operator_name
            or executed.glyph is not scheduled.glyph
            or executed.event_time != scheduled.event_time
            or executed.event_offset != scheduled.event_offset
            or executed.exact_event_time != scheduled.exact_event_time
            or not executed.zero_duration
            or executed.nodes_processed != len(nodes)
        ):
            raise ValueError(
                "executed event disagrees with its scheduled boundary"
            )
    if not _nested_proof_records_are_intact(result.event_execution):
        raise ValueError("nested event-execution proof record is not intact")

    if type(result.remesh) is not DelayedRemeshResult:
        raise TypeError("remesh must be a DelayedRemeshResult")
    if not _nested_proof_records_are_intact(result.remesh):
        raise ValueError("nested REMESH proof record is not intact")
    plan = result.remesh.plan
    configuration = materialize_delayed_remesh_configuration(
        tau_local=plan.tau_local,
        tau_global=plan.tau_global,
        alpha=plan.alpha,
        alpha_source=plan.alpha_source,
        epi_min=plan.epi_min,
        epi_max=plan.epi_max,
        clip_mode=plan.clip_mode,
    )
    if (
        configuration.history_maxlen != transition.history_maxlen
        or result.remesh.status != plan.status
        or not _ordered_identity_is(plan.node_order, nodes)
        or plan.tau_local != transition.tau_local
        or plan.tau_global != transition.tau_global
        or plan.history_length != len(transition.outgoing_exact_history)
        or plan.required_history_length
        != max(transition.tau_local, transition.tau_global) + 1
    ):
        raise ValueError("REMESH plan does not match the history transition")

    for label, observation in (
        ("pre_schedule_epi", result.pre_schedule_epi),
        ("pre_remesh_epi", result.pre_remesh_epi),
        ("post_remesh_epi", result.post_remesh_epi),
    ):
        _validate_weighted_observation(
            observation,
            nodes=nodes,
            weights=weights,
            label=label,
        )
    if (
        transition.appended_exact_pre_remesh_epi
        != _exact_epi_vector(result.pre_remesh_epi.epi_values)
    ):
        raise ValueError("history append is not bound to pre-REMESH EPI")

    if result.history_length_before_cycle != len(
        transition.incoming_exact_history
    ):
        raise ValueError("history_length_before_cycle is inconsistent")
    expected_before_append = min(
        len(transition.incoming_exact_history),
        transition.history_maxlen,
    )
    if result.history_length_before_append != expected_before_append:
        raise ValueError("history_length_before_append is inconsistent")
    if result.history_length_after_append != len(
        transition.outgoing_exact_history
    ):
        raise ValueError("history_length_after_append is inconsistent")
    if (
        result.history_maxlen != transition.history_maxlen
        or result.history_container_rebuilt
        != transition.history_container_rebuilt
        or result.history_oldest_snapshot_evicted
        != transition.oldest_snapshot_evicted
    ):
        raise ValueError("legacy history metadata disagrees with transition")

    local = transition.selected_local_delayed_epi
    global_snapshot = transition.selected_global_delayed_epi
    expected_applied = local is not None and global_snapshot is not None
    if result.remesh.applied != expected_applied:
        raise ValueError("REMESH application disagrees with delayed inputs")
    if expected_applied:
        assert local is not None and global_snapshot is not None
        if len(plan.proposals) != len(nodes):
            raise ValueError("applied REMESH requires one proposal per node")
        for index, proposal in enumerate(plan.proposals):
            if (
                proposal.node is not nodes[index]
                or Fraction.from_float(proposal.epi_now)
                != transition.appended_exact_pre_remesh_epi[index]
                or Fraction.from_float(proposal.epi_local) != local[index]
                or Fraction.from_float(proposal.epi_global)
                != global_snapshot[index]
                or Fraction.from_float(proposal.bounded_epi)
                != _exact_epi_vector(
                    result.post_remesh_epi.epi_values
                )[index]
            ):
                raise ValueError(
                    "REMESH proposal is not bound to exact history inputs"
                )
    elif plan.proposals:
        raise ValueError("insufficient history cannot publish REMESH proposals")
    elif (
        result.pre_remesh_epi.epi_values
        != result.post_remesh_epi.epi_values
    ):
        raise ValueError("REMESH no-op cannot change EPI")

    if result.remesh.applied:
        if type(plan.evidence) is not DelayedRemeshStabilityEvidence:
            raise ValueError(
                "applied REMESH requires exact stability evidence"
            )
        if plan.evidence.metric_weights != weights:
            raise ValueError("REMESH evidence changed the cycle metric")
    elif plan.evidence is not None:
        raise ValueError("REMESH no-op cannot publish stability evidence")

    size = len(nodes)
    channel_names = (
        "capacity_before_schedule",
        "capacity_before_remesh",
        "capacity_after_remesh",
        "pressure_before_schedule",
        "pressure_before_remesh",
        "pressure_after_remesh_before_refresh",
        "pressure_after_optional_refresh",
        "phase_before_schedule",
        "phase_before_remesh",
        "phase_after_remesh_before_refresh",
        "phase_after_optional_refresh",
    )
    for name in channel_names:
        channel = getattr(result, name)
        if (
            type(channel) is not tuple
            or len(channel) != size
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in channel
            )
        ):
            raise ValueError(f"{name} must align with target_nodes")
    if any(
        value < 0.0
        for name in (
            "capacity_before_schedule",
            "capacity_before_remesh",
            "capacity_after_remesh",
        )
        for value in getattr(result, name)
    ):
        raise ValueError("cycle capacity observations must be nonnegative")
    if result.capacity_after_remesh != result.capacity_before_remesh:
        raise ValueError("REMESH changed structural frequency")
    if (
        result.pressure_after_remesh_before_refresh
        != result.pressure_before_remesh
    ):
        raise ValueError("REMESH changed structural pressure")
    if not (
        result.phase_before_remesh
        == result.phase_after_remesh_before_refresh
        == result.phase_after_optional_refresh
    ):
        raise ValueError("REMESH or pressure refresh changed phase")

    schedule_drift = (
        result.pre_remesh_epi.exact_weighted_mean
        - result.pre_schedule_epi.exact_weighted_mean
    )
    remesh_drift = (
        result.post_remesh_epi.exact_weighted_mean
        - result.pre_remesh_epi.exact_weighted_mean
    )
    total_drift = (
        result.post_remesh_epi.exact_weighted_mean
        - result.pre_schedule_epi.exact_weighted_mean
    )
    if (
        result.exact_schedule_weighted_mean_drift != schedule_drift
        or result.exact_remesh_weighted_mean_drift != remesh_drift
        or result.exact_total_weighted_mean_drift != total_drift
        or result.schedule_weighted_mean_drift
        != _optional_diagnostic_float(schedule_drift)
        or result.remesh_weighted_mean_drift
        != _optional_diagnostic_float(remesh_drift)
        or result.total_weighted_mean_drift
        != _optional_diagnostic_float(total_drift)
    ):
        raise ValueError("cycle weighted-mean drifts are inconsistent")

    strict_boolean_names = (
        "schedule_capacity_changed",
        "remesh_capacity_changed",
        "post_remesh_pressure_refresh_requested",
        "post_remesh_epi_time_boundary_recorded",
    )
    if any(
        type(getattr(result, name)) is not bool
        for name in strict_boolean_names
    ):
        raise TypeError("cycle observations must use strict booleans")
    if result.schedule_capacity_changed != (
        result.capacity_before_remesh
        != result.capacity_before_schedule
    ):
        raise ValueError("schedule capacity-change fact is inconsistent")
    if result.remesh_capacity_changed != (
        result.capacity_after_remesh
        != result.capacity_before_remesh
    ):
        raise ValueError("REMESH capacity-change fact is inconsistent")
    if (
        result.post_remesh_epi_time_boundary_recorded
        != result.remesh.epi_time_boundary_recorded
        or result.remesh.applied
        != result.post_remesh_epi_time_boundary_recorded
    ):
        raise ValueError("REMESH EPI-time boundary fact is inconsistent")

    count_names = (
        "schedule_pressure_refresh_callback_invocations",
        "post_remesh_pressure_refresh_callback_invocations",
        "committed_hybrid_event_log_length",
    )
    if any(
        type(getattr(result, name)) is not int
        or getattr(result, name) < 0
        for name in count_names
    ):
        raise ValueError("cycle callback and event-log counts must be nonnegative")
    if (
        result.schedule_pressure_refresh_callback_invocations
        != result.event_execution.pressure_refresh_callback_invocations
    ):
        raise ValueError("schedule pressure-callback count is inconsistent")
    expected_post_refresh_count = int(
        result.post_remesh_pressure_refresh_requested
        and result.remesh.applied
    )
    if (
        result.post_remesh_pressure_refresh_callback_invocations
        != expected_post_refresh_count
    ):
        raise ValueError(
            "post-REMESH pressure callback execution is inconsistent"
        )
    if (
        expected_post_refresh_count == 0
        and result.pressure_after_optional_refresh
        != result.pressure_after_remesh_before_refresh
    ):
        raise ValueError(
            "pressure changed without the explicit post-REMESH callback"
        )

    for name, expected in _CYCLE_FIXED_PROOF_FIELDS.items():
        if getattr(result, name) != expected:
            raise ValueError(f"fixed cycle field {name!r} is inconsistent")


@dataclass(frozen=True, slots=True)
class EventRemeshCycleResult:
    """Immutable evidence for one committed schedule/history/REMESH cycle."""

    target_nodes: tuple[Hashable, ...]
    metric_weights: tuple[float, ...]
    event_execution: OperatorEventExecutionResult
    remesh: DelayedRemeshResult
    history_transition: RemeshHistoryTransitionObservation
    pre_schedule_epi: WeightedEPIObservation
    pre_remesh_epi: WeightedEPIObservation
    post_remesh_epi: WeightedEPIObservation
    exact_schedule_weighted_mean_drift: Fraction
    exact_remesh_weighted_mean_drift: Fraction
    exact_total_weighted_mean_drift: Fraction
    schedule_weighted_mean_drift: float | None
    remesh_weighted_mean_drift: float | None
    total_weighted_mean_drift: float | None
    capacity_before_schedule: tuple[float, ...]
    capacity_before_remesh: tuple[float, ...]
    capacity_after_remesh: tuple[float, ...]
    pressure_before_schedule: tuple[float, ...]
    pressure_before_remesh: tuple[float, ...]
    pressure_after_remesh_before_refresh: tuple[float, ...]
    pressure_after_optional_refresh: tuple[float, ...]
    phase_before_schedule: tuple[float, ...]
    phase_before_remesh: tuple[float, ...]
    phase_after_remesh_before_refresh: tuple[float, ...]
    phase_after_optional_refresh: tuple[float, ...]
    schedule_capacity_changed: bool
    remesh_capacity_changed: bool
    history_length_before_cycle: int
    history_length_before_append: int
    history_length_after_append: int
    history_maxlen: int
    history_container_rebuilt: bool
    history_oldest_snapshot_evicted: bool
    schedule_pressure_refresh_callback_invocations: int
    post_remesh_pressure_refresh_requested: bool
    post_remesh_pressure_refresh_callback_invocations: int
    committed_hybrid_event_log_length: int
    post_remesh_epi_time_boundary_recorded: bool
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    history_convention: str = field(
        default=_HISTORY_CONVENTION,
        init=False,
    )
    schedule_left_history_unchanged: bool = field(default=True, init=False)
    whole_cycle_graph_state_atomic: bool = field(default=True, init=False)
    common_metric_is_frozen: bool = field(default=True, init=False)
    schedule_endpoint_clock_preserved: bool = field(default=True, init=False)
    hybrid_event_log_preserved: bool = field(default=True, init=False)
    pressure_hook_identity_preserved: bool = field(default=True, init=False)
    remesh_configuration_frozen: bool = field(default=True, init=False)
    scope: str = field(default=_SCOPE, init=False)

    def __post_init__(self) -> None:
        values = _cycle_result_fields(self)
        if not proof_stamps_are_identical(
            self._proof_stamp,
            _cycle_result_stamp(values),
        ):
            raise ValueError(
                "event/REMESH cycle proof fields are inconsistent"
            )
        _validate_event_remesh_cycle_result(self)
        if not proof_stamps_are_identical(
            self._proof_stamp,
            _cycle_result_stamp(values),
        ):
            raise ValueError(
                "event/REMESH cycle proof fields changed during validation"
            )

    def _proof_fields_are_intact(self) -> bool:
        """Fail closed after changes to decisive nested or boundary state."""

        try:
            self.__post_init__()
        except BaseException:
            return False
        return True

    @property
    def remesh_history_repetition_certified(self) -> bool:
        """Repeated stability is outside one observed history transition."""

        return False

    @property
    def mixed_runtime_gain_certified(self) -> bool:
        """The represented schedule gain is not composed with delayed REMESH."""

        return False

    @property
    def external_side_effects_rolled_back(self) -> bool:
        """Graph rollback cannot retract already emitted external effects."""

        return False

    @property
    def remesh_applied(self) -> bool:
        """Whether the delayed map committed in this cycle."""

        return self._proof_fields_are_intact() and self.remesh.applied

    @property
    def post_remesh_pressure_refresh_performed(self) -> bool:
        """Whether the explicit post-map pressure callback completed."""

        return bool(
            self._proof_fields_are_intact()
            and self.post_remesh_pressure_refresh_callback_invocations == 1
        )


_EVENT_REMESH_CYCLE_FIELD_NAMES = tuple(
    item.name
    for item in fields(EventRemeshCycleResult)
    if item.name != "_proof_stamp"
)


def _sealed_event_remesh_cycle_result(
    **values: Any,
) -> EventRemeshCycleResult:
    proof_values = dict(_CYCLE_FIXED_PROOF_FIELDS)
    proof_values.update(values)
    if set(proof_values) != set(_EVENT_REMESH_CYCLE_FIELD_NAMES):
        raise RuntimeError(
            "event/REMESH cycle constructor fields are incomplete"
        )
    return EventRemeshCycleResult(
        **values,
        _proof_stamp=_cycle_result_stamp(proof_values),
    )


def _require_graph(graph: Any) -> nx.Graph:
    graph_bases = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    graph_mro = _runtime_class_mro(type(graph))
    if not any(owner is base for owner in graph_mro for base in graph_bases):
        raise TypeError("graph must be a NetworkX graph")
    return graph


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray, complex)):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    if not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    return result


def _diagnostic_float(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} exceeds the finite binary64 diagnostic range"
        ) from exc
    if not math.isfinite(result):
        raise TNFRValueError(
            f"{label} exceeds the finite binary64 diagnostic range"
        )
    return result


def _optional_diagnostic_float(value: Fraction) -> float | None:
    """Return one finite binary64 display, or ``None`` on overflow."""

    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _required_scalar_channel(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
    aliases: tuple[str, ...],
    label: str,
    *,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    values: list[float] = []
    layout = _networkx_runtime_layout(graph)
    observed_nodes = tuple(node for node, _data in layout.node_data)
    if not _ordered_identity_is(observed_nodes, nodes):
        raise TNFRValueError(
            "The event/REMESH cycle requires fixed ordered node support."
        )
    for node, node_data in layout.node_data:
        entries = _selected_mapping_entries(node_data, aliases)
        raw = entries[0][1] if entries else _MISSING
        if raw is _MISSING:
            raise TNFRValueError(f"node {node!r} is missing required {label}")
        value = _finite_real(raw, f"node {node!r} {label}")
        if nonnegative and value < 0.0:
            raise TNFRValueError(
                f"node {node!r} {label} must be nonnegative"
            )
        values.append(value)
    return tuple(values)


def _epi_values(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
) -> tuple[float, ...]:
    values: list[float] = []
    layout = _networkx_runtime_layout(graph)
    observed_nodes = tuple(node for node, _data in layout.node_data)
    if not _ordered_identity_is(observed_nodes, nodes):
        raise TNFRValueError(
            "The event/REMESH cycle requires fixed ordered node support."
        )
    for node, node_data in layout.node_data:
        entries = _selected_mapping_entries(node_data, ALIAS_EPI)
        raw = entries[0][1] if entries else _MISSING
        if raw is _MISSING:
            raise TNFRValueError(f"node {node!r} is missing required EPI")
        values.append(
            require_real_scalar_epi(
                raw,
                operator="Recursivity",
                label=f"node {node!r} EPI",
            )
        )
    return tuple(values)


def _observe_epi(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
    weights: tuple[float, ...],
) -> WeightedEPIObservation:
    values = _epi_values(graph, nodes)
    values_q = tuple(Fraction.from_float(value) for value in values)
    weights_q = tuple(Fraction.from_float(value) for value in weights)
    total_weight = sum(weights_q, Fraction(0))
    mean = sum(
        (
            weight * value
            for weight, value in zip(weights_q, values_q, strict=True)
        ),
        Fraction(0),
    ) / total_weight
    energy = sum(
        (
            weight * (value - mean) * (value - mean)
            for weight, value in zip(weights_q, values_q, strict=True)
        ),
        Fraction(0),
    ) / 2
    return WeightedEPIObservation(
        nodes=nodes,
        epi_values=values,
        metric_weights=weights,
        exact_weighted_mean=mean,
        weighted_mean=_diagnostic_float(mean, "weighted EPI mean"),
        exact_disagreement_energy=energy,
        disagreement_energy=_optional_diagnostic_float(energy),
    )


def _history_signature(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
) -> tuple[
    bool,
    Any,
    tuple[tuple[float, ...], ...],
]:
    layout = _networkx_runtime_layout(graph)
    present, raw = _raw_string_entry(
        layout.graph_mapping,
        "_epi_hist",
        _MISSING,
    )
    if not present:
        return False, _MISSING, ()
    if raw is None:
        return True, raw, ()
    snapshots = _materialize_indexed_history(raw)

    signature: list[tuple[float, ...]] = []
    for index, snapshot in enumerate(snapshots):
        values = _runtime_mapping_values_for_nodes(
            snapshot,
            nodes,
            label=f"_epi_hist[{index}]",
        )
        signature.append(
            tuple(
                require_real_scalar_epi(
                    value,
                    operator="Recursivity",
                    label=f"_epi_hist[{index}][{node!r}]",
                )
                for node, value in zip(nodes, values, strict=True)
            )
        )
    return True, raw, tuple(signature)


def _require_same_history(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
    expected: tuple[
        bool,
        Any,
        tuple[tuple[float, ...], ...],
    ],
    *,
    boundary: str,
) -> tuple[bool, Any, tuple[tuple[float, ...], ...]]:
    observed = _history_signature(graph, nodes)
    expected_present, expected_object, expected_values = expected
    observed_present, observed_object, observed_values = observed
    if (
        observed_present != expected_present
        or observed_object is not expected_object
        or observed_values != expected_values
    ):
        raise TNFRValueError(
            "The operator-event schedule changed REMESH history outside "
            "the explicit history boundary.",
            context={"boundary": boundary},
        )
    return observed


def _require_node_order(
    graph: nx.Graph,
    expected: tuple[Hashable, ...],
    *,
    boundary: str,
) -> None:
    observed = tuple(
        node for node, _data in _networkx_runtime_layout(graph).node_data
    )
    if not _ordered_identity_is(observed, expected):
        raise TNFRValueError(
            "The event/REMESH cycle requires fixed ordered node support.",
            context={
                "boundary": boundary,
                "expected_nodes": expected,
                "observed_nodes": observed,
            },
        )


def _pressure_hook_signature(graph: nx.Graph) -> tuple[bool, Any]:
    return _raw_string_entry(
        _networkx_runtime_layout(graph).graph_mapping,
        "compute_delta_nfr",
        None,
    )


def _read_only_graph_state(
    graph: nx.Graph,
) -> tuple[tuple[Any, ...], list[Any]]:
    """Capture graph/callback state around a bridge-owned read phase."""

    retained_references: list[Any] = []
    return (
        _schedule_preparation_state_signature(
            graph,
            retained_references=retained_references,
        ),
        retained_references,
    )


def _require_read_only_graph_state(
    graph: nx.Graph,
    expected: tuple[tuple[Any, ...], list[Any]],
    *,
    boundary: str,
) -> None:
    """Reject graph mutation caused by observational materialization."""

    expected_signature, retained_references = expected
    del retained_references
    if not proof_stamps_are_identical(
        expected_signature,
        _schedule_preparation_state_signature(graph),
    ):
        raise TNFRValueError(
            "Event/REMESH observational materialization changed graph state.",
            context={"boundary": boundary},
        )


def _require_same_pressure_hook(
    graph: nx.Graph,
    expected: tuple[bool, Any],
    *,
    boundary: str,
) -> None:
    observed = _pressure_hook_signature(graph)
    if observed[0] != expected[0] or observed[1] is not expected[1]:
        raise TNFRValueError(
            "The event/REMESH cycle changed the pressure-refresh hook.",
            context={"boundary": boundary},
        )


def _require_same_configuration(
    graph: nx.Graph,
    expected_signature: tuple[Any, ...],
    *,
    boundary: str,
) -> None:
    if not _remesh_configuration_input_signatures_are_identical(
        _remesh_configuration_input_signature(graph),
        expected_signature,
    ):
        raise TNFRValueError(
            "The event/REMESH cycle changed deterministic REMESH configuration.",
            context={"boundary": boundary},
        )


def _require_same_phase(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
    expected: tuple[float, ...],
    *,
    boundary: str,
) -> tuple[float, ...]:
    observed = _required_scalar_channel(graph, nodes, ALIAS_THETA, "phase")
    if not _contract_values_equal(observed, expected):
        raise TNFRValueError(
            "The delayed REMESH boundary changed phase.",
            context={"boundary": boundary},
        )
    return observed


def _require_schedule_event_log_commit(
    graph: nx.Graph,
    before: tuple[bool, Any, Any],
    event_result: OperatorEventExecutionResult,
) -> tuple[bool, Any, Any]:
    """Bind the immutable schedule result to its graph-owned event log."""

    before_state = before[2]
    if before[0] and (
        type(before_state) is not _RemeshGraphSurfaceState
        or before_state.sequence_items is None
        or before_state.sequence_signature is None
    ):
        raise TNFRValueError("pre-existing hybrid_event_log must be a list")
    before_length = 0 if not before[0] else len(before_state.sequence_items)
    expected_suffix = tuple(event.as_record() for event in event_result.events)
    expected_present = before[0] or bool(event_result.events)
    observed = _snapshot_graph_surface(graph, "hybrid_event_log")
    if observed[0] != expected_present:
        raise TNFRValueError(
            "hybrid_event_log presence disagrees with the schedule result"
        )
    if not expected_present:
        return observed
    if before[0] and observed[1] is not before[1]:
        raise TNFRValueError(
            "hybrid_event_log changed identity during schedule execution"
        )
    observed_state = observed[2]
    if (
        type(observed_state) is not _RemeshGraphSurfaceState
        or observed_state.sequence_items is None
        or list not in _runtime_class_mro(type(observed[1]))
    ):
        raise TNFRValueError(
            "hybrid_event_log content disagrees with the schedule result"
        )
    observed_items = observed_state.sequence_items
    if len(observed_items) != before_length + len(expected_suffix):
        raise TNFRValueError(
            "hybrid_event_log content disagrees with the schedule result"
        )
    opaque_references = (
        before_state.opaque_references
        if before[0]
        else observed_state.opaque_references
    )
    prefix_signature = structural_proof_signature(
        observed_items[:before_length],
        opaque_references=opaque_references,
    )
    suffix_signature = structural_proof_signature(
        observed_items[before_length:],
        opaque_references=opaque_references,
    )
    expected_suffix_signature = structural_proof_signature(
        expected_suffix,
        opaque_references=opaque_references,
    )
    if (
        before[0] and prefix_signature != before_state.sequence_signature
    ) or suffix_signature != expected_suffix_signature:
        raise TNFRValueError(
            "hybrid_event_log content disagrees with the schedule result"
        )
    return observed


def _require_epi_time_right_endpoint(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
    *,
    runtime_time: float,
    epi_values: tuple[float, ...],
) -> tuple[tuple[Hashable, Any, tuple[Any, ...]], ...]:
    """Require the applied REMESH jump as the authoritative same-time tail."""

    snapshots = _snapshot_epi_time_histories(graph, nodes)
    for (node, _history, samples), epi in zip(
        snapshots, epi_values, strict=True
    ):
        if not samples or samples[-1] != (runtime_time, epi):
            raise RuntimeError(
                f"node {node!r} lost the post-REMESH EPI-time endpoint"
            )
    return snapshots


def execute_event_remesh_cycle(
    graph: nx.Graph,
    schedule: OperatorEventSchedule,
    *,
    metric_weights: (
        Mapping[Hashable, Any] | Sequence[Any] | None
    ) = None,
    refresh_pressure_after_remesh: bool = False,
    context: Mapping[str, Any] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    suppress_birth_warnings: bool = False,
    include_flow_certificates: bool = False,
    include_stage_certificates: bool = False,
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = (),
) -> EventRemeshCycleResult:
    """Execute one schedule, one history sample and one delayed REMESH map.

    The schedule must leave ordered node support and the pre-existing REMESH
    history unchanged. The bridge then appends the schedule endpoint exactly
    where the ordinary runtime samples it: immediately before REMESH. A delay
    tau consequently remains history[-(tau + 1)]. The post-map output is not
    appended a second time.

    If requested, the graph's configured pressure callback runs once only
    after an applied REMESH map. It must return successfully before it is
    counted. Any failure restores graph-owned state through the outer
    transaction while retaining the primary exception. Declared physical-flow
    partitions are materialized once inside that same outer transaction and
    delegated unchanged to the schedule executor.
    """

    graph = _require_graph(graph)
    if type(schedule) is not OperatorEventSchedule:
        raise TypeError("schedule must be an OperatorEventSchedule")
    if type(refresh_pressure_after_remesh) is not bool:
        raise TypeError("refresh_pressure_after_remesh must be a bool")
    if type(include_flow_certificates) is not bool:
        raise TypeError("include_flow_certificates must be a bool")
    if type(include_stage_certificates) is not bool:
        raise TypeError("include_stage_certificates must be a bool")

    transaction = GraphTransactionSnapshot(graph)

    try:
        preparation_references: list[Any] = []
        preparation_state = _schedule_preparation_state_signature(
            graph,
            retained_references=preparation_references,
        )
        schedule.__post_init__()
        _require_canonical_nodes_surface(graph)
        runtime_layout = _networkx_runtime_layout(graph)
        nodes = tuple(node for node, _data in runtime_layout.node_data)
        if not nodes:
            raise TNFRValueError(
                "event/REMESH composition requires nonempty node support"
            )
        try:
            materialized_physical_flow_partitions = tuple(
                physical_flow_partitions
            )
        except TypeError as exc:
            raise TypeError(
                "physical_flow_partitions must be an iterable of partitions"
            ) from exc
        for position, partition in enumerate(
            materialized_physical_flow_partitions
        ):
            if type(partition) is not PhysicalFlowPartition:
                raise TypeError(
                    "physical_flow_partitions must contain "
                    "PhysicalFlowPartition records; item "
                    f"{position} has type {type(partition).__qualname__}"
                )
        weights = _materialize_remesh_metric(
            metric_weights,
            nodes,
        )
        history_before = _history_signature(graph, nodes)
        history_length_before = len(history_before[2])
        remesh_configuration_references: list[Any] = []
        remesh_configuration_signature = (
            _remesh_configuration_input_signature(
                graph,
                retained_references=remesh_configuration_references,
            )
        )
        remesh_configuration = _materialize_network_remesh_configuration(graph)
        pressure_hook = _pressure_hook_signature(graph)
        pressure_callback = pressure_hook[1]
        event_log_before = _snapshot_graph_surface(graph, "hybrid_event_log")
        if refresh_pressure_after_remesh and not callable(pressure_callback):
            raise TNFRValueError(
                "refresh_pressure_after_remesh requires a callable "
                "compute_delta_nfr graph hook"
            )

        pre_schedule_epi = _observe_epi(graph, nodes, weights)
        capacity_before = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_VF,
            "structural frequency",
            nonnegative=True,
        )
        pressure_before = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_DNFR,
            "DeltaNFR",
        )
        phase_before_schedule = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_THETA,
            "phase",
        )
        if not proof_stamps_are_identical(
            preparation_state,
            _schedule_preparation_state_signature(graph),
        ):
            raise TNFRValueError(
                "Event/REMESH input materialization changed graph state."
            )
        event_result = execute_operator_event_schedule(
            graph,
            schedule,
            context=context,
            method=method,
            n_jobs=n_jobs,
            suppress_birth_warnings=suppress_birth_warnings,
            include_flow_certificates=include_flow_certificates,
            include_stage_certificates=include_stage_certificates,
            physical_flow_partitions=materialized_physical_flow_partitions,
        )
        if not _ordered_identity_is(event_result.target_nodes, nodes):
            raise RuntimeError(
                "event execution lost the bridge's frozen target order"
            )
        after_schedule_read_state = _read_only_graph_state(graph)
        _require_node_order(graph, nodes, boundary="after_schedule")
        history_after_schedule = _require_same_history(
            graph,
            nodes,
            history_before,
            boundary="after_schedule",
        )
        _require_runtime_clock(
            graph,
            event_result.final_time,
            boundary="event_remesh.after_schedule",
        )
        _require_same_pressure_hook(
            graph,
            pressure_hook,
            boundary="after_schedule",
        )
        _require_same_configuration(
            graph,
            remesh_configuration_signature,
            boundary="after_schedule",
        )
        committed_event_log = _require_schedule_event_log_commit(
            graph,
            event_log_before,
            event_result,
        )
        committed_event_log_state = committed_event_log[2]
        committed_event_log_length = (
            0
            if not committed_event_log[0]
            else len(committed_event_log_state.sequence_items)
            if type(committed_event_log_state) is _RemeshGraphSurfaceState
            and committed_event_log_state.sequence_items is not None
            else -1
        )
        if committed_event_log_length < 0:
            raise RuntimeError("committed hybrid_event_log lost list storage")

        pre_remesh_epi = _observe_epi(graph, nodes, weights)
        capacity_before_remesh = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_VF,
            "structural frequency",
            nonnegative=True,
        )
        pressure_before_remesh = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_DNFR,
            "DeltaNFR",
        )
        phase_before_remesh = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_THETA,
            "phase",
        )
        _require_read_only_graph_state(
            graph,
            after_schedule_read_state,
            boundary="after_schedule_observations",
        )

        from ..dynamics.remesh_history import (
            _append_remesh_epi_history_snapshot_from_materialized,
        )

        history_append = _append_remesh_epi_history_snapshot_from_materialized(
            graph,
            history_maxlen=remesh_configuration.history_maxlen,
            snapshot_items=tuple(
                zip(nodes, pre_remesh_epi.epi_values, strict=True)
            ),
        )
        appended_history_present, appended_history_object = _raw_string_entry(
            _networkx_runtime_layout(graph).graph_mapping,
            "_epi_hist",
            None,
        )
        retained_before_append = history_after_schedule[2][
            -history_append.history_maxlen:
        ]
        expected_appended_values = (
            retained_before_append + (pre_remesh_epi.epi_values,)
        )[-history_append.history_maxlen:]
        appended_history = (
            appended_history_present,
            appended_history_object,
            expected_appended_values,
        )
        if (
            not appended_history[0]
            or type(appended_history[1]) is not deque
            or appended_history[1].maxlen != history_append.history_maxlen
            or len(appended_history[1]) != history_append.history_length_after
            or any(
                observed_node is not expected_node
                for (observed_node, _value), expected_node in zip(
                    history_append.snapshot_items,
                    nodes,
                    strict=True,
                )
            )
            or not _contract_values_equal(
                tuple(value for _node, value in history_append.snapshot_items),
                pre_remesh_epi.epi_values,
            )
            or history_append.history_maxlen
            != remesh_configuration.history_maxlen
        ):
            raise RuntimeError(
                "canonical REMESH history append lost the schedule endpoint"
            )
        history_container_rebuilt = (
            appended_history[1] is not history_before[1]
        )
        history_transition = _build_history_transition(
            nodes=nodes,
            history_before=history_before,
            appended_history=appended_history,
            appended_epi=pre_remesh_epi.epi_values,
            tau_local=remesh_configuration.tau_local,
            tau_global=remesh_configuration.tau_global,
            history_maxlen=history_append.history_maxlen,
            history_container_rebuilt=history_container_rebuilt,
            oldest_snapshot_evicted=(
                history_append.oldest_snapshot_evicted
            ),
        )
        appended_history_surface = _snapshot_graph_surface(
            graph,
            "_epi_hist",
        )

        channels_before_remesh_apply = _snapshot_alias_channels(
            graph,
            nodes,
        )
        edges_before_remesh_apply = _snapshot_edge_state(graph)
        remesh_result = apply_network_remesh(
            graph,
            include_stability_evidence=True,
            metric_weights=weights,
        )
        after_remesh_read_state = _read_only_graph_state(graph)
        if not _contract_values_equal(
            _snapshot_alias_channels(graph, nodes),
            channels_before_remesh_apply,
        ):
            raise TNFRValueError(
                "The REMESH wrapper changed protected non-EPI alias channels."
            )
        if not _contract_values_equal(
            _snapshot_edge_state(graph),
            edges_before_remesh_apply,
        ):
            raise TNFRValueError(
                "The REMESH wrapper changed protected edge state."
            )
        _require_node_order(graph, nodes, boundary="after_remesh")
        _require_same_graph_surface(
            graph,
            "_epi_hist",
            appended_history_surface,
        )
        _require_runtime_clock(
            graph,
            event_result.final_time,
            boundary="event_remesh.after_remesh",
        )
        _require_same_graph_surface(
            graph,
            "hybrid_event_log",
            committed_event_log,
        )
        _require_same_pressure_hook(
            graph,
            pressure_hook,
            boundary="after_remesh",
        )
        _require_same_configuration(
            graph,
            remesh_configuration_signature,
            boundary="after_remesh",
        )
        phase_after_remesh = _require_same_phase(
            graph,
            nodes,
            phase_before_remesh,
            boundary="after_remesh",
        )
        if not _ordered_identity_is(remesh_result.plan.node_order, nodes):
            raise RuntimeError("REMESH plan lost the frozen node order")
        if remesh_result.applied:
            if type(remesh_result.evidence) is not DelayedRemeshStabilityEvidence:
                raise RuntimeError(
                    "applied REMESH omitted exact stability evidence"
                )
            if remesh_result.evidence.metric_weights != weights:
                raise RuntimeError(
                    "REMESH evidence changed the declared metric"
                )

        post_remesh_epi = _observe_epi(graph, nodes, weights)
        if remesh_result.applied:
            if not remesh_result.epi_time_boundary_recorded:
                raise RuntimeError(
                    "applied REMESH omitted its physical EPI-time boundary"
                )
            epi_time_histories = _require_epi_time_right_endpoint(
                graph,
                nodes,
                runtime_time=event_result.final_time,
                epi_values=post_remesh_epi.epi_values,
            )
        else:
            if remesh_result.epi_time_boundary_recorded:
                raise RuntimeError("REMESH no-op recorded a physical boundary")
            epi_time_histories = None
        capacity_after_remesh = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_VF,
            "structural frequency",
            nonnegative=True,
        )
        pressure_before_optional_refresh = _required_scalar_channel(
            graph,
            nodes,
            ALIAS_DNFR,
            "DeltaNFR",
        )
        post_remesh_edges = _snapshot_edge_state(graph)
        post_remesh_metadata = _snapshot_graph_surface(graph, "_REMESH_META")
        post_remesh_alpha_source = _snapshot_graph_surface(
            graph, "_REMESH_ALPHA_SRC"
        )
        post_remesh_telemetry = _snapshot_graph_surface(graph, "history")
        _require_read_only_graph_state(
            graph,
            after_remesh_read_state,
            boundary="after_remesh_observations",
        )

        post_refresh_count = 0
        refresh_guard_failure: BaseException | None = None
        if refresh_pressure_after_remesh and remesh_result.applied:
            _require_same_pressure_hook(
                graph,
                pressure_hook,
                boundary="before_post_remesh_pressure_refresh",
            )
            try:
                _invoke_restricted_pressure_refresh(
                    graph,
                    expected_present=pressure_hook[0],
                    expected_callback=pressure_callback,
                    n_jobs=n_jobs,
                    boundary_label=(
                        "event_remesh.post_remesh_pressure_refresh"
                    ),
                    require_callback_state_preserved=False,
                )
            except TNFRValueError as failure:
                if failure.message != (
                    "A pressure callback changed non-pressure graph state."
                ):
                    raise
                refresh_guard_failure = failure
            except RuntimeError as failure:
                if failure.args != (
                    "configured pressure callback changed during event "
                    "execution",
                ):
                    raise
                refresh_guard_failure = failure
            post_refresh_count += 1
            after_refresh_read_state = _read_only_graph_state(graph)
            _require_node_order(
                graph,
                nodes,
                boundary="after_post_remesh_pressure_refresh",
            )
            _require_same_graph_surface(
                graph,
                "_epi_hist",
                appended_history_surface,
            )
            _require_runtime_clock(
                graph,
                event_result.final_time,
                boundary="event_remesh.after_post_remesh_pressure_refresh",
            )
            _require_same_graph_surface(
                graph,
                "hybrid_event_log",
                committed_event_log,
            )
            _require_same_pressure_hook(
                graph,
                pressure_hook,
                boundary="after_post_remesh_pressure_refresh",
            )
            _require_same_configuration(
                graph,
                remesh_configuration_signature,
                boundary="after_post_remesh_pressure_refresh",
            )
            if not _contract_values_equal(
                _snapshot_edge_state(graph), post_remesh_edges
            ):
                raise TNFRValueError(
                    "The post-REMESH pressure callback changed edge state."
                )
            _require_same_graph_surface(
                graph, "_REMESH_META", post_remesh_metadata
            )
            _require_same_graph_surface(
                graph, "_REMESH_ALPHA_SRC", post_remesh_alpha_source
            )
            _require_same_graph_surface(
                graph, "history", post_remesh_telemetry
            )
            if epi_time_histories is None:
                raise RuntimeError("applied REMESH lost its physical history")
            _require_same_epi_time_histories(graph, epi_time_histories)
            if not _contract_values_equal(
                _epi_values(graph, nodes),
                post_remesh_epi.epi_values,
            ):
                raise TNFRValueError(
                    "The post-REMESH pressure callback changed EPI."
                )
            refreshed_capacity = _required_scalar_channel(
                graph,
                nodes,
                ALIAS_VF,
                "structural frequency",
                nonnegative=True,
            )
            if not _contract_values_equal(
                refreshed_capacity,
                capacity_after_remesh,
            ):
                raise TNFRValueError(
                    "The post-REMESH pressure callback changed capacity."
                )
            phase_after_optional_refresh = _require_same_phase(
                graph,
                nodes,
                phase_before_remesh,
                boundary="after_optional_pressure_refresh",
            )
            pressure_after_optional_refresh = _required_scalar_channel(
                graph,
                nodes,
                ALIAS_DNFR,
                "DeltaNFR",
            )
            _require_read_only_graph_state(
                graph,
                after_refresh_read_state,
                boundary="after_post_remesh_pressure_refresh_observations",
            )
        else:
            phase_after_optional_refresh = phase_after_remesh
            pressure_after_optional_refresh = pressure_before_optional_refresh
        if refresh_guard_failure is not None:
            raise refresh_guard_failure
        schedule_drift = (
            pre_remesh_epi.exact_weighted_mean
            - pre_schedule_epi.exact_weighted_mean
        )
        remesh_drift = (
            post_remesh_epi.exact_weighted_mean
            - pre_remesh_epi.exact_weighted_mean
        )
        total_drift = (
            post_remesh_epi.exact_weighted_mean
            - pre_schedule_epi.exact_weighted_mean
        )
        return _sealed_event_remesh_cycle_result(
            target_nodes=nodes,
            metric_weights=weights,
            event_execution=event_result,
            remesh=remesh_result,
            history_transition=history_transition,
            pre_schedule_epi=pre_schedule_epi,
            pre_remesh_epi=pre_remesh_epi,
            post_remesh_epi=post_remesh_epi,
            exact_schedule_weighted_mean_drift=schedule_drift,
            exact_remesh_weighted_mean_drift=remesh_drift,
            exact_total_weighted_mean_drift=total_drift,
            schedule_weighted_mean_drift=_optional_diagnostic_float(
                schedule_drift
            ),
            remesh_weighted_mean_drift=_optional_diagnostic_float(
                remesh_drift
            ),
            total_weighted_mean_drift=_optional_diagnostic_float(total_drift),
            capacity_before_schedule=capacity_before,
            capacity_before_remesh=capacity_before_remesh,
            capacity_after_remesh=capacity_after_remesh,
            pressure_before_schedule=pressure_before,
            pressure_before_remesh=pressure_before_remesh,
            pressure_after_remesh_before_refresh=(
                pressure_before_optional_refresh
            ),
            pressure_after_optional_refresh=(
                pressure_after_optional_refresh
            ),
            phase_before_schedule=phase_before_schedule,
            phase_before_remesh=phase_before_remesh,
            phase_after_remesh_before_refresh=phase_after_remesh,
            phase_after_optional_refresh=phase_after_optional_refresh,
            schedule_capacity_changed=(
                capacity_before_remesh != capacity_before
            ),
            remesh_capacity_changed=(
                capacity_after_remesh != capacity_before_remesh
            ),
            history_length_before_cycle=history_length_before,
            history_length_before_append=(
                history_append.history_length_before
            ),
            history_length_after_append=(
                history_append.history_length_after
            ),
            history_maxlen=history_append.history_maxlen,
            history_container_rebuilt=history_container_rebuilt,
            history_oldest_snapshot_evicted=(
                history_append.oldest_snapshot_evicted
            ),
            schedule_pressure_refresh_callback_invocations=(
                event_result.pressure_refresh_callback_invocations
            ),
            post_remesh_pressure_refresh_requested=(
                refresh_pressure_after_remesh
            ),
            post_remesh_pressure_refresh_callback_invocations=(
                post_refresh_count
            ),
            committed_hybrid_event_log_length=(
                committed_event_log_length
            ),
            post_remesh_epi_time_boundary_recorded=(
                remesh_result.epi_time_boundary_recorded
            ),
        )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
