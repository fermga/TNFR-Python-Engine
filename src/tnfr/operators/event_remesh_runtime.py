"""Atomic composition of one operator-event schedule and delayed REMESH.

The bridge owns one canonical pre-REMESH history sample. It preserves the
runtime indexing convention and passes one frozen positive diagonal metric to
the delayed-map evidence. Runtime flow accuracy and repeated history-updated
stability remain outside this execution contract.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from numbers import Real
from typing import Any

import networkx as nx

from .._remesh_contract import (
    materialize_delayed_remesh_configuration,
    materialize_positive_diagonal_metric,
)
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..errors import TNFRValueError
from ._epi_domain import require_real_scalar_epi
from .event_runtime import (
    OperatorEventExecutionResult,
    _require_runtime_clock,
    execute_operator_event_schedule,
)
from .event_timing import OperatorEventSchedule
from .network_stage import GraphTransactionSnapshot
from .remesh import (
    DelayedRemeshResult,
    DelayedRemeshStabilityEvidence,
    _contract_values_equal,
    _materialize_network_remesh_configuration,
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


def _exact_epi_vector(values: tuple[float, ...]) -> tuple[Fraction, ...]:
    """Return the exact rational values represented by one binary64 vector."""

    return tuple(Fraction.from_float(value) for value in values)


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
        _validate_history_transition_fields(**values)
        if (
            type(self._proof_stamp) is not tuple
            or self._proof_stamp != _history_transition_stamp(**values)
        ):
            raise ValueError(
                "REMESH history-transition proof fields are inconsistent"
            )

    def _proof_fields_are_intact(self) -> bool:
        """Fail closed after ordinary replacement or field mutation."""

        try:
            self.__post_init__()
        except (AttributeError, TypeError, ValueError, OverflowError):
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


def _safe_opaque_hash(value: Any) -> tuple[Any, ...]:
    """Return a non-raising hash signature for an otherwise opaque value."""

    try:
        return ("hash", hash(value))
    except BaseException as exc:
        return (
            "hash-unavailable",
            type(exc).__module__,
            type(exc).__qualname__,
        )


def _safe_opaque_repr(value: Any) -> str:
    """Return a non-raising representation for an otherwise opaque value."""

    try:
        return repr(value)
    except BaseException as exc:
        return (
            f"<repr-unavailable:{type(exc).__module__}."
            f"{type(exc).__qualname__}>"
        )


def _slot_storage_name(owner: type[Any], declared: str) -> str:
    if declared.startswith("__") and not declared.endswith("__"):
        return f"_{owner.__name__.lstrip('_')}{declared}"
    return declared


def _slotted_object_state(
    value: Any,
    *,
    seen: dict[int, int],
) -> tuple[Any, ...]:
    """Read declared slot storage in deterministic MRO/declaration order."""

    state: list[Any] = []
    for owner in type(value).__mro__:
        declared_slots = owner.__dict__.get("__slots__", ())
        if type(declared_slots) is str:
            slot_names = (declared_slots,)
        else:
            try:
                slot_names = tuple(declared_slots)
            except TypeError:
                slot_names = ()
        for declared in slot_names:
            if declared in ("__dict__", "__weakref__"):
                continue
            if type(declared) is not str:
                state.append(
                    (
                        owner.__module__,
                        owner.__qualname__,
                        _safe_opaque_repr(declared),
                        ("invalid-slot-name",),
                    )
                )
                continue
            storage_name = _slot_storage_name(owner, declared)
            descriptor = owner.__dict__.get(storage_name)
            if descriptor is None:
                slot_value = ("missing-descriptor",)
            else:
                try:
                    observed = descriptor.__get__(value, type(value))
                except AttributeError:
                    slot_value = ("unset",)
                except BaseException as exc:
                    slot_value = (
                        "unreadable",
                        type(exc).__module__,
                        type(exc).__qualname__,
                    )
                else:
                    slot_value = (
                        "value",
                        _proof_value(observed, seen=seen),
                    )
            state.append(
                (
                    owner.__module__,
                    owner.__qualname__,
                    declared,
                    slot_value,
                )
            )
    return tuple(state)


def _proof_value(
    value: Any,
    *,
    seen: dict[int, int] | None = None,
) -> Any:
    """Freeze decisive runtime evidence structurally rather than by identity."""

    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is Fraction:
        return ("fraction", value.numerator, value.denominator)
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if isinstance(value, Enum):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            _proof_value(value.value, seen=seen),
        )

    if seen is None:
        seen = {}
    identity = id(value)
    if identity in seen:
        return ("reference", seen[identity])
    seen[identity] = len(seen)

    if type(value) is tuple:
        return (
            "tuple",
            tuple(_proof_value(item, seen=seen) for item in value),
        )
    if type(value) is list:
        return (
            "list",
            tuple(_proof_value(item, seen=seen) for item in value),
        )
    if isinstance(value, deque):
        return (
            "deque",
            type(value).__module__,
            type(value).__qualname__,
            value.maxlen,
            tuple(_proof_value(item, seen=seen) for item in value),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    _proof_value(key, seen=seen),
                    _proof_value(item, seen=seen),
                )
                for key, item in value.items()
            ),
        )
    if isinstance(value, (set, frozenset)):
        members = tuple(
            _proof_value(item, seen=seen)
            for item in sorted(value, key=_safe_opaque_repr)
        )
        return (type(value).__name__, members)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    item.name,
                    _proof_value(
                        getattr(value, item.name),
                        seen=seen,
                    ),
                )
                for item in fields(value)
                if item.name != "_proof_stamp"
            ),
        )
    namespace = getattr(value, "__dict__", None)
    namespace_state = (
        tuple(
            (
                name,
                _proof_value(item, seen=seen),
            )
            for name, item in namespace.items()
        )
        if isinstance(namespace, Mapping)
        else ()
    )
    slot_state = _slotted_object_state(value, seen=seen)
    if isinstance(namespace, Mapping) or slot_state:
        return (
            "object-state",
            type(value).__module__,
            type(value).__qualname__,
            namespace_state,
            slot_state,
        )
    return (
        "opaque",
        type(value).__module__,
        type(value).__qualname__,
        _safe_opaque_hash(value),
        _safe_opaque_repr(value),
    )


def _nested_proof_records_are_intact(
    value: Any,
    *,
    seen: set[int] | None = None,
) -> bool:
    """Require every nested sealed record to retain its executor-owned proof."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)

    validator = getattr(value, "_proof_fields_are_intact", None)
    if callable(validator):
        try:
            if not bool(validator()):
                return False
        except Exception:
            return False
    if is_dataclass(value) and not isinstance(value, type):
        return all(
            _nested_proof_records_are_intact(
                getattr(value, item.name),
                seen=seen,
            )
            for item in fields(value)
            if item.name != "_proof_stamp"
        )
    if isinstance(value, Mapping):
        return all(
            _nested_proof_records_are_intact(key, seen=seen)
            and _nested_proof_records_are_intact(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, deque, set, frozenset)):
        return all(
            _nested_proof_records_are_intact(item, seen=seen)
            for item in value
        )
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
    if observation.nodes != nodes or observation.metric_weights != weights:
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
        or transition.nodes != nodes
    ):
        raise ValueError("cycle history-transition proof is not intact")

    if type(result.event_execution) is not OperatorEventExecutionResult:
        raise TypeError(
            "event_execution must be an OperatorEventExecutionResult"
        )
    if result.event_execution.target_nodes != nodes:
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
        or plan.node_order != nodes
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
                proposal.node != nodes[index]
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
        _validate_event_remesh_cycle_result(self)
        values = _cycle_result_fields(self)
        if (
            type(self._proof_stamp) is not tuple
            or self._proof_stamp != _cycle_result_stamp(values)
        ):
            raise ValueError(
                "event/REMESH cycle proof fields are inconsistent"
            )

    def _proof_fields_are_intact(self) -> bool:
        """Fail closed after changes to decisive nested or boundary state."""

        try:
            self.__post_init__()
        except Exception:
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

        return self.remesh.applied

    @property
    def post_remesh_pressure_refresh_performed(self) -> bool:
        """Whether the explicit post-map pressure callback completed."""

        return self.post_remesh_pressure_refresh_callback_invocations == 1


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
    if not isinstance(
        graph,
        (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
    ):
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
    for node in nodes:
        raw = get_attr(
            graph.nodes[node],
            aliases,
            _MISSING,
            strict=True,
            conv=lambda value: value,
        )
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
    for node in nodes:
        raw = get_attr(
            graph.nodes[node],
            ALIAS_EPI,
            _MISSING,
            strict=True,
            conv=lambda value: value,
        )
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
    if "_epi_hist" not in graph.graph:
        return False, _MISSING, ()
    raw = graph.graph["_epi_hist"]
    if raw is None:
        return True, raw, ()
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        raise TNFRValueError(
            "_epi_hist must be a replayable indexed history"
        )
    if not hasattr(raw, "__len__") or not hasattr(raw, "__getitem__"):
        raise TNFRValueError(
            "_epi_hist must be a replayable indexed history"
        )
    try:
        snapshots = tuple(raw)
    except (OverflowError, TypeError) as exc:
        raise TNFRValueError(
            "_epi_hist must be a replayable indexed history"
        ) from exc

    node_set = frozenset(nodes)
    signature: list[tuple[float, ...]] = []
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise TNFRValueError(
                f"_epi_hist[{index}] must be a node-to-EPI mapping"
            )
        keys = tuple(snapshot)
        if len(keys) != len(nodes) or frozenset(keys) != node_set:
            raise TNFRValueError(
                f"_epi_hist[{index}] support must equal the initial node support"
            )
        signature.append(
            tuple(
                require_real_scalar_epi(
                    snapshot[node],
                    operator="Recursivity",
                    label=f"_epi_hist[{index}][{node!r}]",
                )
                for node in nodes
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
) -> None:
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


def _require_node_order(
    graph: nx.Graph,
    expected: tuple[Hashable, ...],
    *,
    boundary: str,
) -> None:
    observed = tuple(graph.nodes)
    if observed != expected:
        raise TNFRValueError(
            "The event/REMESH cycle requires fixed ordered node support.",
            context={
                "boundary": boundary,
                "expected_nodes": expected,
                "observed_nodes": observed,
            },
        )


def _pressure_hook_signature(graph: nx.Graph) -> tuple[bool, Any]:
    return "compute_delta_nfr" in graph.graph, graph.graph.get("compute_delta_nfr")


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
    expected: Any,
    *,
    boundary: str,
) -> None:
    if _materialize_network_remesh_configuration(graph) != expected:
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
    if observed != expected:
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

    expected_records = [] if not before[0] else deepcopy(before[2])
    expected_records.extend(event.as_record() for event in event_result.events)
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
    if not isinstance(observed[1], list) or not _contract_values_equal(
        observed[2], expected_records
    ):
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
    transaction while retaining the primary exception.
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

    nodes = tuple(graph.nodes)
    if not nodes:
        raise TNFRValueError(
            "event/REMESH composition requires nonempty node support"
        )
    weights = materialize_positive_diagonal_metric(
        metric_weights,
        nodes,
    )
    history_before = _history_signature(graph, nodes)
    history_length_before = len(history_before[2])
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
    transaction = GraphTransactionSnapshot(graph)

    try:
        event_result = execute_operator_event_schedule(
            graph,
            schedule,
            context=context,
            method=method,
            n_jobs=n_jobs,
            suppress_birth_warnings=suppress_birth_warnings,
            include_flow_certificates=include_flow_certificates,
            include_stage_certificates=include_stage_certificates,
        )
        if event_result.target_nodes != nodes:
            raise RuntimeError(
                "event execution lost the bridge's frozen target order"
            )
        _require_node_order(graph, nodes, boundary="after_schedule")
        _require_same_history(
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
            remesh_configuration,
            boundary="after_schedule",
        )
        committed_event_log = _require_schedule_event_log_commit(
            graph,
            event_log_before,
            event_result,
        )

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

        from ..dynamics.remesh_history import (
            append_remesh_epi_history_snapshot,
        )

        history_append = append_remesh_epi_history_snapshot(graph)
        appended_history = _history_signature(graph, nodes)
        if (
            not appended_history[0]
            or type(appended_history[1]) is not deque
            or appended_history[2][-1] != pre_remesh_epi.epi_values
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
        _require_same_history(
            graph,
            nodes,
            appended_history,
            boundary="after_remesh",
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
            remesh_configuration,
            boundary="after_remesh",
        )
        phase_after_remesh = _require_same_phase(
            graph,
            nodes,
            phase_before_remesh,
            boundary="after_remesh",
        )
        if remesh_result.plan.node_order != nodes:
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

        post_refresh_count = 0
        if refresh_pressure_after_remesh and remesh_result.applied:
            _require_same_pressure_hook(
                graph,
                pressure_hook,
                boundary="before_post_remesh_pressure_refresh",
            )
            assert callable(pressure_callback)
            pressure_callback(graph)
            post_refresh_count += 1
            _require_node_order(
                graph,
                nodes,
                boundary="after_post_remesh_pressure_refresh",
            )
            _require_same_history(
                graph,
                nodes,
                appended_history,
                boundary="after_post_remesh_pressure_refresh",
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
                remesh_configuration,
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
            if _epi_values(graph, nodes) != post_remesh_epi.epi_values:
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
            if refreshed_capacity != capacity_after_remesh:
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
                0 if not committed_event_log[0] else len(committed_event_log[2])
            ),
            post_remesh_epi_time_boundary_recorded=(
                remesh_result.epi_time_boundary_recorded
            ),
        )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
