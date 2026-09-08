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
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any

import networkx as nx

from .._remesh_contract import (
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
    _contract_values_equal,
    _materialize_network_remesh_configuration,
    _require_same_epi_time_histories,
    _require_same_graph_surface,
    _snapshot_edge_state,
    _snapshot_epi_time_histories,
    _snapshot_graph_surface,
    apply_network_remesh,
)

__all__ = [
    "EventRemeshCycleResult",
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
    "integrator side effects are not rolled back. No "
    "field certifies solver accuracy, mixed-word gain, or stability under "
    "repeated cycles with evolving history."
)
_HISTORY_CONVENTION = (
    "append_pre_remesh_snapshot_then_read_delay_at_history[-(tau+1)]"
)
_MISSING = object()


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


@dataclass(frozen=True, slots=True)
class EventRemeshCycleResult:
    """Immutable evidence for one committed schedule/history/REMESH cycle."""

    target_nodes: tuple[Hashable, ...]
    metric_weights: tuple[float, ...]
    event_execution: OperatorEventExecutionResult
    remesh: DelayedRemeshResult
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
    remesh_history_repetition_certified: bool = field(
        default=False,
        init=False,
    )
    mixed_runtime_gain_certified: bool = field(default=False, init=False)
    external_side_effects_rolled_back: bool = field(
        default=False,
        init=False,
    )
    scope: str = field(default=_SCOPE, init=False)

    @property
    def remesh_applied(self) -> bool:
        """Whether the delayed map committed in this cycle."""

        return self.remesh.applied

    @property
    def post_remesh_pressure_refresh_performed(self) -> bool:
        """Whether the explicit post-map pressure callback completed."""

        return self.post_remesh_pressure_refresh_callback_invocations == 1


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
    transaction = GraphTransactionSnapshot(graph)

    try:
        event_result = execute_operator_event_schedule(
            graph,
            schedule,
            context=context,
            method=method,
            n_jobs=n_jobs,
            suppress_birth_warnings=suppress_birth_warnings,
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

        remesh_result = apply_network_remesh(
            graph,
            include_stability_evidence=True,
            metric_weights=weights,
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
        if (
            remesh_result.evidence is not None
            and remesh_result.evidence.metric_weights != weights
        ):
            raise RuntimeError("REMESH evidence changed the declared metric")

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
        return EventRemeshCycleResult(
            target_nodes=nodes,
            metric_weights=weights,
            event_execution=event_result,
            remesh=remesh_result,
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
            history_container_rebuilt=(
                appended_history[1] is not history_before[1]
            ),
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
