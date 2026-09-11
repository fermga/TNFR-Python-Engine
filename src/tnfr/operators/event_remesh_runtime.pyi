from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import networkx as nx

from .event_runtime import OperatorEventExecutionResult
from .event_timing import OperatorEventSchedule, PhysicalFlowPartition
from .remesh import DelayedRemeshResult


@dataclass(frozen=True, slots=True)
class RemeshHistoryTransitionObservation:
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

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def canonical_history_transition_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class WeightedEPIObservation:
    nodes: tuple[Hashable, ...]
    epi_values: tuple[float, ...]
    metric_weights: tuple[float, ...]
    exact_weighted_mean: Fraction
    weighted_mean: float
    exact_disagreement_energy: Fraction
    disagreement_energy: float | None


@dataclass(frozen=True, slots=True)
class EventRemeshCycleResult:
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
    history_convention: str = field(default=..., init=False)
    schedule_left_history_unchanged: bool = field(default=..., init=False)
    whole_cycle_graph_state_atomic: bool = field(default=..., init=False)
    common_metric_is_frozen: bool = field(default=..., init=False)
    schedule_endpoint_clock_preserved: bool = field(default=..., init=False)
    hybrid_event_log_preserved: bool = field(default=..., init=False)
    pressure_hook_identity_preserved: bool = field(default=..., init=False)
    remesh_configuration_frozen: bool = field(default=..., init=False)
    scope: str = field(default=..., init=False)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def remesh_history_repetition_certified(self) -> bool: ...
    @property
    def mixed_runtime_gain_certified(self) -> bool: ...
    @property
    def external_side_effects_rolled_back(self) -> bool: ...
    @property
    def remesh_applied(self) -> bool: ...
    @property
    def post_remesh_pressure_refresh_performed(self) -> bool: ...


def execute_event_remesh_cycle(
    graph: nx.Graph,
    schedule: OperatorEventSchedule,
    *,
    metric_weights: (
        Mapping[Hashable, Any] | Sequence[Any] | None
    ) = ...,
    refresh_pressure_after_remesh: bool = ...,
    context: Mapping[str, Any] | None = ...,
    method: str | None = ...,
    n_jobs: int | None = ...,
    suppress_birth_warnings: bool = ...,
    include_flow_certificates: bool = ...,
    include_stage_certificates: bool = ...,
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = ...,
) -> EventRemeshCycleResult: ...


__all__: list[str]
