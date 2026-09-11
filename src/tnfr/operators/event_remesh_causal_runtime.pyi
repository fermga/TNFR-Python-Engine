from __future__ import annotations

from fractions import Fraction
from typing import Any, Hashable, Iterable, Mapping, Sequence

import networkx as nx

from ..physics.runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleSequenceObservation,
)
from .event_remesh_runtime import EventRemeshCycleResult
from .event_remesh_sequence import ObservedEventRemeshCycleSequence
from .event_timing import OperatorEventSchedule, PhysicalFlowPartition


class EventRemeshCycleExecutionSpec:
    schedule: OperatorEventSchedule
    physical_flow_partitions: Iterable[PhysicalFlowPartition]
    def __init__(
        self,
        schedule: OperatorEventSchedule,
        physical_flow_partitions: Iterable[PhysicalFlowPartition] = ...,
    ) -> None: ...


class CausalEventRemeshCycleReceipt:
    cycle_index: int
    spec: EventRemeshCycleExecutionSpec
    schedule: OperatorEventSchedule
    physical_flow_partitions: tuple[PhysicalFlowPartition, ...]
    cycle_result: EventRemeshCycleResult
    target_nodes: tuple[Hashable, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    @property
    def receipt_binding_certified(self) -> bool: ...


class ExecutedEventRemeshCycleSequence:
    cycle_indices: tuple[int, ...]
    specs: tuple[EventRemeshCycleExecutionSpec, ...]
    schedules: tuple[OperatorEventSchedule, ...]
    physical_flow_partitions_by_cycle: tuple[
        tuple[PhysicalFlowPartition, ...], ...
    ]
    receipts: tuple[CausalEventRemeshCycleReceipt, ...]
    cycles: tuple[EventRemeshCycleResult, ...]
    target_nodes: tuple[Hashable, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    observed_sequence: ObservedEventRemeshCycleSequence
    runtime_telescope_required: bool
    runtime_telescope: RuntimeRemeshScheduleSequenceObservation | None
    conditions: tuple[tuple[str, bool], ...]
    scope: str
    @property
    def causal_cycle_order_certified(self) -> bool: ...
    @property
    def same_graph_execution_provenance_certified(self) -> bool: ...
    @property
    def whole_sequence_graph_state_atomic(self) -> bool: ...
    @property
    def exact_recorded_boundary_continuity_certified(self) -> bool: ...
    @property
    def exact_finite_energy_telescope_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def mesh_convergence_certified(self) -> bool: ...
    @property
    def adaptive_u2_u4_policy_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...
    @property
    def full_graph_state_continuity_certified(self) -> bool: ...
    @property
    def grammar_history_continuity_certified(self) -> bool: ...
    @property
    def concurrent_writer_atomicity_certified(self) -> bool: ...
    @property
    def cryptographic_or_durable_provenance_certified(self) -> bool: ...
    @property
    def external_side_effects_rolled_back(self) -> bool: ...


def execute_event_remesh_cycle_sequence(
    graph: nx.Graph,
    specs: Iterable[EventRemeshCycleExecutionSpec],
    *,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = ...,
    context: Mapping[str, Any] | None = ...,
    method: str | None = ...,
    n_jobs: int | None = ...,
    suppress_birth_warnings: bool = ...,
    require_runtime_telescope: bool = ...,
) -> ExecutedEventRemeshCycleSequence: ...


__all__: tuple[str, ...]
