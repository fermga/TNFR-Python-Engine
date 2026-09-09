from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, TypeAlias

from .event_remesh_runtime import EventRemeshCycleResult
from .event_runtime import ObservedRepresentedEPIScheduleComposition
from .remesh import DelayedRemeshResult

ExactVector: TypeAlias = tuple[Fraction, ...]
ExactHistory: TypeAlias = tuple[ExactVector, ...]


@dataclass(frozen=True, slots=True)
class EventRemeshCycleBoundaryObservation:
    boundary_index: int
    left_cycle_index: int
    right_cycle_index: int
    left_nodes: tuple[Hashable, ...]
    right_nodes: tuple[Hashable, ...]
    exact_left_schedule_end_time: Fraction
    exact_right_schedule_start_time: Fraction
    exact_left_pre_remesh_epi: ExactVector
    exact_left_post_remesh_epi: ExactVector
    exact_right_pre_schedule_epi: ExactVector
    left_outgoing_exact_history: ExactHistory
    right_incoming_exact_history: ExactHistory
    exact_left_capacity_after_remesh: ExactVector
    exact_right_capacity_before_schedule: ExactVector
    exact_left_pressure_after_optional_refresh: ExactVector
    exact_right_pressure_before_schedule: ExactVector
    exact_left_phase_after_optional_refresh: ExactVector
    exact_right_phase_before_schedule: ExactVector
    left_cycle_proof_intact: bool
    right_cycle_proof_intact: bool
    left_cycle_graph_state_atomic: bool
    right_cycle_graph_state_atomic: bool
    left_remesh_applied: bool
    left_post_remesh_pressure_refresh_requested: bool
    left_post_remesh_pressure_refresh_callback_invocations: int
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    scope: str = field(default=..., init=False)

    @property
    def exact_recorded_state_continuity_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ObservedEventRemeshCycleSequence:
    cycle_indices: tuple[int, ...]
    cycles: tuple[EventRemeshCycleResult, ...]
    boundaries: tuple[EventRemeshCycleBoundaryObservation, ...]
    schedule_compositions: tuple[
        ObservedRepresentedEPIScheduleComposition | None, ...
    ]
    remesh_results: tuple[DelayedRemeshResult, ...]
    cycle_exact_metric_weights: tuple[ExactVector, ...]
    cycle_exact_normalized_metric_rays: tuple[ExactVector, ...]
    exact_common_normalized_metric_ray: ExactVector | None
    per_cycle_proof_fields_intact: tuple[bool, ...]
    per_cycle_graph_state_atomic: tuple[bool, ...]
    per_cycle_metric_bound: tuple[bool, ...]
    raw_metric_weights_equal: bool
    nested_schedule_metric_alignment: tuple[bool | None, ...]
    remesh_configurations_equal: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    scope: str = field(default=..., init=False)

    @property
    def exact_recorded_boundary_continuity_certified(self) -> bool: ...
    @property
    def exact_common_metric_cycle_sequence_certified(self) -> bool: ...
    @property
    def all_nested_schedule_metrics_aligned(self) -> bool | None: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def mixed_schedule_remesh_gain_certified(self) -> bool: ...
    @property
    def evolving_history_remesh_gain_certified(self) -> bool: ...
    @property
    def evolving_history_remesh_repetition_certified(self) -> bool: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def future_cycle_stability_certified(self) -> bool: ...
    @property
    def whole_sequence_atomicity_certified(self) -> bool: ...
    @property
    def full_graph_state_continuity_certified(self) -> bool: ...
    @property
    def grammar_history_continuity_certified(self) -> bool: ...
    @property
    def shared_graph_execution_provenance_certified(self) -> bool: ...


def compose_event_remesh_cycle_observations(
    cycles: Iterable[EventRemeshCycleResult],
) -> ObservedEventRemeshCycleSequence: ...
