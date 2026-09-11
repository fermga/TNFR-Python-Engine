from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any

from ..operators.event_remesh_runtime import EventRemeshCycleResult
from ..operators.event_runtime import ObservedRepresentedEPIScheduleComposition
from ..operators.remesh import DelayedRemeshResult
from .event_refinement import ExecutedEventLocalZHIRPhysicalPrejumpObservation


@dataclass(frozen=True, slots=True)
class EventRemeshEPICheckpointObservation:
    mesh_name: str
    checkpoint_kind: str
    parent_interval_index: int | None
    boundary_index: int | None
    time: float
    exact_time: Fraction
    nodes: tuple[Hashable, ...]
    epi_values: tuple[float, ...]
    exact_epi_values: tuple[Fraction, ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def checkpoint_observation_certified(self) -> bool: ...
    @property
    def key(self) -> tuple[str, int | None, Fraction]: ...


@dataclass(frozen=True, slots=True)
class EventRemeshMeshObservation:
    mesh_name: str
    cycle_result: EventRemeshCycleResult = field(repr=False)
    nodes: tuple[Hashable, ...]
    physical_partition_interval_indices: tuple[int, ...]
    exact_partition_boundary_times: tuple[
        tuple[int, tuple[Fraction, ...]], ...
    ]
    checkpoints: tuple[EventRemeshEPICheckpointObservation, ...]
    schedule_composition: ObservedRepresentedEPIScheduleComposition | None = field(
        repr=False
    )
    remesh_result: DelayedRemeshResult = field(repr=False)
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def mesh_observation_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EventRemeshPersistentEPIError:
    left_mesh: str
    right_mesh: str
    checkpoint_kind: str
    parent_interval_index: int | None
    exact_time: Fraction
    nodes: tuple[Hashable, ...]
    exact_left_epi: tuple[Fraction, ...]
    exact_right_epi: tuple[Fraction, ...]
    exact_absolute_epi_errors: tuple[Fraction, ...]
    exact_epi_error_linf: Fraction
    epi_error_linf: float | None
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def error_observation_certified(self) -> bool: ...
    @property
    def checkpoint_key(self) -> tuple[str, int | None, Fraction]: ...


@dataclass(frozen=True, slots=True)
class EventRemeshThreeMeshZHIRObservation:
    event_identity: tuple[int, int, int, str, str]
    nodes: tuple[Hashable, ...]
    coarse_observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    intermediate_observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    fine_observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    coarse_terminal_exact_binary64_rates: tuple[Fraction, ...]
    intermediate_terminal_exact_binary64_rates: tuple[Fraction, ...]
    fine_terminal_exact_binary64_rates: tuple[Fraction, ...]
    coarse_terminal_gate_decisions: tuple[bool, ...]
    intermediate_terminal_gate_decisions: tuple[bool, ...]
    fine_terminal_gate_decisions: tuple[bool, ...]
    terminal_gate_decisions_agree_by_node: tuple[bool, ...]
    terminal_gate_decisions_agree: bool
    exact_coarse_intermediate_terminal_rate_error_linf: Fraction
    exact_intermediate_fine_terminal_rate_error_linf: Fraction
    exact_coarse_fine_terminal_rate_error_linf: Fraction
    coarse_whole_parent_exact_binary64_rates: tuple[Fraction, ...]
    intermediate_whole_parent_exact_binary64_rates: tuple[Fraction, ...]
    fine_whole_parent_exact_binary64_rates: tuple[Fraction, ...]
    coarse_whole_parent_gate_decisions: tuple[bool, ...]
    intermediate_whole_parent_gate_decisions: tuple[bool, ...]
    fine_whole_parent_gate_decisions: tuple[bool, ...]
    whole_parent_gate_decisions_agree_by_node: tuple[bool, ...]
    whole_parent_gate_decisions_agree: bool
    exact_coarse_intermediate_whole_parent_rate_error_linf: Fraction
    exact_intermediate_fine_whole_parent_rate_error_linf: Fraction
    exact_coarse_fine_whole_parent_rate_error_linf: Fraction
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def physical_zhir_comparison_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EventRemeshThreeMeshModalObservation:
    parent_interval_index: int
    applicable: bool
    abstention_reason: str | None
    coarse_exact_segment_durations: tuple[Fraction, ...]
    intermediate_exact_segment_durations: tuple[Fraction, ...]
    fine_exact_segment_durations: tuple[Fraction, ...]
    nodes: tuple[Hashable, ...] | None
    common_decay_rates: tuple[float, ...] | None
    coarse_composed_modal_factors: tuple[float, ...] | None
    intermediate_composed_modal_factors: tuple[float, ...] | None
    fine_composed_modal_factors: tuple[float, ...] | None
    coarse_segment_stability_decisions: tuple[bool, ...] | None
    intermediate_segment_stability_decisions: tuple[bool, ...] | None
    fine_segment_stability_decisions: tuple[bool, ...] | None
    coarse_composed_stable: bool | None
    intermediate_composed_stable: bool | None
    fine_composed_stable: bool | None
    composed_stability_decisions_agree: bool | None
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def common_generator_modal_factors_observed(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EventRemeshThreeMeshRefinementObservation:
    coarse: EventRemeshMeshObservation
    intermediate: EventRemeshMeshObservation
    fine: EventRemeshMeshObservation
    persistent_nodes: tuple[Hashable, ...]
    supports_equal_across_meshes: bool
    coarse_to_intermediate_strict_refinement: bool
    intermediate_to_fine_strict_refinement: bool
    coarse_intermediate_epi_errors: tuple[EventRemeshPersistentEPIError, ...]
    intermediate_fine_epi_errors: tuple[EventRemeshPersistentEPIError, ...]
    coarse_fine_epi_errors: tuple[EventRemeshPersistentEPIError, ...]
    zhir_xi: float | None
    zhir_observations: tuple[EventRemeshThreeMeshZHIRObservation, ...]
    zhir_abstention_reason: str | None
    modal_observations: tuple[EventRemeshThreeMeshModalObservation, ...]
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def three_mesh_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def schedule_compositions(
        self,
    ) -> tuple[ObservedRepresentedEPIScheduleComposition | None, ...]: ...
    @property
    def remesh_results(self) -> tuple[DelayedRemeshResult, ...]: ...
    @property
    def maximum_coarse_intermediate_epi_error_linf(self) -> Fraction | None: ...
    @property
    def maximum_intermediate_fine_epi_error_linf(self) -> Fraction | None: ...
    @property
    def maximum_coarse_fine_epi_error_linf(self) -> Fraction | None: ...
    @property
    def intermediate_fine_error_decreases_at_coarse_checkpoints(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def mesh_convergence_certified(self) -> bool: ...
    @property
    def lyapunov_decrease_certified(self) -> bool: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def future_or_repeated_behavior_certified(self) -> bool: ...
    @property
    def combined_schedule_remesh_gain_certified(self) -> bool: ...
    @property
    def whole_three_mesh_atomicity_certified(self) -> bool: ...
    @property
    def complete_reference_problem_certified(self) -> bool: ...
    @property
    def epi_differences_attributable_only_to_mesh_certified(self) -> bool: ...


def observe_event_remesh_three_mesh_refinement(
    coarse: EventRemeshCycleResult,
    intermediate: EventRemeshCycleResult,
    fine: EventRemeshCycleResult,
    *,
    zhir_xi: Real | None = ...,
) -> EventRemeshThreeMeshRefinementObservation: ...


__all__: tuple[str, ...]
