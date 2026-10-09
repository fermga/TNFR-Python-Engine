from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from tnfr.operators.event_remesh_sequence import ObservedEventRemeshCycleSequence
from tnfr.operators.event_runtime import ObservedRepresentedEPIScheduleComposition
from tnfr.physics.remesh_history_stability import (
    ExactHistory,
    ExactVector,
    UniformRemeshHistoryTransitionObservation,
)
from tnfr.physics.remesh_schedule_stability import (
    RemeshScheduleHistoryStabilityObservation,
)
from tnfr.physics.runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
)

@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleBoundaryObservation:
    source_sequence: ObservedEventRemeshCycleSequence = field(...)
    boundary_index: int
    runtime_bridge: RuntimeRemeshHistoryBridgeObservation = field(...)
    exact_transition: UniformRemeshHistoryTransitionObservation = field(...)
    schedule_composition: ObservedRepresentedEPIScheduleComposition = field(...)
    schedule_balance: RemeshScheduleHistoryStabilityObservation = field(...)
    exact_common_normalized_metric: ExactVector
    exact_schedule_input_head: ExactVector
    exact_scheduled_head: ExactVector
    exact_scheduled_post_history: ExactHistory
    exact_next_cycle_history: ExactHistory
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_gain_based_energy_drop_lower_bound: Fraction
    exact_schedule_augmented_energy_gain_slack: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def boundary_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_recorded_history_advance_certified(self) -> bool: ...
    @property
    def exact_finite_schedule_remesh_balance_certified(self) -> bool: ...
    @property
    def energy_nonincrease_observed(self) -> bool: ...
    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool: ...
    @property
    def shared_graph_execution_provenance_certified(self) -> bool: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleSequenceObservation:
    source_sequence: ObservedEventRemeshCycleSequence = field(...)
    boundaries: tuple[RuntimeRemeshScheduleBoundaryObservation, ...]
    exact_common_normalized_metric: ExactVector
    exact_augmented_energy_initial: Fraction
    exact_augmented_energy_final: Fraction
    exact_total_energy_drop: Fraction
    exact_total_gain_based_energy_drop_lower_bound: Fraction
    exact_total_schedule_augmented_energy_gain_slack: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def sequence_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_recorded_history_advance_certified(self) -> bool: ...
    @property
    def exact_finite_energy_telescope_certified(self) -> bool: ...
    @property
    def energy_nonincrease_observed(self) -> bool: ...
    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool: ...
    @property
    def every_boundary_energy_nonincrease_observed(self) -> bool: ...
    @property
    def shared_graph_execution_provenance_certified(self) -> bool: ...
    @property
    def whole_sequence_atomicity_certified(self) -> bool: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...

def observe_runtime_remesh_schedule_sequence(
    sequence: ObservedEventRemeshCycleSequence,
) -> RuntimeRemeshScheduleSequenceObservation: ...

__all__: tuple[str, ...]
