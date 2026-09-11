from collections.abc import Hashable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from tnfr.operators.event_remesh_runtime import EventRemeshCycleResult
from tnfr.physics.remesh_history_stability import (
    ExactHistory,
    ExactVector,
    UniformRemeshHistoryTransitionObservation,
)

@dataclass(frozen=True, slots=True)
class RuntimeRemeshHistoryBridgeObservation:
    cycle_result: EventRemeshCycleResult = field(..., repr=False)
    exact_transition: UniformRemeshHistoryTransitionObservation = field(
        ..., repr=False
    )
    nodes: tuple[Hashable, ...]
    exact_metric_weights: ExactVector
    exact_history: ExactHistory
    exact_ideal_next_field: ExactVector
    exact_runtime_raw_next_field: ExactVector
    exact_runtime_bounded_next_field: ExactVector
    exact_rounding_residual: ExactVector
    exact_clipping_residual: ExactVector
    exact_total_residual: ExactVector
    exact_max_abs_rounding_residual: Fraction
    exact_max_abs_clipping_residual: Fraction
    exact_max_abs_total_residual: Fraction
    exact_runtime_raw_next_centered_field: ExactVector
    exact_runtime_bounded_next_centered_field: ExactVector
    exact_runtime_raw_next_energy: Fraction
    exact_runtime_bounded_next_energy: Fraction
    exact_lifted_augmented_energy_before: Fraction
    exact_lifted_ideal_augmented_energy_after: Fraction
    exact_lifted_runtime_raw_augmented_energy_after: Fraction
    exact_lifted_runtime_bounded_augmented_energy_after: Fraction
    exact_jensen_dissipation: Fraction
    exact_rounding_augmented_energy_defect: Fraction
    exact_clipping_augmented_energy_defect: Fraction
    exact_total_augmented_energy_defect: Fraction
    exact_lifted_runtime_raw_energy_drop: Fraction
    exact_lifted_runtime_bounded_energy_drop: Fraction
    exact_rounding_augmented_energy_defect_absolute_upper_bound: Fraction
    exact_clipping_augmented_energy_defect_absolute_upper_bound: Fraction
    exact_total_augmented_energy_defect_absolute_upper_bound: Fraction
    exact_lifted_runtime_raw_energy_drop_lower_bound: Fraction
    exact_lifted_runtime_bounded_energy_drop_lower_bound: Fraction
    exact_lifted_runtime_raw_barycenter_drift: ExactVector
    exact_lifted_runtime_bounded_barycenter_drift: ExactVector
    clipping_intervened: bool
    raw_binary64_replay_identified: bool
    bounded_binary64_replay_identified: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def bridge_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def lifted_runtime_raw_energy_nonincrease_observed(self) -> bool: ...
    @property
    def lifted_runtime_bounded_energy_nonincrease_observed(self) -> bool: ...
    @property
    def lifted_runtime_raw_energy_nonincrease_sufficiently_certified(self) -> bool: ...
    @property
    def lifted_runtime_bounded_energy_nonincrease_sufficiently_certified(
        self,
    ) -> bool: ...
    @property
    def hard_clipping_step_nonexpansive_certified(self) -> bool: ...
    @property
    def lifted_runtime_barycenter_preserved_observed(self) -> bool: ...
    @property
    def live_runtime_history_advance_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def companion_temporal_convergence_transferred_to_runtime(self) -> bool: ...
    @property
    def schedule_remesh_composition_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...

def observe_runtime_remesh_history_bridge(
    cycle_result: EventRemeshCycleResult,
) -> RuntimeRemeshHistoryBridgeObservation: ...

__all__: tuple[str, ...]
