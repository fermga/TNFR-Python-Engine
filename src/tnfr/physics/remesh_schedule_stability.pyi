from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any

from tnfr.physics.remesh_history_stability import (
    ExactVector,
    UniformRemeshHistoryTransitionObservation,
)

@dataclass(frozen=True, slots=True)
class RemeshScheduleHistoryStabilityObservation:
    transition: UniformRemeshHistoryTransitionObservation
    exact_runtime_raw_head: ExactVector
    exact_runtime_bounded_head: ExactVector
    exact_scheduled_head: ExactVector
    exact_runtime_raw_head_centered: ExactVector
    exact_runtime_bounded_head_centered: ExactVector
    exact_scheduled_head_centered: ExactVector
    exact_runtime_raw_head_energy: Fraction
    exact_runtime_bounded_head_energy: Fraction
    exact_scheduled_head_energy: Fraction
    exact_schedule_energy_gain_upper_bound: Fraction
    exact_schedule_energy_gain_slack: Fraction
    exact_schedule_augmented_energy_gain_slack: Fraction
    exact_raw_spatial_energy_defect: Fraction
    exact_clipping_spatial_energy_defect: Fraction
    exact_schedule_spatial_energy_defect: Fraction
    exact_total_spatial_energy_defect: Fraction
    exact_raw_augmented_energy_defect: Fraction
    exact_clipping_augmented_energy_defect: Fraction
    exact_schedule_augmented_energy_defect: Fraction
    exact_total_augmented_energy_defect: Fraction
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_schedule_contraction_augmented_margin: Fraction
    exact_gain_based_energy_drop_lower_bound: Fraction
    exact_stationary_history_barycenter_before: ExactVector
    exact_stationary_history_barycenter_after: ExactVector
    exact_stationary_history_barycenter_drift: ExactVector
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def transition_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_energy_balance_certified(self) -> bool: ...
    @property
    def schedule_energy_gain_bound_certified(self) -> bool: ...
    @property
    def energy_nonincrease_observed(self) -> bool: ...
    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool: ...
    @property
    def stationary_history_barycenter_preserved_observed(self) -> bool: ...
    @property
    def runtime_provenance_certified(self) -> bool: ...
    @property
    def repeated_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...

def observe_remesh_schedule_history_transition(
    transition: UniformRemeshHistoryTransitionObservation,
    runtime_raw_head: Iterable[Real],
    runtime_bounded_head: Iterable[Real],
    scheduled_head: Iterable[Real],
    schedule_energy_gain_upper_bound: Real,
) -> RemeshScheduleHistoryStabilityObservation: ...

__all__: tuple[str, ...]
