from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from tnfr.operators.event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence,
)
from tnfr.physics.runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
)


@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleBlockMarginObservation:
    source_execution: ExecutedEventRemeshCycleSequence = field(...)
    start_boundary: int
    boundary_count: int
    boundaries: tuple[RuntimeRemeshScheduleBoundaryObservation, ...] = field(...)
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_gain_based_energy_drop_lower_bound: Fraction
    exact_schedule_augmented_energy_gain_slack: Fraction
    exact_energy_drop: Fraction
    exact_gain_based_energy_drop_fraction_lower_bound: Fraction | None
    exact_observed_energy_drop_fraction: Fraction | None
    exact_endpoint_energy_gain_upper_bound: Fraction | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def block_observation_certified(self) -> bool: ...
    @property
    def same_graph_execution_provenance_certified(self) -> bool: ...
    @property
    def whole_sequence_graph_state_atomic(self) -> bool: ...
    @property
    def exact_finite_block_balance_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def energy_nonincrease_observed(self) -> bool: ...
    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool: ...
    @property
    def positive_normalized_block_margin_certified(self) -> bool: ...
    @property
    def strict_energy_contraction_observed(self) -> bool: ...
    @property
    def zero_energy_preservation_observed(self) -> bool: ...
    @property
    def uniform_class_coercivity_certified(self) -> bool: ...
    @property
    def uniform_repeated_margin_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def mesh_convergence_certified(self) -> bool: ...


def observe_executed_event_remesh_block_margin(
    execution: ExecutedEventRemeshCycleSequence,
    *,
    start_boundary: int = ...,
    boundary_count: int | None = ...,
) -> RuntimeRemeshScheduleBlockMarginObservation: ...


__all__: tuple[str, ...]
