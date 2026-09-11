from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from tnfr.operators.event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence,
)
from tnfr.physics.remesh_history_stability import ExactVector
from tnfr.physics.remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
)
from tnfr.physics.runtime_remesh_schedule_block_margin import (
    RuntimeRemeshScheduleBlockMarginObservation,
)
from tnfr.physics.runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
)


@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleRelativeDefectBlockObservation:
    source_execution: ExecutedEventRemeshCycleSequence = field(...)
    relative_defect_certificate: (
        UniformRemeshScheduleRelativeDefectStabilityCertificate
    ) = field(...)
    block_observation: RuntimeRemeshScheduleBlockMarginObservation = field(...)
    start_boundary: int
    boundary_count: int
    boundaries: tuple[RuntimeRemeshScheduleBoundaryObservation, ...] = field(...)
    exact_common_normalized_metric: ExactVector
    exact_stationary_history_weights: ExactVector
    exact_input_history_energy_vectors: tuple[ExactVector, ...]
    exact_output_history_energy_vectors: tuple[ExactVector, ...]
    exact_remesh_input_energy_upper_bounds: tuple[Fraction, ...]
    exact_ideal_remesh_head_energies: tuple[Fraction, ...]
    exact_runtime_bounded_head_energies: tuple[Fraction, ...]
    exact_pre_schedule_energy_defects: tuple[Fraction, ...]
    exact_relative_energy_defect_ratios: tuple[Fraction | None, ...]
    exact_relative_energy_defect_slacks: tuple[Fraction, ...]
    exact_schedule_energy_gain_upper_bounds: tuple[Fraction, ...]
    exact_effective_head_energy_upper_bounds: tuple[Fraction, ...]
    exact_scheduled_head_energies: tuple[Fraction, ...]
    exact_energy_envelope_vectors: tuple[ExactVector, ...]
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_finite_endpoint_energy_gain_upper_bound: Fraction
    exact_finite_endpoint_energy_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def relative_defect_block_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def same_graph_execution_provenance_certified(self) -> bool: ...
    @property
    def whole_sequence_graph_state_atomic(self) -> bool: ...
    @property
    def every_boundary_relative_defect_bound_verified(self) -> bool: ...
    @property
    def runtime_relative_defect_bound_verified(self) -> bool: ...
    @property
    def runtime_schedule_maps_verified(self) -> bool: ...
    @property
    def exact_finite_energy_envelope_certified(self) -> bool: ...
    @property
    def exact_finite_endpoint_bound_certified(self) -> bool: ...
    @property
    def finite_energy_nonincrease_sufficiently_certified(self) -> bool: ...
    @property
    def finite_strict_energy_contraction_sufficiently_certified(self) -> bool: ...
    @property
    def zero_energy_preservation_observed(self) -> bool: ...
    @property
    def uniform_runtime_relative_defect_class_certified(self) -> bool: ...
    @property
    def runtime_forward_invariant_class_certified(self) -> bool: ...
    @property
    def runtime_forward_invariance_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def repeated_binary64_runtime_stability_certified(self) -> bool: ...
    @property
    def binary64_runtime_stability_certified(self) -> bool: ...
    @property
    def future_runtime_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...
    @property
    def runtime_global_gain_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def mesh_convergence_certified(self) -> bool: ...
    @property
    def adaptive_grammar_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...


def observe_executed_event_remesh_relative_defect_block(
    execution: ExecutedEventRemeshCycleSequence,
    certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    *,
    start_boundary: int = ...,
    boundary_count: int | None = ...,
) -> RuntimeRemeshScheduleRelativeDefectBlockObservation: ...


__all__: tuple[str, ...]
