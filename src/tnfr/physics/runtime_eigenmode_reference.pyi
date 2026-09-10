from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..operators.event_runtime import ExecutedPressureRefreshedFlowPartition
from .reversible_eigenmode_reference import (
    ReversibleSingleEigenmodeEulerReferenceCertificate,
)

ExactVector = tuple[Fraction, ...]
ExactPartition = tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class ExecutedReversibleSingleEigenmodeEulerPartitionObservation:
    reference_certificate: ReversibleSingleEigenmodeEulerReferenceCertificate = field(
        repr=False
    )
    execution: ExecutedPressureRefreshedFlowPartition = field(repr=False)
    partition_index: int
    nodes: tuple[Any, ...]
    exact_segment_durations: ExactPartition
    exact_runtime_boundary_epi: tuple[ExactVector, ...]
    exact_reference_boundary_epi: tuple[ExactVector, ...]
    exact_pressure_realization_residuals: tuple[ExactVector, ...]
    exact_held_input_execution_residuals: tuple[ExactVector, ...]
    exact_local_runtime_defects: tuple[ExactVector, ...]
    exact_endpoint_runtime_defect: ExactVector
    exact_endpoint_runtime_defect_linf: Fraction
    exact_runtime_minus_continuous_endpoint_lower_bound: ExactVector
    exact_runtime_minus_continuous_endpoint_upper_bound: ExactVector
    exact_runtime_continuous_linf_error_lower_bound: Fraction
    exact_runtime_continuous_linf_error_upper_bound: Fraction
    exact_runtime_continuous_h_energy_error_lower_bound: Fraction
    exact_runtime_continuous_h_energy_error_upper_bound: Fraction
    all_segment_exact_affine_maps_identified: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def runtime_partition_binding_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_runtime_defect_decomposition_certified(self) -> bool: ...
    @property
    def exact_continuous_error_enclosure_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ExecutedReversibleSingleEigenmodeEulerReferenceObservation:
    reference_certificate: ReversibleSingleEigenmodeEulerReferenceCertificate = field(
        repr=False
    )
    partition_observations: tuple[
        ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
        ...,
    ]
    nodes: tuple[Any, ...]
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def runtime_reference_binding_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_runtime_defect_decomposition_certified(self) -> bool: ...
    @property
    def exact_continuous_error_enclosures_certified(self) -> bool: ...
    @property
    def all_observed_endpoints_equal_exact_euler_reference(self) -> bool: ...
    @property
    def binary64_asymptotic_convergence_certified(self) -> bool: ...
    @property
    def runtime_mesh_convergence_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def future_or_repeated_stability_certified(self) -> bool: ...
    @property
    def common_causal_execution_provenance_certified(self) -> bool: ...
    @property
    def glyph_or_remesh_dynamics_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...


def observe_executed_reversible_single_eigenmode_euler_reference(
    partitions: Iterable[ExecutedPressureRefreshedFlowPartition],
) -> ExecutedReversibleSingleEigenmodeEulerReferenceObservation: ...


__all__: tuple[str, ...]
