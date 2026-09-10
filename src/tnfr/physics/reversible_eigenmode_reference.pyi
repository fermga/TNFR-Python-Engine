from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[ExactVector, ...]
ExactPartition = tuple[Fraction, ...]
ExactPartitionFamily = tuple[ExactPartition, ...]


@dataclass(frozen=True, slots=True)
class ReversibleSingleEigenmodeEulerReferenceCertificate:
    exact_conductance: ExactMatrix
    exact_nu_f: ExactVector
    exact_initial_epi: ExactVector
    exact_partitions: ExactPartitionFamily
    exact_degrees: ExactVector
    exact_reversible_metric: ExactVector
    exact_normalized_reversible_metric: ExactVector
    exact_generator: ExactMatrix
    exact_weighted_mean: Fraction
    exact_centered_mode: ExactVector
    exact_mode_eigenvalue: Fraction
    exact_mode_residual: ExactVector
    exact_mode_linf_norm: Fraction
    exact_mode_h_energy: Fraction
    exact_total_duration: Fraction
    exact_continuous_factor_lower_bound: Fraction
    exact_continuous_factor_upper_bound: Fraction
    exact_continuous_endpoint_lower_bound: ExactVector
    exact_continuous_endpoint_upper_bound: ExactVector
    exact_partition_hmax: ExactVector
    exact_scaled_partitions: ExactPartitionFamily
    exact_euler_segment_factors: ExactPartitionFamily
    exact_euler_factors: ExactVector
    exact_euler_endpoints: tuple[ExactVector, ...]
    exact_factor_error_lower_bounds: ExactVector
    exact_factor_error_upper_bounds: ExactVector
    exact_quadratic_factor_error_upper_bounds: ExactVector
    exact_hmax_factor_error_upper_bounds: ExactVector
    exact_linf_error_lower_bounds: ExactVector
    exact_linf_error_upper_bounds: ExactVector
    exact_linf_quadratic_error_upper_bounds: ExactVector
    exact_linf_hmax_error_upper_bounds: ExactVector
    exact_h_energy_error_lower_bounds: ExactVector
    exact_h_energy_error_upper_bounds: ExactVector
    exact_h_energy_quadratic_error_upper_bounds: ExactVector
    exact_h_energy_hmax_error_upper_bounds: ExactVector
    exact_euler_factor_improvements: ExactVector
    exact_quadratic_bound_improvements: ExactVector
    exact_linf_quadratic_bound_improvements: ExactVector
    exact_h_energy_quadratic_bound_improvements: ExactVector
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def reference_certificate_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_modal_solution_enclosure_certified(self) -> bool: ...
    @property
    def exact_real_euler_error_bound_certified(self) -> bool: ...
    @property
    def exact_linf_error_bound_certified(self) -> bool: ...
    @property
    def exact_h_energy_error_bound_certified(self) -> bool: ...
    @property
    def strict_proper_subdivision_improvement_certified(self) -> bool: ...
    @property
    def conditional_exact_real_partition_convergence_certified(self) -> bool: ...
    @property
    def binary64_asymptotic_convergence_certified(self) -> bool: ...
    @property
    def arbitrary_or_mixed_mode_initial_data_certified(self) -> bool: ...
    @property
    def directed_or_nonreversible_generator_certified(self) -> bool: ...
    @property
    def changing_generator_or_metric_certified(self) -> bool: ...
    @property
    def glyph_or_remesh_dynamics_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...


def certify_reversible_single_eigenmode_euler_reference(
    conductance: Iterable[Iterable[Fraction]],
    *,
    nu_f: Iterable[Fraction],
    initial_epi: Iterable[Fraction],
    partitions: Iterable[Iterable[Fraction]],
) -> ReversibleSingleEigenmodeEulerReferenceCertificate: ...


__all__: tuple[str, ...]
