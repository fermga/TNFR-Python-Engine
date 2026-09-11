from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..operators.event_remesh_runtime import EventRemeshCycleResult
from .event_remesh_refinement import EventRemeshThreeMeshRefinementObservation
from .remesh_history_stability import ExactVector
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
)


@dataclass(frozen=True, slots=True)
class P2EventRemeshMeshReferenceObservation:
    mesh_name: str
    cycle_result: EventRemeshCycleResult = field(repr=False)
    runtime_bridge: RuntimeRemeshHistoryBridgeObservation = field(repr=False)
    nodes: tuple[Any, Any]
    exact_initial_field: ExactVector
    exact_mean: Fraction
    exact_initial_amplitude: Fraction
    exact_nu_f: Fraction
    exact_lambda: Fraction
    exact_total_duration: Fraction
    exact_segment_durations: tuple[Fraction, ...]
    exact_scaled_segment_durations: tuple[Fraction, ...]
    exact_euler_segment_factors: tuple[Fraction, ...]
    exact_euler_factor: Fraction
    exact_continuous_factor_lower_bound: Fraction
    exact_continuous_factor_upper_bound: Fraction
    exact_factor_error_lower_bound: Fraction
    exact_factor_error_upper_bound: Fraction
    exact_quadratic_factor_error_upper_bound: Fraction
    exact_hmax_factor_error_upper_bound: Fraction
    exact_factor_error_symbolic_coefficients: tuple[Fraction, Fraction]
    exact_observed_pre_remesh_field: ExactVector
    exact_euler_reference_pre_remesh_field: ExactVector
    exact_pre_remesh_error_lower_bound: Fraction
    exact_pre_remesh_error_upper_bound: Fraction
    exact_pre_remesh_quadratic_error_upper_bound: Fraction
    exact_pre_remesh_hmax_error_upper_bound: Fraction
    exact_alpha: Fraction
    exact_beta: Fraction
    exact_clip_lower_bound: Fraction
    exact_clip_upper_bound: Fraction
    exact_ideal_remesh_euler_field: ExactVector
    exact_runtime_remesh_field: ExactVector
    exact_signed_rounding_residual: ExactVector
    exact_signed_clipping_residual: ExactVector
    exact_signed_total_residual: ExactVector
    exact_total_residual_linf: Fraction
    exact_ideal_post_remesh_error_symbolic_coefficients: tuple[
        Fraction,
        Fraction,
    ]
    exact_ideal_post_remesh_error_lower_bound: Fraction
    exact_ideal_post_remesh_error_upper_bound: Fraction
    exact_runtime_post_remesh_error_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def mesh_reference_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_real_euler_error_bound_certified(self) -> bool: ...
    @property
    def exact_ideal_remesh_error_scaling_certified(self) -> bool: ...
    @property
    def runtime_residual_error_bound_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class P2EventRemeshReferenceFamilyObservation:
    refinement: EventRemeshThreeMeshRefinementObservation = field(repr=False)
    meshes: tuple[
        P2EventRemeshMeshReferenceObservation,
        P2EventRemeshMeshReferenceObservation,
        P2EventRemeshMeshReferenceObservation,
    ]
    nodes: tuple[Any, Any]
    exact_initial_field: ExactVector
    exact_mean: Fraction
    exact_initial_amplitude: Fraction
    exact_nu_f: Fraction
    exact_lambda: Fraction
    exact_total_duration: Fraction
    exact_alpha: Fraction
    exact_beta: Fraction
    exact_clip_lower_bound: Fraction
    exact_clip_upper_bound: Fraction
    exact_continuous_factor_lower_bound: Fraction
    exact_continuous_factor_upper_bound: Fraction
    exact_euler_factors: tuple[Fraction, Fraction, Fraction]
    exact_euler_factor_improvements: tuple[Fraction, Fraction]
    exact_quadratic_factor_error_upper_bounds: tuple[
        Fraction,
        Fraction,
        Fraction,
    ]
    exact_quadratic_bound_improvements: tuple[Fraction, Fraction]
    strict_pre_remesh_error_improvement: bool
    strict_ideal_post_remesh_error_improvement: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def reference_family_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_p2_euler_consistency_bound_certified(self) -> bool: ...
    @property
    def strict_proper_subdivision_improvement_certified(self) -> bool: ...
    @property
    def exact_ideal_remesh_error_scaling_certified(self) -> bool: ...
    @property
    def runtime_residual_error_bounds_certified(self) -> bool: ...
    @property
    def binary64_asymptotic_convergence_certified(self) -> bool: ...
    @property
    def arbitrary_glyph_or_mixed_mode_certified(self) -> bool: ...
    @property
    def soft_clipping_certified(self) -> bool: ...
    @property
    def changing_support_or_metric_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def generic_mesh_convergence_certified(self) -> bool: ...
    @property
    def repeated_runtime_stability_certified(self) -> bool: ...
    @property
    def future_stability_certified(self) -> bool: ...


def observe_p2_event_remesh_reference_family(
    coarse: EventRemeshCycleResult,
    intermediate: EventRemeshCycleResult,
    fine: EventRemeshCycleResult,
) -> P2EventRemeshReferenceFamilyObservation: ...


__all__: tuple[str, ...]
