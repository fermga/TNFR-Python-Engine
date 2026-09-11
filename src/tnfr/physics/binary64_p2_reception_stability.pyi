from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Hashable, Literal

from .binary64_remesh_relative_defect import (
    UniformAlphaOneHardClipRemeshClassCertificate,
)
from .remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
)
from .remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
)

Binary64Pair = tuple[float, float]
ExactMatrix2 = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]

def _evaluate_binary64_half_reception_pair(
    values: tuple[float, float],
    *,
    lower: float,
    upper: float,
) -> Binary64Pair: ...

@dataclass(frozen=True, slots=True)
class P2HalfReceptionRemeshStabilityCertificate:
    remesh_class_certificate: UniformAlphaOneHardClipRemeshClassCertificate = field(
        repr=False
    )
    node_order: tuple[Hashable, Hashable]
    exact_normalized_metric: tuple[Fraction, Fraction]
    mutual_singleton_neighbor_indices: tuple[tuple[int], tuple[int]]
    operator_name: Literal["Reception"]
    operator_glyph: Literal["EN"]
    stage_schedule: Literal["two_phase_jacobi"]
    binary64_mix_factor: float
    exact_mix_factor: Fraction
    exact_ideal_consensus_projector: ExactMatrix2
    exact_schedule_energy_gain_upper_bound: Fraction
    exact_pre_schedule_relative_energy_defect_upper_bound: Fraction
    exact_effective_head_energy_gain_upper_bound: Fraction
    active_history_extinction_horizon: int
    policy_certificate: UniformRemeshSchedulePolicyStabilityCertificate = field(
        repr=False
    )
    relative_defect_certificate: (
        UniformRemeshScheduleRelativeDefectStabilityCertificate
    ) = field(repr=False)
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact_after_dependencies_validation(self) -> bool: ...
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def p2_half_reception_remesh_stability_certificate_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def global_binary64_epi_kernel_family_certified(self) -> bool: ...
    @property
    def numeric_consensus_projection_certified(self) -> bool: ...
    @property
    def restricted_epi_kernel_interval_forward_invariant_certified(self) -> bool: ...
    @property
    def restricted_kernel_support_metric_configuration_preservation_certified(
        self,
    ) -> bool: ...
    @property
    def arbitrary_finite_binary64_kernel_repetition_certified(self) -> bool: ...
    @property
    def active_history_exact_extinction_certified(self) -> bool: ...
    def exact_cycle_energy_gain_upper_bound(self, cycle_count: int) -> Fraction: ...
    def evaluate_binary64_schedule_pair(
        self,
        pair: Binary64Pair | list[float],
    ) -> Binary64Pair: ...
    @property
    def signed_zero_bit_preservation_certified(self) -> bool: ...
    @property
    def global_binary64_runtime_affinity_certified(self) -> bool: ...
    @property
    def complete_reception_stage_certified(self) -> bool: ...
    @property
    def grammar_execution_certified(self) -> bool: ...
    @property
    def live_graph_execution_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...

def certify_p2_half_reception_remesh_stability(
    remesh_class_certificate: UniformAlphaOneHardClipRemeshClassCertificate,
) -> P2HalfReceptionRemeshStabilityCertificate: ...

__all__: tuple[str, ...]
