from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any

from ._exact_linear_algebra import ExactSquareMatrix
from .remesh_history_stability import UniformRemeshHistoryStabilityCertificate

@dataclass(frozen=True, slots=True)
class UniformRemeshSchedulePolicyStabilityCertificate:
    remesh_certificate: UniformRemeshHistoryStabilityCertificate = field(...)
    schedule_energy_gain_upper_bound: Fraction
    history_length: int
    universal_block_horizon: int
    remesh_companion_matrix: ExactSquareMatrix
    schedule_energy_domination_matrix: ExactSquareMatrix
    head_avoidance_matrix: ExactSquareMatrix
    remesh_block_power: ExactSquareMatrix
    schedule_block_domination_power: ExactSquareMatrix
    head_avoidance_block_power: ExactSquareMatrix
    exact_uniform_normalized_block_margin_lower_bound: Fraction
    exact_uniform_block_energy_gain_upper_bound: Fraction
    exact_intrablock_prefix_energy_gain_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def policy_stability_certificate_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def conditional_policy_family_spatial_disagreement_nonincrease_certified(
        self,
    ) -> bool: ...
    @property
    def uniform_intrablock_prefix_bound_certified(self) -> bool: ...
    @property
    def uniform_positive_normalized_block_margin_certified(self) -> bool: ...
    @property
    def repeated_exact_model_spatial_disagreement_stability_certified(
        self,
    ) -> bool: ...
    @property
    def geometric_spatial_disagreement_convergence_certified(self) -> bool: ...
    @property
    def alpha_one_spatial_disagreement_decay_certified(self) -> bool: ...
    @property
    def q_one_zero_margin_boundary_certified(self) -> bool: ...
    def exact_cycle_energy_gain_upper_bound(self, cycle_count: int) -> Fraction: ...
    @property
    def runtime_schedule_maps_verified(self) -> bool: ...
    @property
    def binary64_runtime_stability_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def adaptive_grammar_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...

def certify_uniform_remesh_schedule_policy_stability(
    remesh_certificate: UniformRemeshHistoryStabilityCertificate,
    schedule_energy_gain_upper_bound: Real,
) -> UniformRemeshSchedulePolicyStabilityCertificate: ...

__all__: tuple[str, ...]
