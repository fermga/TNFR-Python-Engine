from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from ..operators.event_runtime import ExecutedGlyphStage, OperatorEventExecutionResult
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
)

Binary64Pair = tuple[float, float]
ExactPair = tuple[Fraction, Fraction]

@dataclass(frozen=True, slots=True)
class ExecutedP2HalfReceptionStageCertificate:
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate = field(
        repr=False
    )
    execution_result: OperatorEventExecutionResult = field(repr=False)
    executed_stage: ExecutedGlyphStage = field(repr=False)
    event_index: int
    node_order: tuple[Any, Any]
    binary64_epi_before: Binary64Pair
    binary64_epi_after: Binary64Pair
    exact_epi_before: ExactPair
    exact_epi_after: ExactPair
    exact_normalized_metric: ExactPair
    runtime_neighbor_indices: tuple[tuple[int], tuple[int]]
    exact_mix_factor: Fraction
    clip_mode: Literal["hard"]
    exact_epi_lower_bound: Fraction
    exact_epi_upper_bound: Fraction
    exact_centered_energy_before: Fraction
    exact_centered_energy_after: Fraction
    exact_global_kernel_energy_gain_upper_bound: Fraction
    runtime_kernel_replay_matches_by_bits: bool
    observed_output_binary64_bits_identical: bool
    represented_affine_bridge_available_at_observed_endpoint: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact_after_dependencies_validation(self) -> bool: ...
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def executed_p2_half_reception_stage_certificate_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def finite_executor_bound_epi_stage_certified(self) -> bool: ...
    @property
    def finite_grammar_admission_observed(self) -> bool: ...
    @property
    def canonical_two_phase_reception_stage_observed(self) -> bool: ...
    @property
    def runtime_p2_support_observed(self) -> bool: ...
    @property
    def observed_capacity_and_effective_conductance_preserved(self) -> bool: ...
    @property
    def observed_source_metric_preserved(self) -> bool: ...
    @property
    def source_global_q_zero_applies_to_observed_epi_transition(self) -> bool: ...
    @property
    def observed_numeric_consensus_certified(self) -> bool: ...
    @property
    def finite_schedule_graph_state_atomicity_certified(self) -> bool: ...
    @property
    def complete_reception_stage_stability_class_certified(self) -> bool: ...
    @property
    def executed_remesh_configuration_bound(self) -> bool: ...
    @property
    def reception_auxiliary_state_preservation_certified(self) -> bool: ...
    @property
    def raw_graph_topology_preservation_certified(self) -> bool: ...
    @property
    def current_live_graph_state_bound(self) -> bool: ...
    @property
    def future_or_repeated_live_graph_stability_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...

def certify_executed_p2_half_reception_stage(
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate,
    execution_result: OperatorEventExecutionResult,
    *,
    event_index: int | None = ...,
) -> ExecutedP2HalfReceptionStageCertificate: ...

__all__: tuple[str, ...]
