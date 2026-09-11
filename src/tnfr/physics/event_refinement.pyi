from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any

from ..operators.event_runtime import (
    ExecutedGlyphStage,
    ExecutedNodalFlowInterval,
    ExecutedOperatorEvent,
    ExecutedPressureRefreshedFlowPartition,
    OperatorEventExecutionResult,
)
from ..operators.event_timing import ScheduledOperatorEvent
from ..operators.network_stage import MutationStageDecisionObservation
from .mutation_trigger import MutationTriggerCertificate


@dataclass(frozen=True, slots=True)
class EventLocalZHIRPrejumpObservation:
    event_identity: tuple[int, int, int, str, str]
    flow_interval_index: int
    nodes: tuple[Any, ...]
    substeps: int
    interval_start_time: float
    interval_end_time: float
    duration: float
    exact_interval_start_time: Fraction
    exact_interval_end_time: Fraction
    exact_duration: Fraction
    event_time: float
    exact_event_time: Fraction
    event_offset: Fraction
    exact_initial_epi: tuple[Fraction, ...]
    exact_endpoint_epi: tuple[Fraction, ...]
    exact_capacity: tuple[Fraction, ...]
    exact_pressure: tuple[Fraction, ...]
    exact_conductance: tuple[tuple[Fraction, ...], ...]
    exact_rational_physical_secants: tuple[Fraction, ...]
    binary64_sample_interval: float
    exact_binary64_sample_interval: Fraction
    binary64_epi_deltas: tuple[float, ...]
    exact_binary64_epi_deltas: tuple[Fraction, ...]
    binary64_observed_gate_rates: tuple[float, ...]
    exact_binary64_observed_gate_rates: tuple[Fraction, ...]
    exact_rational_minus_binary64_gate_rates: tuple[Fraction, ...]
    rational_and_binary64_gate_rates_equal_by_node: tuple[bool, ...]
    rational_and_binary64_gate_rates_equal: bool
    xi: float
    exact_xi: Fraction
    binary64_observed_strict_gate_decisions: tuple[bool, ...]
    exact_jump_at_pre_flow_endpoint: bool
    binary64_jump_at_pre_flow_endpoint: bool
    endpoint_subtraction_matches_declared_duration: bool
    scope: str = field(default=..., init=False)
    _proof_stamp: tuple[Any, ...] = field(...)
    binary64_initial_epi: tuple[float, ...] = ()
    binary64_endpoint_epi: tuple[float, ...] = ()
    binary64_capacity: tuple[float, ...] = ()
    binary64_pressure: tuple[float, ...] = ()

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def pre_jump_observation_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EventLocalZHIRHeldPressureComparison:
    comparison_kind: str
    event_identity: tuple[int, int, int, str, str]
    nodes: tuple[Any, ...]
    baseline_substeps: int
    candidate_substeps: int
    substep_counts_differ: bool
    exact_event_time: Fraction
    event_time: float
    exact_jump_placement_equal: bool
    binary64_jump_placement_equal: bool
    exact_duration: Fraction
    baseline_exact_endpoint_epi: tuple[Fraction, ...]
    candidate_exact_endpoint_epi: tuple[Fraction, ...]
    exact_endpoint_epi_difference: tuple[Fraction, ...]
    baseline_exact_rational_physical_secants: tuple[Fraction, ...]
    candidate_exact_rational_physical_secants: tuple[Fraction, ...]
    exact_rational_physical_secant_differences: tuple[Fraction, ...]
    baseline_binary64_observed_gate_rates: tuple[float, ...]
    candidate_binary64_observed_gate_rates: tuple[float, ...]
    baseline_exact_binary64_observed_gate_rates: tuple[Fraction, ...]
    candidate_exact_binary64_observed_gate_rates: tuple[Fraction, ...]
    exact_binary64_gate_rate_differences: tuple[Fraction, ...]
    exact_binary64_gate_rate_difference_bounds: tuple[Fraction, ...]
    xi: float
    exact_xi: Fraction
    baseline_exact_signed_threshold_margins: tuple[Fraction, ...]
    baseline_exact_threshold_distances: tuple[Fraction, ...]
    baseline_observed_strict_gate_decisions: tuple[bool, ...]
    candidate_observed_strict_gate_decisions: tuple[bool, ...]
    observed_strict_gate_decisions_agree_by_node: tuple[bool, ...]
    observed_strict_gate_decisions_agree: bool
    strict_threshold_separation_by_node: tuple[bool, ...]
    strict_threshold_separation: bool
    _event_local_gate_invariance_certified: bool = field(repr=False)
    physical_pressure_reevaluated_partition_established: bool = field(
        default=..., init=False
    )
    diffusion_modal_decisions_applicable: bool = field(default=..., init=False)
    diffusion_modal_decisions_established: bool = field(default=..., init=False)
    solver_accuracy_certified: bool = field(default=..., init=False)
    solver_order_certified: bool = field(default=..., init=False)
    physical_refinement_equivalence_certified: bool = field(
        default=..., init=False
    )
    future_or_repeated_behavior_certified: bool = field(
        default=..., init=False
    )
    u4_readiness_certified: bool = field(default=..., init=False)
    adaptive_policy_certified: bool = field(default=..., init=False)
    scope: str = field(default=..., init=False)
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def event_local_gate_invariance_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EventLocalZHIRPhysicalPrejumpObservation:
    partition_evidence: ExecutedPressureRefreshedFlowPartition = field(
        repr=False
    )
    event_identity: tuple[int, int, int, str, str]
    parent_interval_index: int
    nodes: tuple[Any, ...]
    segment_count: int
    segment_durations: tuple[float, ...]
    exact_segment_durations: tuple[Fraction, ...]
    interval_start_time: float
    interval_end_time: float
    duration: float
    exact_interval_start_time: Fraction
    exact_interval_end_time: Fraction
    exact_duration: Fraction
    event_time: float
    exact_event_time: Fraction
    event_offset: Fraction
    binary64_initial_epi: tuple[float, ...]
    exact_initial_epi: tuple[Fraction, ...]
    binary64_terminal_window_initial_epi: tuple[float, ...]
    exact_terminal_window_initial_epi: tuple[Fraction, ...]
    binary64_terminal_epi: tuple[float, ...]
    exact_terminal_epi: tuple[Fraction, ...]
    binary64_initial_capacity: tuple[float, ...]
    exact_initial_capacity: tuple[Fraction, ...]
    binary64_initial_pressure: tuple[float, ...]
    exact_initial_pressure: tuple[Fraction, ...]
    exact_initial_conductance: tuple[tuple[Fraction, ...], ...]
    binary64_terminal_capacity: tuple[float, ...]
    exact_terminal_capacity: tuple[Fraction, ...]
    binary64_terminal_pressure: tuple[float, ...]
    exact_terminal_pressure: tuple[Fraction, ...]
    exact_terminal_conductance: tuple[tuple[Fraction, ...], ...]
    xi: float
    exact_xi: Fraction
    terminal_sample_start_time: float
    exact_terminal_sample_start_time: Fraction
    terminal_sample_interval: float
    exact_terminal_sample_interval: Fraction
    terminal_binary64_epi_deltas: tuple[float, ...]
    terminal_exact_binary64_epi_deltas: tuple[Fraction, ...]
    terminal_binary64_observed_gate_rates: tuple[float, ...]
    terminal_exact_binary64_observed_gate_rates: tuple[Fraction, ...]
    terminal_exact_rational_physical_secants: tuple[Fraction, ...]
    terminal_exact_rational_minus_binary64_gate_rates: tuple[Fraction, ...]
    terminal_binary64_predicted_rates: tuple[float, ...]
    terminal_exact_binary64_predicted_rates: tuple[Fraction, ...]
    terminal_capacity_active_by_node: tuple[bool, ...]
    terminal_observed_strict_gate_decisions: tuple[bool, ...]
    whole_parent_sample_interval: float
    exact_whole_parent_sample_interval: Fraction
    whole_parent_binary64_epi_deltas: tuple[float, ...]
    whole_parent_exact_binary64_epi_deltas: tuple[Fraction, ...]
    whole_parent_binary64_gate_rates: tuple[float, ...]
    whole_parent_exact_binary64_gate_rates: tuple[Fraction, ...]
    whole_parent_exact_rational_physical_secants: tuple[Fraction, ...]
    whole_parent_exact_rational_minus_binary64_gate_rates: tuple[Fraction, ...]
    whole_parent_strict_gate_decisions: tuple[bool, ...]
    terminal_window_equals_parent_horizon: bool
    exact_jump_at_parent_endpoint: bool
    binary64_jump_at_parent_endpoint: bool
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def common_jump_execution_certified(self) -> bool: ...

    @property
    def u4_readiness_certified(self) -> bool: ...

    @property
    def solver_accuracy_certified(self) -> bool: ...

    @property
    def solver_order_certified(self) -> bool: ...

    @property
    def mesh_convergence_certified(self) -> bool: ...

    @property
    def future_or_repeated_behavior_certified(self) -> bool: ...

    @property
    def scope(self) -> str: ...

    @property
    def physical_pre_jump_observation_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ExecutedEventLocalZHIRPhysicalPrejumpObservation:
    execution_result: OperatorEventExecutionResult = field(repr=False)
    event_index: int
    scheduled_event: ScheduledOperatorEvent = field(repr=False)
    executed_event: ExecutedOperatorEvent = field(repr=False)
    partition_evidence: ExecutedPressureRefreshedFlowPartition = field(
        repr=False
    )
    glyph_stage: ExecutedGlyphStage = field(repr=False)
    physical_observation: EventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    nodes: tuple[Any, ...]
    mutation_decision_observations: tuple[
        MutationStageDecisionObservation, ...
    ] = field(repr=False)
    trigger_certificates: tuple[MutationTriggerCertificate, ...] = field(
        repr=False
    )
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def common_execution_provenance_certified(self) -> bool: ...

    @property
    def solver_accuracy_certified(self) -> bool: ...

    @property
    def solver_order_certified(self) -> bool: ...

    @property
    def mesh_convergence_certified(self) -> bool: ...

    @property
    def u4_readiness_certified(self) -> bool: ...

    @property
    def adaptive_policy_certified(self) -> bool: ...

    @property
    def future_or_repeated_behavior_certified(self) -> bool: ...

    @property
    def scope(self) -> str: ...


@dataclass(frozen=True, slots=True)
class EventLocalZHIRPhysicalRefinementComparison:
    baseline_observation: EventLocalZHIRPrejumpObservation = field(repr=False)
    physical_observation: EventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    comparison_kind: str
    event_identity: tuple[int, int, int, str, str]
    nodes: tuple[Any, ...]
    baseline_runtime_substeps: int
    physical_segment_count: int
    exact_duration: Fraction
    xi: float
    exact_xi: Fraction
    baseline_binary64_gate_rates: tuple[float, ...]
    baseline_exact_binary64_gate_rates: tuple[Fraction, ...]
    baseline_observed_strict_gate_decisions: tuple[bool, ...]
    physical_whole_parent_binary64_gate_rates: tuple[float, ...]
    physical_whole_parent_exact_binary64_gate_rates: tuple[Fraction, ...]
    physical_whole_parent_strict_gate_decisions: tuple[bool, ...]
    exact_whole_parent_gate_rate_differences: tuple[Fraction, ...]
    exact_whole_parent_gate_rate_difference_bounds: tuple[Fraction, ...]
    baseline_exact_signed_threshold_margins: tuple[Fraction, ...]
    baseline_exact_threshold_distances: tuple[Fraction, ...]
    whole_parent_gate_decisions_agree_by_node: tuple[bool, ...]
    whole_parent_gate_decisions_agree: bool
    whole_parent_strict_threshold_separation_by_node: tuple[bool, ...]
    whole_parent_strict_threshold_separation: bool
    _fixed_horizon_gate_agreement_certified: bool = field(repr=False)
    physical_terminal_binary64_gate_rates: tuple[float, ...]
    physical_terminal_exact_binary64_gate_rates: tuple[Fraction, ...]
    physical_terminal_strict_gate_decisions: tuple[bool, ...]
    exact_terminal_gate_rate_differences: tuple[Fraction, ...]
    exact_terminal_gate_rate_difference_bounds: tuple[Fraction, ...]
    terminal_gate_decisions_agree_by_node: tuple[bool, ...]
    terminal_gate_decisions_agree: bool
    terminal_strict_threshold_separation_by_node: tuple[bool, ...]
    terminal_strict_threshold_separation: bool
    observation_windows_equal: bool
    _actual_terminal_gate_invariance_certified: bool = field(repr=False)
    modal_comparison_applicable: bool
    modal_abstention_reason: str | None
    baseline_held_interval_duration: float | None
    common_binary64_modal_decay_rates: tuple[float, ...] | None
    baseline_held_interval_binary64_modal_multipliers: tuple[float, ...] | None
    baseline_held_interval_binary64_maximum_modal_factor: float | None
    baseline_held_interval_modal_stable: bool | None
    physical_segment_modal_decisions: tuple[bool, ...]
    physical_refined_composite_binary64_modal_multipliers: tuple[float, ...] | None
    physical_refined_composite_binary64_maximum_modal_factor: float | None
    physical_refined_composite_modal_stable: bool | None
    modal_stability_decisions_agree: bool | None
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def modal_equivalence_certified(self) -> bool: ...

    @property
    def common_execution_provenance_certified(self) -> bool: ...

    @property
    def solver_accuracy_certified(self) -> bool: ...

    @property
    def solver_order_certified(self) -> bool: ...

    @property
    def mesh_convergence_certified(self) -> bool: ...

    @property
    def future_or_repeated_behavior_certified(self) -> bool: ...

    @property
    def u4_readiness_certified(self) -> bool: ...

    @property
    def adaptive_policy_certified(self) -> bool: ...

    @property
    def scope(self) -> str: ...

    @property
    def fixed_horizon_gate_agreement_certified(self) -> bool: ...

    @property
    def actual_terminal_gate_invariance_certified(self) -> bool: ...


def observe_event_local_zhir_prejump(
    flow: ExecutedNodalFlowInterval,
    event: ScheduledOperatorEvent,
    *,
    xi: Real,
) -> EventLocalZHIRPrejumpObservation: ...


def compare_event_local_zhir_held_pressure_subdivision(
    baseline: EventLocalZHIRPrejumpObservation,
    candidate: EventLocalZHIRPrejumpObservation,
) -> EventLocalZHIRHeldPressureComparison: ...


def observe_event_local_zhir_physical_prejump(
    partition_evidence: ExecutedPressureRefreshedFlowPartition,
    event: ScheduledOperatorEvent,
    *,
    xi: Real,
) -> EventLocalZHIRPhysicalPrejumpObservation: ...


def observe_executed_event_local_zhir_physical_prejump(
    execution_result: OperatorEventExecutionResult,
    *,
    event_index: int,
) -> ExecutedEventLocalZHIRPhysicalPrejumpObservation: ...


def compare_event_local_zhir_physical_refinement(
    baseline: EventLocalZHIRPrejumpObservation,
    physical: EventLocalZHIRPhysicalPrejumpObservation,
) -> EventLocalZHIRPhysicalRefinementComparison: ...


__all__: tuple[str, ...]
