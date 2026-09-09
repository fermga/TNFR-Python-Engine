from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any

from ..operators.event_runtime import ExecutedNodalFlowInterval
from ..operators.event_timing import ScheduledOperatorEvent


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
    scope: str = field(...)
    _proof_stamp: tuple[Any, ...] = field(...)

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
    _event_local_gate_invariance_certified: bool = field(...)
    physical_pressure_reevaluated_partition_established: bool = field(...)
    diffusion_modal_decisions_applicable: bool = field(...)
    diffusion_modal_decisions_established: bool = field(...)
    solver_accuracy_certified: bool = field(...)
    solver_order_certified: bool = field(...)
    physical_refinement_equivalence_certified: bool = field(...)
    future_or_repeated_behavior_certified: bool = field(...)
    u4_readiness_certified: bool = field(...)
    adaptive_policy_certified: bool = field(...)
    scope: str = field(...)
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def event_local_gate_invariance_certified(self) -> bool: ...


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


__all__: tuple[str, ...]
