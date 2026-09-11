from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import networkx as nx

from ..physics.network_stage_stability import AllTargetNeighborStageCertificate
from ..physics.pointwise_stage_stability import (
    PointwiseEPIJumpRealizationCertificate,
)
from ..physics.runtime_flow_stability import NodalFlowIntervalCertificate
from ..physics.runtime_flow_stability import NodalFlowStateSnapshot
from ..types import Glyph
from .event_timing import (
    OperatorEventSchedule,
    PhysicalFlowPartition,
    ScheduledOperatorEvent,
    StructuralFlowInterval,
)
from .network_stage import (
    MutationStageDecisionObservation,
    NetworkStageResult,
    ReceptionStageObservation,
)


@dataclass(frozen=True, slots=True)
class ExecutedOperatorEvent:
    event_index: int
    cycle_index: int
    word_position: int
    operator_name: str
    glyph: Glyph
    event_time: float
    event_offset: Fraction
    exact_event_time: Fraction
    stage_schedule: str
    nodes_processed: int
    zero_duration: bool = field(default=..., init=False)
    history_channel: str = field(default=..., init=False)
    feeds_epi_time_history: bool = field(default=..., init=False)
    _proof_stamp: tuple[Any, ...] = field(...)

    def __post_init__(self) -> None: ...
    def _proof_fields_are_intact(self) -> bool: ...

    @classmethod
    def from_stage(
        cls,
        event: ScheduledOperatorEvent,
        result: NetworkStageResult,
    ) -> ExecutedOperatorEvent: ...
    def as_record(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ExecutedNodalFlowInterval:
    interval: StructuralFlowInterval
    certificate: NodalFlowIntervalCertificate | None
    abstention_reason: str | None
    integrator_name: str
    integrator_provenance_certified: bool
    resolved_method: str | None
    resolved_substeps: int | None
    gamma_is_none: bool | None
    clipping_applied: bool | None
    extended_dynamics_requested: bool
    solver_accuracy_certified: bool = field(default=..., init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=...,
        init=False,
    )
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def runtime_bound_binary64_interval_identified(self) -> bool: ...
    @property
    def runtime_bound_binary64_held_pressure_interval_identified(self) -> bool: ...
    @property
    def runtime_bound_exact_affine_map_identified(self) -> bool: ...
    @property
    def runtime_bound_global_disagreement_contraction_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PressureRefreshBoundaryObservation:
    parent_interval_index: int
    boundary_index: int
    time: float
    exact_time: Fraction
    offset: Fraction
    callback_name: str
    callback_identity: int
    configured_callback: bool
    callback_binding_preserved: bool
    callback_state_preserved: bool
    before: NodalFlowStateSnapshot
    after: NodalFlowStateSnapshot
    node_support_preserved: bool
    epi_preserved: bool
    capacity_preserved: bool
    conductance_preserved: bool
    edge_state_preserved: bool
    graph_configuration_preserved: bool
    dnfr_weights_preserved_or_canonically_initialized: bool
    phase_preserved: bool
    epi_derivatives_preserved: bool
    mutation_history_preserved: bool
    glyph_history_preserved: bool
    other_nodal_state_preserved: bool
    runtime_clock_preserved: bool
    pressure_changed: bool
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def callback_completed(self) -> bool: ...
    @property
    def external_side_effects_rolled_back(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def nonpressure_state_preserved(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PhysicalEulerModalObservation:
    parent_interval_index: int
    segment_index: int
    dt: float
    available: bool
    abstention_reason: str | None
    target_fraction: float | None
    spectral_relative_tolerance: float | None
    spectral_zero_threshold: float | None
    decay_rates: tuple[float, ...] | None
    modal_multipliers: tuple[float, ...] | None
    slowest_decay_rate: float | None
    fastest_decay_rate: float | None
    euler_stability_limit: float | None
    maximum_modal_factor: float | None
    modal_steps: int | None
    policy_window: int | None
    is_euler_stable: bool | None
    scope: str | None
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def modal_diagnostic_established(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ExecutedPressureRefreshedFlowPartition:
    partition: PhysicalFlowPartition
    boundary_observations: tuple[PressureRefreshBoundaryObservation, ...]
    segment_flow_evidence: tuple[ExecutedNodalFlowInterval, ...]
    modal_observations: tuple[PhysicalEulerModalObservation, ...]
    pressure_refresh_callback_invocations: int
    exact_common_metric: tuple[Fraction, ...] | None
    exact_segment_gain_bounds: tuple[Fraction, ...]
    exact_composed_gain_bound: Fraction | None
    _exact_common_metric_gain_product_certified: bool = field(repr=False)
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def solver_order_certified(self) -> bool: ...
    @property
    def mesh_convergence_certified(self) -> bool: ...
    @property
    def future_or_repeated_behavior_certified(self) -> bool: ...
    @property
    def adaptive_u2_u4_policy_certified(self) -> bool: ...
    @property
    def external_side_effects_rolled_back(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def physical_pressure_reevaluated_partition_established(self) -> bool: ...
    @property
    def all_segment_binary64_replays_identified(self) -> bool: ...
    @property
    def all_segment_binary64_intervals_identified(self) -> bool: ...
    @property
    def all_segment_exact_affine_maps_identified(self) -> bool: ...
    @property
    def all_segment_disagreement_contractions_certified(self) -> bool: ...
    @property
    def all_boundaries_binary64_pure_epi_pressure_realized(self) -> bool: ...
    @property
    def all_segment_modal_diagnostics_applicable(self) -> bool: ...
    @property
    def all_segment_modal_decisions_stable(self) -> bool: ...
    @property
    def exact_common_metric_gain_product_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ExecutedGlyphStage:
    event: ExecutedOperatorEvent
    certificate_kind: str | None
    certificate: (
        PointwiseEPIJumpRealizationCertificate
        | AllTargetNeighborStageCertificate
        | None
    )
    certificate_abstention_reason: str | None
    left: NodalFlowStateSnapshot | None
    right: NodalFlowStateSnapshot | None
    endpoint_capture_complete: bool
    exact_runtime_endpoint_bound: bool
    exact_metric_ray_before: tuple[Fraction, ...] | None
    exact_metric_ray_after: tuple[Fraction, ...] | None
    exact_common_metric_bridge: bool
    exact_energy_gain_upper_bound: Fraction | None
    _represented_affine_gain_bound_at_observed_endpoint_certified: bool
    pre_interval_index: int
    post_interval_index: int
    pre_interval_positive: bool
    post_interval_positive: bool
    pre_flow_evidence: ExecutedNodalFlowInterval | None = field(...)
    post_flow_evidence: ExecutedNodalFlowInterval | None = field(...)
    pre_flow_endpoint_continuous: bool | None = ...
    post_flow_endpoint_continuous: bool | None = ...
    pre_flow_metric_compatible: bool | None = ...
    post_flow_metric_compatible: bool | None = ...
    mutation_decision_observations: tuple[
        MutationStageDecisionObservation, ...
    ] = field(...)
    reception_observations: tuple[ReceptionStageObservation, ...] = field(...)
    solver_accuracy_certified: bool = field(default=..., init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=...,
        init=False,
    )
    scope: str = field(default=..., init=False)
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def represented_affine_gain_bound_at_observed_endpoint_certified(
        self,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RepresentedEPIScheduleOperation:
    position: int
    operation_kind: str
    operation_index: int
    operator_name: str | None
    nodes: tuple[Any, ...] | None
    exact_epi_before: tuple[Fraction, ...] | None
    exact_epi_after: tuple[Fraction, ...] | None
    exact_metric_ray_before: tuple[Fraction, ...] | None
    exact_metric_ray_after: tuple[Fraction, ...] | None
    exact_energy_gain_upper_bound: Fraction | None
    ineligibility_reasons: tuple[str, ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False)

    @property
    def represented_affine_gain_certified(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ObservedRepresentedEPIScheduleComposition:
    nodes: tuple[Any, ...]
    positive_flow_interval_indices: tuple[int, ...]
    event_indices: tuple[int, ...]
    operations: tuple[RepresentedEPIScheduleOperation, ...]
    exact_normalized_metric: tuple[Fraction, ...] | None
    exact_operation_energy_gain_factors: tuple[Fraction, ...]
    exact_energy_gain_upper_bound: Fraction | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False)
    scope: str = field(default=..., init=False)

    @property
    def runtime_schedule_global_gain_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def full_multichannel_stability_certified(self) -> bool: ...
    @property
    def future_or_repeated_schedule_stability_certified(self) -> bool: ...

    @property
    def represented_affine_composition_gain_certified(self) -> bool: ...
    @property
    def represented_map_global_disagreement_contraction_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...

@dataclass(frozen=True, slots=True)
class OperatorEventExecutionResult:
    schedule: OperatorEventSchedule
    target_nodes: tuple[Any, ...]
    flow_interval_indices: tuple[int, ...]
    positive_flow_interval_indices: tuple[int, ...]
    events: tuple[ExecutedOperatorEvent, ...]
    final_time: float
    integrator_name: str | None
    pressure_refresh_callback_invocations: int
    flow_certification_requested: bool = ...
    flow_interval_evidence: tuple[ExecutedNodalFlowInterval, ...] = ...
    stage_certification_requested: bool = ...
    glyph_stage_evidence: tuple[ExecutedGlyphStage, ...] = ...
    represented_epi_schedule_composition: (
        ObservedRepresentedEPIScheduleComposition | None
    ) = ...
    physical_flow_partition_indices: tuple[int, ...] = ...
    physical_flow_partition_evidence: tuple[
        ExecutedPressureRefreshedFlowPartition, ...
    ] = ...
    physical_pressure_refresh_callback_invocations: int = ...
    stage_pressure_refresh_callback_invocations: int = ...
    runtime_clock_checked: bool = field(default=..., init=False)
    flow_provenance: str = field(default=..., init=False)
    nodal_flow_inputs: str = field(default=..., init=False)
    whole_schedule_graph_state_atomic: bool = field(
        default=..., init=False
    )
    operator_jumps_have_zero_duration: bool = field(
        default=..., init=False
    )
    solver_accuracy_certified: bool = field(default=..., init=False)
    adaptive_u2_u4_policy: bool = field(default=..., init=False)
    external_side_effects_rolled_back: bool = field(
        default=..., init=False
    )
    future_or_repeated_schedule_stability_certified: bool = field(
        default=...,
        init=False,
    )
    flow_scope: str = field(default=..., init=False)
    _proof_stamp: tuple[Any, ...] = field(...)
    def __post_init__(self) -> None: ...
    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def physical_pressure_reevaluated_partitions_established(
        self,
    ) -> bool | None: ...
    @property
    def all_positive_flow_intervals_binary64_identified(self) -> bool | None: ...
    @property
    def all_positive_flow_intervals_binary64_held_pressure_identified(
        self,
    ) -> bool | None: ...
    @property
    def all_positive_flow_intervals_exact_affine(self) -> bool | None: ...
    @property
    def all_positive_flow_intervals_contracting(self) -> bool | None: ...
    @property
    def all_glyph_stages_represented_affine(self) -> bool | None: ...


def execute_operator_event_schedule(
    graph: nx.Graph,
    schedule: OperatorEventSchedule,
    *,
    context: Mapping[str, Any] | None = ...,
    method: str | None = ...,
    n_jobs: int | None = ...,
    suppress_birth_warnings: bool = ...,
    include_flow_certificates: bool = ...,
    include_stage_certificates: bool = ...,
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = ...,
) -> OperatorEventExecutionResult: ...


__all__: tuple[str, ...]
