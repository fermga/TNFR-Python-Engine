from __future__ import annotations

from collections.abc import Mapping
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
    ScheduledOperatorEvent,
    StructuralFlowInterval,
)
from .network_stage import NetworkStageResult


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

    @property
    def runtime_bound_binary64_interval_identified(self) -> bool: ...
    @property
    def runtime_bound_exact_affine_map_identified(self) -> bool: ...
    @property
    def runtime_bound_global_disagreement_contraction_certified(self) -> bool: ...


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
    represented_affine_gain_bound_at_observed_endpoint_certified: bool
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
    solver_accuracy_certified: bool = field(default=..., init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=...,
        init=False,
    )
    scope: str = field(default=..., init=False)


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
    _proof_stamp: tuple[Any, ...] = field(...)

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
    _proof_stamp: tuple[Any, ...] = field(...)
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

    @property
    def all_positive_flow_intervals_binary64_identified(self) -> bool | None: ...
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
) -> OperatorEventExecutionResult: ...


__all__: tuple[str, str, str, str, str, str, str]
