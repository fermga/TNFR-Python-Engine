"""Atomic runtime binding for finite canonical operator-event schedules.

A schedule alternates configured nodal-flow intervals with zero-duration
canonical operator jumps. Exact rational offsets remain the scheduling
authority; the existing binary64 runtime clock must represent every positive
boundary without collapse. This module coordinates existing integrator and
network-stage contracts. It does not certify numerical accuracy, infer an
operator duration, or adapt grammar U2/U4.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Integral
from typing import TYPE_CHECKING, Any

import networkx as nx

from ..dynamics.integrators import (
    DefaultIntegrator as _CanonicalDefaultIntegrator,
)
from ..dynamics.integrators import (
    prepare_integration_params as _canonical_prepare_integration_params,
)
from ..errors import TNFRValueError
from ..physics._exact_metric import (
    normalized_positive_binary64_metric as _normalized_exact_metric,
    normalized_positive_fraction_metric as _normalized_fraction_metric,
)
from ..types import Glyph
from ..utils._structural_signature import structural_proof_signature
from .event_timing import (
    OperatorEventSchedule,
    ScheduledOperatorEvent,
    StructuralFlowInterval,
    diagnose_operator_event_runtime_clock,
)
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    MutationStageDecisionObservation,
    NetworkStageResult,
)

if TYPE_CHECKING:
    from ..physics.network_stage_stability import (
        AllTargetNeighborStageCertificate,
    )
    from ..physics.pointwise_stage_stability import (
        PointwiseEPIJumpRealizationCertificate,
    )
    from ..physics.runtime_flow_stability import (
        NodalFlowIntervalCertificate,
        NodalFlowStateSnapshot,
    )


_HYBRID_EVENT_LOG = "hybrid_event_log"
_FLOW_PROVENANCE = (
    "tnfr.operators.event_runtime.execute_operator_event_schedule"
)
_NODAL_FLOW_INPUTS = "live_nu_f_and_delta_nfr_at_each_interval_start"
_CANONICAL_DEFAULT_INTEGRATE = _CanonicalDefaultIntegrator.integrate
_FLOW_SCOPE = (
    "configured nodal EPI integrator over each declared positive interval, "
    "with current stored fields held according to that integrator and pressure "
    "refresh confined to explicit operator-stage callbacks; solver accuracy "
    "and adaptive U2/U4 behavior are not certified"
)
_STAGE_SCOPE = (
    "one executed zero-duration glyph stage bound to its captured EPI endpoints "
    "and immediately adjacent observed flow intervals; unsupported glyphs and "
    "incompatible supports or metrics produce explicit abstention"
)
_REPRESENTED_COMPOSITION_SCOPE = (
    "composition of the exact rational EPI maps represented by every operation "
    "in this observed finite schedule. Each positive flow and glyph jump is "
    "bound to exact observed endpoints and one identical normalized rational "
    "metric. This is not a global-affinity or gain theorem for the executable "
    "binary64 runtime map. Solver accuracy, full multichannel stability, future "
    "schedules and repeated execution are not certified"
)


def _has_intact_nodal_flow_certificate(value: Any) -> bool:
    """Recognize only a canonical interval certificate with intact proof fields."""

    from ..physics.runtime_flow_stability import NodalFlowIntervalCertificate

    return bool(
        type(value) is NodalFlowIntervalCertificate
        and value._proof_fields_are_intact()
    )


@dataclass(frozen=True, slots=True)
class _FlowRuntimeMetadata:
    """Trusted metadata read from the actual interval execution path."""

    integrator_name: str
    integrator_provenance_certified: bool
    resolved_method: str | None
    resolved_substeps: int | None
    gamma_is_none: bool | None
    extended_dynamics_requested: bool


@dataclass(frozen=True, slots=True)
class _PendingGlyphStage:
    """Captured event boundary retained until adjacent flows are complete."""

    event: ExecutedOperatorEvent
    result: NetworkStageResult
    left: NodalFlowStateSnapshot | None
    right: NodalFlowStateSnapshot | None
    left_captured: bool
    right_captured: bool


@dataclass(frozen=True, slots=True)
class _GlyphCertificateFacts:
    """Validated common fields extracted from one executor-owned certificate."""

    certificate_kind: str | None
    certificate: Any | None
    abstention_reason: str | None
    nodes: tuple[Any, ...] | None
    exact_epi_before: tuple[Fraction, ...] | None
    exact_epi_after: tuple[Fraction, ...] | None
    exact_metric_ray_before: tuple[Fraction, ...] | None
    exact_metric_ray_after: tuple[Fraction, ...] | None
    intrinsic_common_metric_bridge: bool
    exact_energy_gain_upper_bound: Fraction | None


@dataclass(frozen=True, slots=True)
class ExecutedOperatorEvent:
    """One scheduled zero-duration jump committed by the canonical stage."""

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
    zero_duration: bool = field(default=True, init=False)
    history_channel: str = field(default=_HYBRID_EVENT_LOG, init=False)
    feeds_epi_time_history: bool = field(default=False, init=False)

    @classmethod
    def from_stage(
        cls,
        event: ScheduledOperatorEvent,
        result: NetworkStageResult,
    ) -> "ExecutedOperatorEvent":
        """Bind one immutable schedule event to its accepted stage result."""

        return cls(
            event_index=event.event_index,
            cycle_index=event.cycle_index,
            word_position=event.word_position,
            operator_name=event.operator_name,
            glyph=event.glyph,
            event_time=event.event_time,
            event_offset=event.event_offset,
            exact_event_time=event.exact_event_time,
            stage_schedule=result.schedule,
            nodes_processed=result.nodes_processed,
        )

    def as_record(self) -> dict[str, Any]:
        """Return a detached append-only graph telemetry record."""

        return {
            "event_index": self.event_index,
            "cycle_index": self.cycle_index,
            "word_position": self.word_position,
            "operator_name": self.operator_name,
            "glyph": self.glyph.value,
            "event_time": self.event_time,
            "event_offset": self.event_offset,
            "exact_event_time": self.exact_event_time,
            "stage_schedule": self.stage_schedule,
            "nodes_processed": self.nodes_processed,
            "zero_duration": self.zero_duration,
            "history_channel": self.history_channel,
            "feeds_epi_time_history": self.feeds_epi_time_history,
        }


_EXECUTED_FLOW_PROOF_VERSION = "executed_nodal_flow_interval_v2"


def _executed_flow_interval_stamp(value: Any) -> tuple[Any, ...]:
    """Snapshot one runtime flow wrapper without retaining mutable graph state."""

    if type(value) is not ExecutedNodalFlowInterval:
        raise TypeError("flow evidence must have the canonical runtime type")
    interval = value.interval
    if type(interval) is not StructuralFlowInterval:
        raise TypeError("flow evidence interval must be canonical")
    certificate = value.certificate
    return (
        _EXECUTED_FLOW_PROOF_VERSION,
        structural_proof_signature(interval),
        structural_proof_signature(
            None
            if certificate is None
            else getattr(certificate, "_proof_stamp", None)
        ),
        structural_proof_signature(
            (
                value.abstention_reason,
                value.integrator_name,
                value.integrator_provenance_certified,
                value.resolved_method,
                value.resolved_substeps,
                value.gamma_is_none,
                value.clipping_applied,
                value.extended_dynamics_requested,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ExecutedNodalFlowInterval:
    """Runtime-bound evidence for one executed positive flow interval.

    The wrapper supplies integrator provenance from the execution site. The
    nested certificate remains a standalone endpoint result and therefore
    deliberately leaves its own ``integrator_provenance_certified`` flag
    false. Solver accuracy and future or repeated schedules are outside both
    scopes.
    """

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
    solver_accuracy_certified: bool = field(default=False, init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=False,
        init=False,
    )
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        """Whether this wrapper still matches executor-owned provenance fields."""

        if self.certificate is not None and not _has_intact_nodal_flow_certificate(
            self.certificate
        ):
            return False
        try:
            expected = _executed_flow_interval_stamp(self)
        except Exception:
            return False
        return type(self._proof_stamp) is tuple and self._proof_stamp == expected

    @property
    def runtime_bound_binary64_interval_identified(self) -> bool:
        """Whether the observed endpoint has trusted built-in provenance."""

        return bool(
            self._proof_fields_are_intact()
            and self.integrator_provenance_certified
            and self.certificate is not None
            and _has_intact_nodal_flow_certificate(self.certificate)
            and self.certificate.binary64_runtime_interval_identified
        )

    @property
    def runtime_bound_binary64_held_pressure_interval_identified(self) -> bool:
        """Whether the observed multi-substep held-pressure replay is trusted."""

        return bool(
            self._proof_fields_are_intact()
            and self.integrator_provenance_certified
            and self.certificate is not None
            and _has_intact_nodal_flow_certificate(self.certificate)
            and self.certificate.binary64_held_pressure_runtime_identified
        )

    @property
    def runtime_bound_exact_affine_map_identified(self) -> bool:
        """Whether this execution realizes the certified rational Euler map."""

        return bool(
            self._proof_fields_are_intact()
            and self.integrator_provenance_certified
            and self.certificate is not None
            and _has_intact_nodal_flow_certificate(self.certificate)
            and self.certificate.explicit_euler_map_identified
        )

    @property
    def runtime_bound_global_disagreement_contraction_certified(self) -> bool:
        """Whether that identified map contracts global disagreement."""

        return bool(
            self.runtime_bound_exact_affine_map_identified
            and self.certificate is not None
            and self.certificate.global_disagreement_contraction_certified
        )


@dataclass(frozen=True, slots=True)
class ExecutedGlyphStage:
    """Runtime-bound EPI evidence for one accepted zero-duration glyph stage."""

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
    pre_flow_evidence: ExecutedNodalFlowInterval | None = field(
        default=None, repr=False, compare=False
    )
    post_flow_evidence: ExecutedNodalFlowInterval | None = field(
        default=None, repr=False, compare=False
    )
    pre_flow_endpoint_continuous: bool | None = None
    post_flow_endpoint_continuous: bool | None = None
    pre_flow_metric_compatible: bool | None = None
    post_flow_metric_compatible: bool | None = None
    mutation_decision_observations: tuple[
        MutationStageDecisionObservation, ...
    ] = field(default=(), repr=False)
    solver_accuracy_certified: bool = field(default=False, init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=False,
        init=False,
    )
    scope: str = field(default=_STAGE_SCOPE, init=False)
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        """Whether all executor-owned stage fields retain their sealed values."""

        try:
            if not _executed_glyph_stage_fields_are_valid(self):
                return False
            expected = _executed_glyph_stage_stamp(self)
        except Exception:
            return False
        return type(self._proof_stamp) is tuple and self._proof_stamp == expected

    @property
    def represented_affine_gain_bound_at_observed_endpoint_certified(self) -> bool:
        """Publish the represented gain claim only while the stage seal is intact."""

        return bool(
            self._proof_fields_are_intact()
            and self._represented_affine_gain_bound_at_observed_endpoint_certified
        )


_EXECUTED_GLYPH_STAGE_PROOF_VERSION = "executed_glyph_stage_v1"


def _same_structural_value(left: Any, right: Any) -> bool:
    """Compare detached signatures without invoking identifier equality."""

    return structural_proof_signature(left) == structural_proof_signature(right)


def _executed_glyph_stage_stamp(value: Any) -> tuple[Any, ...]:
    """Snapshot every public and private stage fact except the stamp itself."""

    if type(value) is not ExecutedGlyphStage:
        raise TypeError("glyph-stage evidence must have the canonical runtime type")
    stage_fields = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(ExecutedGlyphStage)
        if item.name != "_proof_stamp"
    )
    return (
        _EXECUTED_GLYPH_STAGE_PROOF_VERSION,
        structural_proof_signature(stage_fields),
    )


def _executed_glyph_stage_fields_are_valid(value: Any) -> bool:
    """Validate stage structure and ZHIR decision-to-endpoint correspondence."""

    if type(value) is not ExecutedGlyphStage:
        return False
    event = value.event
    if (
        type(event) is not ExecutedOperatorEvent
        or not isinstance(event.glyph, Glyph)
        or event.stage_schedule != TWO_PHASE_JACOBI
        or type(event.nodes_processed) is not int
        or event.nodes_processed < 0
    ):
        return False
    if (
        type(value.pre_interval_index) is not int
        or type(value.post_interval_index) is not int
        or value.pre_interval_index != event.event_index
        or value.post_interval_index != event.event_index + 1
    ):
        return False
    booleans = (
        value.endpoint_capture_complete,
        value.exact_runtime_endpoint_bound,
        value.exact_common_metric_bridge,
        value._represented_affine_gain_bound_at_observed_endpoint_certified,
        value.pre_interval_positive,
        value.post_interval_positive,
        value.solver_accuracy_certified,
        value.future_or_repeated_schedule_stability_certified,
    )
    if any(type(item) is not bool for item in booleans):
        return False
    optional_booleans = (
        value.pre_flow_endpoint_continuous,
        value.post_flow_endpoint_continuous,
        value.pre_flow_metric_compatible,
        value.post_flow_metric_compatible,
    )
    if any(item is not None and type(item) is not bool for item in optional_booleans):
        return False
    if value.endpoint_capture_complete != bool(
        value.left is not None and value.right is not None
    ):
        return False
    if value.exact_runtime_endpoint_bound and not value.endpoint_capture_complete:
        return False
    expected_gain_claim = bool(
        value.exact_runtime_endpoint_bound
        and value.exact_common_metric_bridge
        and value.exact_energy_gain_upper_bound is not None
    )
    if (
        value._represented_affine_gain_bound_at_observed_endpoint_certified
        != expected_gain_claim
    ):
        return False
    if not expected_gain_claim and value.exact_energy_gain_upper_bound is not None:
        return False
    if value.pre_flow_evidence is not None and type(
        value.pre_flow_evidence
    ) is not ExecutedNodalFlowInterval:
        return False
    if value.post_flow_evidence is not None and type(
        value.post_flow_evidence
    ) is not ExecutedNodalFlowInterval:
        return False

    from ..physics.runtime_flow_stability import NodalFlowStateSnapshot

    endpoint_nodes: list[tuple[Any, ...]] = []
    for snapshot in (value.left, value.right):
        if snapshot is None:
            continue
        if type(snapshot) is not NodalFlowStateSnapshot:
            return False
        nodes = object.__getattribute__(snapshot, "nodes")
        if type(nodes) is not tuple:
            return False
        endpoint_nodes.append(nodes)
    if len(endpoint_nodes) == 2 and not _same_structural_value(
        endpoint_nodes[0], endpoint_nodes[1]
    ):
        return False

    observations = value.mutation_decision_observations
    if type(observations) is not tuple:
        return False
    if event.glyph is not Glyph.ZHIR:
        return not observations
    if len(observations) != event.nodes_processed:
        return False
    for index, observation in enumerate(observations):
        if (
            type(observation) is not MutationStageDecisionObservation
            or observation.target_index != index
            or observation.glyph is not Glyph.ZHIR
            or not observation._proof_fields_are_intact()
        ):
            return False
        observation_signature = structural_proof_signature(observation.node)
        if any(
            all(
                observation_signature != structural_proof_signature(node)
                for node in nodes
            )
            for nodes in endpoint_nodes
        ):
            return False
    return True



_REPRESENTED_OPERATION_PROOF_VERSION = "represented_epi_schedule_operation_v1"
_REPRESENTED_COMPOSITION_PROOF_VERSION = "represented_epi_schedule_composition_v1"


def _exact_vector_is_well_typed(value: Any) -> bool:
    return bool(type(value) is tuple and all(type(item) is Fraction for item in value))


def _represented_operation_stamp(
    *,
    position: int,
    operation_kind: str,
    operation_index: int,
    operator_name: str | None,
    nodes: tuple[Any, ...] | None,
    exact_epi_before: tuple[Fraction, ...] | None,
    exact_epi_after: tuple[Fraction, ...] | None,
    exact_metric_ray_before: tuple[Fraction, ...] | None,
    exact_metric_ray_after: tuple[Fraction, ...] | None,
    exact_energy_gain_upper_bound: Fraction | None,
    ineligibility_reasons: tuple[str, ...],
) -> tuple[Any, ...]:
    return (
        _REPRESENTED_OPERATION_PROOF_VERSION,
        position,
        operation_kind,
        operation_index,
        operator_name,
        nodes,
        exact_epi_before,
        exact_epi_after,
        exact_metric_ray_before,
        exact_metric_ray_after,
        exact_energy_gain_upper_bound,
        ineligibility_reasons,
    )


def _validate_represented_operation_fields(
    *,
    position: Any,
    operation_kind: Any,
    operation_index: Any,
    operator_name: Any,
    nodes: Any,
    exact_epi_before: Any,
    exact_epi_after: Any,
    exact_metric_ray_before: Any,
    exact_metric_ray_after: Any,
    exact_energy_gain_upper_bound: Any,
    ineligibility_reasons: Any,
) -> None:
    if type(position) is not int or position < 0:
        raise ValueError("operation position must be a nonnegative integer")
    if operation_kind not in ("flow", "glyph"):
        raise ValueError("operation_kind must be 'flow' or 'glyph'")
    if type(operation_index) is not int or operation_index < 0:
        raise ValueError("operation_index must be a nonnegative integer")
    if operation_kind == "flow" and operator_name is not None:
        raise ValueError("flow operations cannot declare an operator_name")
    if operation_kind == "glyph" and (
        type(operator_name) is not str or not operator_name
    ):
        raise ValueError("glyph operations require a nonempty operator_name")
    if nodes is not None and type(nodes) is not tuple:
        raise TypeError("operation nodes must be a tuple or None")

    exact_vectors = (
        ("exact_epi_before", exact_epi_before, False),
        ("exact_epi_after", exact_epi_after, False),
        ("exact_metric_ray_before", exact_metric_ray_before, True),
        ("exact_metric_ray_after", exact_metric_ray_after, True),
    )
    for label, value, is_metric in exact_vectors:
        if value is None:
            continue
        if not _exact_vector_is_well_typed(value):
            raise TypeError(f"{label} must be an exact Fraction tuple or None")
        if nodes is None or len(value) != len(nodes):
            raise ValueError(f"{label} must align with operation nodes")
        if is_metric and _normalized_fraction_metric(value) != value:
            raise ValueError(f"{label} must be a normalized positive metric")
    if exact_energy_gain_upper_bound is not None and (
        type(exact_energy_gain_upper_bound) is not Fraction
        or exact_energy_gain_upper_bound < 0
    ):
        raise ValueError("operation gain must be a nonnegative Fraction or None")
    if (
        type(ineligibility_reasons) is not tuple
        or any(
            type(reason) is not str or not reason
            for reason in ineligibility_reasons
        )
        or len(set(ineligibility_reasons)) != len(ineligibility_reasons)
    ):
        raise ValueError("ineligibility_reasons must contain unique nonempty strings")
    complete = bool(
        nodes is not None
        and exact_epi_before is not None
        and exact_epi_after is not None
        and exact_metric_ray_before is not None
        and exact_metric_ray_after is not None
        and exact_metric_ray_before == exact_metric_ray_after
        and exact_energy_gain_upper_bound is not None
    )
    if ineligibility_reasons and exact_energy_gain_upper_bound is not None:
        raise ValueError("an ineligible operation cannot publish an exact gain")
    if not ineligibility_reasons and not complete:
        raise ValueError("an eligible operation requires complete exact map evidence")


@dataclass(frozen=True, slots=True)
class RepresentedEPIScheduleOperation:
    """One indexed represented EPI map or an explicit abstention.

    Its gain belongs to the exact rational affine map represented by this
    observed operation, not to every input of the binary64 runtime map.
    """

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
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        fields = dict(
            position=self.position,
            operation_kind=self.operation_kind,
            operation_index=self.operation_index,
            operator_name=self.operator_name,
            nodes=self.nodes,
            exact_epi_before=self.exact_epi_before,
            exact_epi_after=self.exact_epi_after,
            exact_metric_ray_before=self.exact_metric_ray_before,
            exact_metric_ray_after=self.exact_metric_ray_after,
            exact_energy_gain_upper_bound=self.exact_energy_gain_upper_bound,
            ineligibility_reasons=self.ineligibility_reasons,
        )
        _validate_represented_operation_fields(**fields)
        if (
            type(self._proof_stamp) is not tuple
            or self._proof_stamp != _represented_operation_stamp(**fields)
        ):
            raise ValueError("represented operation proof fields are inconsistent")

    @property
    def represented_affine_gain_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and not self.ineligibility_reasons
        )

    def _proof_fields_are_intact(self) -> bool:
        fields = dict(
            position=self.position,
            operation_kind=self.operation_kind,
            operation_index=self.operation_index,
            operator_name=self.operator_name,
            nodes=self.nodes,
            exact_epi_before=self.exact_epi_before,
            exact_epi_after=self.exact_epi_after,
            exact_metric_ray_before=self.exact_metric_ray_before,
            exact_metric_ray_after=self.exact_metric_ray_after,
            exact_energy_gain_upper_bound=self.exact_energy_gain_upper_bound,
            ineligibility_reasons=self.ineligibility_reasons,
        )
        try:
            _validate_represented_operation_fields(**fields)
        except (TypeError, ValueError):
            return False
        return bool(
            type(self._proof_stamp) is tuple
            and self._proof_stamp == _represented_operation_stamp(**fields)
        )


def _represented_operation(
    *,
    position: int,
    operation_kind: str,
    operation_index: int,
    operator_name: str | None,
    nodes: tuple[Any, ...] | None,
    exact_epi_before: tuple[Fraction, ...] | None,
    exact_epi_after: tuple[Fraction, ...] | None,
    exact_metric_ray_before: tuple[Fraction, ...] | None,
    exact_metric_ray_after: tuple[Fraction, ...] | None,
    exact_energy_gain_upper_bound: Fraction | None,
    ineligibility_reasons: tuple[str, ...] | list[str],
) -> RepresentedEPIScheduleOperation:
    reasons = tuple(dict.fromkeys(ineligibility_reasons))
    gain = None if reasons else exact_energy_gain_upper_bound
    fields = dict(
        position=position,
        operation_kind=operation_kind,
        operation_index=operation_index,
        operator_name=operator_name,
        nodes=nodes,
        exact_epi_before=exact_epi_before,
        exact_epi_after=exact_epi_after,
        exact_metric_ray_before=exact_metric_ray_before,
        exact_metric_ray_after=exact_metric_ray_after,
        exact_energy_gain_upper_bound=gain,
        ineligibility_reasons=reasons,
    )
    return RepresentedEPIScheduleOperation(
        **fields,
        _proof_stamp=_represented_operation_stamp(**fields),
    )


def _represented_composition_conditions(
    nodes: tuple[Any, ...],
    positive_flow_interval_indices: tuple[int, ...],
    event_indices: tuple[int, ...],
    operations: tuple[RepresentedEPIScheduleOperation, ...],
) -> tuple[tuple[str, bool], ...]:
    positive = set(positive_flow_interval_indices)
    expected_keys: list[tuple[str, int]] = []
    for interval_index in range(len(event_indices) + 1):
        if interval_index in positive:
            expected_keys.append(("flow", interval_index))
        if interval_index < len(event_indices):
            expected_keys.append(("glyph", interval_index))
    complete_cardinality = bool(
        tuple(operation.position for operation in operations)
        == tuple(range(len(operations)))
        and tuple(
            (operation.operation_kind, operation.operation_index)
            for operation in operations
        )
        == tuple(expected_keys)
    )
    flow_operations = tuple(
        operation for operation in operations if operation.operation_kind == "flow"
    )
    glyph_operations = tuple(
        operation for operation in operations if operation.operation_kind == "glyph"
    )
    all_flows = all(item.represented_affine_gain_certified for item in flow_operations)
    all_glyphs = all(
        item.represented_affine_gain_certified
        for item in glyph_operations
    )
    all_operations = bool(
        operations
        and all(
            item.represented_affine_gain_certified
            for item in operations
        )
    )
    nodes_match = bool(operations and all(item.nodes == nodes for item in operations))
    support_continuity = bool(
        operations
        and all(item.nodes is not None for item in operations)
        and all(
            left.nodes == right.nodes
            for left, right in zip(operations, operations[1:])
        )
    )
    endpoints_complete = bool(
        operations
        and all(
            item.exact_epi_before is not None and item.exact_epi_after is not None
            for item in operations
        )
    )
    endpoint_continuity = bool(
        endpoints_complete
        and all(
            left.exact_epi_after == right.exact_epi_before
            for left, right in zip(operations, operations[1:])
        )
    )
    metrics_complete = bool(
        operations
        and all(
            item.exact_metric_ray_before is not None
            and item.exact_metric_ray_after is not None
            and item.exact_metric_ray_before == item.exact_metric_ray_after
            for item in operations
        )
    )
    rays = tuple(
        item.exact_metric_ray_before
        for item in operations
        if item.exact_metric_ray_before is not None
    )
    common_metric = bool(
        metrics_complete and rays and all(ray == rays[0] for ray in rays[1:])
    )
    return (
        ("complete_operation_cardinality", complete_cardinality),
        ("all_positive_flows_exact_affine", all_flows),
        ("all_glyph_stages_represented_affine", all_glyphs),
        ("all_operations_represented_affine", all_operations),
        ("all_operation_nodes_match", nodes_match),
        ("exact_operation_support_continuity", support_continuity),
        ("exact_operation_endpoint_continuity", endpoint_continuity),
        ("one_exact_normalized_metric", common_metric),
        ("nonempty_operation_trace", bool(operations)),
    )


def _represented_composition_stamp(
    *,
    nodes: tuple[Any, ...],
    positive_flow_interval_indices: tuple[int, ...],
    event_indices: tuple[int, ...],
    operations: tuple[RepresentedEPIScheduleOperation, ...],
    exact_normalized_metric: tuple[Fraction, ...] | None,
    exact_operation_energy_gain_factors: tuple[Fraction, ...],
    exact_energy_gain_upper_bound: Fraction | None,
    conditions: tuple[tuple[str, bool], ...],
) -> tuple[Any, ...]:
    return (
        _REPRESENTED_COMPOSITION_PROOF_VERSION,
        nodes,
        positive_flow_interval_indices,
        event_indices,
        tuple(operation._proof_stamp for operation in operations),
        exact_normalized_metric,
        exact_operation_energy_gain_factors,
        exact_energy_gain_upper_bound,
        conditions,
    )


def _complete_represented_operation_factors(
    operations: tuple[RepresentedEPIScheduleOperation, ...],
) -> tuple[Fraction, ...]:
    factors: list[Fraction] = []
    for operation in operations:
        factor = operation.exact_energy_gain_upper_bound
        if factor is None:
            raise ValueError("a certified operation cannot omit its exact gain")
        factors.append(factor)
    return tuple(factors)


@dataclass(frozen=True, slots=True)
class ObservedRepresentedEPIScheduleComposition:
    """Sealed represented-map composition for one observed finite trace."""

    nodes: tuple[Any, ...]
    positive_flow_interval_indices: tuple[int, ...]
    event_indices: tuple[int, ...]
    operations: tuple[RepresentedEPIScheduleOperation, ...]
    exact_normalized_metric: tuple[Fraction, ...] | None
    exact_operation_energy_gain_factors: tuple[Fraction, ...]
    exact_energy_gain_upper_bound: Fraction | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    scope: str = field(default=_REPRESENTED_COMPOSITION_SCOPE, init=False)

    @property
    def runtime_schedule_global_gain_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def full_multichannel_stability_certified(self) -> bool:
        return False

    @property
    def future_or_repeated_schedule_stability_certified(self) -> bool:
        return False

    def __post_init__(self) -> None:
        if type(self.nodes) is not tuple:
            raise TypeError("composition nodes must be a tuple")
        if (
            type(self.event_indices) is not tuple
            or any(type(index) is not int for index in self.event_indices)
            or self.event_indices != tuple(range(len(self.event_indices)))
        ):
            raise ValueError(
                "event_indices must be the complete zero-based event range"
            )
        if (
            type(self.positive_flow_interval_indices) is not tuple
            or any(
                type(index) is not int
                for index in self.positive_flow_interval_indices
            )
            or self.positive_flow_interval_indices
            != tuple(sorted(set(self.positive_flow_interval_indices)))
            or any(
                index < 0 or index > len(self.event_indices)
                for index in self.positive_flow_interval_indices
            )
        ):
            raise ValueError(
                "positive flow indices must be sorted, unique and in range"
            )
        if (
            type(self.operations) is not tuple
            or any(
                type(operation) is not RepresentedEPIScheduleOperation
                or not operation._proof_fields_are_intact()
                for operation in self.operations
            )
        ):
            raise ValueError("composition operations must contain intact proof records")
        if (
            type(self.conditions) is not tuple
            or any(
                type(condition) is not tuple
                or len(condition) != 2
                or type(condition[0]) is not str
                or type(condition[1]) is not bool
                for condition in self.conditions
            )
            or len({name for name, _ in self.conditions}) != len(self.conditions)
        ):
            raise ValueError("composition conditions must be uniquely named booleans")
        expected_conditions = _represented_composition_conditions(
            self.nodes,
            self.positive_flow_interval_indices,
            self.event_indices,
            self.operations,
        )
        if self.conditions != expected_conditions:
            raise ValueError("composition conditions do not match operation evidence")
        certified = all(passed for _, passed in expected_conditions)
        expected_factors = (
            _complete_represented_operation_factors(self.operations)
            if certified
            else ()
        )
        if self.exact_operation_energy_gain_factors != expected_factors:
            raise ValueError(
                "composition factors do not match complete operation evidence"
            )
        expected_metric = (
            self.operations[0].exact_metric_ray_before
            if certified
            else None
        )
        if self.exact_normalized_metric != expected_metric:
            raise ValueError(
                "composition metric does not match complete operation evidence"
            )
        expected_gain = (
            math.prod(expected_factors, start=Fraction(1))
            if certified
            else None
        )
        if self.exact_energy_gain_upper_bound != expected_gain:
            raise ValueError("composition gain does not match its exact factors")
        fields = dict(
            nodes=self.nodes,
            positive_flow_interval_indices=self.positive_flow_interval_indices,
            event_indices=self.event_indices,
            operations=self.operations,
            exact_normalized_metric=self.exact_normalized_metric,
            exact_operation_energy_gain_factors=(
                self.exact_operation_energy_gain_factors
            ),
            exact_energy_gain_upper_bound=self.exact_energy_gain_upper_bound,
            conditions=self.conditions,
        )
        if (
            type(self._proof_stamp) is not tuple
            or self._proof_stamp != _represented_composition_stamp(**fields)
        ):
            raise ValueError("represented composition proof fields are inconsistent")

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except (TypeError, ValueError):
            return False
        return True

    @property
    def represented_affine_composition_gain_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def represented_map_global_disagreement_contraction_certified(self) -> bool:
        return bool(
            self.represented_affine_composition_gain_certified
            and self.exact_energy_gain_upper_bound is not None
            and self.exact_energy_gain_upper_bound < 1
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("composition_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)


@dataclass(frozen=True, slots=True)
class OperatorEventExecutionResult:
    """Immutable evidence that one complete finite schedule committed.

    pressure_refresh_callback_invocations counts explicit stage callbacks that
    returned successfully during this schedule.
    """

    schedule: OperatorEventSchedule
    target_nodes: tuple[Any, ...]
    flow_interval_indices: tuple[int, ...]
    positive_flow_interval_indices: tuple[int, ...]
    events: tuple[ExecutedOperatorEvent, ...]
    final_time: float
    integrator_name: str | None
    pressure_refresh_callback_invocations: int
    flow_certification_requested: bool = False
    flow_interval_evidence: tuple[ExecutedNodalFlowInterval, ...] = ()
    stage_certification_requested: bool = False
    glyph_stage_evidence: tuple[ExecutedGlyphStage, ...] = ()
    represented_epi_schedule_composition: (
        ObservedRepresentedEPIScheduleComposition | None
    ) = None
    runtime_clock_checked: bool = field(default=True, init=False)
    flow_provenance: str = field(default=_FLOW_PROVENANCE, init=False)
    nodal_flow_inputs: str = field(default=_NODAL_FLOW_INPUTS, init=False)
    whole_schedule_graph_state_atomic: bool = field(default=True, init=False)
    operator_jumps_have_zero_duration: bool = field(default=True, init=False)
    solver_accuracy_certified: bool = field(default=False, init=False)
    adaptive_u2_u4_policy: bool = field(default=False, init=False)
    external_side_effects_rolled_back: bool = field(default=False, init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=False,
        init=False,
    )
    flow_scope: str = field(default=_FLOW_SCOPE, init=False)

    def __post_init__(self) -> None:
        """Reject missing, reordered, substituted, or altered stage evidence."""

        if type(self.stage_certification_requested) is not bool:
            raise TypeError("stage_certification_requested must be a bool")
        if type(self.events) is not tuple:
            raise TypeError("events must be a tuple")
        if type(self.glyph_stage_evidence) is not tuple:
            raise TypeError("glyph_stage_evidence must be a tuple")
        if not self.stage_certification_requested:
            if self.glyph_stage_evidence:
                raise ValueError(
                    "disabled stage certification cannot carry glyph-stage evidence"
                )
            if self.represented_epi_schedule_composition is not None:
                raise ValueError(
                    "disabled stage certification cannot carry a represented "
                    "schedule composition"
                )
            return

        if len(self.glyph_stage_evidence) != len(self.events):
            raise ValueError(
                "stage evidence must contain one record per committed event"
            )
        for stage, event in zip(
            self.glyph_stage_evidence,
            self.events,
            strict=True,
        ):
            if type(event) is not ExecutedOperatorEvent:
                raise TypeError("events contain a noncanonical record")
            if (
                type(stage) is not ExecutedGlyphStage
                or not stage._proof_fields_are_intact()
            ):
                raise ValueError("glyph-stage evidence proof fields are not intact")
            try:
                event_matches = _same_structural_value(stage.event, event)
            except Exception as exc:
                raise ValueError(
                    "stage evidence event identity is unreadable"
                ) from exc
            if not event_matches:
                raise ValueError(
                    "stage evidence does not match committed event order"
                )
            if event.nodes_processed != len(self.target_nodes):
                raise ValueError(
                    "committed event target count does not match execution"
                )
            if event.glyph is Glyph.ZHIR:
                observation_nodes = tuple(
                    observation.node
                    for observation in stage.mutation_decision_observations
                )
                try:
                    mutation_targets_match = _same_structural_value(
                        observation_nodes,
                        self.target_nodes,
                    )
                except Exception as exc:
                    raise ValueError(
                        "Mutation stage target support is unreadable"
                    ) from exc
                if not mutation_targets_match:
                    raise ValueError(
                        "Mutation stage observations do not match execution targets"
                    )

        composition = self.represented_epi_schedule_composition
        if (
            type(composition) is not ObservedRepresentedEPIScheduleComposition
            or not composition._proof_fields_are_intact()
        ):
            raise ValueError(
                "stage certification requires one intact represented composition"
            )
        try:
            nodes_match = _same_structural_value(
                composition.nodes,
                self.target_nodes,
            )
        except Exception as exc:
            raise ValueError(
                "represented composition target support is unreadable"
            ) from exc
        if not nodes_match:
            raise ValueError(
                "represented composition target support does not match execution"
            )
        if composition.event_indices != tuple(
            event.event_index for event in self.events
        ):
            raise ValueError(
                "represented composition event indices do not match execution"
            )
        if (
            composition.positive_flow_interval_indices
            != self.positive_flow_interval_indices
        ):
            raise ValueError(
                "represented composition flow indices do not match execution"
            )

    def _all_positive_flow_intervals(self, attribute: str) -> bool | None:
        if not self.flow_certification_requested:
            return None
        if len(self.flow_interval_evidence) != len(
            self.positive_flow_interval_indices
        ):
            return False
        return all(
            bool(getattr(evidence, attribute))
            for evidence in self.flow_interval_evidence
        )

    @property
    def all_positive_flow_intervals_binary64_identified(self) -> bool | None:
        """Aggregate trusted binary64 realization, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_binary64_interval_identified"
        )

    @property
    def all_positive_flow_intervals_binary64_held_pressure_identified(
        self,
    ) -> bool | None:
        """Aggregate trusted held-pressure replays, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_binary64_held_pressure_interval_identified"
        )

    @property
    def all_positive_flow_intervals_exact_affine(self) -> bool | None:
        """Aggregate exact affine identification, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_exact_affine_map_identified"
        )

    @property
    def all_positive_flow_intervals_contracting(self) -> bool | None:
        """Aggregate exact contraction result, or ``None`` if disabled."""

        return self._all_positive_flow_intervals(
            "runtime_bound_global_disagreement_contraction_certified"
        )

    @property
    def all_glyph_stages_represented_affine(self) -> bool | None:
        """Aggregate exact runtime-bound glyph gains, or ``None`` if disabled."""

        if not self.stage_certification_requested:
            return None
        if len(self.glyph_stage_evidence) != len(self.events):
            return False
        for evidence, event in zip(
            self.glyph_stage_evidence,
            self.events,
            strict=True,
        ):
            if (
                type(evidence) is not ExecutedGlyphStage
                or not evidence._proof_fields_are_intact()
            ):
                return False
            try:
                if not _same_structural_value(evidence.event, event):
                    return False
            except Exception:
                return False
            if (
                not evidence
                .represented_affine_gain_bound_at_observed_endpoint_certified
            ):
                return False
        return True


def _require_graph(graph: Any) -> nx.Graph:
    """Return a supported NetworkX graph before any runtime inspection."""

    if not isinstance(
        graph,
        (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
    ):
        raise TypeError("graph must be a NetworkX graph")
    return graph


def _require_runtime_clock(
    graph: nx.Graph,
    expected: float,
    *,
    boundary: str,
) -> float:
    """Require one exact binary64 runtime boundary without resetting it."""

    if "_t" not in graph.graph:
        raise TNFRValueError(
            "Operator-event execution requires an existing binary64 runtime clock.",
            context={"boundary": boundary, "reason": "missing_graph_time"},
        )
    current = graph.graph["_t"]
    if type(current) is not float or not math.isfinite(current):
        raise TNFRValueError(
            "Operator-event execution requires a finite binary64 runtime clock.",
            context={
                "boundary": boundary,
                "runtime_time": repr(current),
                "runtime_time_type": type(current).__name__,
            },
        )
    if current != expected:
        raise TNFRValueError(
            "The live runtime clock does not match the scheduled boundary.",
            context={
                "boundary": boundary,
                "runtime_time": current,
                "scheduled_time": expected,
            },
        )
    return current


def _validate_schedule_clock(schedule: OperatorEventSchedule) -> None:
    """Reject schedules the current binary64 runtime cannot represent."""

    diagnostic = diagnose_operator_event_runtime_clock(schedule)
    if diagnostic.clock_binding_ready:
        return
    raise TNFRValueError(
        "The operator-event schedule cannot bind to the current runtime clock.",
        context={
            "collapsed_positive_interval_indices": (
                diagnostic.collapsed_positive_interval_indices
            ),
            "nonadditive_positive_interval_indices": (
                diagnostic.nonadditive_positive_interval_indices
            ),
            "zhir_event_indices_without_positive_preflow": (
                diagnostic.zhir_event_indices_without_positive_preflow
            ),
            "zhir_event_indices_with_collapsed_preflow": (
                diagnostic.zhir_event_indices_with_collapsed_preflow
            ),
            "zhir_event_indices_with_duration_mismatch": (
                diagnostic.zhir_event_indices_with_duration_mismatch
            ),
        },
        suggestion=(
            "Choose a representable time origin and positive pre-flow for every "
            "Mutation event whose timestamp difference equals its declared duration."
        ),
    )


def _validate_hybrid_log(graph: nx.Graph) -> None:
    """Require the graph event channel to be appendable before any write."""

    if _HYBRID_EVENT_LOG in graph.graph and not isinstance(
        graph.graph[_HYBRID_EVENT_LOG], list
    ):
        raise TNFRValueError(
            "hybrid_event_log must be a list when already configured.",
            context={
                "history_channel": _HYBRID_EVENT_LOG,
                "value_type": type(graph.graph[_HYBRID_EVENT_LOG]).__name__,
            },
        )


def _prepare_word(
    schedule: OperatorEventSchedule,
    context: Mapping[str, Any] | None,
) -> tuple[tuple[Any, ...], Any | None]:
    """Validate one canonical word without stale Mutation preflight."""

    if not schedule.operator_names:
        return (), None

    from ..validation import validate_sequence
    from .grammar_execution import ValidatedSequence
    from .registry import get_operator_class

    operators = tuple(
        get_operator_class(name)() for name in schedule.operator_names
    )
    validation_context = None if context is None else dict(context)
    outcome = validate_sequence(
        list(schedule.operator_names),
        context=validation_context,
    )
    if not outcome.passed:
        raise TNFRValueError(
            "Invalid operator-event sequence: "
            + outcome.summary.get("message", "validation failed"),
            context={
                "sequence": schedule.operator_names,
                "outcome": outcome.summary,
            },
        )
    return operators, ValidatedSequence(
        operators,
        context=validation_context,
    )


def _flow_runtime_metadata(
    graph: nx.Graph,
    interval: StructuralFlowInterval,
    integrator: Any,
    *,
    method: str | None,
) -> _FlowRuntimeMetadata:
    """Derive execution metadata without invoking the integrator twice."""

    from ..gamma import _get_gamma_spec

    integrator_name = type(integrator).__qualname__
    bound_integrate = getattr(integrator, "integrate", None)
    instance_attributes = getattr(integrator, "__dict__", {})
    provenance = bool(
        type(integrator) is _CanonicalDefaultIntegrator
        and "integrate" not in instance_attributes
        and getattr(bound_integrate, "__self__", None) is integrator
        and getattr(bound_integrate, "__func__", None)
        is _CANONICAL_DEFAULT_INTEGRATE
        and _CanonicalDefaultIntegrator.integrate
        is _CANONICAL_DEFAULT_INTEGRATE
    )
    extended_requested = bool(
        graph.graph.get("use_extended_dynamics", False)
    )
    if not provenance:
        return _FlowRuntimeMetadata(
            integrator_name=integrator_name,
            integrator_provenance_certified=False,
            resolved_method=None,
            resolved_substeps=None,
            gamma_is_none=None,
            extended_dynamics_requested=extended_requested,
        )

    _, substeps, _, resolved_method = _canonical_prepare_integration_params(
        graph,
        interval.duration,
        interval.start_time,
        method,
    )
    gamma_spec = _get_gamma_spec(graph)
    gamma_is_none = bool(
        isinstance(gamma_spec, Mapping)
        and gamma_spec.get("type", "none") == "none"
    )
    return _FlowRuntimeMetadata(
        integrator_name=integrator_name,
        integrator_provenance_certified=True,
        resolved_method=resolved_method,
        resolved_substeps=substeps,
        gamma_is_none=gamma_is_none,
        extended_dynamics_requested=extended_requested,
    )


def _capture_interval_endpoint(
    graph: nx.Graph,
) -> tuple[NodalFlowStateSnapshot | None, bool]:
    """Capture a detached endpoint while treating unsupported state as abstention."""

    from ..physics.runtime_flow_stability import capture_nodal_flow_state

    try:
        return capture_nodal_flow_state(graph), True
    except (TypeError, ValueError, nx.NetworkXException):
        return None, False


def _exact_binary64_vector(values: Any) -> tuple[Fraction, ...] | None:
    """Interpret one finite vector through its represented binary64 values."""

    try:
        materialized = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in materialized):
        return None
    return tuple(Fraction.from_float(value) for value in materialized)


def _snapshot_metric_ray(
    snapshot: NodalFlowStateSnapshot | None,
) -> tuple[Fraction, ...] | None:
    """Derive the exact ``d_i / nu_f_i`` ray from one captured endpoint."""

    if snapshot is None or not snapshot.nodes:
        return None
    degrees = tuple(sum(row, Fraction(0)) for row in snapshot.conductance)
    if any(value <= 0 for value in degrees) or any(
        value <= 0 for value in snapshot.exact_nu_f
    ):
        return None
    weights = tuple(
        degree / capacity
        for degree, capacity in zip(
            degrees,
            snapshot.exact_nu_f,
            strict=True,
        )
    )
    return _normalized_fraction_metric(weights)


def _require_nested_certificate_identity(
    result: NetworkStageResult,
    event: ExecutedOperatorEvent,
    certificate: Any,
    *,
    kind: str,
    expected_nodes: tuple[Any, ...] | None,
) -> None:
    """Reject executor-owned evidence bound to a different stage identity."""

    from .operator_contracts import contract_for

    try:
        contract = contract_for(event.operator_name)
    except KeyError as exc:
        raise RuntimeError(
            "scheduled event has no canonical operator contract"
        ) from exc
    failures: list[str] = []
    if result.operator != event.operator_name:
        failures.append("stage_result_operator")
    if result.glyph != event.glyph.value:
        failures.append("stage_result_glyph")
    if result.schedule != event.stage_schedule:
        failures.append("stage_result_schedule")
    if contract.name != event.operator_name:
        failures.append("canonical_operator")
    if contract.glyph != event.glyph.value:
        failures.append("canonical_glyph")
    if getattr(certificate, "operator_name", None) != contract.english_name:
        failures.append("certificate_operator")
    if getattr(certificate, "glyph", None) != event.glyph.value:
        failures.append("certificate_glyph")
    certificate_nodes = getattr(certificate, "nodes", None)
    if type(certificate_nodes) is not tuple:
        failures.append("certificate_nodes_type")
    elif len(certificate_nodes) != result.nodes_processed:
        failures.append("certificate_node_count")
    if expected_nodes is not None and certificate_nodes != expected_nodes:
        failures.append("certificate_target_support")
    if kind == "pointwise":
        if getattr(certificate, "stage_schedule", None) != result.schedule:
            failures.append("certificate_stage_schedule")
        if getattr(certificate, "target_nodes", None) != certificate_nodes:
            failures.append("certificate_target_nodes")
        if event.glyph not in {
            Glyph.AL,
            Glyph.SHA,
            Glyph.VAL,
            Glyph.NUL,
            Glyph.ZHIR,
            Glyph.NAV,
        }:
            failures.append("certificate_family")
    elif event.glyph not in {Glyph.EN, Glyph.RA}:
        failures.append("certificate_family")
    if failures:
        raise RuntimeError(
            "executor-owned EPI certificate identity mismatch: "
            + ",".join(failures)
        )


def _glyph_certificate_facts(
    result: NetworkStageResult,
    event: ExecutedOperatorEvent,
    expected_nodes: tuple[Any, ...] | None,
) -> _GlyphCertificateFacts:
    """Validate and normalize one certificate returned by the stage executor."""

    certificate = result.epi_jump_certificate
    kind = result.epi_jump_certificate_kind
    reason = result.epi_jump_certificate_abstention_reason
    if certificate is None:
        return _GlyphCertificateFacts(
            certificate_kind=kind,
            certificate=None,
            abstention_reason=reason or "glyph_has_no_executor_bound_epi_certificate",
            nodes=None,
            exact_epi_before=None,
            exact_epi_after=None,
            exact_metric_ray_before=None,
            exact_metric_ray_after=None,
            intrinsic_common_metric_bridge=False,
            exact_energy_gain_upper_bound=None,
        )

    if kind == "pointwise":
        from ..physics.pointwise_stage_stability import (
            PointwiseEPIJumpRealizationCertificate,
        )

        if type(certificate) is not PointwiseEPIJumpRealizationCertificate:
            reason = "pointwise_certificate_type_mismatch"
        elif not certificate._proof_fields_are_intact():
            reason = "pointwise_certificate_proof_fields_not_intact"
        else:
            _require_nested_certificate_identity(
                result,
                event,
                certificate,
                kind=kind,
                expected_nodes=expected_nodes,
            )
            pre_flow = certificate.pre_diffusion_certificate
            post_flow = certificate.post_diffusion_certificate
            before = _exact_binary64_vector(certificate.state_before)
            after = _exact_binary64_vector(
                certificate.runtime_proposed_state_after
            )
            pre_metric = (
                None
                if pre_flow is None
                else _normalized_exact_metric(pre_flow.metric_weights)
            )
            post_metric = (
                None
                if post_flow is None
                else _normalized_exact_metric(post_flow.metric_weights)
            )
            bridge = bool(certificate.supports_hybrid_common_metric_bridge)
            if not bridge and reason is None:
                failures = certificate.failed_common_metric_bridge_conditions
                reason = "pointwise_exact_common_metric_gain_not_certified"
                if failures:
                    reason += ":" + ",".join(failures)
            gain = (
                certificate.exact_common_metric_energy_gain_bound
                if bridge
                else None
            )
            return _GlyphCertificateFacts(
                certificate_kind=kind,
                certificate=certificate,
                abstention_reason=reason,
                nodes=tuple(certificate.nodes),
                exact_epi_before=before,
                exact_epi_after=after,
                exact_metric_ray_before=pre_metric,
                exact_metric_ray_after=post_metric,
                intrinsic_common_metric_bridge=bridge,
                exact_energy_gain_upper_bound=gain,
            )
    elif kind == "neighbor":
        from ..physics.network_stage_stability import (
            AllTargetNeighborStageCertificate,
            _validate_bridge_stage_certificate,
        )

        if type(certificate) is not AllTargetNeighborStageCertificate:
            reason = "neighbor_certificate_type_mismatch"
        else:
            try:
                validated = _validate_bridge_stage_certificate(certificate)
            except (AttributeError, TypeError, ValueError, OverflowError):
                reason = "neighbor_certificate_proof_fields_not_intact"
            else:
                _require_nested_certificate_identity(
                    result,
                    event,
                    certificate,
                    kind=kind,
                    expected_nodes=expected_nodes,
                )
                step = validated.step
                jump = step.pre_metric_affine_jump_certificate
                jump_valid = bool(jump.supports_global_gain_theorem)
                bridge = bool(
                    validated.all_local_snapshots_in_affine_model_domain
                    and validated.represented_consensus_subspace_preserved
                    and validated.runtime_exact_match
                    and validated.pre_post_metric_exactly_proportional
                    and jump_valid
                )
                if not bridge and reason is None:
                    reason = "neighbor_exact_common_metric_gain_not_certified"
                return _GlyphCertificateFacts(
                    certificate_kind=kind,
                    certificate=certificate,
                    abstention_reason=reason,
                    nodes=tuple(certificate.nodes),
                    exact_epi_before=_exact_binary64_vector(step.state_before),
                    exact_epi_after=_exact_binary64_vector(
                        step.runtime_accepted_state_after
                    ),
                    exact_metric_ray_before=_normalized_exact_metric(
                        step.pre_diffusion_certificate.metric_weights
                    ),
                    exact_metric_ray_after=_normalized_exact_metric(
                        step.post_diffusion_certificate.metric_weights
                    ),
                    intrinsic_common_metric_bridge=bridge,
                    exact_energy_gain_upper_bound=(
                        jump.exact_quotient_energy_gain_upper_bound
                        if bridge
                        else None
                    ),
                )
    else:
        reason = "unknown_epi_jump_certificate_kind"

    return _GlyphCertificateFacts(
        certificate_kind=kind,
        certificate=certificate,
        abstention_reason=reason,
        nodes=None,
        exact_epi_before=None,
        exact_epi_after=None,
        exact_metric_ray_before=None,
        exact_metric_ray_after=None,
        intrinsic_common_metric_bridge=False,
        exact_energy_gain_upper_bound=None,
    )


def _flow_by_interval(
    evidence: tuple[ExecutedNodalFlowInterval, ...] | list[ExecutedNodalFlowInterval],
) -> dict[int, ExecutedNodalFlowInterval]:
    """Index detached positive-flow evidence without accepting duplicates."""

    indexed: dict[int, ExecutedNodalFlowInterval] = {}
    for item in evidence:
        index = item.interval.index
        if index in indexed:
            raise RuntimeError("duplicate runtime flow interval evidence")
        indexed[index] = item
    return indexed


def _adjacent_flow_checks(
    flow: ExecutedNodalFlowInterval | None,
    endpoint: NodalFlowStateSnapshot | None,
    metric_ray: tuple[Fraction, ...] | None,
    *,
    side: str,
) -> tuple[bool | None, bool | None]:
    """Check exact endpoint and metric continuity at one positive boundary."""

    if flow is None:
        return None, None
    certificate = flow.certificate
    if certificate is None or endpoint is None:
        return False, False
    flow_endpoint = certificate.right if side == "pre" else certificate.left
    endpoint_continuous = flow_endpoint == endpoint
    flow_metric = _normalized_fraction_metric(certificate.exact_metric_weights)
    metric_compatible = bool(
        flow_metric is not None
        and metric_ray is not None
        and flow_metric == metric_ray
    )
    return endpoint_continuous, metric_compatible


def _finalize_glyph_stage(
    pending: _PendingGlyphStage,
    schedule: OperatorEventSchedule,
    flow_index: Mapping[int, ExecutedNodalFlowInterval],
) -> ExecutedGlyphStage:
    """Bind an executor certificate to captured endpoints and adjacent flows."""

    facts = _glyph_certificate_facts(
        pending.result,
        pending.event,
        None if pending.left is None else pending.left.nodes,
    )
    left = pending.left
    right = pending.right
    complete = bool(
        pending.left_captured
        and pending.right_captured
        and left is not None
        and right is not None
    )
    left_ray = _snapshot_metric_ray(left)
    right_ray = _snapshot_metric_ray(right)
    endpoint_bound = bool(
        complete
        and facts.nodes is not None
        and left is not None
        and right is not None
        and facts.nodes == left.nodes == right.nodes
        and facts.exact_epi_before == left.exact_epi
        and facts.exact_epi_after == right.exact_epi
    )
    common_metric = bool(
        facts.intrinsic_common_metric_bridge
        and left_ray is not None
        and right_ray is not None
        and facts.exact_metric_ray_before == left_ray
        and facts.exact_metric_ray_after == right_ray
        and left_ray == right_ray
    )
    gain_certified = bool(
        endpoint_bound
        and common_metric
        and facts.exact_energy_gain_upper_bound is not None
    )
    abstention_reasons: list[str] = []
    if facts.abstention_reason is not None:
        abstention_reasons.append(facts.abstention_reason)
    if not complete:
        abstention_reasons.append("glyph_endpoint_capture_incomplete")
    elif facts.certificate is not None and not endpoint_bound:
        abstention_reasons.append("glyph_certificate_endpoint_mismatch")
    if facts.certificate is not None and not common_metric:
        abstention_reasons.append("glyph_exact_common_metric_bridge_not_certified")
    abstention_reason = (
        ";".join(dict.fromkeys(abstention_reasons))
        if abstention_reasons
        else None
    )

    event_index = pending.event.event_index
    pre_interval = schedule.intervals[event_index]
    post_interval = schedule.intervals[event_index + 1]
    pre_positive = pre_interval.exact_duration > 0
    post_positive = post_interval.exact_duration > 0
    pre_flow = flow_index.get(pre_interval.index) if pre_positive else None
    post_flow = flow_index.get(post_interval.index) if post_positive else None
    pre_endpoint, pre_metric = _adjacent_flow_checks(
        pre_flow,
        left,
        facts.exact_metric_ray_before,
        side="pre",
    )
    post_endpoint, post_metric = _adjacent_flow_checks(
        post_flow,
        right,
        facts.exact_metric_ray_after,
        side="post",
    )
    if pre_positive and pre_flow is None:
        pre_endpoint = False
        pre_metric = False
    if post_positive and post_flow is None:
        post_endpoint = False
        post_metric = False

    stage = ExecutedGlyphStage(
        event=pending.event,
        certificate_kind=facts.certificate_kind,
        certificate=facts.certificate,
        certificate_abstention_reason=abstention_reason,
        left=left,
        right=right,
        endpoint_capture_complete=complete,
        exact_runtime_endpoint_bound=endpoint_bound,
        exact_metric_ray_before=facts.exact_metric_ray_before,
        exact_metric_ray_after=facts.exact_metric_ray_after,
        exact_common_metric_bridge=common_metric,
        exact_energy_gain_upper_bound=(
            facts.exact_energy_gain_upper_bound if gain_certified else None
        ),
        _represented_affine_gain_bound_at_observed_endpoint_certified=gain_certified,
        pre_interval_index=pre_interval.index,
        post_interval_index=post_interval.index,
        pre_interval_positive=pre_positive,
        post_interval_positive=post_positive,
        pre_flow_evidence=pre_flow,
        post_flow_evidence=post_flow,
        pre_flow_endpoint_continuous=pre_endpoint,
        post_flow_endpoint_continuous=post_endpoint,
        pre_flow_metric_compatible=pre_metric,
        post_flow_metric_compatible=post_metric,
        mutation_decision_observations=(
            pending.result.mutation_decision_observations
        ),
    )
    return replace(
        stage,
        _proof_stamp=_executed_glyph_stage_stamp(stage),
    )


def _flow_composition_operation(
    *,
    position: int,
    interval: StructuralFlowInterval,
    expected_nodes: tuple[Any, ...],
    evidence: ExecutedNodalFlowInterval | None,
) -> RepresentedEPIScheduleOperation:
    reasons: list[str] = []
    certificate = None if evidence is None else evidence.certificate
    if evidence is None:
        reasons.append("flow_evidence_missing")
    elif certificate is None:
        reasons.append("flow_certificate_unavailable")
        if evidence.abstention_reason is not None:
            reasons.append(f"flow_certificate_abstained:{evidence.abstention_reason}")
    elif not _has_intact_nodal_flow_certificate(certificate):
        reasons.append("flow_certificate_proof_fields_not_intact")

    operation_nodes: tuple[Any, ...] | None = None
    exact_before: tuple[Fraction, ...] | None = None
    exact_after: tuple[Fraction, ...] | None = None
    metric_ray: tuple[Fraction, ...] | None = None
    gain: Fraction | None = None
    if (
        evidence is not None
        and certificate is not None
        and _has_intact_nodal_flow_certificate(certificate)
    ):
        left_nodes = tuple(certificate.left.nodes)
        right_nodes = tuple(certificate.right.nodes)
        if left_nodes != right_nodes:
            reasons.append("flow_support_changed")
        else:
            operation_nodes = left_nodes
            exact_before = certificate.left.exact_epi
            exact_after = certificate.right.exact_epi
        if operation_nodes != expected_nodes:
            reasons.append("flow_node_order_mismatch")
        if evidence.interval != interval:
            reasons.append("flow_interval_identity_mismatch")
        if not evidence.integrator_provenance_certified:
            reasons.append("flow_integrator_provenance_not_certified")
        if not evidence.runtime_bound_exact_affine_map_identified:
            reasons.append("flow_represented_affine_map_not_identified")
            reasons.extend(
                f"flow_model_condition_failed:{name}"
                for name in certificate.euler_map_abstention_reasons
            )
        metric_ray = _normalized_fraction_metric(certificate.exact_metric_weights)
        if metric_ray is None:
            reasons.append("flow_exact_positive_metric_unavailable")
        gain = certificate.exact_quotient_energy_gain_upper_bound
        if gain is None:
            reasons.append("flow_exact_energy_gain_unavailable")
        if operation_nodes is None:
            exact_before = None
            exact_after = None
            metric_ray = None

    return _represented_operation(
        position=position,
        operation_kind="flow",
        operation_index=interval.index,
        operator_name=None,
        nodes=operation_nodes,
        exact_epi_before=exact_before,
        exact_epi_after=exact_after,
        exact_metric_ray_before=metric_ray,
        exact_metric_ray_after=metric_ray,
        exact_energy_gain_upper_bound=gain,
        ineligibility_reasons=reasons,
    )


def _glyph_composition_operation(
    *,
    position: int,
    event: ScheduledOperatorEvent,
    expected_nodes: tuple[Any, ...],
    evidence: ExecutedGlyphStage | None,
) -> RepresentedEPIScheduleOperation:
    reasons: list[str] = []
    if evidence is None:
        reasons.append("glyph_stage_evidence_missing")
        return _represented_operation(
            position=position,
            operation_kind="glyph",
            operation_index=event.event_index,
            operator_name=event.operator_name,
            nodes=None,
            exact_epi_before=None,
            exact_epi_after=None,
            exact_metric_ray_before=None,
            exact_metric_ray_after=None,
            exact_energy_gain_upper_bound=None,
            ineligibility_reasons=reasons,
        )

    if (
        type(evidence) is not ExecutedGlyphStage
        or not evidence._proof_fields_are_intact()
    ):
        reasons.append("glyph_stage_proof_fields_not_intact")
        return _represented_operation(
            position=position,
            operation_kind="glyph",
            operation_index=event.event_index,
            operator_name=event.operator_name,
            nodes=None,
            exact_epi_before=None,
            exact_epi_after=None,
            exact_metric_ray_before=None,
            exact_metric_ray_after=None,
            exact_energy_gain_upper_bound=None,
            ineligibility_reasons=reasons,
        )

    committed = evidence.event
    if (
        committed.event_index != event.event_index
        or committed.cycle_index != event.cycle_index
        or committed.word_position != event.word_position
        or committed.operator_name != event.operator_name
        or committed.glyph is not event.glyph
        or committed.event_time != event.event_time
        or committed.event_offset != event.event_offset
        or committed.exact_event_time != event.exact_event_time
        or committed.stage_schedule != TWO_PHASE_JACOBI
        or committed.nodes_processed != len(expected_nodes)
    ):
        reasons.append("glyph_event_identity_mismatch")
    if evidence.certificate_abstention_reason is not None:
        reasons.append(
            "glyph_certificate_abstained:"
            + evidence.certificate_abstention_reason
        )
    if not evidence.endpoint_capture_complete:
        reasons.append("glyph_endpoint_capture_incomplete")
    if not evidence.exact_runtime_endpoint_bound:
        reasons.append("glyph_runtime_endpoint_not_bound")
    if not evidence.exact_common_metric_bridge:
        reasons.append("glyph_exact_common_metric_bridge_not_certified")
    if not evidence.represented_affine_gain_bound_at_observed_endpoint_certified:
        reasons.append("glyph_represented_affine_gain_not_certified")
    if evidence.exact_energy_gain_upper_bound is None:
        reasons.append("glyph_exact_energy_gain_unavailable")

    left = evidence.left
    right = evidence.right
    operation_nodes: tuple[Any, ...] | None = None
    exact_before: tuple[Fraction, ...] | None = None
    exact_after: tuple[Fraction, ...] | None = None
    if left is None or right is None:
        reasons.append("glyph_support_observation_incomplete")
    elif left.nodes != right.nodes:
        reasons.append("glyph_support_changed")
    else:
        operation_nodes = left.nodes
        exact_before = left.exact_epi
        exact_after = right.exact_epi
    if operation_nodes != expected_nodes:
        reasons.append("glyph_node_order_mismatch")
    metric_before = evidence.exact_metric_ray_before
    metric_after = evidence.exact_metric_ray_after
    if operation_nodes is None:
        exact_before = None
        exact_after = None
        metric_before = None
        metric_after = None

    return _represented_operation(
        position=position,
        operation_kind="glyph",
        operation_index=event.event_index,
        operator_name=event.operator_name,
        nodes=operation_nodes,
        exact_epi_before=exact_before,
        exact_epi_after=exact_after,
        exact_metric_ray_before=metric_before,
        exact_metric_ray_after=metric_after,
        exact_energy_gain_upper_bound=evidence.exact_energy_gain_upper_bound,
        ineligibility_reasons=reasons,
    )


def _compose_observed_represented_epi_schedule(
    schedule: OperatorEventSchedule,
    nodes: tuple[Any, ...],
    flows: tuple[ExecutedNodalFlowInterval, ...],
    stages: tuple[ExecutedGlyphStage, ...],
) -> ObservedRepresentedEPIScheduleComposition:
    """Compose only the rational maps represented by the complete observed trace."""

    flow_index = _flow_by_interval(flows)
    stage_index = {item.event.event_index: item for item in stages}
    if len(stage_index) != len(stages):
        raise RuntimeError("duplicate runtime glyph-stage evidence")
    positive_indices = tuple(
        interval.index
        for interval in schedule.intervals
        if interval.exact_duration > 0
    )
    if any(index not in positive_indices for index in flow_index):
        raise RuntimeError("unexpected runtime flow interval evidence")
    if any(index < 0 or index >= schedule.event_count for index in stage_index):
        raise RuntimeError("unexpected runtime glyph-stage evidence")

    operations: list[RepresentedEPIScheduleOperation] = []
    for interval in schedule.intervals:
        if interval.exact_duration > 0:
            operations.append(
                _flow_composition_operation(
                    position=len(operations),
                    interval=interval,
                    expected_nodes=nodes,
                    evidence=flow_index.get(interval.index),
                )
            )
        if interval.index < schedule.event_count:
            event = schedule.events[interval.index]
            operations.append(
                _glyph_composition_operation(
                    position=len(operations),
                    event=event,
                    expected_nodes=nodes,
                    evidence=stage_index.get(event.event_index),
                )
            )

    operation_tuple = tuple(operations)
    event_indices = tuple(range(schedule.event_count))
    conditions = _represented_composition_conditions(
        nodes,
        positive_indices,
        event_indices,
        operation_tuple,
    )
    certified = all(passed for _, passed in conditions)
    factors = (
        _complete_represented_operation_factors(operation_tuple)
        if certified
        else ()
    )
    metric = operation_tuple[0].exact_metric_ray_before if certified else None
    gain = math.prod(factors, start=Fraction(1)) if certified else None
    fields = dict(
        nodes=nodes,
        positive_flow_interval_indices=positive_indices,
        event_indices=event_indices,
        operations=operation_tuple,
        exact_normalized_metric=metric,
        exact_operation_energy_gain_factors=factors,
        exact_energy_gain_upper_bound=gain,
        conditions=conditions,
    )
    return ObservedRepresentedEPIScheduleComposition(
        **fields,
        _proof_stamp=_represented_composition_stamp(**fields),
    )


def _clipping_intervened(
    left: NodalFlowStateSnapshot,
    right: NodalFlowStateSnapshot,
    interval: StructuralFlowInterval,
    metadata: _FlowRuntimeMetadata,
) -> bool | None:
    """Detect clipping against the sequential unclipped held-pressure replay."""

    substeps = metadata.resolved_substeps
    if not (
        metadata.integrator_provenance_certified
        and metadata.resolved_method == "euler"
        and type(substeps) is int
        and substeps >= 1
        and metadata.gamma_is_none is True
        and left.nodes == right.nodes
    ):
        return None

    from ..mathematics.unified_numerical import np

    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            rate = np.multiply(
                np.asarray(left.nu_f, dtype=float),
                np.asarray(left.delta_nfr, dtype=float),
            )
            rate = np.add(rate, np.zeros_like(rate))
            dt_step = interval.duration / substeps
            replay = np.asarray(left.epi, dtype=float)
            for _ in range(substeps):
                increment = np.multiply(dt_step, rate)
                replay = np.add(replay, increment)
    except (FloatingPointError, TypeError, ValueError, OverflowError):
        return None
    if not bool(np.all(np.isfinite(replay))):
        return None
    return any(
        float(expected).hex() != observed.hex()
        for expected, observed in zip(replay, right.epi, strict=True)
    )


def _execute_flow_interval(
    graph: nx.Graph,
    interval: StructuralFlowInterval,
    integrator: Any,
    *,
    method: str | None,
    n_jobs: int | None,
    include_flow_certificate: bool,
) -> ExecutedNodalFlowInterval | None:
    """Advance one positive interval and optionally bind endpoint evidence."""

    from ..dynamics.runtime import _record_mutation_flow_boundary

    _require_runtime_clock(
        graph,
        interval.start_time,
        boundary=f"interval[{interval.index}].start",
    )
    if interval.exact_duration == 0:
        return None

    metadata = (
        _flow_runtime_metadata(
            graph,
            interval,
            integrator,
            method=method,
        )
        if include_flow_certificate
        else None
    )
    _record_mutation_flow_boundary(graph)
    if include_flow_certificate:
        left, left_captured = _capture_interval_endpoint(graph)
    else:
        left, left_captured = None, False

    integrator.integrate(
        graph,
        dt=interval.duration,
        t=interval.start_time,
        method=method,
        n_jobs=n_jobs,
    )

    _require_runtime_clock(
        graph,
        interval.end_time,
        boundary=f"interval[{interval.index}].end",
    )
    if include_flow_certificate:
        right, right_captured = _capture_interval_endpoint(graph)
    else:
        right, right_captured = None, False
    _record_mutation_flow_boundary(graph)

    if not include_flow_certificate:
        return None
    if metadata is None:
        raise RuntimeError("flow certification metadata was not prepared")
    if not left_captured or not right_captured:
        if not left_captured and not right_captured:
            reason = "left_and_right_state_capture_failed"
        elif not left_captured:
            reason = "left_state_capture_failed"
        else:
            reason = "right_state_capture_failed"
        evidence = ExecutedNodalFlowInterval(
            interval=interval,
            certificate=None,
            abstention_reason=reason,
            integrator_name=metadata.integrator_name,
            integrator_provenance_certified=(
                metadata.integrator_provenance_certified
            ),
            resolved_method=metadata.resolved_method,
            resolved_substeps=metadata.resolved_substeps,
            gamma_is_none=metadata.gamma_is_none,
            clipping_applied=None,
            extended_dynamics_requested=(
                metadata.extended_dynamics_requested
            ),
        )
        return replace(
            evidence,
            _proof_stamp=_executed_flow_interval_stamp(evidence),
        )
    if left is None or right is None:
        raise RuntimeError("captured flow endpoint is unexpectedly absent")

    clipping_applied = _clipping_intervened(
        left,
        right,
        interval,
        metadata,
    )
    from ..physics.runtime_flow_stability import (
        certify_observed_nodal_flow_interval,
    )

    certificate = certify_observed_nodal_flow_interval(
        left,
        right,
        duration=interval.duration,
        integrator_name=(
            "DefaultIntegrator"
            if metadata.integrator_provenance_certified
            else metadata.integrator_name
        ),
        method=metadata.resolved_method,
        substeps=metadata.resolved_substeps,
        gamma_is_none=metadata.gamma_is_none,
        clipping_applied=clipping_applied,
        extended_dynamics_requested=(
            metadata.extended_dynamics_requested
        ),
    )
    evidence = ExecutedNodalFlowInterval(
        interval=interval,
        certificate=certificate,
        abstention_reason=None,
        integrator_name=metadata.integrator_name,
        integrator_provenance_certified=(
            metadata.integrator_provenance_certified
        ),
        resolved_method=metadata.resolved_method,
        resolved_substeps=metadata.resolved_substeps,
        gamma_is_none=metadata.gamma_is_none,
        clipping_applied=clipping_applied,
        extended_dynamics_requested=metadata.extended_dynamics_requested,
    )
    return replace(
        evidence,
        _proof_stamp=_executed_flow_interval_stamp(evidence),
    )


def _require_exact_stage(
    event: ScheduledOperatorEvent,
    result: NetworkStageResult,
    *,
    target_count: int,
) -> None:
    """Reject a replacement, override or partial target realization."""

    if (
        result.operator != event.operator_name
        or result.glyph != event.glyph.value
        or result.schedule != TWO_PHASE_JACOBI
        or result.nodes_processed != target_count
    ):
        raise TNFRValueError(
            "The scheduled operator event was not realized by its exact "
            "canonical all-target stage.",
            context={
                "event_index": event.event_index,
                "operator_name": event.operator_name,
                "glyph": event.glyph.value,
                "observed_operator": result.operator,
                "observed_glyph": result.glyph,
                "observed_schedule": result.schedule,
                "observed_nodes_processed": result.nodes_processed,
                "expected_nodes_processed": target_count,
            },
        )


def execute_operator_event_schedule(
    graph: nx.Graph,
    schedule: OperatorEventSchedule,
    *,
    context: Mapping[str, Any] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    suppress_birth_warnings: bool = False,
    include_flow_certificates: bool = False,
    include_stage_certificates: bool = False,
) -> OperatorEventExecutionResult:
    """Execute one finite flow/jump schedule inside a whole-schedule rollback.

    The live graph time must already equal schedule.start_time. Every positive
    flow is delegated once to the graph's configured nodal EPI integrator with
    the schedule duration; the resulting binary64 clock must equal the
    scheduled endpoint. Operator jumps use the same fixed initial target tuple
    and canonical dispatcher as ordinary network words. Any failure restores
    graph-owned state, topology, histories, event telemetry and runtime caches.
    A secondary rollback fault is attached to the primary failure instead of
    replacing it. Warnings and effects emitted by external integrators,
    callbacks, monitors or resources remain outside that graph transaction.

    Event timestamps identify zero-duration jumps in hybrid_event_log. Boundary
    sampling may retain the same physical coordinate as the right endpoint of
    one flow and the left endpoint of the next, but a same-time EPI jump resets
    that node's history and can never become an epi_time_history secant.
    ``include_flow_certificates`` captures detached endpoints around each
    positive interval. ``include_stage_certificates`` additionally requests
    executor-bound EPI certificates for every supported glyph, captures each
    jump boundary, and implies interval capture so adjacent evidence can be
    checked. Both result channels remain detached from graph metadata and make
    no solver-accuracy or repeated-stability claim.
    """

    graph = _require_graph(graph)
    if type(schedule) is not OperatorEventSchedule:
        raise TypeError("schedule must be an OperatorEventSchedule")
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping or None")
    if method is not None and type(method) is not str:
        raise TypeError("method must be a string or None")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral)
    ):
        raise TypeError("n_jobs must be an integer or None")
    if type(suppress_birth_warnings) is not bool:
        raise TypeError("suppress_birth_warnings must be a bool")
    if type(include_flow_certificates) is not bool:
        raise TypeError("include_flow_certificates must be a bool")
    if type(include_stage_certificates) is not bool:
        raise TypeError("include_stage_certificates must be a bool")
    schedule.__post_init__()
    _validate_schedule_clock(schedule)
    _require_runtime_clock(
        graph,
        schedule.start_time,
        boundary="schedule.start",
    )
    _validate_hybrid_log(graph)
    operators, execution_word = _prepare_word(schedule, context)

    from ..dynamics.runtime import (
        _record_mutation_flow_boundary,
        _resolve_integrator_instance,
    )
    from .word_execution import execute_network_operator_stage

    targets = tuple(graph.nodes())
    configured_compute_delta_nfr = graph.graph.get("compute_delta_nfr")
    pressure_refresh_callback_invocations = 0
    if callable(configured_compute_delta_nfr):

        def compute_delta_nfr(live_graph: nx.Graph) -> None:
            nonlocal pressure_refresh_callback_invocations
            configured_compute_delta_nfr(live_graph)
            pressure_refresh_callback_invocations += 1

    else:
        compute_delta_nfr = None
    transaction = GraphTransactionSnapshot(graph)
    events_committed: list[ExecutedOperatorEvent] = []
    flow_interval_evidence: list[ExecutedNodalFlowInterval] = []
    pending_glyph_stages: list[_PendingGlyphStage] = []
    effective_flow_certification = bool(
        include_flow_certificates or include_stage_certificates
    )
    positive_intervals = tuple(
        interval.index
        for interval in schedule.intervals
        if interval.exact_duration > 0
    )
    integrator = None

    try:
        if positive_intervals:
            integrator = _resolve_integrator_instance(graph)

        with warnings.catch_warnings():
            if suppress_birth_warnings:
                warnings.filterwarnings(
                    "ignore",
                    message=r".*has no sources.*",
                )
            for interval in schedule.intervals:
                if interval.exact_duration > 0:
                    if integrator is None:
                        raise RuntimeError(
                            "positive flow interval has no configured integrator"
                        )
                    evidence = _execute_flow_interval(
                        graph,
                        interval,
                        integrator,
                        method=method,
                        n_jobs=n_jobs,
                        include_flow_certificate=(
                            effective_flow_certification
                        ),
                    )
                    if evidence is not None:
                        flow_interval_evidence.append(evidence)
                else:
                    _require_runtime_clock(
                        graph,
                        interval.start_time,
                        boundary=f"interval[{interval.index}].start",
                    )
                    _require_runtime_clock(
                        graph,
                        interval.end_time,
                        boundary=f"interval[{interval.index}].end",
                    )

                if interval.index >= schedule.event_count:
                    continue

                event = schedule.events[interval.index]
                operator = operators[event.word_position]
                sequence_step = (
                    None
                    if execution_word is None
                    else execution_word.step(event.word_position)
                )
                if include_stage_certificates:
                    stage_left, stage_left_captured = _capture_interval_endpoint(
                        graph
                    )
                else:
                    stage_left, stage_left_captured = None, False
                stage_kwargs = {
                    "sequence_context": sequence_step,
                    "compute_delta_nfr": compute_delta_nfr,
                }
                if include_stage_certificates:
                    stage_kwargs["include_epi_jump_certificate"] = True
                result = execute_network_operator_stage(
                    graph,
                    operator,
                    targets,
                    **stage_kwargs,
                )
                _require_exact_stage(
                    event,
                    result,
                    target_count=len(targets),
                )
                _require_runtime_clock(
                    graph,
                    event.event_time,
                    boundary=f"event[{event.event_index}]",
                )
                if include_stage_certificates:
                    stage_right, stage_right_captured = _capture_interval_endpoint(
                        graph
                    )
                else:
                    stage_right, stage_right_captured = None, False
                _record_mutation_flow_boundary(graph)
                committed = ExecutedOperatorEvent.from_stage(event, result)
                if include_stage_certificates:
                    pending_glyph_stages.append(
                        _PendingGlyphStage(
                            event=committed,
                            result=result,
                            left=stage_left,
                            right=stage_right,
                            left_captured=stage_left_captured,
                            right_captured=stage_right_captured,
                        )
                    )
                sink = graph.graph.setdefault(_HYBRID_EVENT_LOG, [])
                if not isinstance(sink, list):
                    raise TNFRValueError(
                        "hybrid_event_log changed type during event execution.",
                        context={
                            "event_index": event.event_index,
                            "value_type": type(sink).__name__,
                        },
                    )
                sink.append(committed.as_record())
                events_committed.append(committed)

        _require_runtime_clock(
            graph,
            schedule.end_time,
            boundary="schedule.end",
        )
        flow_tuple = tuple(flow_interval_evidence)
        flow_index = _flow_by_interval(flow_tuple)
        glyph_stage_tuple = tuple(
            _finalize_glyph_stage(pending, schedule, flow_index)
            for pending in pending_glyph_stages
        )
        represented_composition = (
            _compose_observed_represented_epi_schedule(
                schedule,
                targets,
                flow_tuple,
                glyph_stage_tuple,
            )
            if include_stage_certificates
            else None
        )
        return OperatorEventExecutionResult(
            schedule=schedule,
            target_nodes=targets,
            flow_interval_indices=tuple(
                interval.index for interval in schedule.intervals
            ),
            positive_flow_interval_indices=positive_intervals,
            events=tuple(events_committed),
            final_time=schedule.end_time,
            integrator_name=(
                None if integrator is None else type(integrator).__qualname__
            ),
            pressure_refresh_callback_invocations=(
                pressure_refresh_callback_invocations
            ),
            flow_certification_requested=effective_flow_certification,
            flow_interval_evidence=flow_tuple,
            stage_certification_requested=include_stage_certificates,
            glyph_stage_evidence=glyph_stage_tuple,
            represented_epi_schedule_composition=represented_composition,
        )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


__all__ = (
    "ExecutedGlyphStage",
    "ExecutedNodalFlowInterval",
    "ExecutedOperatorEvent",
    "ObservedRepresentedEPIScheduleComposition",
    "RepresentedEPIScheduleOperation",
    "OperatorEventExecutionResult",
    "execute_operator_event_schedule",
)
