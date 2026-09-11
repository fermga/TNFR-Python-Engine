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
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from functools import cached_property, partial
from numbers import Integral
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CellType,
    FunctionType,
    MethodType,
)
from typing import TYPE_CHECKING, Any

import networkx as nx

from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
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
from ..utils._structural_signature import (
    binary64_vectors_are_identical as _binary64_vectors_are_identical,
    proof_stamps_are_identical,
    structural_object_state_signature,
    structural_proof_signature,
)
from .event_timing import (
    OperatorEventSchedule,
    PhysicalFlowPartition,
    ScheduledOperatorEvent,
    StructuralFlowInterval,
    build_physical_flow_partition,
    diagnose_operator_event_runtime_clock,
)
from ._reception_kernel import RECEPTION_NO_SOURCES_WARNING_PATTERN
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    MutationStageDecisionObservation,
    NetworkStageResult,
    ReceptionStageObservation,
    _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES,
    _graph_factory_items,
    _graph_factory_state_signature,
    _graph_transaction_protected_values,
    _networkx_internal_mapping_items,
    _networkx_internal_mapping_state_signature,
    _networkx_runtime_layout,
    _runtime_class_mro,
    _runtime_class_namespace,
    _runtime_instance_namespace,
    _runtime_mapping_items,
    _runtime_owner_qualified_slot_state,
    _runtime_stored_attribute,
    _set_runtime_mapping_item,
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
    "with pressure held inside each integrator call and refreshed only at "
    "declared physical boundaries or explicit operator-stage callbacks; "
    "solver accuracy and adaptive U2/U4 behavior are not certified"
)
_PRESSURE_REFRESH_BOUNDARY_SCOPE = (
    "one executor-owned pressure-hook invocation at an explicit physical "
    "boundary; completion and preservation of non-pressure nodal state, edge "
    "state, graph configuration and capturable callback-owned state are "
    "observed. Known derived pressure/cache "
    "metadata may change without failing this check but remains inside graph "
    "rollback; external callback effects remain outside rollback"
)
_PRESSURE_REFRESHED_PARTITION_SCOPE = (
    "one declared flow interval executed as explicit physical segments with "
    "DeltaNFR refreshed at every boundary, including the terminal "
    "boundary. The sealed trace identifies finite executed binary64 steps and "
    "per-segment frozen modal diagnostics only; it does not prove solver order, "
    "mesh convergence, future behavior or adaptive U2/U4 policy"
)
_PRESSURE_REFRESH_BOUNDARY_PROOF_VERSION = (
    "pressure_refresh_boundary_observation_v2"
)
_PRESSURE_REFRESHED_PARTITION_PROOF_VERSION = (
    "executed_pressure_refreshed_flow_partition_v1"
)
_EXECUTED_OPERATOR_EVENT_PROOF_VERSION = "executed_operator_event_v1"
_OPERATOR_EVENT_EXECUTION_RESULT_PROOF_VERSION = (
    "operator_event_execution_result_v1"
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


def _raw_proof_stamp_or_none(value: Any) -> tuple[Any, ...] | None:
    """Read a proof stamp without letting a deleted slot escape validation."""

    try:
        raw_stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return raw_stamp if type(raw_stamp) is tuple else None


def _has_intact_nodal_flow_certificate(value: Any) -> bool:
    """Recognize only a canonical interval certificate with intact proof fields."""

    from ..physics.runtime_flow_stability import NodalFlowIntervalCertificate

    try:
        return bool(
            type(value) is NodalFlowIntervalCertificate
            and value._proof_fields_are_intact()
        )
    except BaseException:
        return False


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
    _proof_stamp: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __getattribute__(self, name: str) -> Any:
        """Expose historical dataclass contracts through the proof seal."""

        if name == "zero_duration":
            try:
                raw = object.__getattribute__(self, name)
                checker = object.__getattribute__(self, "_proof_fields_are_intact")
                return bool(raw is True and checker())
            except BaseException:
                return False
        if name == "feeds_epi_time_history":
            return False
        if name == "history_channel":
            return _HYBRID_EVENT_LOG
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        raw_stamp = _raw_proof_stamp_or_none(self)
        if not (type(raw_stamp) is tuple and tuple.__len__(raw_stamp) == 0):
            try:
                expected = _sealed_dataclass_stamp(
                    self,
                    _EXECUTED_OPERATOR_EVENT_PROOF_VERSION,
                )
            except BaseException as exc:
                raise ValueError(
                    "executed event proof fields are inconsistent"
                ) from exc
            if not proof_stamps_are_identical(raw_stamp, expected):
                raise ValueError("executed event proof fields are inconsistent")

        integer_fields = (
            self.event_index,
            self.cycle_index,
            self.word_position,
            self.nodes_processed,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError("event indices and node count must be nonnegative ints")
        if type(self.operator_name) is not str or not self.operator_name:
            raise TypeError("operator_name must be a nonempty string")
        if not isinstance(self.glyph, Glyph):
            raise TypeError("glyph must be a Glyph")
        if type(self.event_time) is not float or not math.isfinite(self.event_time):
            raise ValueError("event_time must be a finite binary64 float")
        if type(self.event_offset) is not Fraction or self.event_offset < 0:
            raise ValueError("event_offset must be a nonnegative Fraction")
        if type(self.exact_event_time) is not Fraction:
            raise TypeError("exact_event_time must be a Fraction")
        if self.stage_schedule != TWO_PHASE_JACOBI:
            raise ValueError("event stage_schedule must be two_phase_jacobi")
        if (
            object.__getattribute__(self, "zero_duration") is not True
            or object.__getattribute__(self, "history_channel")
            != _HYBRID_EVENT_LOG
            or object.__getattribute__(self, "feeds_epi_time_history") is not False
        ):
            raise ValueError("executed event contract fields are inconsistent")

    def _proof_fields_are_intact(self) -> bool:
        try:
            raw_stamp = _raw_proof_stamp_or_none(self)
            if type(raw_stamp) is not tuple or tuple.__len__(raw_stamp) == 0:
                return False
            self.__post_init__()
        except BaseException:
            return False
        return True

    @classmethod
    def from_stage(
        cls,
        event: ScheduledOperatorEvent,
        result: NetworkStageResult,
    ) -> "ExecutedOperatorEvent":
        """Bind one immutable schedule event to its accepted stage result."""

        executed = cls(
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
        return replace(
            executed,
            _proof_stamp=_sealed_dataclass_stamp(
                executed,
                _EXECUTED_OPERATOR_EVENT_PROOF_VERSION,
            ),
        )

    def as_record(self) -> dict[str, Any]:
        """Return a detached append-only graph telemetry record."""

        if not self._proof_fields_are_intact():
            raise ValueError("executed event proof fields are not intact")

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
            else _raw_proof_stamp_or_none(certificate)
        ),
        structural_proof_signature(
            (
                value.abstention_reason,
                value.integrator_name,
                object.__getattribute__(
                    value,
                    "integrator_provenance_certified",
                ),
                value.resolved_method,
                value.resolved_substeps,
                value.gamma_is_none,
                value.clipping_applied,
                value.extended_dynamics_requested,
                object.__getattribute__(value, "solver_accuracy_certified"),
                object.__getattribute__(
                    value,
                    "future_or_repeated_schedule_stability_certified",
                ),
            )
        ),
    )


def _executed_flow_wrapper_seal_is_intact(value: Any) -> bool:
    """Check wrapper fields without promoting its optional nested certificate."""

    try:
        return bool(
            type(value) is ExecutedNodalFlowInterval
            and proof_stamps_are_identical(
                _raw_proof_stamp_or_none(value),
                _executed_flow_interval_stamp(value),
            )
        )
    except BaseException:
        return False


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

    def __getattribute__(self, name: str) -> Any:
        """Expose the compatible provenance field through its proof seal."""

        if name == "integrator_provenance_certified":
            try:
                raw = object.__getattribute__(self, name)
                checker = object.__getattribute__(self, "_proof_fields_are_intact")
                return raw is True and checker()
            except BaseException:
                return False
        if name in {
            "solver_accuracy_certified",
            "future_or_repeated_schedule_stability_certified",
        }:
            return False
        return object.__getattribute__(self, name)

    def _proof_fields_are_intact(self) -> bool:
        """Whether this wrapper still matches executor-owned provenance fields."""

        try:
            expected = _executed_flow_interval_stamp(self)
            if not proof_stamps_are_identical(
                _raw_proof_stamp_or_none(self),
                expected,
            ):
                return False
            certificate = object.__getattribute__(self, "certificate")
            if certificate is not None and not _has_intact_nodal_flow_certificate(
                certificate
            ):
                return False
            if (
                object.__getattribute__(self, "solver_accuracy_certified") is not False
                or object.__getattribute__(
                    self,
                    "future_or_repeated_schedule_stability_certified",
                )
                is not False
            ):
                return False
        except BaseException:
            return False
        return True

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


def _nested_runtime_proof_token(value: Any, expected_type: type[Any]) -> Any:
    """Reference one exact sealed child without serializing its fields twice."""

    if type(value) is not expected_type:
        return structural_proof_signature(value)
    stamp = _raw_proof_stamp_or_none(value)
    return (
        "tnfr-nested-proof-stamp-v1",
        expected_type.__module__,
        expected_type.__qualname__,
        stamp if type(stamp) is tuple else None,
    )


def _nested_runtime_proof_sequence(
    value: Any,
    expected_type: type[Any],
) -> Any:
    """Seal an exact tuple of child records through their independent seals."""

    if type(value) is not tuple:
        return structural_proof_signature(value)
    return (
        "tnfr-nested-proof-sequence-v1",
        tuple(
            _nested_runtime_proof_token(item, expected_type)
            for item in tuple.__iter__(value)
        ),
    )


def _sealed_runtime_field_signature(
    owner_type: type[Any],
    name: str,
    value: Any,
) -> Any:
    """Use compact child seals only at closed, validator-owned field sites."""

    if owner_type is ExecutedPressureRefreshedFlowPartition:
        sequence_types = {
            "boundary_observations": PressureRefreshBoundaryObservation,
            "segment_flow_evidence": ExecutedNodalFlowInterval,
            "modal_observations": PhysicalEulerModalObservation,
        }
        expected = sequence_types.get(name)
        if expected is not None:
            return _nested_runtime_proof_sequence(value, expected)
    elif owner_type is ExecutedGlyphStage:
        direct_types = {
            "event": ExecutedOperatorEvent,
            "pre_flow_evidence": ExecutedNodalFlowInterval,
            "post_flow_evidence": ExecutedNodalFlowInterval,
        }
        expected = direct_types.get(name)
        if expected is not None:
            if value is None:
                return structural_proof_signature(None)
            return _nested_runtime_proof_token(value, expected)
        if name == "mutation_decision_observations":
            return _nested_runtime_proof_sequence(
                value,
                MutationStageDecisionObservation,
            )
        if name == "reception_observations":
            return _nested_runtime_proof_sequence(
                value,
                ReceptionStageObservation,
            )
    elif owner_type is OperatorEventExecutionResult:
        sequence_types = {
            "events": ExecutedOperatorEvent,
            "flow_interval_evidence": ExecutedNodalFlowInterval,
            "glyph_stage_evidence": ExecutedGlyphStage,
            "physical_flow_partition_evidence": (
                ExecutedPressureRefreshedFlowPartition
            ),
        }
        expected = sequence_types.get(name)
        if expected is not None:
            return _nested_runtime_proof_sequence(value, expected)
        if name == "represented_epi_schedule_composition":
            if value is None:
                return structural_proof_signature(None)
            return _nested_runtime_proof_token(
                value,
                ObservedRepresentedEPIScheduleComposition,
            )
    return structural_proof_signature(value)


def _sealed_dataclass_stamp(value: Any, version: str) -> tuple[Any, ...]:
    """Seal every field, reusing validated seals of exact nested records."""

    owner_type = type(value)
    return (
        version,
        tuple(
            (
                item.name,
                _sealed_runtime_field_signature(
                    owner_type,
                    item.name,
                    object.__getattribute__(value, item.name),
                ),
            )
            for item in fields(owner_type)
            if item.name != "_proof_stamp"
        ),
    )


@dataclass(frozen=True, slots=True)
class PressureRefreshBoundaryObservation:
    """Executor-owned evidence for one explicit pressure refresh boundary."""

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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    @property
    def callback_completed(self) -> bool:
        """Record that this observation follows a returned callback."""

        return self._proof_fields_are_intact()

    @property
    def external_side_effects_rolled_back(self) -> bool:
        """Exclude callback effects outside graph-owned state."""

        return False

    @property
    def scope(self) -> str:
        """Describe the boundary observation's claim boundary."""

        return _PRESSURE_REFRESH_BOUNDARY_SCOPE

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _sealed_dataclass_stamp(
                self,
                _PRESSURE_REFRESH_BOUNDARY_PROOF_VERSION,
            )
            return proof_stamps_are_identical(
                _raw_proof_stamp_or_none(self),
                expected,
            )
        except BaseException:
            return False

    @property
    def nonpressure_state_preserved(self) -> bool:
        """Whether the callback changed only pressure-related state."""

        return bool(
            self._proof_fields_are_intact()
            and self.callback_completed
            and self.callback_binding_preserved
            and self.callback_state_preserved
            and self.node_support_preserved
            and self.epi_preserved
            and self.capacity_preserved
            and self.conductance_preserved
            and self.edge_state_preserved
            and self.graph_configuration_preserved
            and self.dnfr_weights_preserved_or_canonically_initialized
            and self.phase_preserved
            and self.epi_derivatives_preserved
            and self.mutation_history_preserved
            and self.glyph_history_preserved
            and self.other_nodal_state_preserved
            and self.runtime_clock_preserved
        )


@dataclass(frozen=True, slots=True)
class PhysicalEulerModalObservation:
    """Immutable binary64 modal diagnostic at one refreshed segment start."""

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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _sealed_dataclass_stamp(
                self,
                "physical_euler_modal_observation_v1",
            )
            if not proof_stamps_are_identical(
                _raw_proof_stamp_or_none(self),
                expected,
            ):
                return False
        except BaseException:
            return False
        if (
            type(self.parent_interval_index) is not int
            or self.parent_interval_index < 0
            or type(self.segment_index) is not int
            or self.segment_index < 0
            or type(self.dt) is not float
            or not math.isfinite(self.dt)
            or self.dt <= 0.0
            or type(self.available) is not bool
        ):
            return False
        if self.available:
            if (
                self.abstention_reason is not None
                or type(self.decay_rates) is not tuple
                or not self.decay_rates
                or type(self.modal_multipliers) is not tuple
                or len(self.decay_rates) != len(self.modal_multipliers)
                or type(self.is_euler_stable) is not bool
            ):
                return False
            scalar_fields = (
                self.target_fraction,
                self.spectral_relative_tolerance,
                self.spectral_zero_threshold,
                self.slowest_decay_rate,
                self.fastest_decay_rate,
                self.euler_stability_limit,
                self.maximum_modal_factor,
            )
            if any(
                type(value) is not float or not math.isfinite(value)
                for value in scalar_fields
            ):
                return False
            if (
                self.target_fraction is None
                or not 0.0 < self.target_fraction < 1.0
                or self.spectral_relative_tolerance is None
                or not 0.0 < self.spectral_relative_tolerance < 1.0
                or self.spectral_zero_threshold is None
                or self.spectral_zero_threshold <= 0.0
                or any(
                    type(value) is not float
                    or not math.isfinite(value)
                    or value <= 0.0
                    for value in self.decay_rates
                )
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in self.modal_multipliers
                )
                or tuple(sorted(self.decay_rates)) != self.decay_rates
                or self.slowest_decay_rate != self.decay_rates[0]
                or self.fastest_decay_rate != self.decay_rates[-1]
                or self.euler_stability_limit
                != 2.0 / self.fastest_decay_rate
                or self.maximum_modal_factor
                != max(abs(value) for value in self.modal_multipliers)
                or self.is_euler_stable
                != (self.maximum_modal_factor < 1.0)
                or tuple(
                    1.0 - self.dt * rate for rate in self.decay_rates
                )
                != self.modal_multipliers
                or (
                    self.modal_steps is not None
                    and (
                        type(self.modal_steps) is not int
                        or self.modal_steps < 1
                    )
                )
                or (self.is_euler_stable and self.modal_steps is None)
                or (not self.is_euler_stable and self.modal_steps is not None)
                or type(self.policy_window) is not int
                or self.policy_window < 1
                or type(self.scope) is not str
                or not self.scope
            ):
                return False
        elif (
            type(self.abstention_reason) is not str
            or not self.abstention_reason
            or self.is_euler_stable is not None
            or any(
                value is not None
                for value in (
                    self.target_fraction,
                    self.spectral_relative_tolerance,
                    self.spectral_zero_threshold,
                    self.decay_rates,
                    self.modal_multipliers,
                    self.slowest_decay_rate,
                    self.fastest_decay_rate,
                    self.euler_stability_limit,
                    self.maximum_modal_factor,
                    self.modal_steps,
                    self.policy_window,
                    self.scope,
                )
            )
        ):
            return False
        return True

    @property
    def modal_diagnostic_established(self) -> bool:
        """Whether one available frozen modal diagnostic retains its seal."""

        return bool(self._proof_fields_are_intact() and self.available)


def _nodal_flow_snapshots_are_identical(left: Any, right: Any) -> bool:
    """Compare canonical flow snapshots by sealed value, not object identity."""

    from ..physics.runtime_flow_stability import (
        _nodal_flow_snapshot_proof_signature,
    )

    try:
        return bool(
            _nodal_flow_snapshot_proof_signature(left)
            == _nodal_flow_snapshot_proof_signature(right)
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _partition_gain_facts(
    segments: tuple[ExecutedNodalFlowInterval, ...],
) -> tuple[
    tuple[Fraction, ...] | None,
    tuple[Fraction, ...],
    Fraction | None,
    bool,
]:
    """Derive a common-metric product from intact exact segment maps."""

    rays: list[tuple[Fraction, ...]] = []
    gains: list[Fraction] = []
    for segment in segments:
        certificate = segment.certificate
        if (
            not segment.runtime_bound_exact_affine_map_identified
            or certificate is None
            or certificate.exact_quotient_energy_gain_upper_bound is None
        ):
            return None, (), None, False
        left_ray = _snapshot_metric_ray(certificate.left)
        right_ray = _snapshot_metric_ray(certificate.right)
        if left_ray is None or right_ray is None or left_ray != right_ray:
            return None, (), None, False
        rays.append(left_ray)
        gains.append(certificate.exact_quotient_energy_gain_upper_bound)
    if not rays or any(ray != rays[0] for ray in rays[1:]):
        return None, (), None, False
    factors = tuple(gains)
    return rays[0], factors, math.prod(factors, start=Fraction(1)), True


def _pressure_refreshed_partition_fields_are_valid(value: Any) -> bool:
    """Validate temporal coverage, refreshes and segment endpoint bridges."""

    if type(value) is not ExecutedPressureRefreshedFlowPartition:
        return False
    partition = value.partition
    if type(partition) is not PhysicalFlowPartition:
        return False
    try:
        canonical_partition = build_physical_flow_partition(
            partition.parent_interval,
            partition.segment_durations,
        )
    except (TypeError, ValueError):
        return False
    if partition != canonical_partition:
        return False
    segments = partition.segments
    boundaries = value.boundary_observations
    flows = value.segment_flow_evidence
    modal = value.modal_observations
    if (
        type(boundaries) is not tuple
        or type(flows) is not tuple
        or type(modal) is not tuple
        or len(boundaries) != len(segments) + 1
        or len(flows) != len(segments)
        or len(modal) != len(segments)
        or value.pressure_refresh_callback_invocations != len(boundaries)
    ):
        return False
    callback_identities: list[int] = []
    for index, boundary in enumerate(boundaries):
        if (
            type(boundary) is not PressureRefreshBoundaryObservation
            or not boundary.nonpressure_state_preserved
            or boundary.parent_interval_index != partition.parent_interval.index
            or boundary.boundary_index != index
        ):
            return False
        expected_time = (
            segments[index].start_time
            if index < len(segments)
            else segments[-1].end_time
        )
        expected_exact_time = (
            segments[index].exact_start_time
            if index < len(segments)
            else segments[-1].exact_end_time
        )
        expected_offset = (
            segments[index].start_offset
            if index < len(segments)
            else segments[-1].end_offset
        )
        if (
            structural_proof_signature(boundary.time)
            != structural_proof_signature(expected_time)
            or boundary.exact_time != expected_exact_time
            or boundary.offset != expected_offset
        ):
            return False
        callback_identities.append(boundary.callback_identity)
    if any(identity != callback_identities[0] for identity in callback_identities[1:]):
        return False

    for index, (segment, flow, decision) in enumerate(
        zip(segments, flows, modal, strict=True)
    ):
        if (
            type(flow) is not ExecutedNodalFlowInterval
            or not flow._proof_fields_are_intact()
            or flow.interval != segment
            or flow.certificate is None
            or type(decision) is not PhysicalEulerModalObservation
            or not decision._proof_fields_are_intact()
            or decision.parent_interval_index != partition.parent_interval.index
            or decision.segment_index != index
            or structural_proof_signature(decision.dt)
            != structural_proof_signature(segment.duration)
        ):
            return False
        certificate = flow.certificate
        if (
            not _nodal_flow_snapshots_are_identical(
                boundaries[index].after,
                certificate.left,
            )
            or not _nodal_flow_snapshots_are_identical(
                boundaries[index + 1].before,
                certificate.right,
            )
        ):
            return False

    metric, gains, product, certified = _partition_gain_facts(flows)
    return bool(
        value.exact_common_metric == metric
        and value.exact_segment_gain_bounds == gains
        and value.exact_composed_gain_bound == product
        and value._exact_common_metric_gain_product_certified == certified
    )


@dataclass(frozen=True, slots=True)
class ExecutedPressureRefreshedFlowPartition:
    """Sealed execution trace for one explicitly partitioned physical flow."""

    partition: PhysicalFlowPartition
    boundary_observations: tuple[PressureRefreshBoundaryObservation, ...]
    segment_flow_evidence: tuple[ExecutedNodalFlowInterval, ...]
    modal_observations: tuple[PhysicalEulerModalObservation, ...]
    pressure_refresh_callback_invocations: int
    exact_common_metric: tuple[Fraction, ...] | None
    exact_segment_gain_bounds: tuple[Fraction, ...]
    exact_composed_gain_bound: Fraction | None
    _exact_common_metric_gain_product_certified: bool = field(repr=False)
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    @property
    def solver_accuracy_certified(self) -> bool:
        """Keep solver accuracy outside finite partition evidence."""

        return False

    @property
    def solver_order_certified(self) -> bool:
        """Keep numerical order outside finite partition evidence."""

        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        """Keep mesh convergence outside one finite partition."""

        return False

    @property
    def future_or_repeated_behavior_certified(self) -> bool:
        """Keep future and repeated behavior outside this trace."""

        return False

    @property
    def adaptive_u2_u4_policy_certified(self) -> bool:
        """Keep adaptive grammar policy outside this trace."""

        return False

    @property
    def external_side_effects_rolled_back(self) -> bool:
        """Exclude external callback effects from rollback claims."""

        return False

    @property
    def scope(self) -> str:
        """Describe the physical partition's finite-evidence scope."""

        return _PRESSURE_REFRESHED_PARTITION_SCOPE

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _sealed_dataclass_stamp(
                self,
                _PRESSURE_REFRESHED_PARTITION_PROOF_VERSION,
            )
            if not proof_stamps_are_identical(
                _raw_proof_stamp_or_none(self),
                expected,
            ):
                return False
            if not _pressure_refreshed_partition_fields_are_valid(self):
                return False
        except BaseException:
            return False
        return True

    @property
    def physical_pressure_reevaluated_partition_established(self) -> bool:
        """Whether this is an intact executor-owned physical partition trace."""

        return self._proof_fields_are_intact()

    @property
    def all_segment_binary64_replays_identified(self) -> bool:
        """Whether every explicit segment matches its built-in binary64 replay."""

        return bool(
            self._proof_fields_are_intact()
            and all(
                flow.runtime_bound_binary64_held_pressure_interval_identified
                for flow in self.segment_flow_evidence
            )
        )

    @property
    def all_segment_binary64_intervals_identified(self) -> bool:
        """Whether each segment has trusted built-in runtime provenance."""

        return bool(
            self._proof_fields_are_intact()
            and all(
                flow.runtime_bound_binary64_interval_identified
                for flow in self.segment_flow_evidence
            )
        )

    @property
    def all_segment_exact_affine_maps_identified(self) -> bool:
        """Whether every refreshed segment realizes its rational Euler map."""

        return bool(
            self._proof_fields_are_intact()
            and all(
                flow.runtime_bound_exact_affine_map_identified
                for flow in self.segment_flow_evidence
            )
        )

    @property
    def all_segment_disagreement_contractions_certified(self) -> bool:
        """Whether every identified segment contracts disagreement."""

        return bool(
            self._proof_fields_are_intact()
            and all(
                flow.runtime_bound_global_disagreement_contraction_certified
                for flow in self.segment_flow_evidence
            )
        )

    @property
    def all_boundaries_binary64_pure_epi_pressure_realized(self) -> bool:
        """Whether refreshed pressure equals the binary64 pure-EPI replay by bits."""

        if not self._proof_fields_are_intact():
            return False
        return all(
            _binary64_vectors_are_identical(
                boundary.after.delta_nfr,
                boundary.after.binary64_pure_epi_pressure,
            )
            for boundary in self.boundary_observations
        )

    @property
    def all_segment_modal_diagnostics_applicable(self) -> bool:
        """Whether each frozen pure-EPI segment has a modal diagnostic."""

        return bool(
            self._proof_fields_are_intact()
            and all(
                item.modal_diagnostic_established
                for item in self.modal_observations
            )
        )

    @property
    def all_segment_modal_decisions_stable(self) -> bool:
        """Whether every executed segment realizes a stable Euler diagnostic."""

        return bool(
            self.all_segment_modal_diagnostics_applicable
            and self.all_segment_binary64_replays_identified
            and all(item.is_euler_stable is True for item in self.modal_observations)
        )

    @property
    def exact_common_metric_gain_product_certified(self) -> bool:
        """Whether exact segment gain bounds compose in one metric."""

        return bool(
            self._proof_fields_are_intact()
            and self._exact_common_metric_gain_product_certified
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
    reception_observations: tuple[ReceptionStageObservation, ...] = field(
        default=(),
        repr=False,
    )
    solver_accuracy_certified: bool = field(default=False, init=False)
    future_or_repeated_schedule_stability_certified: bool = field(
        default=False,
        init=False,
    )
    scope: str = field(default=_STAGE_SCOPE, init=False)
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __getattribute__(self, name: str) -> Any:
        """Expose historical contract fields without permitting promotion."""

        if name in {
            "solver_accuracy_certified",
            "future_or_repeated_schedule_stability_certified",
        }:
            return False
        if name == "scope":
            return _STAGE_SCOPE
        return object.__getattribute__(self, name)

    def _proof_fields_are_intact(self) -> bool:
        """Whether all executor-owned stage fields retain their sealed values."""

        try:
            expected = _executed_glyph_stage_stamp(self)
            if not proof_stamps_are_identical(
                _raw_proof_stamp_or_none(self),
                expected,
            ):
                return False
            if not _executed_glyph_stage_fields_are_valid(self):
                return False
        except BaseException:
            return False
        return True

    @property
    def represented_affine_gain_bound_at_observed_endpoint_certified(self) -> bool:
        """Publish the represented gain claim only while the stage seal is intact."""

        return bool(
            self._proof_fields_are_intact()
            and self._represented_affine_gain_bound_at_observed_endpoint_certified
        )


_EXECUTED_GLYPH_STAGE_PROOF_VERSION = "executed_glyph_stage_v2"


def _same_structural_value(left: Any, right: Any) -> bool:
    """Compare detached signatures without invoking identifier equality."""

    return proof_stamps_are_identical(
        structural_proof_signature(left),
        structural_proof_signature(right),
    )


def _executed_glyph_stage_stamp(value: Any) -> tuple[Any, ...]:
    """Snapshot every public and private stage fact except the stamp itself."""

    if type(value) is not ExecutedGlyphStage:
        raise TypeError("glyph-stage evidence must have the canonical runtime type")
    return (
        _EXECUTED_GLYPH_STAGE_PROOF_VERSION,
        tuple(
            (
                item.name,
                _sealed_runtime_field_signature(
                    ExecutedGlyphStage,
                    item.name,
                    object.__getattribute__(value, item.name),
                ),
            )
            for item in fields(ExecutedGlyphStage)
            if item.name != "_proof_stamp"
        ),
    )


def _same_binary64_scalar(left: Any, right: Any) -> bool:
    """Compare scalar payloads by their exact represented binary64 bits."""

    try:
        return _binary64_vectors_are_identical(
            (float(left),),
            (float(right),),
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _reception_observation_matches_neighbor_step(
    observation: ReceptionStageObservation,
    index: int,
    step: Any,
) -> bool:
    """Bind one EN observation to its exact all-target certificate row."""

    try:
        local = step.local_certificates[index]
        runtime_neighbors = tuple(local.runtime_neighbors)
        neighbor_indices = tuple(local.runtime_neighbor_indices)
        state_before = tuple(float(item) for item in local.state_before)
        neighbor_values = tuple(
            state_before[neighbor_index]
            for neighbor_index in neighbor_indices
        )
        accepted_after = float(step.runtime_accepted_state_after[index])
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        local.target_index == index
        and _same_structural_value(local.target, observation.node)
        and _same_structural_value(runtime_neighbors, observation.neighbors)
        and _same_structural_value(
            step.runtime_neighbor_sets[index],
            observation.neighbors,
        )
        and _same_binary64_scalar(
            local.unweighted_runtime_neighbor_mean,
            observation.neighbor_epi_mean,
        )
        and _same_binary64_scalar(
            state_before[index],
            observation.target_epi_before,
        )
        and _same_binary64_scalar(
            local.runtime_target_value,
            observation.target_epi_after,
        )
        and _same_binary64_scalar(
            accepted_after,
            observation.target_epi_after,
        )
        and _binary64_vectors_are_identical(
            neighbor_values,
            observation.neighbor_epi_values,
        )
        and _binary64_vectors_are_identical(
            neighbor_values,
            observation.neighbor_dominant_values,
        )
        and local.epi_kind_before == observation.target_epi_kind_before
        and local.epi_kind_after == observation.target_epi_kind_after
    )


def _executed_glyph_stage_fields_are_valid(value: Any) -> bool:
    """Validate stage structure and ZHIR decision-to-endpoint correspondence."""

    if type(value) is not ExecutedGlyphStage:
        return False
    event = value.event
    if (
        type(event) is not ExecutedOperatorEvent
        or not event._proof_fields_are_intact()
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
        object.__getattribute__(value, "solver_accuracy_certified"),
        object.__getattribute__(
            value,
            "future_or_repeated_schedule_stability_certified",
        ),
    )
    if any(type(item) is not bool for item in booleans):
        return False
    if (
        object.__getattribute__(value, "solver_accuracy_certified") is not False
        or object.__getattribute__(
            value,
            "future_or_repeated_schedule_stability_certified",
        )
        is not False
        or object.__getattribute__(value, "scope") != _STAGE_SCOPE
    ):
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
    if (
        value.pre_flow_evidence is not None
        and not _executed_flow_wrapper_seal_is_intact(
            value.pre_flow_evidence
        )
    ):
        return False
    if value.post_flow_evidence is not None and type(
        value.post_flow_evidence
    ) is not ExecutedNodalFlowInterval:
        return False
    if (
        value.post_flow_evidence is not None
        and not _executed_flow_wrapper_seal_is_intact(
            value.post_flow_evidence
        )
    ):
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
    if event.glyph is Glyph.ZHIR:
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
    elif observations:
        return False

    reception_observations = value.reception_observations
    if type(reception_observations) is not tuple:
        return False
    if event.glyph is Glyph.EN:
        if len(reception_observations) != event.nodes_processed:
            return False
        left = value.left
        right = value.right
        if left is None or right is None:
            return False
        if (
            len(left.nodes) != event.nodes_processed
            or len(right.nodes) != event.nodes_processed
        ):
            return False

        neighbor_step = None
        if value.certificate is not None or value.certificate_kind is not None:
            from ..physics.network_stage_stability import (
                AllTargetNeighborStageCertificate,
                _validate_bridge_stage_certificate,
            )

            certificate = value.certificate
            if (
                value.certificate_kind != "neighbor"
                or type(certificate) is not AllTargetNeighborStageCertificate
                or certificate.glyph != Glyph.EN.value
            ):
                return False
            try:
                neighbor_step = _validate_bridge_stage_certificate(
                    certificate
                ).step
            except (AttributeError, TypeError, ValueError, OverflowError):
                return False
        for index, observation in enumerate(reception_observations):
            if (
                type(observation) is not ReceptionStageObservation
                or observation.target_index != index
                or observation.glyph is not Glyph.EN
                or not observation._proof_fields_are_intact()
            ):
                return False
            if (
                not _same_structural_value(observation.node, left.nodes[index])
                or not _same_structural_value(
                    observation.node,
                    right.nodes[index],
                )
                or Fraction.from_float(observation.target_epi_before)
                != left.exact_epi[index]
                or Fraction.from_float(observation.target_epi_after)
                != right.exact_epi[index]
                or not _same_binary64_scalar(
                    observation.target_epi_before,
                    left.epi[index],
                )
                or not _same_binary64_scalar(
                    observation.target_epi_after,
                    right.epi[index],
                )
            ):
                return False
            if neighbor_step is not None and not (
                _reception_observation_matches_neighbor_step(
                    observation,
                    index,
                    neighbor_step,
                )
            ):
                return False
    elif reception_observations:
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
        structural_proof_signature(nodes),
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


def _represented_operation_fields(value: Any) -> dict[str, Any]:
    """Read one operation's sealed fields without invoking custom accessors."""

    if type(value) is not RepresentedEPIScheduleOperation:
        raise TypeError("represented operation must have the canonical type")
    return {
        name: object.__getattribute__(value, name)
        for name in (
            "position",
            "operation_kind",
            "operation_index",
            "operator_name",
            "nodes",
            "exact_epi_before",
            "exact_epi_after",
            "exact_metric_ray_before",
            "exact_metric_ray_after",
            "exact_energy_gain_upper_bound",
            "ineligibility_reasons",
        )
    }


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
        try:
            fields = _represented_operation_fields(self)
            expected = _represented_operation_stamp(**fields)
        except BaseException as exc:
            raise ValueError(
                "represented operation proof fields are inconsistent"
            ) from exc
        if not proof_stamps_are_identical(
            _raw_proof_stamp_or_none(self),
            expected,
        ):
            raise ValueError("represented operation proof fields are inconsistent")
        _validate_represented_operation_fields(**fields)

    @property
    def represented_affine_gain_certified(self) -> bool:
        """Whether this operation retains complete represented-map evidence."""

        return bool(
            self._proof_fields_are_intact()
            and not self.ineligibility_reasons
        )

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except BaseException:
            return False
        return True


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
    if type(operations) is not tuple:
        raise TypeError("composition operations must be a tuple")
    operation_stamps: list[tuple[Any, ...]] = []
    for operation in tuple.__iter__(operations):
        if type(operation) is not RepresentedEPIScheduleOperation:
            raise TypeError("composition operation must have the canonical type")
        operation_stamps.append(
            _raw_proof_stamp_or_none(operation)
        )
    return (
        _REPRESENTED_COMPOSITION_PROOF_VERSION,
        structural_proof_signature(nodes),
        positive_flow_interval_indices,
        event_indices,
        tuple(operation_stamps),
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


def _represented_composition_fields(value: Any) -> dict[str, Any]:
    """Read one composition's sealed fields without custom accessors."""

    if type(value) is not ObservedRepresentedEPIScheduleComposition:
        raise TypeError("represented composition must have the canonical type")
    return {
        name: object.__getattribute__(value, name)
        for name in (
            "nodes",
            "positive_flow_interval_indices",
            "event_indices",
            "operations",
            "exact_normalized_metric",
            "exact_operation_energy_gain_factors",
            "exact_energy_gain_upper_bound",
            "conditions",
        )
    }


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

    def __getattribute__(self, name: str) -> Any:
        """Expose the canonical historical scope while retaining raw sealing."""

        if name == "scope":
            return _REPRESENTED_COMPOSITION_SCOPE
        return object.__getattribute__(self, name)

    @property
    def runtime_schedule_global_gain_certified(self) -> bool:
        """Keep global executable-map gain outside represented evidence."""

        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        """Keep solver accuracy outside represented composition."""

        return False

    @property
    def full_multichannel_stability_certified(self) -> bool:
        """Keep full multichannel stability outside EPI-map evidence."""

        return False

    @property
    def future_or_repeated_schedule_stability_certified(self) -> bool:
        """Keep future and repeated schedules outside this trace."""

        return False

    def __post_init__(self) -> None:
        try:
            fields = _represented_composition_fields(self)
            expected = _represented_composition_stamp(**fields)
        except BaseException as exc:
            raise ValueError(
                "represented composition proof fields are inconsistent"
            ) from exc
        if not proof_stamps_are_identical(
            _raw_proof_stamp_or_none(self),
            expected,
        ):
            raise ValueError("represented composition proof fields are inconsistent")

        if (
            object.__getattribute__(self, "scope")
            != _REPRESENTED_COMPOSITION_SCOPE
        ):
            raise ValueError("represented composition scope is inconsistent")
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

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except BaseException:
            return False
        return True

    @property
    def represented_affine_composition_gain_certified(self) -> bool:
        """Whether every represented operation composes in one metric."""

        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def represented_map_global_disagreement_contraction_certified(self) -> bool:
        """Whether the represented gain product is strictly contracting."""

        return bool(
            self.represented_affine_composition_gain_certified
            and self.exact_energy_gain_upper_bound is not None
            and self.exact_energy_gain_upper_bound < 1
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Return failed represented-composition condition names."""

        if not self._proof_fields_are_intact():
            return ("composition_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)


@dataclass(frozen=True, slots=True)
class OperatorEventExecutionResult:
    """Immutable evidence that one complete finite schedule committed.

    ``pressure_refresh_callback_invocations`` counts every executor-owned
    pressure callback that returned successfully during this schedule.
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
    physical_flow_partition_indices: tuple[int, ...] = ()
    physical_flow_partition_evidence: tuple[
        ExecutedPressureRefreshedFlowPartition, ...
    ] = ()
    physical_pressure_refresh_callback_invocations: int = 0
    stage_pressure_refresh_callback_invocations: int = 0
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
    _proof_stamp: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __getattribute__(self, name: str) -> Any:
        """Preserve the historical dataclass schema with fail-closed reads."""

        if name in {
            "solver_accuracy_certified",
            "adaptive_u2_u4_policy",
            "external_side_effects_rolled_back",
            "future_or_repeated_schedule_stability_certified",
        }:
            return False
        canonical_text = {
            "flow_provenance": _FLOW_PROVENANCE,
            "nodal_flow_inputs": _NODAL_FLOW_INPUTS,
            "flow_scope": _FLOW_SCOPE,
        }
        if name in canonical_text:
            return canonical_text[name]
        if name in {
            "runtime_clock_checked",
            "whole_schedule_graph_state_atomic",
            "operator_jumps_have_zero_duration",
        }:
            try:
                raw = object.__getattribute__(self, name)
                checker = object.__getattribute__(self, "_proof_fields_are_intact")
                return bool(raw is True and checker())
            except BaseException:
                return False
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        """Reject missing, reordered, substituted, or altered stage evidence."""

        self._validate_existing_proof_stamp()

        expected_contracts = {
            "runtime_clock_checked": True,
            "flow_provenance": _FLOW_PROVENANCE,
            "nodal_flow_inputs": _NODAL_FLOW_INPUTS,
            "whole_schedule_graph_state_atomic": True,
            "operator_jumps_have_zero_duration": True,
            "solver_accuracy_certified": False,
            "adaptive_u2_u4_policy": False,
            "external_side_effects_rolled_back": False,
            "future_or_repeated_schedule_stability_certified": False,
            "flow_scope": _FLOW_SCOPE,
        }
        if any(
            object.__getattribute__(self, name) != expected
            for name, expected in expected_contracts.items()
        ):
            raise ValueError("event execution result contract fields are inconsistent")

        if type(self.schedule) is not OperatorEventSchedule:
            raise TypeError("schedule must be an OperatorEventSchedule")
        self.schedule.__post_init__()
        if type(self.target_nodes) is not tuple:
            raise TypeError("target_nodes must be a tuple")
        if type(self.flow_interval_indices) is not tuple:
            raise TypeError("flow_interval_indices must be a tuple")
        expected_flow_indices = tuple(
            interval.index for interval in self.schedule.intervals
        )
        if self.flow_interval_indices != expected_flow_indices:
            raise ValueError("flow interval indices do not match the schedule")
        if type(self.positive_flow_interval_indices) is not tuple:
            raise TypeError("positive_flow_interval_indices must be a tuple")
        expected_positive_indices = tuple(
            interval.index
            for interval in self.schedule.intervals
            if interval.exact_duration > 0
        )
        if self.positive_flow_interval_indices != expected_positive_indices:
            raise ValueError(
                "positive flow interval indices do not match the schedule"
            )
        if (
            type(self.final_time) is not float
            or structural_proof_signature(self.final_time)
            != structural_proof_signature(self.schedule.end_time)
        ):
            raise ValueError("final_time does not match the schedule endpoint")
        if type(self.events) is not tuple:
            raise TypeError("events must be a tuple")
        if len(self.events) != len(self.schedule.events):
            raise ValueError("committed events do not cover the schedule")
        for observed, scheduled in zip(
            self.events,
            self.schedule.events,
            strict=True,
        ):
            if type(observed) is not ExecutedOperatorEvent:
                raise TypeError("events contain a noncanonical record")
            if not observed._proof_fields_are_intact():
                raise ValueError("committed event proof fields are not intact")
            if (
                observed.event_index != scheduled.event_index
                or observed.cycle_index != scheduled.cycle_index
                or observed.word_position != scheduled.word_position
                or observed.operator_name != scheduled.operator_name
                or observed.glyph is not scheduled.glyph
                or structural_proof_signature(observed.event_time)
                != structural_proof_signature(scheduled.event_time)
                or observed.event_offset != scheduled.event_offset
                or observed.exact_event_time != scheduled.exact_event_time
                or observed.stage_schedule != TWO_PHASE_JACOBI
                or observed.nodes_processed != len(self.target_nodes)
            ):
                raise ValueError("committed event does not match the schedule")
        if type(self.flow_certification_requested) is not bool:
            raise TypeError("flow_certification_requested must be a bool")
        if type(self.flow_interval_evidence) is not tuple:
            raise TypeError("flow_interval_evidence must be a tuple")
        if type(self.physical_flow_partition_indices) is not tuple:
            raise TypeError("physical_flow_partition_indices must be a tuple")
        if type(self.physical_flow_partition_evidence) is not tuple:
            raise TypeError("physical_flow_partition_evidence must be a tuple")
        if (
            type(self.pressure_refresh_callback_invocations) is not int
            or self.pressure_refresh_callback_invocations < 0
        ):
            raise ValueError(
                "pressure_refresh_callback_invocations must be nonnegative"
            )
        if (
            type(self.physical_pressure_refresh_callback_invocations) is not int
            or self.physical_pressure_refresh_callback_invocations < 0
        ):
            raise ValueError(
                "physical pressure refresh invocation count must be nonnegative"
            )
        if (
            type(self.stage_pressure_refresh_callback_invocations) is not int
            or self.stage_pressure_refresh_callback_invocations < 0
        ):
            raise ValueError(
                "stage pressure refresh invocation count must be nonnegative"
            )
        for evidence in self.flow_interval_evidence:
            if type(evidence) is not ExecutedNodalFlowInterval:
                raise TypeError(
                    "flow_interval_evidence must contain canonical runtime records"
                )
            if not _executed_flow_wrapper_seal_is_intact(evidence):
                raise ValueError(
                    "flow interval evidence proof fields are not intact"
                )
            interval = evidence.interval
            if (
                type(interval) is not StructuralFlowInterval
                or interval.index < 0
                or interval.index >= len(self.schedule.intervals)
                or not _same_structural_value(
                    interval,
                    self.schedule.intervals[interval.index],
                )
            ):
                raise ValueError("flow interval evidence does not match the schedule")
            certificate = evidence.certificate
            if _has_intact_nodal_flow_certificate(certificate):
                try:
                    nodes_match = _same_structural_value(
                        certificate.nodes,
                        self.target_nodes,
                    )
                except Exception as exc:
                    raise ValueError(
                        "flow interval target support is unreadable"
                    ) from exc
                if not nodes_match:
                    raise ValueError(
                        "flow interval evidence does not match execution targets"
                    )
        if any(
            type(evidence) is not ExecutedPressureRefreshedFlowPartition
            or not evidence.physical_pressure_reevaluated_partition_established
            for evidence in self.physical_flow_partition_evidence
        ):
            raise ValueError("physical partition evidence proof fields are not intact")
        physical_indices = tuple(
            evidence.partition.parent_interval.index
            for evidence in self.physical_flow_partition_evidence
        )
        if self.physical_flow_partition_indices != physical_indices:
            raise ValueError(
                "physical partition indices do not match execution evidence"
            )
        if physical_indices != tuple(sorted(set(physical_indices))):
            raise ValueError(
                "physical partition indices must be sorted and unique"
            )
        for evidence in self.physical_flow_partition_evidence:
            parent = evidence.partition.parent_interval
            if (
                parent.index < 0
                or parent.index >= len(self.schedule.intervals)
                or not _same_structural_value(
                    parent,
                    self.schedule.intervals[parent.index],
                )
            ):
                raise ValueError(
                    "physical partition evidence does not match the schedule"
                )
            try:
                physical_targets_match = _same_structural_value(
                    evidence.boundary_observations[0].after.nodes,
                    self.target_nodes,
                )
            except Exception as exc:
                raise ValueError(
                    "physical partition target support is unreadable"
                ) from exc
            if not physical_targets_match:
                raise ValueError(
                    "physical partition evidence does not match execution targets"
                )
        observed_physical_invocations = sum(
            evidence.pressure_refresh_callback_invocations
            for evidence in self.physical_flow_partition_evidence
        )
        if (
            self.physical_pressure_refresh_callback_invocations
            != observed_physical_invocations
        ):
            raise ValueError(
                "physical pressure refresh count does not match partition evidence"
            )
        if self.pressure_refresh_callback_invocations != (
            self.physical_pressure_refresh_callback_invocations
            + self.stage_pressure_refresh_callback_invocations
        ):
            raise ValueError(
                "total pressure refresh count must equal physical plus stage "
                "refreshes"
            )
        regular_indices = tuple(
            evidence.interval.index for evidence in self.flow_interval_evidence
        )
        if len(regular_indices) != len(set(regular_indices)):
            raise ValueError("flow interval evidence contains duplicate indices")
        if regular_indices != tuple(sorted(regular_indices)):
            raise ValueError("flow interval evidence is not in schedule order")
        if set(regular_indices).intersection(physical_indices):
            raise ValueError("one interval cannot have two execution paths")
        if self.flow_certification_requested:
            expected_regular_indices = tuple(
                index
                for index in self.positive_flow_interval_indices
                if index not in physical_indices
            )
            if regular_indices != expected_regular_indices:
                raise ValueError(
                    "flow evidence must cover every positive scheduled interval"
                )
        elif regular_indices or physical_indices:
            raise ValueError("disabled flow certification cannot carry flow evidence")

        if type(self.stage_certification_requested) is not bool:
            raise TypeError("stage_certification_requested must be a bool")
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
            if event.glyph is Glyph.EN:
                observation_nodes = tuple(
                    observation.node
                    for observation in stage.reception_observations
                )
                try:
                    reception_targets_match = _same_structural_value(
                        observation_nodes,
                        self.target_nodes,
                    )
                except Exception as exc:
                    raise ValueError(
                        "Reception stage target support is unreadable"
                    ) from exc
                if not reception_targets_match:
                    raise ValueError(
                        "Reception stage observations do not match execution targets"
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
        expected_composition = _compose_observed_represented_epi_schedule(
            self.schedule,
            self.target_nodes,
            self.flow_interval_evidence,
            self.glyph_stage_evidence,
            self.physical_flow_partition_evidence,
        )
        if not proof_stamps_are_identical(
            _raw_proof_stamp_or_none(composition),
            _raw_proof_stamp_or_none(expected_composition),
        ):
            raise ValueError(
                "represented composition does not match execution evidence"
            )

    def _validate_existing_proof_stamp(self) -> None:
        """Reject replacement of any field on an already sealed result."""

        raw_stamp = _raw_proof_stamp_or_none(self)
        if type(raw_stamp) is tuple and tuple.__len__(raw_stamp) == 0:
            return
        try:
            expected = _sealed_dataclass_stamp(
                self,
                _OPERATOR_EVENT_EXECUTION_RESULT_PROOF_VERSION,
            )
        except BaseException as exc:
            raise ValueError(
                "event execution result proof fields are inconsistent"
            ) from exc
        if not proof_stamps_are_identical(raw_stamp, expected):
            raise ValueError("event execution result proof fields are inconsistent")

    def _proof_fields_are_intact(self) -> bool:
        """Whether the result and every decisive nested field retain their seal."""

        try:
            raw_stamp = _raw_proof_stamp_or_none(self)
            if type(raw_stamp) is not tuple or tuple.__len__(raw_stamp) == 0:
                return False
            self.__post_init__()
        except BaseException:
            return False
        return True

    def _all_positive_flow_intervals(self, attribute: str) -> bool | None:
        if not self._proof_fields_are_intact():
            return False
        if not self.flow_certification_requested:
            return None
        regular = {
            item.interval.index: item for item in self.flow_interval_evidence
        }
        physical = {
            item.partition.parent_interval.index: item
            for item in self.physical_flow_partition_evidence
        }
        if tuple(sorted((*regular, *physical))) != (
            self.positive_flow_interval_indices
        ):
            return False
        physical_attribute = {
            "runtime_bound_binary64_interval_identified": (
                "all_segment_binary64_intervals_identified"
            ),
            "runtime_bound_binary64_held_pressure_interval_identified": (
                "all_segment_binary64_replays_identified"
            ),
            "runtime_bound_exact_affine_map_identified": (
                "all_segment_exact_affine_maps_identified"
            ),
            "runtime_bound_global_disagreement_contraction_certified": (
                "all_segment_disagreement_contractions_certified"
            ),
        }[attribute]
        return all(
            bool(getattr(regular[index], attribute))
            if index in regular
            else bool(getattr(physical[index], physical_attribute))
            for index in self.positive_flow_interval_indices
        )

    @property
    def physical_pressure_reevaluated_partitions_established(self) -> bool | None:
        """Aggregate intact physical partition evidence, or ``None`` if absent."""

        if not self._proof_fields_are_intact():
            return False
        if not self.physical_flow_partition_evidence:
            return None
        return bool(
            self.physical_flow_partition_indices
            and all(
                item.physical_pressure_reevaluated_partition_established
                for item in self.physical_flow_partition_evidence
            )
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

        if not self._proof_fields_are_intact():
            return False
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

    entries = _selected_mapping_entries(
        _runtime_graph_mapping(graph),
        ("_t",),
    )
    if not entries:
        raise TNFRValueError(
            "Operator-event execution requires an existing binary64 runtime clock.",
            context={"boundary": boundary, "reason": "missing_graph_time"},
        )
    current = entries[0][1]
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

    entries = _selected_mapping_entries(
        _runtime_graph_mapping(graph),
        (_HYBRID_EVENT_LOG,),
    )
    if entries and not isinstance(entries[0][1], list):
        raise TNFRValueError(
            "hybrid_event_log must be a list when already configured.",
            context={
                "history_channel": _HYBRID_EVENT_LOG,
                "value_type": type(entries[0][1]).__name__,
            },
        )


def _materialize_physical_flow_partitions(
    schedule: OperatorEventSchedule,
    values: Iterable[PhysicalFlowPartition],
) -> tuple[PhysicalFlowPartition, ...]:
    """Validate and index explicit physical partitions before graph mutation."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("physical_flow_partitions must be an iterable of partitions")
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "physical_flow_partitions must be an iterable of partitions"
        ) from exc
    indexed: dict[int, PhysicalFlowPartition] = {}
    for position, partition in enumerate(materialized):
        if type(partition) is not PhysicalFlowPartition:
            raise TypeError(
                "physical_flow_partitions must contain PhysicalFlowPartition "
                f"records; item {position} has type "
                f"{type(partition).__qualname__}"
            )
        partition.__post_init__()
        index = partition.parent_interval.index
        if index < 0 or index >= len(schedule.intervals):
            raise ValueError(
                "physical partition parent interval index is outside the schedule"
            )
        if partition.parent_interval != schedule.intervals[index]:
            raise ValueError(
                "physical partition parent does not match the scheduled interval"
            )
        if partition.parent_interval.exact_duration <= 0:
            raise ValueError("physical partitions require positive parent intervals")
        if index in indexed:
            raise ValueError(
                "physical_flow_partitions contain duplicate parent intervals"
            )
        indexed[index] = partition
    return tuple(indexed[index] for index in sorted(indexed))


def _prepare_word(
    schedule: OperatorEventSchedule,
    context: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], Any | None]:
    """Validate one canonical word from an already detached context."""

    if not schedule.operator_names:
        return (), None

    from ..validation import validate_sequence
    from .grammar_execution import ValidatedSequence
    from .registry import get_operator_class

    operators = tuple(
        get_operator_class(name)() for name in schedule.operator_names
    )
    outcome = validate_sequence(
        list(schedule.operator_names),
        context=context,
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
        context=context,
    )


def _resolve_schedule_integrator_instance(
    graph: nx.Graph,
    *,
    cache_result: bool = True,
) -> Any:
    """Resolve the configured integrator without virtual graph access.

    ``cache_result=False`` separates potentially user-defined construction from
    the executor-owned cache write.  Event schedules use that mode so they can
    prove that a class or factory did not mutate graph-owned state before the
    cache becomes an authorised runtime change.
    """

    import inspect

    from ..dynamics import integrators as integrator_module
    from ..dynamics.runtime import _call_integrator_factory

    graph_mapping = _runtime_graph_mapping(graph)
    cache_entries = _selected_mapping_entries(
        graph_mapping,
        ("_integrator_cache",),
    )
    candidate_entries = _selected_mapping_entries(
        graph_mapping,
        ("integrator",),
    )
    cache_entry = cache_entries[0][1] if cache_entries else None
    candidate = candidate_entries[0][1] if candidate_entries else None
    if (
        type(cache_entry) is tuple
        and len(cache_entry) == 2
        and cache_entry[0] is candidate
        and isinstance(cache_entry[1], integrator_module.AbstractIntegrator)
    ):
        return cache_entry[1]

    if isinstance(candidate, integrator_module.AbstractIntegrator):
        instance = candidate
    elif inspect.isclass(candidate) and issubclass(
        candidate,
        integrator_module.AbstractIntegrator,
    ):
        instance = candidate()
    elif callable(candidate):
        instance = _call_integrator_factory(candidate, graph)
    elif candidate is None:
        instance = integrator_module.DefaultIntegrator()
    else:
        raise TypeError(
            "Graph integrator must be an AbstractIntegrator, subclass or callable"
        )

    if not isinstance(instance, integrator_module.AbstractIntegrator):
        raise TypeError(
            "Configured integrator must implement AbstractIntegrator.integrate"
        )
    if cache_result:
        _set_runtime_mapping_item(
            graph_mapping,
            "_integrator_cache",
            (candidate, instance),
        )
    return instance


def _cache_schedule_integrator_instance(
    graph: nx.Graph,
    integrator: Any,
) -> None:
    """Commit only the canonical cache binding after sealed resolution."""

    graph_mapping = _runtime_graph_mapping(graph)
    candidate_entries = _selected_mapping_entries(
        graph_mapping,
        ("integrator",),
    )
    candidate = candidate_entries[0][1] if candidate_entries else None
    cache_entries = _selected_mapping_entries(
        graph_mapping,
        ("_integrator_cache",),
    )
    cache_entry = cache_entries[0][1] if cache_entries else None
    if (
        type(cache_entry) is tuple
        and len(cache_entry) == 2
        and cache_entry[0] is candidate
        and cache_entry[1] is integrator
    ):
        return
    _set_runtime_mapping_item(
        graph_mapping,
        "_integrator_cache",
        (candidate, integrator),
    )


def _resolve_integrator_method(integrator: Any) -> Any:
    """Bind ``integrate`` without invoking an instance attribute override."""

    namespace = _runtime_instance_namespace(integrator)
    if namespace is not None and "integrate" in namespace:
        method = dict.__getitem__(namespace, "integrate")
        if not callable(method):
            raise TNFRValueError("Configured integrator integrate is not callable.")
        return method

    for owner in _runtime_class_mro(type(integrator)):
        descriptor = _runtime_class_namespace(owner).get("integrate")
        if descriptor is None:
            continue
        if type(descriptor) is not FunctionType:
            raise TNFRValueError(
                "Configured integrator integrate descriptor cannot be bound "
                "without invoking user code."
            )
        return MethodType(descriptor, integrator)
    raise TNFRValueError("Configured integrator has no integrate method.")


def _flow_metadata_graph(graph: nx.Graph) -> nx.Graph:
    """Copy graph configuration to an inert graph for metadata resolution."""

    metadata_graph = nx.Graph()
    metadata_mapping = _runtime_graph_mapping(metadata_graph)
    for key, value in _runtime_mapping_items(_runtime_graph_mapping(graph)):
        if type(key) is str:
            _set_runtime_mapping_item(metadata_mapping, key, value)
    return metadata_graph


def _flow_runtime_metadata(
    graph: nx.Graph,
    interval: StructuralFlowInterval,
    integrator: Any,
    integrate_method: Any,
    *,
    method: str | None,
) -> _FlowRuntimeMetadata:
    """Derive execution metadata without invoking the integrator twice."""

    from ..gamma import _get_gamma_spec

    integrator_name = _safe_type_text(type(integrator), _TYPE_QUALNAME_DESCRIPTOR)
    instance_attributes = _runtime_instance_namespace(integrator) or {}
    bound_function = (
        _read_builtin_descriptor(_METHOD_FUNCTION_DESCRIPTOR, integrate_method)
        if type(integrate_method) is MethodType
        else None
    )
    bound_receiver = (
        _read_builtin_descriptor(_METHOD_RECEIVER_DESCRIPTOR, integrate_method)
        if type(integrate_method) is MethodType
        else None
    )
    provenance = bool(
        type(integrator) is _CanonicalDefaultIntegrator
        and "integrate" not in instance_attributes
        and bound_receiver is integrator
        and bound_function is _CANONICAL_DEFAULT_INTEGRATE
        and _CanonicalDefaultIntegrator.integrate
        is _CANONICAL_DEFAULT_INTEGRATE
    )
    graph_mapping = _runtime_graph_mapping(graph)
    extended_entries = _selected_mapping_entries(
        graph_mapping,
        ("use_extended_dynamics",),
    )
    extended_requested = bool(extended_entries[0][1]) if extended_entries else False
    if not provenance:
        return _FlowRuntimeMetadata(
            integrator_name=integrator_name,
            integrator_provenance_certified=False,
            resolved_method=None,
            resolved_substeps=None,
            gamma_is_none=None,
            extended_dynamics_requested=extended_requested,
        )

    metadata_graph = _flow_metadata_graph(graph)
    _, substeps, _, resolved_method = _canonical_prepare_integration_params(
        metadata_graph,
        interval.duration,
        interval.start_time,
        method,
    )
    gamma_spec = _get_gamma_spec(metadata_graph)
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

    def capture() -> tuple[NodalFlowStateSnapshot | None, bool]:
        try:
            capture_graph, _nodes = _nodal_flow_capture_graph(graph)
            return capture_nodal_flow_state(capture_graph), True
        except (TypeError, ValueError, nx.NetworkXException):
            return None, False

    return _run_readonly_graph_observation(
        graph,
        capture,
        label="nodal flow endpoint capture",
    )


def _nodal_flow_capture_graph(
    graph: nx.Graph,
) -> tuple[nx.Graph, tuple[Any, ...]]:
    """Build an inert scalar-flow view from stored NetworkX mappings."""

    layout = _networkx_runtime_layout(graph)
    if layout.multigraph and layout.directed:
        capture_graph: nx.Graph = nx.MultiDiGraph()
        add_node = nx.MultiDiGraph.add_node
        add_edge = nx.MultiDiGraph.add_edge
    elif layout.multigraph:
        capture_graph = nx.MultiGraph()
        add_node = nx.MultiGraph.add_node
        add_edge = nx.MultiGraph.add_edge
    elif layout.directed:
        capture_graph = nx.DiGraph()
        add_node = nx.DiGraph.add_node
        add_edge = nx.DiGraph.add_edge
    else:
        capture_graph = nx.Graph()
        add_node = nx.Graph.add_node
        add_edge = nx.Graph.add_edge

    capture_node_keys = frozenset((*ALIAS_EPI, *ALIAS_VF, *ALIAS_DNFR))
    nodes = tuple(node for node, _data in layout.node_data)
    for node, data in layout.node_data:
        attributes = {
            key: value
            for key, value in _runtime_mapping_items(data)
            if type(key) is str and key in capture_node_keys
        }
        add_node(capture_graph, node, **attributes)

    for edge in layout.edges:
        if layout.multigraph:
            source, target, key, data = edge
        else:
            source, target, data = edge
            key = None
        weight_entries = _selected_mapping_entries(data, ("weight",))
        attributes = (
            {"weight": weight_entries[0][1]} if weight_entries else {}
        )
        if layout.multigraph:
            add_edge(
                capture_graph,
                source,
                target,
                key=key,
                **attributes,
            )
        else:
            add_edge(capture_graph, source, target, **attributes)
    return capture_graph, nodes


def _require_pressure_boundary_snapshot(
    graph: nx.Graph,
    *,
    parent_interval_index: int,
    boundary_index: int,
    side: str,
) -> NodalFlowStateSnapshot:
    """Capture the scalar nodal state required by a physical boundary."""

    from ..physics.runtime_flow_stability import capture_nodal_flow_state

    def capture() -> NodalFlowStateSnapshot:
        try:
            capture_graph, _nodes = _nodal_flow_capture_graph(graph)
            return capture_nodal_flow_state(capture_graph)
        except (TypeError, ValueError, nx.NetworkXException) as exc:
            raise TNFRValueError(
                "Physical pressure refresh requires a capturable scalar "
                "nodal state.",
                context={
                    "parent_interval_index": parent_interval_index,
                    "boundary_index": boundary_index,
                    "side": side,
                    "reason": str(exc),
                },
            ) from exc

    return _run_readonly_graph_observation(
        graph,
        capture,
        label="physical pressure-boundary capture",
    )


def _selected_mapping_entries(
    mapping: MutableMapping[Any, Any],
    keys: tuple[str, ...],
) -> tuple[tuple[str, Any], ...]:
    """Read selected exact-string keys without virtual mapping dispatch."""

    items = _runtime_mapping_items(mapping)
    return tuple(
        (selected, value)
        for selected in keys
        for key, value in items
        if type(key) is str and key == selected
    )


def _runtime_graph_mapping(graph: nx.Graph) -> MutableMapping[Any, Any]:
    """Read ``Graph.graph`` storage without invoking graph attribute hooks."""

    mapping = _runtime_stored_attribute(graph, "graph")
    if not isinstance(mapping, MutableMapping):
        raise TNFRValueError("graph attribute storage is not a mutable mapping")
    _runtime_mapping_items(mapping)
    return mapping


def _graph_node_data_items(
    graph: nx.Graph,
) -> tuple[tuple[Any, MutableMapping[Any, Any]], ...]:
    """Read node attribute mappings through the safe NetworkX layout."""

    return _networkx_runtime_layout(graph).node_data


def _node_alias_state_signature(
    graph: nx.Graph,
    aliases: tuple[str, ...],
    *,
    node_data: tuple[tuple[Any, MutableMapping[Any, Any]], ...] | None = None,
) -> tuple[Any, ...]:
    """Capture every present spelling in one canonical nodal alias group."""

    return structural_proof_signature(
        tuple(
            (
                node,
                _selected_mapping_entries(data, aliases),
            )
            for node, data in (
                _graph_node_data_items(graph) if node_data is None else node_data
            )
        )
    )


def _node_history_signature(
    graph: nx.Graph,
    keys: tuple[str, ...],
    *,
    node_data: tuple[tuple[Any, MutableMapping[Any, Any]], ...] | None = None,
) -> tuple[Any, ...]:
    """Capture every present selected history channel in the live graph."""

    return structural_proof_signature(
        tuple(
            (
                node,
                selected,
            )
            for node, data in (
                _graph_node_data_items(graph) if node_data is None else node_data
            )
            if (selected := _selected_mapping_entries(data, keys))
        )
    )


def _nodal_state_excluding_keys_signature(
    graph: nx.Graph,
    excluded_keys: frozenset[str],
    *,
    node_data: tuple[tuple[Any, MutableMapping[Any, Any]], ...] | None = None,
) -> tuple[Any, ...]:
    """Capture all nodal data except explicitly executor-owned channels."""

    return structural_proof_signature(
        tuple(
            (
                node,
                structural_object_state_signature(data),
                tuple(
                    (key, value)
                    for key, value in _runtime_mapping_items(data)
                    if not (type(key) is str and key in excluded_keys)
                ),
            )
            for node, data in (
                _graph_node_data_items(graph) if node_data is None else node_data
            )
        )
    )


def _other_nodal_state_signature(
    graph: nx.Graph,
    *,
    node_data: tuple[tuple[Any, MutableMapping[Any, Any]], ...] | None = None,
) -> tuple[Any, ...]:
    """Capture all nodal data outside the pressure alias group."""

    return _nodal_state_excluding_keys_signature(
        graph,
        frozenset(ALIAS_DNFR),
        node_data=node_data,
    )


_PRESSURE_REFRESH_DERIVED_GRAPH_KEYS = frozenset(
    {
        "_DNFR_META",
        "_dnfr_hook_name",
        "_dnfr_prep_dirty",
        "_dnfrmax",
        "_dnfrmax_node",
        "_sel_norms",
    }
)

_INTEGRATOR_MUTABLE_NODE_KEYS = frozenset(
    (*ALIAS_EPI, *ALIAS_DEPI, *ALIAS_D2EPI)
)
_INTEGRATOR_RUNTIME_GRAPH_KEYS = frozenset(
    {
        "_t",
        "integrator",
        "_integrator_cache",
        # ``DefaultIntegrator`` may normalize these rebuildable Γ caches while
        # reading the persistent ``GAMMA`` configuration.  The configuration
        # itself remains protected by the flow contract.
        "_gamma_raw",
        "_gamma_spec",
        "_gamma_spec_hash",
    }
)


def _filtered_graph_configuration_signature(
    graph: nx.Graph,
    *,
    excluded_keys: frozenset[str],
    opaque_references: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    """Capture graph configuration while exempting declared runtime channels."""

    graph_mapping = _runtime_graph_mapping(graph)
    graph_data = tuple(
        (key, value)
        for key, value in _runtime_mapping_items(graph_mapping)
        if not (type(key) is str and key in excluded_keys)
    )
    graph_mapping_state = structural_object_state_signature(graph_mapping)
    graph_namespace = _runtime_instance_namespace(graph)
    if graph_namespace is None:
        raise TNFRValueError("graph instance has no readable namespace")
    instance_data = tuple(
        (key, value)
        for key, value in graph_namespace.items()
        if key not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        or key == "_last_operator_applied"
    )
    slot_data = tuple(
        item
        for item in _runtime_owner_qualified_slot_state(graph)
        if item[0][4] not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        or item[0][4] == "_last_operator_applied"
    )
    return structural_proof_signature(
        (
            type(graph),
            id(graph_namespace),
            graph_data,
            graph_mapping_state,
            instance_data,
            slot_data,
            _graph_factory_state_signature(graph),
            _networkx_internal_mapping_state_signature(graph),
        ),
        opaque_references=opaque_references,
    )


def _graph_configuration_signature(graph: nx.Graph) -> tuple[Any, ...]:
    """Capture persistent graph and instance configuration outside caches."""

    from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS

    excluded = (
        GRAPH_RUNTIME_CACHE_KEYS
        | _PRESSURE_REFRESH_DERIVED_GRAPH_KEYS
        | {"compute_delta_nfr", "_dnfr_weights"}
    )
    return _filtered_graph_configuration_signature(
        graph,
        excluded_keys=frozenset(excluded),
    )


def _dnfr_weights_signature(
    graph: nx.Graph,
) -> tuple[bool, bool, tuple[Any, ...]]:
    """Capture cached pressure weights separately from graph configuration."""

    entries = _selected_mapping_entries(
        _runtime_graph_mapping(graph),
        ("_dnfr_weights",),
    )
    present = bool(entries)
    value = entries[0][1] if entries else None
    return present, value is None, structural_proof_signature(value)


def _edge_state_signature(
    graph: nx.Graph,
    *,
    layout: Any | None = None,
) -> tuple[Any, ...]:
    """Capture topology and every edge attribute without invoking equality."""

    runtime_layout = (
        _networkx_runtime_layout(graph) if layout is None else layout
    )
    if runtime_layout.multigraph:
        edges = tuple(
            (
                source,
                target,
                key,
                structural_object_state_signature(data),
                _runtime_mapping_items(data),
            )
            for source, target, key, data in runtime_layout.edges
        )
    else:
        edges = tuple(
            (
                source,
                target,
                structural_object_state_signature(data),
                _runtime_mapping_items(data),
            )
            for source, target, data in runtime_layout.edges
        )
    return structural_proof_signature(
        (runtime_layout.directed, runtime_layout.multigraph, edges)
    )


def _runtime_clock_signature(graph: nx.Graph) -> tuple[Any, ...]:
    entries = _selected_mapping_entries(
        _runtime_graph_mapping(graph),
        ("_t",),
    )
    return structural_proof_signature(
        (bool(entries), entries[0][1] if entries else None)
    )


def _retained_identity_signature(
    value: Any,
    *,
    retained_references: list[Any],
    opaque_references: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Sign one live value by structure and identity while retaining it."""

    return structural_proof_signature(
        value,
        opaque_references=opaque_references,
        identity_sensitive=True,
        _retained_references=retained_references,
    )


def _retained_object_state_signature(
    value: Any,
    *,
    retained_references: list[Any],
    opaque_references: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Sign one container/object identity and its directly owned state."""

    return structural_object_state_signature(
        value,
        opaque_references=opaque_references,
        identity_sensitive=True,
        _retained_references=retained_references,
    )


def _remove_transient_networkx_cached_views(
    graph: nx.Graph,
    transient_views: Iterable[tuple[str, Any]],
) -> None:
    """Remove only cached views materialized by a runtime preservation guard."""

    namespace = _runtime_instance_namespace(graph)
    if namespace is None:
        raise TNFRValueError("graph instance has no readable namespace")
    for name, value in transient_views:
        if name in namespace and namespace[name] is value:
            dict.__delitem__(namespace, name)


def _networkx_cached_view_items(
    graph: nx.Graph,
    *,
    transient_views: list[tuple[str, Any]] | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Materialize trusted NetworkX views and record guard-created caches."""

    graph_bases = {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}
    descriptors: dict[str, Any] = {}
    for owner in _runtime_class_mro(type(graph)):
        if owner not in graph_bases:
            continue
        for name, descriptor in _runtime_class_namespace(owner).items():
            if type(descriptor) is cached_property:
                descriptors.setdefault(name, descriptor)
    namespace = _runtime_instance_namespace(graph)
    if namespace is None:
        raise TNFRValueError("graph instance has no readable namespace")
    initially_present = frozenset(namespace)
    items: list[tuple[str, Any]] = []
    created: dict[str, Any] = {}
    try:
        for name, descriptor in descriptors.items():
            value = type(descriptor).__get__(descriptor, graph, type(graph))
            items.append((name, value))
            for candidate in descriptors:
                if (
                    candidate not in initially_present
                    and candidate in namespace
                    and candidate not in created
                ):
                    created[candidate] = namespace[candidate]
    except BaseException as exc:
        _remove_transient_networkx_cached_views(graph, created.items())
        raise TNFRValueError(
            "NetworkX cached view could not be materialized safely",
            context={"view": name, "reason_type": type(exc).__qualname__},
        ) from exc
    if transient_views is not None:
        transient_views.extend(created.items())
    return tuple(items)


def _protected_graph_identity_signature(
    graph: nx.Graph,
    *,
    excluded_node_keys: frozenset[str],
    excluded_graph_keys: frozenset[str],
    retained_references: list[Any],
    opaque_references: tuple[Any, ...],
    layout: Any | None = None,
    transient_networkx_cached_views: list[tuple[str, Any]] | None = None,
) -> tuple[Any, ...]:
    """Seal protected graph values and their cross-domain alias topology."""

    runtime_layout = (
        _networkx_runtime_layout(graph) if layout is None else layout
    )
    cached_view_items = _networkx_cached_view_items(
        graph,
        transient_views=transient_networkx_cached_views,
    )

    def signature(value: Any) -> tuple[Any, ...]:
        return _retained_identity_signature(
            value,
            retained_references=retained_references,
            opaque_references=opaque_references,
        )

    def object_state(value: Any) -> tuple[Any, ...]:
        return _retained_object_state_signature(
            value,
            retained_references=retained_references,
            opaque_references=opaque_references,
        )

    node_records = tuple(
        (
            signature(node),
            object_state(data),
            tuple(
                (signature(key), signature(value))
                for key, value in _runtime_mapping_items(data)
                if not (
                    type(key) is str and key in excluded_node_keys
                )
            ),
        )
        for node, data in runtime_layout.node_data
    )
    if runtime_layout.multigraph:
        edge_records = tuple(
            (
                signature(source),
                signature(target),
                signature(key),
                object_state(data),
                tuple(
                    (signature(name), signature(value))
                    for name, value in _runtime_mapping_items(data)
                ),
            )
            for source, target, key, data in runtime_layout.edges
        )
    else:
        edge_records = tuple(
            (
                signature(source),
                signature(target),
                object_state(data),
                tuple(
                    (signature(name), signature(value))
                    for name, value in _runtime_mapping_items(data)
                ),
            )
            for source, target, data in runtime_layout.edges
        )

    graph_mapping = runtime_layout.graph_mapping
    graph_records = tuple(
        (signature(key), signature(value))
        for key, value in _runtime_mapping_items(graph_mapping)
        if not (type(key) is str and key in excluded_graph_keys)
    )
    namespace = _runtime_instance_namespace(graph)
    if namespace is None:
        raise TNFRValueError("graph instance has no readable namespace")
    instance_records = tuple(
        (signature(key), signature(value))
        for key, value in namespace.items()
        if key not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        or key == "_last_operator_applied"
    )
    slot_records = tuple(
        (
            structural_proof_signature(label),
            present,
            signature(value) if present else None,
        )
        for label, present, value in _runtime_owner_qualified_slot_state(graph)
        if label[4] not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        or label[4] == "_last_operator_applied"
    )
    factory_records = tuple(
        (name, signature(value))
        for name, value in _graph_factory_items(graph)
    )
    internal_mapping_records = tuple(
        (structural_proof_signature(label), object_state(value))
        for label, value in _networkx_internal_mapping_items(
            graph,
            layout=runtime_layout,
        )
    )
    adjacency_binding_records = tuple(
        (
            signature(node),
            object_state(neighbors),
            tuple(
                (signature(neighbor), signature(edge_storage))
                for neighbor, edge_storage in _runtime_mapping_items(neighbors)
            ),
        )
        for node, neighbors in runtime_layout.adjacency_inner
    )
    predecessor_binding_records = tuple(
        (
            signature(node),
            object_state(neighbors),
            tuple(
                (signature(neighbor), signature(edge_storage))
                for neighbor, edge_storage in _runtime_mapping_items(neighbors)
            ),
        )
        for node, neighbors in runtime_layout.predecessor_inner
    )
    cached_view_opaque_references = (
        *opaque_references,
        graph,
        runtime_layout.graph_mapping,
        runtime_layout.node_outer,
        runtime_layout.adjacency_outer,
        *(
            ()
            if runtime_layout.predecessor_outer is None
            else (runtime_layout.predecessor_outer,)
        ),
        *(data for _node, data in runtime_layout.node_data),
        *(mapping for _node, mapping in runtime_layout.adjacency_inner),
        *(mapping for _node, mapping in runtime_layout.predecessor_inner),
        *(
            mapping
            for _node, _neighbor, mapping in runtime_layout.adjacency_edge_keys
        ),
        *(
            mapping
            for _node, _neighbor, mapping in runtime_layout.predecessor_edge_keys
        ),
        *(edge[-1] for edge in runtime_layout.edges),
    )
    cached_view_records = tuple(
        (
            name,
            _retained_object_state_signature(
                value,
                retained_references=retained_references,
                opaque_references=cached_view_opaque_references,
            ),
        )
        for name, value in cached_view_items
    )
    auxiliary_records = tuple(
        (
            name,
            name in namespace,
            object_state(namespace[name]) if name in namespace else None,
        )
        for name in ("__networkx_cache__", "__networkx_backend__")
    )
    root_mapping_bindings = []
    for name in ("graph", "_node", "_adj", "_succ", "_pred"):
        try:
            value = _runtime_stored_attribute(graph, name)
        except TNFRValueError:
            root_mapping_bindings.append((name, False, None))
        else:
            root_mapping_bindings.append((name, True, object_state(value)))
    return (
        signature(type(graph)),
        id(namespace),
        object_state(graph_mapping),
        graph_records,
        instance_records,
        slot_records,
        factory_records,
        internal_mapping_records,
        adjacency_binding_records,
        predecessor_binding_records,
        tuple(root_mapping_bindings),
        cached_view_records,
        auxiliary_records,
        node_records,
        edge_records,
    )


def _integrator_binding_signature(
    graph: nx.Graph,
    integrator: Any,
) -> tuple[Any, ...]:
    """Capture configured integrator bindings without freezing its own state."""

    graph_mapping = _runtime_graph_mapping(graph)
    entries = _selected_mapping_entries(
        graph_mapping,
        ("integrator", "_integrator_cache"),
    )
    return tuple(
        (
            key,
            id(value),
            (
                tuple(id(item) for item in value)
                if key == "_integrator_cache" and type(value) is tuple
                else None
            ),
        )
        for key, value in entries
    )


def _run_readonly_graph_observation(
    graph: nx.Graph,
    operation: Any,
    *,
    label: str,
) -> Any:
    """Run one diagnostic read and reject any graph-owned side effect."""

    retained: list[Any] = []
    transient_views: list[tuple[str, Any]] = []
    layout = _networkx_runtime_layout(graph)
    before = _protected_graph_identity_signature(
        graph,
        excluded_node_keys=frozenset(),
        excluded_graph_keys=frozenset(),
        retained_references=retained,
        opaque_references=(graph,),
        layout=layout,
        transient_networkx_cached_views=transient_views,
    )
    try:
        result = operation()
        after = _protected_graph_identity_signature(
            graph,
            excluded_node_keys=frozenset(),
            excluded_graph_keys=frozenset(),
            retained_references=[],
            opaque_references=(graph,),
        )
        if before != after:
            raise TNFRValueError(
                f"Event-schedule {label} changed graph state."
            )
        return result
    finally:
        _remove_transient_networkx_cached_views(graph, transient_views)


def _canonical_graph_storage_references(
    graph: nx.Graph,
    layout: Any,
) -> tuple[Any, ...]:
    """Return structural storage aliases already signed by their own channel."""

    return (
        graph,
        layout.graph_mapping,
        layout.node_outer,
        layout.adjacency_outer,
        *(
            ()
            if layout.predecessor_outer is None
            else (layout.predecessor_outer,)
        ),
        *(data for _node, data in layout.node_data),
        *(mapping for _node, mapping in layout.adjacency_inner),
        *(mapping for _node, mapping in layout.predecessor_inner),
        *(
            mapping
            for _node, _neighbor, mapping in layout.adjacency_edge_keys
        ),
        *(
            mapping
            for _node, _neighbor, mapping in layout.predecessor_edge_keys
        ),
        *(edge[-1] for edge in layout.edges),
    )


def _record_guarded_mutation_flow_boundary(graph: nx.Graph) -> None:
    """Record the executor-owned history channel under a complete state guard.

    History materialisation can inspect graph-owned containers supplied by a
    caller.  The central recorder is allowed to replace only each node's
    ``epi_time_history`` value; every other graph binding and reachable mutable
    value must remain observationally identical.
    """

    from ..dynamics.runtime import _record_mutation_flow_boundary

    retained: list[Any] = []
    transient_views: list[tuple[str, Any]] = []
    layout = _networkx_runtime_layout(graph)
    opaque_references = _canonical_graph_storage_references(graph, layout)
    before = _protected_graph_identity_signature(
        graph,
        excluded_node_keys=frozenset({"epi_time_history"}),
        excluded_graph_keys=frozenset(),
        retained_references=retained,
        opaque_references=opaque_references,
        layout=layout,
        transient_networkx_cached_views=transient_views,
    )
    try:
        _record_mutation_flow_boundary(graph)
        after = _protected_graph_identity_signature(
            graph,
            excluded_node_keys=frozenset({"epi_time_history"}),
            excluded_graph_keys=frozenset(),
            retained_references=[],
            opaque_references=opaque_references,
        )
        if before != after:
            raise TNFRValueError(
                "Mutation boundary recording changed graph state outside "
                "epi_time_history."
            )
    finally:
        _remove_transient_networkx_cached_views(
            graph,
            tuple(
                (name, value)
                for name, value in transient_views
                if name != "nodes"
            ),
        )


def _append_guarded_hybrid_event_record(
    graph: nx.Graph,
    record: dict[str, Any],
) -> None:
    """Append one event record without mutating an aliased graph channel."""

    retained: list[Any] = []
    transient_views: list[tuple[str, Any]] = []
    layout = _networkx_runtime_layout(graph)
    opaque_references = _canonical_graph_storage_references(graph, layout)
    before = _protected_graph_identity_signature(
        graph,
        excluded_node_keys=frozenset(),
        excluded_graph_keys=frozenset({_HYBRID_EVENT_LOG}),
        retained_references=retained,
        opaque_references=opaque_references,
        layout=layout,
        transient_networkx_cached_views=transient_views,
    )
    try:
        graph_mapping = layout.graph_mapping
        sink_entries = _selected_mapping_entries(
            graph_mapping,
            (_HYBRID_EVENT_LOG,),
        )
        if sink_entries:
            sink = sink_entries[0][1]
        else:
            sink = []
            _set_runtime_mapping_item(
                graph_mapping,
                _HYBRID_EVENT_LOG,
                sink,
            )
        if not isinstance(sink, list):
            raise TNFRValueError(
                "hybrid_event_log changed type during event execution.",
                context={"value_type": type(sink).__name__},
            )
        list.append(sink, record)
        after = _protected_graph_identity_signature(
            graph,
            excluded_node_keys=frozenset(),
            excluded_graph_keys=frozenset({_HYBRID_EVENT_LOG}),
            retained_references=[],
            opaque_references=opaque_references,
        )
        if before != after:
            raise TNFRValueError(
                "hybrid_event_log must not alias another graph-owned channel."
            )
    finally:
        _remove_transient_networkx_cached_views(graph, transient_views)


def _capture_integrator_flow_state(
    graph: nx.Graph,
    integrator: Any,
    *,
    retained_references: list[Any] | None = None,
    transient_networkx_cached_views: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture channels a schedule integrator must preserve during one flow."""

    try:
        layout = _networkx_runtime_layout(graph)
        node_data = layout.node_data
        from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS

        retained = [] if retained_references is None else retained_references
        excluded_graph_keys = frozenset(
            _INTEGRATOR_RUNTIME_GRAPH_KEYS | GRAPH_RUNTIME_CACHE_KEYS
        )

        return {
            "node_support": structural_proof_signature(
                tuple(node for node, _data in node_data)
            ),
            "capacity": _node_alias_state_signature(
                graph,
                ALIAS_VF,
                node_data=node_data,
            ),
            "pressure": _node_alias_state_signature(
                graph,
                ALIAS_DNFR,
                node_data=node_data,
            ),
            "phase": _node_alias_state_signature(
                graph,
                ALIAS_THETA,
                node_data=node_data,
            ),
            "epi_metadata": structural_proof_signature(
                (
                    _node_alias_state_signature(
                        graph,
                        ALIAS_EPI_KIND,
                        node_data=node_data,
                    ),
                    _node_alias_state_signature(
                        graph,
                        ALIAS_SOURCE_GLYPH,
                        node_data=node_data,
                    ),
                )
            ),
            "epi_time_history": _node_history_signature(
                graph,
                ("epi_time_history",),
                node_data=node_data,
            ),
            "protected_nodal_state": (
                _nodal_state_excluding_keys_signature(
                    graph,
                    _INTEGRATOR_MUTABLE_NODE_KEYS,
                    node_data=node_data,
                )
            ),
            "edges": _edge_state_signature(graph, layout=layout),
            "graph_configuration": (
                _filtered_graph_configuration_signature(
                    graph,
                    excluded_keys=excluded_graph_keys,
                    opaque_references=(graph, integrator),
                )
            ),
            "protected_identity": _protected_graph_identity_signature(
                graph,
                excluded_node_keys=_INTEGRATOR_MUTABLE_NODE_KEYS,
                excluded_graph_keys=excluded_graph_keys,
                retained_references=retained,
                opaque_references=(graph, integrator),
                layout=layout,
                transient_networkx_cached_views=(
                    transient_networkx_cached_views
                ),
            ),
            "integrator_binding": _integrator_binding_signature(
                graph,
                integrator,
            ),
        }
    except TNFRValueError:
        raise
    except Exception as exc:
        raise TNFRValueError(
            "Event-schedule integrator state could not be captured safely.",
            context={"reason_type": type(exc).__qualname__},
        ) from exc


def _require_integrator_flow_contract(
    graph: nx.Graph,
    integrator: Any,
    before: Mapping[str, Any],
    *,
    interval_index: int,
    transient_networkx_cached_views: tuple[tuple[str, Any], ...] = (),
) -> None:
    """Reject graph mutations outside the integrator's EPI-flow channels."""

    after = _capture_integrator_flow_state(graph, integrator)
    if before["epi_time_history"] != after["epi_time_history"]:
        raise TNFRValueError(
            "Event-schedule integrators must not write epi_time_history; "
            "the executor owns physical boundary samples.",
            context={"interval_index": interval_index},
        )

    checks = {
        "node_support_preserved": before["node_support"]
        == after["node_support"],
        "capacity_preserved": before["capacity"] == after["capacity"],
        "pressure_preserved": before["pressure"] == after["pressure"],
        "phase_preserved": before["phase"] == after["phase"],
        "epi_metadata_preserved": before["epi_metadata"]
        == after["epi_metadata"],
        "protected_nodal_state_preserved": (
            before["protected_nodal_state"]
            == after["protected_nodal_state"]
        ),
        "edge_state_preserved": before["edges"] == after["edges"],
        "graph_configuration_preserved": (
            before["graph_configuration"]
            == after["graph_configuration"]
        ),
        "protected_identity_preserved": (
            before["protected_identity"] == after["protected_identity"]
        ),
        "integrator_binding_preserved": (
            before["integrator_binding"] == after["integrator_binding"]
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise TNFRValueError(
            "Event-schedule integrators may update only EPI, EPI "
            "derivatives, and the runtime clock during a flow interval.",
            context={
                "interval_index": interval_index,
                "failed_preservation_checks": failed,
            },
        )
    _remove_transient_networkx_cached_views(
        graph,
        transient_networkx_cached_views,
    )


_TYPE_MODULE_DESCRIPTOR = type.__dict__["__module__"]
_TYPE_QUALNAME_DESCRIPTOR = type.__dict__["__qualname__"]
_FUNCTION_MODULE_DESCRIPTOR = FunctionType.__dict__["__module__"]
_FUNCTION_QUALNAME_DESCRIPTOR = FunctionType.__dict__["__qualname__"]
_BUILTIN_FUNCTION_MODULE_DESCRIPTOR = BuiltinFunctionType.__dict__["__module__"]
_BUILTIN_FUNCTION_QUALNAME_DESCRIPTOR = BuiltinFunctionType.__dict__[
    "__qualname__"
]
_METHOD_FUNCTION_DESCRIPTOR = MethodType.__dict__["__func__"]
_METHOD_RECEIVER_DESCRIPTOR = MethodType.__dict__["__self__"]
_BUILTIN_METHOD_RECEIVER_DESCRIPTOR = BuiltinMethodType.__dict__["__self__"]
_PARTIAL_FUNCTION_DESCRIPTOR = partial.__dict__["func"]
_PARTIAL_ARGUMENTS_DESCRIPTOR = partial.__dict__["args"]
_PARTIAL_KEYWORDS_DESCRIPTOR = partial.__dict__["keywords"]
_CELL_CONTENTS_DESCRIPTOR = CellType.__dict__["cell_contents"]
_FUNCTION_STATE_DESCRIPTOR_NAMES = (
    "__annotations__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__doc__",
    "__kwdefaults__",
    "__module__",
    "__name__",
    "__qualname__",
    "__type_params__",
)


def _safe_type_text(kind: type[Any], descriptor: Any) -> str:
    """Read intrinsic class text without invoking a custom metaclass."""

    value = type(descriptor).__get__(descriptor, kind, type(kind))
    if type(value) is not str:
        raise TNFRValueError("callback type metadata must be a string")
    return value


def _read_builtin_descriptor(
    descriptor: Any,
    value: Any,
) -> Any:
    """Read a known built-in descriptor without dynamic instance lookup."""

    return type(descriptor).__get__(descriptor, value, type(value))


def _callback_label(callback: Any) -> str:
    """Return a callback label without invoking callback-owned user code."""

    if isinstance(callback, partial):
        wrapped = _read_builtin_descriptor(_PARTIAL_FUNCTION_DESCRIPTOR, callback)
        return f"functools.partial({_callback_label(wrapped)})"
    if isinstance(callback, MethodType):
        function = _read_builtin_descriptor(_METHOD_FUNCTION_DESCRIPTOR, callback)
        return _callback_label(function)
    if type(callback) is FunctionType:
        module = _read_builtin_descriptor(_FUNCTION_MODULE_DESCRIPTOR, callback)
        qualname = _read_builtin_descriptor(_FUNCTION_QUALNAME_DESCRIPTOR, callback)
        if type(module) is str and type(qualname) is str:
            return f"{module}.{qualname}"
    if isinstance(callback, BuiltinFunctionType):
        module = _read_builtin_descriptor(
            _BUILTIN_FUNCTION_MODULE_DESCRIPTOR,
            callback,
        )
        qualname = _read_builtin_descriptor(
            _BUILTIN_FUNCTION_QUALNAME_DESCRIPTOR,
            callback,
        )
        if type(module) is str and type(qualname) is str:
            return f"{module}.{qualname}"
    kind = type(callback)
    module = _safe_type_text(kind, _TYPE_MODULE_DESCRIPTOR)
    qualname = _safe_type_text(kind, _TYPE_QUALNAME_DESCRIPTOR)
    return f"{module}.{qualname}"


def _pressure_callback_opaque_references(
    graph: nx.Graph,
    callback: Any,
) -> tuple[Any, ...]:
    """Return graph-owned roots signed separately by the refresh contract."""

    return _graph_transaction_protected_values(
        graph,
        excluded=(callback,),
    )


def _function_owned_state_signature(
    function: FunctionType,
    *,
    opaque_references: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Capture mutable intrinsic bindings and closure contents of a function."""

    intrinsic: list[tuple[str, Any]] = []
    closure: tuple[Any, ...] | None = None
    for name in _FUNCTION_STATE_DESCRIPTOR_NAMES:
        descriptor = FunctionType.__dict__.get(name)
        if descriptor is None:
            continue
        value = _read_builtin_descriptor(descriptor, function)
        if name == "__closure__":
            closure = value
        else:
            intrinsic.append((name, value))
    cell_values: list[tuple[str, Any]] = []
    for cell in closure or ():
        try:
            value = _read_builtin_descriptor(_CELL_CONTENTS_DESCRIPTOR, cell)
        except ValueError:
            cell_values.append(("empty", None))
        else:
            cell_values.append(("value", value))
    return (
        "function",
        structural_object_state_signature(
            function,
            opaque_references=opaque_references,
            identity_sensitive=True,
        ),
        tuple(
            (
                name,
                structural_proof_signature(
                    value,
                    opaque_references=opaque_references,
                    identity_sensitive=True,
                ),
            )
            for name, value in intrinsic
        ),
        tuple(
            (
                state,
                structural_proof_signature(
                    value,
                    opaque_references=opaque_references,
                    identity_sensitive=True,
                ),
            )
            for state, value in cell_values
        ),
    )


def _callback_owned_state_signature(
    callback: Any,
    *,
    opaque_references: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Capture direct and safely reachable state owned by a callback."""

    if isinstance(callback, partial):
        function = _read_builtin_descriptor(_PARTIAL_FUNCTION_DESCRIPTOR, callback)
        arguments = _read_builtin_descriptor(_PARTIAL_ARGUMENTS_DESCRIPTOR, callback)
        keywords = _read_builtin_descriptor(_PARTIAL_KEYWORDS_DESCRIPTOR, callback)
        return (
            "partial",
            _callback_owned_state_signature(
                function,
                opaque_references=opaque_references,
            ),
            structural_proof_signature(
                arguments,
                opaque_references=opaque_references,
                identity_sensitive=True,
            ),
            structural_proof_signature(
                keywords,
                opaque_references=opaque_references,
                identity_sensitive=True,
            ),
            structural_object_state_signature(
                callback,
                opaque_references=opaque_references,
                identity_sensitive=True,
            ),
        )
    if isinstance(callback, MethodType):
        function = _read_builtin_descriptor(_METHOD_FUNCTION_DESCRIPTOR, callback)
        receiver = _read_builtin_descriptor(_METHOD_RECEIVER_DESCRIPTOR, callback)
        return (
            "bound_method",
            _function_owned_state_signature(
                function,
                opaque_references=opaque_references,
            ),
            structural_proof_signature(
                receiver,
                opaque_references=opaque_references,
                identity_sensitive=True,
            ),
        )
    if type(callback) is FunctionType:
        return _function_owned_state_signature(
            callback,
            opaque_references=opaque_references,
        )
    if isinstance(callback, BuiltinMethodType):
        receiver = _read_builtin_descriptor(
            _BUILTIN_METHOD_RECEIVER_DESCRIPTOR,
            callback,
        )
        return (
            "builtin_bound_method",
            structural_proof_signature(
                receiver,
                opaque_references=opaque_references,
                identity_sensitive=True,
            ),
        )
    return (
        "callable",
        structural_proof_signature(
            callback,
            opaque_references=opaque_references,
            identity_sensitive=True,
        ),
    )


def _schedule_preparation_state_signature(
    graph: nx.Graph,
    *,
    retained_references: list[Any] | None = None,
) -> tuple[Any, ...]:
    """Seal graph and callback state around user-supplied iterable consumption."""

    callback_present = False
    callback: Any = None
    for key, value in _runtime_mapping_items(_runtime_graph_mapping(graph)):
        if type(key) is str and key == "compute_delta_nfr":
            callback_present = True
            callback = value
            break
    callback_signature: tuple[Any, ...] | None = None
    if callback_present and callable(callback):
        callback_signature = _callback_owned_state_signature(
            callback,
            opaque_references=_pressure_callback_opaque_references(
                graph,
                callback,
            ),
        )
    return (
        structural_proof_signature(
            graph,
            identity_sensitive=True,
            _retained_references=retained_references,
        ),
        callback_present,
        callback_signature,
    )


def _capture_pressure_refresh_state(
    graph: nx.Graph,
    *,
    retained_references: list[Any] | None = None,
    opaque_references: tuple[Any, ...] = (),
    transient_networkx_cached_views: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture graph-owned state that a pressure callback must preserve."""

    layout = _networkx_runtime_layout(graph)
    node_data = layout.node_data
    from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS

    retained = [] if retained_references is None else retained_references
    excluded_graph_keys = frozenset(
        GRAPH_RUNTIME_CACHE_KEYS
        | _PRESSURE_REFRESH_DERIVED_GRAPH_KEYS
        | {"compute_delta_nfr", "_dnfr_weights"}
    )
    return {
        "node_support": structural_proof_signature(
            tuple(node for node, _data in node_data)
        ),
        "epi": structural_proof_signature(
            (
                _node_alias_state_signature(
                    graph,
                    ALIAS_EPI,
                    node_data=node_data,
                ),
                _node_alias_state_signature(
                    graph,
                    ALIAS_EPI_KIND,
                    node_data=node_data,
                ),
                _node_alias_state_signature(
                    graph,
                    ALIAS_SOURCE_GLYPH,
                    node_data=node_data,
                ),
            )
        ),
        "capacity": _node_alias_state_signature(
            graph,
            ALIAS_VF,
            node_data=node_data,
        ),
        "phase": _node_alias_state_signature(
            graph,
            ALIAS_THETA,
            node_data=node_data,
        ),
        "derivatives": structural_proof_signature(
            (
                _node_alias_state_signature(
                    graph,
                    ALIAS_DEPI,
                    node_data=node_data,
                ),
                _node_alias_state_signature(
                    graph,
                    ALIAS_D2EPI,
                    node_data=node_data,
                ),
            )
        ),
        "mutation_history": _node_history_signature(
            graph,
            ("epi_time_history", "epi_history", "_epi_history"),
            node_data=node_data,
        ),
        "glyph_history": _node_history_signature(
            graph,
            ("glyph_history",),
            node_data=node_data,
        ),
        "other": _other_nodal_state_signature(
            graph,
            node_data=node_data,
        ),
        "graph_configuration": _graph_configuration_signature(graph),
        "protected_identity": _protected_graph_identity_signature(
            graph,
            excluded_node_keys=frozenset(ALIAS_DNFR),
            excluded_graph_keys=excluded_graph_keys,
            retained_references=retained,
            opaque_references=opaque_references,
            layout=layout,
            transient_networkx_cached_views=transient_networkx_cached_views,
        ),
        "dnfr_weights": _dnfr_weights_signature(graph),
        "edges": _edge_state_signature(graph, layout=layout),
        "clock": _runtime_clock_signature(graph),
    }


def _pressure_callback_binding_matches(
    graph: nx.Graph,
    *,
    expected_present: bool,
    expected_callback: Any,
) -> bool:
    """Check the executor-frozen pressure-hook presence and identity."""

    entries = _selected_mapping_entries(
        _runtime_graph_mapping(graph),
        ("compute_delta_nfr",),
    )
    present = bool(entries)
    callback = entries[0][1] if entries else None
    return bool(
        present is expected_present
        and (
            not expected_present
            or callback is expected_callback
        )
    )


def _require_pressure_callback_binding(
    graph: nx.Graph,
    *,
    expected_present: bool,
    expected_callback: Any,
) -> None:
    """Reject a changed pressure hook before any replacement can execute."""

    if not _pressure_callback_binding_matches(
        graph,
        expected_present=expected_present,
        expected_callback=expected_callback,
    ):
        raise RuntimeError(
            "configured pressure callback changed during event execution"
        )


def _invoke_restricted_pressure_refresh(
    graph: nx.Graph,
    *,
    expected_present: bool,
    expected_callback: Any,
    n_jobs: int | None,
    boundary_label: str,
    require_callback_state_preserved: bool = True,
) -> tuple[Any, dict[str, bool], str]:
    """Invoke one frozen pressure hook and reject non-pressure mutations."""

    if type(require_callback_state_preserved) is not bool:
        raise TypeError("require_callback_state_preserved must be a bool")

    from ..dynamics.dnfr import default_compute_delta_nfr
    from ..dynamics.runtime import _refresh_delta_nfr

    _require_pressure_callback_binding(
        graph,
        expected_present=expected_present,
        expected_callback=expected_callback,
    )
    callback_label = _callback_label(expected_callback)
    opaque_references = _pressure_callback_opaque_references(
        graph,
        expected_callback,
    )
    graph_identity_opaque_references = (graph, expected_callback)
    retained_callback_bindings: list[Any] = []
    structural_proof_signature(
        expected_callback,
        opaque_references=opaque_references,
        identity_sensitive=True,
        _retained_references=retained_callback_bindings,
    )
    retained_graph_bindings: list[Any] = []
    transient_networkx_cached_views: list[tuple[str, Any]] = []
    before = _capture_pressure_refresh_state(
        graph,
        retained_references=retained_graph_bindings,
        opaque_references=graph_identity_opaque_references,
        transient_networkx_cached_views=transient_networkx_cached_views,
    )
    callback_state_before = (
        _callback_owned_state_signature(
            expected_callback,
            opaque_references=opaque_references,
        )
        if require_callback_state_preserved
        else None
    )
    callback = _refresh_delta_nfr(graph, n_jobs=n_jobs)
    if callback is not expected_callback:
        raise RuntimeError(
            "configured pressure callback changed during event execution"
        )
    _require_pressure_callback_binding(
        graph,
        expected_present=expected_present,
        expected_callback=expected_callback,
    )
    after = _capture_pressure_refresh_state(
        graph,
        opaque_references=graph_identity_opaque_references,
    )
    callback_state_after = (
        _callback_owned_state_signature(
            expected_callback,
            opaque_references=opaque_references,
        )
        if require_callback_state_preserved
        else None
    )

    weights_before = before["dnfr_weights"]
    weights_after = after["dnfr_weights"]
    weights_preserved_or_initialized = bool(
        weights_before == weights_after
        or (
            (weights_before[0] is False or weights_before[1] is True)
            and weights_after[0] is True
            and weights_after[1] is False
            and callback is default_compute_delta_nfr
        )
    )
    checks = {
        "callback_binding_preserved": True,
        "callback_state_preserved": bool(
            not require_callback_state_preserved
            or callback_state_before == callback_state_after
        ),
        "node_support_preserved": before["node_support"]
        == after["node_support"],
        "epi_preserved": before["epi"] == after["epi"],
        "capacity_preserved": before["capacity"] == after["capacity"],
        "edge_state_preserved": before["edges"] == after["edges"],
        "graph_configuration_preserved": before["graph_configuration"]
        == after["graph_configuration"],
        "protected_identity_preserved": before["protected_identity"]
        == after["protected_identity"],
        "dnfr_weights_preserved_or_canonically_initialized": (
            weights_preserved_or_initialized
        ),
        "phase_preserved": before["phase"] == after["phase"],
        "epi_derivatives_preserved": before["derivatives"]
        == after["derivatives"],
        "mutation_history_preserved": before["mutation_history"]
        == after["mutation_history"],
        "glyph_history_preserved": before["glyph_history"]
        == after["glyph_history"],
        "other_nodal_state_preserved": before["other"] == after["other"],
        "runtime_clock_preserved": before["clock"] == after["clock"],
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise TNFRValueError(
            "A pressure callback changed non-pressure graph state.",
            context={
                "boundary": boundary_label,
                "failed_preservation_checks": failed,
            },
        )
    _remove_transient_networkx_cached_views(
        graph,
        transient_networkx_cached_views,
    )
    return callback, checks, callback_label


def _refresh_pressure_boundary(
    graph: nx.Graph,
    partition: PhysicalFlowPartition,
    boundary_index: int,
    *,
    expected_callback_present: bool,
    expected_callback: Any,
    n_jobs: int | None,
) -> PressureRefreshBoundaryObservation:
    """Refresh pressure once and reject any non-pressure graph mutation."""

    segment_count = partition.segment_count
    if boundary_index < 0 or boundary_index > segment_count:
        raise IndexError("physical boundary index is outside the partition")
    segment = partition.segments[
        boundary_index if boundary_index < segment_count else segment_count - 1
    ]
    boundary_time = (
        segment.start_time
        if boundary_index < segment_count
        else segment.end_time
    )
    exact_time = (
        segment.exact_start_time
        if boundary_index < segment_count
        else segment.exact_end_time
    )
    offset = (
        segment.start_offset
        if boundary_index < segment_count
        else segment.end_offset
    )
    _require_runtime_clock(
        graph,
        boundary_time,
        boundary=(
            f"physical_partition[{partition.parent_interval.index}]"
            f".boundary[{boundary_index}]"
        ),
    )

    before = _require_pressure_boundary_snapshot(
        graph,
        parent_interval_index=partition.parent_interval.index,
        boundary_index=boundary_index,
        side="before",
    )
    boundary_label = (
        f"physical_partition[{partition.parent_interval.index}]"
        f".boundary[{boundary_index}]"
    )
    callback, checks, callback_label = _invoke_restricted_pressure_refresh(
        graph,
        expected_present=expected_callback_present,
        expected_callback=expected_callback,
        n_jobs=n_jobs,
        boundary_label=boundary_label,
    )
    after = _require_pressure_boundary_snapshot(
        graph,
        parent_interval_index=partition.parent_interval.index,
        boundary_index=boundary_index,
        side="after",
    )
    observation = PressureRefreshBoundaryObservation(
        parent_interval_index=partition.parent_interval.index,
        boundary_index=boundary_index,
        time=boundary_time,
        exact_time=exact_time,
        offset=offset,
        callback_name=callback_label,
        callback_identity=id(callback),
        configured_callback=bool(
            expected_callback_present and callable(expected_callback)
        ),
        callback_binding_preserved=checks["callback_binding_preserved"],
        callback_state_preserved=checks["callback_state_preserved"],
        before=before,
        after=after,
        node_support_preserved=bool(
            checks["node_support_preserved"]
            and _same_structural_value(before.nodes, after.nodes)
        ),
        epi_preserved=bool(
            checks["epi_preserved"]
            and _binary64_vectors_are_identical(before.epi, after.epi)
        ),
        capacity_preserved=bool(
            checks["capacity_preserved"]
            and _binary64_vectors_are_identical(before.nu_f, after.nu_f)
        ),
        conductance_preserved=before.conductance == after.conductance,
        edge_state_preserved=checks["edge_state_preserved"],
        graph_configuration_preserved=checks[
            "graph_configuration_preserved"
        ],
        dnfr_weights_preserved_or_canonically_initialized=(
            checks["dnfr_weights_preserved_or_canonically_initialized"]
        ),
        phase_preserved=checks["phase_preserved"],
        epi_derivatives_preserved=checks["epi_derivatives_preserved"],
        mutation_history_preserved=checks["mutation_history_preserved"],
        glyph_history_preserved=checks["glyph_history_preserved"],
        other_nodal_state_preserved=checks["other_nodal_state_preserved"],
        runtime_clock_preserved=checks["runtime_clock_preserved"],
        pressure_changed=not _binary64_vectors_are_identical(
            before.delta_nfr,
            after.delta_nfr,
        ),
    )
    observation = replace(
        observation,
        _proof_stamp=_sealed_dataclass_stamp(
            observation,
            _PRESSURE_REFRESH_BOUNDARY_PROOF_VERSION,
        ),
    )
    if not observation.nonpressure_state_preserved:
        raise TNFRValueError(
            "A physical pressure callback changed non-pressure graph state.",
            context={
                "parent_interval_index": partition.parent_interval.index,
                "boundary_index": boundary_index,
                "failed_preservation_checks": tuple(
                    name
                    for name in (
                        "node_support_preserved",
                        "epi_preserved",
                        "capacity_preserved",
                        "conductance_preserved",
                    )
                    if not getattr(observation, name)
                ),
            },
        )
    return observation


def _physical_modal_observation(
    graph: nx.Graph,
    partition: PhysicalFlowPartition,
    segment_index: int,
    boundary: PressureRefreshBoundaryObservation,
) -> PhysicalEulerModalObservation:
    """Capture one frozen pure-EPI Euler diagnostic without changing policy."""

    segment = partition.segments[segment_index]
    available = _binary64_vectors_are_identical(
        boundary.after.delta_nfr,
        boundary.after.binary64_pure_epi_pressure,
    )
    reason: str | None = None
    diagnostic = None
    if not available:
        reason = "refreshed_pressure_is_not_binary64_pure_epi_diffusion"
    else:
        from ..physics.structural_diffusion import (
            diagnose_euler_relaxation_window,
        )

        try:
            diagnostic = diagnose_euler_relaxation_window(
                graph,
                dt=segment.duration,
            )
        except (TypeError, ValueError, nx.NetworkXException) as exc:
            available = False
            reason = (
                "frozen_euler_modal_diagnostic_unavailable:"
                f"{type(exc).__qualname__}:{exc}"
            )

    observation = PhysicalEulerModalObservation(
        parent_interval_index=partition.parent_interval.index,
        segment_index=segment_index,
        dt=segment.duration,
        available=available,
        abstention_reason=reason,
        target_fraction=(
            None if diagnostic is None else diagnostic.target_fraction
        ),
        spectral_relative_tolerance=(
            None
            if diagnostic is None
            else diagnostic.spectral_relative_tolerance
        ),
        spectral_zero_threshold=(
            None if diagnostic is None else diagnostic.spectral_zero_threshold
        ),
        decay_rates=(
            None
            if diagnostic is None
            else tuple(float(value) for value in diagnostic.decay_rates)
        ),
        modal_multipliers=(
            None
            if diagnostic is None
            else tuple(float(value) for value in diagnostic.modal_multipliers)
        ),
        slowest_decay_rate=(
            None if diagnostic is None else diagnostic.slowest_decay_rate
        ),
        fastest_decay_rate=(
            None if diagnostic is None else diagnostic.fastest_decay_rate
        ),
        euler_stability_limit=(
            None if diagnostic is None else diagnostic.euler_stability_limit
        ),
        maximum_modal_factor=(
            None if diagnostic is None else diagnostic.maximum_modal_factor
        ),
        modal_steps=None if diagnostic is None else diagnostic.modal_steps,
        policy_window=None if diagnostic is None else diagnostic.policy_window,
        is_euler_stable=(
            None if diagnostic is None else diagnostic.is_euler_stable
        ),
        scope=None if diagnostic is None else diagnostic.scope,
    )
    sealed = replace(
        observation,
        _proof_stamp=_sealed_dataclass_stamp(
            observation,
            "physical_euler_modal_observation_v1",
        ),
    )
    if sealed._proof_fields_are_intact():
        return sealed

    abstention = PhysicalEulerModalObservation(
        parent_interval_index=partition.parent_interval.index,
        segment_index=segment_index,
        dt=segment.duration,
        available=False,
        abstention_reason="frozen_euler_modal_diagnostic_not_representable",
        target_fraction=None,
        spectral_relative_tolerance=None,
        spectral_zero_threshold=None,
        decay_rates=None,
        modal_multipliers=None,
        slowest_decay_rate=None,
        fastest_decay_rate=None,
        euler_stability_limit=None,
        maximum_modal_factor=None,
        modal_steps=None,
        policy_window=None,
        is_euler_stable=None,
        scope=None,
    )
    return replace(
        abstention,
        _proof_stamp=_sealed_dataclass_stamp(
            abstention,
            "physical_euler_modal_observation_v1",
        ),
    )


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
    endpoint_continuous = bool(
        _same_structural_value(flow_endpoint.nodes, endpoint.nodes)
        and flow_endpoint.exact_epi == endpoint.exact_epi
        and _binary64_vectors_are_identical(flow_endpoint.epi, endpoint.epi)
    )
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
    pre_flow_index: Mapping[int, ExecutedNodalFlowInterval],
    post_flow_index: Mapping[int, ExecutedNodalFlowInterval] | None = None,
) -> ExecutedGlyphStage:
    """Bind an executor certificate to captured endpoints and adjacent flows."""

    if post_flow_index is None:
        post_flow_index = pre_flow_index

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
    pre_flow = (
        pre_flow_index.get(pre_interval.index) if pre_positive else None
    )
    post_flow = (
        post_flow_index.get(post_interval.index) if post_positive else None
    )
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
        reception_observations=pending.result.reception_observations,
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


def _physical_partition_composition_operation(
    *,
    position: int,
    interval: StructuralFlowInterval,
    expected_nodes: tuple[Any, ...],
    evidence: ExecutedPressureRefreshedFlowPartition | None,
) -> RepresentedEPIScheduleOperation:
    """Represent a complete physical partition as its exact segment product."""

    reasons: list[str] = []
    if evidence is None:
        reasons.append("physical_partition_evidence_missing")
        return _represented_operation(
            position=position,
            operation_kind="flow",
            operation_index=interval.index,
            operator_name=None,
            nodes=None,
            exact_epi_before=None,
            exact_epi_after=None,
            exact_metric_ray_before=None,
            exact_metric_ray_after=None,
            exact_energy_gain_upper_bound=None,
            ineligibility_reasons=reasons,
        )
    if not evidence.physical_pressure_reevaluated_partition_established:
        reasons.append("physical_partition_proof_fields_not_intact")
    if evidence.partition.parent_interval != interval:
        reasons.append("physical_partition_parent_interval_mismatch")
    if not evidence.all_segment_exact_affine_maps_identified:
        reasons.append("physical_segment_affine_map_not_identified")
    if not evidence.exact_common_metric_gain_product_certified:
        reasons.append("physical_partition_common_metric_gain_not_certified")

    first = evidence.segment_flow_evidence[0].certificate
    last = evidence.segment_flow_evidence[-1].certificate
    operation_nodes: tuple[Any, ...] | None = None
    exact_before: tuple[Fraction, ...] | None = None
    exact_after: tuple[Fraction, ...] | None = None
    if first is None or last is None:
        reasons.append("physical_partition_endpoint_certificate_unavailable")
    elif first.left.nodes != last.right.nodes:
        reasons.append("physical_partition_support_changed")
    else:
        operation_nodes = first.left.nodes
        exact_before = first.left.exact_epi
        exact_after = last.right.exact_epi
    if operation_nodes != expected_nodes:
        reasons.append("physical_partition_node_order_mismatch")
    if operation_nodes is None:
        exact_before = None
        exact_after = None

    metric = evidence.exact_common_metric
    gain = evidence.exact_composed_gain_bound
    if metric is None:
        reasons.append("physical_partition_exact_metric_unavailable")
    if gain is None:
        reasons.append("physical_partition_exact_gain_unavailable")
    return _represented_operation(
        position=position,
        operation_kind="flow",
        operation_index=interval.index,
        operator_name=None,
        nodes=operation_nodes,
        exact_epi_before=exact_before,
        exact_epi_after=exact_after,
        exact_metric_ray_before=metric,
        exact_metric_ray_after=metric,
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
    physical_partitions: tuple[ExecutedPressureRefreshedFlowPartition, ...] = (),
) -> ObservedRepresentedEPIScheduleComposition:
    """Compose only the rational maps represented by the complete observed trace."""

    flow_index = _flow_by_interval(flows)
    physical_index = {
        item.partition.parent_interval.index: item
        for item in physical_partitions
    }
    if len(physical_index) != len(physical_partitions):
        raise RuntimeError("duplicate physical partition evidence")
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
    if any(index not in positive_indices for index in physical_index):
        raise RuntimeError("unexpected physical partition evidence")
    if set(flow_index).intersection(physical_index):
        raise RuntimeError("flow interval has duplicate execution evidence")
    if any(index < 0 or index >= schedule.event_count for index in stage_index):
        raise RuntimeError("unexpected runtime glyph-stage evidence")

    operations: list[RepresentedEPIScheduleOperation] = []
    for interval in schedule.intervals:
        if interval.exact_duration > 0:
            if interval.index in physical_index:
                operations.append(
                    _physical_partition_composition_operation(
                        position=len(operations),
                        interval=interval,
                        expected_nodes=nodes,
                        evidence=physical_index[interval.index],
                    )
                )
            else:
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
    target_nodes: tuple[Any, ...],
    method: str | None,
    n_jobs: int | None,
    include_flow_certificate: bool,
) -> ExecutedNodalFlowInterval | None:
    """Advance one positive interval and optionally bind endpoint evidence."""

    _require_runtime_clock(
        graph,
        interval.start_time,
        boundary=f"interval[{interval.index}].start",
    )
    if interval.exact_duration == 0:
        return None

    metadata_references: list[Any] = []
    metadata_state = _schedule_preparation_state_signature(
        graph,
        retained_references=metadata_references,
    )
    integrate_method = _resolve_integrator_method(integrator)
    metadata = _flow_runtime_metadata(
        graph,
        interval,
        integrator,
        integrate_method,
        method=method,
    )
    if metadata_state != _schedule_preparation_state_signature(graph):
        raise TNFRValueError(
            "Event-schedule flow metadata resolution changed graph state."
        )
    residual_required = not metadata.integrator_provenance_certified
    capture_endpoints = bool(include_flow_certificate or residual_required)
    _record_guarded_mutation_flow_boundary(graph)
    if capture_endpoints:
        left, left_captured = _capture_interval_endpoint(graph)
    else:
        left, left_captured = None, False
    if residual_required and not left_captured:
        raise TNFRValueError(
            "A custom event-schedule integrator requires a capturable nodal "
            "state for its held-input residual.",
            context={
                "interval_index": interval.index,
                "endpoint": "left",
            },
        )
    retained_flow_bindings: list[Any] = []
    transient_networkx_cached_views: list[tuple[str, Any]] = []
    integrator_state_before = _capture_integrator_flow_state(
        graph,
        integrator,
        retained_references=retained_flow_bindings,
        transient_networkx_cached_views=transient_networkx_cached_views,
    )

    integrate_method(
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
    _require_integrator_flow_contract(
        graph,
        integrator,
        integrator_state_before,
        interval_index=interval.index,
        transient_networkx_cached_views=tuple(
            transient_networkx_cached_views
        ),
    )
    if capture_endpoints:
        right, right_captured = _capture_interval_endpoint(graph)
    else:
        right, right_captured = None, False
    if residual_required:
        if not right_captured or left is None or right is None:
            raise TNFRValueError(
                "A custom event-schedule integrator requires a capturable "
                "nodal state for its held-input residual.",
                context={
                    "interval_index": interval.index,
                    "endpoint": "right",
                },
            )
        from ..physics.runtime_flow_stability import (
            certify_observed_nodal_flow_interval,
        )

        residual = certify_observed_nodal_flow_interval(
            left,
            right,
            duration=interval.duration,
            integrator_name=metadata.integrator_name,
            method=metadata.resolved_method,
            substeps=metadata.resolved_substeps,
            gamma_is_none=metadata.gamma_is_none,
            clipping_applied=None,
            extended_dynamics_requested=(
                metadata.extended_dynamics_requested
            ),
        )
        if not residual.exact_nodal_equation_realized:
            raise TNFRValueError(
                "A custom event-schedule integrator must realize the exact "
                "held-input nodal equation over each flow interval.",
                context={
                    "interval_index": interval.index,
                    "nodal_residual": (
                        residual.exact_nodal_equation_residual
                    ),
                },
            )
    _record_guarded_mutation_flow_boundary(graph)

    if not include_flow_certificate:
        return None
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
            integrator_provenance_certified=object.__getattribute__(
                evidence,
                "integrator_provenance_certified",
            ),
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
        integrator_provenance_certified=object.__getattribute__(
            evidence,
            "integrator_provenance_certified",
        ),
        _proof_stamp=_executed_flow_interval_stamp(evidence),
    )


def _execute_pressure_refreshed_flow_partition(
    graph: nx.Graph,
    partition: PhysicalFlowPartition,
    integrator: Any,
    *,
    target_nodes: tuple[Any, ...],
    expected_callback_present: bool,
    expected_callback: Any,
    method: str | None,
    n_jobs: int | None,
) -> ExecutedPressureRefreshedFlowPartition:
    """Execute physical segments with pressure refreshed at every boundary."""

    boundaries: list[PressureRefreshBoundaryObservation] = []
    flows: list[ExecutedNodalFlowInterval] = []
    modal: list[PhysicalEulerModalObservation] = []
    for segment_index, segment in enumerate(partition.segments):
        boundary = _refresh_pressure_boundary(
            graph,
            partition,
            segment_index,
            expected_callback_present=expected_callback_present,
            expected_callback=expected_callback,
            n_jobs=n_jobs,
        )
        boundaries.append(boundary)
        modal.append(
            _physical_modal_observation(
                graph,
                partition,
                segment_index,
                boundary,
            )
        )
        flow = _execute_flow_interval(
            graph,
            segment,
            integrator,
            target_nodes=target_nodes,
            method=method,
            n_jobs=n_jobs,
            include_flow_certificate=True,
        )
        if flow is None:
            raise RuntimeError("a positive physical segment produced no evidence")
        flows.append(flow)

    boundaries.append(
        _refresh_pressure_boundary(
            graph,
            partition,
            partition.segment_count,
            expected_callback_present=expected_callback_present,
            expected_callback=expected_callback,
            n_jobs=n_jobs,
        )
    )
    flow_tuple = tuple(flows)
    metric, gains, product, gain_certified = _partition_gain_facts(flow_tuple)
    evidence = ExecutedPressureRefreshedFlowPartition(
        partition=partition,
        boundary_observations=tuple(boundaries),
        segment_flow_evidence=flow_tuple,
        modal_observations=tuple(modal),
        pressure_refresh_callback_invocations=len(boundaries),
        exact_common_metric=metric,
        exact_segment_gain_bounds=gains,
        exact_composed_gain_bound=product,
        _exact_common_metric_gain_product_certified=gain_certified,
    )
    evidence = replace(
        evidence,
        _proof_stamp=_sealed_dataclass_stamp(
            evidence,
            _PRESSURE_REFRESHED_PARTITION_PROOF_VERSION,
        ),
    )
    if not evidence.physical_pressure_reevaluated_partition_established:
        raise RuntimeError("physical partition evidence failed internal validation")
    return evidence


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
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = (),
) -> OperatorEventExecutionResult:
    """Execute one finite flow/jump schedule inside a whole-schedule rollback.

    The live graph time must already equal schedule.start_time. Every positive
    unpartitioned flow is delegated once to the configured nodal EPI integrator.
    A declared physical partition instead delegates each segment independently
    and refreshes DeltaNFR at every boundary, including the terminal boundary.
    The resulting binary64 clock must equal every scheduled endpoint. Operator
    jumps use the same fixed initial target tuple and canonical dispatcher as
    ordinary network words. Any failure restores graph-owned state, topology,
    histories, event telemetry and runtime caches.
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
    transaction = GraphTransactionSnapshot(graph)
    try:
        preparation_references: list[Any] = []
        preparation_state = _schedule_preparation_state_signature(
            graph,
            retained_references=preparation_references,
        )
        schedule.__post_init__()
        _validate_schedule_clock(schedule)
        partitions = _materialize_physical_flow_partitions(
            schedule,
            physical_flow_partitions,
        )
        materialized_context = None if context is None else dict(context)
        operators, execution_word = _prepare_word(
            schedule,
            materialized_context,
        )
        if preparation_state != _schedule_preparation_state_signature(graph):
            raise TNFRValueError(
                "Schedule input materialization changed graph state."
            )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
    try:
        partition_index = {
            item.parent_interval.index: item for item in partitions
        }
        _require_runtime_clock(
            graph,
            schedule.start_time,
            boundary="schedule.start",
        )
        _validate_hybrid_log(graph)

        from ..dynamics.dnfr import default_compute_delta_nfr
        from .word_execution import execute_network_operator_stage

        runtime_layout = _networkx_runtime_layout(graph)
        targets = tuple(node for node, _data in runtime_layout.node_data)
        configured_callback_entries = _selected_mapping_entries(
            runtime_layout.graph_mapping,
            ("compute_delta_nfr",),
        )
        configured_callback_present = bool(configured_callback_entries)
        configured_compute_delta_nfr = (
            configured_callback_entries[0][1]
            if configured_callback_entries
            else default_compute_delta_nfr
        )
        stage_pressure_refresh_callback_invocations = 0
        if configured_callback_present and callable(configured_compute_delta_nfr):

            def compute_delta_nfr(live_graph: nx.Graph) -> None:
                """Run the frozen stage pressure callback and verify its binding."""

                nonlocal stage_pressure_refresh_callback_invocations
                _invoke_restricted_pressure_refresh(
                    live_graph,
                    expected_present=configured_callback_present,
                    expected_callback=configured_compute_delta_nfr,
                    n_jobs=n_jobs,
                    boundary_label="operator_stage_pressure_refresh",
                )
                stage_pressure_refresh_callback_invocations += 1

        else:
            compute_delta_nfr = None
        events_committed: list[ExecutedOperatorEvent] = []
        flow_interval_evidence: list[ExecutedNodalFlowInterval] = []
        physical_partition_evidence: list[
            ExecutedPressureRefreshedFlowPartition
        ] = []
        pending_glyph_stages: list[_PendingGlyphStage] = []
        effective_flow_certification = bool(
            include_flow_certificates
            or include_stage_certificates
            or partitions
        )
        positive_intervals = tuple(
            interval.index
            for interval in schedule.intervals
            if interval.exact_duration > 0
        )
        integrator = None
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise

    try:
        if positive_intervals:
            integrator_resolution_references: list[Any] = []
            integrator_resolution_state = (
                _schedule_preparation_state_signature(
                    graph,
                    retained_references=integrator_resolution_references,
                )
            )
            integrator = _resolve_schedule_integrator_instance(
                graph,
                cache_result=False,
            )
            if integrator_resolution_state != (
                _schedule_preparation_state_signature(graph)
            ):
                raise TNFRValueError(
                    "Event-schedule integrator resolution changed graph state."
                )
            _cache_schedule_integrator_instance(graph, integrator)

        with warnings.catch_warnings():
            if suppress_birth_warnings:
                warnings.filterwarnings(
                    "ignore",
                    message=RECEPTION_NO_SOURCES_WARNING_PATTERN,
                )
            for interval in schedule.intervals:
                _require_pressure_callback_binding(
                    graph,
                    expected_present=configured_callback_present,
                    expected_callback=configured_compute_delta_nfr,
                )
                if interval.exact_duration > 0:
                    if integrator is None:
                        raise RuntimeError(
                            "positive flow interval has no configured integrator"
                        )
                    partition = partition_index.get(interval.index)
                    if partition is not None:
                        physical_partition_evidence.append(
                            _execute_pressure_refreshed_flow_partition(
                                graph,
                                partition,
                                integrator,
                                target_nodes=targets,
                                expected_callback_present=(
                                    configured_callback_present
                                ),
                                expected_callback=configured_compute_delta_nfr,
                                method=method,
                                n_jobs=n_jobs,
                            )
                        )
                    else:
                        evidence = _execute_flow_interval(
                            graph,
                            interval,
                            integrator,
                            target_nodes=targets,
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

                _require_pressure_callback_binding(
                    graph,
                    expected_present=configured_callback_present,
                    expected_callback=configured_compute_delta_nfr,
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
                    "transaction_snapshot": transaction,
                }
                if include_stage_certificates:
                    stage_kwargs["include_epi_jump_certificate"] = True
                result = execute_network_operator_stage(
                    graph,
                    operator,
                    targets,
                    **stage_kwargs,
                )
                _require_pressure_callback_binding(
                    graph,
                    expected_present=configured_callback_present,
                    expected_callback=configured_compute_delta_nfr,
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
                _record_guarded_mutation_flow_boundary(graph)
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
                _append_guarded_hybrid_event_record(
                    graph,
                    committed.as_record(),
                )
                events_committed.append(committed)

        _require_pressure_callback_binding(
            graph,
            expected_present=configured_callback_present,
            expected_callback=configured_compute_delta_nfr,
        )
        _require_runtime_clock(
            graph,
            schedule.end_time,
            boundary="schedule.end",
        )
        flow_tuple = tuple(flow_interval_evidence)
        physical_tuple = tuple(physical_partition_evidence)
        flow_index = _flow_by_interval(flow_tuple)
        pre_flow_index = dict(flow_index)
        post_flow_index = dict(flow_index)
        for item in physical_tuple:
            parent_index = item.partition.parent_interval.index
            if parent_index in pre_flow_index or parent_index in post_flow_index:
                raise RuntimeError("flow interval has duplicate execution evidence")
            pre_flow_index[parent_index] = item.segment_flow_evidence[-1]
            post_flow_index[parent_index] = item.segment_flow_evidence[0]
        glyph_stage_tuple = tuple(
            _finalize_glyph_stage(
                pending,
                schedule,
                pre_flow_index,
                post_flow_index,
            )
            for pending in pending_glyph_stages
        )
        represented_composition = (
            _compose_observed_represented_epi_schedule(
                schedule,
                targets,
                flow_tuple,
                glyph_stage_tuple,
                physical_tuple,
            )
            if include_stage_certificates
            else None
        )
        execution_result = OperatorEventExecutionResult(
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
                stage_pressure_refresh_callback_invocations
                + sum(
                    item.pressure_refresh_callback_invocations
                    for item in physical_tuple
                )
            ),
            flow_certification_requested=effective_flow_certification,
            flow_interval_evidence=flow_tuple,
            physical_flow_partition_indices=tuple(partition_index),
            physical_flow_partition_evidence=physical_tuple,
            physical_pressure_refresh_callback_invocations=sum(
                item.pressure_refresh_callback_invocations
                for item in physical_tuple
            ),
            stage_pressure_refresh_callback_invocations=(
                stage_pressure_refresh_callback_invocations
            ),
            stage_certification_requested=include_stage_certificates,
            glyph_stage_evidence=glyph_stage_tuple,
            represented_epi_schedule_composition=represented_composition,
        )
        return replace(
            execution_result,
            _proof_stamp=_sealed_dataclass_stamp(
                execution_result,
                _OPERATOR_EVENT_EXECUTION_RESULT_PROOF_VERSION,
            ),
        )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


__all__ = (
    "ExecutedGlyphStage",
    "ExecutedNodalFlowInterval",
    "ExecutedOperatorEvent",
    "ExecutedPressureRefreshedFlowPartition",
    "ObservedRepresentedEPIScheduleComposition",
    "PhysicalEulerModalObservation",
    "PressureRefreshBoundaryObservation",
    "RepresentedEPIScheduleOperation",
    "OperatorEventExecutionResult",
    "execute_operator_event_schedule",
)
