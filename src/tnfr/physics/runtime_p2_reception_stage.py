r"""Finite causal binding of one executed Reception stage to the P2 kernel.

The global theorem in :mod:`binary64_p2_reception_stability` concerns an
abstract two-node numeric kernel.  This module supplies the narrower runtime
bridge: it selects one Reception event from one sealed
``OperatorEventExecutionResult`` and verifies that the executor's captured
stage endpoints realize that kernel on the observed pair.

The bridge checks the executor-owned event/stage correspondence, exact
two-phase Jacobi dispatch, mutual singleton runtime neighbours, binary64
``0.5``, the common hard interval, the fixed exact diffusion-metric ray, and
bit-for-bit replay of both observed output coordinates.  The mathematical
conclusion remains numeric consensus and exact zero centered EPI energy;
opposite signed-zero outputs are therefore allowed.

This is evidence for one completed event inside one finite atomic schedule.
It does not bind REMESH history/configuration to the executed graph, certify
all Reception telemetry or monitor effects, retain current live-graph
identity, or establish future/repeated stage stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any, Literal

from ..errors import TNFRValueError
from ..operators.event_runtime import (
    ExecutedGlyphStage,
    OperatorEventExecutionResult,
)
from ..types import Glyph
from ..utils._structural_signature import (
    binary64_vectors_are_identical,
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
    _evaluate_binary64_half_reception_pair,
)
from .network_stage_stability import (
    AllTargetNeighborStageCertificate,
    AllTargetNeighborStageStep,
    _validate_bridge_stage_certificate,
)
from .reception_realization import ReceptionEPIRealizationCertificate
from .runtime_flow_stability import NodalFlowStateSnapshot

__all__ = (
    "ExecutedP2HalfReceptionStageCertificate",
    "certify_executed_p2_half_reception_stage",
)

Binary64Pair = tuple[float, float]
ExactPair = tuple[Fraction, Fraction]

_PROOF_VERSION = "executed_p2_half_reception_stage_v1"
_SCOPE = (
    "One selected Reception event in one sealed completed operator-event "
    "execution: the canonical executor admitted the event in its finite "
    "grammar context, retained the all-target two-phase Jacobi stage, and "
    "captured an EPI transition that matches the global P2 half-Reception "
    "kernel coordinate by coordinate in binary64. The observed output has "
    "numeric consensus and exact centered EPI energy zero in the source "
    "metric. The result binds this finite EPI stage to executor-owned atomic "
    "schedule provenance. It does not bind the source REMESH history or "
    "configuration to the graph, certify auxiliary Reception telemetry, "
    "monitor effects or raw topology, retain current live-graph identity, "
    "or prove future/repeated stage, solver, or full TNFR stability."
)

_CONDITION_NAMES = (
    "global_p2_kernel_certificate_is_intact",
    "operator_event_execution_result_is_intact",
    "selected_stage_belongs_to_the_same_execution",
    "selected_event_is_zero_duration_reception_en",
    "selected_stage_has_exact_captured_runtime_endpoints",
    "execution_support_matches_the_ordered_p2_nodes",
    "reception_stage_is_admissible_two_phase_jacobi",
    "executor_neighbor_certificate_is_present_and_bound",
    "runtime_neighbors_are_mutual_singletons",
    "runtime_reception_mix_is_exact_binary64_one_half",
    "runtime_hard_clip_matches_the_source_interval",
    "captured_input_belongs_to_the_source_interval",
    "capacity_and_effective_conductance_are_preserved",
    "captured_metric_ray_matches_the_source_metric",
    "accepted_output_matches_the_production_kernel_by_bits",
    "accepted_output_is_numeric_consensus",
    "accepted_output_has_zero_centered_energy",
    "executor_reports_whole_schedule_graph_state_atomicity",
)


def _same(left: Any, right: Any) -> bool:
    """Compare structural values without invoking identifier equality."""

    try:
        return proof_stamps_are_identical(
            structural_proof_signature(left),
            structural_proof_signature(right),
        )
    except BaseException:
        return False


def _strict_conditions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_CONDITION_NAMES)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == _CONDITION_NAMES[index]
            for index, item in enumerate(value)
        )
    )


def _proof_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not ExecutedP2HalfReceptionStageCertificate:
        raise TypeError("proof value must have its canonical result type")
    opaque = (
        object.__getattribute__(value, "kernel_certificate"),
        object.__getattribute__(value, "execution_result"),
        object.__getattribute__(value, "executed_stage"),
    )
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(ExecutedP2HalfReceptionStageCertificate)
        if item.name != "_proof_stamp"
    )
    return (
        _PROOF_VERSION,
        structural_proof_signature(payload, opaque_references=opaque),
    )


def _seal(
    value: "ExecutedP2HalfReceptionStageCertificate",
) -> "ExecutedP2HalfReceptionStageCertificate":
    return replace(value, _proof_stamp=_proof_stamp(value))


def _sealed(value: Any) -> bool:
    try:
        return proof_stamps_are_identical(
            object.__getattribute__(value, "_proof_stamp"),
            _proof_stamp(value),
        )
    except BaseException:
        return False


def _binary64_pair_from_snapshot(
    snapshot: Any,
    *,
    label: str,
) -> tuple[Binary64Pair, ExactPair]:
    if type(snapshot) is not NodalFlowStateSnapshot:
        raise TNFRValueError(f"{label} must be a canonical nodal-flow snapshot")
    values = object.__getattribute__(snapshot, "epi")
    exact = object.__getattribute__(snapshot, "exact_epi")
    if (
        type(values) is not tuple
        or len(values) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in values)
        or type(exact) is not tuple
        or len(exact) != 2
        or any(type(value) is not Fraction for value in exact)
    ):
        raise TNFRValueError(f"{label} must contain two finite binary64 EPI values")
    typed_values = (values[0], values[1])
    typed_exact = (exact[0], exact[1])
    if typed_exact != tuple(Fraction.from_float(value) for value in typed_values):
        raise TNFRValueError(f"{label} exact EPI values do not match binary64")
    return typed_values, typed_exact


def _binary64_pair_from_stage_vector(value: Any, *, label: str) -> Binary64Pair:
    try:
        materialized = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{label} is not a finite binary64 pair") from exc
    if len(materialized) != 2 or not all(
        math.isfinite(item) for item in materialized
    ):
        raise TNFRValueError(f"{label} is not a finite binary64 pair")
    return materialized[0], materialized[1]


def _exact_conductance(snapshot: NodalFlowStateSnapshot) -> tuple[ExactPair, ExactPair]:
    value = object.__getattribute__(snapshot, "conductance")
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(row) is not tuple or len(row) != 2 for row in value)
        or any(
            type(entry) is not Fraction
            for row in value
            for entry in row
        )
    ):
        raise TNFRValueError("captured conductance must be an exact 2x2 matrix")
    return (value[0][0], value[0][1]), (value[1][0], value[1][1])


def _exact_capacities(snapshot: NodalFlowStateSnapshot) -> ExactPair:
    represented = object.__getattribute__(snapshot, "nu_f")
    value = object.__getattribute__(snapshot, "exact_nu_f")
    if (
        type(represented) is not tuple
        or len(represented) != 2
        or any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in represented
        )
        or type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not Fraction or item <= 0 for item in value)
    ):
        raise TNFRValueError(
            "captured capacities must be two positive binary64 values"
        )
    if value != tuple(Fraction.from_float(item) for item in represented):
        raise TNFRValueError("captured exact capacities do not match binary64")
    return value[0], value[1]


def _metric_ray(
    conductance: tuple[ExactPair, ExactPair],
    capacities: ExactPair,
) -> ExactPair:
    if not (
        conductance[0][0] == 0
        and conductance[1][1] == 0
        and conductance[0][1] == conductance[1][0]
        and conductance[0][1] > 0
    ):
        raise TNFRValueError(
            "captured conductance must be the symmetric positive P2 matrix"
        )
    degrees = tuple(sum(row, Fraction(0)) for row in conductance)
    if any(value <= 0 for value in degrees):
        raise TNFRValueError("captured support must have positive row strengths")
    weights = tuple(
        degree / capacity
        for degree, capacity in zip(degrees, capacities, strict=True)
    )
    total = sum(weights, Fraction(0))
    return weights[0] / total, weights[1] / total


def _normalized_binary64_metric(value: Any, *, label: str) -> ExactPair:
    try:
        represented = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{label} is not a positive binary64 pair") from exc
    if len(represented) != 2 or any(
        not math.isfinite(item) or item <= 0.0 for item in represented
    ):
        raise TNFRValueError(f"{label} is not a positive binary64 pair")
    exact = tuple(Fraction.from_float(item) for item in represented)
    total = sum(exact, Fraction(0))
    return exact[0] / total, exact[1] / total


def _centered_energy(values: ExactPair, metric: ExactPair) -> Fraction:
    center = sum(
        (
            weight * value
            for weight, value in zip(metric, values, strict=True)
        ),
        Fraction(0),
    )
    return sum(
        (
            weight * (value - center) ** 2
            for weight, value in zip(metric, values, strict=True)
        ),
        Fraction(0),
    ) / 2


def _runtime_neighbor_indices(
    step: AllTargetNeighborStageStep,
    nodes: tuple[Any, Any],
) -> tuple[tuple[int], tuple[int]]:
    observed = step.runtime_neighbor_sets
    if (
        type(observed) is not tuple
        or len(observed) != 2
        or any(type(row) is not tuple or len(row) != 1 for row in observed)
        or not _same(observed[0][0], nodes[1])
        or not _same(observed[1][0], nodes[0])
    ):
        raise TNFRValueError(
            "Reception runtime neighbours must be the mutual P2 singletons"
        )
    return ((1,), (0,))


@dataclass(frozen=True, slots=True)
class _ExecutedP2StageModel:
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


def _matches_model(
    value: "ExecutedP2HalfReceptionStageCertificate",
    expected: _ExecutedP2StageModel,
) -> bool:
    """Compare stored conclusions with one already derived strict model."""

    observed_payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(_ExecutedP2StageModel)
    )
    expected_payload = tuple(
        (item.name, object.__getattribute__(expected, item.name))
        for item in fields(_ExecutedP2StageModel)
    )
    return _same(observed_payload, expected_payload)


def _model_from_validated_dependencies(
    kernel: P2HalfReceptionRemeshStabilityCertificate,
    execution: OperatorEventExecutionResult,
    stage: ExecutedGlyphStage,
    event_index: int,
) -> _ExecutedP2StageModel:
    """Derive the adapter after same-call validation of both parent proofs."""

    evidence = object.__getattribute__(execution, "glyph_stage_evidence")
    events = object.__getattribute__(execution, "events")
    if (
        type(event_index) is not int
        or event_index < 0
        or event_index >= len(evidence)
        or event_index >= len(events)
        or stage is not evidence[event_index]
    ):
        raise TNFRValueError("selected stage does not belong to this execution")
    event = object.__getattribute__(stage, "event")
    committed_event = events[event_index]
    if event is not committed_event:
        raise TNFRValueError(
            "selected stage event does not retain same-invocation identity"
        )
    if (
        event.event_index != event_index
        or event.operator_name != "reception"
        or event.glyph is not Glyph.EN
        or object.__getattribute__(event, "zero_duration") is not True
    ):
        raise TNFRValueError("selected event must be zero-duration Reception/EN")
    if (
        object.__getattribute__(execution, "stage_certification_requested")
        is not True
        or object.__getattribute__(stage, "endpoint_capture_complete") is not True
        or object.__getattribute__(stage, "exact_runtime_endpoint_bound") is not True
    ):
        raise TNFRValueError(
            "selected Reception stage lacks exact captured runtime endpoints"
        )

    left = object.__getattribute__(stage, "left")
    right = object.__getattribute__(stage, "right")
    if (
        type(left) is not NodalFlowStateSnapshot
        or type(right) is not NodalFlowStateSnapshot
    ):
        raise TNFRValueError("selected Reception stage endpoints are unavailable")
    source_nodes = object.__getattribute__(kernel, "node_order")
    if type(source_nodes) is not tuple or len(source_nodes) != 2:
        raise TNFRValueError("global P2 certificate has invalid node support")
    nodes = (source_nodes[0], source_nodes[1])
    for observed_nodes in (
        object.__getattribute__(execution, "target_nodes"),
        object.__getattribute__(left, "nodes"),
        object.__getattribute__(right, "nodes"),
    ):
        if not _same(observed_nodes, nodes):
            raise TNFRValueError(
                "executed node order does not match the global P2 certificate"
            )

    certificate = object.__getattribute__(stage, "certificate")
    if (
        object.__getattribute__(stage, "certificate_kind") != "neighbor"
        or type(certificate) is not AllTargetNeighborStageCertificate
        or certificate.operator_name != "Reception"
        or certificate.glyph != "EN"
        or not _same(certificate.nodes, nodes)
        or certificate.repetitions_requested != 1
        or certificate.repetitions_observed != 1
        or certificate.repetitions_completed != 1
        or certificate.fixed_support_declared is not True
        or certificate.all_stages_admissible is not True
        or type(certificate.steps) is not tuple
        or len(certificate.steps) != 1
    ):
        raise TNFRValueError(
            "selected executor evidence is not one admissible Reception stage"
        )
    try:
        validated = _validate_bridge_stage_certificate(certificate)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(
            "executor Reception certificate failed deep revalidation"
        ) from exc
    step = validated.step
    if (
        type(step) is not AllTargetNeighborStageStep
        or step is not certificate.steps[0]
        or step.index != 0
        or not _same(step.nodes, nodes)
        or any(
            type(local) is not ReceptionEPIRealizationCertificate
            for local in step.local_certificates
        )
        or step.runtime_stage_admissible is not True
        or step.atomic_rejection_nodes != ()
        or event.stage_schedule != kernel.stage_schedule
        or event.stage_schedule != "two_phase_jacobi"
        or event.nodes_processed != 2
    ):
        raise TNFRValueError(
            "Reception event is not an admissible two-phase P2 stage"
        )

    neighbor_indices = _runtime_neighbor_indices(step, nodes)
    if (
        type(step.mix_factor) is not float
        or not binary64_vectors_are_identical(
            (step.mix_factor,),
            (object.__getattribute__(kernel, "binary64_mix_factor"),),
        )
        or step.exact_mix_factor != Fraction(1, 2)
        or step.exact_mix_factor != kernel.exact_mix_factor
    ):
        raise TNFRValueError(
            "Reception runtime mix must be exact binary64 one half"
        )

    source = object.__getattribute__(kernel, "remesh_class_certificate")
    configuration = object.__getattribute__(source, "configuration")
    lower = configuration.epi_min
    upper = configuration.epi_max
    if (
        step.clip_mode != "hard"
        or source.clip_mode != "hard"
        or type(step.epi_lower_bound) is not float
        or type(step.epi_upper_bound) is not float
        or not binary64_vectors_are_identical(
            (step.epi_lower_bound, step.epi_upper_bound),
            (lower, upper),
        )
        or Fraction.from_float(step.epi_lower_bound) != source.epi_min
        or Fraction.from_float(step.epi_upper_bound) != source.epi_max
    ):
        raise TNFRValueError(
            "Reception hard-clamp interval does not match the P2 source class"
        )

    binary_before, exact_before = _binary64_pair_from_snapshot(
        left,
        label="left endpoint",
    )
    binary_after, exact_after = _binary64_pair_from_snapshot(
        right,
        label="right endpoint",
    )
    if any(value < lower or value > upper for value in binary_before):
        raise TNFRValueError("captured Reception input is outside the source class")

    certified_before = _binary64_pair_from_stage_vector(
        step.state_before,
        label="certified Reception input",
    )
    if not binary64_vectors_are_identical(certified_before, binary_before):
        raise TNFRValueError(
            "Reception certificate input does not match the captured endpoint"
        )
    accepted = _binary64_pair_from_stage_vector(
        step.runtime_accepted_state_after,
        label="accepted Reception state",
    )
    if not binary64_vectors_are_identical(accepted, binary_after):
        raise TNFRValueError(
            "accepted Reception proposal does not match the captured endpoint"
        )
    expected = _evaluate_binary64_half_reception_pair(
        binary_before,
        lower=lower,
        upper=upper,
    )
    replay_matches = binary64_vectors_are_identical(expected, binary_after)
    if not replay_matches:
        raise TNFRValueError(
            "executed Reception endpoint does not match the P2 kernel replay"
        )

    left_capacity = _exact_capacities(left)
    right_capacity = _exact_capacities(right)
    left_conductance = _exact_conductance(left)
    right_conductance = _exact_conductance(right)
    if left_capacity != right_capacity or left_conductance != right_conductance:
        raise TNFRValueError(
            "Reception stage changed capacity or effective conductance"
        )
    metric = _metric_ray(left_conductance, left_capacity)
    source_metric = object.__getattribute__(kernel, "exact_normalized_metric")
    if (
        type(source_metric) is not tuple
        or len(source_metric) != 2
        or metric != source_metric
        or object.__getattribute__(stage, "exact_metric_ray_before") != metric
        or object.__getattribute__(stage, "exact_metric_ray_after") != metric
    ):
        raise TNFRValueError(
            "captured diffusion metric does not match the P2 source metric"
        )
    certificate_pre_metric = _normalized_binary64_metric(
        step.pre_diffusion_certificate.metric_weights,
        label="Reception certificate pre-metric",
    )
    certificate_post_metric = _normalized_binary64_metric(
        step.post_diffusion_certificate.metric_weights,
        label="Reception certificate post-metric",
    )
    if certificate_pre_metric != metric or certificate_post_metric != metric:
        raise TNFRValueError(
            "Reception certificate metric does not match captured endpoints"
        )

    energy_before = _centered_energy(exact_before, metric)
    energy_after = _centered_energy(exact_after, metric)
    numeric_consensus = binary_after[0] == binary_after[1]
    if not numeric_consensus or energy_after != 0:
        raise TNFRValueError(
            "executed P2 Reception output did not reach numeric consensus"
        )
    bitwise_consensus = binary64_vectors_are_identical(
        (binary_after[0],),
        (binary_after[1],),
    )
    wrapper_affine_bridge = object.__getattribute__(
        stage,
        "_represented_affine_gain_bound_at_observed_endpoint_certified",
    )
    derived_affine_bridge = bool(
        validated.all_local_snapshots_in_affine_model_domain
        and validated.represented_consensus_subspace_preserved
        and validated.runtime_exact_match
        and validated.pre_post_metric_exactly_proportional
        and step.pre_metric_affine_jump_certificate.supports_global_gain_theorem
    )
    expected_affine_gain = (
        step.pre_metric_affine_jump_certificate
        .exact_quotient_energy_gain_upper_bound
        if derived_affine_bridge
        else None
    )
    expected_abstention_reason = (
        None
        if derived_affine_bridge
        else (
            "neighbor_exact_common_metric_gain_not_certified;"
            "glyph_exact_common_metric_bridge_not_certified"
        )
    )
    observed_abstention_reason = object.__getattribute__(
        stage,
        "certificate_abstention_reason",
    )
    if (
        type(wrapper_affine_bridge) is not bool
        or wrapper_affine_bridge != derived_affine_bridge
        or object.__getattribute__(stage, "exact_common_metric_bridge")
        != derived_affine_bridge
        or object.__getattribute__(stage, "exact_energy_gain_upper_bound")
        != expected_affine_gain
        or (
            observed_abstention_reason is not None
            and type(observed_abstention_reason) is not str
        )
        or observed_abstention_reason != expected_abstention_reason
    ):
        raise TNFRValueError("executor affine-bridge observation is invalid")

    atomic = (
        object.__getattribute__(execution, "whole_schedule_graph_state_atomic")
        is True
    )
    if not atomic:
        raise TNFRValueError("execution lacks whole-schedule graph atomicity")
    conditions = tuple((name, True) for name in _CONDITION_NAMES)
    return _ExecutedP2StageModel(
        event_index=event_index,
        node_order=nodes,
        binary64_epi_before=binary_before,
        binary64_epi_after=binary_after,
        exact_epi_before=exact_before,
        exact_epi_after=exact_after,
        exact_normalized_metric=(metric[0], metric[1]),
        runtime_neighbor_indices=neighbor_indices,
        exact_mix_factor=Fraction(1, 2),
        clip_mode="hard",
        exact_epi_lower_bound=source.epi_min,
        exact_epi_upper_bound=source.epi_max,
        exact_centered_energy_before=energy_before,
        exact_centered_energy_after=energy_after,
        exact_global_kernel_energy_gain_upper_bound=Fraction(0),
        runtime_kernel_replay_matches_by_bits=replay_matches,
        observed_output_binary64_bits_identical=bitwise_consensus,
        represented_affine_bridge_available_at_observed_endpoint=(
            derived_affine_bridge
        ),
        conditions=conditions,
    )


@dataclass(frozen=True, slots=True)
class ExecutedP2HalfReceptionStageCertificate:
    """Sealed finite adapter from one runtime EN stage to the global P2 kernel."""

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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact_after_dependencies_validation(self) -> bool:
        """Rederive after validating both nested proofs in the same call."""

        try:
            if type(self) is not ExecutedP2HalfReceptionStageCertificate:
                return False
            if not _sealed(self) or not _strict_conditions(self.conditions):
                return False
            kernel = object.__getattribute__(self, "kernel_certificate")
            execution = object.__getattribute__(self, "execution_result")
            stage = object.__getattribute__(self, "executed_stage")
            if (
                type(kernel) is not P2HalfReceptionRemeshStabilityCertificate
                or type(execution) is not OperatorEventExecutionResult
                or type(stage) is not ExecutedGlyphStage
            ):
                return False
            expected = _model_from_validated_dependencies(
                kernel,
                execution,
                stage,
                object.__getattribute__(self, "event_index"),
            )
            return _matches_model(self, expected)
        except BaseException:
            return False

    def _proof_fields_are_intact(self) -> bool:
        try:
            kernel = object.__getattribute__(self, "kernel_certificate")
            execution = object.__getattribute__(self, "execution_result")
            if (
                type(kernel) is not P2HalfReceptionRemeshStabilityCertificate
                or not kernel._proof_fields_are_intact()
                or not all(passed for _name, passed in kernel.conditions)
                or type(execution) is not OperatorEventExecutionResult
                or not execution._proof_fields_are_intact()
            ):
                return False
            return self._proof_fields_are_intact_after_dependencies_validation()
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def executed_p2_half_reception_stage_certificate_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _name, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("executed_p2_reception_stage_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def finite_executor_bound_epi_stage_certified(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def finite_grammar_admission_observed(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def canonical_two_phase_reception_stage_observed(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def runtime_p2_support_observed(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def observed_capacity_and_effective_conductance_preserved(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def observed_source_metric_preserved(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def source_global_q_zero_applies_to_observed_epi_transition(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def observed_numeric_consensus_certified(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def finite_schedule_graph_state_atomicity_certified(self) -> bool:
        return self.executed_p2_half_reception_stage_certificate_certified

    @property
    def complete_reception_stage_stability_class_certified(self) -> bool:
        return False

    @property
    def executed_remesh_configuration_bound(self) -> bool:
        return False

    @property
    def reception_auxiliary_state_preservation_certified(self) -> bool:
        return False

    @property
    def raw_graph_topology_preservation_certified(self) -> bool:
        return False

    @property
    def current_live_graph_state_bound(self) -> bool:
        return False

    @property
    def future_or_repeated_live_graph_stability_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def _select_reception_event_index(
    execution: OperatorEventExecutionResult,
    event_index: int | None,
) -> int:
    evidence = object.__getattribute__(execution, "glyph_stage_evidence")
    if event_index is None:
        indices = tuple(
            index
            for index, stage in enumerate(evidence)
            if type(stage) is ExecutedGlyphStage
            and object.__getattribute__(stage, "event").glyph is Glyph.EN
        )
        if len(indices) != 1:
            raise TNFRValueError(
                "event_index is required unless execution has exactly one "
                "Reception event"
            )
        return indices[0]
    if type(event_index) is not int:
        raise TNFRValueError("event_index must be an integer or None")
    if event_index < 0 or event_index >= len(evidence):
        raise TNFRValueError("event_index is outside glyph-stage evidence")
    return event_index


def certify_executed_p2_half_reception_stage(
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate,
    execution_result: OperatorEventExecutionResult,
    *,
    event_index: int | None = None,
) -> ExecutedP2HalfReceptionStageCertificate:
    """Bind one executed EN stage to the global half-Reception P2 kernel."""

    if type(kernel_certificate) is not P2HalfReceptionRemeshStabilityCertificate:
        raise TNFRValueError(
            "kernel_certificate must be an exact P2 half-Reception certificate"
        )
    if not kernel_certificate._proof_fields_are_intact() or not all(
        passed for _name, passed in kernel_certificate.conditions
    ):
        raise TNFRValueError(
            "kernel_certificate is unsealed, tampered, or inconsistent"
        )
    if type(execution_result) is not OperatorEventExecutionResult:
        raise TNFRValueError(
            "execution_result must be an exact operator-event execution result"
        )
    if not execution_result._proof_fields_are_intact():
        raise TNFRValueError(
            "execution_result is unsealed, tampered, or inconsistent"
        )
    selected_index = _select_reception_event_index(execution_result, event_index)
    stage = execution_result.glyph_stage_evidence[selected_index]
    model = _model_from_validated_dependencies(
        kernel_certificate,
        execution_result,
        stage,
        selected_index,
    )
    result = ExecutedP2HalfReceptionStageCertificate(
        kernel_certificate=kernel_certificate,
        execution_result=execution_result,
        executed_stage=stage,
        event_index=model.event_index,
        node_order=model.node_order,
        binary64_epi_before=model.binary64_epi_before,
        binary64_epi_after=model.binary64_epi_after,
        exact_epi_before=model.exact_epi_before,
        exact_epi_after=model.exact_epi_after,
        exact_normalized_metric=model.exact_normalized_metric,
        runtime_neighbor_indices=model.runtime_neighbor_indices,
        exact_mix_factor=model.exact_mix_factor,
        clip_mode=model.clip_mode,
        exact_epi_lower_bound=model.exact_epi_lower_bound,
        exact_epi_upper_bound=model.exact_epi_upper_bound,
        exact_centered_energy_before=model.exact_centered_energy_before,
        exact_centered_energy_after=model.exact_centered_energy_after,
        exact_global_kernel_energy_gain_upper_bound=(
            model.exact_global_kernel_energy_gain_upper_bound
        ),
        runtime_kernel_replay_matches_by_bits=(
            model.runtime_kernel_replay_matches_by_bits
        ),
        observed_output_binary64_bits_identical=(
            model.observed_output_binary64_bits_identical
        ),
        represented_affine_bridge_available_at_observed_endpoint=(
            model.represented_affine_bridge_available_at_observed_endpoint
        ),
        conditions=model.conditions,
    )
    sealed = _seal(result)
    if not (
        _sealed(sealed)
        and _strict_conditions(sealed.conditions)
        and _matches_model(sealed, model)
    ):
        raise RuntimeError("constructed executed P2 Reception proof is inconsistent")
    return sealed
