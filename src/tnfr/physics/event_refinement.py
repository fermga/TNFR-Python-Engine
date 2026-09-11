"""Exact pre-jump observations for held-pressure and physical ZHIR paths.

The Mutation gate uses timestamped binary64 subtraction and division.  That
arithmetic need not equal the rational secant between exact representations of
the same endpoint floats.  This module records both without conflating them. It
also compares an executor-sealed pressure-refreshed physical partition with a
matched held-pressure baseline while keeping terminal-segment and whole-parent
secants separate. Internal integrator substeps retain interval-start pressure
and therefore are not physical diffusion refinements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Real
from typing import Any

import networkx as nx

from ..operators.event_runtime import (
    ExecutedGlyphStage,
    ExecutedNodalFlowInterval,
    ExecutedOperatorEvent,
    ExecutedPressureRefreshedFlowPartition,
    OperatorEventExecutionResult,
)
from ..operators.event_timing import ScheduledOperatorEvent
from ..operators.network_stage import MutationStageDecisionObservation
from ..types import Glyph
from ..utils._structural_signature import (
    binary64_vectors_are_identical as _binary64_vectors_are_identical,
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .mutation_trigger import (
    MutationTriggerCertificate,
    MutationTriggerEvidence,
    certify_mutation_trigger,
)
from .runtime_flow_stability import (
    NodalFlowIntervalCertificate,
    capture_nodal_flow_state,
)
from .structural_diffusion import diagnose_euler_relaxation_window

__all__ = (
    "ExecutedEventLocalZHIRPhysicalPrejumpObservation",
    "EventLocalZHIRHeldPressureComparison",
    "EventLocalZHIRPhysicalPrejumpObservation",
    "EventLocalZHIRPhysicalRefinementComparison",
    "EventLocalZHIRPrejumpObservation",
    "compare_event_local_zhir_held_pressure_subdivision",
    "compare_event_local_zhir_physical_refinement",
    "observe_executed_event_local_zhir_physical_prejump",
    "observe_event_local_zhir_physical_prejump",
    "observe_event_local_zhir_prejump",
)


_OBSERVATION_PROOF_VERSION = "event_local_zhir_prejump_observation_v4"
_COMPARISON_PROOF_VERSION = "event_local_zhir_held_pressure_comparison_v5"
_PHYSICAL_OBSERVATION_PROOF_VERSION = (
    "event_local_zhir_physical_prejump_observation_v2"
)
_EXECUTED_PHYSICAL_OBSERVATION_PROOF_VERSION = (
    "executed_event_local_zhir_physical_prejump_observation_v1"
)
_PHYSICAL_COMPARISON_PROOF_VERSION = (
    "event_local_zhir_physical_refinement_comparison_v2"
)
_COMPARISON_KIND = "internal_held_pressure_subdivision"
_PHYSICAL_COMPARISON_KIND = "pressure_reevaluated_physical_partition"
_OBSERVATION_SCOPE = (
    "one sealed binary64 sampling boundary paired offline by coordinate with a "
    "scheduled ZHIR jump, with exact represented and actual binary64 gate "
    "secants kept separate; common schedule-execution provenance, operator "
    "admission and U4 readiness are not certified"
)
_COMPARISON_SCOPE = (
    "offline coordinate-paired comparison of two sealed built-in held-pressure "
    "binary64 realizations at the same pre-ZHIR boundary with different "
    "internal substep counts; common schedule-execution provenance is not "
    "certified. "
    "Internal substeps retain interval-start pressure. Physical pressure-"
    "reevaluated refinement, diffusion modal decisions, solver accuracy or "
    "order, future or repeated behavior, U4 readiness and adaptive policy are "
    "not certified"
)
_PHYSICAL_OBSERVATION_SCOPE = (
    "one executor-sealed pressure-refreshed physical partition paired offline "
    "by coordinate with a scheduled ZHIR jump. The actual Mutation window is "
    "the terminal physical segment; the whole-parent-horizon secant is a "
    "separate offline diagnostic. Common jump execution, U4 readiness, solver "
    "accuracy or order, mesh convergence and future behavior are not certified"
)
_EXECUTED_PHYSICAL_OBSERVATION_SCOPE = (
    "one executor-sealed schedule execution linking an immediately preceding "
    "pressure-refreshed physical partition to its committed ZHIR stage and "
    "ordered per-node Mutation trigger decisions. This finite common-execution "
    "observation does not certify solver accuracy or order, mesh convergence, "
    "general U4 readiness, adaptive policy or future behavior"
)
_PHYSICAL_COMPARISON_SCOPE = (
    "finite comparison of one held-pressure pre-ZHIR observation with one "
    "executor-sealed pressure-refreshed physical partition sharing its parent "
    "interval, event, initial state and threshold. Whole-parent-horizon and "
    "actual terminal-window gate decisions remain separate. The held outer "
    "map uses the parent duration; the physical diagnostic composes segment "
    "multipliers only for one fixed pure-EPI generator realized by the trusted "
    "held-pressure runtime. Solver accuracy or order, mesh convergence, modal "
    "equivalence, U4 readiness, adaptive policy and future behavior are not "
    "certified"
)


def _materialized_binary64(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be representable as binary64") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _difference(
    right: tuple[Fraction, ...],
    left: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        right_value - left_value
        for right_value, left_value in zip(right, left, strict=True)
    )


def _observation_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not EventLocalZHIRPrejumpObservation:
        raise TypeError("observation must have the canonical result type")
    return (
        _OBSERVATION_PROOF_VERSION,
        tuple(
            (
                item.name,
                structural_proof_signature(
                    object.__getattribute__(value, item.name)
                ),
            )
            for item in fields(EventLocalZHIRPrejumpObservation)
            if item.name != "_proof_stamp"
        ),
    )


@dataclass(frozen=True, slots=True)
class EventLocalZHIRPrejumpObservation:
    """Sealed evidence at one scheduled Mutation's immediately prior flow.

    ``exact_rational_physical_secants`` uses exact represented endpoints and
    duration.  The ``binary64_*`` fields reproduce the actual timestamp and
    EPI subtraction followed by division used by ``mutation_trigger``.
    The historical ``scope`` dataclass field remains visible to schema
    introspection; its raw value participates in the seal and its public read
    remains fixed.
    """

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
    scope: str = field(default=_OBSERVATION_SCOPE, init=False)
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    binary64_initial_epi: tuple[float, ...] = ()
    binary64_endpoint_epi: tuple[float, ...] = ()
    binary64_capacity: tuple[float, ...] = ()
    binary64_pressure: tuple[float, ...] = ()

    def __getattribute__(self, name: str) -> Any:
        """Expose the historical fixed scope while sealing its raw storage."""

        if name == "scope":
            return _OBSERVATION_SCOPE
        return object.__getattribute__(self, name)

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _observation_stamp(self)
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        return proof_stamps_are_identical(observed, expected)

    @property
    def pre_jump_observation_certified(self) -> bool:
        """Whether this observation retains its construction proof."""

        return self._proof_fields_are_intact()


def _validated_flow_certificate(flow: Any) -> NodalFlowIntervalCertificate:
    if type(flow) is not ExecutedNodalFlowInterval:
        raise TypeError("flow must be an ExecutedNodalFlowInterval")
    if not flow._proof_fields_are_intact():
        raise ValueError("flow evidence is unsealed, tampered, or stale")
    if not flow.runtime_bound_binary64_held_pressure_interval_identified:
        raise ValueError(
            "flow must identify the built-in held-pressure binary64 runtime"
        )
    certificate = flow.certificate
    if type(certificate) is not NodalFlowIntervalCertificate:
        raise ValueError("flow lacks a canonical nested interval certificate")
    if not certificate._proof_fields_are_intact():
        raise ValueError("nested flow certificate is tampered or stale")

    interval = flow.interval
    if interval.exact_duration <= 0:
        raise ValueError("pre-ZHIR flow duration must be positive")
    if (
        interval.duration != certificate.duration
        or interval.exact_duration != certificate.exact_duration
        or structural_proof_signature(certificate.left.nodes)
        != structural_proof_signature(certificate.right.nodes)
        or flow.resolved_substeps != certificate.substeps
        or type(certificate.substeps) is not int
        or certificate.substeps <= 0
    ):
        raise ValueError("flow interval and nested certificate are inconsistent")
    if not certificate.nodes:
        raise ValueError("pre-ZHIR flow must contain at least one node")
    return certificate


def observe_event_local_zhir_prejump(
    flow: ExecutedNodalFlowInterval,
    event: ScheduledOperatorEvent,
    *,
    xi: Real,
) -> EventLocalZHIRPrejumpObservation:
    """Seal actual gate arithmetic before ZHIR, admitted or rejected."""

    certificate = _validated_flow_certificate(flow)
    if type(event) is not ScheduledOperatorEvent:
        raise TypeError("event must be a ScheduledOperatorEvent")
    if event.glyph is not Glyph.ZHIR or event.operator_name != "mutation":
        raise ValueError("event must be the canonical scheduled ZHIR operator")

    interval = flow.interval
    exact_jump_placement = bool(
        event.event_index == interval.index
        and event.exact_event_time == interval.exact_end_time
        and event.event_offset == interval.end_offset
    )
    binary64_jump_placement = (
        structural_proof_signature(event.event_time)
        == structural_proof_signature(interval.end_time)
    )
    if not exact_jump_placement or not binary64_jump_placement:
        raise ValueError("scheduled ZHIR is not immediately after the flow endpoint")

    sample_interval = interval.end_time - interval.start_time
    endpoint_subtraction_matches = bool(
        math.isfinite(sample_interval)
        and sample_interval > 0.0
        and structural_proof_signature(sample_interval)
        == structural_proof_signature(interval.duration)
    )
    if not endpoint_subtraction_matches:
        raise ValueError(
            "binary64 endpoint subtraction must equal the declared flow duration"
        )

    xi_value = _materialized_binary64(xi, "xi")
    if xi_value < 0.0:
        raise ValueError("xi must be nonnegative")
    exact_xi = Fraction.from_float(xi_value)
    epi_deltas: list[float] = []
    gate_rates: list[float] = []
    for left_epi, right_epi in zip(
        certificate.left.epi,
        certificate.right.epi,
        strict=True,
    ):
        epi_delta = right_epi - left_epi
        observed_rate = epi_delta / sample_interval
        if not math.isfinite(epi_delta) or not math.isfinite(observed_rate):
            raise ValueError("binary64 Mutation gate arithmetic must remain finite")
        epi_deltas.append(epi_delta)
        gate_rates.append(observed_rate)

    exact_epi_deltas = tuple(Fraction.from_float(value) for value in epi_deltas)
    exact_gate_rates = tuple(Fraction.from_float(value) for value in gate_rates)
    rational_secants = tuple(
        (right_epi - left_epi) / certificate.exact_duration
        for left_epi, right_epi in zip(
            certificate.left.exact_epi,
            certificate.right.exact_epi,
            strict=True,
        )
    )
    rational_minus_runtime = tuple(
        rational - runtime
        for rational, runtime in zip(
            rational_secants,
            exact_gate_rates,
            strict=True,
        )
    )
    rate_equal_by_node = tuple(value == 0 for value in rational_minus_runtime)

    observation = EventLocalZHIRPrejumpObservation(
        event_identity=(
            event.event_index,
            event.cycle_index,
            event.word_position,
            event.operator_name,
            event.glyph.value,
        ),
        flow_interval_index=interval.index,
        nodes=certificate.nodes,
        substeps=certificate.substeps,
        interval_start_time=interval.start_time,
        interval_end_time=interval.end_time,
        duration=interval.duration,
        exact_interval_start_time=interval.exact_start_time,
        exact_interval_end_time=interval.exact_end_time,
        exact_duration=certificate.exact_duration,
        event_time=event.event_time,
        exact_event_time=event.exact_event_time,
        event_offset=event.event_offset,
        exact_initial_epi=certificate.left.exact_epi,
        exact_endpoint_epi=certificate.right.exact_epi,
        exact_capacity=certificate.left.exact_nu_f,
        exact_pressure=certificate.left.exact_delta_nfr,
        exact_conductance=certificate.left.conductance,
        exact_rational_physical_secants=rational_secants,
        binary64_sample_interval=sample_interval,
        exact_binary64_sample_interval=Fraction.from_float(sample_interval),
        binary64_epi_deltas=tuple(epi_deltas),
        exact_binary64_epi_deltas=exact_epi_deltas,
        binary64_observed_gate_rates=tuple(gate_rates),
        exact_binary64_observed_gate_rates=exact_gate_rates,
        exact_rational_minus_binary64_gate_rates=rational_minus_runtime,
        rational_and_binary64_gate_rates_equal_by_node=rate_equal_by_node,
        rational_and_binary64_gate_rates_equal=all(rate_equal_by_node),
        xi=xi_value,
        exact_xi=exact_xi,
        binary64_observed_strict_gate_decisions=tuple(
            value > xi_value for value in gate_rates
        ),
        exact_jump_at_pre_flow_endpoint=exact_jump_placement,
        binary64_jump_at_pre_flow_endpoint=binary64_jump_placement,
        endpoint_subtraction_matches_declared_duration=endpoint_subtraction_matches,
        binary64_initial_epi=certificate.left.epi,
        binary64_endpoint_epi=certificate.right.epi,
        binary64_capacity=certificate.left.nu_f,
        binary64_pressure=certificate.left.delta_nfr,
    )
    return replace(observation, _proof_stamp=_observation_stamp(observation))


def _comparison_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not EventLocalZHIRHeldPressureComparison:
        raise TypeError("comparison must have the canonical result type")
    return (
        _COMPARISON_PROOF_VERSION,
        tuple(
            (
                item.name,
                structural_proof_signature(
                    object.__getattribute__(value, item.name)
                ),
            )
            for item in fields(EventLocalZHIRHeldPressureComparison)
            if item.name != "_proof_stamp"
        ),
    )


@dataclass(frozen=True, slots=True)
class EventLocalZHIRHeldPressureComparison:
    """Sealed comparison of two represented binary64 Mutation gate outcomes.

    Historical fixed-claim fields remain visible to dataclass introspection.
    Their raw values participate in the proof seal while public reads remain
    fixed, so low-level mutation invalidates rather than promotes evidence.
    """

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
        default=False,
        init=False,
    )
    diffusion_modal_decisions_applicable: bool = field(default=False, init=False)
    diffusion_modal_decisions_established: bool = field(default=False, init=False)
    solver_accuracy_certified: bool = field(default=False, init=False)
    solver_order_certified: bool = field(default=False, init=False)
    physical_refinement_equivalence_certified: bool = field(default=False, init=False)
    future_or_repeated_behavior_certified: bool = field(default=False, init=False)
    u4_readiness_certified: bool = field(default=False, init=False)
    adaptive_policy_certified: bool = field(default=False, init=False)
    scope: str = field(default=_COMPARISON_SCOPE, init=False)
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __getattribute__(self, name: str) -> Any:
        """Expose fixed historical claims without allowing promotion."""

        if name in {
            "physical_pressure_reevaluated_partition_established",
            "diffusion_modal_decisions_applicable",
            "diffusion_modal_decisions_established",
            "solver_accuracy_certified",
            "solver_order_certified",
            "physical_refinement_equivalence_certified",
            "future_or_repeated_behavior_certified",
            "u4_readiness_certified",
            "adaptive_policy_certified",
        }:
            return False
        if name == "scope":
            return _COMPARISON_SCOPE
        return object.__getattribute__(self, name)

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _comparison_stamp(self)
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        return proof_stamps_are_identical(observed, expected)

    @property
    def event_local_gate_invariance_certified(self) -> bool:
        """Return the strict local gate claim only while its seal is intact."""

        return bool(
            self._proof_fields_are_intact()
            and self._event_local_gate_invariance_certified
        )


def _require_comparable_observations(
    baseline: Any,
    candidate: Any,
) -> tuple[
    EventLocalZHIRPrejumpObservation,
    EventLocalZHIRPrejumpObservation,
]:
    for label, value in (("baseline", baseline), ("candidate", candidate)):
        if type(value) is not EventLocalZHIRPrejumpObservation:
            raise TypeError(f"{label} must be an EventLocalZHIRPrejumpObservation")
        if not value._proof_fields_are_intact():
            raise ValueError(f"{label} observation is tampered or stale")

    requirements = (
        ("event_identity", baseline.event_identity == candidate.event_identity),
        (
            "exact_event_time",
            baseline.exact_event_time == candidate.exact_event_time,
        ),
        (
            "event_time",
            structural_proof_signature(baseline.event_time)
            == structural_proof_signature(candidate.event_time),
        ),
        ("event_offset", baseline.event_offset == candidate.event_offset),
        (
            "flow_interval_index",
            baseline.flow_interval_index == candidate.flow_interval_index,
        ),
        ("exact_duration", baseline.exact_duration == candidate.exact_duration),
        (
            "node_support",
            structural_proof_signature(baseline.nodes)
            == structural_proof_signature(candidate.nodes),
        ),
        (
            "exact_initial_epi",
            baseline.exact_initial_epi == candidate.exact_initial_epi,
        ),
        (
            "binary64_initial_epi",
            _binary64_vectors_are_identical(
                baseline.binary64_initial_epi,
                candidate.binary64_initial_epi,
            ),
        ),
        ("exact_capacity", baseline.exact_capacity == candidate.exact_capacity),
        (
            "binary64_capacity",
            _binary64_vectors_are_identical(
                baseline.binary64_capacity,
                candidate.binary64_capacity,
            ),
        ),
        ("exact_pressure", baseline.exact_pressure == candidate.exact_pressure),
        (
            "binary64_pressure",
            _binary64_vectors_are_identical(
                baseline.binary64_pressure,
                candidate.binary64_pressure,
            ),
        ),
        (
            "exact_conductance",
            baseline.exact_conductance == candidate.exact_conductance,
        ),
        ("exact_xi", baseline.exact_xi == candidate.exact_xi),
        (
            "binary64_xi",
            structural_proof_signature(baseline.xi)
            == structural_proof_signature(candidate.xi),
        ),
        ("different_substeps", baseline.substeps != candidate.substeps),
    )
    failed = tuple(name for name, passed in requirements if not passed)
    if failed:
        raise ValueError(
            "pre-ZHIR observations are outside the held-pressure subdivision "
            f"comparison domain: {', '.join(failed)}"
        )
    return baseline, candidate


def compare_event_local_zhir_held_pressure_subdivision(
    baseline: EventLocalZHIRPrejumpObservation,
    candidate: EventLocalZHIRPrejumpObservation,
) -> EventLocalZHIRHeldPressureComparison:
    """Compare two sealed actual binary64 gate rates at one ZHIR boundary."""

    baseline, candidate = _require_comparable_observations(baseline, candidate)
    endpoint_difference = _difference(
        candidate.exact_endpoint_epi,
        baseline.exact_endpoint_epi,
    )
    rational_secant_differences = _difference(
        candidate.exact_rational_physical_secants,
        baseline.exact_rational_physical_secants,
    )
    gate_rate_differences = _difference(
        candidate.exact_binary64_observed_gate_rates,
        baseline.exact_binary64_observed_gate_rates,
    )
    gate_rate_bounds = tuple(abs(value) for value in gate_rate_differences)
    signed_margins = tuple(
        value - baseline.exact_xi
        for value in baseline.exact_binary64_observed_gate_rates
    )
    threshold_distances = tuple(abs(value) for value in signed_margins)
    decision_agreement_by_node = tuple(
        left == right
        for left, right in zip(
            baseline.binary64_observed_strict_gate_decisions,
            candidate.binary64_observed_strict_gate_decisions,
            strict=True,
        )
    )
    decision_agreement = all(decision_agreement_by_node)
    strict_separation_by_node = tuple(
        bound < distance
        for bound, distance in zip(
            gate_rate_bounds,
            threshold_distances,
            strict=True,
        )
    )
    strict_separation = all(strict_separation_by_node)
    invariance = bool(decision_agreement and strict_separation)

    comparison = EventLocalZHIRHeldPressureComparison(
        comparison_kind=_COMPARISON_KIND,
        event_identity=baseline.event_identity,
        nodes=baseline.nodes,
        baseline_substeps=baseline.substeps,
        candidate_substeps=candidate.substeps,
        substep_counts_differ=True,
        exact_event_time=baseline.exact_event_time,
        event_time=baseline.event_time,
        exact_jump_placement_equal=bool(
            baseline.exact_jump_at_pre_flow_endpoint
            and candidate.exact_jump_at_pre_flow_endpoint
            and baseline.exact_event_time == candidate.exact_event_time
        ),
        binary64_jump_placement_equal=bool(
            baseline.binary64_jump_at_pre_flow_endpoint
            and candidate.binary64_jump_at_pre_flow_endpoint
            and structural_proof_signature(baseline.event_time)
            == structural_proof_signature(candidate.event_time)
        ),
        exact_duration=baseline.exact_duration,
        baseline_exact_endpoint_epi=baseline.exact_endpoint_epi,
        candidate_exact_endpoint_epi=candidate.exact_endpoint_epi,
        exact_endpoint_epi_difference=endpoint_difference,
        baseline_exact_rational_physical_secants=(
            baseline.exact_rational_physical_secants
        ),
        candidate_exact_rational_physical_secants=(
            candidate.exact_rational_physical_secants
        ),
        exact_rational_physical_secant_differences=rational_secant_differences,
        baseline_binary64_observed_gate_rates=(
            baseline.binary64_observed_gate_rates
        ),
        candidate_binary64_observed_gate_rates=(
            candidate.binary64_observed_gate_rates
        ),
        baseline_exact_binary64_observed_gate_rates=(
            baseline.exact_binary64_observed_gate_rates
        ),
        candidate_exact_binary64_observed_gate_rates=(
            candidate.exact_binary64_observed_gate_rates
        ),
        exact_binary64_gate_rate_differences=gate_rate_differences,
        exact_binary64_gate_rate_difference_bounds=gate_rate_bounds,
        xi=baseline.xi,
        exact_xi=baseline.exact_xi,
        baseline_exact_signed_threshold_margins=signed_margins,
        baseline_exact_threshold_distances=threshold_distances,
        baseline_observed_strict_gate_decisions=(
            baseline.binary64_observed_strict_gate_decisions
        ),
        candidate_observed_strict_gate_decisions=(
            candidate.binary64_observed_strict_gate_decisions
        ),
        observed_strict_gate_decisions_agree_by_node=decision_agreement_by_node,
        observed_strict_gate_decisions_agree=decision_agreement,
        strict_threshold_separation_by_node=strict_separation_by_node,
        strict_threshold_separation=strict_separation,
        _event_local_gate_invariance_certified=invariance,
    )
    return replace(comparison, _proof_stamp=_comparison_stamp(comparison))


def _binary64_secant(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    start_time: float,
    end_time: float,
    declared_duration: float,
    label: str,
) -> tuple[
    float,
    Fraction,
    tuple[float, ...],
    tuple[Fraction, ...],
    tuple[float, ...],
    tuple[Fraction, ...],
]:
    """Reproduce Mutation's subtraction/division order for one sample pair."""

    if len(left) != len(right):
        raise ValueError(f"{label} EPI endpoints require equal support")
    sample_interval = end_time - start_time
    if (
        not math.isfinite(sample_interval)
        or sample_interval <= 0.0
        or structural_proof_signature(sample_interval)
        != structural_proof_signature(declared_duration)
    ):
        raise ValueError(
            f"{label} timestamp subtraction must equal its declared duration"
        )
    deltas: list[float] = []
    rates: list[float] = []
    for left_value, right_value in zip(left, right, strict=True):
        delta = right_value - left_value
        rate = delta / sample_interval
        if not math.isfinite(delta) or not math.isfinite(rate):
            raise ValueError(f"{label} Mutation arithmetic must remain finite")
        deltas.append(delta)
        rates.append(rate)
    return (
        sample_interval,
        Fraction.from_float(sample_interval),
        tuple(deltas),
        tuple(Fraction.from_float(value) for value in deltas),
        tuple(rates),
        tuple(Fraction.from_float(value) for value in rates),
    )


def _exact_secants(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
    duration: Fraction,
) -> tuple[Fraction, ...]:
    if duration <= 0:
        raise ValueError("secant duration must be positive")
    return tuple(
        (right_value - left_value) / duration
        for left_value, right_value in zip(left, right, strict=True)
    )


def _physical_observation_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not EventLocalZHIRPhysicalPrejumpObservation:
        raise TypeError("physical observation must have the canonical result type")
    return (
        _PHYSICAL_OBSERVATION_PROOF_VERSION,
        tuple(
            (item.name, structural_proof_signature(getattr(value, item.name)))
            for item in fields(EventLocalZHIRPhysicalPrejumpObservation)
            if item.name != "_proof_stamp"
        ),
    )


@dataclass(frozen=True, slots=True)
class EventLocalZHIRPhysicalPrejumpObservation:
    """Sealed ZHIR evidence after a pressure-refreshed physical partition.

    Claims excluded by this observation and its scope are slotless read-only
    properties, so an instance cannot promote them directly.
    """

    partition_evidence: ExecutedPressureRefreshedFlowPartition = field(repr=False)
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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _physical_observation_stamp(self)
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        if not proof_stamps_are_identical(observed, expected):
            return False
        return bool(
            type(self.partition_evidence)
            is ExecutedPressureRefreshedFlowPartition
            and self.partition_evidence._proof_fields_are_intact()
        )

    @property
    def common_jump_execution_certified(self) -> bool:
        """Keep common jump execution outside offline coordinate pairing."""

        return False

    @property
    def u4_readiness_certified(self) -> bool:
        """Keep U4 readiness outside physical pre-jump evidence."""

        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        """Keep solver accuracy outside one finite physical partition."""

        return False

    @property
    def solver_order_certified(self) -> bool:
        """Keep numerical order outside one finite physical partition."""

        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        """Keep mesh convergence outside one finite physical partition."""

        return False

    @property
    def future_or_repeated_behavior_certified(self) -> bool:
        """Keep future and repeated behavior outside this observation."""

        return False

    @property
    def scope(self) -> str:
        """Describe the physical observation's fixed claim boundary."""

        return _PHYSICAL_OBSERVATION_SCOPE

    @property
    def physical_pre_jump_observation_certified(self) -> bool:
        """Whether the physical partition and derived gate evidence are intact."""

        return self._proof_fields_are_intact()


def observe_event_local_zhir_physical_prejump(
    partition_evidence: ExecutedPressureRefreshedFlowPartition,
    event: ScheduledOperatorEvent,
    *,
    xi: Real,
) -> EventLocalZHIRPhysicalPrejumpObservation:
    """Observe terminal-window and parent-horizon ZHIR rates separately."""

    if type(partition_evidence) is not ExecutedPressureRefreshedFlowPartition:
        raise TypeError(
            "partition_evidence must be an ExecutedPressureRefreshedFlowPartition"
        )
    if not partition_evidence.physical_pressure_reevaluated_partition_established:
        raise ValueError("physical partition evidence is unsealed, tampered, or stale")
    if type(event) is not ScheduledOperatorEvent:
        raise TypeError("event must be a ScheduledOperatorEvent")
    if event.glyph is not Glyph.ZHIR or event.operator_name != "mutation":
        raise ValueError("event must be the canonical scheduled ZHIR operator")

    partition = partition_evidence.partition
    parent = partition.parent_interval
    exact_jump_placement = bool(
        event.event_index == parent.index
        and event.exact_event_time == parent.exact_end_time
        and event.event_offset == parent.end_offset
    )
    binary_jump_placement = bool(
        structural_proof_signature(event.event_time)
        == structural_proof_signature(parent.end_time)
    )
    if not exact_jump_placement or not binary_jump_placement:
        raise ValueError("scheduled ZHIR is not at the physical parent endpoint")

    boundaries = partition_evidence.boundary_observations
    flows = partition_evidence.segment_flow_evidence
    if not boundaries or not flows:
        raise ValueError("physical pre-ZHIR partition must contain flow evidence")
    initial = boundaries[0].after
    terminal = boundaries[-1].after
    last_certificate = flows[-1].certificate
    if last_certificate is None:
        raise ValueError("terminal physical segment lacks endpoint evidence")
    terminal_left = last_certificate.left
    if not initial.nodes:
        raise ValueError("physical pre-ZHIR partition must contain at least one node")
    if (
        structural_proof_signature(initial.nodes)
        != structural_proof_signature(terminal.nodes)
        or structural_proof_signature(initial.nodes)
        != structural_proof_signature(terminal_left.nodes)
    ):
        raise ValueError("physical pre-ZHIR partition changed ordered node support")

    xi_value = _materialized_binary64(xi, "xi")
    if xi_value < 0.0:
        raise ValueError("xi must be nonnegative")
    exact_xi = Fraction.from_float(xi_value)
    last_segment = partition.segments[-1]
    (
        terminal_interval,
        exact_terminal_interval,
        terminal_deltas,
        exact_terminal_deltas,
        terminal_rates,
        exact_terminal_rates,
    ) = _binary64_secant(
        terminal_left.epi,
        terminal.epi,
        start_time=last_segment.start_time,
        end_time=last_segment.end_time,
        declared_duration=last_segment.duration,
        label="terminal physical segment",
    )
    (
        whole_interval,
        exact_whole_interval,
        whole_deltas,
        exact_whole_deltas,
        whole_rates,
        exact_whole_rates,
    ) = _binary64_secant(
        initial.epi,
        terminal.epi,
        start_time=parent.start_time,
        end_time=parent.end_time,
        declared_duration=parent.duration,
        label="whole parent interval",
    )

    terminal_rational = _exact_secants(
        terminal_left.exact_epi,
        terminal.exact_epi,
        last_segment.exact_duration,
    )
    whole_rational = _exact_secants(
        initial.exact_epi,
        terminal.exact_epi,
        parent.exact_duration,
    )
    trigger_certificates = tuple(
        certify_mutation_trigger(
            current_epi=current_epi,
            nu_f=nu_f,
            delta_nfr=pressure,
            xi=xi_value,
            epi_time_history=(
                (last_segment.start_time, previous_epi),
                (last_segment.end_time, current_epi),
            ),
        )
        for previous_epi, current_epi, nu_f, pressure in zip(
            terminal_left.epi,
            terminal.epi,
            terminal.nu_f,
            terminal.delta_nfr,
            strict=True,
        )
    )
    if any(
        certificate.observed_depi_dt is None
        or not certificate.evidence_valid
        or not certificate.physical_time_resolved
        for certificate in trigger_certificates
    ):
        raise RuntimeError("canonical Mutation rejected valid terminal evidence")
    trigger_rates = tuple(
        certificate.observed_depi_dt for certificate in trigger_certificates
    )
    if any(rate is None for rate in trigger_rates):
        raise RuntimeError("canonical Mutation omitted a terminal observed rate")
    typed_trigger_rates = tuple(float(rate) for rate in trigger_rates)
    if not _binary64_vectors_are_identical(typed_trigger_rates, terminal_rates):
        raise RuntimeError("terminal observation diverged from canonical Mutation")

    terminal_window_equals_parent = bool(
        last_segment.exact_start_time == parent.exact_start_time
        and last_segment.exact_duration == parent.exact_duration
        and structural_proof_signature(last_segment.start_time)
        == structural_proof_signature(parent.start_time)
        and structural_proof_signature(last_segment.duration)
        == structural_proof_signature(parent.duration)
    )
    observation = EventLocalZHIRPhysicalPrejumpObservation(
        partition_evidence=partition_evidence,
        event_identity=(
            event.event_index,
            event.cycle_index,
            event.word_position,
            event.operator_name,
            event.glyph.value,
        ),
        parent_interval_index=parent.index,
        nodes=initial.nodes,
        segment_count=partition.segment_count,
        segment_durations=partition.segment_durations,
        exact_segment_durations=tuple(
            segment.exact_duration for segment in partition.segments
        ),
        interval_start_time=parent.start_time,
        interval_end_time=parent.end_time,
        duration=parent.duration,
        exact_interval_start_time=parent.exact_start_time,
        exact_interval_end_time=parent.exact_end_time,
        exact_duration=parent.exact_duration,
        event_time=event.event_time,
        exact_event_time=event.exact_event_time,
        event_offset=event.event_offset,
        binary64_initial_epi=initial.epi,
        exact_initial_epi=initial.exact_epi,
        binary64_terminal_window_initial_epi=terminal_left.epi,
        exact_terminal_window_initial_epi=terminal_left.exact_epi,
        binary64_terminal_epi=terminal.epi,
        exact_terminal_epi=terminal.exact_epi,
        binary64_initial_capacity=initial.nu_f,
        exact_initial_capacity=initial.exact_nu_f,
        binary64_initial_pressure=initial.delta_nfr,
        exact_initial_pressure=initial.exact_delta_nfr,
        exact_initial_conductance=initial.conductance,
        binary64_terminal_capacity=terminal.nu_f,
        exact_terminal_capacity=terminal.exact_nu_f,
        binary64_terminal_pressure=terminal.delta_nfr,
        exact_terminal_pressure=terminal.exact_delta_nfr,
        exact_terminal_conductance=terminal.conductance,
        xi=xi_value,
        exact_xi=exact_xi,
        terminal_sample_start_time=last_segment.start_time,
        exact_terminal_sample_start_time=last_segment.exact_start_time,
        terminal_sample_interval=terminal_interval,
        exact_terminal_sample_interval=exact_terminal_interval,
        terminal_binary64_epi_deltas=terminal_deltas,
        terminal_exact_binary64_epi_deltas=exact_terminal_deltas,
        terminal_binary64_observed_gate_rates=terminal_rates,
        terminal_exact_binary64_observed_gate_rates=exact_terminal_rates,
        terminal_exact_rational_physical_secants=terminal_rational,
        terminal_exact_rational_minus_binary64_gate_rates=tuple(
            rational - runtime
            for rational, runtime in zip(
                terminal_rational,
                exact_terminal_rates,
                strict=True,
            )
        ),
        terminal_binary64_predicted_rates=tuple(
            certificate.predicted_depi_dt for certificate in trigger_certificates
        ),
        terminal_exact_binary64_predicted_rates=tuple(
            Fraction.from_float(certificate.predicted_depi_dt)
            for certificate in trigger_certificates
        ),
        terminal_capacity_active_by_node=tuple(
            certificate.capacity_active for certificate in trigger_certificates
        ),
        terminal_observed_strict_gate_decisions=tuple(
            certificate.threshold_gate_satisfied
            for certificate in trigger_certificates
        ),
        whole_parent_sample_interval=whole_interval,
        exact_whole_parent_sample_interval=exact_whole_interval,
        whole_parent_binary64_epi_deltas=whole_deltas,
        whole_parent_exact_binary64_epi_deltas=exact_whole_deltas,
        whole_parent_binary64_gate_rates=whole_rates,
        whole_parent_exact_binary64_gate_rates=exact_whole_rates,
        whole_parent_exact_rational_physical_secants=whole_rational,
        whole_parent_exact_rational_minus_binary64_gate_rates=tuple(
            rational - runtime
            for rational, runtime in zip(
                whole_rational,
                exact_whole_rates,
                strict=True,
            )
        ),
        whole_parent_strict_gate_decisions=tuple(
            rate > xi_value for rate in whole_rates
        ),
        terminal_window_equals_parent_horizon=terminal_window_equals_parent,
        exact_jump_at_parent_endpoint=exact_jump_placement,
        binary64_jump_at_parent_endpoint=binary_jump_placement,
    )
    return replace(
        observation,
        _proof_stamp=_physical_observation_stamp(observation),
    )


def _executed_physical_observation_stamp(value: Any) -> tuple[Any, ...]:
    """Seal the common-execution links and their derived physical observation."""

    if type(value) is not ExecutedEventLocalZHIRPhysicalPrejumpObservation:
        raise TypeError(
            "executed physical observation must have the canonical result type"
        )
    execution = object.__getattribute__(value, "execution_result")
    scheduled = object.__getattribute__(value, "scheduled_event")
    executed = object.__getattribute__(value, "executed_event")
    partition = object.__getattribute__(value, "partition_evidence")
    stage = object.__getattribute__(value, "glyph_stage")
    physical = object.__getattribute__(value, "physical_observation")
    nodes = object.__getattribute__(value, "nodes")
    decisions = object.__getattribute__(value, "mutation_decision_observations")
    certificates = object.__getattribute__(value, "trigger_certificates")
    return (
        _EXECUTED_PHYSICAL_OBSERVATION_PROOF_VERSION,
        object.__getattribute__(value, "event_index"),
        (id(execution), object.__getattribute__(execution, "_proof_stamp")),
        id(scheduled),
        (id(executed), object.__getattribute__(executed, "_proof_stamp")),
        (id(partition), object.__getattribute__(partition, "_proof_stamp")),
        (id(stage), object.__getattribute__(stage, "_proof_stamp")),
        (id(physical), object.__getattribute__(physical, "_proof_stamp")),
        (id(nodes), tuple(id(node) for node in nodes)),
        (
            id(decisions),
            tuple(
                (
                    id(decision),
                    object.__getattribute__(decision, "_proof_stamp"),
                    id(decision.trigger_certificate),
                )
                for decision in decisions
            ),
        ),
        (id(certificates), tuple(id(certificate) for certificate in certificates)),
    )


def _single_event_index_match(
    values: tuple[Any, ...],
    event_index: int,
    *,
    label: str,
) -> Any:
    """Select exactly one canonical record without relying on tuple position."""

    matches = tuple(
        value
        for value in values
        if object.__getattribute__(value, "event_index") == event_index
    )
    if not matches:
        raise ValueError(f"execution has no {label} for event_index={event_index}")
    if len(matches) != 1:
        raise ValueError(
            f"execution has duplicate {label} records for event_index={event_index}"
        )
    return matches[0]


def _single_partition_match(
    values: tuple[ExecutedPressureRefreshedFlowPartition, ...],
    event_index: int,
) -> ExecutedPressureRefreshedFlowPartition:
    """Select the unique physical parent interval immediately before an event."""

    matches = tuple(
        value
        for value in values
        if value.partition.parent_interval.index == event_index
    )
    if not matches:
        raise ValueError(
            "ZHIR event has no immediately preceding physical flow partition"
        )
    if len(matches) != 1:
        raise ValueError(
            "ZHIR event has duplicate immediately preceding physical flow "
            "partitions"
        )
    return matches[0]


def _single_stage_match(
    values: tuple[ExecutedGlyphStage, ...],
    event_index: int,
) -> ExecutedGlyphStage:
    """Select exactly one executor-owned stage for the requested event."""

    matches = tuple(
        value
        for value in values
        if value.event.event_index == event_index
    )
    if not matches:
        raise ValueError(f"execution has no glyph stage for event_index={event_index}")
    if len(matches) != 1:
        raise ValueError(
            f"execution has duplicate glyph stages for event_index={event_index}"
        )
    return matches[0]


def _mutation_trigger_value_signature(
    value: MutationTriggerCertificate,
) -> tuple[Any, ...]:
    """Sign one trigger by value while retaining binary64 distinctions."""

    if type(value) is not MutationTriggerCertificate:
        raise TypeError("trigger must be a MutationTriggerCertificate")
    evidence = value.evidence
    if evidence is not None and type(evidence) is not MutationTriggerEvidence:
        raise TypeError("trigger evidence must have the canonical type")
    evidence_signature = (
        None
        if evidence is None
        else tuple(
            (
                item.name,
                structural_proof_signature(
                    object.__getattribute__(evidence, item.name)
                ),
            )
            for item in fields(MutationTriggerEvidence)
        )
    )
    return tuple(
        (
            item.name,
            (
                evidence_signature
                if item.name == "evidence"
                else structural_proof_signature(
                    object.__getattribute__(value, item.name)
                )
            ),
        )
        for item in fields(MutationTriggerCertificate)
    )


def _trigger_matches_physical_terminal_node(
    certificate: MutationTriggerCertificate,
    physical: EventLocalZHIRPhysicalPrejumpObservation,
    index: int,
) -> bool:
    """Rebuild one terminal-window trigger and compare every certificate fact."""

    if type(certificate) is not MutationTriggerCertificate:
        return False
    try:
        expected = certify_mutation_trigger(
            current_epi=physical.binary64_terminal_epi[index],
            nu_f=physical.binary64_terminal_capacity[index],
            delta_nfr=physical.binary64_terminal_pressure[index],
            xi=physical.xi,
            epi_time_history=(
                (
                    physical.terminal_sample_start_time,
                    physical.binary64_terminal_window_initial_epi[index],
                ),
                (
                    physical.interval_end_time,
                    physical.binary64_terminal_epi[index],
                ),
            ),
        )
        return bool(
            _mutation_trigger_value_signature(certificate)
            == _mutation_trigger_value_signature(expected)
        )
    except (IndexError, TypeError, ValueError, OverflowError):
        return False


def _executed_physical_observation_links_are_intact(value: Any) -> bool:
    """Validate every identity and ordering link back to one sealed execution."""

    if type(value) is not ExecutedEventLocalZHIRPhysicalPrejumpObservation:
        return False
    try:
        execution = object.__getattribute__(value, "execution_result")
        event_index = object.__getattribute__(value, "event_index")
        scheduled_event = object.__getattribute__(value, "scheduled_event")
        executed_event = object.__getattribute__(value, "executed_event")
        partition = object.__getattribute__(value, "partition_evidence")
        stage = object.__getattribute__(value, "glyph_stage")
        physical = object.__getattribute__(value, "physical_observation")
        nodes = object.__getattribute__(value, "nodes")
        decisions = object.__getattribute__(
            value, "mutation_decision_observations"
        )
        certificates = object.__getattribute__(value, "trigger_certificates")
        if (
            type(execution) is not OperatorEventExecutionResult
            or not execution._proof_fields_are_intact()
            or type(event_index) is not int
            or event_index < 0
            or not execution.stage_certification_requested
            or type(scheduled_event) is not ScheduledOperatorEvent
            or type(executed_event) is not ExecutedOperatorEvent
            or type(partition) is not ExecutedPressureRefreshedFlowPartition
            or type(stage) is not ExecutedGlyphStage
            or type(physical) is not EventLocalZHIRPhysicalPrejumpObservation
            or type(nodes) is not tuple
            or type(decisions) is not tuple
            or type(certificates) is not tuple
            or not physical.physical_pre_jump_observation_certified
            or not stage._proof_fields_are_intact()
        ):
            return False

        selected_scheduled = _single_event_index_match(
            execution.schedule.events,
            event_index,
            label="scheduled event",
        )
        selected_executed = _single_event_index_match(
            execution.events,
            event_index,
            label="committed event",
        )
        selected_partition = _single_partition_match(
            execution.physical_flow_partition_evidence,
            event_index,
        )
        selected_stage = _single_stage_match(
            execution.glyph_stage_evidence,
            event_index,
        )
        if (
            scheduled_event is not selected_scheduled
            or executed_event is not selected_executed
            or partition is not selected_partition
            or stage is not selected_stage
            or stage.event is not executed_event
            or physical.partition_evidence is not partition
            or nodes is not execution.target_nodes
            or decisions is not stage.mutation_decision_observations
            or scheduled_event.glyph is not Glyph.ZHIR
            or scheduled_event.operator_name != "mutation"
            or executed_event.glyph is not Glyph.ZHIR
            or executed_event.operator_name != "mutation"
            or physical.parent_interval_index != event_index
            or stage.pre_interval_index != event_index
            or not partition.segment_flow_evidence
            or stage.pre_flow_evidence
            is not partition.segment_flow_evidence[-1]
            or stage.pre_flow_endpoint_continuous is not True
            or len(nodes) == 0
            or len(decisions) != len(nodes)
            or len(certificates) != len(nodes)
            or len(physical.nodes) != len(nodes)
        ):
            return False

        expected_identity = (
            scheduled_event.event_index,
            scheduled_event.cycle_index,
            scheduled_event.word_position,
            scheduled_event.operator_name,
            scheduled_event.glyph.value,
        )
        if physical.event_identity != expected_identity:
            return False
        for index, (node, physical_node, decision, certificate) in enumerate(
            zip(
                nodes,
                physical.nodes,
                decisions,
                certificates,
                strict=True,
            )
        ):
            if (
                physical_node is not node
                or type(decision) is not MutationStageDecisionObservation
                or decision is not stage.mutation_decision_observations[index]
                or decision.target_index != index
                or decision.node is not node
                or not decision._proof_fields_are_intact()
                or certificate is not decision.trigger_certificate
                or not _trigger_matches_physical_terminal_node(
                    certificate,
                    physical,
                    index,
                )
            ):
                return False
    except Exception:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ExecutedEventLocalZHIRPhysicalPrejumpObservation:
    """One physical pre-ZHIR window bound to its actual committed jump.

    The object retains the executor result itself and verifies all nested
    records by identity. Its positive claim is therefore limited to common
    execution provenance for this finite partition and this one ZHIR stage.
    """

    execution_result: OperatorEventExecutionResult = field(repr=False)
    event_index: int
    scheduled_event: ScheduledOperatorEvent = field(repr=False)
    executed_event: ExecutedOperatorEvent = field(repr=False)
    partition_evidence: ExecutedPressureRefreshedFlowPartition = field(repr=False)
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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _executed_physical_observation_stamp(self)
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        if not proof_stamps_are_identical(observed, expected):
            return False
        try:
            return _executed_physical_observation_links_are_intact(self)
        except BaseException:
            return False

    @property
    def common_execution_provenance_certified(self) -> bool:
        """Whether the physical partition and ZHIR decisions share one run."""

        return self._proof_fields_are_intact()

    @property
    def solver_accuracy_certified(self) -> bool:
        """Keep solver accuracy outside one finite common-execution trace."""

        return False

    @property
    def solver_order_certified(self) -> bool:
        """Keep solver order outside one finite common-execution trace."""

        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        """Keep mesh convergence outside one executed partition."""

        return False

    @property
    def u4_readiness_certified(self) -> bool:
        """Keep general U4 readiness outside one accepted ZHIR decision."""

        return False

    @property
    def adaptive_policy_certified(self) -> bool:
        """Keep adaptive policy outside this fixed schedule execution."""

        return False

    @property
    def future_or_repeated_behavior_certified(self) -> bool:
        """Keep future and repeated behavior outside this finite trace."""

        return False

    @property
    def scope(self) -> str:
        """Describe the common-execution observation's fixed claim boundary."""

        return _EXECUTED_PHYSICAL_OBSERVATION_SCOPE


def observe_executed_event_local_zhir_physical_prejump(
    execution_result: OperatorEventExecutionResult,
    *,
    event_index: int,
) -> ExecutedEventLocalZHIRPhysicalPrejumpObservation:
    """Bind terminal physical ZHIR evidence to one actual schedule execution."""

    if type(execution_result) is not OperatorEventExecutionResult:
        raise TypeError(
            "execution_result must be an OperatorEventExecutionResult"
        )
    if type(event_index) is not int:
        raise TypeError("event_index must be an int")
    if event_index < 0:
        raise ValueError("event_index must be a nonnegative int")
    if not execution_result._proof_fields_are_intact():
        raise ValueError("execution result is unsealed, tampered, or stale")
    if not execution_result.stage_certification_requested:
        raise ValueError("execution must include stage certificates")

    scheduled_event = _single_event_index_match(
        execution_result.schedule.events,
        event_index,
        label="scheduled event",
    )
    executed_event = _single_event_index_match(
        execution_result.events,
        event_index,
        label="committed event",
    )
    if (
        scheduled_event.glyph is not Glyph.ZHIR
        or scheduled_event.operator_name != "mutation"
        or executed_event.glyph is not Glyph.ZHIR
        or executed_event.operator_name != "mutation"
    ):
        raise ValueError("event_index must select a committed canonical ZHIR event")

    partition = _single_partition_match(
        execution_result.physical_flow_partition_evidence,
        event_index,
    )
    stage = _single_stage_match(
        execution_result.glyph_stage_evidence,
        event_index,
    )
    decisions = stage.mutation_decision_observations
    if not decisions:
        raise ValueError("ZHIR stage lacks Mutation decision observations")
    trigger_certificates = tuple(
        decision.trigger_certificate for decision in decisions
    )
    xi = trigger_certificates[0].xi
    physical = observe_event_local_zhir_physical_prejump(
        partition,
        scheduled_event,
        xi=xi,
    )
    candidate = ExecutedEventLocalZHIRPhysicalPrejumpObservation(
        execution_result=execution_result,
        event_index=event_index,
        scheduled_event=scheduled_event,
        executed_event=executed_event,
        partition_evidence=partition,
        glyph_stage=stage,
        physical_observation=physical,
        nodes=execution_result.target_nodes,
        mutation_decision_observations=decisions,
        trigger_certificates=trigger_certificates,
    )
    if not _executed_physical_observation_links_are_intact(candidate):
        raise ValueError(
            "execution does not preserve the physical partition-to-ZHIR links"
        )
    sealed = replace(
        candidate,
        _proof_stamp=_executed_physical_observation_stamp(candidate),
    )
    if not sealed.common_execution_provenance_certified:
        raise RuntimeError("failed to seal common physical ZHIR execution evidence")
    return sealed


@dataclass(frozen=True, slots=True)
class _ModalComparisonFacts:
    applicable: bool
    abstention_reason: str | None
    held_interval_duration: float | None
    common_decay_rates: tuple[float, ...] | None
    held_interval_multipliers: tuple[float, ...] | None
    held_interval_maximum_factor: float | None
    held_interval_stable: bool | None
    physical_segment_decisions: tuple[bool, ...]
    refined_composite_multipliers: tuple[float, ...] | None
    refined_composite_maximum_factor: float | None
    refined_composite_stable: bool | None
    stability_decisions_agree: bool | None


def _modal_abstention(
    reason: str,
    *,
    physical_segment_decisions: tuple[bool, ...] = (),
) -> _ModalComparisonFacts:
    return _ModalComparisonFacts(
        False,
        reason,
        None,
        None,
        None,
        None,
        None,
        physical_segment_decisions,
        None,
        None,
        None,
        None,
    )


def _physical_segments_share_baseline_generator(
    baseline: EventLocalZHIRPrejumpObservation,
    physical: EventLocalZHIRPhysicalPrejumpObservation,
) -> bool:
    """Whether every observed segment uses the baseline's fixed generator."""

    evidence = physical.partition_evidence
    expected_nodes = structural_proof_signature(baseline.nodes)

    def _snapshot_matches(snapshot: Any) -> bool:
        return bool(
            structural_proof_signature(snapshot.nodes) == expected_nodes
            and snapshot.exact_nu_f == baseline.exact_capacity
            and _binary64_vectors_are_identical(
                snapshot.nu_f,
                baseline.binary64_capacity,
            )
            and snapshot.conductance == baseline.exact_conductance
        )

    segment_starts = evidence.boundary_observations[:-1]
    if len(segment_starts) != len(evidence.segment_flow_evidence):
        return False
    if any(not _snapshot_matches(boundary.after) for boundary in segment_starts):
        return False
    for flow in evidence.segment_flow_evidence:
        certificate = flow.certificate
        if (
            certificate is None
            or not certificate.capacity_unchanged
            or not certificate.fixed_conductance
            or not _snapshot_matches(certificate.left)
            or not _snapshot_matches(certificate.right)
        ):
            return False
    return True


def _modal_comparison_facts(
    baseline: EventLocalZHIRPrejumpObservation,
    physical: EventLocalZHIRPhysicalPrejumpObservation,
) -> _ModalComparisonFacts:
    """Compare the held outer map with the physical segment product."""

    partition_evidence = physical.partition_evidence
    candidate_observations = partition_evidence.modal_observations
    if not partition_evidence.all_segment_modal_diagnostics_applicable:
        return _modal_abstention(
            "physical_segment_modal_diagnostics_unavailable"
        )
    candidate_decisions = tuple(
        bool(observation.is_euler_stable)
        for observation in candidate_observations
    )
    if not _physical_segments_share_baseline_generator(baseline, physical):
        return _modal_abstention(
            "physical_modal_generator_changed",
            physical_segment_decisions=candidate_decisions,
        )
    if not partition_evidence.all_segment_binary64_replays_identified:
        return _modal_abstention(
            "physical_segment_held_pressure_runtime_unidentified",
            physical_segment_decisions=candidate_decisions,
        )
    first_modal = candidate_observations[0]
    target_fraction = first_modal.target_fraction
    spectral_tolerance = first_modal.spectral_relative_tolerance
    common_rates = first_modal.decay_rates
    if (
        target_fraction is None
        or spectral_tolerance is None
        or common_rates is None
        or not common_rates
    ):
        return _modal_abstention(
            "physical_modal_configuration_unavailable",
            physical_segment_decisions=candidate_decisions,
        )
    if any(
        structural_proof_signature(observation.target_fraction)
        != structural_proof_signature(target_fraction)
        or structural_proof_signature(observation.spectral_relative_tolerance)
        != structural_proof_signature(spectral_tolerance)
        for observation in candidate_observations
    ):
        return _modal_abstention(
            "physical_modal_configuration_changed",
            physical_segment_decisions=candidate_decisions,
        )
    if any(
        observation.decay_rates is None
        or not _binary64_vectors_are_identical(
            common_rates,
            observation.decay_rates,
        )
        for observation in candidate_observations
    ):
        return _modal_abstention(
            "physical_modal_decay_rates_changed",
            physical_segment_decisions=candidate_decisions,
        )

    dimension = len(baseline.nodes)
    matrix = baseline.exact_conductance
    if (
        len(matrix) != dimension
        or any(len(row) != dimension for row in matrix)
        or any(matrix[i][i] != 0 for i in range(dimension))
        or any(
            matrix[i][j] < 0 or matrix[i][j] != matrix[j][i]
            for i in range(dimension)
            for j in range(dimension)
        )
    ):
        return _modal_abstention(
            "baseline_transport_is_not_simple_symmetric_nonnegative",
            physical_segment_decisions=candidate_decisions,
        )

    graph = nx.Graph()
    for node, epi, capacity, pressure in zip(
        baseline.nodes,
        baseline.binary64_initial_epi,
        baseline.binary64_capacity,
        baseline.binary64_pressure,
        strict=True,
    ):
        graph.add_node(
            node,
            EPI=epi,
            nu_f=capacity,
            delta_nfr=pressure,
        )
    for left_index in range(dimension):
        for right_index in range(left_index + 1, dimension):
            weight = float(matrix[left_index][right_index])
            if weight != 0.0:
                graph.add_edge(
                    baseline.nodes[left_index],
                    baseline.nodes[right_index],
                    weight=weight,
                )
    try:
        snapshot = capture_nodal_flow_state(graph)
        if not _binary64_vectors_are_identical(
            baseline.binary64_pressure,
            snapshot.binary64_pure_epi_pressure,
        ):
            raise ValueError("baseline pressure is not the binary64 pure-EPI channel")
        diagnostic = diagnose_euler_relaxation_window(
            graph,
            dt=baseline.duration,
            target_fraction=target_fraction,
            tolerance=spectral_tolerance,
        )
    except (TypeError, ValueError, OverflowError, nx.NetworkXException) as exc:
        return _modal_abstention(
            f"baseline_modal_diagnostic_unavailable:{type(exc).__name__}",
            physical_segment_decisions=candidate_decisions,
        )

    baseline_rates = tuple(float(value) for value in diagnostic.decay_rates)
    if not _binary64_vectors_are_identical(baseline_rates, common_rates):
        return _modal_abstention(
            "baseline_and_physical_modal_decay_rates_differ",
            physical_segment_decisions=candidate_decisions,
        )

    multiplier_rows: list[tuple[float, ...]] = []
    for observation in candidate_observations:
        multipliers = observation.modal_multipliers
        if multipliers is None or len(multipliers) != len(common_rates):
            return _modal_abstention(
                "physical_modal_multiplier_dimension_changed",
                physical_segment_decisions=candidate_decisions,
            )
        multiplier_rows.append(multipliers)
    composite: list[float] = []
    for mode_index in range(len(common_rates)):
        gain = 1.0
        for multipliers in multiplier_rows:
            gain *= multipliers[mode_index]
        if not math.isfinite(gain):
            return _modal_abstention(
                "physical_modal_composite_is_nonfinite",
                physical_segment_decisions=candidate_decisions,
            )
        composite.append(gain)
    composite_multipliers = tuple(composite)
    composite_maximum = max(abs(value) for value in composite_multipliers)
    composite_stable = composite_maximum < 1.0
    held_stable = diagnostic.is_euler_stable
    return _ModalComparisonFacts(
        True,
        None,
        baseline.duration,
        baseline_rates,
        tuple(float(value) for value in diagnostic.modal_multipliers),
        diagnostic.maximum_modal_factor,
        held_stable,
        candidate_decisions,
        composite_multipliers,
        composite_maximum,
        composite_stable,
        held_stable == composite_stable,
    )


def _physical_comparison_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not EventLocalZHIRPhysicalRefinementComparison:
        raise TypeError("physical comparison must have the canonical result type")
    return (
        _PHYSICAL_COMPARISON_PROOF_VERSION,
        tuple(
            (item.name, structural_proof_signature(getattr(value, item.name)))
            for item in fields(EventLocalZHIRPhysicalRefinementComparison)
            if item.name != "_proof_stamp"
        ),
    )


@dataclass(frozen=True, slots=True)
class EventLocalZHIRPhysicalRefinementComparison:
    """Sealed held-pressure versus physical-refinement ZHIR comparison.

    Claims excluded by this comparison and its scope are slotless read-only
    properties, so an instance cannot promote them directly.
    """

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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _physical_comparison_stamp(self)
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        if not proof_stamps_are_identical(observed, expected):
            return False
        try:
            return bool(
                type(self.baseline_observation)
                is EventLocalZHIRPrejumpObservation
                and self.baseline_observation._proof_fields_are_intact()
                and type(self.physical_observation)
                is EventLocalZHIRPhysicalPrejumpObservation
                and self.physical_observation._proof_fields_are_intact()
            )
        except BaseException:
            return False

    @property
    def modal_equivalence_certified(self) -> bool:
        """Keep modal-map equivalence outside stability-decision evidence."""

        return False

    @property
    def common_execution_provenance_certified(self) -> bool:
        """Keep common execution provenance outside this offline comparison."""

        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        """Keep solver accuracy outside this finite comparison."""

        return False

    @property
    def solver_order_certified(self) -> bool:
        """Keep numerical order outside this finite comparison."""

        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        """Keep mesh convergence outside this finite comparison."""

        return False

    @property
    def future_or_repeated_behavior_certified(self) -> bool:
        """Keep future and repeated behavior outside this comparison."""

        return False

    @property
    def u4_readiness_certified(self) -> bool:
        """Keep U4 readiness outside this finite comparison."""

        return False

    @property
    def adaptive_policy_certified(self) -> bool:
        """Keep adaptive policy outside this finite comparison."""

        return False

    @property
    def scope(self) -> str:
        """Describe the physical comparison's fixed claim boundary."""

        return _PHYSICAL_COMPARISON_SCOPE

    @property
    def fixed_horizon_gate_agreement_certified(self) -> bool:
        """Whether baseline and offline whole-parent decisions safely agree."""

        return bool(
            self._proof_fields_are_intact()
            and self._fixed_horizon_gate_agreement_certified
        )

    @property
    def actual_terminal_gate_invariance_certified(self) -> bool:
        """Whether equal sampling windows support the actual gate claim."""

        return bool(
            self._proof_fields_are_intact()
            and self._actual_terminal_gate_invariance_certified
        )


def _require_physical_comparison_inputs(
    baseline: Any,
    physical: Any,
) -> tuple[
    EventLocalZHIRPrejumpObservation,
    EventLocalZHIRPhysicalPrejumpObservation,
]:
    if type(baseline) is not EventLocalZHIRPrejumpObservation:
        raise TypeError("baseline must be an EventLocalZHIRPrejumpObservation")
    if not baseline._proof_fields_are_intact():
        raise ValueError("baseline observation is tampered or stale")
    if type(physical) is not EventLocalZHIRPhysicalPrejumpObservation:
        raise TypeError(
            "physical must be an EventLocalZHIRPhysicalPrejumpObservation"
        )
    if not physical._proof_fields_are_intact():
        raise ValueError("physical observation is tampered or stale")

    requirements = (
        ("event_identity", baseline.event_identity == physical.event_identity),
        (
            "parent_interval_index",
            baseline.flow_interval_index == physical.parent_interval_index,
        ),
        (
            "exact_interval_start_time",
            baseline.exact_interval_start_time
            == physical.exact_interval_start_time,
        ),
        (
            "exact_interval_end_time",
            baseline.exact_interval_end_time == physical.exact_interval_end_time,
        ),
        ("exact_duration", baseline.exact_duration == physical.exact_duration),
        (
            "binary64_interval_start_time",
            structural_proof_signature(baseline.interval_start_time)
            == structural_proof_signature(physical.interval_start_time),
        ),
        (
            "binary64_interval_end_time",
            structural_proof_signature(baseline.interval_end_time)
            == structural_proof_signature(physical.interval_end_time),
        ),
        (
            "binary64_duration",
            structural_proof_signature(baseline.duration)
            == structural_proof_signature(physical.duration),
        ),
        (
            "node_support",
            structural_proof_signature(baseline.nodes)
            == structural_proof_signature(physical.nodes),
        ),
        ("exact_initial_epi", baseline.exact_initial_epi == physical.exact_initial_epi),
        (
            "binary64_initial_epi",
            _binary64_vectors_are_identical(
                baseline.binary64_initial_epi,
                physical.binary64_initial_epi,
            ),
        ),
        (
            "exact_initial_capacity",
            baseline.exact_capacity == physical.exact_initial_capacity,
        ),
        (
            "binary64_initial_capacity",
            _binary64_vectors_are_identical(
                baseline.binary64_capacity,
                physical.binary64_initial_capacity,
            ),
        ),
        (
            "exact_initial_pressure",
            baseline.exact_pressure == physical.exact_initial_pressure,
        ),
        (
            "binary64_initial_pressure",
            _binary64_vectors_are_identical(
                baseline.binary64_pressure,
                physical.binary64_initial_pressure,
            ),
        ),
        (
            "exact_initial_conductance",
            baseline.exact_conductance == physical.exact_initial_conductance,
        ),
        ("exact_xi", baseline.exact_xi == physical.exact_xi),
        (
            "binary64_xi",
            structural_proof_signature(baseline.xi)
            == structural_proof_signature(physical.xi),
        ),
    )
    failed = tuple(name for name, passed in requirements if not passed)
    if failed:
        raise ValueError(
            "observations are outside the physical-refinement comparison "
            f"domain: {', '.join(failed)}"
        )
    return baseline, physical


def compare_event_local_zhir_physical_refinement(
    baseline: EventLocalZHIRPrejumpObservation,
    physical: EventLocalZHIRPhysicalPrejumpObservation,
) -> EventLocalZHIRPhysicalRefinementComparison:
    """Compare held pressure with an explicit pressure-refreshed partition."""

    baseline, physical = _require_physical_comparison_inputs(baseline, physical)
    whole_differences = _difference(
        physical.whole_parent_exact_binary64_gate_rates,
        baseline.exact_binary64_observed_gate_rates,
    )
    whole_bounds = tuple(abs(value) for value in whole_differences)
    margins = tuple(
        rate - baseline.exact_xi
        for rate in baseline.exact_binary64_observed_gate_rates
    )
    distances = tuple(abs(value) for value in margins)
    whole_agreements = tuple(
        left == right
        for left, right in zip(
            baseline.binary64_observed_strict_gate_decisions,
            physical.whole_parent_strict_gate_decisions,
            strict=True,
        )
    )
    whole_separations = tuple(
        bound < distance
        for bound, distance in zip(whole_bounds, distances, strict=True)
    )
    whole_agreement = all(whole_agreements)
    whole_separation = all(whole_separations)
    fixed_horizon_claim = bool(whole_agreement and whole_separation)

    terminal_differences = _difference(
        physical.terminal_exact_binary64_observed_gate_rates,
        baseline.exact_binary64_observed_gate_rates,
    )
    terminal_bounds = tuple(abs(value) for value in terminal_differences)
    terminal_agreements = tuple(
        left == right
        for left, right in zip(
            baseline.binary64_observed_strict_gate_decisions,
            physical.terminal_observed_strict_gate_decisions,
            strict=True,
        )
    )
    terminal_separations = tuple(
        bound < distance
        for bound, distance in zip(terminal_bounds, distances, strict=True)
    )
    terminal_agreement = all(terminal_agreements)
    terminal_separation = all(terminal_separations)
    windows_equal = bool(
        physical.terminal_window_equals_parent_horizon
        and baseline.exact_interval_start_time
        == physical.exact_terminal_sample_start_time
        and baseline.exact_duration == physical.exact_terminal_sample_interval
        and structural_proof_signature(baseline.interval_start_time)
        == structural_proof_signature(physical.terminal_sample_start_time)
        and structural_proof_signature(baseline.binary64_sample_interval)
        == structural_proof_signature(physical.terminal_sample_interval)
    )
    actual_claim = bool(
        windows_equal and terminal_agreement and terminal_separation
    )

    modal = _modal_comparison_facts(baseline, physical)
    comparison = EventLocalZHIRPhysicalRefinementComparison(
        baseline_observation=baseline,
        physical_observation=physical,
        comparison_kind=_PHYSICAL_COMPARISON_KIND,
        event_identity=baseline.event_identity,
        nodes=baseline.nodes,
        baseline_runtime_substeps=baseline.substeps,
        physical_segment_count=physical.segment_count,
        exact_duration=baseline.exact_duration,
        xi=baseline.xi,
        exact_xi=baseline.exact_xi,
        baseline_binary64_gate_rates=baseline.binary64_observed_gate_rates,
        baseline_exact_binary64_gate_rates=(
            baseline.exact_binary64_observed_gate_rates
        ),
        baseline_observed_strict_gate_decisions=(
            baseline.binary64_observed_strict_gate_decisions
        ),
        physical_whole_parent_binary64_gate_rates=(
            physical.whole_parent_binary64_gate_rates
        ),
        physical_whole_parent_exact_binary64_gate_rates=(
            physical.whole_parent_exact_binary64_gate_rates
        ),
        physical_whole_parent_strict_gate_decisions=(
            physical.whole_parent_strict_gate_decisions
        ),
        exact_whole_parent_gate_rate_differences=whole_differences,
        exact_whole_parent_gate_rate_difference_bounds=whole_bounds,
        baseline_exact_signed_threshold_margins=margins,
        baseline_exact_threshold_distances=distances,
        whole_parent_gate_decisions_agree_by_node=whole_agreements,
        whole_parent_gate_decisions_agree=whole_agreement,
        whole_parent_strict_threshold_separation_by_node=whole_separations,
        whole_parent_strict_threshold_separation=whole_separation,
        _fixed_horizon_gate_agreement_certified=fixed_horizon_claim,
        physical_terminal_binary64_gate_rates=(
            physical.terminal_binary64_observed_gate_rates
        ),
        physical_terminal_exact_binary64_gate_rates=(
            physical.terminal_exact_binary64_observed_gate_rates
        ),
        physical_terminal_strict_gate_decisions=(
            physical.terminal_observed_strict_gate_decisions
        ),
        exact_terminal_gate_rate_differences=terminal_differences,
        exact_terminal_gate_rate_difference_bounds=terminal_bounds,
        terminal_gate_decisions_agree_by_node=terminal_agreements,
        terminal_gate_decisions_agree=terminal_agreement,
        terminal_strict_threshold_separation_by_node=terminal_separations,
        terminal_strict_threshold_separation=terminal_separation,
        observation_windows_equal=windows_equal,
        _actual_terminal_gate_invariance_certified=actual_claim,
        modal_comparison_applicable=modal.applicable,
        modal_abstention_reason=modal.abstention_reason,
        baseline_held_interval_duration=modal.held_interval_duration,
        common_binary64_modal_decay_rates=modal.common_decay_rates,
        baseline_held_interval_binary64_modal_multipliers=(
            modal.held_interval_multipliers
        ),
        baseline_held_interval_binary64_maximum_modal_factor=(
            modal.held_interval_maximum_factor
        ),
        baseline_held_interval_modal_stable=modal.held_interval_stable,
        physical_segment_modal_decisions=modal.physical_segment_decisions,
        physical_refined_composite_binary64_modal_multipliers=(
            modal.refined_composite_multipliers
        ),
        physical_refined_composite_binary64_maximum_modal_factor=(
            modal.refined_composite_maximum_factor
        ),
        physical_refined_composite_modal_stable=(
            modal.refined_composite_stable
        ),
        modal_stability_decisions_agree=modal.stability_decisions_agree,
    )
    return replace(
        comparison,
        _proof_stamp=_physical_comparison_stamp(comparison),
    )
