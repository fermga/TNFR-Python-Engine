"""Exact pre-jump observations for held-pressure ZHIR subdivisions.

The Mutation gate uses timestamped binary64 subtraction and division.  That
arithmetic need not equal the rational secant between exact representations of
the same endpoint floats.  This module records both without conflating them.
Internal integrator substeps retain interval-start pressure and therefore are
not physical diffusion refinements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Real
from typing import Any

from ..operators.event_runtime import ExecutedNodalFlowInterval
from ..operators.event_timing import ScheduledOperatorEvent
from ..types import Glyph
from ..utils._structural_signature import structural_proof_signature
from .runtime_flow_stability import NodalFlowIntervalCertificate

__all__ = (
    "EventLocalZHIRHeldPressureComparison",
    "EventLocalZHIRPrejumpObservation",
    "compare_event_local_zhir_held_pressure_subdivision",
    "observe_event_local_zhir_prejump",
)


_OBSERVATION_PROOF_VERSION = "event_local_zhir_prejump_observation_v2"
_COMPARISON_PROOF_VERSION = "event_local_zhir_held_pressure_comparison_v3"
_COMPARISON_KIND = "internal_held_pressure_subdivision"
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
            (item.name, structural_proof_signature(getattr(value, item.name)))
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

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _observation_stamp(self)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return type(self._proof_stamp) is tuple and self._proof_stamp == expected

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
    )
    return replace(observation, _proof_stamp=_observation_stamp(observation))


def _comparison_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not EventLocalZHIRHeldPressureComparison:
        raise TypeError("comparison must have the canonical result type")
    return (
        _COMPARISON_PROOF_VERSION,
        tuple(
            (item.name, structural_proof_signature(getattr(value, item.name)))
            for item in fields(EventLocalZHIRHeldPressureComparison)
            if item.name != "_proof_stamp"
        ),
    )


@dataclass(frozen=True, slots=True)
class EventLocalZHIRHeldPressureComparison:
    """Sealed comparison of two represented binary64 Mutation gate outcomes."""

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

    def _proof_fields_are_intact(self) -> bool:
        try:
            expected = _comparison_stamp(self)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return type(self._proof_stamp) is tuple and self._proof_stamp == expected

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
        ("exact_capacity", baseline.exact_capacity == candidate.exact_capacity),
        ("exact_pressure", baseline.exact_pressure == candidate.exact_pressure),
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
