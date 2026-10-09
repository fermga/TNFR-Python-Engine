"""Validated four-history observations on the complete three-C9 support.

Two shared prefixes and four carried suffixes use the existing source-box
Picard/Taylor kernel. Supplied full-state boxes are outer covers, not acquired
source certificates. No analytic response prediction or sensor law is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import MAX_ORDER
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics._validated_taylor import (
    ValidatedBoxTaylorStep,
    validated_box_taylor_step,
)
from ._sine_flow import _full_sine_field
from .phase_cycle_geometry import PhaseCycleGeometry, _derive
from .relational_observations import _interval, _ordered
from .relational_sine_class_mediation import _EDGES, _NODES

__all__ = ("SineClassFourHistoryReadout", "bound_sine_class_four_history_readout")

_LOGGER = logging.getLogger(__name__)
_MAX_STEPS = 4096


@dataclass(frozen=True)
class _ClassReadoutSegment:
    """A complete segment, a retained partial segment, or unattempted ancestry."""

    label: str
    parent_segment_index: int | None
    start_time: Q
    end_time: Q
    form_jump: Q
    pre_event_box: tuple[I, ...] | None
    initial_box: tuple[I, ...] | None
    steps: tuple[ValidatedBoxTaylorStep, ...]
    completed_time: Q | None
    completed_endpoint_box: tuple[I, ...] | None
    completed_receiver_increment_bounds: I | None
    final_state_bounds: tuple[I, ...] | None
    failed_initial_box: tuple[I, ...] | None
    failed_time: Q | None
    failed_tube: tuple[I, ...] | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class SineClassFourHistoryReadout:
    """Four endpoint observations of one common source family, or honest prefixes.

    Forms and continuous phase lifts contain 27 coordinates each. Every member
    of the same supplied box has four mathematical continuations; independent
    endpoint intervals conservatively discard correlations between them. Their
    Cartesian product does not claim joint realization of its corners.

    A reached family with component balls and zero sums may be covered by this
    box. Such containment and acquisition remain external premises; the whole
    Cartesian box need not consist of acquired states. Widths include source
    range, wrapping, arithmetic and truncation, and contain no sensor error.
    """

    initial_form_bounds: tuple[I, ...]
    initial_phase_bounds: tuple[I, ...]
    first_probe_amplitude: Q
    second_probe_amplitude: Q
    delay: Q
    total_duration: Q
    time_step: Q
    order: int
    max_steps: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    source_box: tuple[I, ...]
    initial_receiver_form_bounds: I
    history_event_amplitudes: tuple[tuple[Q, Q], ...]
    segments: tuple[_ClassReadoutSegment, ...]
    planned_unique_step_count: int
    attempted_step_count: int
    completed_step_count: int
    completed_segment_count: int
    completed_history_readout_bounds: tuple[tuple[str, I], ...]
    endpoint_readout_bounds: tuple[I, ...] | None
    raw_endpoint_mixed_bounds: I | None
    suffix_receiver_increment_bounds: tuple[I, ...] | None
    mixed_readout_bounds: I | None
    failed_segment_index: int | None
    unattempted_segment_indices: tuple[int, ...]
    status: str
    unavailable_reasons: tuple[str, ...]
    history_order: tuple[str, ...] = ("neither", "first_only", "second_only", "both")
    history_segment_indices: tuple[tuple[int, int], ...] = (
        (0, 2),
        (1, 3),
        (0, 4),
        (1, 5),
    )
    mixed_readout_coefficients: tuple[int, ...] = (1, -1, -1, 1)
    donor_node: int = 4
    receiver_node: int = 22
    capacity: tuple[Q, ...] = (Q(1),) * 27
    state_order: tuple[str, ...] = tuple(f"x_{i}" for i in _NODES) + tuple(
        f"theta_{i}" for i in _NODES
    )
    clock: str = "tau=e*t; e=1023/1024"
    arithmetic_method: str = INTERVAL_METHOD
    method: str = "shared_prefix_full54_source_box_Picard_Taylor_v1"
    scope: tuple[str, ...] = (
        "one_primitive_full54_source_box_on_fixed_three_C9_support",
        "shared_complete_sine_law_both_rows_in_structural_tau_clock",
        "outer_source_cover_does_not_assert_acquisition_zero_sums_or_cycle_class",
        "two_shared_prefixes_then_four_full_state_suffixes_without_reset",
        "only_donor_form_changes_at_supplied_events_all_phases_and_hidden_states_carry",
        "fixed_step_clipped_only_at_declared_delay_and_final_time",
        "global_unique_step_attempt_budget_no_adaptive_retry_or_response_tuning",
        "first_numerical_or_budget_failure_stops_all_later_steps_and_events",
        "completed_steps_and_failed_attempt_source_tube_reason_are_retained",
        "all_four_endpoints_required_before_mixed_observation_is_available",
        "suffix_increments_cancel_shared_prefix_receiver_coordinates_symbolically",
        "endpoint_interval_product_relaxes_common_source_correlations_not_joint_realizability",
        "interval_width_includes_source_range_wrapping_arithmetic_and_remainder",
        "global_smooth_tube_domain_does_not_certify_acute_identity_or_work",
        "no_prediction_inverse_source_handoff_sensor_noise_or_physical_observation_input",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-four-history-readout.v1",
            "report": _project(self),
        }


def _admit_class_readout_inputs(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    first_probe_amplitude,
    second_probe_amplitude,
    delay,
    total_duration,
    time_step,
    order,
    max_steps,
):
    """Normalize the complete primitive boundary before any field execution."""
    channels = []
    for raw, label in (
        (initial_form_bounds, "initial_form_bounds"),
        (initial_phase_bounds, "initial_phase_bounds"),
    ):
        rows = _ordered(raw, label, limit=28)
        if len(rows) != 27:
            raise ValueError(f"{label} must contain exactly 27 endpoint pairs")
        channels.append(
            tuple(_interval(row, f"{label}[{i}]") for i, row in enumerate(rows))
        )
    a, b, s, total, h = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (first_probe_amplitude, "first_probe_amplitude"),
            (second_probe_amplitude, "second_probe_amplitude"),
            (delay, "delay"),
            (total_duration, "total_duration"),
            (time_step, "time_step"),
        )
    )
    if not 0 <= s <= total <= 2:
        raise ValueError("require 0<=delay<=total_duration<=2")
    if not 0 < h <= 2:
        raise ValueError("time_step must lie in (0,2]")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("order must be an ordinary integer in 1..16")
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError("max_steps must be an ordinary integer in 1..4096")
    form, phase = channels
    return form, phase, a, b, s, total, h, order, max_steps


def bound_sine_class_four_history_readout(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    first_probe_amplitude,
    second_probe_amplitude,
    delay,
    total_duration,
    time_step,
    order,
    max_steps,
) -> SineClassFourHistoryReadout:
    """Enclose four complete histories from nine independently admitted inputs.

    Source channels each have 27 ordered real endpoint pairs. Amplitudes are
    signed finite reals. Require 0<=delay<=total_duration<=2, 0<time_step<=2,
    ordinary integer order1..16 and max_steps1..4096. The latter caps all shared
    kernel attempts across the six-segment tree, including a failed attempt.
    Zero-duration segments apply their event without invoking a flow step.

    First failure retains available prefixes and completed individual readings;
    subsequent events are not applied. The complete four-reading tuple and its
    mixed enclosure are unavailable until every suffix reaches total_duration.
    """
    form, phase, a, b, s, total, h, order, max_steps = _admit_class_readout_inputs(
        initial_form_bounds=initial_form_bounds,
        initial_phase_bounds=initial_phase_bounds,
        first_probe_amplitude=first_probe_amplitude,
        second_probe_amplitude=second_probe_amplitude,
        delay=delay,
        total_duration=total_duration,
        time_step=time_step,
        order=order,
        max_steps=max_steps,
    )
    source = form + phase
    geometry = _derive(_NODES, tuple(sorted(tuple(sorted(edge)) for edge in _EDGES)))
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    flow, domain = _full_sine_field(model, geometry, degrees)

    def count(duration):
        ratio = duration / h
        return (ratio.numerator + ratio.denominator - 1) // ratio.denominator

    plan = (
        ("prefix_unprobed", None, Q(0), s, Q(0)),
        ("prefix_first", None, Q(0), s, a),
        ("neither", 0, s, total, Q(0)),
        ("first_only", 1, s, total, Q(0)),
        ("second_only", 0, s, total, b),
        ("both", 1, s, total, b),
    )
    segments = []
    attempted = completed = 0
    failed_index = None
    reasons = []
    for index, (label, parent, start, end, jump) in enumerate(plan):
        pre_event = initial = state = failed_initial = failed_time = failed_tube = None
        increment = None
        current = None
        steps = []
        reason = None
        status = "not_attempted"
        if failed_index is None:
            if start < end and attempted >= max_steps:
                status, reason = (
                    "budget_exhausted",
                    "unique_step_budget_exhausted_before_event",
                )
            else:
                state = (
                    source if parent is None else segments[parent].final_state_bounds
                )
                pre_event = state
                if jump:
                    state = tuple(
                        value + jump if i == 4 else value
                        for i, value in enumerate(state)
                    )
                initial, current = state, start
                increment = I(0)
                status = "admitted"
                while current < end:
                    if attempted >= max_steps:
                        status, reason = (
                            "budget_exhausted",
                            "unique_step_budget_exhausted",
                        )
                        failed_initial, failed_time = state, current
                        break
                    width = min(h, end - current)
                    attempted += 1
                    step, failed_tube, reason = validated_box_taylor_step(
                        state, width, flow, domain, order=order, time=current
                    )
                    if step is None:
                        status = "unavailable"
                        failed_initial, failed_time = state, current
                        if reason is None:
                            reason = "shared_step_unavailable"
                        break
                    steps.append(step)
                    increment += step.increment[22]
                    state, current = step.endpoint, current + width
                    completed += 1
                    if completed % 64 == 0:
                        _LOGGER.info(
                            "Validated %d unique class-readout steps", completed
                        )
            if reason is not None:
                failed_index = index
                reasons.append(f"{label}: {reason}")
        segments.append(
            _ClassReadoutSegment(
                label=label,
                parent_segment_index=parent,
                start_time=start,
                end_time=end,
                form_jump=jump,
                pre_event_box=pre_event,
                initial_box=initial,
                steps=tuple(steps),
                completed_time=current,
                completed_endpoint_box=state,
                completed_receiver_increment_bounds=increment,
                final_state_bounds=state if status == "admitted" else None,
                failed_initial_box=failed_initial,
                failed_time=failed_time,
                failed_tube=failed_tube,
                status=status,
                reason=reason,
            )
        )
    readings = tuple(
        (segment.label, segment.final_state_bounds[22])
        for segment in segments[2:]
        if segment.status == "admitted"
    )
    endpoints = tuple(value for _, value in readings) if len(readings) == 4 else None
    raw_mixed = (
        None
        if endpoints is None
        else endpoints[0] - endpoints[1] - endpoints[2] + endpoints[3]
    )
    increments = (
        None
        if endpoints is None
        else tuple(
            segment.completed_receiver_increment_bounds for segment in segments[2:]
        )
    )
    mixed = (
        None
        if increments is None
        else increments[0] - increments[1] - increments[2] + increments[3]
    )
    return SineClassFourHistoryReadout(
        initial_form_bounds=form,
        initial_phase_bounds=phase,
        first_probe_amplitude=a,
        second_probe_amplitude=b,
        delay=s,
        total_duration=total,
        time_step=h,
        order=order,
        max_steps=max_steps,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        source_box=source,
        initial_receiver_form_bounds=form[22],
        history_event_amplitudes=((Q(0), Q(0)), (a, Q(0)), (Q(0), b), (a, b)),
        segments=tuple(segments),
        planned_unique_step_count=2 * count(s) + 4 * count(total - s),
        attempted_step_count=attempted,
        completed_step_count=completed,
        completed_segment_count=sum(
            segment.status == "admitted" for segment in segments
        ),
        completed_history_readout_bounds=readings,
        endpoint_readout_bounds=endpoints,
        raw_endpoint_mixed_bounds=raw_mixed,
        suffix_receiver_increment_bounds=increments,
        mixed_readout_bounds=mixed,
        failed_segment_index=failed_index,
        unattempted_segment_indices=tuple(
            i for i, segment in enumerate(segments) if segment.status == "not_attempted"
        ),
        status="admitted" if endpoints is not None else "unavailable",
        unavailable_reasons=tuple(reasons),
    )
