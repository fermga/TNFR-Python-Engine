"""Independent complete-law port observations for two primitive sources.

Each original 54-coordinate state receives one central form impulse, then
uses the shared source-box Picard/Taylor kernel on a fixed partition. The
producer consumes absolute continuous phase lifts, not interface residuals
or predictions. Source class, acquisition and any target conversion remain
external premises.
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

__all__ = ("SineClassPortReadout", "bound_sine_class_port_readout")

_PORTS = (4, 13, 22)
_MAX_STEPS = 8192
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClassPortHistory:
    """One carried history, a completed prefix, or an unapplied event."""

    source_index: int
    pre_event_box: tuple[I, ...] | None
    initial_box: tuple[I, ...] | None
    steps: tuple[ValidatedBoxTaylorStep, ...]
    attempted_step_count: int
    completed_time: Q | None
    completed_endpoint_box: tuple[I, ...] | None
    final_state_bounds: tuple[I, ...] | None
    readout_bounds: I | None
    failed_initial_box: tuple[I, ...] | None
    failed_time: Q | None
    failed_tube: tuple[I, ...] | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class SineClassPortReadout:
    """Two endpoint port readings, retaining independent full-state sources.

    Each source has 27 forms and 27 absolute continuous phase lifts at 0-.
    A primitive box is an outer cover, not evidence of acquisition, winding,
    mean constraints or independently preparable corners. The two sources
    may differ. The same supplied impulse changes their central forms at
    time zero; every phase and hidden coordinate then carries unchanged
    between validated steps.

    Widths include source uncertainty, wrapping, arithmetic and truncation.
    No sensor error or causal prediction is used to narrow these intervals.
    """

    initial_form_bounds: tuple[tuple[I, ...], ...]
    initial_phase_bounds: tuple[tuple[I, ...], ...]
    port_impulse: tuple[Q, ...]
    horizon: Q
    time_step: Q
    order: int
    max_steps: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    histories: tuple[_ClassPortHistory, ...]
    planned_step_count: int
    attempted_step_count: int
    completed_step_count: int
    completed_source_count: int
    completed_source_readout_bounds: tuple[tuple[int, I], ...]
    endpoint_readout_bounds: tuple[I, I] | None
    failed_source_index: int | None
    unattempted_source_indices: tuple[int, ...]
    status: str
    unavailable_reasons: tuple[str, ...]
    source_order: tuple[int, int] = (0, 1)
    port_nodes: tuple[int, int, int] = _PORTS
    readout_node: int = 13
    capacity: tuple[Q, ...] = (Q(1),) * 27
    state_order: tuple[str, ...] = tuple(f"x_{i}" for i in _NODES) + tuple(
        f"theta_{i}" for i in _NODES
    )
    clock: str = "tau=e*t; e=1023/1024"
    source_coordinates: str = (
        "pre-input form x and absolute continuous phase theta at 0-"
    )
    arithmetic_method: str = INTERVAL_METHOD
    method: str = "two_source_single_event_full54_Picard_Taylor_v1"
    scope: tuple[str, ...] = (
        "both_primitive_full54_sources_are_admitted_before_any_field_execution",
        "source_order_is_an_association_not_a_cycle_class_or_acquisition_verdict",
        "absolute_phase_lifts_must_be_constructed_outside_from_any_target_and_residual",
        "one_supplied_form_impulse_at_ports4_13_22_with_phases_unchanged",
        "all54_coordinates_carry_through_the_shared_complete_sine_law",
        "fixed_step_clipped_only_at_final_horizon_without_adaptive_retry",
        "one_global_attempt_budget_counts_successful_and_failed_kernel_calls",
        "first_failure_stops_all_later_flow_calls_and_unapplied_events",
        "completed_individual_readings_remain_available_after_a_later_failure",
        "both_complete_sources_required_for_the_complete_endpoint_pair",
        "retained_derivative_and_Picard_generation_remain_execution_premises",
        "source_range_wrapping_arithmetic_and_truncation_are_not_sensor_error",
        "no_prediction_kernel_reduction_target_root_or_source_handoff_is_consumed",
        "global_smooth_domain_is_not_an_acute_identity_work_or_physical_certificate",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-class-port-readout.v1", "report": _project(self)}


def _admit_port_readout_inputs(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    port_impulse,
    horizon,
    time_step,
    order,
    max_steps,
):
    """Normalize all consumed primitives before model or field construction."""
    channels = []
    for raw, label in (
        (initial_form_bounds, "initial_form_bounds"),
        (initial_phase_bounds, "initial_phase_bounds"),
    ):
        sources = _ordered(raw, label, limit=3)
        if len(sources) != 2:
            raise ValueError(f"{label} must contain exactly two source covers")
        admitted = []
        for source_index, source in enumerate(sources):
            rows = _ordered(source, f"{label}[{source_index}]", limit=28)
            if len(rows) != 27:
                raise ValueError(
                    f"{label}[{source_index}] must contain 27 endpoint pairs"
                )
            admitted.append(
                tuple(
                    _interval(row, f"{label}[{source_index}][{i}]")
                    for i, row in enumerate(rows)
                )
            )
        channels.append(tuple(admitted))
    impulses = _ordered(port_impulse, "port_impulse", limit=4)
    if len(impulses) != 3:
        raise ValueError("port_impulse must contain three signed scalars")
    impulses = tuple(
        exact_or_represented_real(value, "port_impulse") for value in impulses
    )
    duration = exact_or_represented_real(horizon, "horizon")
    width = exact_or_represented_real(time_step, "time_step")
    if not 0 <= duration <= 2:
        raise ValueError("horizon must lie in [0,2]")
    if not 0 < width <= 2:
        raise ValueError("time_step must lie in (0,2]")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("order must be an ordinary integer in 1..16")
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError("max_steps must be an ordinary integer in 1..8192")
    return (*channels, impulses, duration, width, order, max_steps)


def bound_sine_class_port_readout(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    port_impulse,
    horizon,
    time_step,
    order,
    max_steps,
) -> SineClassPortReadout:
    """Enclose two single-event histories from seven mandatory primitive inputs.

    Source channels each contain two ordered 27-row real endpoint-pair covers.
    Phase pairs are absolute continuous lifts in radians. Three signed finite
    amplitudes apply at nodes4,13,22; zero means no jump. Require 0<=horizon<=2,
    0<time_step<=2, ordinary order1..16 and max_steps1..8192. The cap counts all
    attempts across both histories. A zero horizon applies the event without
    a flow call. Exhaustion before a positive-duration history applies no
    event there. First numerical failure retains its attempted state/tube
    and every completed step, without retry or later source execution.
    """
    forms, phases, impulse, total, width, order, cap = _admit_port_readout_inputs(
        initial_form_bounds=initial_form_bounds,
        initial_phase_bounds=initial_phase_bounds,
        port_impulse=port_impulse,
        horizon=horizon,
        time_step=time_step,
        order=order,
        max_steps=max_steps,
    )
    geometry = _derive(_NODES, tuple(sorted(tuple(sorted(edge)) for edge in _EDGES)))
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    flow, domain = _full_sine_field(model, geometry, degrees)
    jumps = dict(zip(_PORTS, impulse))
    histories, reasons = [], []
    attempted = completed = 0
    failed_source = None
    for index in range(2):
        pre_event = initial = state = current = None
        failed_initial = failed_time = failed_tube = None
        steps = []
        history_attempts = 0
        reason = None
        status = "not_attempted"
        if failed_source is None:
            if total > 0 and attempted >= cap:
                status, reason = (
                    "budget_exhausted",
                    "total_step_budget_exhausted_before_event",
                )
            else:
                pre_event = forms[index] + phases[index]
                state = tuple(
                    value + jumps[i] if jumps.get(i) else value
                    for i, value in enumerate(pre_event)
                )
                initial, current = state, Q(0)
                status = "admitted"
                while current < total:
                    if attempted >= cap:
                        status, reason = (
                            "budget_exhausted",
                            "total_step_budget_exhausted",
                        )
                        failed_initial, failed_time = state, current
                        break
                    step_width = min(width, total - current)
                    attempted += 1
                    history_attempts += 1
                    step, failed_tube, reason = validated_box_taylor_step(
                        state, step_width, flow, domain, order=order, time=current
                    )
                    if step is None:
                        status = "unavailable"
                        failed_initial, failed_time = state, current
                        if reason is None:
                            reason = "shared_step_unavailable"
                        break
                    steps.append(step)
                    state, current = step.endpoint, current + step_width
                    completed += 1
                    if completed % 64 == 0:
                        _LOGGER.info(
                            "Validated %d complete-law port-readout steps", completed
                        )
            if reason is not None:
                failed_source = index
                reasons.append(f"source_{index}: {reason}")
        final = state if status == "admitted" else None
        histories.append(
            _ClassPortHistory(
                source_index=index,
                pre_event_box=pre_event,
                initial_box=initial,
                steps=tuple(steps),
                attempted_step_count=history_attempts,
                completed_time=current,
                completed_endpoint_box=state,
                final_state_bounds=final,
                readout_bounds=final[13] if final is not None else None,
                failed_initial_box=failed_initial,
                failed_time=failed_time,
                failed_tube=failed_tube,
                status=status,
                reason=reason,
            )
        )
    readings = tuple(
        (item.source_index, item.readout_bounds)
        for item in histories
        if item.status == "admitted"
    )
    ratio = total / width
    planned = 2 * ((ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    return SineClassPortReadout(
        initial_form_bounds=forms,
        initial_phase_bounds=phases,
        port_impulse=impulse,
        horizon=total,
        time_step=width,
        order=order,
        max_steps=cap,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        histories=tuple(histories),
        planned_step_count=planned,
        attempted_step_count=attempted,
        completed_step_count=completed,
        completed_source_count=len(readings),
        completed_source_readout_bounds=readings,
        endpoint_readout_bounds=(
            tuple(value for _, value in readings) if len(readings) == 2 else None
        ),
        failed_source_index=failed_source,
        unattempted_source_indices=tuple(
            item.source_index for item in histories if item.initial_box is None
        ),
        status="admitted" if len(readings) == 2 else "unavailable",
        unavailable_reasons=tuple(reasons),
    )
