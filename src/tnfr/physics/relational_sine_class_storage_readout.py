"""Matched full/tangent loss integrals from primitive complete source covers.

Eight 55-coordinate histories use the shared fixed-step box Taylor kernel.
The passive loss coordinate avoids subtraction of large absolute potentials;
it does not preserve every correlation between independently enclosed models.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import MAX_ORDER, Jet
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ..mathematics._validated_taylor import (
    ValidatedBoxTaylorStep,
    validated_box_taylor_step,
)
from ._sine_flow import _full_sine_field
from .phase_cycle_geometry import PhaseCycleGeometry, _derive
from .relational_observations import _interval, _ordered
from .relational_sine_class_cubic_response import _cubic_parameters, _CubicParameters
from .relational_sine_class_mediation import _EDGES, _NODES
from .relational_sine_comparison import _sine_form_gradient

__all__ = ("SineClassStorageReadout", "bound_sine_class_storage_readout")

_HISTORIES = ("neither", "donor_only", "receiver_only", "both")
_MODELS = ("full", "tangent")
_MAX_STEPS = 8192


def _storage_target(parameters):
    pi = pi_interval()
    return tuple(
        2 * Q(parameters.classes[c] * (j - 4), 9) * pi
        for c in range(3)
        for j in range(9)
    )


def _storage_loss_fields(model, geometry, parameters):
    """Augment both complete nodal rows with the same passive loss law."""
    full, _ = _full_sine_field(model, geometry, parameters.degrees)
    neighbors = tuple(
        tuple(j if i == node else i for i, j in _EDGES if node in (i, j))
        for node in _NODES
    )

    def loss(gradient):
        return sum(
            (value**2 / degree for value, degree in zip(gradient, parameters.degrees)),
            gradient[0] * 0,
        )

    def full_flow(state):
        if len(state) != 55:
            raise ValueError("complete storage field requires 55 coordinates")
        gradient = _sine_form_gradient(state[:27], neighbors)
        return full(state[:54]) + (loss(gradient),)

    def tangent_flow(state):
        if len(state) != 55:
            raise ValueError("complete storage field requires 55 coordinates")
        form, phase = state[:27], state[27:54]
        gradient = _sine_form_gradient(form, neighbors)
        zero = Jet.constant(0, state[0].order) if isinstance(state[0], Jet) else I(0)
        currents = [zero] * 27
        for (i, j), cosine in zip(_EDGES, parameters.edge_cosines):
            current = (phase[j] - phase[i]) * cosine
            currents[i] += current
            currents[j] -= current
        gamma, degrees = parameters.gamma, parameters.degrees
        return (
            tuple(
                (-q + current * gamma) / d
                for q, current, d in zip(gradient, currents, degrees)
            )
            + tuple(q * gamma / d for q, d in zip(gradient, degrees))
            + (loss(gradient),)
        )

    def domain(_):
        # Both augmented fields are globally smooth. This asserts neither
        # acuteness, trapping, work admission nor physical source preparation.
        return (Q(1),)

    return (full_flow, tangent_flow), domain


@dataclass(frozen=True)
class _StorageHistory:
    history_index: int
    history_name: str
    model_name: str
    event_amplitudes: tuple[Q, Q]
    pre_event_box: tuple[I, ...] | None
    initial_box: tuple[I, ...] | None
    steps: tuple[ValidatedBoxTaylorStep, ...]
    attempted_step_count: int
    completed_time: Q | None
    completed_endpoint_box: tuple[I, ...] | None
    completed_loss_increment_bounds: I | None
    final_state_bounds: tuple[I, ...] | None
    loss_integral_bounds: I | None
    failed_initial_box: tuple[I, ...] | None
    failed_time: Q | None
    failed_tube: tuple[I, ...] | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class SineClassStorageReadout:
    """One matched source, eight histories and a conditional mixed loss readout.

    Primitive phase deviations are original radians relative to the named wound
    target, at time 0-. The full model uses theta=target+deviation; its tangent
    partner uses that same deviation w and the target Hessian. Both models retain
    all 54 original nodal coordinates. The target and primitive source covers are
    constructed once, then shared as associations across the eight independent
    enclosures; their Cartesian corners are not independent physical preparations.

    The reported storage excess is minus the mixed integrated loss. That identity
    requires the same complete source and matched simultaneous events, as declared
    here, but is not an acquisition or storage-sensor certificate.
    """

    mediator_class: int
    initial_form_bounds: tuple[I, ...]
    initial_phase_deviation_bounds: tuple[I, ...]
    donor_amplitude: Q
    receiver_amplitude: Q
    horizon: Q
    time_step: Q
    order: int
    max_steps: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    parameters: _CubicParameters
    target_phase_bounds: tuple[I, ...]
    full_source_box: tuple[I, ...]
    tangent_source_box: tuple[I, ...]
    histories: tuple[_StorageHistory, ...]
    planned_step_count: int
    attempted_step_count: int
    completed_step_count: int
    completed_history_count: int
    completed_history_loss_bounds: tuple[tuple[int, I], ...]
    loss_integral_bounds: tuple[I, ...] | None
    raw_endpoint_loss_bounds: tuple[I, ...] | None
    integrated_excess_loss_bounds: I | None
    excess_storage_bounds: I | None
    failed_history_index: int | None
    unattempted_history_indices: tuple[int, ...]
    status: str
    unavailable_reasons: tuple[str, ...]
    history_order: tuple[str, ...] = _HISTORIES
    model_order: tuple[str, ...] = _MODELS
    intervention_nodes: tuple[int, int] = (4, 22)
    capacity: tuple[Q, ...] = (Q(1),) * 27
    full_state_order: tuple[str, ...] = tuple(
        f"{c}_{i}" for c in ("x", "theta") for i in _NODES
    ) + ("cumulative_loss",)
    tangent_state_order: tuple[str, ...] = tuple(
        f"{c}_{i}" for c in ("v", "w") for i in _NODES
    ) + ("cumulative_loss",)
    clock: str = "tau=e*t; e=1023/1024"
    source_coordinates: str = "pre-input x and y=theta-Theta_k in radians at 0-"
    loss_law: str = "sum_i (L*x)_i**2/d_i, using each model's own form"
    arithmetic_method: str = INTERVAL_METHOD
    method: str = "matched_full_tangent_eight_history_loss55_Picard_Taylor_v1"
    scope: tuple[str, ...] = (
        "all_source_event_clock_and_numerical_primitives_are_admitted_before_execution",
        "full_absolute_phase_and_tangent_phase_deviation_have_original_radian_units",
        "all54_nodal_coordinates_and_one_passive_loss_integral_carry_without_reset",
        "same_target_and_complete_source_parameters_for_all_eight_histories",
        "full_then_tangent_for_each_neither_donor_receiver_both_history",
        "fixed_step_clipped_at_horizon_without_retry_or_adaptive_refinement",
        "global_attempt_cap_and_first_failure_stop_later_flows_and_events",
        "mixed_output_available_only_after_all_eight_complete_histories",
        "Taylor_loss_increments_remove_accumulator_baselines_not_model_source_correlations",
        "storage_balance_cancels_the_common_initial_gap_and_matched_event_work",
        "all_interval_widths_include_source_wrapping_arithmetic_and_truncation",
        "smooth_domain_is_not_acquisition_identity_work_or_sensor_admission",
        "no_analytic_prediction_or_observed_response_narrows_the_produced_bounds",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-storage-readout.v1",
            "report": _project(self),
        }


def _admit_storage_readout_inputs(
    *,
    mediator_class,
    initial_form_bounds,
    initial_phase_deviation_bounds,
    donor_amplitude,
    receiver_amplitude,
    horizon,
    time_step,
    order,
    max_steps,
):
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator_class must be ordinary integer one or two")
    sources = []
    for raw, name in (
        (initial_form_bounds, "initial_form_bounds"),
        (initial_phase_deviation_bounds, "initial_phase_deviation_bounds"),
    ):
        rows = _ordered(raw, name, limit=28)
        if len(rows) != 27:
            raise ValueError(f"{name} requires exactly 27 endpoint pairs")
        sources.append(
            tuple(_interval(row, f"{name}[{i}]") for i, row in enumerate(rows))
        )
    a, b, total, width = (
        _exact(value, name)
        for value, name in (
            (donor_amplitude, "donor_amplitude"),
            (receiver_amplitude, "receiver_amplitude"),
            (horizon, "horizon"),
            (time_step, "time_step"),
        )
    )
    if not 0 <= total <= 2 or not 0 < width <= 2:
        raise ValueError("horizon must lie in [0,2] and time_step in (0,2]")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("order must be an ordinary integer in 1..16")
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError("max_steps must be an ordinary integer in 1..8192")
    return mediator_class, *sources, a, b, total, width, order, max_steps


def _mixed_loss(losses):
    return sum(
        (
            value if index in (0, 3, 5, 6) else -value
            for index, value in enumerate(losses)
        ),
        I(0),
    )


def bound_sine_class_storage_readout(
    *,
    mediator_class,
    initial_form_bounds,
    initial_phase_deviation_bounds,
    donor_amplitude,
    receiver_amplitude,
    horizon,
    time_step,
    order,
    max_steps,
) -> SineClassStorageReadout:
    """Enclose a matched full/tangent loss comparison from nine primitives."""
    k, forms, phases, a, b, total, width, order, cap = _admit_storage_readout_inputs(
        mediator_class=mediator_class,
        initial_form_bounds=initial_form_bounds,
        initial_phase_deviation_bounds=initial_phase_deviation_bounds,
        donor_amplitude=donor_amplitude,
        receiver_amplitude=receiver_amplitude,
        horizon=horizon,
        time_step=time_step,
        order=order,
        max_steps=max_steps,
    )
    parameters = _cubic_parameters(k)
    target = _storage_target(parameters)
    full_source = forms + tuple(t + y for t, y in zip(target, phases)) + (I(0),)
    tangent_source = forms + phases + (I(0),)
    geometry = _derive(_NODES, tuple(sorted(tuple(sorted(edge)) for edge in _EDGES)))
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    fields, domain = _storage_loss_fields(model, geometry, parameters)
    histories, reasons = [], []
    attempted = completed = 0
    failure = None
    for index in range(8):
        word, model_index = divmod(index, 2)
        impulses = ((Q(0), Q(0)), (a, Q(0)), (Q(0), b), (a, b))[word]
        pre_event = initial = state = current = accumulated = None
        failed_initial = failed_time = failed_tube = None
        steps, count, reason, status = [], 0, None, "not_attempted"
        if failure is None:
            if total > 0 and attempted >= cap:
                status, reason = (
                    "budget_exhausted",
                    "total_step_budget_exhausted_before_event",
                )
            else:
                pre_event = (full_source, tangent_source)[model_index]
                jumps = dict(zip((4, 22), impulses))
                state = tuple(
                    value + jumps[i] if jumps.get(i) else value
                    for i, value in enumerate(pre_event)
                )
                initial, current, accumulated = state, Q(0), I(0)
                status = "admitted"
                while current < total:
                    if attempted >= cap:
                        status, reason = (
                            "budget_exhausted",
                            "total_step_budget_exhausted",
                        )
                        failed_initial, failed_time = state, current
                        break
                    h = min(width, total - current)
                    attempted += 1
                    count += 1
                    step, failed_tube, reason = validated_box_taylor_step(
                        state, h, fields[model_index], domain, order=order, time=current
                    )
                    if step is None:
                        status = "unavailable"
                        failed_initial, failed_time = state, current
                        reason = reason or "shared_step_unavailable"
                        break
                    steps.append(step)
                    accumulated += step.increment[54]
                    state, current = step.endpoint, current + h
                    completed += 1
            if reason is not None:
                failure = index
                reasons.append(f"history_{index}: {reason}")
        final = state if status == "admitted" else None
        histories.append(
            _StorageHistory(
                index,
                _HISTORIES[word],
                _MODELS[model_index],
                impulses,
                pre_event,
                initial,
                tuple(steps),
                count,
                current,
                state,
                accumulated,
                final,
                accumulated if final is not None else None,
                failed_initial,
                failed_time,
                failed_tube,
                status,
                reason,
            )
        )
    readings = tuple(
        (row.history_index, row.loss_integral_bounds)
        for row in histories
        if row.status == "admitted"
    )
    losses = tuple(value for _, value in readings) if len(readings) == 8 else None
    mixed = _mixed_loss(losses) if losses is not None else None
    ratio = total / width
    planned = 8 * ((ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    return SineClassStorageReadout(
        k,
        forms,
        phases,
        a,
        b,
        total,
        width,
        order,
        cap,
        model,
        geometry,
        parameters,
        target,
        full_source,
        tangent_source,
        tuple(histories),
        planned,
        attempted,
        completed,
        len(readings),
        readings,
        losses,
        (
            tuple(row.final_state_bounds[54] for row in histories)
            if losses is not None
            else None
        ),
        mixed,
        -mixed if mixed is not None else None,
        failure,
        tuple(row.history_index for row in histories if row.initial_box is None),
        "admitted" if losses is not None else "unavailable",
        tuple(reasons),
    )
