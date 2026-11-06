"""One fresh two-direction full-state metric connection evaluation.

The two forecasts start from the same admitted complete state. Observations
concern their validated prefixes; no caller-supplied forecast or cached verdict
can substitute for executing this declared evaluation. An endpoint enclosure
does not become an independently prepared winding-zero source ball.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ..mathematics._rational_interval import INTERVAL_METHOD, pi_interval
from ..mathematics._validated_metric import _admit_rate_search
from .relational_sine_comparison import (
    SineExchangeComparison,
    _comparison_from_state,
    _comparison_neighbors,
    _sine_state_from_rows,
    _validate_comparison_labels,
)
from .relational_sine_metric_forecast import (
    SineSaddleMetricForecast,
    forecast_sine_saddle_metric,
)
from .relational_sine_partition import _private_leaf_support
from .relational_sine_regional import _branches

__all__ = ("SineMetricConnection", "forecast_sine_metric_connection")


@dataclass(frozen=True)
class _CycleObservation:
    step_index: int
    time: Q
    duration: Q
    edge_turn_offsets: tuple[int, ...] | None
    winding: int | None
    acute_margin_lower_bound: Q | None


def _cycle_observation(box, cycle_indices, *, step_index, time, duration):
    """Read fresh producer boxes, not an independently supplied certificate."""
    if len(box) != 20:
        raise ValueError("connection observation requires all twenty coordinates")
    phases = box[10:]
    pairs = tuple(zip(cycle_indices, cycle_indices[1:] + cycle_indices[:1]))
    offsets, _, winding, margin = _branches(
        tuple(phases[j] - phases[i] for i, j in pairs)
    )
    return _CycleObservation(step_index, time, duration, offsets, winding, margin)


def _connection_observations(forward, backward, cycle_indices):
    """Select the first declared endpoint and first contiguous acute window."""
    endpoints = tuple(
        _cycle_observation(
            step.endpoint,
            cycle_indices,
            step_index=index,
            time=step.time + step.duration,
            duration=Q(0),
        )
        for index, step in enumerate(forward.steps)
    )
    tubes = tuple(
        _cycle_observation(
            step.tube,
            cycle_indices,
            step_index=index,
            time=step.time,
            duration=step.duration,
        )
        for index, step in enumerate(backward.steps)
    )
    first_endpoint = next((row for row in endpoints if row.winding == 0), None)
    window = None
    start, end, offsets, margin = None, None, None, None
    threshold = pi_interval().hi
    for row in tubes:
        acute = (
            row.winding == 1
            and row.acute_margin_lower_bound is not None
            and row.acute_margin_lower_bound > 0
        )
        if not acute:
            start = end = offsets = margin = None
            continue
        if start is None or row.time != end or row.edge_turn_offsets != offsets:
            start, margin = row.time, row.acute_margin_lower_bound
        else:
            margin = min(margin, row.acute_margin_lower_bound)
        end, offsets = row.time + row.duration, row.edge_turn_offsets
        if end - start >= threshold:
            window = (start, end, margin)
            break
    return endpoints, tubes, first_endpoint, window


@dataclass(frozen=True)
class SineMetricConnection:
    """Fresh same-orbit passage evidence, distinct from source preparation."""

    source: SineExchangeComparison
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    forward: SineSaddleMetricForecast
    backward: SineSaddleMetricForecast
    forward_endpoint_observations: tuple[_CycleObservation, ...]
    backward_tube_observations: tuple[_CycleObservation, ...]
    forward_zero_winding_step_index: int | None
    forward_zero_winding_time: Q | None
    backward_retention_start: Q | None
    backward_retention_end: Q | None
    backward_acute_margin_lower_bound: Q | None
    forward_horizon_complete: bool
    backward_horizon_complete: bool
    declared_horizons_complete: bool
    same_orbit_connection_certified: bool
    independent_zero_winding_source_ball_certified: bool
    outcome: str
    reasons: tuple[str, ...]
    retained_scaled_duration: Q = Q(1)
    required_original_duration_upper_bound: Q = pi_interval().hi
    clock: str = "original structural t; tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "two_fresh_full_state_forecasts_from_one_readmitted_source_and_unchanged_law",
        "forward_and_reverse_fields_are_evaluated_once_with_the_same_fixed_budget",
        "first_forward_winding_zero_endpoint_and_first_backward_whole_tube_acute_winding_one_window",
        "time_reversal_changes_forms_sign_and_preserves_phases_and_winding",
        "the_connection_refers_to_the_same_orbit_not_unrelated_endpoint_selections",
        "validated_prefix_observations_and_complete_declared_horizons_are_distinct",
        "endpoint_enclosures_are_not_arbitrary_prepared_winding_zero_source_boxes",
        "no_event_time_autonomous_preparation_infinite_retention_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_label_groups(self.cycle)
        self.forward.to_dict()
        self.backward.to_dict()
        return {"schema": "tnfr.sine-metric-connection.v1", "report": _project(self)}


def forecast_sine_metric_connection(
    source,
    *,
    cycle,
    duration,
    time_step,
    initial_coordinate_radius,
    growth_rate_bounds=(Q(0), Q(1)),
    growth_bisections=8,
    order=16,
    max_steps=256,
) -> SineMetricConnection:
    """Execute one compound prospective evaluation, retaining both responses.

    A missing observation on complete horizons is only a finite noncertificate.
    A numerical failure remains unavailable. If both observations are already
    certified on validated prefixes, that fact survives later step failure;
    ``declared_horizons_complete`` still exposes the incomplete study budget.
    """
    source, _, indices, _, _ = _private_leaf_support(source, cycle)
    growth_rate_bounds = _admit_rate_search(growth_rate_bounds, growth_bisections)
    source = _comparison_from_state(
        _sine_state_from_rows(
            source.nodes,
            source.edges,
            source.epi,
            source.phase,
            source.capacity,
            _comparison_neighbors(source),
        ),
        source.reference_model,
    )
    cycle = tuple(source.nodes[index] for index in indices)
    options = dict(
        cycle=cycle,
        duration=duration,
        time_step=time_step,
        initial_coordinate_radius=initial_coordinate_radius,
        phase_radius=None,
        growth_rate_bounds=growth_rate_bounds,
        growth_bisections=growth_bisections,
        order=order,
        max_steps=max_steps,
    )
    forward = forecast_sine_saddle_metric(source, direction=1, **options)
    backward = forecast_sine_saddle_metric(source, direction=-1, **options)
    for result, direction in ((forward, 1), (backward, -1)):
        if (
            result.initial_center != source.epi + source.phase
            or result.direction != direction
        ):
            raise ArithmeticError(
                "fresh forecast lost its common initial-state association"
            )
    endpoints, tubes, first, window = _connection_observations(
        forward, backward, indices
    )
    forward_complete = (
        forward.admitted and forward.validated_duration == forward.duration
    )
    backward_complete = (
        backward.admitted and backward.validated_duration == backward.duration
    )
    complete = forward_complete and backward_complete
    certified = first is not None and window is not None
    outcome = (
        "same_orbit_connection_certified"
        if certified
        else "no_certificate_on_declared_horizons" if complete else "unavailable"
    )
    reasons = tuple(
        reason
        for passed, reason in (
            (first is not None, "forward_zero_winding_endpoint_not_certified"),
            (
                window is not None,
                "backward_whole_time_acute_winding_one_retention_not_certified",
            ),
            (forward_complete, "forward_declared_horizon_incomplete"),
            (backward_complete, "backward_declared_horizon_incomplete"),
        )
        if not passed
    )
    return SineMetricConnection(
        source=source,
        cycle=cycle,
        cycle_indices=indices,
        forward=forward,
        backward=backward,
        forward_endpoint_observations=endpoints,
        backward_tube_observations=tubes,
        forward_zero_winding_step_index=None if first is None else first.step_index,
        forward_zero_winding_time=None if first is None else first.time,
        backward_retention_start=None if window is None else window[0],
        backward_retention_end=None if window is None else window[1],
        backward_acute_margin_lower_bound=None if window is None else window[2],
        forward_horizon_complete=forward_complete,
        backward_horizon_complete=backward_complete,
        declared_horizons_complete=complete,
        same_orbit_connection_certified=certified,
        independent_zero_winding_source_ball_certified=False,
        outcome=outcome,
        reasons=reasons,
    )
