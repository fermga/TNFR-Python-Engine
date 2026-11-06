"""Whole-time saddle-domain forecasts retaining full-state metric uncertainty.

The conservative sine law and its twenty-coordinate sensitivity metric are
re-admitted before execution. Coordinate boxes certify Picard inclusion and
project observations; only the metric radius is carried between steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real
from ..mathematics._interval_taylor import MAX_ORDER, Jet
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ..mathematics._validated_metric import (
    ValidatedMetricTaylorStep,
    _admit_rate_search,
    validated_metric_taylor_step,
)
from .relational_sine_comparison import (
    SineExchangeComparison,
    _comparison_neighbors,
    _validate_comparison_labels,
)
from .relational_sine_forecast import MAX_FORECAST_STEPS, _sine_flow
from .relational_sine_sensitivity import (
    SineSaddleSensitivity,
    assess_sine_saddle_sensitivity,
)

__all__ = ("SineSaddleMetricForecast", "forecast_sine_saddle_metric")


@dataclass(frozen=True)
class SineSaddleMetricForecast:
    """A fixed-budget full-state enclosure under the declared saddle law.

    Duration is elapsed original structural time. Direction chooses the
    forward or reverse autonomous field, without changing the signed form
    convention. Endpoints are observations of retained metric balls, not
    input boxes for the next uncertainty propagation.
    """

    source: SineExchangeComparison
    sensitivity: SineSaddleSensitivity
    duration: Q
    time_step: Q
    direction: int
    order: int
    max_steps: int
    initial_coordinate_radius: Q
    initial_center: tuple[Q, ...]
    initial_metric_radius: Q
    growth_rate_upper_bound: Q | None
    growth_mode: str
    growth_rate_bounds: tuple[Q, Q] | None
    growth_bisections: int
    steps: tuple[ValidatedMetricTaylorStep, ...]
    validated_duration: Q
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q
    endpoint: tuple[I, ...]
    failed_tube: tuple[I, ...] | None
    status: str
    reasons: tuple[str, ...]
    clock: str = "original structural t; direction times elapsed duration"
    method: str = "full_sine_saddle_Picard_Taylor_retained_metric128_v1"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_unit_C5_private_leaf_support_and_unit_held_capacities",
        "complete_conservative_sine_law_without_forcing_or_events",
        "all_twenty_form_phase_coordinates_and_both_origins_retained",
        "independent_initial_coordinate_errors_enclosed_in_a_metric_ball_once",
        "fixed_phase_tube_or_actual_whole_tube_Jacobian_growth_is_declared_before_execution",
        "computed_Jacobian_growth_already_uses_the_original_structural_clock",
        "center_Taylor_and_remainder_rounding_are_added_in_the_same_metric",
        "endpoint_boxes_are_observations_not_reboxed_initial_metric_radii",
        "fixed_budget_without_adaptive_retries_or_automatic_domain_enlargement",
        "partial_validated_duration_is_not_the_requested_full_horizon",
        "forecast_does_not_certify_formation_retention_or_physical_identification",
        "distinct_report_type_from_the_existing_coordinate_box_forecast",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        self.sensitivity.to_dict()
        return {
            "schema": "tnfr.sine-saddle-metric-forecast.v1",
            "report": _project(self),
        }


def _saddle_flow(values, *, source, neighbors, direction):
    """Reuse the original-clock sine field with an exact held capacity row."""
    one = Jet.constant(1, values[0].order) if isinstance(values[0], Jet) else I(1)
    rows = _sine_flow(
        tuple(values) + (one,),
        neighbors=neighbors,
        visible_capacity=source.capacity[:-1],
        model=source.reference_model,
    )
    return tuple(direction * row for row in rows[:-1])


def forecast_sine_saddle_metric(
    source,
    *,
    cycle,
    duration,
    time_step,
    initial_coordinate_radius=Q(0),
    phase_radius=Q(1, 1000),
    growth_rate_bounds=None,
    growth_bisections=8,
    direction=1,
    order=12,
    max_steps=MAX_FORECAST_STEPS,
) -> SineSaddleMetricForecast:
    """Validate a finite full-state response with declared growth admission.

    The source is actual initial state, unlike the static sensitivity reader's
    support anchor. Every consumed primitive and theorem is rebuilt. A domain
    failure leaves the last valid metric ball and elapsed time available;
    it cannot trigger a larger phase radius, shorter step or a different law.
    With phase_radius=None, each whole Picard tube instead receives an actual
    interval-Jacobian growth proof in the fixed metric. That global smooth
    sine mode uses the declared rate bracket and bounded PSD bisections.
    """
    duration = exact_or_represented_real(duration, "duration")
    time_step = exact_or_represented_real(time_step, "time_step")
    initial_coordinate_radius = exact_or_represented_real(
        initial_coordinate_radius, "initial_coordinate_radius"
    )
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_FORECAST_STEPS:
        raise ValueError(f"max_steps must be an integer from 1 to {MAX_FORECAST_STEPS}")
    if duration <= 0 or time_step <= 0 or duration / time_step > max_steps:
        raise ValueError("require positive duration and time_step within max_steps")
    if initial_coordinate_radius < 0:
        raise ValueError("initial_coordinate_radius must be nonnegative")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("direction must be the exact integer -1 or +1")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError(f"Taylor order must be an integer from 1 to {MAX_ORDER}")
    dynamic = phase_radius is None
    if dynamic:
        growth_rate_bounds = _admit_rate_search(
            (Q(0), Q(1)) if growth_rate_bounds is None else growth_rate_bounds,
            growth_bisections,
        )
    elif growth_rate_bounds is not None:
        raise ValueError("a computed growth bracket requires phase_radius=None")
    elif type(growth_bisections) is not int or growth_bisections != 8:
        raise ValueError("growth_bisections applies only to computed growth")
    sensitivity = assess_sine_saddle_sensitivity(
        source, cycle=cycle, phase_radius=Q(0) if dynamic else phase_radius
    )
    source = sensitivity.source
    center = source.epi + source.phase
    initial_center = center
    radius = sensitivity.infinity_to_metric_upper_bound * initial_coordinate_radius
    initial_radius = radius
    factors = sensitivity.metric_to_coordinate_upper_bounds
    endpoint = tuple(
        I(value - factor * radius, value + factor * radius)
        for value, factor in zip(center, factors)
    )
    pi = pi_interval()
    growth = None if dynamic else sensitivity.nonlinear_growth_rate_upper_bound / pi.lo
    saddle_phases = tuple(
        2 * turn * pi for turn in sensitivity.saddle.target_phase_turns
    )
    neighbors = _comparison_neighbors(source)
    size = len(source.nodes)

    @lru_cache(maxsize=32)
    def flow(values):
        return _saddle_flow(
            values, source=source, neighbors=neighbors, direction=direction
        )

    def domain(values):
        if dynamic:
            # The complete sine field is smooth on every retained real phase
            # lift. Its actual whole-tube Jacobian supplies the growth proof.
            return (Q(1),)
        return tuple(
            sensitivity.phase_radius - (phase - target).abs_max
            for phase, target in zip(values[size:], saddle_phases)
        )

    elapsed, steps, failed, reasons = Q(0), [], None, []
    if not sensitivity.nonlinear_tube_implication_certified:
        reasons.append("full_state_saddle_metric_not_certified")
    while elapsed < duration and not reasons:
        step = min(time_step, duration - elapsed)
        certificate, failed, reason = validated_metric_taylor_step(
            center,
            radius,
            sensitivity.full_metric,
            step,
            flow,
            domain,
            growth_rate=growth,
            growth_rate_bounds=growth_rate_bounds,
            growth_bisections=growth_bisections,
            order=order,
            time=elapsed,
            domain_failure="whole_time_saddle_phase_tube_not_admitted",
        )
        if certificate is None:
            reasons.append(reason)
            break
        steps.append(certificate)
        center, radius = certificate.endpoint_center, certificate.endpoint_radius
        endpoint = certificate.endpoint
        elapsed += step
    return SineSaddleMetricForecast(
        source=source,
        sensitivity=sensitivity,
        duration=duration,
        time_step=time_step,
        direction=direction,
        order=order,
        max_steps=max_steps,
        initial_coordinate_radius=initial_coordinate_radius,
        initial_center=initial_center,
        initial_metric_radius=initial_radius,
        growth_rate_upper_bound=(
            max(step.growth_rate_upper_bound for step in steps)
            if dynamic and steps
            else growth
        ),
        growth_mode="whole_tube_jacobian" if dynamic else "fixed_saddle_phase_tube",
        growth_rate_bounds=growth_rate_bounds,
        growth_bisections=growth_bisections,
        steps=tuple(steps),
        validated_duration=elapsed,
        endpoint_center=center,
        endpoint_radius=radius,
        endpoint=endpoint,
        failed_tube=failed,
        status="unavailable" if reasons else "admitted",
        reasons=tuple(reasons),
        method=(
            "full_sine_Picard_Taylor_retained_metric_tube_LMI128_v1"
            if dynamic
            else "full_sine_saddle_Picard_Taylor_retained_metric128_v1"
        ),
    )
