"""Necessary geometry/gain/clock refinement from four finite scalar readings.

The extra half-window reading supplies a finite second difference, never an
observed derivative. All four observations belong to one complete trajectory.
Intersecting its necessary constraints need not preserve joint realizability.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, sin
from .relational_observations import _ordered
from .relational_sine_clock_inference import (
    SineClockInference,
    infer_sine_geometry_gain_clock,
)
from .relational_sine_two_port_inference import _GAMMA_UPPER, _primitive_bounds

__all__ = ("SineCurvatureInference", "infer_sine_geometry_gain_clock_curvature")


def _positive_quotient_bounds(numerator, denominator):
    """Divide exact endpoint pairs without rounding a positive divisor to zero."""
    if not 0 < denominator[0] <= denominator[1]:
        raise ArithmeticError("positive quotient denominator was not admitted")
    corners = tuple(value / divisor for value in numerator for divisor in denominator)
    return I(min(corners), max(corners))


@dataclass(frozen=True)
class SineCurvatureInference:
    """Outer refinement by a finite first-window curvature constraint.

    The four primitive readings occur at observed times 0,H/2,H,2H. The
    clock envelope reuses indices 0,2,3. The same errors are embedded in
    both sets of coefficient rows, although their subsequent interval
    intersection conservatively discards joint error/state feasibility.

    ``curvature_refinement_available`` means the curvature constraint was
    incorporated, not that every marginal became strictly narrower. Failed
    coefficient arithmetic returns unavailable with the coarse clock report
    retained; the parent refined outer bounds then remain absent.
    """

    bulk_angle_bounds: tuple[Q, Q]
    receiver_short_angle_bounds: tuple[Q, Q]
    form_radius: Q
    phase_radius: Q
    phase_increments: tuple[Q, Q]
    probe_duration: Q
    recorded_reading_bounds: tuple[tuple[Q, Q], ...]
    readout_error_bound: Q
    readout_gain_bounds: tuple[Q, Q]
    clock_rate_bounds: tuple[Q, Q]
    clock_envelope: SineClockInference
    observation_times: tuple[Q, Q, Q, Q]
    reading_midpoints: tuple[Q, Q, Q, Q]
    reading_radii: tuple[Q, Q, Q, Q]
    curvature_midpoint: Q
    curvature_observation_radius: Q
    sensor_corrected_curvature_bounds: I
    embedded_inverse_reading_coefficients: tuple[tuple[I, I, I, I], ...] | None
    first_window_sine_norm_candidate: Q
    initial_curvature_error_candidate: Q
    third_derivative_bound_candidate: Q
    normalized_curvature_error_candidate: Q
    normalized_curvature_error_upper_bound: Q | None
    effective_gain_denominator_bounds: tuple[Q, Q] | None
    curvature_coefficient_bounds: I | None
    normalized_curvature_observation_bounds: I | None
    clock_rate_constraint_bounds: I | None
    nominal_bulk_angle_outer_bounds: I | None
    actual_long_arc_mean_outer_bounds: I | None
    effective_gain_outer_bounds: I | None
    readout_gain_outer_bounds: I | None
    clock_rate_outer_bounds: I | None
    source_admitted: bool
    finite_curvature_bound_certified: bool
    base_inverse_enclosure_available: bool
    curvature_coefficient_positive: bool
    curvature_refinement_available: bool
    inverse_enclosure_available: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    incompatibility_reasons: tuple[str, ...]
    curvature_reading_coefficients: tuple[Q, Q, Q, Q] = (Q(1), Q(-2), Q(1), Q(0))
    arithmetic_method: str = INTERVAL_METHOD
    clock: str = "tau=rho*s; rho is one held positive rate"
    effective_gain_relation: str = "J=G*rho"
    scope: tuple[str, ...] = (
        "four_readings_on_one_complete_two_pulse_trajectory_at_zero_H_over_two_H_two_H",
        "first_and_second_phase_events_remain_at_zero_and_H_without_reset",
        "finite_second_difference_is_not_a_supplied_derivative_observation",
        "all_four_readings_are_admitted_before_the_three_reading_clock_envelope",
        "exact_midpoint_second_difference_cancels_arbitrary_held_offset",
        "four_reading_error_coefficients_are_retained_before_marginal_relaxation",
        "complete_initial_state_error_and_third_derivative_remainder_are_retained",
        "clock_refinement_uses_positive_factors_and_exact_endpoint_division",
        "necessary_constraint_intersection_does_not_assert_joint_error_or_state_realizability",
        "original_angle_and_actual_pre_probe_arc_mean_remain_the_child_statistics",
        "unavailable_refinement_retains_the_coarse_clock_report_separately",
        "no_clock_grid_shooting_repeated_fit_or_additional_response_evaluation",
        "no_physical_calibration_source_acquisition_or_unique_law_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-curvature-inference.v1", "report": _project(self)}


def infer_sine_geometry_gain_clock_curvature(
    *,
    bulk_angle_bounds,
    receiver_short_angle_bounds,
    form_radius,
    phase_radius,
    phase_increments,
    probe_duration,
    recorded_reading_bounds,
    readout_error_bound,
    readout_gain_bounds,
    clock_rate_bounds,
) -> SineCurvatureInference:
    """Refine necessary clock/gain marginals using four primitive readings.

    Domains match the clock inverse except that exactly four reading pairs
    are required, ordered at 0,H/2,H,2H in observed time. The first-window
    curvature midpoint is c_H-2*c_half+c_0; its observation radius is
    tau_0+2*tau_half+tau_H+4*delta. These combinations are exact before any
    interval normalization. The last reading has zero curvature coefficient.

    The additional constraint is Z=rho*K(b,a1) plus an admitted finite error,
    where Z=4*C/(H^2*J), J=G*rho. Positive effective-gain and clock domains
    are intersected before exact endpoint division. No observed derivative,
    cached incoming report or intermediate state is accepted.
    """
    raw = _ordered(recorded_reading_bounds, "recorded_reading_bounds", limit=5)
    if len(raw) != 4:
        raise ValueError(
            "recorded_reading_bounds must contain exactly four endpoint pairs"
        )
    readings = tuple(
        _primitive_bounds(pair, f"recorded_reading_bounds[{index}]")
        for index, pair in enumerate(raw)
    )
    child = infer_sine_geometry_gain_clock(
        bulk_angle_bounds=bulk_angle_bounds,
        receiver_short_angle_bounds=receiver_short_angle_bounds,
        form_radius=form_radius,
        phase_radius=phase_radius,
        phase_increments=phase_increments,
        probe_duration=probe_duration,
        recorded_reading_bounds=tuple(readings[index] for index in (0, 2, 3)),
        readout_error_bound=readout_error_bound,
        readout_gain_bounds=readout_gain_bounds,
        clock_rate_bounds=clock_rate_bounds,
    )
    x, y, horizon = child.form_radius, child.phase_radius, child.probe_duration
    a1 = child.phase_increments[0]
    rate_lo, rate_hi = child.clock_rate_bounds
    gain_lo, gain_hi = child.readout_gain_bounds
    structural_hi = child.structural_probe_duration_bounds[1]
    base = child.constraint_envelope
    qmax, g = base.whole_window_form_norm_candidate, _GAMMA_UPPER
    sine_bound = min(Q(7), 4 + 4 * a1 + 2 * y + 8 * g * structural_hi * qmax)
    initial_error = 4 * (1 + g * g) * x + 4 * g * y
    third_bound = (
        8 * (1 + 2 * g * g) * qmax
        + 4 * g * (1 + g * g) * sine_bound
        + 8 * g**3 * qmax * qmax
    )
    error = rate_hi * initial_error + rate_hi**2 * horizon * third_bound / 2
    midpoints = tuple((lo + hi) / 2 for lo, hi in readings)
    radii = tuple((hi - lo) / 2 for lo, hi in readings)
    midpoint = midpoints[2] - 2 * midpoints[1] + midpoints[0]
    observation_radius = (
        radii[0] + 2 * radii[1] + radii[2] + 4 * child.readout_error_bound
    )
    corrected = I(midpoint - observation_radius, midpoint + observation_radius)
    embedded = None
    if base.transformed_reading_coefficients is not None:
        embedded = tuple(
            (row[0], I(0), row[1], row[2])
            for row in base.transformed_reading_coefficients
        )
    coefficient = normalized = rate_constraint = denominator = None
    nominal = actual = effective = inferred_gain = inferred_rate = None
    coefficient_positive = False
    status = child.status
    unavailable, incompatible = list(child.unavailable_reasons), list(
        child.incompatibility_reasons
    )
    if child.inverse_enclosure_available:
        effective_lo = max(child.effective_gain_outer_bounds.lo, gain_lo * rate_lo)
        effective_hi = min(child.effective_gain_outer_bounds.hi, gain_hi * rate_hi)
        coarse_rate_lo = max(rate_lo, child.clock_rate_outer_bounds.lo)
        coarse_rate_hi = min(rate_hi, child.clock_rate_outer_bounds.hi)
        coarse_gain_lo = max(gain_lo, child.readout_gain_outer_bounds.lo)
        coarse_gain_hi = min(gain_hi, child.readout_gain_outer_bounds.hi)
        denominator = (effective_lo, effective_hi)
        angle = child.nominal_bulk_angle_outer_bounds
        amplitude = I(a1)
        coefficient = base.gamma_bounds * (
            2 * sin(I(a1 / 2)) ** 2 * (1 + 3 * cos(amplitude)) * sin(angle)
            + sin(amplitude) * (2 + 3 * cos(amplitude)) * cos(angle)
        )
        coefficient_positive = coefficient.lo > 0
        if (
            effective_lo > effective_hi
            or coarse_rate_lo > coarse_rate_hi
            or coarse_gain_lo > coarse_gain_hi
        ):
            status = "incompatible"
            incompatible.append(
                "coarse_clock_marginals_exclude_their_positive_primitive_domains"
            )
        elif not coefficient_positive:
            status = "unavailable"
            unavailable.append(
                "positive_curvature_coefficient_is_unresolved_at_interval_precision"
            )
        else:
            # Normalize exact midpoint endpoints before I construction. In
            # particular, neither H^2 nor a tiny positive J becomes a rounded
            # interval denominator containing zero.
            normalized = _positive_quotient_bounds(
                (
                    4 * (midpoint - observation_radius) / horizon**2,
                    4 * (midpoint + observation_radius) / horizon**2,
                ),
                denominator,
            )
            rate_constraint = _positive_quotient_bounds(
                (normalized.lo - error, normalized.hi + error),
                (coefficient.lo, coefficient.hi),
            )
            refined_lo = max(coarse_rate_lo, rate_constraint.lo)
            refined_hi = min(coarse_rate_hi, rate_constraint.hi)
            if refined_lo > refined_hi:
                status = "incompatible"
                incompatible.append(
                    "finite_curvature_excludes_the_clock_prior_and_coarse_envelope"
                )
            else:
                refined_gain_lo = max(coarse_gain_lo, effective_lo / refined_hi)
                refined_gain_hi = min(coarse_gain_hi, effective_hi / refined_lo)
                if refined_gain_lo > refined_gain_hi:
                    status = "incompatible"
                    incompatible.append(
                        "curvature_clock_and_effective_gain_exclude_the_gain_prior"
                    )
                else:
                    nominal, actual = (
                        child.nominal_bulk_angle_outer_bounds,
                        child.actual_long_arc_mean_outer_bounds,
                    )
                    effective = child.effective_gain_outer_bounds
                    inferred_rate = I(refined_lo, refined_hi)
                    inferred_gain = I(refined_gain_lo, refined_gain_hi)
                    status = "bounded_candidate"
    available = status == "bounded_candidate"
    return SineCurvatureInference(
        bulk_angle_bounds=child.bulk_angle_bounds,
        receiver_short_angle_bounds=child.receiver_short_angle_bounds,
        form_radius=x,
        phase_radius=y,
        phase_increments=child.phase_increments,
        probe_duration=horizon,
        recorded_reading_bounds=readings,
        readout_error_bound=child.readout_error_bound,
        readout_gain_bounds=child.readout_gain_bounds,
        clock_rate_bounds=child.clock_rate_bounds,
        clock_envelope=child,
        observation_times=(Q(0), horizon / 2, horizon, 2 * horizon),
        reading_midpoints=midpoints,
        reading_radii=radii,
        curvature_midpoint=midpoint,
        curvature_observation_radius=observation_radius,
        sensor_corrected_curvature_bounds=corrected,
        embedded_inverse_reading_coefficients=embedded,
        first_window_sine_norm_candidate=sine_bound,
        initial_curvature_error_candidate=initial_error,
        third_derivative_bound_candidate=third_bound,
        normalized_curvature_error_candidate=error,
        normalized_curvature_error_upper_bound=error if child.source_admitted else None,
        effective_gain_denominator_bounds=denominator,
        curvature_coefficient_bounds=coefficient,
        normalized_curvature_observation_bounds=normalized,
        clock_rate_constraint_bounds=rate_constraint,
        nominal_bulk_angle_outer_bounds=nominal,
        actual_long_arc_mean_outer_bounds=actual,
        effective_gain_outer_bounds=effective,
        readout_gain_outer_bounds=inferred_gain,
        clock_rate_outer_bounds=inferred_rate,
        source_admitted=child.source_admitted,
        finite_curvature_bound_certified=child.source_admitted,
        base_inverse_enclosure_available=child.inverse_enclosure_available,
        curvature_coefficient_positive=coefficient_positive,
        curvature_refinement_available=available,
        inverse_enclosure_available=available,
        status=status,
        unavailable_reasons=tuple(unavailable),
        incompatibility_reasons=tuple(incompatible),
    )
