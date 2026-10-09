"""Response-free sufficient resolution budgets for the fixed aperture model.

Only declared source, sensor, numerical and clock-drift budgets are consumed.
The exact rational corollary does not run an inverse, select a response or
certify the outward arithmetic of an eventual numerical inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from .relational_sine_two_port_inference import (
    _BULK_DOMAIN,
    _GAMMA_UPPER,
    _RECEIVER_DOMAIN,
)

__all__ = ("SineApertureBudget", "assess_sine_aperture_budget")


@dataclass(frozen=True)
class SineApertureBudget:
    """Conditional ideal width bounds, independent of any observed response.

    A common half-width t bounds each supplied recorded-average enclosure,
    including propagated source uncertainty and numerical enclosure error.
    It is a supplied bound, not a separately proved solver-error estimate;
    the four additive sensor errors bounded by delta remain distinct.
    The fixed chart, pulse pair, gain/rate priors and targets are explicit
    model/policy choices. Positive source and quotient margins admit the
    sufficient corollary; passing its targets does not supply an observation
    or guarantee executable inverse availability at finite precision.
    """

    probe_duration: Q
    form_radius: Q
    phase_radius: Q
    readout_error_bound: Q
    averaged_reading_halfwidth_bound: Q
    clock_rate_derivative_bound: Q
    combined_average_error_bound: Q
    source_phase_margin: Q
    whole_window_form_norm_bound: Q
    structural_readout_speed_bound: Q
    second_derivative_bound: Q
    third_derivative_bound: Q
    clock_average_error_scale: Q
    endpoint_reconstruction_error_bound: Q
    midpoint_reconstruction_error_bound: Q
    second_window_reconstruction_error_bound: Q
    virtual_numerical_error_radii: tuple[Q, ...]
    virtual_sensor_error_radii: tuple[Q, ...]
    virtual_clock_error_radii: tuple[Q, ...]
    virtual_reconstruction_error_radii: tuple[Q, ...]
    virtual_point_error_radii: tuple[Q, ...]
    specialized_form_norm_bound: Q
    finite_response_error_bound: Q
    transformed_error_radius: Q
    transformed_radius_margin: Q
    specialized_sine_norm_bound: Q
    specialized_third_derivative_bound: Q
    initial_curvature_error_bound: Q
    normalized_curvature_error_bound: Q
    curvature_observation_error_term: Q
    curvature_clock_drift_term: Q
    curvature_reconstruction_term: Q
    curvature_numerator_diameter_bound: Q
    curvature_numerator_lower_candidate: Q
    corrected_curvature_numerator_margin: Q
    effective_gain_width_candidate: Q
    nominal_angle_width_candidate: Q | None
    actual_angle_width_candidate: Q | None
    mean_clock_width_candidate: Q | None
    readout_gain_width_candidate: Q | None
    effective_gain_width_upper_bound: Q | None
    nominal_angle_width_upper_bound: Q | None
    actual_angle_width_upper_bound: Q | None
    mean_clock_width_upper_bound: Q | None
    readout_gain_width_upper_bound: Q | None
    noise_overlap_sensor_error_threshold: Q
    noise_overlap_witness_admitted: bool
    source_chart_eligible: bool
    transformed_radius_eligible: bool
    curvature_quotients_eligible: bool
    sufficient_bound_eligible: bool
    sufficient_resolution_certified: bool
    status: str
    ineligibility_reasons: tuple[str, ...]
    unmet_resolution_targets: tuple[str, ...]
    bulk_angle_bounds: tuple[Q, Q] = _BULK_DOMAIN
    receiver_short_angle_bounds: tuple[Q, Q] = _RECEIVER_DOMAIN
    phase_increments: tuple[Q, Q] = (Q(1, 4), Q(3, 4))
    readout_gain_bounds: tuple[Q, Q] = (Q(1), Q(2))
    clock_rate_bounds: tuple[Q, Q] = (Q(1, 2), Q(2))
    source_phase_margin_lower_bound: Q = Q(1, 256)
    natural_curvature_coefficient_lower_bound: Q = Q(1, 18000)
    actual_angle_width_target: Q = Q(1, 1024)
    effective_gain_width_target: Q = Q(1, 2048)
    readout_gain_width_target: Q = Q(1, 16)
    mean_clock_width_target: Q = Q(1, 64)
    noise_overlap_gain_rate_pairs: tuple[tuple[Q, Q], ...] = (
        (Q(3, 2), Q(1)),
        (Q(1), Q(3, 2)),
    )
    noise_overlap_common_effective_gain: Q = Q(3, 2)
    noise_overlap_required_gain_and_mean_clock_width: Q = Q(1, 2)
    arithmetic_method: str = (
        "exact_rational_sufficient_aperture_resolution_corollary_v1"
    )
    inferred_clock_statistic: str = "rho_bar_1=(1/H)*integral_0^H rho(s) ds"
    effective_gain_relation: str = "J=G*rho_bar_1"
    scope: tuple[str, ...] = (
        "fixed_complete_two_port_law_chart_pulse_pair_and_gain_rate_priors",
        "fixed_normalized_boxcars_over_first_thirds_and_full_second_window",
        "only_declared_source_sensor_numerical_and_C1_clock_drift_budgets",
        "common_numerical_halfwidth_and_sensor_error_are_separate_primitives",
        "source_chart_guard_uses_pi_less_than_355_over_113_and_Y_less_than_1_over_256",
        "ideal_exact_arithmetic_conditional_width_bounds_without_readings",
        "positive_source_chart_transformed_radius_and_curvature_quotient_margins_are_required",
        "passing_targets_does_not_prove_compatible_data_or_joint_realizability",
        "eventual_outward_inverse_availability_and_width_need_separate_checks",
        "failed_sufficient_certificate_is_not_nonidentifiability_or_impossibility",
        "separate_noise_overlap_witness_is_existential_for_some_admissible_common_record",
        "witness_uses_zero_source_residuals_constant_clocks_same_geometry_and_held_offset",
        "witness_compares_only_physical_sensor_error_not_numerical_halfwidth",
        "witness_gain_and_mean_rate_pair_diameter_does_not_apply_to_every_record",
        "zero_residual_witness_does_not_require_certification_of_the_entire_source_ball",
        "no_response_fitting_horizon_search_producer_inverse_or_cached_report_input",
        "no_sensor_calibration_source_acquisition_or_physical_clock_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-aperture-budget.v1", "report": _project(self)}


def assess_sine_aperture_budget(
    *,
    probe_duration,
    form_radius,
    phase_radius,
    readout_error_bound,
    averaged_reading_halfwidth_bound,
    clock_rate_derivative_bound,
) -> SineApertureBudget:
    """Evaluate a fixed-model sufficient budget without selecting observations.

    All six primitives are mandatory. Require 0<H<=1/4 and finite nonnegative
    source radii X,Y, sensor error delta, common numerical half-width t and
    clock derivative bound Lambda. Fractions remain exact; other real inputs
    use shared represented-real admission. Boolean and nonfinite values reject.

    The source chart requires Y<1/256. The full-prior donor short-edge margin
    is 11-7*pi/2>1/226>1/256, using pi<355/113; all other edge margins are
    larger. Further strict radius and quotient guards precede certified width
    fields. Failed guards or resolution targets are explicit abstentions of
    this sufficient method, not lower bounds on achievable inference error.
    The separate noise-overlap witness uses zero residuals within every
    admitted nonnegative source budget, including balls that fail the
    sufficient chart guard. Its threshold consumes sensor error alone.
    """
    h, x, y, noise, numerical, drift = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (probe_duration, "probe_duration"),
            (form_radius, "form_radius"),
            (phase_radius, "phase_radius"),
            (readout_error_bound, "readout_error_bound"),
            (averaged_reading_halfwidth_bound, "averaged_reading_halfwidth_bound"),
            (clock_rate_derivative_bound, "clock_rate_derivative_bound"),
        )
    )
    if not 0 < h <= Q(1, 4):
        raise ValueError("probe_duration must lie in (0,1/4]")
    if min(x, y, noise, numerical, drift) < 0:
        raise ValueError(
            "source radii and sensor, numerical and clock-drift bounds must be nonnegative"
        )

    g, epsilon = _GAMMA_UPPER, noise + numerical
    source_margin = Q(1, 256) - y
    q0 = x + 28 * g * h
    speed = 2 * q0 + 7 * g
    m2 = 4 * (1 + g * g) * q0 + 14 * g
    m3 = 8 * (1 + 2 * g * g) * q0 + 28 * g * (1 + g * g) + 8 * g**3 * q0**2
    clock = 2 * speed * drift * h * h
    endpoint = 4 * m3 * h**3 / 27
    midpoint = m3 * h**3 / 144
    last = 4 * m2 * h * h / 3
    row_norms = (Q(10, 3), Q(7, 6), Q(10, 3), Q(16, 3))
    numerical_radii = tuple(value * numerical for value in row_norms)
    sensor_radii = tuple(value * noise for value in row_norms)
    clock_radii = tuple(
        value * clock for value in (Q(91, 324), Q(11, 81), Q(91, 324), Q(361, 324))
    )
    reconstruction = (endpoint, midpoint, endpoint, endpoint + last)
    radii = tuple(
        sum(values)
        for values in zip(numerical_radii, sensor_radii, clock_radii, reconstruction)
    )

    qs = (x + 4 * g * h * (7 + 2 * y)) / (1 - 64 * g * g * h * h)
    finite_error = 4 * h * qs + 4 * g * h * y + 32 * g * g * h * h * qs
    radius = 18000 * (2 * finite_error + radii[2] + radii[3]) / h
    radius_margin = Q(1, 8) - radius
    sine_bound = min(Q(7), 5 + 2 * y + 16 * g * h * qs)
    special_m3 = (
        8 * (1 + 2 * g * g) * qs + 4 * g * (1 + g * g) * sine_bound + 8 * g**3 * qs**2
    )
    initial_error = 4 * (1 + g * g) * x + 4 * g * y
    curvature_error = 2 * initial_error + 2 * h * special_m3
    observation_term = 72 * epsilon / (h * h)
    drift_term = Q(40, 3) * speed * drift
    reconstruction_term = Q(67, 27) * m3 * h
    diameter = observation_term + drift_term + reconstruction_term
    k0 = Q(1, 18000)
    numerator_lower = (k0 / 2 - curvature_error) / 2 - diameter
    corrected_margin = numerator_lower / 4 - curvature_error

    wj = 4 * radius
    wb = actual = wrate = wgain = None
    if radius_margin > 0:
        wb = 2 * radius / (Q(1, 4) - 2 * radius)
        actual = wb + y / 4
        relative = 11 * g * wb / (8 * k0)
        product = 2 * wj + relative + 2 * wj * relative
        wrate = (
            2 * diameter / k0
            + (2 + curvature_error / k0) * product
            + 2 * curvature_error / k0
        )
        wgain = 2 * wj + 4 * wrate
    source_ok, radius_ok = source_margin > 0, radius_margin > 0
    quotient_ok = numerator_lower > 0 and corrected_margin > 0
    eligible = source_ok and radius_ok and quotient_ok
    ineligible = tuple(
        label
        for valid, label in (
            (source_ok, "source_phase_radius_not_below_fixed_chart_margin"),
            (radius_ok, "transformed_error_radius_not_below_one_eighth"),
            (numerator_lower > 0, "curvature_numerator_lower_bound_not_positive"),
            (corrected_margin > 0, "corrected_curvature_numerator_margin_not_positive"),
        )
        if not valid
    )
    targets = (
        ("actual_angle", actual, Q(1, 1024)),
        ("effective_gain", wj, Q(1, 2048)),
        ("readout_gain", wgain, Q(1, 16)),
        ("mean_clock", wrate, Q(1, 64)),
    )
    unmet = tuple(
        name for name, value, target in targets if value is None or value >= target
    )
    certified = eligible and not unmet
    overlap_threshold = Q(245, 8) * g * h**2 + Q(1365, 32) * g**3 * h**3
    return SineApertureBudget(
        probe_duration=h,
        form_radius=x,
        phase_radius=y,
        readout_error_bound=noise,
        averaged_reading_halfwidth_bound=numerical,
        clock_rate_derivative_bound=drift,
        combined_average_error_bound=epsilon,
        source_phase_margin=source_margin,
        whole_window_form_norm_bound=q0,
        structural_readout_speed_bound=speed,
        second_derivative_bound=m2,
        third_derivative_bound=m3,
        clock_average_error_scale=clock,
        endpoint_reconstruction_error_bound=endpoint,
        midpoint_reconstruction_error_bound=midpoint,
        second_window_reconstruction_error_bound=last,
        virtual_numerical_error_radii=numerical_radii,
        virtual_sensor_error_radii=sensor_radii,
        virtual_clock_error_radii=clock_radii,
        virtual_reconstruction_error_radii=reconstruction,
        virtual_point_error_radii=radii,
        specialized_form_norm_bound=qs,
        finite_response_error_bound=finite_error,
        transformed_error_radius=radius,
        transformed_radius_margin=radius_margin,
        specialized_sine_norm_bound=sine_bound,
        specialized_third_derivative_bound=special_m3,
        initial_curvature_error_bound=initial_error,
        normalized_curvature_error_bound=curvature_error,
        curvature_observation_error_term=observation_term,
        curvature_clock_drift_term=drift_term,
        curvature_reconstruction_term=reconstruction_term,
        curvature_numerator_diameter_bound=diameter,
        curvature_numerator_lower_candidate=numerator_lower,
        corrected_curvature_numerator_margin=corrected_margin,
        effective_gain_width_candidate=wj,
        nominal_angle_width_candidate=wb,
        actual_angle_width_candidate=actual,
        mean_clock_width_candidate=wrate,
        readout_gain_width_candidate=wgain,
        effective_gain_width_upper_bound=wj if eligible else None,
        nominal_angle_width_upper_bound=wb if eligible else None,
        actual_angle_width_upper_bound=actual if eligible else None,
        mean_clock_width_upper_bound=wrate if eligible else None,
        readout_gain_width_upper_bound=wgain if eligible else None,
        noise_overlap_sensor_error_threshold=overlap_threshold,
        noise_overlap_witness_admitted=noise >= overlap_threshold,
        source_chart_eligible=source_ok,
        transformed_radius_eligible=radius_ok,
        curvature_quotients_eligible=quotient_ok,
        sufficient_bound_eligible=eligible,
        sufficient_resolution_certified=certified,
        status="certified_sufficient_budget" if certified else "not_certified",
        ineligibility_reasons=ineligible,
        unmet_resolution_targets=unmet,
    )
