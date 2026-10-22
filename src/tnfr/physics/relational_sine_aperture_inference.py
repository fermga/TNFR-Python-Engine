"""Necessary geometry/gain/mean-clock constraints from four finite boxcars.

Exact quadratic moments reconstruct a constant-mean comparison history.
Structural derivative bounds are applied separately on either side of the
phase event; no observed-clock third derivative or point sensor is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from .relational_observations import _ordered
from .relational_sine_curvature_inference import (
    SineCurvatureInference,
    infer_sine_geometry_gain_clock_curvature,
)
from .relational_sine_two_port_inference import (
    _BULK_DOMAIN,
    _GAMMA_UPPER,
    _RECEIVER_DOMAIN,
    _primitive_bounds,
)

__all__ = ("SineApertureInference", "infer_sine_geometry_gain_clock_aperture")

_RECONSTRUCTION = (
    (Q(11, 6), Q(-7, 6), Q(1, 3), Q(0)),
    (Q(-1, 24), Q(13, 12), Q(-1, 24), Q(0)),
    (Q(1, 3), Q(-7, 6), Q(11, 6), Q(0)),
    (Q(-1, 3), Q(7, 6), Q(-11, 6), Q(2)),
)


@dataclass(frozen=True)
class SineApertureInference:
    """Necessary marginals for normalized boxcar observations of one history.

    The four apertures are [0,H/3], [H/3,2H/3], [2H/3,H], [H,2H].
    A single positive gain and held offset apply to all averages; each has
    one additive error bounded by the original sensor allowance. The clock
    is positive C1 with supplied global rate and derivative bounds.

    ``reference_envelope`` consumes virtual noiseless point bands for a
    constant first-window-mean clock. Its zero sensor allowance does not
    remove the original four errors: their complete linear images already
    widen those bands. Marginal interval propagation may discard their
    correlations and does not establish a joint realization.
    """

    bulk_angle_bounds: tuple[Q, Q]
    receiver_short_angle_bounds: tuple[Q, Q]
    form_radius: Q
    phase_radius: Q
    phase_increments: tuple[Q, Q]
    probe_duration: Q
    averaged_reading_bounds: tuple[tuple[Q, Q], ...]
    readout_error_bound: Q
    readout_gain_bounds: tuple[Q, Q]
    clock_rate_bounds: tuple[Q, Q]
    clock_rate_derivative_bound: Q
    reference_envelope: SineCurvatureInference
    aperture_windows: tuple[tuple[Q, Q], ...]
    virtual_observation_times: tuple[Q, Q, Q, Q]
    reconstruction_matrix: tuple[tuple[Q, ...], ...]
    averaged_reading_midpoints: tuple[Q, ...]
    averaged_reading_radii: tuple[Q, ...]
    structural_duration_upper_bound: Q
    whole_window_form_norm_candidate: Q
    whole_window_form_norm_upper_bound: Q | None
    structural_readout_speed_candidate: Q
    structural_readout_speed_upper_bound: Q | None
    second_derivative_bound_candidate: Q
    second_derivative_bound_upper_bound: Q | None
    third_derivative_bound_candidate: Q
    third_derivative_bound_upper_bound: Q | None
    averaged_exposure_discrepancy_bounds: tuple[Q, ...]
    averaged_clock_discrepancy_candidates: tuple[Q, ...]
    averaged_clock_discrepancy_upper_bounds: tuple[Q, ...] | None
    projected_numerical_radii: tuple[Q, ...]
    projected_sensor_error_radii: tuple[Q, ...]
    projected_clock_discrepancy_candidates: tuple[Q, ...]
    projected_clock_discrepancy_upper_bounds: tuple[Q, ...] | None
    reconstruction_error_candidates: tuple[Q, ...]
    reconstruction_error_upper_bounds: tuple[Q, ...] | None
    virtual_reading_midpoints: tuple[Q, ...]
    point_reference_reading_bounds: tuple[tuple[Q, Q], ...]
    original_curvature_average_coefficients: tuple[Q, Q, Q, Q]
    embedded_inverse_average_coefficients: tuple[tuple[I, ...], ...] | None
    nominal_bulk_angle_outer_bounds: I | None
    actual_long_arc_mean_outer_bounds: I | None
    effective_gain_outer_bounds: I | None
    readout_gain_outer_bounds: I | None
    first_window_mean_clock_rate_outer_bounds: I | None
    source_admitted: bool
    clock_drift_transfer_certified: bool
    aperture_reconstruction_certified: bool
    finite_curvature_bound_certified: bool
    rank_deficient: bool
    rank_certified: bool
    inverse_enclosure_available: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    incompatibility_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    clock: str = "tau(s)=integral_0^s rho(u) du; positive C1 rho with supplied bounds"
    inferred_clock_statistic: str = "rho_bar_1=(1/H)*integral_0^H rho(s) ds"
    effective_gain_relation: str = "J=G*rho_bar_1"
    complete_observed_rows: tuple[str, str] = (
        "dx/ds=rho(s)*(-A*x+gamma*f(theta))",
        "dtheta/ds=rho(s)*gamma*A*x",
    )
    scope: tuple[str, ...] = (
        "four_fixed_normalized_boxcar_averages_are_the_original_observations",
        "one_held_positive_gain_and_offset_and_four_scalar_sensor_errors",
        "global_positive_C1_clock_rate_and_derivative_bounds_are_supplied_premises",
        "original_complete_source_and_phase_events_at_zero_and_H_are_preserved",
        "form_is_continuous_at_H_but_derivative_bounds_are_applied_on_each_side",
        "no_prehistory_posthorizon_data_or_observed_clock_third_derivative",
        "constant_first_window_mean_reference_has_the_same_complete_event_states",
        "quadratic_moments_reconstruct_first_three_points_with_cubic_remainders",
        "last_point_uses_second_window_endpoint_identity_with_quadratic_remainder",
        "virtual_point_bands_already_include_all_original_sensor_errors",
        "reference_zero_sensor_allowance_refers_to_noiseless_comparison_points",
        "explicit_error_coefficient_maps_do_not_restore_discarded_joint_correlations",
        "reference_envelope_is_a_necessary_constraint_not_an_observed_point_trajectory",
        "first_window_mean_clock_rate_is_not_an_instantaneous_rate",
        "actual_long_arc_mean_retains_original_phase_radius_over_eight",
        "marginals_do_not_certify_joint_state_sensor_or_clock_profile_realizability",
        "no_response_producer_cached_report_or_physical_calibration_input",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-aperture-inference.v1", "report": _project(self)}


def infer_sine_geometry_gain_clock_aperture(
    *,
    bulk_angle_bounds,
    receiver_short_angle_bounds,
    form_radius,
    phase_radius,
    phase_increments,
    probe_duration,
    averaged_reading_bounds,
    readout_error_bound,
    readout_gain_bounds,
    clock_rate_bounds,
    clock_rate_derivative_bound,
) -> SineApertureInference:
    """Infer necessary marginals from fixed finite-aperture observations.

    All eleven inputs are mandatory primitives. Domains match the bounded
    clock-drift inverse except that four averaged reading pairs replace its
    point readings. Each supplied interval encloses a recorded normalized
    average over the corresponding fixed aperture; additive sensor error is
    separate. Require H>0 and rho_plus*H<=1/2.

    Exact rational moments and comparison allowances produce four virtual
    point bands. One fresh curvature inverse consumes these with zero new
    sensor error. Its rate output refers to the actual first-window mean.
    Source-independent arithmetic candidates are certified only after that
    child re-admits the original phase chart and source norm budgets.
    """
    b = _primitive_bounds(bulk_angle_bounds, "bulk_angle_bounds")
    receiver = _primitive_bounds(
        receiver_short_angle_bounds, "receiver_short_angle_bounds"
    )
    amplitudes = _primitive_bounds(phase_increments, "phase_increments")
    gain = _primitive_bounds(readout_gain_bounds, "readout_gain_bounds")
    rates = _primitive_bounds(clock_rate_bounds, "clock_rate_bounds")
    raw = _ordered(averaged_reading_bounds, "averaged_reading_bounds", limit=5)
    if len(raw) != 4:
        raise ValueError(
            "averaged_reading_bounds must contain exactly four endpoint pairs"
        )
    readings = tuple(
        _primitive_bounds(pair, f"averaged_reading_bounds[{index}]")
        for index, pair in enumerate(raw)
    )
    x, y, horizon, noise, drift = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_radius, "form_radius"),
            (phase_radius, "phase_radius"),
            (probe_duration, "probe_duration"),
            (readout_error_bound, "readout_error_bound"),
            (clock_rate_derivative_bound, "clock_rate_derivative_bound"),
        )
    )
    if not _BULK_DOMAIN[0] <= b[0] <= b[1] <= _BULK_DOMAIN[1]:
        raise ValueError("bulk_angle_bounds must lie within [11/8,3/2] radians")
    if not _RECEIVER_DOMAIN[0] <= receiver[0] <= receiver[1] <= _RECEIVER_DOMAIN[1]:
        raise ValueError("receiver_short_angle_bounds must lie within [2/3,1] radians")
    if min(x, y, noise, drift) < 0:
        raise ValueError(
            "form_radius, phase_radius, readout_error_bound and clock_rate_derivative_bound must be nonnegative"
        )
    if not 0 < amplitudes[0] <= amplitudes[1] <= 1:
        raise ValueError("cumulative phase_increments must satisfy 0<a1<=a2<=1")
    if gain[0] <= 0 or rates[0] <= 0:
        raise ValueError(
            "readout_gain_bounds and clock_rate_bounds must be strictly positive"
        )
    if horizon <= 0 or rates[1] * horizon > Q(1, 2):
        raise ValueError(
            "require positive probe_duration and clock_upper*duration<=1/2"
        )

    hstar, g = rates[1] * horizon, _GAMMA_UPPER
    form_bound = x + 14 * g * hstar
    speed = 2 * form_bound + 7 * g
    second = 4 * (1 + g * g) * form_bound + 14 * g
    third = (
        8 * (1 + 2 * g * g) * form_bound
        + 28 * g * (1 + g * g)
        + 8 * g**3 * form_bound**2
    )
    rate_range = rates[1] - rates[0]
    exposure = tuple(
        min(coefficient * drift * horizon**2, 2 * coefficient * rate_range * horizon)
        for coefficient in (Q(7, 108), Q(13, 108), Q(7, 108))
    ) + (
        min(5 * drift * horizon**2 / 12, rate_range * horizon / 2),
    )
    clock_average = tuple(gain[1] * speed * radius for radius in exposure)
    midpoints = tuple((lo + hi) / 2 for lo, hi in readings)
    radii = tuple((hi - lo) / 2 for lo, hi in readings)
    virtual = tuple(
        sum(w * c for w, c in zip(row, midpoints)) for row in _RECONSTRUCTION
    )

    def project_radii(values):
        return tuple(
            sum(abs(w) * value for w, value in zip(row, values))
            for row in _RECONSTRUCTION
        )

    numerical = project_radii(radii)
    sensor = project_radii((noise,) * 4)
    clock = project_radii(clock_average)
    endpoint = gain[1] * third * hstar**3 / 108
    reconstruction = (
        endpoint,
        gain[1] * third * hstar**3 / 2304,
        endpoint,
        endpoint + gain[1] * second * hstar**2 / 6,
    )
    point_radii = tuple(
        sum(parts) for parts in zip(numerical, sensor, clock, reconstruction)
    )
    comparison = tuple(
        (c - radius, c + radius) for c, radius in zip(virtual, point_radii)
    )
    reference = infer_sine_geometry_gain_clock_curvature(
        bulk_angle_bounds=b,
        receiver_short_angle_bounds=receiver,
        form_radius=x,
        phase_radius=y,
        phase_increments=amplitudes,
        probe_duration=horizon,
        recorded_reading_bounds=comparison,
        readout_error_bound=Q(0),
        readout_gain_bounds=gain,
        clock_rate_bounds=rates,
    )
    embedded = None
    if reference.embedded_inverse_reading_coefficients is not None:
        embedded = tuple(
            tuple(
                sum((row[k] * _RECONSTRUCTION[k][j] for k in range(4)), I(0))
                for j in range(4)
            )
            for row in reference.embedded_inverse_reading_coefficients
        )
    admitted = reference.source_admitted
    return SineApertureInference(
        bulk_angle_bounds=b,
        receiver_short_angle_bounds=receiver,
        form_radius=x,
        phase_radius=y,
        phase_increments=amplitudes,
        probe_duration=horizon,
        averaged_reading_bounds=readings,
        readout_error_bound=noise,
        readout_gain_bounds=gain,
        clock_rate_bounds=rates,
        clock_rate_derivative_bound=drift,
        reference_envelope=reference,
        aperture_windows=(
            (Q(0), horizon / 3),
            (horizon / 3, 2 * horizon / 3),
            (2 * horizon / 3, horizon),
            (horizon, 2 * horizon),
        ),
        virtual_observation_times=(Q(0), horizon / 2, horizon, 2 * horizon),
        reconstruction_matrix=_RECONSTRUCTION,
        averaged_reading_midpoints=midpoints,
        averaged_reading_radii=radii,
        structural_duration_upper_bound=2 * hstar,
        whole_window_form_norm_candidate=form_bound,
        whole_window_form_norm_upper_bound=form_bound if admitted else None,
        structural_readout_speed_candidate=speed,
        structural_readout_speed_upper_bound=speed if admitted else None,
        second_derivative_bound_candidate=second,
        second_derivative_bound_upper_bound=second if admitted else None,
        third_derivative_bound_candidate=third,
        third_derivative_bound_upper_bound=third if admitted else None,
        averaged_exposure_discrepancy_bounds=exposure,
        averaged_clock_discrepancy_candidates=clock_average,
        averaged_clock_discrepancy_upper_bounds=clock_average if admitted else None,
        projected_numerical_radii=numerical,
        projected_sensor_error_radii=sensor,
        projected_clock_discrepancy_candidates=clock,
        projected_clock_discrepancy_upper_bounds=clock if admitted else None,
        reconstruction_error_candidates=reconstruction,
        reconstruction_error_upper_bounds=reconstruction if admitted else None,
        virtual_reading_midpoints=virtual,
        point_reference_reading_bounds=comparison,
        original_curvature_average_coefficients=(Q(9, 4), Q(-9, 2), Q(9, 4), Q(0)),
        embedded_inverse_average_coefficients=embedded,
        nominal_bulk_angle_outer_bounds=reference.nominal_bulk_angle_outer_bounds,
        actual_long_arc_mean_outer_bounds=reference.actual_long_arc_mean_outer_bounds,
        effective_gain_outer_bounds=reference.effective_gain_outer_bounds,
        readout_gain_outer_bounds=reference.readout_gain_outer_bounds,
        first_window_mean_clock_rate_outer_bounds=reference.clock_rate_outer_bounds,
        source_admitted=admitted,
        clock_drift_transfer_certified=admitted,
        aperture_reconstruction_certified=admitted,
        finite_curvature_bound_certified=reference.finite_curvature_bound_certified,
        rank_deficient=reference.clock_envelope.rank_deficient,
        rank_certified=reference.clock_envelope.rank_certified,
        inverse_enclosure_available=reference.inverse_enclosure_available,
        status=reference.status,
        unavailable_reasons=reference.unavailable_reasons,
        incompatibility_reasons=reference.incompatibility_reasons,
    )
