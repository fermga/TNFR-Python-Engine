"""Necessary geometry/gain/mean-clock bounds under supplied bounded clock drift.

Autonomous complete-flow exposure compares the actual clock with its first
window mean. The existing curvature inverse consumes widened reading bands
for that reference history; it does not identify an instantaneous clock rate.
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

__all__ = ("SineClockDriftInference", "infer_sine_geometry_gain_clock_drift")


@dataclass(frozen=True)
class SineClockDriftInference:
    """Outer marginals conditional on a positive C1 clock with bounded drift.

    The supplied rate prior holds pointwise on [0,2H], and the supplied
    derivative bound is |rho'(s)|<=L in observed-time units. No clock profile
    is consumed or verified. The inferred rate is the first-window mean
    rho_bar_1=(1/H)*integral_0^H rho(s) ds, not any pointwise value.

    ``reference_envelope`` concerns a constant-mean reference history with
    the same complete source and both events. Its widened observations
    retain original sensor error separately. Necessary marginal constraints
    need not have a joint state, error, gain or clock-profile realization.
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
    clock_rate_derivative_bound: Q
    reference_envelope: SineCurvatureInference
    observation_times: tuple[Q, Q, Q, Q]
    structural_duration_upper_bound: Q
    whole_window_form_norm_candidate: Q
    whole_window_form_norm_upper_bound: Q | None
    structural_readout_speed_candidate: Q
    structural_readout_speed_upper_bound: Q | None
    exposure_discrepancy_bounds: tuple[Q, Q, Q, Q]
    recorded_discrepancy_candidates: tuple[Q, Q, Q, Q]
    recorded_discrepancy_upper_bounds: tuple[Q, Q, Q, Q] | None
    comparison_reading_bounds: tuple[tuple[Q, Q], ...]
    nominal_bulk_angle_outer_bounds: I | None
    actual_long_arc_mean_outer_bounds: I | None
    effective_gain_outer_bounds: I | None
    readout_gain_outer_bounds: I | None
    first_window_mean_clock_rate_outer_bounds: I | None
    source_admitted: bool
    clock_drift_transfer_certified: bool
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
        "global_positive_clock_rate_and_C1_derivative_bounds_are_supplied_premises",
        "fixed_complete_two_port_law_and_both_rows_share_one_clock_exposure",
        "four_point_readings_at_zero_H_over_two_H_two_H_are_not_exposure_averages",
        "original_source_and_phase_events_at_zero_and_H_are_preserved",
        "constant_first_window_mean_reference_agrees_exactly_at_zero_and_H",
        "second_event_starts_from_the_same_complete_state_in_both_histories",
        "only_half_time_and_final_reading_bands_receive_clock_comparison_allowances",
        "original_sensor_error_gain_and_offset_premises_remain_unchanged",
        "reference_envelope_is_a_constraint_on_a_comparison_history_not_actual_constant_clock_flow",
        "reference_status_and_reasons_apply_to_the_widened_necessary_constraints",
        "effective_gain_uses_first_window_mean_rate_not_pointwise_clock_rate",
        "original_actual_arc_mean_retains_the_original_phase_radius_over_eight",
        "marginal_intersections_do_not_certify_joint_state_error_or_clock_profile_realizability",
        "no_profile_fit_clock_grid_response_producer_or_cached_report_input",
        "no_physical_clock_calibration_acquisition_or_unique_law_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-clock-drift-inference.v1",
            "report": _project(self),
        }


def infer_sine_geometry_gain_clock_drift(
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
    clock_rate_derivative_bound,
) -> SineClockDriftInference:
    """Infer necessary first-window mean-rate bounds with a supplied drift cap.

    All eleven inputs are mandatory primitives. The first ten domains match
    the four-reading curvature inverse; the new derivative bound L is finite
    and nonnegative. Require H>0 and rho_plus*H<=1/2. The pointwise clock
    prior and C1 derivative bound hold throughout both observed windows.

    Exact reading expansions use exposure discrepancies bounded by both the
    derivative and rate range. The original scalar sensor-error allowance
    is passed unchanged to one fresh constant-mean reference calculation.
    Zero drift or a singleton rate prior gives zero expansion exactly.
    """
    b = _primitive_bounds(bulk_angle_bounds, "bulk_angle_bounds")
    receiver = _primitive_bounds(
        receiver_short_angle_bounds, "receiver_short_angle_bounds"
    )
    amplitudes = _primitive_bounds(phase_increments, "phase_increments")
    gain = _primitive_bounds(readout_gain_bounds, "readout_gain_bounds")
    rates = _primitive_bounds(clock_rate_bounds, "clock_rate_bounds")
    raw = _ordered(recorded_reading_bounds, "recorded_reading_bounds", limit=5)
    if len(raw) != 4:
        raise ValueError(
            "recorded_reading_bounds must contain exactly four endpoint pairs"
        )
    readings = tuple(
        _primitive_bounds(pair, f"recorded_reading_bounds[{index}]")
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

    total = 2 * rates[1] * horizon
    form_bound = x + 7 * _GAMMA_UPPER * total
    speed = 2 * form_bound + 7 * _GAMMA_UPPER
    rate_range = rates[1] - rates[0]
    exposure = (
        Q(0),
        min(drift * horizon**2 / 8, rate_range * horizon / 4),
        Q(0),
        min(drift * horizon**2, rate_range * horizon),
    )
    discrepancy = tuple(gain[1] * speed * radius for radius in exposure)
    comparison = tuple(
        (lo - radius, hi + radius) for (lo, hi), radius in zip(readings, discrepancy)
    )
    reference = infer_sine_geometry_gain_clock_curvature(
        bulk_angle_bounds=b,
        receiver_short_angle_bounds=receiver,
        form_radius=x,
        phase_radius=y,
        phase_increments=amplitudes,
        probe_duration=horizon,
        recorded_reading_bounds=comparison,
        readout_error_bound=noise,
        readout_gain_bounds=gain,
        clock_rate_bounds=rates,
    )
    admitted = reference.source_admitted
    return SineClockDriftInference(
        bulk_angle_bounds=b,
        receiver_short_angle_bounds=receiver,
        form_radius=x,
        phase_radius=y,
        phase_increments=amplitudes,
        probe_duration=horizon,
        recorded_reading_bounds=readings,
        readout_error_bound=noise,
        readout_gain_bounds=gain,
        clock_rate_bounds=rates,
        clock_rate_derivative_bound=drift,
        reference_envelope=reference,
        observation_times=(Q(0), horizon / 2, horizon, 2 * horizon),
        structural_duration_upper_bound=total,
        whole_window_form_norm_candidate=form_bound,
        whole_window_form_norm_upper_bound=form_bound if admitted else None,
        structural_readout_speed_candidate=speed,
        structural_readout_speed_upper_bound=speed if admitted else None,
        exposure_discrepancy_bounds=exposure,
        recorded_discrepancy_candidates=discrepancy,
        recorded_discrepancy_upper_bounds=discrepancy if admitted else None,
        comparison_reading_bounds=comparison,
        nominal_bulk_angle_outer_bounds=reference.nominal_bulk_angle_outer_bounds,
        actual_long_arc_mean_outer_bounds=reference.actual_long_arc_mean_outer_bounds,
        effective_gain_outer_bounds=reference.effective_gain_outer_bounds,
        readout_gain_outer_bounds=reference.readout_gain_outer_bounds,
        first_window_mean_clock_rate_outer_bounds=reference.clock_rate_outer_bounds,
        source_admitted=admitted,
        clock_drift_transfer_certified=admitted,
        finite_curvature_bound_certified=reference.finite_curvature_bound_certified,
        rank_deficient=reference.clock_envelope.rank_deficient,
        rank_certified=reference.clock_envelope.rank_certified,
        inverse_enclosure_available=reference.inverse_enclosure_available,
        status=reference.status,
        unavailable_reasons=reference.unavailable_reasons,
        incompatibility_reasons=reference.incompatibility_reasons,
    )
