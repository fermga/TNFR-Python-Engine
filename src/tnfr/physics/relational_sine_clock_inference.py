"""Necessary two-pulse geometry and gain bounds with one uncertain clock rate.

The known-clock inverse supplies an auxiliary constraint envelope, not a
replacement trajectory. Full finite-time dependence remains in its error
bound; only the leading response combines sensor gain and clock rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from .relational_sine_two_port_inference import _primitive_bounds
from .relational_sine_two_pulse_inference import (
    SineTwoPulseInference,
    infer_sine_two_pulse_geometry_gain,
)

__all__ = ("SineClockInference", "infer_sine_geometry_gain_clock")


@dataclass(frozen=True)
class SineClockInference:
    """Conditional outer marginals with ``tau=rho*s`` and fixed gamma.

    ``probe_duration`` is the observed duration H of each successive window.
    One positive clock rate rho and one sensor gain G are held throughout.
    ``constraint_envelope`` uses h*=rho_plus*H and auxiliary gain
    L=G*rho/rho_plus. Its gain fields refer to L, and its known-clock
    trajectory wording does not assert that the actual trajectory reaches
    h*. It is consumed only as a necessary algebraic constraint envelope.

    The effective gain is J=G*rho. Its bounds and the separate G/rho
    projections are necessary marginals, not a joint realization claim or
    an exact full-flow gain/clock symmetry. All geometry concerns the
    original source, including the Y/8 actual-mean expansion.
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
    structural_probe_duration_bounds: tuple[Q, Q]
    auxiliary_gain_prior_bounds: tuple[Q, Q]
    effective_gain_prior_bounds: tuple[Q, Q]
    constraint_envelope: SineTwoPulseInference
    finite_remainder_over_clock_candidate: Q
    finite_remainder_over_clock_upper_bound: Q | None
    nominal_bulk_angle_outer_bounds: I | None
    actual_long_arc_mean_outer_bounds: I | None
    effective_gain_outer_bounds: I | None
    readout_gain_outer_bounds: I | None
    clock_rate_outer_bounds: I | None
    source_admitted: bool
    finite_response_certified: bool
    rank_deficient: bool
    rank_certified: bool
    whole_window_acute_certified: bool
    inverse_enclosure_available: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    incompatibility_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    clock: str = "tau=rho*s; rho is one held positive rate"
    effective_gain_relation: str = "J=G*rho"
    complete_observed_rows: tuple[str, str] = (
        "dx/ds=rho*(-A*x+gamma*f(theta))",
        "dtheta/ds=rho*gamma*A*x",
    )
    scope: tuple[str, ...] = (
        "fixed_gamma_and_complete_two_port_law_under_one_unknown_constant_clock_rate",
        "two_events_at_observed_times_zero_and_H_without_source_reset",
        "auxiliary_envelope_gain_is_L_equal_G_times_rho_over_rho_upper",
        "auxiliary_upper_duration_is_not_the_actual_unknown_trajectory_duration",
        "monotone_finite_remainder_over_clock_rate_retains_full_flow_dependence",
        "three_reading_errors_remain_in_sensor_units_without_clock_division",
        "effective_gain_J_is_G_times_rho_not_sensor_gain_alone",
        "separate_gain_and_clock_bounds_project_one_necessary_product_relation",
        "coordinate_product_need_not_be_jointly_realizable",
        "no_exact_full_flow_gain_clock_symmetry_or_separate_point_identification",
        "original_source_actual_arc_mean_retains_original_phase_radius_over_eight",
        "no_acquisition_calibration_trajectory_producer_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-clock-inference.v1", "report": _project(self)}


def infer_sine_geometry_gain_clock(
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
) -> SineClockInference:
    """Project geometry, effective gain and separate gain/clock outer bounds.

    All ten inputs are mandatory primitives. Here ``probe_duration`` is H in
    observed time s. Require H>0, positive ordered gain and clock priors, and
    rho_upper*H<=1/2. The shared two-pulse inverse admits the original source,
    cumulative phase inputs and three reading pairs with unchanged domains.

    At h*=rho_upper*H, the auxiliary gain prior is
    [G_lower*rho_lower/rho_upper, G_upper]. Monotonicity of E(rho*H)/rho
    proves G*E(rho*H)<=L*E(h*), so its finite envelope covers every admitted
    clock. No recorded reading or sensor error is divided by a clock rate.
    Exact endpoint products/quotients precede outward interval projection.
    """
    rates = _primitive_bounds(clock_rate_bounds, "clock_rate_bounds")
    gain = _primitive_bounds(readout_gain_bounds, "readout_gain_bounds")
    duration = exact_or_represented_real(probe_duration, "probe_duration")
    if rates[0] <= 0:
        raise ValueError("clock_rate_bounds must be strictly positive")
    if gain[0] <= 0:
        raise ValueError("readout_gain_bounds must be strictly positive")
    if duration <= 0 or rates[1] * duration > Q(1, 2):
        raise ValueError(
            "require positive probe_duration and clock_upper*duration<=1/2"
        )
    structural = tuple(rate * duration for rate in rates)
    auxiliary_gain = (gain[0] * rates[0] / rates[1], gain[1])
    effective_prior = (gain[0] * rates[0], gain[1] * rates[1])
    envelope = infer_sine_two_pulse_geometry_gain(
        bulk_angle_bounds=bulk_angle_bounds,
        receiver_short_angle_bounds=receiver_short_angle_bounds,
        form_radius=form_radius,
        phase_radius=phase_radius,
        phase_increments=phase_increments,
        probe_duration=structural[1],
        recorded_reading_bounds=recorded_reading_bounds,
        readout_error_bound=readout_error_bound,
        readout_gain_bounds=auxiliary_gain,
    )
    effective = inferred_gain = inferred_rate = None
    nominal = envelope.nominal_bulk_angle_outer_bounds
    actual = envelope.actual_long_arc_mean_outer_bounds
    status = envelope.status
    incompatible = envelope.incompatibility_reasons
    if envelope.inverse_enclosure_available:
        auxiliary = envelope.readout_gain_outer_bounds
        # Retain exact products while projecting the relation. Constructing a
        # tiny positive clock interval first could introduce a zero divisor.
        lo, hi = rates[1] * auxiliary.lo, rates[1] * auxiliary.hi
        gain_lo, gain_hi = max(gain[0], lo / rates[1]), min(gain[1], hi / rates[0])
        rate_lo, rate_hi = max(rates[0], lo / gain[1]), min(rates[1], hi / gain[0])
        if gain_lo > gain_hi or rate_lo > rate_hi:
            status = "incompatible"
            incompatible += ("effective_gain_excludes_the_supplied_gain_clock_product",)
            nominal = actual = None
        else:
            effective = I(lo, hi)
            inferred_gain = I(gain_lo, gain_hi)
            inferred_rate = I(rate_lo, rate_hi)
    error_over_clock = envelope.finite_remainder_candidate / rates[1]
    return SineClockInference(
        bulk_angle_bounds=envelope.bulk_angle_bounds,
        receiver_short_angle_bounds=envelope.receiver_short_angle_bounds,
        form_radius=envelope.form_radius,
        phase_radius=envelope.phase_radius,
        phase_increments=envelope.phase_increments,
        probe_duration=duration,
        recorded_reading_bounds=envelope.recorded_reading_bounds,
        readout_error_bound=envelope.readout_error_bound,
        readout_gain_bounds=gain,
        clock_rate_bounds=rates,
        structural_probe_duration_bounds=structural,
        auxiliary_gain_prior_bounds=auxiliary_gain,
        effective_gain_prior_bounds=effective_prior,
        constraint_envelope=envelope,
        finite_remainder_over_clock_candidate=error_over_clock,
        finite_remainder_over_clock_upper_bound=(
            error_over_clock if envelope.finite_response_certified else None
        ),
        nominal_bulk_angle_outer_bounds=nominal,
        actual_long_arc_mean_outer_bounds=actual,
        effective_gain_outer_bounds=effective,
        readout_gain_outer_bounds=inferred_gain,
        clock_rate_outer_bounds=inferred_rate,
        source_admitted=envelope.source_admitted,
        finite_response_certified=envelope.finite_response_certified,
        rank_deficient=envelope.rank_deficient,
        rank_certified=envelope.rank_certified,
        whole_window_acute_certified=envelope.whole_window_acute_certified,
        inverse_enclosure_available=status == "bounded_candidate",
        status=status,
        unavailable_reasons=envelope.unavailable_reasons,
        incompatibility_reasons=incompatible,
    )
