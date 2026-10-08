"""Joint necessary geometry/gain bounds from two successive phase probes.

Three readings share their middle observation, a held gain and a held offset.
The two windows retain the full evolving source; no equilibrium or preparation
reset separates the pulses. Coordinate projections are not an existence proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    arg,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry
from .relational_observations import _ordered
from .relational_sine_two_port_inference import (
    _BULK_DOMAIN,
    _FORCING_UPPER,
    _GAMMA_UPPER,
    _RECEIVER_DOMAIN,
    _inference_geometry,
    _primitive_bounds,
)

__all__ = ("SineTwoPulseInference", "infer_sine_two_pulse_geometry_gain")


def _intersection(left, right):
    lower, upper = max(left.lo, right.lo), min(left.hi, right.hi)
    return None if lower > upper else I(lower, upper)


@dataclass(frozen=True)
class SineTwoPulseInference:
    """Conditional outer marginals for one original geometry and one held gain.

    ``phase_increments`` contains cumulative amplitudes a1,a2. The actual
    second jump is (a2-a1)*q after one elapsed ``probe_duration``; a second
    equally long window follows. Equality denotes a zero second jump and
    leaves the leading inverse rank deficient, not the full flow excluded.

    Three primitive recorded-reading intervals refer to before, middle and
    final observations of q-transpose-x. One unknown positive gain and one
    arbitrary offset remain held. Each scalar reading has its own bounded
    additive error. Exact midpoints cancel the common offset before outward
    coefficient arithmetic; shared-middle error coefficients remain explicit.

    Form/phase radii and the actual eight-long-arc mean refer to the original
    source. Every marginal encloses necessary compatibility only; their
    Cartesian product need not be jointly realizable by any full trajectory.
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
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    nodal_angle_affine_coefficients: tuple[tuple[Q, Q, Q], ...]
    edge_angle_affine_coefficients: tuple[tuple[Q, Q, Q], ...]
    edge_integer_offsets: tuple[int, ...]
    named_cycle_periods: tuple[Q, ...]
    laplacian: tuple[tuple[Q, ...], ...]
    normalized_laplacian: tuple[tuple[Q, ...], ...]
    normalized_upper_slack_matrix: tuple[tuple[Q, ...], ...]
    normalized_sine_symbol_coefficients: tuple[tuple[Q, ...], ...]
    readout_forcing_moments: tuple[tuple[Q, ...], ...]
    dipole: tuple[Q, ...]
    nominal_edge_angle_bounds: tuple[I, ...]
    nominal_acute_margin: Q
    source_acute_margin: Q
    total_duration: Q
    actual_phase_jumps: tuple[Q, Q]
    whole_window_form_norm_candidate: Q
    whole_window_phase_norm_candidate: Q
    whole_window_acute_margin_candidate: Q
    finite_remainder_candidate: Q
    finite_remainder_upper_bound: Q | None
    gamma_bounds: I
    response_scale_bounds: tuple[I, I]
    response_matrix_bounds: tuple[tuple[I, I], tuple[I, I]]
    determinant_angle_factor_bounds: I
    determinant_bounds: I
    inverse_matrix_bounds: tuple[tuple[I, I], tuple[I, I]] | None
    inverse_row_sum_upper_bounds: tuple[Q, Q] | None
    reading_midpoints: tuple[Q, Q, Q]
    reading_radii: tuple[Q, Q, Q]
    recorded_increment_midpoints: tuple[Q, Q]
    transformed_reading_coefficients: tuple[tuple[I, I, I], ...] | None
    transformed_centers: tuple[I, I] | None
    transformed_observation_error_radii: tuple[Q, Q] | None
    transformed_flow_error_radii: tuple[Q, Q] | None
    raw_transformed_bounds: tuple[I, I] | None
    prior_transformed_bounds: tuple[I, I]
    transformed_coordinate_bounds: tuple[I, I] | None
    nominal_bulk_angle_outer_bounds: I | None
    actual_long_arc_mean_outer_bounds: I | None
    readout_gain_outer_bounds: I | None
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
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t"
    capacity: tuple[Q, ...] = (Q(1),) * 18
    transformed_coordinate_order: tuple[str, str] = (
        "gain_times_cos_bulk",
        "gain_times_sin_bulk",
    )
    scope: tuple[str, ...] = (
        "same_noncritical_affine_source_family_and_full_two_port_sine_law",
        "two_successive_equal_windows_without_source_or_equilibrium_reset",
        "cumulative_amplitudes_determine_first_jump_and_second_amplitude_difference",
        "three_scalar_readings_share_one_held_positive_gain_and_one_held_offset",
        "shared_middle_observation_error_retained_before_coordinate_projection",
        "exact_reading_midpoint_differences_cancel_arbitrary_common_offset",
        "full_original_source_errors_and_both_evolving_rows_enter_uniform_remainder",
        "positive_gain_prior_is_supplied_information_not_inferred_physical_calibration",
        "coordinate_box_can_discard_joint_error_correlations_without_claiming_realizability",
        "bounded_candidate_gives_necessary_marginals_not_existence_or_point_identification",
        "actual_pre_probe_arc_mean_expands_nominal_angle_by_original_phase_radius_over_eight",
        "leading_rank_failure_does_not_exclude_a_full_nonlinear_response",
        "no_target_root_trajectory_producer_hidden_source_or_cached_report_consumed",
        "whole_window_acuity_is_optional_and_not_an_inverse_admission_gate",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-pulse-inference.v1", "report": _project(self)}


def infer_sine_two_pulse_geometry_gain(
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
) -> SineTwoPulseInference:
    """Project necessary joint compatibility from nine mandatory primitives.

    Angle priors lie in b=[11/8,3/2],c=[2/3,1] radians; radii and errors are
    nonnegative. The two cumulative amplitudes satisfy 0<a1<=a2<=1 and each
    window has duration 0<h<=1/2 in fast time. Three ordered reading pairs
    and positive gain endpoints undergo shared exact/represented admission.

    Equal amplitudes, unresolved positive interval determinants, source chart
    failure or unresolved argument arithmetic return ``unavailable``. Strict
    exclusion by a necessary constraint yields ``incompatible``. A retained
    ``bounded_candidate`` supplies marginal outer bounds, not joint existence.
    No response acquisition, empirical fit or inverse refinement is executed.
    """
    b = _primitive_bounds(bulk_angle_bounds, "bulk_angle_bounds")
    receiver = _primitive_bounds(
        receiver_short_angle_bounds, "receiver_short_angle_bounds"
    )
    amplitudes = _primitive_bounds(phase_increments, "phase_increments")
    gain = _primitive_bounds(readout_gain_bounds, "readout_gain_bounds")
    raw_readings = _ordered(recorded_reading_bounds, "recorded_reading_bounds", limit=4)
    if len(raw_readings) != 3:
        raise ValueError(
            "recorded_reading_bounds must contain exactly three endpoint pairs"
        )
    readings = tuple(
        _primitive_bounds(raw, f"recorded_reading_bounds[{i}]")
        for i, raw in enumerate(raw_readings)
    )
    x, y, h, noise = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_radius, "form_radius"),
            (phase_radius, "phase_radius"),
            (probe_duration, "probe_duration"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    if not (_BULK_DOMAIN[0] <= b[0] <= b[1] <= _BULK_DOMAIN[1]):
        raise ValueError("bulk_angle_bounds must lie within [11/8,3/2] radians")
    if not (_RECEIVER_DOMAIN[0] <= receiver[0] <= receiver[1] <= _RECEIVER_DOMAIN[1]):
        raise ValueError("receiver_short_angle_bounds must lie within [2/3,1] radians")
    if min(x, y, noise) < 0:
        raise ValueError(
            "form_radius, phase_radius and readout_error_bound must be nonnegative"
        )
    a1, a2 = amplitudes
    if not 0 < a1 <= a2 <= 1:
        raise ValueError("cumulative phase_increments must satisfy 0<a1<=a2<=1")
    if not 0 < h <= Q(1, 2):
        raise ValueError("probe_duration must lie in (0,1/2]")
    if gain[0] <= 0:
        raise ValueError("readout_gain_bounds must be strictly positive")
    geometry = _inference_geometry()
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    _sine_model_coefficients(model, positive_loss=True)
    pi = pi_interval()
    coordinates = (pi, I(*b), I(*receiver))
    edge_bounds = tuple(
        sum(
            (
                coefficient * coordinate
                for coefficient, coordinate in zip(row, coordinates)
            ),
            I(0),
        )
        for row in geometry["edge_angle_affine_coefficients"]
    )
    nominal_margin = min((pi / 2).lo - edge.abs_max for edge in edge_bounds)
    source_margin = nominal_margin - y
    source_ok = source_margin > 0
    g, duration = _GAMMA_UPPER, 2 * h
    qmax = (x + g * duration * (_FORCING_UPPER + 4 * a2 + 2 * y)) / (
        1 - 4 * g * g * duration * duration
    )
    pmax = y + 2 * a2 + 2 * g * duration * qmax
    error = 2 * h * qmax + 2 * g * h * y + 4 * g * g * duration * h * qmax
    gamma = 1 / (1023 * pi)
    scales = tuple(2 * gamma * h * sin(I(Q(3, 2) * a)) for a in amplitudes)
    matrix = tuple(
        (scale * cos(I(a / 2)), scale * sin(I(a / 2)))
        for a, scale in zip(amplitudes, scales)
    )
    deficient = a1 == a2
    factor = I(0) if deficient else sin(I((a2 - a1) / 2))
    determinant = scales[0] * scales[1] * factor
    rank_ok = determinant.lo > 0
    midpoints = tuple((lo + hi) / 2 for lo, hi in readings)
    radii = tuple((hi - lo) / 2 for lo, hi in readings)
    differences = (midpoints[1] - midpoints[0], midpoints[2] - midpoints[1])
    prior = (
        I(*gain) * I(cos(I(b[1])).lo, cos(I(b[0])).hi),
        I(*gain) * I(sin(I(b[0])).lo, sin(I(b[1])).hi),
    )
    unavailable, incompatible = [], []
    if not source_ok:
        unavailable.append(
            "original_source_phase_ball_is_not_certified_inside_the_declared_acute_chart"
        )
    if deficient:
        unavailable.append(
            "equal_cumulative_amplitudes_leave_the_leading_response_rank_deficient"
        )
    elif not rank_ok:
        unavailable.append(
            "positive_response_determinant_is_unresolved_at_interval_precision"
        )
    inverse = row_norms = reading_coefficients = None
    centers = observation_radii = flow_radii = None
    raw = clipped = None
    angle = actual = gain_outer = None
    status = "unavailable"
    if rank_ok:
        inverse = (
            (matrix[1][1] / determinant, -matrix[0][1] / determinant),
            (-matrix[1][0] / determinant, matrix[0][0] / determinant),
        )
        row_norms = tuple(left.abs_max + right.abs_max for left, right in inverse)
        reading_coefficients = tuple(
            (left, right - left, -right) for left, right in inverse
        )
    if not unavailable:
        centers = tuple(
            -left * differences[0] - right * differences[1] for left, right in inverse
        )
        observation_radii = tuple(
            sum(
                (radius + noise) * coefficient.abs_max
                for radius, coefficient in zip(radii, row)
            )
            for row in reading_coefficients
        )
        flow_radii = tuple(gain[1] * error * value for value in row_norms)
        raw = tuple(
            center + I(-sensor - flow, sensor + flow)
            for center, sensor, flow in zip(centers, observation_radii, flow_radii)
        )
        intersections = tuple(
            _intersection(value, bound) for value, bound in zip(raw, prior)
        )
        if any(value is None for value in intersections):
            incompatible.append(
                "transformed_response_excludes_the_supplied_positive_gain_and_angle_priors"
            )
            status = "incompatible"
        else:
            clipped = intersections
            try:
                projected_angle = arg(*clipped)
                projected_gain = sqrt(clipped[0] ** 2 + clipped[1] ** 2)
            except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
                unavailable.append(f"joint_angle_gain_projection_unavailable: {exc}")
            else:
                angle = _intersection(projected_angle, I(*b))
                gain_outer = _intersection(projected_gain, I(*gain))
                if angle is None or gain_outer is None:
                    incompatible.append(
                        "joint_angle_or_gain_projection_excludes_its_supplied_prior"
                    )
                    angle = gain_outer = None
                    status = "incompatible"
                else:
                    actual = angle + I(-y / 8, y / 8)
                    status = "bounded_candidate"
    return SineTwoPulseInference(
        bulk_angle_bounds=b,
        receiver_short_angle_bounds=receiver,
        form_radius=x,
        phase_radius=y,
        phase_increments=amplitudes,
        probe_duration=h,
        recorded_reading_bounds=readings,
        readout_error_bound=noise,
        readout_gain_bounds=gain,
        reference_model=model,
        **geometry,
        nominal_edge_angle_bounds=edge_bounds,
        nominal_acute_margin=nominal_margin,
        source_acute_margin=source_margin,
        total_duration=duration,
        actual_phase_jumps=(a1, a2 - a1),
        whole_window_form_norm_candidate=qmax,
        whole_window_phase_norm_candidate=pmax,
        whole_window_acute_margin_candidate=nominal_margin - pmax,
        finite_remainder_candidate=error,
        finite_remainder_upper_bound=error if source_ok else None,
        gamma_bounds=gamma,
        response_scale_bounds=scales,
        response_matrix_bounds=matrix,
        determinant_angle_factor_bounds=factor,
        determinant_bounds=determinant,
        inverse_matrix_bounds=inverse,
        inverse_row_sum_upper_bounds=row_norms,
        reading_midpoints=midpoints,
        reading_radii=radii,
        recorded_increment_midpoints=differences,
        transformed_reading_coefficients=reading_coefficients,
        transformed_centers=centers,
        transformed_observation_error_radii=observation_radii,
        transformed_flow_error_radii=flow_radii,
        raw_transformed_bounds=raw,
        prior_transformed_bounds=prior,
        transformed_coordinate_bounds=clipped,
        nominal_bulk_angle_outer_bounds=angle,
        actual_long_arc_mean_outer_bounds=actual,
        readout_gain_outer_bounds=gain_outer,
        source_admitted=source_ok,
        finite_response_certified=source_ok,
        rank_deficient=deficient,
        rank_certified=rank_ok,
        whole_window_acute_certified=source_ok and nominal_margin - pmax > 0,
        inverse_enclosure_available=status == "bounded_candidate",
        status=status,
        unavailable_reasons=tuple(unavailable),
        incompatibility_reasons=tuple(incompatible),
    )
