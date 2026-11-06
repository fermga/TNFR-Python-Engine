"""Direct mixed-state passage and capture for the original C9 preparation.

Proof: theory/nodal/SINE_PREPARATION_PATH_CAPTURE.md. The source is rebuilt
at tau=200 and continued through y=theta+z, z=gamma*x. No reference trajectory,
nominal reset, phase-only replacement law or supplied endpoint is consumed.
The two stages end at memberwise times; a common later deadline follows from
full-state storage trapping, not from identifying their hitting times.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ._sine_preparation_path_selection import (
    _assess_preparation_path_selection,
    _PreparationPathSelection,
)
from .relational_sine_formed_classes import _GAP
from .reversible_eigenmode_reference import _negative_exp_bounds

_INITIAL_TIME = Q(200)
_FIRST_DURATION = Q(8)
_SECOND_DURATION = Q(128)
_GAP_LOWER = Q(9, 10)
_GAP_UPPER = Q(13, 10)
_CLOSING_UPPER = Q(63, 100)
_CLOSING_TARGET = Q(-7, 4)
_PASSAGE_SPEED = Q(3, 10)
_ACUTE_CAP = Q(7, 5)
_INITIAL_DISTANCE = Q(15)
_CONTRACTION = Q(1, 30)


@dataclass(frozen=True)
class _PreparationPathCapture:
    """Sufficient continuation certificate; failed margins remain unresolved.

    Bounds for stage two describe each member's actual passage time plus 128
    slow units. Only after the strict capture margins pass does storage
    nonincrease carry that result to the common deadline and all later times.
    Candidate arithmetic remains visible when its sufficient premises fail.
    """

    source: _PreparationPathSelection
    uniform_scaled_form_norm_upper_bound: Q
    mixed_field_error_upper_bound: Q
    initial_mixed_phase_error_norm_upper_bound: Q
    initial_ordinary_gap_bounds: tuple[I, ...]
    initial_closing_coordinate_bounds: I
    initial_closing_lower_margin: Q
    initial_closing_upper_margin: Q
    initial_ordinary_upper_margin: Q
    initial_coupled_lower_margin: Q
    stage_one_coupled_lower_bound: Q
    stage_one_ordinary_lower_bound: Q
    stage_one_ordinary_upper_bound: Q
    stage_one_sine_speed_lower_bounds: tuple[Q, Q, Q]
    stage_one_passage_speed_lower_bound: Q
    stage_one_passage_time_upper_bound: Q
    stage_two_initial_gap_bounds: I
    stage_two_actual_gap_absolute_upper_bound: Q
    stage_two_curvature_lower_bound: Q
    stage_two_initial_distance_squared_upper_bound: Q
    stage_two_decay_bounds: I
    stage_two_phase_distance_upper_bound: Q
    actual_phase_distance_at_capture_upper_bound: Q
    form_norm_upper_bound: Q
    form_storage_upper_bound: Q
    phase_storage_upper_bound: Q
    total_storage_upper_bound: Q
    sector_boundary_storage_lower_bound: Q
    capture_margin: Q
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_policy_certified: bool
    passage_certified: bool
    capture_certified: bool
    captured_winding: int | None
    slow_time_deadline: Q
    scaled_time_deadline_bounds: I
    original_time_deadline_bounds: I
    status: str
    unresolved_conditions: tuple[str, ...]
    initial_scaled_time: Q = _INITIAL_TIME
    stage_one_duration: Q = _FIRST_DURATION
    stage_two_duration: Q = _SECOND_DURATION
    closing_target: Q = _CLOSING_TARGET
    acute_gap_cap: Q = _ACUTE_CAP
    contraction_rate: Q = _CONTRACTION
    scope: tuple[str, ...] = (
        "same_original_Cayley_source_and_latent_errors_rebuilt_at_tau_200",
        "actual_mixed_state_y_equals_theta_plus_gamma_centered_form",
        "mixed_error_from_exact_initial_identity_without_a_second_fast_transient",
        "all_eighteen_original_coordinates_and_conserved_means_retained",
        "closing_edge_passage_is_smooth_flow_not_a_hybrid_event",
        "stage_two_begins_at_each_actual_first_hitting_time",
        "distance_bound_at_memberwise_capture_time_not_a_common_sampled_endpoint",
        "strict_full_state_storage_barrier_extends_capture_to_common_deadline",
        "acute_winding_one_capture_not_fixed_small_ball_entry",
        "failed_sufficient_margins_are_unresolved_not_nonformation",
        "no_reference_trajectory_parameter_sweep_or_frozen_producer",
    )


def _assess_preparation_path_capture(
    *, parameter_interval, form_error_bound, phase_error_bound
) -> _PreparationPathCapture:
    """Re-admit the original preparation and test the fixed two-stage proof.

    The path owner admits the exact interval and nonnegative finite residual
    budgets before any continuation arithmetic. Times, source, law, capacities
    and the proof constants are fixed; no incoming report replaces primitives.
    """
    source = _assess_preparation_path_selection(
        parameter_interval=parameter_interval,
        scaled_time=_INITIAL_TIME,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
    )
    pi = pi_interval()
    eta, forcing = source.eta_bounds.hi, source.forcing_norm_bounds.hi
    z = max(source.scaled_form_radius_upper_bound, eta * forcing / _GAP)
    error = 2 * z
    # y(tau)=theta(0)+gamma*x(0)+eta*integral_0^tau f(theta). The
    # homogeneous form term cancels exactly here; adding Rz would charge it twice.
    initial_error = (
        source.scaled_initial_error_norm_upper_bound
        + source.initial_phase_error_norm_upper_bound
        + eta * forcing * _INITIAL_TIME
    )
    edge_error = I(-initial_error, initial_error)
    dipole = tuple(Q(int(j == 3) - int(j == 4)) for j in range(9))
    ordinary = tuple(
        source.ramp_angle_bounds
        + source.lift_correction_bounds * (dipole[j + 1] - dipole[j])
        + edge_error
        for j in range(8)
    )
    closing = 8 * source.ramp_angle_bounds - 3 * pi + edge_error
    initial_margins = (
        ("initial_closing_positive", closing.lo),
        ("initial_closing_below_declared_ceiling", _CLOSING_UPPER - closing.hi),
        (
            "initial_ordinary_gaps_below_declared_ceiling",
            _GAP_UPPER - max(value.hi for value in ordinary),
        ),
        (
            "initial_coupled_gap_lower_bound",
            min(value.lo for value in ordinary) - closing.hi / 2 - _GAP_LOWER,
        ),
    )
    coupled_lower = _GAP_LOWER - Q(3, 2) * error * _FIRST_DURATION
    ordinary_lower = coupled_lower + _CLOSING_TARGET / 2
    ordinary_upper = _GAP_UPPER + error * _FIRST_DURATION
    sine_speeds = (
        (sin(I(Q(607, 500))) - sin(I(_CLOSING_UPPER))).lo,
        sin(I(Q(649, 1000))).lo,
        sin(I(Q(1, 2))).lo,
    )
    speed = min(sine_speeds) - error
    passage_time = (_CLOSING_UPPER - _CLOSING_TARGET) / _PASSAGE_SPEED
    first_margins = initial_margins + (
        ("stage_one_coupled_gap_floor", coupled_lower - Q(899, 1000)),
        ("stage_one_ordinary_positive_floor", ordinary_lower - Q(3, 125)),
        ("stage_one_ordinary_acute_ceiling", pi.lo / 2 - ordinary_upper),
        ("stage_one_strict_passage_speed", speed - _PASSAGE_SPEED),
        ("stage_one_strict_hitting_deadline", _FIRST_DURATION - passage_time),
    )
    # At q=-7/4 the closing edge has the winding-one representative 7/4-pi.
    # Within the acute chart the gap extrema drift by at most error per unit
    # slow time. The actual phase adds at most z to every mixed edge gap.
    initial_gaps = I(Q(7, 4) - pi.hi, ordinary_upper)
    actual_cap = initial_gaps.abs_max + _SECOND_DURATION * error + z
    curvature = cos(I(_ACUTE_CAP)).lo
    initial_squared = (45 * (_ACUTE_CAP + 2 * pi / 9) ** 2).hi
    decay = I(*_negative_exp_bounds(_CONTRACTION * _SECOND_DURATION))
    phase_distance = _INITIAL_DISTANCE * decay.hi + error / _CONTRACTION
    actual_distance = phase_distance + z
    inverse_gamma = 1023 * pi
    form_norm = inverse_gamma.hi * z / sqrt(I(2)).lo
    form_storage = 2 * form_norm**2
    phase_storage = (9 * (1 - cos(2 * pi / 9))).hi + actual_distance**2
    storage = form_storage + phase_storage
    barrier = source.sector_barrier_lower_bounds[source.candidate_windings.index(1)]
    capture_margin = barrier - storage
    all_margins = first_margins + (
        ("stage_two_strict_acute_tube", _ACUTE_CAP - actual_cap),
        ("stage_two_cap_is_acute", pi.lo / 2 - _ACUTE_CAP),
        ("stage_two_uniform_phase_curvature", curvature - Q(1, 6)),
        ("stage_two_initial_distance", _INITIAL_DISTANCE**2 - initial_squared),
        ("strict_full_state_capture_barrier", capture_margin),
    )
    source_ok = source.source_budget_certified and source.initial_zero_winding_certified
    passage = source_ok and all(margin > 0 for _, margin in first_margins)
    capture = source_ok and all(margin > 0 for _, margin in all_margins)
    unresolved = tuple(name for name, margin in all_margins if margin <= 0) + tuple(
        name
        for name, ok in (
            ("original_source_storage_policy", source.source_budget_certified),
            (
                "original_zero_winding_preparation",
                source.initial_zero_winding_certified,
            ),
        )
        if not ok
    )
    deadline = _FIRST_DURATION + _SECOND_DURATION
    # Use the exact model formula, never division by an outward eta interval.
    tau = _INITIAL_TIME + deadline * 1023**2 * pi**2
    original_time = tau / Q(1023, 1024)
    return _PreparationPathCapture(
        source=source,
        uniform_scaled_form_norm_upper_bound=z,
        mixed_field_error_upper_bound=error,
        initial_mixed_phase_error_norm_upper_bound=initial_error,
        initial_ordinary_gap_bounds=ordinary,
        initial_closing_coordinate_bounds=closing,
        initial_closing_lower_margin=initial_margins[0][1],
        initial_closing_upper_margin=initial_margins[1][1],
        initial_ordinary_upper_margin=initial_margins[2][1],
        initial_coupled_lower_margin=initial_margins[3][1],
        stage_one_coupled_lower_bound=coupled_lower,
        stage_one_ordinary_lower_bound=ordinary_lower,
        stage_one_ordinary_upper_bound=ordinary_upper,
        stage_one_sine_speed_lower_bounds=sine_speeds,
        stage_one_passage_speed_lower_bound=speed,
        stage_one_passage_time_upper_bound=passage_time,
        stage_two_initial_gap_bounds=initial_gaps,
        stage_two_actual_gap_absolute_upper_bound=actual_cap,
        stage_two_curvature_lower_bound=curvature,
        stage_two_initial_distance_squared_upper_bound=initial_squared,
        stage_two_decay_bounds=decay,
        stage_two_phase_distance_upper_bound=phase_distance,
        actual_phase_distance_at_capture_upper_bound=actual_distance,
        form_norm_upper_bound=form_norm,
        form_storage_upper_bound=form_storage,
        phase_storage_upper_bound=phase_storage,
        total_storage_upper_bound=storage,
        sector_boundary_storage_lower_bound=barrier,
        capture_margin=capture_margin,
        sufficient_margins=all_margins,
        source_policy_certified=source_ok,
        passage_certified=passage,
        capture_certified=capture,
        captured_winding=1 if capture else None,
        slow_time_deadline=deadline,
        scaled_time_deadline_bounds=tau,
        original_time_deadline_bounds=original_time,
        status="certified_winding_one_capture" if capture else "unavailable",
        unresolved_conditions=unresolved,
    )
