"""Own-law uniform capture along the original energy-isometric C9 source.

Proof: theory/nodal/SINE_PREPARATION_PATH_ROBUSTNESS.md. The complete
ninth-harmonic potential family is re-admitted through global current and
Hessian bounds, its source cost, mixed-state passage and full-storage barrier.
No sine transit report, capture verdict, trajectory or frozen producer is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_class_constitutive_comparison import _current_derivative_bound, _deformation
from ._sine_preparation import _prepared_transit_radii
from ._sine_preparation_mode_selection import _STORAGE_BUDGET
from ._sine_preparation_path_selection import (
    _PreparationPathSource,
    _prepare_path_source,
)
from ._sine_sector_boundary import _sector_boundary_bounds
from .phase_cycle_geometry import _derive
from .relational_observations import _ordered
from .relational_sine_formed_classes import _EDGES, _GAP, _NODES
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
_CONTRACTION = Q(1, 32)
_CURVATURE_SPLIT = Q(3, 2)


@dataclass(frozen=True)
class _ConstitutivePreparationPathCapture:
    """Uniform sufficient capture over the declared closed law interval.

    Every bound is rebuilt from primitive source and law data. Upper budgets
    increase with the interval's upper deformation; lower margins decrease.
    Candidate arithmetic is retained on failure, without claiming capture.
    Stage-two bounds refer to each actual hitting time plus 128 slow units;
    the common deadline follows only after full-state trapping is proved.
    Phase and scaled-form norm fields use D=2I; form_norm_upper_bound is
    Euclidean. In particular the endpoint phase error is a D-norm bound.
    """

    deformation_interval: tuple[Q, Q]
    source: _PreparationPathSource
    preparation_dimension: int
    full_relative_preparation: bool
    current_deviation_upper_bound: Q
    current_absolute_upper_bound: Q
    phase_hessian_upper_bound: Q
    global_field_lipschitz_upper_bound: Q
    forcing_norm_bounds: I
    initial_potential_deformation_upper_bound: Q
    common_initial_storage_upper_bound: Q
    source_budget_margin_bounds: I
    source_budget_certified: bool
    initial_zero_winding_certified: bool
    exponential_decay_bounds: I
    scaled_form_radius_upper_bound: Q
    endpoint_phase_error_norm_upper_bound: Q
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
    stage_one_current_speed_lower_bounds: tuple[Q, Q, Q]
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
    sector_boundary_face_lower_bounds: tuple[Q, ...]
    sector_boundary_storage_lower_bound: Q
    capture_margin: Q
    inner_acute_curvature_lower_bound: Q
    law_margins: tuple[tuple[str, Q], ...]
    sufficient_margins: tuple[tuple[str, Q], ...]
    uniform_law_bounds_certified: bool
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
    common_source_storage_budget: Q = _STORAGE_BUDGET
    law: str = "normalized_matched_ninth_harmonic_reciprocal_exchange"
    clock: str = "tau=e*t; e=1023/1024; sigma=gamma**2*(tau-200)"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_original_Cayley_source_and_latent_errors_for_every_admitted_law",
        "exact_deformation_endpoints_admitted_before_interval_materialization",
        "uniform_law_bounds_by_monotonicity_not_a_deformation_sweep",
        "law_independent_source_geometry_but_own_force_transit_and_storage",
        "source_coordinate_error_norms_not_enlarged_by_current_amplitude",
        "exact_quartic_source_potential_correction_retained_before_rounding",
        "actual_mixed_state_y_equals_theta_plus_gamma_centered_form",
        "closing_current_is_j_pi_minus_q_not_j_q",
        "all_eighteen_original_coordinates_and_conserved_means_retained",
        "own_global_Lipschitz_bound_and_local_contraction_rate",
        "whole_acute_curvature_proved_by_inner_and_harmonic_outer_intervals",
        "nonnegative_deformation_transfers_a_boundary_inequality_not_a_verdict",
        "closing_edge_passage_is_smooth_flow_not_a_hybrid_event",
        "stage_two_begins_at_each_actual_first_hitting_time",
        "strict_full_state_storage_barrier_extends_capture_to_common_deadline",
        "acute_winding_one_capture_not_fixed_small_ball_entry",
        "failed_sufficient_margins_are_unresolved_not_nonformation",
        "no_sine_specific_transit_capture_dwell_contact_or_interaction_transfer",
        "no_reference_trajectory_parameter_sweep_or_frozen_producer",
    )


def _assess_constitutive_preparation_path_capture(
    *, deformation_interval, parameter_interval, form_error_bound, phase_error_bound
) -> _ConstitutivePreparationPathCapture:
    """Rebuild own-law bounds for every deformation in an exact closed interval.

    Both intervals are ordered finite represented reals in [0,1]. Source
    errors are finite nonnegative reals. The original source, law family,
    clocks and prospective two-stage budgets are fixed, with no report or
    caller-selected continuation horizon replacing their premises.
    """
    raw = _ordered(deformation_interval, "deformation_interval", limit=3)
    if len(raw) != 2:
        raise ValueError("deformation_interval must contain exactly two endpoints")
    lower, upper = tuple(_deformation(value) for value in raw)
    if not lower <= upper <= 1:
        raise ValueError("deformation_interval requires 0 <= lower <= upper <= 1")
    source = _prepare_path_source(
        parameter_interval=parameter_interval,
        scaled_time=_INITIAL_TIME,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
    )
    pi = pi_interval()
    eta = source.eta_bounds.hi
    # Keep exact rational deviations, including values smaller than the
    # interval grid. Source norms themselves retain the unit metric bound.
    current = _current_derivative_bound(upper, 0)
    deviation = current - 1
    hessian = _current_derivative_bound(upper, 1)
    lipschitz = 2 * hessian
    forcing = source.metric_unit_norm_bounds * current
    correction = 6 * upper * source.phase_error_bound**4
    source_storage = (
        source.initial_form_storage_upper_bound
        + source.quadratic_initial_phase_storage_upper_bound
        + correction
    )
    source_margin = I(_STORAGE_BUDGET - source_storage)
    source_budget = source_margin.lo > 0
    initial_zero = source.initial_zero_winding_margin_bounds.lo > 0
    decay = I(*_negative_exp_bounds(_GAP * _INITIAL_TIME))
    scaled_radius, phase_radius = _prepared_transit_radii(
        time=_INITIAL_TIME,
        decay_upper=decay.hi,
        gap_lower=_GAP,
        initial_norm_upper=source.scaled_initial_norm_upper_bound,
        feedback_upper=eta,
        forcing_upper=forcing.hi,
    )
    endpoint_phase_error = (
        phase_radius
        + source.scaled_initial_error_norm_upper_bound
        + source.initial_phase_error_norm_upper_bound
    )
    z = max(scaled_radius, eta * forcing.hi / _GAP)
    error = lipschitz * z
    # In y the homogeneous form term cancels exactly; the endpoint phase
    # error above retains it because theta, unlike y, still subtracts z.
    initial_error = (
        source.scaled_initial_error_norm_upper_bound
        + source.initial_phase_error_norm_upper_bound
        + eta * forcing.hi * _INITIAL_TIME
    )
    edge_error = I(-initial_error, initial_error)
    dipole = tuple(Q(int(j == 3) - int(j == 4)) for j in _NODES)
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
    # The actual closing current is j(pi-q). It shares sin(q), but not
    # the ninth-harmonic correction, with j(q). Both deviations are charged.
    current_speeds = tuple(value - 2 * deviation for value in sine_speeds)
    speed = min(current_speeds) - error
    passage_time = (_CLOSING_UPPER - _CLOSING_TARGET) / _PASSAGE_SPEED
    first_margins = initial_margins + (
        ("stage_one_coupled_gap_floor", coupled_lower - Q(899, 1000)),
        ("stage_one_ordinary_positive_floor", ordinary_lower - Q(3, 125)),
        ("stage_one_ordinary_acute_ceiling", pi.lo / 2 - ordinary_upper),
        ("stage_one_strict_passage_speed", speed - _PASSAGE_SPEED),
        ("stage_one_strict_hitting_deadline", _FIRST_DURATION - passage_time),
    )
    initial_gaps = I(Q(7, 4) - pi.hi, ordinary_upper)
    actual_cap = initial_gaps.abs_max + _SECOND_DURATION * error + z
    curvature = cos(I(_ACUTE_CAP)).lo - (hessian - 1)
    initial_squared = (45 * (_ACUTE_CAP + 2 * pi / 9) ** 2).hi
    relaxation_decay = I(*_negative_exp_bounds(_CONTRACTION * _SECOND_DURATION))
    phase_distance = _INITIAL_DISTANCE * relaxation_decay.hi + error / _CONTRACTION
    actual_distance = phase_distance + z
    form_norm = (1023 * pi).hi * z / sqrt(I(2)).lo
    form_storage = 2 * form_norm**2
    phase_storage = (9 * (1 - cos(2 * pi / 9))).hi + hessian * actual_distance**2
    storage = form_storage + phase_storage

    geometry = _derive(_NODES, _EDGES)
    target = source.path.source.target_phase_turns_by_class[0]
    principal = source.path.source.target_principal_edge_turns_by_class[0]
    offsets = tuple(
        turn - target[j] + target[i] for (i, j), turn in zip(_EDGES, principal)
    )
    if any(value.denominator != 1 for value in offsets):
        raise ArithmeticError("the fixed winding-one sector offsets must be integral")
    cycles = tuple(
        sum(sign * int(offset) for sign, offset in zip(row, offsets))
        for row in geometry.cycle_rows
    )
    faces = _sector_boundary_bounds(geometry, cycles)
    # V_lambda >= 1-cos globally: this reuses only a proved boundary
    # inequality. The endpoint H_lambda and strict capture test are own-law.
    barrier = min(faces)
    capture_margin = barrier - storage
    interior_curvature = cos(I(_CURVATURE_SPLIT)).lo - (hessian - 1)
    # For 3/2 <= |delta| <= pi/2, cos(9delta)>=0 and
    # cos(18delta)<=0. Hence the correction to cos(delta) is nonnegative;
    # strictly inside the acute cell cos(delta)>0. These angles check the
    # harmonic intervals without claiming a positive closed-face minimum.
    law_margins = (
        ("acute_inner_interval_curvature", interior_curvature),
        ("acute_split_below_boundary", pi.lo / 2 - _CURVATURE_SPLIT),
        ("ninth_harmonic_lower_angle", 9 * _CURVATURE_SPLIT - 4 * pi.hi),
        (
            "eighteenth_harmonic_lower_angle",
            18 * _CURVATURE_SPLIT - Q(17, 2) * pi.hi,
        ),
    )
    all_margins = (
        law_margins
        + first_margins
        + (
            ("stage_two_strict_acute_tube", _ACUTE_CAP - actual_cap),
            ("stage_two_cap_is_acute", pi.lo / 2 - _ACUTE_CAP),
            ("stage_two_uniform_phase_curvature", _GAP * curvature - _CONTRACTION),
            ("stage_two_initial_distance", _INITIAL_DISTANCE**2 - initial_squared),
            ("strict_full_state_capture_barrier", capture_margin),
        )
    )
    law_ok = all(margin > 0 for _, margin in law_margins)
    source_ok = source_budget and initial_zero
    passage = source_ok and law_ok and all(margin > 0 for _, margin in first_margins)
    capture = source_ok and all(margin > 0 for _, margin in all_margins)
    unresolved = tuple(name for name, margin in all_margins if margin <= 0) + tuple(
        name
        for name, ok in (
            ("original_source_storage_policy", source_budget),
            ("original_zero_winding_preparation", initial_zero),
        )
        if not ok
    )
    deadline = _FIRST_DURATION + _SECOND_DURATION
    tau = _INITIAL_TIME + deadline * 1023**2 * pi**2
    original_time = tau / Q(1023, 1024)
    return _ConstitutivePreparationPathCapture(
        deformation_interval=(lower, upper),
        source=source,
        preparation_dimension=8 * int(source.form_error_bound > 0)
        + 8 * int(source.phase_error_bound > 0),
        full_relative_preparation=source.form_error_bound > 0
        and source.phase_error_bound > 0,
        current_deviation_upper_bound=deviation,
        current_absolute_upper_bound=current,
        phase_hessian_upper_bound=hessian,
        global_field_lipschitz_upper_bound=lipschitz,
        forcing_norm_bounds=forcing,
        initial_potential_deformation_upper_bound=correction,
        common_initial_storage_upper_bound=source_storage,
        source_budget_margin_bounds=source_margin,
        source_budget_certified=source_budget,
        initial_zero_winding_certified=initial_zero,
        exponential_decay_bounds=decay,
        scaled_form_radius_upper_bound=scaled_radius,
        endpoint_phase_error_norm_upper_bound=endpoint_phase_error,
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
        stage_one_current_speed_lower_bounds=current_speeds,
        stage_one_passage_speed_lower_bound=speed,
        stage_one_passage_time_upper_bound=passage_time,
        stage_two_initial_gap_bounds=initial_gaps,
        stage_two_actual_gap_absolute_upper_bound=actual_cap,
        stage_two_curvature_lower_bound=curvature,
        stage_two_initial_distance_squared_upper_bound=initial_squared,
        stage_two_decay_bounds=relaxation_decay,
        stage_two_phase_distance_upper_bound=phase_distance,
        actual_phase_distance_at_capture_upper_bound=actual_distance,
        form_norm_upper_bound=form_norm,
        form_storage_upper_bound=form_storage,
        phase_storage_upper_bound=phase_storage,
        total_storage_upper_bound=storage,
        sector_boundary_face_lower_bounds=faces,
        sector_boundary_storage_lower_bound=barrier,
        capture_margin=capture_margin,
        inner_acute_curvature_lower_bound=interior_curvature,
        law_margins=law_margins,
        sufficient_margins=all_margins,
        uniform_law_bounds_certified=law_ok,
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
