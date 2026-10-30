"""Conditional error against an independently evolved child-symmetric core.

The reference starts at the exact projection of the supplied fine state.
Both complete rows evolve with unit capacities and no events. This reader
bounds their mathematical difference without evaluating either trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._comparison_flow import decay_convolution_upper
from ..mathematics._rational_interval import I
from ._organized_source_capture import _organized_source_offset_edge_angles
from ._organized_source_child_quotient import (
    _lift_organized_source_child_quotient,
    _OrganizedSourceChildQuotientState,
)
from ._organized_source_child_relaxation import (
    _bound_organized_source_child_relaxation,
    _OrganizedSourceChildRelaxationBound,
)
from ._organized_source_port import (
    _observe_organized_source_port,
    _OrganizedSourceObservation,
    _source_emission_geometry,
)
from ._repeated_organized_composition import _vector
from ._sine_lyapunov import _sine_common_quadratic_coefficients
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds


@dataclass(frozen=True)
class _OrganizedSourceChildTrackingBound:
    """Fresh initial evidence and optional uniform/pointwise error bounds.

    Error norms use the 45-node lift in both channels, equivalently the
    orbit-mass-weighted 36-node norm. The independently evolved reference
    starts at ``reference_state``; its later state is not computed here.
    Uniform bounds remain available when only exponential work admission
    fails. All source-dependent scaling remains exact rational arithmetic.
    """

    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    capacities: tuple[Q, ...]
    elapsed_time: Q
    initial_relaxation: _OrganizedSourceChildRelaxationBound
    reference_state: _OrganizedSourceChildQuotientState
    reference_observation: _OrganizedSourceObservation
    reference_edge_angle_bounds: tuple[I, ...]
    reference_excess_bounds: I
    initial_form_total: Q
    reference_form_total: Q
    initial_phase_offset_total: Q
    reference_phase_offset_total: Q
    matched_means: bool
    geometry_certified: bool
    reference_in_acute_cell: bool
    reference_below_energy_ceiling: bool
    full_gap_lower_bound: Q | None
    metric_lower: Q | None
    metric_upper: Q | None
    hidden_decay_rate_lower: Q | None
    error_decay_rate_lower: Q | None
    forcing_decay_rate: Q | None
    rate_separation: Q | None
    hidden_norm_squared_coefficient: Q | None
    forcing_coefficient: Q | None
    uniform_metric_upper_bound: Q | None
    uniform_squared_error_upper_bound: Q | None
    error_exponent: Q | None
    forcing_exponent: Q | None
    error_decay_bounds: tuple[Q, Q] | None
    forcing_decay_bounds: tuple[Q, Q] | None
    convolution_upper_bound: Q | None
    pointwise_metric_upper_bound: Q | None
    pointwise_squared_error_upper_bound: Q | None
    domain_certified: bool
    exponential_work_admitted: bool
    certified: bool
    unavailable_reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "all_primitives_and_exact_unit_capacities_admitted_before_geometry",
        "same_complete_quarter_rate_reciprocal_law_with_Gamma_absent",
        "no_inputs_events_or_capacity_changes_in_either_solution",
        "independent_symmetric_reference_initialized_at_exact_child_projection",
        "both_raw_lifted_channel_means_match_and_remain_conserved",
        "fine_and_reference_acute_energy_domains_are_retained_separately",
        "full_support_gap_applies_on_the_zero_mean_error_leaf",
        "forty_five_node_norm_equals_orbit_mass_weighted_quotient_norm",
        "edge_forcing_controls_accumulated_error_of_both_complete_rows",
        "uniform_bound_survives_pointwise_exponential_work_unavailability",
        "exact_rational_hidden_scaling_without_a_dyadic_error_floor",
        "no_trajectory_evaluated_error_guard_time_or_hybrid_comparison",
    )


def _bound_organized_source_child_tracking(
    *, form, phase_offsets, capacities, elapsed_time
):
    """Bound Pi*z(t)-w(t) with w(0)=Pi*z(0) and no subsequent events.

    The shared hidden estimate gives ||(a,b)||^2 <= C0 exp(-r_h*t).
    On the equal-mean error leaf, the full-support common metric and the
    actual edge-forcing inequality yield
    W_error' <= -r_e W_error + 10*gamma^2*C0^2*exp(-2*r_h*t).
    Here r_h=15*r_e and W_error(0)=0. The shared convolution owner encloses
    its integral. Its all-time bound is F/(2*r_h-r_e); division by the
    metric lower coefficient converts it to the joint squared norm.

    The old short-window Lipschitz allowance is not used. The entire fine
    and reference domain follows from fresh acute/energy admission and
    the same passive continuous law, not from an evaluated response.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    mobilities = _vector(capacities, "capacities", 45)
    duration = _exact(elapsed_time, "elapsed_time")
    if duration < 0:
        raise ValueError("elapsed_time must be nonnegative")
    if any(value != 1 for value in mobilities):
        raise ValueError("child tracking requires exactly unit held capacities")

    initial = _bound_organized_source_child_relaxation(
        form=values, phase_offsets=offsets, capacities=mobilities, elapsed_time=Q(0)
    )
    state = initial.orbit.state
    reference = _lift_organized_source_child_quotient(
        form_coordinates=state.form_coordinates,
        phase_offset_coordinates=state.phase_offset_coordinates,
        capacity_coordinates=state.capacity_coordinates,
    )
    observation = _observe_organized_source_port(
        form=reference.form, phase_offsets=reference.phase_offsets
    )
    geometry = state.geometry.fine_geometry
    support_ok, _diameter, cycle_gap, child_gap, _squares = _source_emission_geometry(
        geometry
    )
    geometry_ok = (
        support_ok
        and initial.geometry_certified
        and geometry == reference.geometry.fine_geometry
        and geometry == observation.geometry
    )
    gap = min(cycle_gap, child_gap) if geometry_ok else None
    angles = _organized_source_offset_edge_angles(geometry, reference.phase_offsets)
    acute = all(
        angle.abs_max < initial.barrier.cell_edge_gap_bounds.lo for angle in angles
    )
    excess = observation.joint_storage_bounds - initial.barrier.target_storage_bounds
    below = excess.hi < initial.energy_ceiling
    form_total, reference_form_total = sum(values, Q(0)), sum(reference.form, Q(0))
    phase_total = sum(offsets, Q(0))
    reference_phase_total = sum(reference.phase_offsets, Q(0))
    matched = (
        form_total == reference_form_total and phase_total == reference_phase_total
    )
    gamma = initial.gamma_bounds
    hidden_rate = initial.decay_rate_lower
    lower = upper = error_rate = forcing_rate = separation = None
    hidden_coefficient = forcing_coefficient = None
    if geometry_ok and hidden_rate is not None:
        lower, upper, homogeneous_rate = _sine_common_quadratic_coefficients(
            gamma_lower=gamma.lo,
            gamma_upper=gamma.hi,
            cosine_lower=initial.cosine_lower_bound,
            gap_lower=gap,
            capacity_lower=Q(1),
            capacity_upper=Q(1),
        )
        # Half of the homogeneous derivative pays for the two Young bounds
        # on the actual form-only edge forcing. The phase row is retained.
        error_rate = homogeneous_rate / 2
        forcing_rate = 2 * hidden_rate
        separation = forcing_rate - error_rate
        hidden_coefficient = initial.initial_metric_upper / lower
        forcing_coefficient = 10 * gamma.hi**2 * hidden_coefficient**2

    reasons = list(initial.unavailable_reasons)
    for passed, reason in (
        (geometry_ok, "full_support_gap_geometry_unavailable"),
        (acute, "reference_not_certified_in_fixed_branch_acute_cell"),
        (below, "reference_excess_not_strictly_below_fixed_ceiling"),
        (matched, "projected_reference_means_do_not_match"),
        (
            error_rate is not None
            and error_rate > 0
            and hidden_rate == 15 * error_rate,
            "positive_hidden_and_error_rate_relation_unavailable",
        ),
    ):
        if not passed:
            reasons.append(reason)
    domain = initial.domain_certified and not reasons
    uniform_metric = uniform_squared = None
    if domain:
        uniform_metric = forcing_coefficient / separation
        uniform_squared = uniform_metric / lower
    error_exponent = error_rate * duration if error_rate is not None else None
    forcing_exponent = forcing_rate * duration if forcing_rate is not None else None
    work_ok = (
        error_exponent is not None
        and forcing_exponent is not None
        and max(error_exponent, forcing_exponent) <= _MAX_RATIONAL_EXPONENT
    )
    if not work_ok:
        reasons.append("negative_exponential_work_limit_unavailable")
    certified = domain and work_ok
    error_decay = forcing_decay = convolution = point_metric = point_squared = None
    if certified:
        error_decay = _negative_exp_bounds(error_exponent)
        forcing_decay = _negative_exp_bounds(forcing_exponent)
        convolution = decay_convolution_upper(
            error_rate, forcing_rate, duration, error_decay, forcing_decay
        )
        # All source scaling stays outside interval arithmetic, preserving
        # exact zero and quartic scaling for subgrid hidden amplitudes.
        point_metric = forcing_coefficient * convolution
        point_squared = point_metric / lower
    return _OrganizedSourceChildTrackingBound(
        form=values,
        phase_offsets=offsets,
        capacities=mobilities,
        elapsed_time=duration,
        initial_relaxation=initial,
        reference_state=reference,
        reference_observation=observation,
        reference_edge_angle_bounds=angles,
        reference_excess_bounds=excess,
        initial_form_total=form_total,
        reference_form_total=reference_form_total,
        initial_phase_offset_total=phase_total,
        reference_phase_offset_total=reference_phase_total,
        matched_means=matched,
        geometry_certified=geometry_ok,
        reference_in_acute_cell=acute,
        reference_below_energy_ceiling=below,
        full_gap_lower_bound=gap,
        metric_lower=lower,
        metric_upper=upper,
        hidden_decay_rate_lower=hidden_rate,
        error_decay_rate_lower=error_rate,
        forcing_decay_rate=forcing_rate,
        rate_separation=separation,
        hidden_norm_squared_coefficient=hidden_coefficient,
        forcing_coefficient=forcing_coefficient,
        uniform_metric_upper_bound=uniform_metric,
        uniform_squared_error_upper_bound=uniform_squared,
        error_exponent=error_exponent,
        forcing_exponent=forcing_exponent,
        error_decay_bounds=error_decay,
        forcing_decay_bounds=forcing_decay,
        convolution_upper_bound=convolution,
        pointwise_metric_upper_bound=point_metric,
        pointwise_squared_error_upper_bound=point_squared,
        domain_certified=domain,
        exponential_work_admitted=work_ok,
        certified=certified,
        unavailable_reasons=tuple(reasons),
    )
