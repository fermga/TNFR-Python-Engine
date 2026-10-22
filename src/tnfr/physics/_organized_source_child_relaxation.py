"""Conditional quantitative forgetting of organized-source child asymmetry.

The shared complete source law and its retained acute energy domain imply a
common quadratic decay bound even while the sine secant stiffness changes.
This reader evaluates that inequality from primitive state, not a response,
and bounds the instantaneous error of discarding the hidden coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I
from ._organized_source_capture import (
    _derive_organized_source_capture_barrier,
    _organized_source_offset_edge_angles,
    _OrganizedSourceCaptureBarrier,
)
from ._organized_source_child_orbit import (
    _encode_organized_source_child_orbit,
    _observe_organized_source_child_orbit,
    _OrganizedSourceChildOrbitObservation,
)
from ._repeated_organized_composition import _vector
from ._sine_lyapunov import (
    _sine_common_quadratic_coefficients,
    _sine_common_quadratic_initial_upper,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

_ENERGY_CEILING = Q(1, 100)
_COSINE_LOWER = Q(1, 2)
_HIDDEN_GAP_LOWER = Q(2)


@dataclass(frozen=True)
class _OrganizedSourceChildRelaxationBound:
    """Primitive evidence and an optional conditional squared-norm allowance.

    Initial metric bounds and exponential endpoints remain exact rational
    pairs, preserving nonzero quantities below the interval grid. Initial
    derivative intervals come from both complete fine rows at the supplied
    state. The decayed fields are absent when geometric or work admission
    fails. A certified allowance is not an evaluated hidden trajectory.

    Projected defects compare the full law projected at its current state
    with the same law at that state's symmetric projection. They do not
    compare two separately evolved solutions. The Coherence field bounds
    the difference of the target resultant arguments; multiplying by an
    independently admitted phase coefficient gives its increment defect.
    """

    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    capacities: tuple[Q, ...]
    elapsed_time: Q
    orbit: _OrganizedSourceChildOrbitObservation
    barrier: _OrganizedSourceCaptureBarrier
    edge_angle_bounds: tuple[I, ...]
    excess_bounds: I
    energy_ceiling: Q
    geometry_certified: bool
    in_acute_cell: bool
    below_energy_ceiling: bool
    boundary_ceiling_margin: Q
    collar_angle_upper: Q
    collar_margin: Q
    carrier_cosine_lower_bound: Q
    gamma_bounds: I
    cosine_lower_bound: Q
    hidden_gap_lower_bound: Q
    paired_capacity_lower: Q
    paired_capacity_upper: Q
    capacity_upper: Q
    initial_hidden_norm_squared: Q
    initial_weighted_square: Q
    initial_weighted_cross: Q
    initial_metric_bounds: tuple[Q, Q] | None
    initial_metric_upper: Q | None
    initial_metric_derivative_bounds: I
    initial_euclidean_squared_derivative_bounds: I
    metric_lower: Q | None
    metric_upper: Q | None
    decay_rate_lower: Q | None
    exponent: Q | None
    domain_certified: bool
    exponential_work_admitted: bool
    certified: bool
    decay_bounds: tuple[Q, Q] | None
    metric_upper_bound: Q | None
    hidden_norm_squared_upper_bound: Q | None
    projected_form_defect_upper_bound: Q | None
    projected_phase_defect_upper_bound: Q | None
    coherence_argument_defect_upper_bound: Q | None
    unavailable_reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "all_primitive_state_and_elapsed_time_admitted_before_geometry",
        "same_complete_quarter_rate_reciprocal_law_with_Gamma_absent",
        "positive_held_capacities_equal_on_the_two_exchanged_children",
        "raw_carrier_offset_cell_and_energy_excess_below_one_hundredth",
        "common_quadratic_metric_does_not_freeze_the_time_varying_secant_stiffness",
        "exact_rational_source_scaling_and_shared_negative_exponential_bounds",
        "initial_derivatives_retain_both_complete_fine_rows",
        "instantaneous_projected_law_defect_not_separately_evolved_core_error",
        "coherence_argument_defect_uses_the_conditional_resultant_floor_four",
        "a_separately_admitted_marked_word_keeps_hidden_state_and_paired_capacity",
        "no_live_word_admission_context_authentication_or_occurrence_selector",
        "no_trajectory_finite_time_symmetry_acquisition_or_guard_time_certificate",
    )


def _bound_organized_source_child_relaxation(
    *, form, phase_offsets, capacities, elapsed_time
):
    """Bound hidden relaxation on the admitted all-future acute sublevel.

    With V the fixed paired capacity, H=L_C9+2I and H/2<=K(t)<=H,
    W=a^T V^-1 a+b^T V^-1 b+2*gamma*a^T V^-1 b satisfies
    W'<=-r*W, r=nu_min*gamma^2/[4*(1+gamma)]. No derivative of K is
    required. The low-energy edge collar proves these stiffness bounds,
    and the shared capture barrier supplies their uninterrupted retention.

    A marked ideal word may be covered only with its separate admission,
    unchanged a,b,V and retained domain. The supplied elapsed time then
    counts continuous structural time; no event is executed by this reader.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    mobilities = _vector(capacities, "capacities", 45)
    duration = _exact(elapsed_time, "elapsed_time")
    if duration < 0:
        raise ValueError("elapsed_time must be nonnegative")
    state = _encode_organized_source_child_orbit(
        form=values, phase_offsets=offsets, capacities=mobilities
    )
    orbit = _observe_organized_source_child_orbit(
        form_coordinates=state.form_coordinates,
        phase_offset_coordinates=state.phase_offset_coordinates,
        capacity_coordinates=state.capacity_coordinates,
        hidden_gram=state.hidden_gram,
    )
    barrier = _derive_organized_source_capture_barrier()
    geometry = state.geometry.fine_geometry
    geometry_ok = (
        geometry == orbit.state.geometry.fine_geometry
        and geometry == orbit.fine.unit_capacity_reference.geometry
        and geometry == barrier.geometry
    )
    angles = _organized_source_offset_edge_angles(geometry, offsets)
    acute = all(angle.abs_max < barrier.cell_edge_gap_bounds.lo for angle in angles)
    excess = orbit.joint_storage_bounds - barrier.target_storage_bounds
    below = excess.hi < _ENERGY_CEILING
    boundary_margin = barrier.boundary_energy_lower_bound - _ENERGY_CEILING
    carrier_cosine = 1 - barrier.cycle_storage_minimum_bounds.hi / 9
    # The existing cycle Taylor bound gives sum_e u_e^2 <= 4E < 1/25.
    # The entire child-exchange chord therefore lies below this same collar.
    collar = geometry.alpha_bounds.hi + Q(1, 5)
    collar_margin = 2 * barrier.cell_edge_gap_bounds.lo / 3 - collar
    gamma = geometry.gamma_bounds
    gamma_ok = 0 < gamma.lo <= gamma.hi <= Q(1, 2)

    paired = tuple(
        state.capacities[block[0]] for block in state.geometry.blocks if len(block) == 2
    )
    capacity_lower, capacity_upper = min(paired), max(paired)
    maximum_capacity = max(mobilities)
    a, b = state.hidden_coordinates[:9], state.hidden_coordinates[9:]
    da, db = orbit.hidden_tau_rate_bounds[:9], orbit.hidden_tau_rate_bounds[9:]
    squared = sum((x * x + y * y for x, y in zip(a, b)), Q(0))
    weighted_square = sum(
        ((x * x + y * y) / nu for x, y, nu in zip(a, b, paired)), Q(0)
    )
    weighted_cross = sum((x * y / nu for x, y, nu in zip(a, b, paired)), Q(0))
    euclidean_derivative = 2 * sum(
        (x * dx + y * dy for x, y, dx, dy in zip(a, b, da, db)), I(0)
    )
    metric_derivative = 2 * sum(
        (
            (x / nu + gamma * (y / nu)) * dx + (y / nu + gamma * (x / nu)) * dy
            for x, y, dx, dy, nu in zip(a, b, da, db, paired)
        ),
        I(0),
    )

    metric_lower = metric_upper = rate = initial_upper = initial_bounds = None
    if gamma_ok:
        metric_lower, metric_upper, rate = _sine_common_quadratic_coefficients(
            gamma_lower=gamma.lo,
            gamma_upper=gamma.hi,
            cosine_lower=_COSINE_LOWER,
            gap_lower=_HIDDEN_GAP_LOWER,
            capacity_lower=capacity_lower,
            capacity_upper=capacity_upper,
        )
        initial_upper = _sine_common_quadratic_initial_upper(
            gamma_lower=gamma.lo,
            gamma_upper=gamma.hi,
            weighted_square=weighted_square,
            weighted_cross=weighted_cross,
        )
        opposite_upper = _sine_common_quadratic_initial_upper(
            gamma_lower=gamma.lo,
            gamma_upper=gamma.hi,
            weighted_square=weighted_square,
            weighted_cross=-weighted_cross,
        )
        initial_bounds = (2 * weighted_square - opposite_upper, initial_upper)

    conditions = (
        (geometry_ok, "shared_geometry_unavailable"),
        (barrier.barrier_certified, "acute_boundary_barrier_unavailable"),
        (acute, "state_not_certified_in_fixed_branch_acute_cell"),
        (below, "state_excess_not_strictly_below_fixed_ceiling"),
        (boundary_margin > 0, "fixed_ceiling_not_below_acute_boundary"),
        (carrier_cosine > Q(3, 4), "carrier_cosine_taylor_bound_unavailable"),
        (collar_margin > 0, "half_stiffness_edge_collar_unavailable"),
        (gamma_ok, "positive_gamma_at_most_one_half_unavailable"),
    )
    reasons = [reason for passed, reason in conditions if not passed]
    domain = not reasons
    exponent = rate * duration if rate is not None else None
    work_ok = exponent is not None and exponent <= _MAX_RATIONAL_EXPONENT
    if not work_ok:
        reasons.append("negative_exponential_work_limit_unavailable")
    certified = domain and work_ok
    decay = returned_metric = returned_squared = form_defect = None
    phase_defect = argument_defect = None
    if certified:
        decay = _negative_exp_bounds(exponent)
        # Keep source scaling outside dyadic interval arithmetic: exact
        # symmetry remains zero and arbitrarily small asymmetry stays small.
        returned_metric = initial_upper * decay[1]
        returned_squared = returned_metric / metric_lower
        form_defect = 5 * gamma.hi * maximum_capacity * returned_squared / 4
        phase_defect = Q(0)
        argument_defect = returned_squared / 4

    return _OrganizedSourceChildRelaxationBound(
        form=values,
        phase_offsets=offsets,
        capacities=mobilities,
        elapsed_time=duration,
        orbit=orbit,
        barrier=barrier,
        edge_angle_bounds=angles,
        excess_bounds=excess,
        energy_ceiling=_ENERGY_CEILING,
        geometry_certified=geometry_ok,
        in_acute_cell=acute,
        below_energy_ceiling=below,
        boundary_ceiling_margin=boundary_margin,
        collar_angle_upper=collar,
        collar_margin=collar_margin,
        carrier_cosine_lower_bound=carrier_cosine,
        gamma_bounds=gamma,
        cosine_lower_bound=_COSINE_LOWER,
        hidden_gap_lower_bound=_HIDDEN_GAP_LOWER,
        paired_capacity_lower=capacity_lower,
        paired_capacity_upper=capacity_upper,
        capacity_upper=maximum_capacity,
        initial_hidden_norm_squared=squared,
        initial_weighted_square=weighted_square,
        initial_weighted_cross=weighted_cross,
        initial_metric_bounds=initial_bounds,
        initial_metric_upper=initial_upper,
        initial_metric_derivative_bounds=metric_derivative,
        initial_euclidean_squared_derivative_bounds=euclidean_derivative,
        metric_lower=metric_lower,
        metric_upper=metric_upper,
        decay_rate_lower=rate,
        exponent=exponent,
        domain_certified=domain,
        exponential_work_admitted=work_ok,
        certified=certified,
        decay_bounds=decay,
        metric_upper_bound=returned_metric,
        hidden_norm_squared_upper_bound=returned_squared,
        projected_form_defect_upper_bound=form_defect,
        projected_phase_defect_upper_bound=phase_defect,
        coherence_argument_defect_upper_bound=argument_defect,
        unavailable_reasons=tuple(reasons),
    )
