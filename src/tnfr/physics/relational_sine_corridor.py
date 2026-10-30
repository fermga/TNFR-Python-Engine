"""Conditional finite passage through a conservative C5 saddle corridor.

The source is an exact member of a supplied sign/reflection invariant family.
The proof retains all five receivers, five leaves and their contact feedback.
It bounds a conjugate momentum on a declared phase strip; it neither integrates
a trajectory nor certifies acute acquisition or transverse uncertainty boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import TYPE_CHECKING, Any

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from .relational_sine_comparison import (
    SineExchangeComparison,
    _comparison_from_state,
    _comparison_neighbors,
    _sine_phase_rate_numerators,
    _sine_state_from_rows,
    _validate_comparison_labels,
)
from .relational_sine_partition import _private_leaf_support
from .relational_sine_regional import _branches
from .relational_sine_symmetry import (
    SineInvolutionReduction,
    assess_sine_involution_reduction,
)

if TYPE_CHECKING:
    from .relational_sine_resonance import SineC5LeafSaddle

__all__ = (
    "SineSaddleCorridor",
    "SineSaddleRetentionBand",
    "SineSaddleFormation",
    "SineSaddlePreparation",
    "assess_sine_saddle_corridor",
    "assess_sine_saddle_retention_band",
    "assess_sine_saddle_formation",
    "prepare_sine_saddle_state",
)


def _cycle_floor(value):
    """The minimum cycle potential at fixed u on this signed family."""
    return 5 - cos(2 * I(value)) - 4 * cos(I(value) / 2)


def _signed_reduction(source, contacts):
    permutation = list(range(len(source.nodes)))
    for position, (node, leaf) in enumerate(contacts):
        permutation[node] = contacts[4 - position][0]
        permutation[leaf] = contacts[4 - position][1]
    return assess_sine_involution_reduction(
        source, permutation_indices=tuple(permutation), sign=-1
    )


@dataclass(frozen=True)
class SineSaddleCorridor:
    """A sufficient exact-source directed exit, in scaled time tau=t/pi.

    A certified exit has winding zero. The initial state has winding one,
    and reversal supplies the opposite finite connection. Neither endpoint
    is asserted acute. Independent errors that break the source symmetry
    require a separate complete-field comparison argument.
    """

    source: SineExchangeComparison
    reduction: SineInvolutionReduction
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    contacts: tuple[tuple[Any, Any], ...]
    lower_phase: Q
    upper_phase: Q
    form_coordinates: tuple[Q, ...]
    phase_coordinates: tuple[Q, ...]
    corridor_coordinates: tuple[Q, ...]
    initial_cycle_gap_bounds: tuple[I, ...]
    initial_edge_turn_offsets: tuple[int, ...] | None
    initial_winding: int | None
    full_form_storage: Q
    full_phase_storage_bounds: I
    full_storage_bounds: I
    initial_momentum: Q
    initial_reaction_velocity: Q
    initial_momentum_rate_bounds: I
    left_potential_floor_bounds: I
    right_potential_floor_bounds: I
    force_bracket_bounds: I
    force_lower_bound: Q
    momentum_abs_upper_bound: Q
    left_exit_momentum_upper_bound: Q
    directed_momentum_margin: Q
    residence_time_upper_bound: Q | None
    original_time_upper_bound: Q | None
    source_family_certified: bool
    source_strip_certified: bool
    initial_branches_certified: bool
    branch_protection_certified: bool
    positive_force_certified: bool
    left_exit_excluded: bool
    directed_exit_certified: bool
    exit_winding: int | None
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_unit_conservative_sine_law_on_C5_and_five_private_leaves",
        "exact_source_fixed_by_combined_sign_and_reflection_not_an_independent_box",
        "real_phase_lifts_and_zero_fixed_node_origins_are_declared",
        "source_energy_rebuilt_from_full_primitive_state_no_cached_verdict_consumed",
        "momentum_monotonicity_does_not_assert_reaction_coordinate_monotonicity",
        "positive_force_and_constrained_energy_bound_exclude_the_lower_exit",
        "finite_upper_exit_changes_winding_one_to_zero_with_contact_feedback_retained",
        "time_bound_is_an_upper_bound_not_a_computed_hitting_time",
        "no_trajectory_acute_retention_autonomous_preparation_or_physical_identification",
        "transverse_source_errors_require_a_separate_full_field_comparison",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_comparison_labels(self.reduction.source)
        _validate_label_groups(self.cycle, *self.contacts)
        return {"schema": "tnfr.sine-saddle-corridor.v1", "report": _project(self)}


def assess_sine_saddle_corridor(
    source, *, cycle, lower_phase, upper_phase
) -> SineSaddleCorridor:
    """Certify a sufficient nonlinear passage without evaluating a trajectory.

    The declared strip satisfies -2*pi/3 < lower_phase < -pi/2 < upper_phase < 0.
    Canonical reduced coordinates are receiver (u,v,0,-v,-u), leaf
    (p,q,0,-q,-p), and the corresponding four form coordinates (a,b,c,d).
    Source hypotheses that fail return unavailable; malformed laws, support,
    scalars or strip declarations reject. All comparisons are outward.
    """
    source, edges, indices, pairs, contacts = _private_leaf_support(source, cycle)
    state = _sine_state_from_rows(
        source.nodes,
        source.edges,
        source.epi,
        source.phase,
        source.capacity,
        _comparison_neighbors(source),
    )
    source = _comparison_from_state(state, source.reference_model)
    lower = exact_or_represented_real(lower_phase, "lower_phase")
    upper = exact_or_represented_real(upper_phase, "upper_phase")
    pi = pi_interval()
    if not (-2 * pi.lo / 3 < lower < -pi.hi / 2 and -pi.lo / 2 < upper < 0):
        raise ValueError(
            "corridor requires -2*pi/3 < lower_phase < -pi/2 < upper_phase < 0"
        )

    reduction = _signed_reduction(source, contacts)
    family = reduction.source_trajectory_reduction_certified
    representatives = (indices[0], indices[1], contacts[0][1], contacts[1][1])
    forms = tuple(source.epi[i] for i in representatives)
    phases = tuple(source.phase[i] for i in representatives)
    a, b, c, d = forms
    u, v, leaf_u, leaf_v = phases
    coordinates = (u, v - u / 2, leaf_u - u, leaf_v - v)
    source_strip = lower < u < -pi.hi / 2

    gaps = tuple(I(source.phase[j] - source.phase[i]) for i, j in pairs)
    offsets, _, winding, _ = _branches(gaps)
    initial_branches = offsets == (0, 0, 0, 0, -1) and winding == 1
    form_storage = source.form_storage
    phase_storage = source.phase_storage
    storage = source.storage
    energy_upper = storage.hi
    branch_protection = energy_upper < 4
    momentum = 6 * a + 3 * b + 2 * c + d
    reaction_velocity = _sine_phase_rate_numerators(
        source.reference_model, source.degrees, source.form_gradient, source.capacity
    )[indices[0]]

    # Recompute the actual full row even for a source outside the invariant
    # family; such a diagnostic does not inherit the conditional force theorem.
    momentum_rate = pi * sum(
        (
            coefficient * source.form_rates[index]
            for coefficient, index in zip((6, 3, 2, 1), representatives)
        ),
        I(0),
    )

    left_floor, right_floor = _cycle_floor(lower), _cycle_floor(upper)
    left_cosine = cos(I(lower))
    force_bracket = 6 + 8 * left_cosine + 6 * left_cosine**2 - energy_upper
    prefactor_right = -sin(I(upper) / 2) / (2 * cos(I(upper) / 2))
    prefactor_left = -sin(I(lower) / 2) / (2 * cos(I(lower) / 2))
    prefactor = I(prefactor_right.lo, prefactor_left.hi)
    force_lower = (prefactor * I(force_bracket.lo)).lo
    positive_force = force_lower > 0
    # Negative allowances mean the face is energetically inaccessible; zero
    # remains a conservative allowance for the sufficient momentum test.
    right_allowance = max(Q(0), energy_upper - right_floor.lo)
    left_allowance = max(Q(0), energy_upper - left_floor.lo)
    momentum_upper = sqrt(I(53 * right_allowance)).hi
    left_momentum_upper = sqrt(I(44 * left_allowance)).hi
    momentum_margin = momentum - left_momentum_upper
    left_excluded = family and source_strip and positive_force and momentum_margin > 0
    prerequisites = (
        (family, "exact_signed_source_membership_not_certified"),
        (source_strip, "initial_reaction_coordinate_outside_declared_strip"),
        (initial_branches, "initial_winding_one_path_lifts_not_certified"),
        (branch_protection, "paired_path_branch_protection_not_certified"),
        (positive_force, "uniform_positive_momentum_force_not_certified"),
        (momentum_margin > 0, "lower_exit_momentum_exclusion_not_certified"),
        (momentum_upper >= momentum, "outward_momentum_budget_not_certified"),
    )
    reasons = tuple(reason for passed, reason in prerequisites if not passed)
    certified = not reasons
    duration = (momentum_upper - momentum) / force_lower if certified else None
    return SineSaddleCorridor(
        source=source,
        reduction=reduction,
        cycle=tuple(source.nodes[i] for i in indices),
        cycle_indices=indices,
        contacts=tuple((source.nodes[i], source.nodes[j]) for i, j in contacts),
        lower_phase=lower,
        upper_phase=upper,
        form_coordinates=forms,
        phase_coordinates=phases,
        corridor_coordinates=coordinates,
        initial_cycle_gap_bounds=gaps,
        initial_edge_turn_offsets=offsets,
        initial_winding=winding,
        full_form_storage=form_storage,
        full_phase_storage_bounds=phase_storage,
        full_storage_bounds=storage,
        initial_momentum=momentum,
        initial_reaction_velocity=reaction_velocity,
        initial_momentum_rate_bounds=momentum_rate,
        left_potential_floor_bounds=left_floor,
        right_potential_floor_bounds=right_floor,
        force_bracket_bounds=force_bracket,
        force_lower_bound=force_lower,
        momentum_abs_upper_bound=momentum_upper,
        left_exit_momentum_upper_bound=left_momentum_upper,
        directed_momentum_margin=momentum_margin,
        residence_time_upper_bound=duration,
        original_time_upper_bound=None if duration is None else pi.hi * duration,
        source_family_certified=family,
        source_strip_certified=source_strip,
        initial_branches_certified=initial_branches,
        branch_protection_certified=branch_protection,
        positive_force_certified=positive_force,
        left_exit_excluded=left_excluded,
        directed_exit_certified=certified,
        exit_winding=0 if certified else None,
        status="certified" if certified else "unavailable",
        reasons=reasons,
    )


@dataclass(frozen=True)
class SineSaddleRetentionBand:
    """A conditional dwell theorem, not a certificate about the captured state.

    The actual connecting orbit must reach the declared deep coordinate in
    the same signed family, energy sublevel and principal path component.
    This report admits the support/law and checks the fixed theorem constants.
    It does not supply that orbit, its event times or its checkpoint state.
    """

    source: SineExchangeComparison
    reduction: SineInvolutionReduction
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    contacts: tuple[tuple[Any, Any], ...]
    lower_phase: Q
    upper_phase: Q
    deep_phase: Q
    full_storage_upper_bound: Q
    paired_face_surplus_bounds: I
    closing_gap_margin_bounds: I
    cycle_gap_margin_lower_bound: Q
    minimum_cycle_storage_bounds: I
    computed_reaction_speed_upper_bound: Q
    reaction_speed_upper_bound: Q
    backward_band_residence_lower_bound: Q
    forward_band_residence_lower_bound: Q
    scaled_dwell_lower_bound: Q
    checkpoint_offset_from_deep_hit: Q
    retained_scaled_duration: Q
    target_form_error_bound: Q
    target_phase_error_bound: Q
    error_growth_upper_bound: Q
    target_cycle_gap_error_upper_bound: Q
    target_retention_margin_lower_bound: Q
    band_geometry_certified: bool
    finite_dwell_implication_certified: bool
    independent_target_ball_implication_certified: bool
    theorem_certified: bool
    captured_source_retention_certified: bool
    captured_source_formation_certified: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    conditional_premises: tuple[str, ...] = (
        "one_actual_complete_orbit_in_the_signed_C5_private_leaf_family",
        "full_conserved_storage_at_most_the_declared_ceiling",
        "four_raw_path_gaps_initially_in_minus_pi_pi_and_retained_by_storage_below_four",
        "actual_deep_hit_on_that_same_complete_orbit",
        "checkpoint_is_one_half_scaled_time_before_the_actual_deep_hit",
        "fixed_positive_held_unit_capacities_no_events_inputs_or_changed_law",
    )
    scope: tuple[str, ...] = (
        "captured_source_anchors_support_and_law_only_not_a_claim_of_band_membership",
        "constants_are_one_sufficient_theorem_choice_not_physical_parameters",
        "deep_hit_to_either_band_face_requires_positive_travel_in_both_time_directions",
        "nominal_retention_window_runs_from_deep_minus_one_half_to_deep_plus_one_half",
        "independent_errors_cover_every_fine_node_form_and_phase_coordinate",
        "target_ball_uses_complete_field_Lipschitz_two_in_tau_not_symmetry_of_perturbations",
        "no_actual_checkpoint_event_time_preparation_or_trajectory_is_evaluated",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_comparison_labels(self.reduction.source)
        _validate_label_groups(self.cycle, *self.contacts)
        return {
            "schema": "tnfr.sine-saddle-retention-band.v1",
            "report": _project(self),
        }


def assess_sine_saddle_retention_band(source, *, cycle) -> SineSaddleRetentionBand:
    """Check a reusable conditional acute dwell band on the admitted support.

    The report concerns the implication from an actual deep hit, not
    membership or future motion of ``source``. All current-state certification
    flags remain false, even when the conditional theorem is certified.
    """
    source, _, indices, _, contacts = _private_leaf_support(source, cycle)
    reduction = _signed_reduction(source, contacts)
    lower, upper, deep = Q(-13, 5), Q(-12, 5), Q(-5, 2)
    energy = Q(35001, 10000)
    pi = pi_interval()
    # On this entire band cos(u)<0 and sin(u)+1/2<0. Therefore the
    # paired acute-face cost7/2+2(sin(u)+1/2)^2 is smallest at its left end.
    surplus = Q(7, 2) + 2 * (sin(I(lower)) + Q(1, 2)) ** 2 - energy
    closing_margin = -2 * upper - 3 * pi / 2
    margin = Q(1, 12000)
    minimum_storage = 5 * (1 - cos(2 * pi / 5))
    speed = sqrt(4 * (energy - minimum_storage) / 9).hi
    declared_speed = Q(1, 7)
    backward = (upper - deep) / declared_speed
    forward = min(deep - lower, upper - deep) / declared_speed
    dwell = backward + forward
    duration, radius, growth = Q(1), Q(1, 2**18), Q(9)
    error = 2 * radius * growth
    retained_margin = margin - error

    geometry = (
        reduction.family_invariance_certified
        and cos(I(lower, upper)).hi < 0
        and (sin(I(lower)) + Q(1, 2)).hi < 0
        and -pi.lo < lower < deep < upper < -3 * pi.hi / 4
        and surplus.lo > Q(1, 3000)
        and closing_margin.lo > margin
        and surplus.lo / 4 > margin
        and energy < 4
    )
    dwell_certified = (
        geometry
        and speed < declared_speed
        and backward > duration / 2
        and forward > duration / 2
        and dwell > duration
    )
    target_certified = dwell_certified and retained_margin > 0
    reasons = tuple(
        reason
        for passed, reason in (
            (geometry, "conditional_band_geometry_not_certified"),
            (dwell_certified, "conditional_finite_dwell_not_certified"),
            (target_certified, "conditional_full_target_ball_not_certified"),
        )
        if not passed
    )
    return SineSaddleRetentionBand(
        source=source,
        reduction=reduction,
        cycle=tuple(source.nodes[i] for i in indices),
        cycle_indices=indices,
        contacts=tuple((source.nodes[i], source.nodes[j]) for i, j in contacts),
        lower_phase=lower,
        upper_phase=upper,
        deep_phase=deep,
        full_storage_upper_bound=energy,
        paired_face_surplus_bounds=surplus,
        closing_gap_margin_bounds=closing_margin,
        cycle_gap_margin_lower_bound=margin,
        minimum_cycle_storage_bounds=minimum_storage,
        computed_reaction_speed_upper_bound=speed,
        reaction_speed_upper_bound=declared_speed,
        backward_band_residence_lower_bound=backward,
        forward_band_residence_lower_bound=forward,
        scaled_dwell_lower_bound=dwell,
        checkpoint_offset_from_deep_hit=Q(-1, 2),
        retained_scaled_duration=duration,
        target_form_error_bound=radius,
        target_phase_error_bound=radius,
        error_growth_upper_bound=growth,
        target_cycle_gap_error_upper_bound=error,
        target_retention_margin_lower_bound=retained_margin,
        band_geometry_certified=geometry,
        finite_dwell_implication_certified=dwell_certified,
        independent_target_ball_implication_certified=target_certified,
        theorem_certified=target_certified,
        captured_source_retention_certified=False,
        captured_source_formation_certified=False,
        status="certified" if target_certified else "unavailable",
        reasons=reasons,
    )


@dataclass(frozen=True)
class SineSaddleFormation:
    """A correlated preparation and same-orbit finite formation existence proof.

    ``source`` only anchors the complete support and law. The mathematical
    preparation uses the saddle's one correlated algebraic eigenvector.
    Neither that recipe nor the event-defined zero-winding source center is
    identified with the captured source. The positive independent source
    radius is retained symbolically as prefactor*exp(-exponent).
    """

    source: SineExchangeComparison
    saddle: SineC5LeafSaddle
    retention_band: SineSaddleRetentionBand
    epsilon: Q
    phase_direction_quadratic_bounds: I
    momentum_direction_bounds: I
    initial_storage_excess_lower_bound: Q
    initial_storage_excess_upper_bound: Q
    full_storage_upper_bound: Q
    local_remainder_bound: Q
    normalized_local_remainder_bound: Q
    outer_phase_displacement_coefficient_lower_bound: Q
    outer_momentum_coefficient_lower_bound: Q
    inner_phase_displacement_coefficient_upper_bound: Q
    inner_negative_momentum_coefficient_lower_bound: Q
    momentum_exit_allowance_coefficient_upper_bound: Q
    outer_momentum_gate_margin: Q
    inner_momentum_gate_margin: Q
    local_phase_error_bound: Q
    outer_force_lower_bound: Q
    inner_force_lower_bound: Q
    connection_time_upper_bound: Q
    source_radius_prefactor: Q
    source_radius_exponent: Q
    initial_source_branch_margin_lower_bound: Q
    target_radius: Q
    retained_scaled_duration: Q
    correlated_preparation_certified: bool
    local_connection_certified: bool
    outer_corridor_connection_certified: bool
    inner_corridor_connection_certified: bool
    formation_existence_certified: bool
    independent_source_ball_existence_certified: bool
    captured_source_formation_certified: bool
    captured_source_retention_certified: bool
    numerical_source_center_available: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    preparation_recipe: str = "z_saddle + epsilon*(3*X_lambda, -v_lambda)"
    source_center_recipe: str = (
        "time_reversal_of_the_same_orbit_first_outer_exit_at_u=-3/2"
    )
    checkpoint_recipe: str = (
        "one_half_scaled_time_before_the_same_reversed_orbit_first_u=-5/2_hit"
    )
    source_radius_recipe: str = "source_radius_prefactor*exp(-source_radius_exponent)"
    scope: tuple[str, ...] = (
        "admitted_source_anchors_support_and_law_not_the_hypothetical_preparation",
        "one_correlated_algebraic_direction_not_an_independent_choice_from_its_intervals",
        "local_forward_and_backward_bounds_apply_to_one_exact_complete_orbit",
        "opposite_actual_local_endpoints_meet_outer_and_inner_nonlinear_corridor_conditions",
        "reversal_connects_the_outer_zero_winding_endpoint_to_the_inner_acute_band",
        "target_box_covers_all_twenty_form_and_phase_coordinates_at_unit_held_capacity",
        "positive_independent_source_radius_is_symbolic_never_float_underflow_to_zero",
        "event_defined_centers_and_times_are_not_numerically_materialized",
        "no_trajectory_forecast_autonomous_preparation_generic_attraction_or_physical_identification",
        "finite_retention_does_not_establish_permanent_identity_or_a_naturally_selected_law",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        # Delegate nested label admission to the actual evidence owners.
        self.saddle.to_dict()
        self.retention_band.to_dict()
        return {"schema": "tnfr.sine-saddle-formation.v1", "report": _project(self)}


def _local_connection_coefficients(momentum_lower, size, normalized_remainder):
    """Shared homogeneous local gates for the correlated recipe and its ball."""
    exponential_lower = sum(
        (Q(21, 25) ** k / divisor for k, divisor in enumerate((1, 1, 2, 6))), Q(0)
    )
    outer_displacement = (
        exponential_lower - 2 / exponential_lower - normalized_remainder
    )
    outer_momentum = (
        momentum_lower * (exponential_lower + 2 / exponential_lower)
        - 12 * normalized_remainder
    )
    inner_displacement = (
        1 / exponential_lower - 2 * exponential_lower + normalized_remainder
    )
    inner_momentum = (
        momentum_lower * (1 / exponential_lower + 2 * exponential_lower)
        - 12 * normalized_remainder
    )
    deficit = Q(3, 4) + Q(17, 12) * size
    allowance = 44 * (5 + deficit)
    return (
        outer_displacement,
        outer_momentum,
        inner_displacement,
        inner_momentum,
        deficit,
        allowance,
    )


def assess_sine_saddle_formation(
    source, *, cycle, epsilon=Q(1, 2**32)
) -> SineSaddleFormation:
    """Verify the finite same-orbit formation recipe and its open-source bridge.

    This static theorem reader performs no solver call. The preparation,
    algebraic saddle, two actual local endpoints and event-defined source
    refer to one mathematical orbit. Captured source membership or future
    motion is never inferred. Epsilon is an exact positive preparation size,
    at most2^-32; it is not a constitutive parameter of the nodal law.
    """
    from .relational_sine_resonance import assess_sine_c5_leaf_saddle

    source, _, indices, _, contacts = _private_leaf_support(source, cycle)
    cycle = tuple(source.nodes[i] for i in indices)
    size = exact_or_represented_real(epsilon, "epsilon")
    if not 0 < size <= Q(1, 2**32):
        raise ValueError("epsilon must lie in (0,2^-32]")
    saddle = assess_sine_c5_leaf_saddle(source, cycle=cycle)
    band = assess_sine_saddle_retention_band(source, cycle=cycle)
    count = len(source.nodes)
    representatives = (indices[0], indices[1], contacts[0][1], contacts[1][1])
    form_direction = tuple(saddle.unstable_direction_bounds[i] for i in representatives)
    phase_direction = tuple(
        saddle.unstable_direction_bounds[count + i] for i in representatives
    )
    quadratic = -sum(
        (
            phase_direction[i] * coefficient * phase_direction[j]
            for i, row in enumerate(saddle.odd_phase_hessian)
            for j, coefficient in enumerate(row)
        ),
        I(0),
    )
    momentum = sum(
        (weight * value for weight, value in zip((6, 3, 2, 1), form_direction)),
        I(0),
    )
    energy_low = 4 * quadratic.lo * size**2 - Q(40, 3) * size**3
    energy_high = 4 * quadratic.hi * size**2 + Q(40, 3) * size**3
    energy_ceiling = Q(7, 2) + 5 * size**2
    # Keep all epsilon comparisons homogeneous and exact. Adding epsilon to
    # an outward pi interval would lose arbitrarily small admissible sizes.
    remainder_coefficient = 9 * 729 * 728
    normalized_remainder = remainder_coefficient * size
    remainder = normalized_remainder * size
    (
        outer_displacement,
        outer_momentum,
        inner_displacement,
        inner_momentum,
        potential_deficit_coefficient,
        exit_allowance,
    ) = _local_connection_coefficients(momentum.lo, size, normalized_remainder)
    outer_margin = outer_momentum**2 - exit_allowance
    inner_margin = inner_momentum**2 - exit_allowance
    local_phase_error = 2187 * size
    pi = pi_interval()

    prepared = (
        saddle.status == "certified"
        and quadratic.lo > 1
        and quadratic.hi < Q(6, 5)
        and momentum.lo > Q(53, 10)
        and phase_direction[0] == I(1)
        and max(value.abs_max for value in phase_direction) <= 1
        and max(value.abs_max for value in form_direction) < 1
        and saddle.growth_rate_bounds.lo > Q(7, 25)
        and 0 < energy_low <= energy_high < 5 * size**2
        and energy_ceiling < band.full_storage_upper_bound
    )
    local = (
        prepared
        and normalized_remainder < Q(1, 512)
        and local_phase_error < pi.lo / 12
        and -2 * pi.hi / 3 - local_phase_error > band.deep_phase
        and -2 * pi.lo / 3 + local_phase_error < -pi.hi / 2
        and potential_deficit_coefficient < 1
    )
    outer = (
        local
        and outer_displacement > 1
        and outer_momentum > 0
        and outer_margin > 0
        and sqrt(I(3)).lo - 5 * size > 1
        and (-sin(I(Q(-3, 4))) / (2 * cos(I(Q(-3, 4))))).lo > Q(1, 3)
    )
    # V0' is strictly concave on the inner strip. Its right-face value at
    # u_saddle-epsilon is bounded below by(3/2-17epsilon/4)*epsilon;
    # the fixed left-face value is positive and larger than epsilon.
    left_force = 2 * (sin(2 * I(band.deep_phase)) + sin(I(band.deep_phase) / 2))
    inner = (
        local
        and inner_displacement < -1
        and inner_momentum > 0
        and inner_margin > 0
        and Q(3, 2) - Q(17, 4) * size > 1
        and left_force.lo > size
    )
    time_budget = 53 * energy_ceiling < 14**2 and 6 * size + 56 < 64
    formation = outer and inner and time_budget and band.theorem_certified
    source_radius_prefactor = Q(1, 2**19)
    radius_bridge = (
        source_radius_prefactor * 2 == band.target_form_error_bound
        and 2 * source_radius_prefactor < Q(1, 10)
        and pi.lo - 3 > Q(1, 10)
        and (2 * (1 + cos(I(Q(1, 10))))).lo > energy_ceiling
    )
    open_source = formation and radius_bridge
    reasons = tuple(
        reason
        for passed, reason in (
            (prepared, "correlated_preparation_energy_not_certified"),
            (local, "same_orbit_local_endpoints_not_certified"),
            (outer, "actual_outer_endpoint_corridor_gate_not_certified"),
            (inner, "actual_inner_endpoint_corridor_gate_not_certified"),
            (time_budget, "finite_connection_time_budget_not_certified"),
            (band.theorem_certified, "independent_target_retention_not_certified"),
            (radius_bridge, "symbolic_independent_source_radius_not_certified"),
        )
        if not passed
    )
    return SineSaddleFormation(
        source=source,
        saddle=saddle,
        retention_band=band,
        epsilon=size,
        phase_direction_quadratic_bounds=quadratic,
        momentum_direction_bounds=momentum,
        initial_storage_excess_lower_bound=energy_low,
        initial_storage_excess_upper_bound=energy_high,
        full_storage_upper_bound=energy_ceiling,
        local_remainder_bound=remainder,
        normalized_local_remainder_bound=normalized_remainder,
        outer_phase_displacement_coefficient_lower_bound=outer_displacement,
        outer_momentum_coefficient_lower_bound=outer_momentum,
        inner_phase_displacement_coefficient_upper_bound=inner_displacement,
        inner_negative_momentum_coefficient_lower_bound=inner_momentum,
        momentum_exit_allowance_coefficient_upper_bound=exit_allowance,
        outer_momentum_gate_margin=outer_margin,
        inner_momentum_gate_margin=inner_margin,
        local_phase_error_bound=local_phase_error,
        outer_force_lower_bound=size / 3,
        inner_force_lower_bound=size,
        connection_time_upper_bound=64 / size,
        source_radius_prefactor=source_radius_prefactor,
        source_radius_exponent=128 / size,
        initial_source_branch_margin_lower_bound=Q(1, 10),
        target_radius=band.target_form_error_bound,
        retained_scaled_duration=band.retained_scaled_duration,
        correlated_preparation_certified=prepared,
        local_connection_certified=local,
        outer_corridor_connection_certified=outer,
        inner_corridor_connection_certified=inner,
        formation_existence_certified=formation,
        independent_source_ball_existence_certified=open_source,
        captured_source_formation_certified=False,
        captured_source_retention_certified=False,
        numerical_source_center_available=False,
        status="certified" if open_source else "unavailable",
        reasons=reasons,
    )


@dataclass(frozen=True)
class SineSaddlePreparation:
    """An explicit rational near-saddle state with two local corridor gates.

    The captured source anchors support and law. ``prepared_state`` is the
    newly constructed state, with exact rational primitives and rates in the
    original clock. Only the gate bounds use scaled time tau=t/pi. The recipe
    intervals enclose one correlated algebraic state, not independent choices
    of its eigenvector entries. Their midpoint is compared to that state with
    an explicitly retained full-state error.
    """

    source: SineExchangeComparison
    formation: SineSaddleFormation
    prepared_state: SineExchangeComparison
    epsilon: Q
    form_recipe_bounds: tuple[I, ...]
    phase_recipe_bounds: tuple[I, ...]
    preparation_error_bound: Q
    local_endpoint_error_bound: Q
    normalized_local_endpoint_error_bound: Q
    full_storage_bounds: I
    storage_excess_bounds: I
    initial_cycle_gap_bounds: tuple[I, ...]
    initial_edge_turn_offsets: tuple[int, ...] | None
    initial_winding: int | None
    exact_signed_reconstruction_certified: bool
    initial_branches_certified: bool
    actual_energy_certified: bool
    local_phase_error_bound: Q
    outer_phase_displacement_coefficient_lower_bound: Q
    outer_momentum_coefficient_lower_bound: Q
    inner_phase_displacement_coefficient_upper_bound: Q
    inner_negative_momentum_coefficient_lower_bound: Q
    momentum_exit_allowance_coefficient_upper_bound: Q
    outer_momentum_gate_margin: Q
    inner_momentum_gate_margin: Q
    outer_corridor_gate_certified: bool
    inner_corridor_gate_certified: bool
    preparation_certified: bool
    numerical_zero_winding_source_available: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi; prepared_state rates retain original t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "captured_source_anchors_support_and_law_and_is_not_replaced_or_advanced",
        "exact_rational_midpoints_preserve_the_declared_signed_involution",
        "one_correlated_eigenvector_is_enclosed_not_independent_eigenvector_selection",
        "full_state_preparation_error_is_propagated_by_global_Lipschitz_two_to_both_tau_plus_minus_three",
        "actual_prepared_energy_is_rebuilt_from_primitive_full_edge_storage",
        "the_prepared_state_has_winding_one_not_a_numerical_winding_zero_source",
        "local_corridor_gates_are_analytic_bounds_not_evaluated_endpoint_states",
        "no_solver_event_time_trajectory_frozen_response_or_physical_identification",
        "unavailable_means_this_fixed_precision_cannot_certify_the_preparation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        _validate_comparison_labels(self.prepared_state)
        self.formation.to_dict()
        return {"schema": "tnfr.sine-saddle-preparation.v1", "report": _project(self)}


def prepare_sine_saddle_state(
    source, *, cycle, epsilon=Q(1, 2**32)
) -> SineSaddlePreparation:
    """Materialize and statically admit a rational near-saddle preparation.

    Epsilon must lie in (0,2^-32]. Smaller values are not silently rounded
    into a successful certificate: unresolved energy or phase margins return
    unavailable. No trajectory, winding-zero source or event is evaluated.
    """
    source, _, indices, pairs, contacts = _private_leaf_support(source, cycle)
    formation = assess_sine_saddle_formation(
        source, cycle=tuple(source.nodes[i] for i in indices), epsilon=epsilon
    )
    size, saddle = formation.epsilon, formation.saddle
    count, pi = len(source.nodes), pi_interval()
    form_bounds = tuple(
        3 * size * value for value in saddle.unstable_direction_bounds[:count]
    )
    phase_bounds = tuple(
        2 * turn * pi - size * value
        for turn, value in zip(
            saddle.target_phase_turns, saddle.unstable_direction_bounds[count:]
        )
    )
    forms, phases = [Q(0)] * count, [Q(0)] * count
    for block in (indices, tuple(leaf for _, leaf in contacts)):
        for left, right in ((block[0], block[4]), (block[1], block[3])):
            forms[left] = (form_bounds[left].lo + form_bounds[left].hi) / 2
            phases[left] = (phase_bounds[left].lo + phase_bounds[left].hi) / 2
            forms[right], phases[right] = -forms[left], -phases[left]
    forms, phases = tuple(forms), tuple(phases)
    preparation_error = max(
        max(abs(value - bound.lo), abs(value - bound.hi))
        for value, bound in zip(forms + phases, form_bounds + phase_bounds)
    )
    prepared = _comparison_from_state(
        _sine_state_from_rows(
            source.nodes,
            source.edges,
            forms,
            phases,
            source.capacity,
            _comparison_neighbors(source),
        ),
        source.reference_model,
    )
    family = _signed_reduction(prepared, contacts).source_trajectory_reduction_certified
    gaps = tuple(I(phases[j] - phases[i]) for i, j in pairs)
    offsets, _, winding, _ = _branches(gaps)
    branches = offsets == (0, 0, 0, 0, -1) and winding == 1
    endpoint_error = formation.local_remainder_bound + 729 * preparation_error
    normalized_error = endpoint_error / size
    (
        outer_displacement,
        outer_momentum,
        inner_displacement,
        inner_momentum,
        deficit,
        allowance,
    ) = _local_connection_coefficients(
        formation.momentum_direction_bounds.lo, size, normalized_error
    )
    outer_margin = outer_momentum**2 - allowance
    inner_margin = inner_momentum**2 - allowance
    phase_error = formation.local_phase_error_bound + 729 * preparation_error
    energy = (
        Q(7, 2) < prepared.storage.lo <= prepared.storage.hi < Q(7, 2) + 5 * size**2
    )
    local = (
        formation.correlated_preparation_certified
        and family
        and branches
        and energy
        and phase_error < pi.lo / 12
        and -2 * pi.hi / 3 - phase_error > formation.retention_band.deep_phase
        and -2 * pi.lo / 3 + phase_error < -pi.hi / 2
        and deficit < 1
    )
    outer = (
        local
        and formation.outer_corridor_connection_certified
        and outer_displacement > 1
        and outer_momentum > 0
        and outer_margin > 0
    )
    inner = (
        local
        and formation.inner_corridor_connection_certified
        and inner_displacement < -1
        and inner_momentum > 0
        and inner_margin > 0
    )
    reasons = tuple(
        reason
        for passed, reason in (
            (
                formation.correlated_preparation_certified,
                "correlated_recipe_not_certified",
            ),
            (family, "exact_signed_reconstruction_not_certified"),
            (branches, "prepared_winding_one_branches_not_certified"),
            (energy, "actual_nearcritical_energy_not_resolved"),
            (local, "local_full_state_phase_guards_not_certified"),
            (outer, "rational_preparation_outer_corridor_gate_not_certified"),
            (inner, "rational_preparation_inner_corridor_gate_not_certified"),
        )
        if not passed
    )
    return SineSaddlePreparation(
        source=source,
        formation=formation,
        prepared_state=prepared,
        epsilon=size,
        form_recipe_bounds=form_bounds,
        phase_recipe_bounds=phase_bounds,
        preparation_error_bound=preparation_error,
        local_endpoint_error_bound=endpoint_error,
        normalized_local_endpoint_error_bound=normalized_error,
        full_storage_bounds=prepared.storage,
        storage_excess_bounds=prepared.storage - Q(7, 2),
        initial_cycle_gap_bounds=gaps,
        initial_edge_turn_offsets=offsets,
        initial_winding=winding,
        exact_signed_reconstruction_certified=family,
        initial_branches_certified=branches,
        actual_energy_certified=energy,
        local_phase_error_bound=phase_error,
        outer_phase_displacement_coefficient_lower_bound=outer_displacement,
        outer_momentum_coefficient_lower_bound=outer_momentum,
        inner_phase_displacement_coefficient_upper_bound=inner_displacement,
        inner_negative_momentum_coefficient_lower_bound=inner_momentum,
        momentum_exit_allowance_coefficient_upper_bound=allowance,
        outer_momentum_gate_margin=outer_margin,
        inner_momentum_gate_margin=inner_margin,
        outer_corridor_gate_certified=outer,
        inner_corridor_gate_certified=inner,
        preparation_certified=outer and inner,
        numerical_zero_winding_source_available=False,
        status="certified" if outer and inner else "unavailable",
        reasons=reasons,
    )
