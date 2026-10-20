"""Detached five-C9 capture and source-free continuous retention admission.

ORGANIZED_SOURCE_CAPTURE.md owns the frozen source, acute-chart boundary
proof and disconnected control. No trajectory, support event or source
reset is performed, and no incoming report replaces primitive admission.
ORGANIZED_SOURCE_PORT.md owns the separate reflected hybrid domain and the
full-leaf raw-phase continuation. Both reuse the support barrier without
replaying the acquired source; their event and state premises differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from ._organized_source_port import (
    _ORIGINAL_CLOCK_FACTOR,
    _admit_organized_source_capacities,
    _assess_organized_source_attachment,
    _derive_organized_source_port,
    _observe_organized_source_capacity,
    _observe_organized_source_phase_state,
    _observe_organized_source_port,
    _observe_organized_source_raw_capacity,
    _OrganizedSourceAttachment,
    _OrganizedSourceCapacityObservation,
    _OrganizedSourceGeometry,
    _OrganizedSourceObservation,
    _OrganizedSourcePhaseState,
    _OrganizedSourceRawCapacityObservation,
    _OrganizedSourceReflectionState,
    _restrict_organized_source_reflection,
    _source_emission_geometry,
)
from ._repeated_organized_composition import _bounded_sine_edge_fields, _vector
from .relational_observations import _ordered

_TARGET_ORIGIN = Q(1, 40)
_RETENTION_CEILING = Q(1, 100)


@dataclass(frozen=True)
class _OrganizedSourceCaptureBarrier:
    """Complete acute-cell faces, without any source or entry verdict."""

    geometry: _OrganizedSourceGeometry
    target_form: tuple[Q, ...]
    target_phase_offsets: tuple[Q, ...]
    target_phase_lift_bounds: tuple[I, ...]
    target_storage_bounds: I
    cycle_storage_minimum_bounds: I
    positive_cycle_face_storage_bounds: I
    negative_cycle_face_storage_bounds: I
    cycle_boundary_excess_bounds: I
    matching_boundary_excess: Q
    boundary_energy_lower_bound: Q
    cell_edge_gap_bounds: I
    target_certified: bool
    barrier_certified: bool
    target_windings: tuple[int, ...] = (1,) * 5
    signed_internal_boundary_faces: int = 90
    signed_matching_boundary_faces: int = 90
    scope: tuple[str, ...] = (
        "all_ninety_signed_internal_and_ninety_signed_matching_faces",
        "fixed_internal_cycle_sums_two_pi_on_the_continuous_lift_chart",
        "five_disjoint_cycle_minima_plus_nonnegative_matching_storage",
        "forms_remain_unrestricted_until_the_full_storage_sublevel_is_applied",
        "target_and_boundary_geometry_alone_do_not_admit_the_actual_source",
    )


def _derive_organized_source_capture_barrier():
    """Rebuild the actual fine support and its complete Jensen boundary."""
    geometry = _derive_organized_source_port()
    pi, alpha = pi_interval(), geometry.alpha_bounds
    cycle = 9 * (1 - cos(alpha))
    positive = 1 + 8 * (1 - cos(3 * pi / 16))
    negative = 1 + 8 * (1 - cos(5 * pi / 16))
    excess = positive - cycle
    # On a marked cycle each oriented target current is sin(alpha).
    # Matching target currents vanish. Check its exact node divergence.
    divergence = [Q(0)] * 45
    compatible = True
    for a, b, weight in geometry.weighted_edges:
        if a // 9 == b // 9 and (b - a) % 9 == 1:
            divergence[a] += weight
            divergence[b] -= weight
        elif a // 9 != b // 9 and a % 9 == b % 9:
            continue
        else:
            compatible = False
    target_ok = (
        compatible and not any(divergence) and 0 < alpha.lo < alpha.hi < pi.lo / 2
    )
    barrier_ok = target_ok and negative.lo > positive.hi and excess.lo > 0
    return _OrganizedSourceCaptureBarrier(
        geometry=geometry,
        target_form=(Q(0),) * 45,
        target_phase_offsets=(_TARGET_ORIGIN,) * 45,
        target_phase_lift_bounds=tuple(
            alpha * (node % 9 - 4) + _TARGET_ORIGIN for node in range(45)
        ),
        target_storage_bounds=5 * cycle,
        cycle_storage_minimum_bounds=cycle,
        positive_cycle_face_storage_bounds=positive,
        negative_cycle_face_storage_bounds=negative,
        cycle_boundary_excess_bounds=excess,
        matching_boundary_excess=Q(1),
        boundary_energy_lower_bound=min(excess.lo, Q(1)),
        cell_edge_gap_bounds=pi / 2,
        target_certified=target_ok,
        barrier_certified=barrier_ok,
    )


@dataclass(frozen=True)
class _OrganizedSourceRawRetention:
    """Conditional full-leaf maintenance at raw phases with declared lift turns."""

    observation: _OrganizedSourcePhaseState
    barrier: _OrganizedSourceCaptureBarrier
    phase_turns: tuple[int, ...]
    raw_phase_sum: Q
    phase_turn_sum: int
    phase_lift_bounds: tuple[I, ...]
    lifted_phase_mean_bounds: I
    edge_angle_bounds: tuple[I, ...]
    energy_ceiling: Q
    excess_bounds: I
    spectral_gap_lower_bound: Q | None
    geometry_certified: bool
    in_acute_cell: bool
    below_energy_ceiling: bool
    domain_boundary_margin: Q
    admitted: bool
    form_squared_upper_bound: Q | None
    form_absolute_upper_bound: Q | None
    retained_windings: tuple[int, ...] | None
    asymptotic_phase_lift_bounds: tuple[I, ...] | None
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: tuple[str, ...] = (
        "all_raw_coordinates_and_ordinary_integer_turns_admitted_before_geometry",
        "exact_zero_total_form_required_without_reflection_or_projection",
        "fixed_full_support_unit_capacities_and_complete_law_rebuilt",
        "external_runtime_support_capacity_clock_and_state_association_is_caller_owned",
        "edge_turn_differences_cancel_before_interval_construction",
        "raw_phase_sum_and_integer_turn_sum_retain_the_new_lifted_mean",
        "source_free_boundary_and_existing_full_graph_spectral_gap_reused",
        "barrier_template_origin_is_only_a_translated_energy_and_face_reference",
        "conditional_exact_continuous_flow_without_future_events_or_modulo_resets",
        "convergence_uses_the_admitted_endpoint_mean_not_the_old_source_origin",
        "no_reflection_childwise_mean_or_hybrid_event_passivity_claim",
        "no_trajectory_source_replay_grammar_occurrence_or_physical_identity_claim",
    )


def _assess_organized_source_raw_retention(*, form, phases, phase_turns):
    """Admit an exact continuous continuation from raw phase representatives.

    The function fixes the existing ninety-edge, unit-capacity source law and
    all three clock conventions. Callers separately bind these premises to an
    external runtime state. Every lift is ``phases[i] + 2*pi*phase_turns[i]``;
    turns are supplied ordinary integers, never inferred by wrapping. Future
    flow keeps this continuous lift rather than repeating a modulo operation.

    Invalid primitives, malformed turns and nonzero total form reject before
    geometry or observation. Finite states outside the strict acute cell or
    energy sublevel return unavailable continuation bounds. Inside the cell,
    Jensen's inequality bounds form storage by the excess. The existing full
    graph gap bounds squared forms by twice that excess divided by the gap.
    Exact loss and the source-free boundary then preserve the cell and rails;
    the fixed-mean convergence argument supplies the new asymptotic phase lift.
    No reflected identity, event passivity or future event admission follows.
    """
    values = _vector(form, "form", 45)
    angles = _vector(phases, "phases", 45)
    turns = _admit_organized_source_turns(phase_turns)
    if sum(values, Q(0)) != 0:
        raise ValueError("raw retention requires exactly zero total form")

    observation = _observe_organized_source_phase_state(form=values, phases=angles)
    chart = _organized_source_raw_chart(observation, turns)
    raw_sum, turn_sum = sum(angles, Q(0)), sum(turns)
    mean = (I(raw_sum) + 2 * pi_interval() * turn_sum) / 45
    admitted, gap = chart["admitted"], chart["spectral_gap_lower_bound"]
    return _OrganizedSourceRawRetention(
        observation=observation,
        phase_turns=turns,
        raw_phase_sum=raw_sum,
        phase_turn_sum=turn_sum,
        lifted_phase_mean_bounds=mean,
        **chart,
        form_squared_upper_bound=2 * _RETENTION_CEILING / gap if admitted else None,
        form_absolute_upper_bound=Q(3, 10) if admitted else None,
        retained_windings=(1,) * 5 if admitted else None,
        asymptotic_phase_lift_bounds=(
            tuple(
                observation.geometry.alpha_bounds * (i % 9 - 4) + mean
                for i in range(45)
            )
            if admitted
            else None
        ),
        status="certified_conditional_raw_retention" if admitted else "unavailable",
    )


def _admit_organized_source_turns(phase_turns):
    turns = _ordered(phase_turns, "phase_turns", limit=46)
    if len(turns) != 45 or any(type(turn) is not int for turn in turns):
        raise ValueError("phase_turns requires 45 ordinary integer coordinates")
    return turns


def _organized_source_raw_chart(observation, turns):
    """Share fixed-support lift, barrier and spectral admission from fresh state."""
    angles = observation.phases
    barrier = _derive_organized_source_capture_barrier()
    geometry = observation.geometry
    geometry_ok, _diameter, cycle_gap, child_gap, _degrees = _source_emission_geometry(
        geometry
    )
    geometry_ok = geometry_ok and geometry == barrier.geometry
    gap = min(cycle_gap, child_gap) if geometry_ok else None
    pi = pi_interval()
    edge_angles = tuple(
        I(angles[b] - angles[a])
        + 2
        * pi
        * (turns[b] - turns[a] + int(a // 9 == b // 9 and a % 9 == 8 and b % 9 == 0))
        for a, b, _weight in geometry.weighted_edges
    )
    lifted = tuple(I(angle) + 2 * pi * turn for angle, turn in zip(angles, turns))
    in_cell = all(angle.abs_max < pi.lo / 2 for angle in edge_angles)
    excess = observation.joint_storage_bounds - barrier.target_storage_bounds
    ceiling = _RETENTION_CEILING
    below = excess.hi < ceiling
    margin = barrier.boundary_energy_lower_bound - ceiling
    unresolved = tuple(
        label
        for passed, label in (
            (geometry_ok, "fixed_full_support_capacity_clock_or_gap_unavailable"),
            (barrier.barrier_certified, "source_free_acute_boundary_unavailable"),
            (in_cell, "declared_lift_not_certified_in_fixed_branch_acute_cell"),
            (below, "state_excess_not_strictly_below_fixed_ceiling"),
            (margin > 0, "fixed_ceiling_not_below_acute_boundary"),
        )
        if not passed
    )
    admitted = not unresolved
    return dict(
        barrier=barrier,
        phase_lift_bounds=lifted,
        edge_angle_bounds=edge_angles,
        energy_ceiling=ceiling,
        excess_bounds=excess,
        spectral_gap_lower_bound=gap,
        geometry_certified=geometry_ok,
        in_acute_cell=in_cell,
        below_energy_ceiling=below,
        domain_boundary_margin=margin,
        admitted=admitted,
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _OrganizedSourceRawCapacityRetention:
    """Positive shared-capacity continuation on its exact weighted form leaf."""

    observation: _OrganizedSourceRawCapacityObservation
    barrier: _OrganizedSourceCaptureBarrier
    phase_turns: tuple[int, ...]
    inverse_capacity_weights: tuple[Q, ...]
    inverse_capacity_weight_sum: Q
    weighted_raw_phase_sum: Q
    weighted_phase_turn_sum: Q
    weighted_carrier_slot_sum: Q
    phase_lift_bounds: tuple[I, ...]
    weighted_phase_origin_bounds: I
    edge_angle_bounds: tuple[I, ...]
    energy_ceiling: Q
    excess_bounds: I
    spectral_gap_lower_bound: Q | None
    capacity_ratio: Q
    geometry_certified: bool
    in_acute_cell: bool
    below_energy_ceiling: bool
    domain_boundary_margin: Q
    geometric_retention_certified: bool
    form_rails_certified: bool
    admitted: bool
    form_squared_upper_bound: Q | None
    form_absolute_upper_bound: Q | None
    retained_windings: tuple[int, ...] | None
    converges_to_target: bool | None
    asymptotic_phase_lift_bounds: tuple[I, ...] | None
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: tuple[str, ...] = (
        "strictly_positive_held_capacities_with_explicit_shared_phase_mobility",
        "all_primitives_and_exact_zero_inverse_capacity_form_total_readmitted",
        "raw_phase_lifts_use_supplied_ordinary_integer_turns_without_projection",
        "weighted_target_origin_subtracts_the_nonzero_weighted_carrier_mean",
        "capacity_ratio_scales_the_full_support_spectral_form_bound",
        "geometric_retention_and_convergence_do_not_imply_form_rails_for_any_ratio",
        "conditional_exact_continuous_flow_without_further_events_or_modulo_resets",
        "no_zero_capacity_limit_event_execution_occurrence_or_source_claim",
    )


def _assess_organized_source_raw_capacity_retention(
    *, form, phases, phase_turns, capacities
):
    """Certify the declared positive shared-capacity law from primitive state.

    The phase row shares each node's held capacity; that premise is additional
    to the nodal form identity. The invariant form leaf is exactly sum(x/nu)=0.
    At fixed positive capacities, the weighted phase-origin invariant selects
    the translated winding-one equilibrium. A ratio-dependent sufficient rail
    bound is checked separately from geometric retention and convergence.
    """
    values = _vector(form, "form", 45)
    angles = _vector(phases, "phases", 45)
    turns = _admit_organized_source_turns(phase_turns)
    mobilities = _admit_organized_source_capacities(capacities, "shared_capacity")
    if any(value <= 0 for value in mobilities):
        raise ValueError("raw weighted retention requires strictly positive capacities")
    weights = tuple(1 / nu for nu in mobilities)
    if sum((w * x for w, x in zip(weights, values)), Q(0)) != 0:
        raise ValueError(
            "raw weighted retention requires exactly zero sum(form/capacity)"
        )
    observation = _observe_organized_source_raw_capacity(
        form=values,
        phases=angles,
        capacities=mobilities,
        phase_mobility="shared_capacity",
    )
    reference = observation.unit_capacity_reference
    chart = _organized_source_raw_chart(reference, turns)
    geometric = chart.pop("admitted")
    unresolved = chart.pop("unresolved_conditions")
    phase_fields = _organized_source_weighted_phase_fields(angles, turns, weights)
    origin = phase_fields["weighted_phase_origin_bounds"]
    ratio = max(mobilities) / min(mobilities)
    bound = (
        ratio * 2 * _RETENTION_CEILING / chart["spectral_gap_lower_bound"]
        if geometric
        else None
    )
    rails = bound is not None and bound < 1
    if geometric and not rails:
        unresolved += ("capacity_ratio_does_not_certify_unit_form_rails",)
    admitted = geometric and rails
    return _OrganizedSourceRawCapacityRetention(
        observation=observation,
        phase_turns=turns,
        inverse_capacity_weights=weights,
        **phase_fields,
        **chart,
        capacity_ratio=ratio,
        geometric_retention_certified=geometric,
        form_rails_certified=rails,
        admitted=admitted,
        form_squared_upper_bound=bound,
        form_absolute_upper_bound=sqrt(I(bound)).hi if geometric else None,
        retained_windings=(1,) * 5 if geometric else None,
        converges_to_target=True if geometric else None,
        asymptotic_phase_lift_bounds=(
            tuple(
                reference.geometry.alpha_bounds * (i % 9 - 4) + origin
                for i in range(45)
            )
            if geometric
            else None
        ),
        status=(
            "certified_conditional_raw_shared_capacity_retention"
            if admitted
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )


def _organized_source_weighted_phase_fields(angles, turns, weights):
    """Keep the lifted invariant and carrier subtraction in exact affine form."""
    weight_sum = sum(weights, Q(0))
    raw_sum = sum((w * angle for w, angle in zip(weights, angles)), Q(0))
    turn_sum = sum((w * turn for w, turn in zip(weights, turns)), Q(0))
    slot_sum = sum((w * (i % 9 - 4) for i, w in enumerate(weights)), Q(0))
    # Cancel affine pi coefficients exactly before interval construction.
    origin = (
        I(raw_sum) + pi_interval() * (2 * turn_sum - 2 * slot_sum / 9)
    ) / weight_sum
    return dict(
        inverse_capacity_weight_sum=weight_sum,
        weighted_raw_phase_sum=raw_sum,
        weighted_phase_turn_sum=turn_sum,
        weighted_carrier_slot_sum=slot_sum,
        weighted_phase_origin_bounds=origin,
    )


@dataclass(frozen=True)
class _OrganizedSourceWeightedMeanRetention:
    """Conditional continuation on an arbitrary positive-capacity mean leaf."""

    observation: _OrganizedSourceRawCapacityObservation
    barrier: _OrganizedSourceCaptureBarrier
    phase_turns: tuple[int, ...]
    inverse_capacity_weights: tuple[Q, ...]
    inverse_capacity_weight_sum: Q
    weighted_form_total: Q
    weighted_form_mean: Q
    weighted_raw_phase_sum: Q
    weighted_phase_turn_sum: Q
    weighted_carrier_slot_sum: Q
    phase_lift_bounds: tuple[I, ...]
    weighted_phase_origin_bounds: I
    edge_angle_bounds: tuple[I, ...]
    energy_ceiling: Q
    excess_bounds: I
    spectral_gap_lower_bound: Q | None
    capacity_ratio: Q
    geometry_certified: bool
    in_acute_cell: bool
    below_energy_ceiling: bool
    domain_boundary_margin: Q
    geometric_retention_certified: bool
    form_rails_certified: bool
    admitted: bool
    centered_form_squared_upper_bound: Q | None
    centered_form_absolute_upper_bound: Q | None
    form_absolute_upper_bound: Q | None
    retained_windings: tuple[int, ...] | None
    converges_to_target: bool | None
    asymptotic_form: tuple[Q, ...] | None
    asymptotic_phase_lift_bounds: tuple[I, ...] | None
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: tuple[str, ...] = (
        "all_primitives_and_strictly_positive_held_capacities_admitted_first",
        "shared_capacity_phase_mobility_is_an_explicit_complete_law_premise",
        "raw_form_is_observed_without_centering_projection_or_mean_restriction",
        "exact_inverse_capacity_form_mean_is_conserved_on_the_continuous_segment",
        "centered_norm_bound_uses_the_full_support_gap_and_initial_excess_upper_bound",
        "unit_form_rails_require_the_mean_plus_centered_radius_to_be_less_than_one",
        "geometric_retention_and_convergence_are_separate_from_the_unit_rail_test",
        "weighted_phase_origin_subtracts_the_carrier_on_the_declared_fixed_lift",
        "conditional_exact_continuous_flow_without_future_events_or_modulo_resets",
        "no_zero_capacity_limit_event_execution_occurrence_or_source_claim",
    )


def _assess_organized_source_weighted_mean_retention(
    *, form, phases, phase_turns, capacities
):
    """Certify the fixed reciprocal law without imposing a zero form mean.

    At positive held capacities the conserved mean is
    ``m=sum(x_i/nu_i)/sum(1/nu_i)``. For ``z=x-m`` the same weighted least
    squares argument and full-support spectral gap give
    ``||z||^2 <= 2*rho*E_initial/gap`` throughout the trapped sublevel.
    Observation retains the supplied x; centering is only a proof coordinate.
    Convergence is to the constant form m and the carrier translated by the
    conserved weighted phase-minus-carrier mean. Neither a nonzero m nor
    failure of the sufficient unit-rail bound defeats geometric retention.

    This additive certificate leaves the old weighted-zero API and its fixed
    ceiling-based bounds intact. Here the actual outward initial excess upper
    endpoint bounds the centered norm. Its nonnegativity is checked instead
    of silently clamping an inconsistent bound before taking its square root.
    """
    values = _vector(form, "form", 45)
    angles = _vector(phases, "phases", 45)
    turns = _admit_organized_source_turns(phase_turns)
    mobilities = _admit_organized_source_capacities(capacities, "shared_capacity")
    if any(value <= 0 for value in mobilities):
        raise ValueError(
            "weighted mean retention requires strictly positive capacities"
        )
    weights = tuple(1 / nu for nu in mobilities)
    total = sum((w * x for w, x in zip(weights, values)), Q(0))
    mean = total / sum(weights, Q(0))
    observation = _observe_organized_source_raw_capacity(
        form=values,
        phases=angles,
        capacities=mobilities,
        phase_mobility="shared_capacity",
    )
    reference = observation.unit_capacity_reference
    chart = _organized_source_raw_chart(reference, turns)
    geometric = chart.pop("admitted")
    unresolved = chart.pop("unresolved_conditions")
    phase_fields = _organized_source_weighted_phase_fields(angles, turns, weights)
    origin = phase_fields["weighted_phase_origin_bounds"]
    ratio = max(mobilities) / min(mobilities)
    bound = radius = absolute = None
    if geometric:
        if chart["excess_bounds"].hi < 0:
            raise ValueError("acute-cell excess upper bound cannot be negative")
        bound = (
            ratio * 2 * chart["excess_bounds"].hi / chart["spectral_gap_lower_bound"]
        )
        radius = sqrt(I(bound)).hi
        absolute = abs(mean) + radius
    rails = absolute is not None and absolute < 1
    if geometric and not rails:
        unresolved += ("weighted_mean_and_excess_do_not_certify_unit_form_rails",)
    admitted = geometric and rails
    return _OrganizedSourceWeightedMeanRetention(
        observation=observation,
        phase_turns=turns,
        inverse_capacity_weights=weights,
        weighted_form_total=total,
        weighted_form_mean=mean,
        **phase_fields,
        **chart,
        capacity_ratio=ratio,
        geometric_retention_certified=geometric,
        form_rails_certified=rails,
        admitted=admitted,
        centered_form_squared_upper_bound=bound,
        centered_form_absolute_upper_bound=radius,
        form_absolute_upper_bound=absolute,
        retained_windings=(1,) * 5 if geometric else None,
        converges_to_target=True if geometric else None,
        asymptotic_form=(mean,) * 45 if geometric else None,
        asymptotic_phase_lift_bounds=(
            tuple(
                reference.geometry.alpha_bounds * (i % 9 - 4) + origin
                for i in range(45)
            )
            if geometric
            else None
        ),
        status=(
            "certified_conditional_weighted_mean_retention"
            if admitted
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _OrganizedSourceReflectionRetention:
    """Conditional acute-cell and form-rail retention from fresh primitive state."""

    state: _OrganizedSourceReflectionState
    observation: _OrganizedSourceObservation
    barrier: _OrganizedSourceCaptureBarrier
    energy_ceiling: Q
    excess_bounds: I
    edge_angle_bounds: tuple[I, ...]
    in_acute_cell: bool
    below_energy_ceiling: bool
    domain_boundary_margin: Q
    admitted: bool
    form_squared_upper_bound: Q | None
    form_absolute_upper_bound: Q | None
    retained_windings: tuple[int, ...] | None
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: tuple[str, ...] = (
        "raw_state_strictly_restricted_without_wrapping_or_projection",
        "source_free_full_support_barrier_and_fine_storage_rebuilt",
        "fixed_origin_reflection_and_strict_acute_cell_and_storage_sublevel",
        "fixed_slot_paths_bound_forms_before_event_passivity_is_invoked",
        "same_law_continuous_flow_and_finite_independently_admitted_ideal_pair_EN",
        "future_policy_grammar_and_event_occurrence_remain_independent",
        "represented_events_require_separate_sector_held_field_and_work_checks",
        "no_acquired_source_replay_formation_or_physical_identity_claim",
    )


def _assess_organized_source_reflection_retention(*, form, phase_offsets):
    """Admit one state to the fixed joint winding-and-form-rail domain.

    Invalid primitives or a state outside the exact reflection chart reject.
    Valid chart states failing the acute-cell or fixed energy ceiling return
    unavailable evidence. A small or negative excess alone cannot establish
    winding identity outside the declared continuous-lift cell.
    """
    state = _restrict_organized_source_reflection(
        form=form, phase_offsets=phase_offsets
    )
    barrier = _derive_organized_source_capture_barrier()
    observation = _observe_organized_source_port(
        form=state.form, phase_offsets=state.phase_offsets
    )
    geometry = barrier.geometry
    angles = tuple(
        (
            geometry.alpha_bounds + state.phase_offsets[b] - state.phase_offsets[a]
            if a // 9 == b // 9
            else I(state.phase_offsets[b] - state.phase_offsets[a])
        )
        for a, b, _ in geometry.weighted_edges
    )
    in_cell = all(angle.abs_max < barrier.cell_edge_gap_bounds.lo for angle in angles)
    excess = observation.joint_storage_bounds - barrier.target_storage_bounds
    ceiling = _RETENTION_CEILING
    below = excess.hi < ceiling
    margin = barrier.boundary_energy_lower_bound - ceiling
    unresolved = tuple(
        label
        for passed, label in (
            (barrier.barrier_certified, "source_free_acute_boundary_unavailable"),
            (in_cell, "state_not_certified_in_fixed_branch_acute_cell"),
            (below, "state_excess_not_strictly_below_fixed_ceiling"),
            (margin > 0, "fixed_ceiling_not_below_acute_boundary"),
        )
        if not passed
    )
    admitted = not unresolved
    return _OrganizedSourceReflectionRetention(
        state=state,
        observation=observation,
        barrier=barrier,
        energy_ceiling=ceiling,
        excess_bounds=excess,
        edge_angle_bounds=angles,
        in_acute_cell=in_cell,
        below_energy_ceiling=below,
        domain_boundary_margin=margin,
        admitted=admitted,
        form_squared_upper_bound=8 * ceiling if admitted else None,
        form_absolute_upper_bound=Q(3, 10) if admitted else None,
        retained_windings=(1,) * 5 if admitted else None,
        status=(
            "certified_conditional_reflection_retention" if admitted else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _OrganizedSourceReflectionCoherence:
    """Conditional ideal phase action, distinct from runtime IL admission."""

    domain: _OrganizedSourceReflectionRetention
    phase_locking_coefficient: Q
    resultants_real_bounds: tuple[I, I] | None = None
    resultants_imaginary_bounds: tuple[I, I] | None = None
    phase_error_bounds: tuple[I, I] | None = None
    phase_displacement_bounds: tuple[I, I] | None = None
    phase_offset_after_bounds: tuple[I, ...] | None = None
    storage_jump_bounds: I | None = None
    isolated_storage_jump_bounds: tuple[I, I] | None = None
    shared_edge_correction_bounds: I | None = None
    work_decomposition_residual_bounds: I | None = None
    phase_current_jump_bounds: tuple[I, ...] | None = None
    form_tau_rate_jump_bounds: tuple[I, ...] | None = None
    phase_tau_rate_jump_bounds: tuple[I, ...] | None = None
    form_slow_rate_jump_bounds: tuple[I, ...] | None = None
    phase_slow_rate_jump_bounds: tuple[I, ...] | None = None
    form_original_rate_jump_bounds: tuple[I, ...] | None = None
    phase_original_rate_jump_bounds: tuple[I, ...] | None = None
    ideal_retention_certified: bool = False
    retained_windings: tuple[int, ...] | None = None
    storage_jump_upper_bound: Q | None = None
    strict_work_decrease_certified: bool = False
    status: str = "unavailable"
    unresolved_conditions: tuple[str, ...] = ()
    scope: tuple[str, ...] = (
        "raw_fine_state_and_supplied_coefficient_readmitted_without_cached_reports",
        "ideal_common_snapshot_phase_action_on_the_fixed_pair_zero_one",
        "exact_reflection_implies_correlated_opposite_phase_displacements",
        "transcendental_endpoint_offsets_retained_as_outward_intervals_not_midpoints",
        "joint_touched_edge_work_includes_the_shared_edge_correction",
        "algebraic_nonincrease_bound_is_separate_from_numeric_work_enclosure",
        "fresh_declared_pressure_not_the_operator_stored_pressure_contraction",
        "held_forms_unit_capacities_support_and_clocks_without_elapsed_flow",
        "no_runtime_admission_grammar_word_completion_or_event_occurrence_claim",
        "represented_endpoints_require_separate_lift_sector_and_work_checks",
    )


def _assess_organized_source_reflection_coherence(
    *, form, phase_offsets, phase_locking_coefficient
):
    """Assess one ideal mirrored Coherence phase map on the maintained domain.

    The caller supplies the phase coefficient; this observer installs no
    runtime defaults or operator. Strict reflection and the acute storage
    domain are rebuilt from all raw fine coordinates before any endpoint is
    enclosed. Invalid primitives reject; valid states outside that domain
    return unavailable evidence without endpoint or future bounds.

    On the admitted domain the two relative resultants are conjugate, have
    real part above four, and their arguments have magnitude below 1/8.
    The joint work theorem therefore bounds every snapshot reset prefix by
    zero and transfers the existing barrier. Correlated opposite arguments
    express that exact identity, not a projection of a represented endpoint.
    """
    coefficient = _exact(phase_locking_coefficient, "phase_locking_coefficient")
    if not 0 <= coefficient <= 1:
        raise ValueError("phase_locking_coefficient must lie in [0, 1]")
    domain = _assess_organized_source_reflection_retention(
        form=form, phase_offsets=phase_offsets
    )
    if not domain.admitted:
        return _OrganizedSourceReflectionCoherence(
            domain=domain,
            phase_locking_coefficient=coefficient,
            unresolved_conditions=domain.unresolved_conditions,
        )

    fine = domain.observation
    geometry, offsets = fine.geometry, domain.state.phase_offsets
    edges, alpha = geometry.weighted_edges, geometry.alpha_bounds

    def gap(a, b):
        return alpha * ((b % 9) - (a % 9)) + offsets[b] - offsets[a]

    neighbors = tuple(
        tuple(b if a == target else a for a, b, _ in edges if target in (a, b))
        for target in (0, 1)
    )
    real = tuple(
        sum((cos(gap(target, neighbor)) for neighbor in row), I(0))
        for target, row in enumerate(neighbors)
    )
    imaginary = (fine.phase_current_bounds[0], fine.phase_current_bounds[1])
    if any(value.lo <= 0 for value in real):
        return _OrganizedSourceReflectionCoherence(
            domain=domain,
            phase_locking_coefficient=coefficient,
            resultants_real_bounds=real,
            resultants_imaginary_bounds=imaginary,
            unresolved_conditions=("positive_relative_resultants_not_certified",),
        )
    angle = arg(real[0], imaginary[0])
    errors = (angle, -angle)
    zero_row = (I(0),) * 45
    if coefficient == 0:
        displacement = (I(0), I(0))
        endpoint = tuple(I(value) for value in offsets)
        work = correction = residual = I(0)
        isolated = (I(0), I(0))
        current_jump = form_jump = zero_row
    else:
        left = coefficient * angle
        displacement = (left, -left)
        changes = displacement + (I(0),) * 43
        endpoint = tuple(I(value) + change for value, change in zip(offsets, changes))
        after_current, _potential, _edge_currents = _bounded_sine_edge_fields(
            phase_offsets=endpoint,
            weighted_edges=edges,
            alpha_bounds=alpha,
        )
        current_jump = tuple(
            after - before
            for before, after in zip(fine.phase_current_bounds, after_current)
        )
        form_jump = tuple(
            geometry.clock_factor * geometry.gamma_bounds * change
            for change in current_jump
        )
        work = sum(
            (
                weight * (cos(gap(a, b)) - cos(gap(a, b) + changes[b] - changes[a]))
                for a, b, weight in edges
                if a in (0, 1) or b in (0, 1)
            ),
            I(0),
        )
        isolated = tuple(
            sum(
                (
                    cos(gap(target, neighbor))
                    - cos(gap(target, neighbor) - displacement[target])
                    for neighbor in row
                ),
                I(0),
            )
            for target, row in enumerate(neighbors)
        )
        left, right = displacement
        correction = (
            -4 * cos(gap(0, 1) + (right - left) / 2) * sin(left / 2) * sin(right / 2)
        )
        residual = work - sum(isolated, I(0)) - correction
    slow = 2 / geometry.gamma_bounds**2
    return _OrganizedSourceReflectionCoherence(
        domain=domain,
        phase_locking_coefficient=coefficient,
        resultants_real_bounds=real,
        resultants_imaginary_bounds=imaginary,
        phase_error_bounds=errors,
        phase_displacement_bounds=displacement,
        phase_offset_after_bounds=endpoint,
        storage_jump_bounds=work,
        isolated_storage_jump_bounds=isolated,
        shared_edge_correction_bounds=correction,
        work_decomposition_residual_bounds=residual,
        phase_current_jump_bounds=current_jump,
        form_tau_rate_jump_bounds=form_jump,
        phase_tau_rate_jump_bounds=zero_row,
        form_slow_rate_jump_bounds=tuple(slow * value for value in form_jump),
        phase_slow_rate_jump_bounds=zero_row,
        form_original_rate_jump_bounds=tuple(
            _ORIGINAL_CLOCK_FACTOR * value for value in form_jump
        ),
        phase_original_rate_jump_bounds=zero_row,
        ideal_retention_certified=True,
        retained_windings=domain.retained_windings,
        storage_jump_upper_bound=Q(0),
        strict_work_decrease_certified=coefficient > 0 and not angle.contains(0),
        status="certified_conditional_reflection_coherence",
    )


@dataclass(frozen=True)
class _OrganizedSourceSharedCapacityRetention:
    """Same geometric domain under an explicitly supplied shared-capacity law."""

    pair_capacity: Q
    domain: _OrganizedSourceReflectionRetention
    continuation: _OrganizedSourceCapacityObservation
    retention_certified: bool
    retained_windings: tuple[int, ...] | None
    form_absolute_upper_bound: Q | None
    converges_to_target: bool | None
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: tuple[str, ...] = (
        "pair_zero_one_has_equal_held_capacity_other_capacities_are_unit",
        "declared_pressure_unchanged_shared_capacity_phase_law_is_an_added_premise",
        "raw_state_rebuilds_reflection_acute_cell_storage_and_both_actual_rows",
        "geometric_retention_includes_zero_capacity_without_inverse_capacity",
        "target_convergence_requires_strictly_positive_held_capacity",
        "capacity_only_jump_has_zero_storage_work_under_verified_held_fields",
        "no_silence_execution_grammar_completion_or_occurrence_claim",
    )


def _assess_organized_source_shared_capacity_retention(
    *, form, phase_offsets, pair_capacity
):
    """Transfer the fixed domain to an explicit shared-mobility continuation.

    The pair capacity is a normalized held value in [0, 1], not a request
    to execute Silence. Zero retains the geometric certificate when available,
    but does not inherit convergence to the unit model's target. This observer
    cannot authenticate runtime support, held fields or event provenance.
    """
    capacity = _exact(pair_capacity, "pair_capacity")
    if not 0 <= capacity <= 1:
        raise ValueError("pair_capacity must lie in [0, 1]")
    domain = _assess_organized_source_reflection_retention(
        form=form, phase_offsets=phase_offsets
    )
    continuation = _observe_organized_source_capacity(
        form=domain.state.form,
        phase_offsets=domain.state.phase_offsets,
        capacities=(capacity, capacity) + (Q(1),) * 43,
        phase_mobility="shared_capacity",
    )
    unresolved = domain.unresolved_conditions
    if capacity == 0:
        unresolved += ("strictly_positive_capacity_required_for_target_convergence",)
    return _OrganizedSourceSharedCapacityRetention(
        pair_capacity=capacity,
        domain=domain,
        continuation=continuation,
        retention_certified=domain.admitted,
        retained_windings=domain.retained_windings,
        form_absolute_upper_bound=domain.form_absolute_upper_bound,
        converges_to_target=True if domain.admitted and capacity > 0 else None,
        status=(
            "certified_conditional_shared_capacity_retention"
            if domain.admitted
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _OrganizedSourceDisconnectedControl:
    """Same initialized source; separate conserved means forbid alignment."""

    weighted_edges: tuple[tuple[int, int, Q], ...]
    components: tuple[tuple[int, ...], ...]
    component_form_means: tuple[Q, Q]
    component_phase_offset_means: tuple[Q, Q]
    common_target_phase_offset: Q
    target_component_mean_mismatches: tuple[Q, Q]
    donor_host_phase_difference: Q
    joined_alignment_excluded: bool
    clock_factor: Q = Q(1, 4)
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2"
    scope: tuple[str, ...] = (
        "omit_only_the_nine_donor_contacts_at_the_same_actual_initialized_state",
        "same_uniform_quarter_factor_in_both_rows_and_unchanged_source_clocks",
        "separate_component_means_are_not_connected_graph_invariants",
        "invariant_mean_obstruction_not_an_evaluated_disconnected_trajectory",
        "no_later_edge_removal_event_or_disconnected_capture_claim",
    )


@dataclass(frozen=True)
class _OrganizedSourceCapture:
    """Fresh source, exact residual allowances and conditional future flags."""

    source: _OrganizedSourceAttachment
    barrier: _OrganizedSourceCaptureBarrier
    control: _OrganizedSourceDisconnectedControl
    source_host_nominal_excess_upper_bound: Q
    source_host_residual_excess_upper_bound: Q
    source_host_excess_upper_bound: Q
    source_donor_excess_upper_bound: Q
    pre_attachment_excess_upper_bound: Q
    initial_excess_upper_bound: Q
    capture_margin: Q
    initial_internal_phase_gap_bounds: I
    initial_host_matching_gap_upper_bound: Q
    initial_contact_phase_gap_bounds: I
    initial_donor_host_phase_difference: Q
    asymptotic_donor_host_phase_difference: Q | None
    source_certified: bool
    conditional_source_in_cell: bool
    source_in_cell: bool
    barrier_certified: bool
    capture_certified: bool
    winding_retained: bool
    converges_to_target: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    protocol_revision: str = "e7e7ce6e"
    scope: tuple[str, ...] = (
        "all_ten_original_primitives_admitted_before_fresh_source_rebuilding",
        "all_eighty_independent_relative_preparation_directions_retained",
        "actual_host_at_s_four_and_independent_clock_transformed_recovered_donor",
        "reached_host_storage_rebuilt_from_gap_and_full_fine_error_bounds",
        "exact_tiny_residual_allowances_retained_separately_from_nominal_bounds",
        "positive_attachment_work_added_without_crediting_previous_loss",
        "complete_ninety_coordinate_flow_on_the_connected_conserved_leaf",
        "all_future_five_child_winding_retention_and_asymptotic_joined_capture",
        "no_all_time_parent_registration_or_finite_recovery_deadline_claim",
        "failed_sufficient_certificate_is_unavailable_not_proved_identity_loss",
        "detached_proof_no_trajectory_or_autonomous_event_selection",
    )


def _assess_organized_source_capture(
    *,
    host_form_error_bounds,
    host_phase_error_bounds,
    donor_form_error_bound,
    donor_phase_error_bound,
):
    """Re-admit all ten primitives before rebuilding the actual source."""
    host_rx = _vector(host_form_error_bounds, "host_form_error_bounds", 4)
    host_rt = _vector(host_phase_error_bounds, "host_phase_error_bounds", 4)
    donor_rx = _exact(donor_form_error_bound, "donor_form_error_bound")
    donor_rt = _exact(donor_phase_error_bound, "donor_phase_error_bound")
    if min(*host_rx, *host_rt, donor_rx, donor_rt) < 0:
        raise ValueError("original preparation error bounds must be nonnegative")
    source = _assess_organized_source_attachment(
        host_form_error_bounds=host_rx,
        host_phase_error_bounds=host_rt,
        donor_form_error_bound=donor_rx,
        donor_phase_error_bound=donor_rt,
    )
    barrier = _derive_organized_source_capture_barrier()
    host, gamma = source.host, barrier.geometry.gamma_bounds
    inner = host.nominal_inner_gap_at_entry_bounds.hi
    outer = host.nominal_outer_gap_at_entry_bounds.hi
    form_error, phase_error = (
        host.full_form_error_upper_bound,
        host.full_phase_error_upper_bound,
    )
    donor_error = source.donor_recovered_norm_bound
    # Match the source owner's upward interval-square envelope in its
    # contact work, rather than introducing a second nominal form bound.
    form_scale = gamma.hi * (1 + 2 * (gamma**2).hi)
    nominal = 9 * (1 + form_scale**2) * (inner**2 + outer**2)
    # Exact expansion of the edgewise bound in the theorem. In particular,
    # residual linear terms do not vanish when reached child means differ.
    residual = 74 * (form_error**2 + phase_error**2) + 36 * (inner + outer) * (
        form_scale * form_error + phase_error
    )
    host_excess, donor_excess = nominal + residual, 4 * donor_error**2
    before = host_excess + donor_excess
    initial = before + source.attachment_work_upper_bound
    margin = barrier.boundary_energy_lower_bound - initial
    alpha, half_pi = barrier.geometry.alpha_bounds, barrier.cell_edge_gap_bounds
    fine_error = max(phase_error, donor_error)
    internal = alpha + I(-2 * fine_error, 2 * fine_error)
    matching = max(inner, outer) + 2 * phase_error
    contact = I(
        source.contact_phase_gap_lower_bound, source.contact_phase_gap_upper_bound
    )
    in_cell = (
        0 < internal.lo <= internal.hi < half_pi.lo
        and matching < half_pi.lo
        and 0 < contact.lo <= contact.hi < half_pi.lo
    )
    source_ok = source.admitted
    unresolved = tuple(
        label
        for passed, label in (
            (source_ok, "actual_original_sources_and_attachment_not_certified"),
            (barrier.target_certified, "complete_joined_target_not_certified"),
            (barrier.barrier_certified, "complete_acute_chart_boundary_not_certified"),
            (in_cell, "actual_source_not_certified_inside_acute_chart"),
            (margin > 0, "actual_source_storage_not_strictly_below_boundary"),
        )
        if not passed
    )
    capture = not unresolved
    control = _OrganizedSourceDisconnectedControl(
        weighted_edges=source.geometry.host_edges + source.geometry.donor_edges,
        components=(tuple(range(36)), tuple(range(36, 45))),
        component_form_means=(Q(0), Q(0)),
        component_phase_offset_means=(Q(0), source.donor_phase_origin),
        common_target_phase_offset=_TARGET_ORIGIN,
        target_component_mean_mismatches=(
            -_TARGET_ORIGIN,
            source.donor_phase_origin - _TARGET_ORIGIN,
        ),
        donor_host_phase_difference=source.donor_phase_origin,
        joined_alignment_excluded=source_ok,
    )
    return _OrganizedSourceCapture(
        source=source,
        barrier=barrier,
        control=control,
        source_host_nominal_excess_upper_bound=nominal,
        source_host_residual_excess_upper_bound=residual,
        source_host_excess_upper_bound=host_excess,
        source_donor_excess_upper_bound=donor_excess,
        pre_attachment_excess_upper_bound=before,
        initial_excess_upper_bound=initial,
        capture_margin=margin,
        initial_internal_phase_gap_bounds=internal,
        initial_host_matching_gap_upper_bound=matching,
        initial_contact_phase_gap_bounds=contact,
        initial_donor_host_phase_difference=source.donor_phase_origin,
        asymptotic_donor_host_phase_difference=Q(0) if capture else None,
        source_certified=source_ok,
        conditional_source_in_cell=in_cell,
        source_in_cell=source_ok and in_cell,
        barrier_certified=barrier.barrier_certified,
        capture_certified=capture,
        winding_retained=capture,
        converges_to_target=capture,
        status="certified_organized_source_capture" if capture else "unavailable",
        unresolved_conditions=unresolved,
    )
