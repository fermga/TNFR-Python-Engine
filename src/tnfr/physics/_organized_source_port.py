"""Detached complete port inherited from one additional organized C9.

ORGANIZED_SOURCE_PORT.md owns the frozen support, source and theorem. These
observers retain fine state or explicitly restrict to the rigid carrier;
they install no support, event, trajectory or subsequent identity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, sin
from ._organized_parent_acquisition import (
    _OrganizedParentAcquisition,
    _assess_organized_parent_acquisition,
)
from ._repeated_organized_composition import (
    _bounded_sine_edge_fields,
    _derive_repeated_composition,
    _vector,
)
from ._sine_constitutive_recovery import (
    _ConstitutiveRecovery,
    _assess_constitutive_recovery,
)
from ._sine_preparation import _prepared_form_costs

_ORIGINAL_CLOCK_FACTOR = Q(1023, 1024)
_ROW_FACTOR = Q(1, 4)
_DONOR_ORIGIN = Q(1, 8)
_RECOVERED_RADIUS = Q(1, 10**70)
_COLLECTIVE_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))


@dataclass(frozen=True)
class _OrganizedSourceGeometry:
    """Fixed 90-edge support and uniform row factor, not degree normalization."""

    host_edges: tuple[tuple[int, int, Q], ...]
    donor_edges: tuple[tuple[int, int, Q], ...]
    port_edges: tuple[tuple[int, int, Q], ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    weighted_degrees: tuple[Q, ...]
    support_degrees: tuple[int, ...]
    alpha_bounds: I
    gamma_bounds: I
    collective_edges: tuple[tuple[int, int], ...] = _COLLECTIVE_EDGES
    node_order: tuple[int, ...] = tuple(range(45))
    held_capacities: tuple[Q, ...] = (Q(1),) * 45
    clock_factor: Q = _ROW_FACTOR
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2"
    scope: tuple[str, ...] = (
        "host_matching_support_preserved_with_one_supplied_donor_contact",
        "uniform_quarter_factor_in_both_rows_at_all_nonuniform_degrees",
        "held_unit_capacities_no_inputs_and_no_support_occurrence_law",
        "node_9c_plus_j_with_c_in_zero_through_four_and_j_in_zero_through_eight",
    )


def _derive_organized_source_port():
    host = _derive_repeated_composition(interface="matching")
    donor = tuple((36 + j, 36 + (j + 1) % 9, Q(1)) for j in range(9))
    ports = tuple((j, 36 + j, Q(1)) for j in range(9))
    edges = host.weighted_edges + donor + ports
    return _OrganizedSourceGeometry(
        host_edges=host.weighted_edges,
        donor_edges=donor,
        port_edges=ports,
        weighted_edges=edges,
        weighted_degrees=tuple(
            sum((w for a, b, w in edges if node in (a, b)), Q(0)) for node in range(45)
        ),
        support_degrees=tuple(
            sum(node in (a, b) for a, b, _ in edges) for node in range(45)
        ),
        alpha_bounds=host.alpha_bounds,
        gamma_bounds=host.gamma_bounds,
    )


@dataclass(frozen=True)
class _OrganizedSourceObservation:
    """Complete fine rows, both port channels, and regional storage accounts."""

    geometry: _OrganizedSourceGeometry
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    phase_lift_bounds: tuple[I, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    contact_form_differences: tuple[Q, ...]
    contact_phase_differences: tuple[Q, ...]
    host_port_form_tau_bounds: tuple[I, ...]
    host_port_phase_tau_bounds: tuple[I, ...]
    donor_port_form_tau_bounds: tuple[I, ...]
    donor_port_phase_tau_bounds: tuple[I, ...]
    child_form_means: tuple[Q, ...]
    child_phase_offset_means: tuple[Q, ...]
    child_form_tau_rate_bounds: tuple[I, ...]
    child_phase_tau_rate_bounds: tuple[I, ...]
    conserved_form_total: Q
    conserved_phase_total: Q
    host_storage_bounds: I
    donor_storage_bounds: I
    interaction_storage_bounds: I
    joint_storage_bounds: I
    host_internal_form_gradient: tuple[Q, ...]
    donor_internal_form_gradient: tuple[Q, ...]
    host_internal_phase_current_bounds: tuple[I, ...]
    donor_internal_phase_current_bounds: tuple[I, ...]
    host_internal_dissipation_tau: Q
    donor_internal_dissipation_tau: Q
    host_port_power_tau_bounds: I
    donor_port_power_tau_bounds: I
    host_storage_tau_rate_bounds: I
    donor_storage_tau_rate_bounds: I
    interaction_storage_tau_rate_bounds: I
    joint_dissipation_tau: Q
    joint_storage_tau_rate: Q
    joint_storage_slow_rate_bounds: I
    joint_storage_original_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    regional_balance_residual_bounds: I
    scope: tuple[str, ...] = (
        "all_ninety_primitive_coordinates_admitted_before_interval_construction",
        "lift_equals_alpha_times_j_minus_four_plus_supplied_real_offset",
        "both_contact_rows_and_equal_opposite_donor_reaction_retained",
        "regional_internal_storage_has_signed_port_power_not_an_isolated_loss_law",
        "new_edge_storage_is_separate_from_host_and_donor_internal_storage",
        "conserved_totals_are_sums_of_five_child_means_not_raw_node_sums",
        "fine_mean_projection_does_not_assert_exact_collective_closure",
        "static_observation_does_not_admit_source_history_or_future_identity",
    )


def _observe_organized_source_port(*, form, phase_offsets):
    """Observe the declared full law from signed forms and real lifted offsets."""
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    geometry = _derive_organized_source_port()
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    current, potential, edge_currents = _bounded_sine_edge_fields(
        phase_offsets=offsets,
        weighted_edges=geometry.weighted_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    dx = tuple(k * (-q + gamma * v) for q, v in zip(gradient, current))
    dt = tuple(k * gamma * q for q in gradient)
    difference = tuple(values[36 + j] - values[j] for j in range(9))
    phase_gap = tuple(offsets[36 + j] - offsets[j] for j in range(9))
    contact_current = edge_currents[-9:]
    port_dx = tuple(k * (d + gamma * v) for d, v in zip(difference, contact_current))
    port_dt = tuple(-k * gamma * d for d in difference)
    host_gradient = tuple(
        gradient[i] + difference[i] if i < 9 else gradient[i] for i in range(36)
    )
    donor_gradient = tuple(gradient[36 + j] - difference[j] for j in range(9))
    host_current = tuple(
        current[i] - contact_current[i] if i < 9 else current[i] for i in range(36)
    )
    donor_current = tuple(current[36 + j] + contact_current[j] for j in range(9))
    host_power = sum(
        (
            host_gradient[j] * port_dx[j] - host_current[j] * port_dt[j]
            for j in range(9)
        ),
        I(0),
    )
    donor_power = sum(
        (
            -donor_gradient[j] * port_dx[j] + donor_current[j] * port_dt[j]
            for j in range(9)
        ),
        I(0),
    )
    host_loss = k * sum((q**2 for q in host_gradient), Q(0))
    donor_loss = k * sum((q**2 for q in donor_gradient), Q(0))
    interaction_rate = sum(
        (
            difference[j] * (dx[36 + j] - dx[j])
            + contact_current[j] * (dt[36 + j] - dt[j])
            for j in range(9)
        ),
        I(0),
    )
    loss = k * sum((q**2 for q in gradient), Q(0))
    direct = sum((q * a - v * b for q, v, a, b in zip(gradient, current, dx, dt)), I(0))
    total_costs = tuple(I(a) + b for a, b in zip(costs, potential))
    slow = 2 / gamma**2

    def means(vector, zero):
        return tuple(
            sum(vector[start : start + 9], zero) / 9 for start in range(0, 45, 9)
        )

    xmeans, tmeans = means(values, Q(0)), means(offsets, Q(0))
    return _OrganizedSourceObservation(
        geometry=geometry,
        form=values,
        phase_offsets=offsets,
        phase_lift_bounds=tuple(
            geometry.alpha_bounds * (i % 9 - 4) + value
            for i, value in enumerate(offsets)
        ),
        form_gradient=gradient,
        phase_current_bounds=current,
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dt,
        form_slow_rate_bounds=tuple(slow * rate for rate in dx),
        phase_slow_rate_bounds=tuple(slow * rate for rate in dt),
        form_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * rate for rate in dx),
        phase_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * rate for rate in dt),
        contact_form_differences=difference,
        contact_phase_differences=phase_gap,
        host_port_form_tau_bounds=port_dx,
        host_port_phase_tau_bounds=port_dt,
        donor_port_form_tau_bounds=tuple(-rate for rate in port_dx),
        donor_port_phase_tau_bounds=tuple(-rate for rate in port_dt),
        child_form_means=xmeans,
        child_phase_offset_means=tmeans,
        child_form_tau_rate_bounds=means(dx, I(0)),
        child_phase_tau_rate_bounds=means(dt, I(0)),
        conserved_form_total=sum(xmeans, Q(0)),
        conserved_phase_total=sum(tmeans, Q(0)),
        host_storage_bounds=sum(total_costs[:72], I(0)),
        donor_storage_bounds=sum(total_costs[72:81], I(0)),
        interaction_storage_bounds=sum(total_costs[81:], I(0)),
        joint_storage_bounds=sum(total_costs, I(0)),
        host_internal_form_gradient=host_gradient,
        donor_internal_form_gradient=donor_gradient,
        host_internal_phase_current_bounds=host_current,
        donor_internal_phase_current_bounds=donor_current,
        host_internal_dissipation_tau=host_loss,
        donor_internal_dissipation_tau=donor_loss,
        host_port_power_tau_bounds=host_power,
        donor_port_power_tau_bounds=donor_power,
        host_storage_tau_rate_bounds=-host_loss + host_power,
        donor_storage_tau_rate_bounds=-donor_loss + donor_power,
        interaction_storage_tau_rate_bounds=interaction_rate,
        joint_dissipation_tau=loss,
        joint_storage_tau_rate=-loss,
        joint_storage_slow_rate_bounds=-slow * loss,
        joint_storage_original_rate=-_ORIGINAL_CLOCK_FACTOR * loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
        regional_balance_residual_bounds=(
            -host_loss + host_power - donor_loss + donor_power + interaction_rate + loss
        ),
    )


@dataclass(frozen=True)
class _OrganizedSourceCollective:
    """Exact restriction to five uniform forms and five carrier origins."""

    geometry: _OrganizedSourceGeometry
    form_means: tuple[Q, ...]
    phase_origins: tuple[Q, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    conserved_form_total: Q
    conserved_phase_total: Q
    donor_form_difference: Q
    donor_phase_difference: Q
    host_port_form_tau_bounds: I
    host_port_phase_tau_bounds: I
    donor_port_form_tau_bounds: I
    donor_port_phase_tau_bounds: I
    internal_cycle_storage_bounds: I
    interaction_storage_bounds: I
    joint_storage_bounds: I
    joint_dissipation_tau: Q
    joint_storage_tau_rate: Q
    joint_storage_slow_rate_bounds: I
    joint_storage_original_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    scope: tuple[str, ...] = (
        "exact_invariant_winding_one_carrier_family_only",
        "ten_collective_coordinates_retain_both_donor_channels",
        "conserved_totals_sum_five_child_means_not_raw_node_sums",
        "eight_host_coordinates_close_only_when_both_totals_are_retained",
        "fine_child_means_outside_the_carrier_need_additional_correlations",
        "no_acquired_source_reset_or_future_perturbation_identity_claim",
    )


def _observe_organized_source_collective(*, form_means, phase_origins):
    """Evaluate the exact carrier restriction, not a generic mean closure."""
    values = _vector(form_means, "form_means", 5)
    origins = _vector(phase_origins, "phase_origins", 5)
    return _observe_admitted_collective(values, origins)


def _observe_admitted_collective(values, origins):
    geometry = _derive_organized_source_port()
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    edges = tuple((a, b, Q(1)) for a, b in geometry.collective_edges)
    costs, gradient = _prepared_form_costs(form=values, weighted_edges=edges)
    current, potentials = [I(0) for _ in range(5)], []
    for a, b in geometry.collective_edges:
        gap = I(origins[b] - origins[a])
        edge_current = sin(gap)
        current[a] += edge_current
        current[b] -= edge_current
        potentials.append(1 - cos(gap))
    dx = tuple(k * (-q + gamma * v) for q, v in zip(gradient, current))
    dt = tuple(k * gamma * q for q in gradient)
    difference, gap = values[4] - values[0], origins[4] - origins[0]
    port_dx, port_dt = k * (difference + gamma * sin(I(gap))), -k * gamma * difference
    cycles = 45 * (1 - cos(geometry.alpha_bounds))
    interaction = 9 * (costs[-1] + potentials[-1])
    storage = cycles + 9 * (sum(costs, Q(0)) + sum(potentials, I(0)))
    loss = 9 * k * sum((q**2 for q in gradient), Q(0))
    direct = 9 * sum(
        (q * a - v * b for q, v, a, b in zip(gradient, current, dx, dt)), I(0)
    )
    slow = 2 / gamma**2
    return _OrganizedSourceCollective(
        geometry=geometry,
        form_means=values,
        phase_origins=origins,
        form_gradient=gradient,
        phase_current_bounds=tuple(current),
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dt,
        form_slow_rate_bounds=tuple(slow * rate for rate in dx),
        phase_slow_rate_bounds=tuple(slow * rate for rate in dt),
        form_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * rate for rate in dx),
        phase_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * rate for rate in dt),
        conserved_form_total=sum(values, Q(0)),
        conserved_phase_total=sum(origins, Q(0)),
        donor_form_difference=difference,
        donor_phase_difference=gap,
        host_port_form_tau_bounds=port_dx,
        host_port_phase_tau_bounds=port_dt,
        donor_port_form_tau_bounds=-port_dx,
        donor_port_phase_tau_bounds=-port_dt,
        internal_cycle_storage_bounds=cycles,
        interaction_storage_bounds=interaction,
        joint_storage_bounds=storage,
        joint_dissipation_tau=loss,
        joint_storage_tau_rate=-loss,
        joint_storage_slow_rate_bounds=-slow * loss,
        joint_storage_original_rate=-_ORIGINAL_CLOCK_FACTOR * loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
    )


def _reconstruct_organized_source_collective(
    *, host_form_means, host_phase_origins, conserved_form_total, conserved_phase_total
):
    """Recover donor means from sums of FIVE CHILD MEANS and four host means.

    The retained constants equal raw 45-node totals divided by nine. This
    exact reconstruction closes the carrier family; generic fine projections
    preserve these totals but still discard phase correlations.
    """
    forms = _vector(host_form_means, "host_form_means", 4)
    phases = _vector(host_phase_origins, "host_phase_origins", 4)
    form_total = _exact(conserved_form_total, "conserved_form_total")
    phase_total = _exact(conserved_phase_total, "conserved_phase_total")
    return _observe_admitted_collective(
        forms + (form_total - sum(forms, Q(0)),),
        phases + (phase_total - sum(phases, Q(0)),),
    )


@dataclass(frozen=True)
class _OrganizedSourceAttachment:
    """Actual independently recovered source admission at a supplied contact."""

    host_form_error_bounds: tuple[Q, ...]
    host_phase_error_bounds: tuple[Q, ...]
    donor_form_error_bound: Q
    donor_phase_error_bound: Q
    host: _OrganizedParentAcquisition
    donor: _ConstitutiveRecovery
    geometry: _OrganizedSourceGeometry
    relative_preparation_dimension: int
    original_preparation_storage_upper_bound: Q
    donor_original_preparation_storage_upper_bound: Q
    host_elapsed_acquisition_tau_bounds: I
    host_elapsed_acquisition_original_bounds: I
    donor_formation_tau: Q
    donor_recovery_duration_tau: Q
    donor_handoff_tau: Q
    donor_handoff_original: Q
    host_fine_form_error_upper_bound: Q
    host_fine_phase_error_upper_bound: Q
    donor_recovered_norm_bound: Q
    contact_form_gap_absolute_upper_bound: Q
    contact_phase_gap_lower_bound: Q
    contact_phase_gap_upper_bound: Q
    attachment_work_lower_bound: Q
    attachment_work_upper_bound: Q
    host_pre_attachment_storage_upper_bound: Q
    donor_pre_attachment_excess_storage_upper_bound: Q
    donor_pre_attachment_storage_bounds: I
    joined_initial_storage_upper_bound: Q
    sources_certified: bool
    initial_child_identities_certified: bool
    positive_attachment_work_certified: bool
    attachment_work_bound_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    donor_phase_origin: Q = _DONOR_ORIGIN
    donor_form_mean: Q = Q(0)
    conserved_form_total: Q = Q(0)
    conserved_phase_total: Q = _DONOR_ORIGIN
    donor_clock_dilation: Q = Q(2)
    attachment_work_upper_envelope: Q = Q(1, 10)
    protocol_revision: str = "02a6415f"
    scope: tuple[str, ...] = (
        "all_ten_raw_allowances_admitted_before_either_source_owner_is_called",
        "four_original_independent_host_sources_and_one_independent_donor_source",
        "host_actual_acquisition_at_s_four_with_all_carried_errors_retained",
        "donor_quarter_row_is_old_half_row_under_old_tau_equals_new_tau_over_two",
        "both_donor_rows_and_formation_and_dwell_times_share_the_clock_dilation",
        "donor_phase_origin_one_eighth_is_supplied_without_resetting_fine_state",
        "preparation_start_times_align_handoffs_without_unproved_extra_waiting",
        "actual_new_edge_work_is_separate_and_not_funded_by_prior_dissipation",
        "one_tenth_is_a_derived_work_envelope_not_a_frozen_budget",
        "candidate_bounds_are_conditional_when_original_sources_are_unavailable",
        "nested_source_future_bounds_keep_their_old_support_and_law_premises",
        "no_post_attachment_maintenance_response_or_autonomous_occurrence_claim",
    )


def _assess_organized_source_attachment(
    *,
    host_form_error_bounds,
    host_phase_error_bounds,
    donor_form_error_bound,
    donor_phase_error_bound,
):
    """Rebuild both preparation histories before admitting their new contact."""
    host_rx = _vector(host_form_error_bounds, "host_form_error_bounds", 4)
    host_rt = _vector(host_phase_error_bounds, "host_phase_error_bounds", 4)
    donor_rx = _exact(donor_form_error_bound, "donor_form_error_bound")
    donor_rt = _exact(donor_phase_error_bound, "donor_phase_error_bound")
    if min(*host_rx, *host_rt, donor_rx, donor_rt) < 0:
        raise ValueError("original preparation error bounds must be nonnegative")
    host = _assess_organized_parent_acquisition(
        form_error_bounds=host_rx, phase_error_bounds=host_rt
    )
    donor = _assess_constitutive_recovery(
        deformation=0, form_error_bound=donor_rx, phase_error_bound=donor_rt
    )
    geometry = _derive_organized_source_port()
    gamma, radius = geometry.gamma_bounds, _RECOVERED_RADIUS
    eta = gamma**2
    phase_nominal = (
        host.nominal_inner_gap_at_entry_bounds.hi
        + host.nominal_outer_gap_at_entry_bounds.hi
    ) / 2
    form_gap = (
        gamma.hi * (1 + 2 * eta.hi) * phase_nominal
        + host.full_form_error_upper_bound
        + radius
    )
    phase_error = host.full_phase_error_upper_bound + radius
    phase_lower = _DONOR_ORIGIN - phase_error
    phase_upper = _DONOR_ORIGIN + phase_nominal + phase_error
    work_upper = Q(9, 2) * (form_gap**2 + phase_upper**2)
    phase_acute = 0 < phase_lower <= phase_upper < geometry.alpha_bounds.lo
    work_lower = 9 * (1 - cos(I(phase_lower))).lo if phase_acute else Q(0)
    donor_ok = (
        donor.recovery_certified
        and donor.endpoint_form_norm_squared_upper_bound is not None
        and donor.endpoint_phase_norm_squared_upper_bound is not None
        and donor.endpoint_form_norm_squared_upper_bound < radius**2
        and donor.endpoint_phase_norm_squared_upper_bound < radius**2
    )
    sources_ok = host.entry_certified and host.child_identities_retained and donor_ok
    positive = sources_ok and work_lower > 0
    work_ok = sources_ok and work_upper < Q(1, 10)
    admitted = sources_ok and positive and work_ok
    donor_storage = 9 * (1 - cos(geometry.alpha_bounds)) + I(0, 4 * radius**2)
    host_storage_upper = host.complete_initial_storage_bounds.hi
    donor_preparation = donor.source.initial_storage_upper_bounds[0]
    unresolved = tuple(
        label
        for passed, label in (
            (
                host.entry_certified and host.child_identities_retained,
                "host_actual_acquired_source_not_certified",
            ),
            (donor_ok, "donor_clock_transformed_original_recovery_not_certified"),
            (
                phase_acute and work_lower > 0,
                "strict_positive_contact_work_not_certified",
            ),
            (work_upper < Q(1, 10), "contact_work_upper_bound_not_certified"),
        )
        if not passed
    )
    return _OrganizedSourceAttachment(
        host_form_error_bounds=host_rx,
        host_phase_error_bounds=host_rt,
        donor_form_error_bound=donor_rx,
        donor_phase_error_bound=donor_rt,
        host=host,
        donor=donor,
        geometry=geometry,
        relative_preparation_dimension=host.relative_preparation_dimension
        + donor.source.preparation_dimension,
        original_preparation_storage_upper_bound=host.original_preparation_storage_upper_bound
        + donor_preparation,
        donor_original_preparation_storage_upper_bound=donor_preparation,
        host_elapsed_acquisition_tau_bounds=host.elapsed_scaled_acquisition_time_bounds,
        host_elapsed_acquisition_original_bounds=host.elapsed_original_acquisition_time_bounds,
        donor_formation_tau=2 * donor.formation_time,
        donor_recovery_duration_tau=2 * donor.relaxation_duration,
        donor_handoff_tau=2 * donor.scaled_handoff_time,
        donor_handoff_original=2 * donor.original_handoff_time,
        host_fine_form_error_upper_bound=host.full_form_error_upper_bound,
        host_fine_phase_error_upper_bound=host.full_phase_error_upper_bound,
        donor_recovered_norm_bound=radius,
        contact_form_gap_absolute_upper_bound=form_gap,
        contact_phase_gap_lower_bound=phase_lower,
        contact_phase_gap_upper_bound=phase_upper,
        attachment_work_lower_bound=work_lower,
        attachment_work_upper_bound=work_upper,
        host_pre_attachment_storage_upper_bound=host_storage_upper,
        donor_pre_attachment_excess_storage_upper_bound=4 * radius**2,
        donor_pre_attachment_storage_bounds=donor_storage,
        joined_initial_storage_upper_bound=host_storage_upper
        + donor_storage.hi
        + work_upper,
        sources_certified=sources_ok,
        initial_child_identities_certified=sources_ok,
        positive_attachment_work_certified=positive,
        attachment_work_bound_certified=work_ok,
        admitted=admitted,
        status="certified_organized_source_attachment" if admitted else "unavailable",
        unresolved_conditions=unresolved,
    )
