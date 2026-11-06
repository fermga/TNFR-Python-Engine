"""Detached complete port inherited from one additional organized C9.

ORGANIZED_SOURCE_PORT.md owns the frozen support, source and theorem. These
observers retain fine state or explicitly restrict to the rigid carrier;
they install no support, event, trajectory or subsequent identity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_parent_acquisition import (
    _assess_organized_parent_acquisition,
    _OrganizedParentAcquisition,
)
from ._repeated_organized_composition import (
    _bounded_sine_edge_fields,
    _derive_repeated_composition,
    _vector,
)
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)
from ._sine_preparation import _prepared_form_costs

_ORIGINAL_CLOCK_FACTOR = Q(1023, 1024)
_ROW_FACTOR = Q(1, 4)
_DONOR_ORIGIN = Q(1, 8)
_RECOVERED_RADIUS = Q(1, 10**70)
_COLLECTIVE_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
_REFLECTION_PAIRS = ((0, 1), (2, 8), (3, 7), (4, 6))
_REFLECTION_FIXED_SLOT = 5
_REFLECTION_ORIGIN = Q(1, 40)
_REFLECTION_NODES = tuple(9 * c + j for c in range(5) for j, _ in _REFLECTION_PAIRS)


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


def _organized_source_rate_rows(*, geometry, gradient, current):
    """Assemble the declared unit-capacity rows from caller-admitted fields."""
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    form = tuple(k * (-q + gamma * value) for q, value in zip(gradient, current))
    phase = tuple(k * gamma * q for q in gradient)
    slow = 2 / gamma**2
    return {
        "form_tau_rate_bounds": form,
        "phase_tau_rate_bounds": phase,
        "form_slow_rate_bounds": tuple(slow * rate for rate in form),
        "phase_slow_rate_bounds": tuple(slow * rate for rate in phase),
        "form_original_rate_bounds": tuple(
            _ORIGINAL_CLOCK_FACTOR * rate for rate in form
        ),
        "phase_original_rate_bounds": tuple(
            _ORIGINAL_CLOCK_FACTOR * rate for rate in phase
        ),
    }


@dataclass(frozen=True)
class _OrganizedSourcePhaseState:
    """Fresh complete-law fields at supplied raw primitive phase coordinates."""

    geometry: _OrganizedSourceGeometry
    form: tuple[Q, ...]
    phases: tuple[Q, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    edge_phase_current_bounds: tuple[I, ...]
    form_edge_storage: tuple[Q, ...]
    phase_edge_storage_bounds: tuple[I, ...]
    edge_storage_bounds: tuple[I, ...]
    pressure_tau_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    joint_storage_bounds: I
    joint_dissipation_tau: Q
    joint_storage_tau_rate: Q
    joint_storage_slow_rate_bounds: I
    joint_storage_original_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    scope: tuple[str, ...] = (
        "all_ninety_raw_coordinates_admitted_before_geometry_or_intervals",
        "mathematical_trigonometry_of_supplied_phases_without_carrier_reencoding",
        "fixed_ninety_edge_support_and_held_unit_capacities",
        "fresh_declared_pressure_is_not_an_operator_stored_pressure_channel",
        "both_complete_law_rows_and_all_three_clock_conventions_retained",
        "no_continuous_lift_association_reflection_winding_or_retention_claim",
        "no_event_execution_grammar_admission_or_source_occurrence_claim",
    )


def _observe_organized_source_phase_state(*, form, phases):
    """Observe the fixed full law directly at raw finite phase coordinates.

    Exact rationals and represented reals use the same primitive admission as
    the ideal-offset observer. Supplied phases are not wrapped, projected or
    interpreted as carrier offsets. Mathematical sine and cosine enclose their
    actual pair differences; they are not evaluations at an associated ideal
    reflection state. Any event lift or relation to that state needs a separate
    certificate. The observer installs neither state nor a continuation law.
    """
    values = _vector(form, "form", 45)
    angles = _vector(phases, "phases", 45)
    geometry = _derive_organized_source_port()
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    current, potential, edge_currents = _bounded_sine_edge_fields(
        phase_offsets=angles,
        weighted_edges=geometry.weighted_edges,
        alpha_bounds=I(0),
    )
    rows = _organized_source_rate_rows(
        geometry=geometry, gradient=gradient, current=current
    )
    dx, dt = rows["form_tau_rate_bounds"], rows["phase_tau_rate_bounds"]
    total_costs = tuple(I(cost) + value for cost, value in zip(costs, potential))
    loss = geometry.clock_factor * sum((q**2 for q in gradient), Q(0))
    direct = sum((q * a - v * b for q, v, a, b in zip(gradient, current, dx, dt)), I(0))
    return _OrganizedSourcePhaseState(
        geometry=geometry,
        form=values,
        phases=angles,
        form_gradient=gradient,
        phase_current_bounds=current,
        edge_phase_current_bounds=edge_currents,
        form_edge_storage=costs,
        phase_edge_storage_bounds=potential,
        edge_storage_bounds=total_costs,
        pressure_tau_bounds=dx,
        **rows,
        joint_storage_bounds=sum(total_costs, I(0)),
        joint_dissipation_tau=loss,
        joint_storage_tau_rate=-loss,
        joint_storage_slow_rate_bounds=-(2 / geometry.gamma_bounds**2) * loss,
        joint_storage_original_rate=-_ORIGINAL_CLOCK_FACTOR * loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
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
    rows = _organized_source_rate_rows(
        geometry=geometry, gradient=gradient, current=current
    )
    dx, dt = rows["form_tau_rate_bounds"], rows["phase_tau_rate_bounds"]
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
        **rows,
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
class _OrganizedSourceCapacityObservation:
    """Detached complete rows for one explicit held-capacity phase premise."""

    unit_capacity_reference: _OrganizedSourceObservation
    capacities: tuple[Q, ...]
    phase_mobility: str
    pressure_tau_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    dissipation_tau: Q
    storage_tau_rate_bounds: I
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    inverse_capacity_weighted_form_total: Q | None
    inverse_capacity_weighted_phase_offset_total: Q | None
    scope: tuple[str, ...] = (
        "all_raw_form_phase_and_capacity_coordinates_admitted_before_fine_observation",
        "held_nonnegative_capacities_on_the_fixed_source_port_support_and_clocks",
        "pressure_is_the_declared_diffusive_sine_map_not_native_resultant_pressure",
        "phase_mobility_is_an_independent_supplied_complete_law_premise",
        "unit_capacity_reference_rates_are_not_the_changed_capacity_law",
        "inverse_capacity_totals_only_for_positive_shared_capacity_segments",
        "zero_capacity_freezes_phase_only_under_the_shared_capacity_premise",
        "no_event_execution_grammar_occurrence_or_retention_certificate",
    )


def _admit_organized_source_capacities(capacities, phase_mobility):
    """Admit the supplied mobility branch before constructing any observation."""
    if type(phase_mobility) is not str or phase_mobility not in (
        "shared_capacity",
        "unit_capacity",
    ):
        raise ValueError("phase_mobility must be 'shared_capacity' or 'unit_capacity'")
    mobilities = _vector(capacities, "capacities", 45)
    if any(value < 0 for value in mobilities):
        raise ValueError("capacities must be nonnegative")
    return mobilities


def _organized_source_capacity_rows(*, reference, mobilities, phase_mobility):
    """Share complete-law algebra across ideal-offset and raw-phase observers."""
    gradient, current = reference.form_gradient, reference.phase_current_bounds
    gamma, k = reference.geometry.gamma_bounds, reference.geometry.clock_factor
    pressure = tuple(k * (-q + gamma * s) for q, s in zip(gradient, current))
    dx = tuple(nu * p for nu, p in zip(mobilities, pressure))
    phase_capacities = (
        mobilities if phase_mobility == "shared_capacity" else (Q(1),) * 45
    )
    dtheta = tuple(k * gamma * nu * q for nu, q in zip(phase_capacities, gradient))
    loss = k * sum((nu * q**2 for nu, q in zip(mobilities, gradient)), Q(0))
    storage_rate = I(-loss)
    if phase_mobility == "unit_capacity":
        storage_rate += (
            k
            * gamma
            * sum(
                (s * (nu - 1) * q for s, nu, q in zip(current, mobilities, gradient)),
                I(0),
            )
        )
    direct = sum(
        (q * a - s * b for q, s, a, b in zip(gradient, current, dx, dtheta)),
        I(0),
    )
    slow = 2 / gamma**2
    return dict(
        pressure_tau_bounds=pressure,
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dtheta,
        form_slow_rate_bounds=tuple(slow * rate for rate in dx),
        phase_slow_rate_bounds=tuple(slow * rate for rate in dtheta),
        form_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * rate for rate in dx),
        phase_original_rate_bounds=tuple(
            _ORIGINAL_CLOCK_FACTOR * rate for rate in dtheta
        ),
        dissipation_tau=loss,
        storage_tau_rate_bounds=storage_rate,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct - storage_rate,
    )


def _observe_organized_source_capacity(
    *, form, phase_offsets, capacities, phase_mobility
):
    """Observe supplied held capacities without replacing the pressure law.

    ``shared_capacity`` multiplies both complete rows by the same local
    capacity. ``unit_capacity`` multiplies only the nodal form row and retains
    the previous phase row. The latter has an additional storage cross term.
    Neither choice is selected by the nodal identity or an operator label.

    Zero capacity is admitted. Inverse-capacity weighted totals are reported
    only for the positive shared branch, where they are conserved between
    events. Phase totals use the declared real offsets, not wrapped angles.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    mobilities = _admit_organized_source_capacities(capacities, phase_mobility)
    reference = _observe_organized_source_port(form=values, phase_offsets=offsets)
    form_total = phase_total = None
    if phase_mobility == "shared_capacity" and all(nu > 0 for nu in mobilities):
        form_total = sum((x / nu for x, nu in zip(values, mobilities)), Q(0))
        phase_total = sum((delta / nu for delta, nu in zip(offsets, mobilities)), Q(0))
    return _OrganizedSourceCapacityObservation(
        unit_capacity_reference=reference,
        capacities=mobilities,
        phase_mobility=phase_mobility,
        **_organized_source_capacity_rows(
            reference=reference, mobilities=mobilities, phase_mobility=phase_mobility
        ),
        inverse_capacity_weighted_form_total=form_total,
        inverse_capacity_weighted_phase_offset_total=phase_total,
    )


@dataclass(frozen=True)
class _OrganizedSourceRawCapacityObservation:
    """Complete held-capacity rows evaluated at raw represented phases."""

    unit_capacity_reference: _OrganizedSourcePhaseState
    capacities: tuple[Q, ...]
    phase_mobility: str
    pressure_tau_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    dissipation_tau: Q
    storage_tau_rate_bounds: I
    storage_slow_rate_bounds: I
    storage_original_rate_bounds: I
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    inverse_capacity_weighted_form_total: Q | None
    inverse_capacity_weighted_raw_phase_total: Q | None
    scope: tuple[str, ...] = (
        "all_raw_form_phase_and_capacity_values_admitted_before_observation",
        "fixed_support_and_clocks_reuse_the_unit_reference_geometry_not_its_rates",
        "phase_mobility_is_an_explicit_complete_law_premise_not_an_operator_effect",
        "pressure_is_independent_of_capacity_and_held_unit_reference_rates",
        "raw_phase_total_is_not_a_continuous_lift_without_declared_integer_turns",
        "inverse_totals_available_only_on_positive_shared_capacity_segments",
        "no_carrier_offset_reconstruction_event_execution_or_retention_claim",
    )


def _observe_organized_source_raw_capacity(*, form, phases, capacities, phase_mobility):
    """Reuse raw trigonometric fields with an explicit held-capacity law.

    Zero capacity is admitted here, but cannot support an inverse-capacity
    leaf or positive-mobility convergence certificate. Integer phase turns
    belong to the separate continuation admission, never inferred here.
    """
    values = _vector(form, "form", 45)
    angles = _vector(phases, "phases", 45)
    mobilities = _admit_organized_source_capacities(capacities, phase_mobility)
    reference = _observe_organized_source_phase_state(form=values, phases=angles)
    rows = _organized_source_capacity_rows(
        reference=reference, mobilities=mobilities, phase_mobility=phase_mobility
    )
    form_total = phase_total = None
    if phase_mobility == "shared_capacity" and all(nu > 0 for nu in mobilities):
        form_total = sum((x / nu for x, nu in zip(values, mobilities)), Q(0))
        phase_total = sum((angle / nu for angle, nu in zip(angles, mobilities)), Q(0))
    return _OrganizedSourceRawCapacityObservation(
        unit_capacity_reference=reference,
        capacities=mobilities,
        phase_mobility=phase_mobility,
        **rows,
        storage_slow_rate_bounds=(
            2 / reference.geometry.gamma_bounds**2 * rows["storage_tau_rate_bounds"]
        ),
        storage_original_rate_bounds=(
            _ORIGINAL_CLOCK_FACTOR * rows["storage_tau_rate_bounds"]
        ),
        inverse_capacity_weighted_form_total=form_total,
        inverse_capacity_weighted_raw_phase_total=phase_total,
    )


@dataclass(frozen=True)
class _OrganizedSourceFormJump:
    """Raw endpoint accounting under declared held phase, support and capacity."""

    before: _OrganizedSourceObservation
    after: _OrganizedSourceObservation
    form_displacement: tuple[Q, ...]
    gradient_jump: tuple[Q, ...]
    storage_jump: Q
    form_tau_rate_jump_bounds: tuple[I, ...]
    phase_tau_rate_jump_bounds: tuple[I, ...]
    form_slow_rate_jump_bounds: tuple[I, ...]
    phase_slow_rate_jump_bounds: tuple[I, ...]
    form_original_rate_jump_bounds: tuple[I, ...]
    phase_original_rate_jump_bounds: tuple[I, ...]
    scope: tuple[str, ...] = (
        "all_primitive_endpoints_admitted_before_fine_law_observation",
        "one_declared_phase_lift_fixed_support_and_held_unit_capacities",
        "both_fresh_rows_rebuilt_not_stored_runtime_pressure",
        "exact_form_storage_jump_includes_all_target_cross_terms",
        "correlated_rate_differences_use_the_unchanged_sine_current",
        "no_event_authentication_grammar_admission_or_occurrence_certificate",
        "no_event_duration_or_future_maintenance_claim",
    )


def _observe_organized_source_form_jump(*, form_before, form_after, phase_offsets):
    """Rebuild the same-law balance for two supplied form endpoints.

    Callers relating this calculation to an executed event must separately
    establish that its phase, support and capacities were held. No graph,
    operator label or cached report substitutes for those event premises.
    """
    initial = _vector(form_before, "form_before", 45)
    final = _vector(form_after, "form_after", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    before = _observe_organized_source_port(form=initial, phase_offsets=offsets)
    after = _observe_organized_source_port(form=final, phase_offsets=offsets)
    displacement = tuple(b - a for a, b in zip(initial, final))
    gradient_jump = tuple(
        b - a for a, b in zip(before.form_gradient, after.form_gradient)
    )
    work = sum(
        (
            h * (q + dq / 2)
            for h, q, dq in zip(displacement, before.form_gradient, gradient_jump)
        ),
        Q(0),
    )
    gamma, k = before.geometry.gamma_bounds, before.geometry.clock_factor
    dx = tuple(I(-k * dq) for dq in gradient_jump)
    dtheta = tuple(k * gamma * dq for dq in gradient_jump)
    slow = 2 / gamma**2
    return _OrganizedSourceFormJump(
        before=before,
        after=after,
        form_displacement=displacement,
        gradient_jump=gradient_jump,
        storage_jump=work,
        form_tau_rate_jump_bounds=dx,
        phase_tau_rate_jump_bounds=dtheta,
        form_slow_rate_jump_bounds=tuple(slow * rate for rate in dx),
        phase_slow_rate_jump_bounds=tuple(slow * rate for rate in dtheta),
        form_original_rate_jump_bounds=tuple(
            _ORIGINAL_CLOCK_FACTOR * rate for rate in dx
        ),
        phase_original_rate_jump_bounds=tuple(
            _ORIGINAL_CLOCK_FACTOR * rate for rate in dtheta
        ),
    )


@dataclass(frozen=True)
class _OrganizedSourceAcceleration:
    """Same-law instantaneous derivatives rebuilt from primitive fine state."""

    state: _OrganizedSourceObservation
    gradient_tau_rate_bounds: tuple[I, ...]
    phase_current_tau_rate_bounds: tuple[I, ...]
    form_tau_acceleration_bounds: tuple[I, ...]
    phase_tau_acceleration_bounds: tuple[I, ...]
    form_slow_acceleration_bounds: tuple[I, ...]
    phase_slow_acceleration_bounds: tuple[I, ...]
    form_original_acceleration_bounds: tuple[I, ...]
    phase_original_acceleration_bounds: tuple[I, ...]
    child_form_tau_acceleration_bounds: tuple[I, ...]
    child_phase_tau_acceleration_bounds: tuple[I, ...]
    scope: tuple[str, ...] = (
        "primitive_fine_state_readmitted_and_complete_first_rows_rebuilt",
        "held_support_unit_capacities_uniform_quarter_mobility_and_no_inputs",
        "phase_current_derivative_retains_each_edge_phase_and_relative_rate",
        "both_accelerations_are_time_derivatives_not_taylor_coefficients",
        "constant_clock_changes_apply_squared_rate_factors",
        "predicted_local_derivatives_not_observed_history_or_finite_time_bounds",
        "no_source_acquisition_event_selection_or_operator_admission",
    )


def _observe_organized_source_acceleration(*, form, phase_offsets):
    """Differentiate both complete rows, accepting no cached observation.

    The first-row owner admits every primitive before deriving geometry or
    intervals. Edge incidence then differentiates its form gradient and sine
    current together, retaining the actual conductance and continuous lift.
    """
    state = _observe_organized_source_port(form=form, phase_offsets=phase_offsets)
    geometry = state.geometry
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    dx, dtheta = state.form_tau_rate_bounds, state.phase_tau_rate_bounds
    dq, ds = [I(0) for _ in state.form], [I(0) for _ in state.form]
    for a, b, weight in geometry.weighted_edges:
        gap = (
            geometry.alpha_bounds * ((b % 9) - (a % 9))
            + state.phase_offsets[b]
            - state.phase_offsets[a]
        )
        gradient_rate = weight * (dx[a] - dx[b])
        current_rate = weight * cos(gap) * (dtheta[b] - dtheta[a])
        dq[a] += gradient_rate
        dq[b] -= gradient_rate
        ds[a] += current_rate
        ds[b] -= current_rate
    ddx = tuple(k * (-q + gamma * s) for q, s in zip(dq, ds))
    ddtheta = tuple(k * gamma * q for q in dq)
    slow_factor = (2 / gamma**2) ** 2
    original_factor = _ORIGINAL_CLOCK_FACTOR**2

    def means(row):
        return tuple(sum(row[start : start + 9], I(0)) / 9 for start in range(0, 45, 9))

    return _OrganizedSourceAcceleration(
        state=state,
        gradient_tau_rate_bounds=tuple(dq),
        phase_current_tau_rate_bounds=tuple(ds),
        form_tau_acceleration_bounds=ddx,
        phase_tau_acceleration_bounds=ddtheta,
        form_slow_acceleration_bounds=tuple(slow_factor * value for value in ddx),
        phase_slow_acceleration_bounds=tuple(slow_factor * value for value in ddtheta),
        form_original_acceleration_bounds=tuple(
            original_factor * value for value in ddx
        ),
        phase_original_acceleration_bounds=tuple(
            original_factor * value for value in ddtheta
        ),
        child_form_tau_acceleration_bounds=means(ddx),
        child_phase_tau_acceleration_bounds=means(ddtheta),
    )


@dataclass(frozen=True)
class _OrganizedSourceReflectionState:
    """Exact coordinates and full real lift of the declared reflection sector."""

    form_coordinates: tuple[Q, ...]
    phase_coordinates: tuple[Q, ...]
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]


def _lift_organized_source_reflection(*, form_coordinates, phase_coordinates):
    """Lift twenty forms and twenty deviations from the fixed offset origin.

    Coordinate order is child-major, then slots (0, 2, 3, 4). This chart
    retains forty independent real values, not twenty bare NFR nodes.
    Sector admission alone supplies no winding or operator-history evidence.
    """
    values = _vector(form_coordinates, "form_coordinates", 20)
    deviations = _vector(phase_coordinates, "phase_coordinates", 20)
    form, offsets = [Q(0)] * 45, [_REFLECTION_ORIGIN] * 45
    for c in range(5):
        for k, (a, b) in enumerate(_REFLECTION_PAIRS):
            value, deviation = values[4 * c + k], deviations[4 * c + k]
            form[9 * c + a], form[9 * c + b] = value, -value
            offsets[9 * c + a] = _REFLECTION_ORIGIN + deviation
            offsets[9 * c + b] = _REFLECTION_ORIGIN - deviation
    return _OrganizedSourceReflectionState(
        form_coordinates=values,
        phase_coordinates=deviations,
        form=tuple(form),
        phase_offsets=tuple(offsets),
    )


def _restrict_organized_source_reflection(*, form, phase_offsets):
    """Restrict an admitted fine state; reject rather than project off-sector data.

    Real offsets belong to the declared continuous lift. Circularly equivalent
    representatives are not silently wrapped into it. Admission precedes every
    equality check, and no cached report or verdict is consumed.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    for c in range(5):
        fixed = 9 * c + _REFLECTION_FIXED_SLOT
        if values[fixed] != 0 or offsets[fixed] != _REFLECTION_ORIGIN:
            raise ValueError("state is outside the fixed-origin reflection sector")
        for a, b in _REFLECTION_PAIRS:
            a, b = 9 * c + a, 9 * c + b
            if (
                values[a] != -values[b]
                or offsets[a] + offsets[b] != 2 * _REFLECTION_ORIGIN
            ):
                raise ValueError("state is outside the fixed-origin reflection sector")
    return _lift_organized_source_reflection(
        form_coordinates=tuple(values[i] for i in _REFLECTION_NODES),
        phase_coordinates=tuple(
            offsets[i] - _REFLECTION_ORIGIN for i in _REFLECTION_NODES
        ),
    )


@dataclass(frozen=True)
class _OrganizedSourceReflectionObservation:
    """Exact-sector rows selected from the shared full-state law observation."""

    state: _OrganizedSourceReflectionState
    fine: _OrganizedSourceObservation
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    scope: tuple[str, ...] = (
        "exact_sufficient_coordinates_on_the_fixed_origin_signed_reflection_sector",
        "both_nonlinear_rows_inherited_from_the_unchanged_fine_law",
        "full_support_lift_clocks_currents_and_storage_retained_in_fine_observation",
        "constant_child_means_do_not_determine_fine_internal_response",
        "no_minimality_winding_retention_formation_or_physical_identity_claim",
        "no_operator_history_grammar_or_event_occurrence_supplied",
    )


def _observe_organized_source_reflection(*, form_coordinates, phase_coordinates):
    """Evaluate the exact conditional chart through the existing fine owner.

    The mathematical intertwining proof belongs to ORGANIZED_SOURCE_PORT.md;
    interval overlap alone is not its certificate. Optional acceleration is
    available separately from the raw full state and is not computed here.
    """
    state = _lift_organized_source_reflection(
        form_coordinates=form_coordinates, phase_coordinates=phase_coordinates
    )
    fine = _observe_organized_source_port(
        form=state.form, phase_offsets=state.phase_offsets
    )

    def select(row):
        return tuple(row[i] for i in _REFLECTION_NODES)

    return _OrganizedSourceReflectionObservation(
        state=state,
        fine=fine,
        form_tau_rate_bounds=select(fine.form_tau_rate_bounds),
        phase_tau_rate_bounds=select(fine.phase_tau_rate_bounds),
        form_slow_rate_bounds=select(fine.form_slow_rate_bounds),
        phase_slow_rate_bounds=select(fine.phase_slow_rate_bounds),
        form_original_rate_bounds=select(fine.form_original_rate_bounds),
        phase_original_rate_bounds=select(fine.phase_original_rate_bounds),
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


@dataclass(frozen=True)
class _OrganizedSourceEmission:
    """Necessary form reachability for one fixed nine-target Emission action.

    The source is rebuilt from primitive preparation allowances. Candidate
    bounds remain visible when preparation is unavailable, but then no actual
    envelope, target admission or realization exclusion is certified.
    """

    source: _OrganizedSourceAttachment
    target_nodes: tuple[int, ...]
    cycle_gap_lower_bound: Q
    collective_diameter: int
    collective_gap_lower_bound: Q
    form_laplacian_gap_lower_bound: Q
    sine_current_norm_squared_upper_bound: Q
    sine_current_norm_upper_bound: Q
    initial_form_norm_upper_bound: Q
    initial_target_form_absolute_upper_bound: Q
    forced_form_norm_upper_bound: Q | None
    all_time_form_norm_upper_bound: Q | None
    form_decay_tau_lower_bound: Q
    target_mean_displacement_upper_bound: Q | None
    emission_boost: Q
    nonclipping_margin: Q
    strict_form_margin: Q
    basal_capacity_margin: Q
    obstruction_margin: Q | None
    coarse_initial_form_norm_upper_bound: Q
    coarse_all_time_form_norm_upper_bound: Q
    coarse_mean_displacement_upper_bound: Q
    coarse_envelope_conditions: tuple[tuple[str, bool], ...]
    geometry_certified: bool
    source_certified: bool
    form_target_admitted: bool
    envelope_certified: bool
    realization_excluded: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: tuple[str, ...] = (
        "unchanged_actual_attached_five_C9_source_and_all_original_allowances",
        "full_forty_five_form_coordinates_not_a_rigid_carrier_replacement",
        "all_finite_durations_from_the_actual_attachment_state",
        "nine_host_child_zero_form_increments_by_the_resolved_default_AL_gain",
        "necessary_child_mean_condition_not_sufficient_full_operator_realization",
        "donor_reaction_and_all_other_coordinates_may_evolve_in_the_obstruction",
        "ideal_continuous_sine_law_not_native_runtime_or_represented_trajectory",
        "same_law_pressure_handoff_differs_from_the_raw_stored_pressure_AL_state",
        "no_operator_invocation_grammar_trace_or_autonomous_activation_is_invented",
        "smaller_emission_gains_other_sources_and_other_laws_remain_unresolved",
    )


def _source_emission_geometry(geometry):
    """Check the actual Cartesian support before using its gap certificate."""
    expected_cycles = {
        tuple(sorted((9 * child + slot, 9 * child + (slot + 1) % 9)))
        for child in range(5)
        for slot in range(9)
    }
    expected_matching = {
        (9 * a + slot, 9 * b + slot) for a, b in _COLLECTIVE_EDGES for slot in range(9)
    }
    edges = geometry.weighted_edges
    actual = {tuple(sorted((a, b))) for a, b, _ in edges}
    degrees = tuple(sum(node in (a, b) for a, b, _ in edges) for node in range(45))
    adjacency = [set() for _ in range(5)]
    for a, b in _COLLECTIVE_EDGES:
        adjacency[a].add(b)
        adjacency[b].add(a)
    diameter = 0
    for origin in range(5):
        distances = {origin: 0}
        frontier = [origin]
        for node in frontier:
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    frontier.append(neighbor)
        if len(distances) != 5:
            return False, 0, Q(0), Q(0), Q(0)
        diameter = max(diameter, max(distances.values()))
    pi = pi_interval()
    alpha, gamma = 2 * pi / 9, Q(1, 1023) / pi
    cycle_gap = (2 - 2 * cos(alpha)).lo
    collective_gap = Q(4, 5 * diameter)
    square_sum = sum((Q(degree**2) for degree in degrees), Q(0))
    valid = (
        geometry.node_order == tuple(range(45))
        and geometry.held_capacities == (Q(1),) * 45
        and geometry.clock_factor == Q(1, 4)
        and geometry.alpha_bounds == alpha
        and geometry.gamma_bounds == gamma
        and geometry.collective_edges == _COLLECTIVE_EDGES
        and len(edges) == len(actual) == 90
        and all(weight == 1 for _, _, weight in edges)
        and actual == expected_cycles | expected_matching
        and geometry.support_degrees == degrees
        and geometry.weighted_degrees == tuple(map(Q, degrees))
        and diameter == 3
        and cycle_gap > Q(1, 3)
        and square_sum == 738
        and square_sum < 28**2
    )
    return valid, diameter, cycle_gap, collective_gap, square_sum


def _assess_organized_source_emission(
    *,
    host_form_error_bounds,
    host_phase_error_bounds,
    donor_form_error_bound,
    donor_phase_error_bound,
):
    """Re-admit the fixed source and bound every finite-duration AL form target.

    On the zero-form-mean leaf, Duhamel bounds the full form norm by the
    convex interpolation between its initial norm and gamma*28/gap. The
    factor 1/4 fixes the decay clock and cancels from the forced envelope.
    No incoming report, amplitude override, endpoint time or trajectory is
    accepted. The comparison concerns ideal addition of the represented
    default boost; floating addition is not claimed exact for every source.
    """
    from ..config.thresholds import EPI_LATENT_MAX, VF_BASAL_THRESHOLD
    from ..constants import DEFAULTS
    from ..operators import _operator_epi_clip_policy
    from ..operators.factor_contracts import resolve_operator_factors

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
    geometry, host = source.geometry, source.host
    geometry_ok, diameter, cycle_gap, collective_gap, square_sum = (
        _source_emission_geometry(geometry)
    )
    gap = min(Q(1, 3), collective_gap) if geometry_ok else Q(0)
    gamma = geometry.gamma_bounds
    inner = host.nominal_inner_gap_at_entry_bounds.hi
    outer = host.nominal_outer_gap_at_entry_bounds.hi
    # Rebuild from the nested source proof, not summary contact-gap caches.
    nominal = gamma.hi * (1 + 2 * (gamma**2).hi) * (inner + outer) / 2
    initial = 6 * nominal + host.full_form_error_upper_bound + _RECOVERED_RADIUS
    target_initial = nominal + host.full_form_error_upper_bound
    donor = source.donor
    donor_ok = (
        donor.recovery_certified
        and donor.endpoint_form_norm_squared_upper_bound is not None
        and donor.endpoint_phase_norm_squared_upper_bound is not None
        and donor.endpoint_form_norm_squared_upper_bound < _RECOVERED_RADIUS**2
        and donor.endpoint_phase_norm_squared_upper_bound < _RECOVERED_RADIUS**2
    )
    source_ok = (
        source.admitted
        and source.sources_certified
        and host.entry_certified
        and host.child_identities_retained
        and donor_ok
        and source.conserved_form_total == 0
    )
    current_norm = Q(28)
    forced = gamma.hi * current_norm / gap if gap > 0 else None
    all_time = max(initial, forced) if forced is not None else None
    displacement = (all_time + initial) / 3 if all_time is not None else None
    boost = _exact(resolve_operator_factors(None, "AL")["AL_boost"], "AL_boost")
    lower, upper, mode, _ = _operator_epi_clip_policy(DEFAULTS)
    nonclip = min(-target_initial - Q(lower), Q(upper) - target_initial - boost)
    strict_form = _exact(EPI_LATENT_MAX, "EPI_LATENT_MAX") - target_initial
    basal = Q(1) - _exact(VF_BASAL_THRESHOLD, "VF_BASAL_THRESHOLD")
    target_ok = (
        mode == "hard" and boost > 0 and nonclip > 0 and strict_form > 0 and basal >= 0
    )
    margin = boost - displacement if displacement is not None else None
    coarse_initial, coarse_all_time = Q(1, 50000), Q(7, 200)
    coarse_displacement = (coarse_all_time + coarse_initial) / 3
    conditions = (
        ("gamma_below_one_over_3000", 0 < gamma.lo <= gamma.hi < Q(1, 3000)),
        (
            "outward_gamma_square_below_one_over_ten_million",
            (gamma**2).hi < Q(1, 10**7),
        ),
        ("actual_source_norm_below_one_over_50000", initial < coarse_initial),
        (
            "forced_form_norm_below_seven_over_200",
            forced is not None and forced < coarse_all_time,
        ),
        ("default_boost_above_one_sixteenth", boost > Q(1, 16)),
        ("coarse_mean_bound_below_one_eightieth", coarse_displacement < Q(1, 80)),
    )
    envelope = source_ok and geometry_ok and all(passed for _, passed in conditions)
    excluded = envelope and target_ok and margin is not None and margin > 0
    unresolved = tuple(
        label
        for passed, label in (
            (source_ok, "actual_original_source_not_certified"),
            (geometry_ok, "actual_product_support_or_complete_law_not_certified"),
            (
                all(passed for _, passed in conditions),
                "coarse_form_envelope_not_certified",
            ),
            (target_ok, "default_emission_form_target_not_admitted"),
            (
                margin is not None and margin > 0,
                "default_emission_not_excluded_by_mean_bound",
            ),
        )
        if not passed
    )
    return _OrganizedSourceEmission(
        source=source,
        target_nodes=tuple(range(9)),
        cycle_gap_lower_bound=cycle_gap,
        collective_diameter=diameter,
        collective_gap_lower_bound=collective_gap,
        form_laplacian_gap_lower_bound=gap,
        sine_current_norm_squared_upper_bound=square_sum,
        sine_current_norm_upper_bound=current_norm,
        initial_form_norm_upper_bound=initial,
        initial_target_form_absolute_upper_bound=target_initial,
        forced_form_norm_upper_bound=forced,
        all_time_form_norm_upper_bound=all_time,
        form_decay_tau_lower_bound=geometry.clock_factor * gap,
        target_mean_displacement_upper_bound=displacement,
        emission_boost=boost,
        nonclipping_margin=nonclip,
        strict_form_margin=strict_form,
        basal_capacity_margin=basal,
        obstruction_margin=margin,
        coarse_initial_form_norm_upper_bound=coarse_initial,
        coarse_all_time_form_norm_upper_bound=coarse_all_time,
        coarse_mean_displacement_upper_bound=coarse_displacement,
        coarse_envelope_conditions=conditions,
        geometry_certified=geometry_ok,
        source_certified=source_ok,
        form_target_admitted=source_ok and geometry_ok and target_ok,
        envelope_certified=envelope,
        realization_excluded=excluded,
        status="excluded_organized_source_emission" if excluded else "unavailable",
        unresolved_conditions=unresolved,
    )
