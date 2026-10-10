"""Detached finite external interaction of an acquired, observed organization.

ORGANIZED_EXTERNAL_INTERACTION.md owns the frozen contact and source history.
All fine coordinates, supplied registration, observation backaction and contact
work remain live premises. No trajectory, event selector or source reset runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._organized_outer_observation import (
    _FINE_PHASE_BOUND,
    _FORM_BOUND,
    _OUTER_ANGLE_BOUND,
)
from ._organized_outer_winding import (
    _OrganizedOuterWinding,
    _assess_organized_outer_winding,
    _derive_organized_outer_winding,
)
from ._organized_sequential_observation import (
    _OrganizedSequentialObservationProtocol,
    _assess_organized_sequential_observation_protocol,
)
from ._repeated_organized_composition import _bounded_sine_edge_fields
from ._sine_constitutive_recovery import (
    _ConstitutiveRecovery,
    _assess_constitutive_recovery,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from ._sine_preparation import _prepared_form_costs

_SIZE = 54
_ALLOWANCE = Q(1, 10**10)
_Q = Q(1, 10**70)
_HORIZON = Q(1, 1000)
_READOUT_ERROR = Q(1, 10**10)
_RESOLUTION = Q(1, 10**7)
_ORIGINAL_CLOCK_FACTOR = Q(1023, 1024)


@dataclass(frozen=True)
class _OrganizedExternalInteractionGeometry:
    internal_edges: tuple[tuple[int, int, Q], ...]
    outer_edges: tuple[tuple[int, int, Q], ...]
    contact_edges: tuple[tuple[int, int, Q], ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    internal_cycles: tuple[tuple[int, ...], ...]
    outer_cycles: tuple[tuple[int, ...], ...]
    alpha_bounds: I
    gamma_bounds: I
    external_phase_origin_bounds: I
    node_order: tuple[int, ...] = tuple(range(_SIZE))
    support_degrees: tuple[int, ...] = (5,) * 45 + (7,) * 9
    weighted_degrees: tuple[Q, ...] = (Q(5),) * 45 + (Q(7),) * 9
    held_capacities: tuple[Q, ...] = (Q(1),) * _SIZE
    clock_factor: Q = Q(1, 4)
    clock: str = "tau=(1023/1024)*t; both complete rows use the uniform quarter factor"
    phase_domain: str = "regular"
    scope: tuple[str, ...] = (
        "unit_C9_cartesian_wheel_W6_with_forty_five_supplied_external_contacts",
        "six_form_phase_pairs_on_the_carrier_not_one_scalar_composite_state",
        "all_fifty_four_forms_and_phases_retained_off_the_carrier",
        "external_pi_over_two_origin_supplied_from_primitive_preparation",
        "geometry_installs_no_support_event_or_occurrence_law",
    )


def _derive_organized_external_interaction():
    """Build the fixed contact support without a source or execution verdict."""
    host = _derive_organized_outer_winding()
    internal = host.internal_edges + tuple(
        (45 + j, 45 + (j + 1) % 9, Q(1)) for j in range(9)
    )
    contact = tuple((9 * c + j, 45 + j, Q(1)) for c in range(5) for j in range(9))
    return _OrganizedExternalInteractionGeometry(
        internal_edges=internal,
        outer_edges=host.outer_edges,
        contact_edges=contact,
        weighted_edges=internal + host.outer_edges + contact,
        internal_cycles=host.internal_cycles + (tuple(range(45, 54)),),
        outer_cycles=host.outer_cycles,
        alpha_bounds=host.alpha_bounds,
        gamma_bounds=host.gamma_bounds,
        external_phase_origin_bounds=pi_interval() / 2,
    )


@dataclass(frozen=True)
class _OrganizedExternalInteractionObservation:
    geometry: _OrganizedExternalInteractionGeometry
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    phase_lift_bounds: tuple[I, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    child_form_means: tuple[Q, ...]
    # Raw offset means omit the supplied donor origin (pi/2 in the original
    # protocol); these are not the complete lifted phase means of the children.
    child_phase_offset_means: tuple[Q, ...]
    child_form_tau_rate_bounds: tuple[I, ...]
    child_phase_tau_rate_bounds: tuple[I, ...]
    contact_form_gradient: tuple[Q, ...]
    contact_phase_current_bounds: tuple[I, ...]
    contact_form_tau_rate_bounds: tuple[I, ...]
    contact_phase_tau_rate_bounds: tuple[I, ...]
    contact_child_form_tau_rate_bounds: tuple[I, ...]
    contact_child_phase_tau_rate_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    attachment_work_bounds: I
    storage_tau_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    scope: tuple[str, ...] = (
        "admitted_full_detached_state_without_acquisition_or_mean_leaf_assertion",
        "raw_rational_offsets_relative_to_carrier_and_symbolic_external_pi_over_two",
        "complete_contact_rows_and_equal_opposite_boundary_reaction",
        "instantaneous_storage_and_rates_do_not_certify_future_identity",
    )


def _observe_organized_external_interaction(*, form, phase_offsets):
    """Observe the full joined law after admitting all 108 primitive scalars."""
    values = _vector(form, "form", _SIZE)
    offsets = _vector(phase_offsets, "phase_offsets", _SIZE)
    geometry = _derive_organized_external_interaction()
    return _observe_admitted_external_interaction(values, offsets, geometry)


def _observe_admitted_external_interaction(values, offsets, geometry):
    """Shared full-state field kernel for caller-admitted contact preparations.

    The calling observer owns primitive admission and freshly derived support
    and origin. This kernel neither authenticates incoming reports nor asserts
    source acquisition, identity retention or future response.
    """
    lifted_offsets = offsets[:45] + tuple(
        value + geometry.external_phase_origin_bounds for value in offsets[45:]
    )
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    current, potential, _ = _bounded_sine_edge_fields(
        phase_offsets=lifted_offsets,
        weighted_edges=geometry.weighted_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    boundary_costs, boundary_gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.contact_edges
    )
    boundary_current, boundary_potential, _ = _bounded_sine_edge_fields(
        phase_offsets=lifted_offsets,
        weighted_edges=geometry.contact_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    gamma, k = geometry.gamma_bounds, geometry.clock_factor

    def rows(gradients, currents):
        return (
            tuple(k * (-g + gamma * s) for g, s in zip(gradients, currents)),
            tuple(k * gamma * g for g in gradients),
        )

    def means(row, zero):
        return tuple(sum(row[9 * c : 9 * c + 9], zero) / 9 for c in range(6))

    dx, dt = rows(gradient, current)
    boundary_dx, boundary_dt = rows(boundary_gradient, boundary_current)
    loss = k * sum((g**2 for g in gradient), Q(0))
    direct = sum((g * x - s * t for g, s, x, t in zip(gradient, current, dx, dt)), I(0))
    form_storage = sum(costs, Q(0))
    phase_storage = sum(potential, I(0))
    return _OrganizedExternalInteractionObservation(
        geometry=geometry,
        form=values,
        phase_offsets=offsets,
        phase_lift_bounds=tuple(
            geometry.alpha_bounds * (node % 9 - 4) + value
            for node, value in enumerate(lifted_offsets)
        ),
        form_gradient=gradient,
        phase_current_bounds=current,
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dt,
        form_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * a for a in dx),
        phase_original_rate_bounds=tuple(_ORIGINAL_CLOCK_FACTOR * a for a in dt),
        child_form_means=means(values, Q(0)),
        child_phase_offset_means=means(offsets, Q(0)),
        child_form_tau_rate_bounds=means(dx, I(0)),
        child_phase_tau_rate_bounds=means(dt, I(0)),
        contact_form_gradient=boundary_gradient,
        contact_phase_current_bounds=boundary_current,
        contact_form_tau_rate_bounds=boundary_dx,
        contact_phase_tau_rate_bounds=boundary_dt,
        contact_child_form_tau_rate_bounds=means(boundary_dx, I(0)),
        contact_child_phase_tau_rate_bounds=means(boundary_dt, I(0)),
        form_storage=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=form_storage + phase_storage,
        attachment_work_bounds=sum(boundary_costs, Q(0))
        + sum(boundary_potential, I(0)),
        storage_tau_rate=-loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
    )


@dataclass(frozen=True)
class _OrganizedExternalInteraction:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    geometry: _OrganizedExternalInteractionGeometry
    host_acquisition: _OrganizedOuterWinding
    donor_recovery: _ConstitutiveRecovery
    observation_protocol: _OrganizedSequentialObservationProtocol
    relative_preparation_dimension: int
    source_inclusion_margins: tuple[tuple[str, Q], ...]
    source_admission_flags: tuple[bool, ...]
    observation_form_upper_bound: Q
    carried_phase_drift_upper_bound: Q
    outer_one_host_phase_offset_bounds: tuple[I, ...]
    outer_zero_host_phase_offset_bounds: tuple[I, ...]
    donor_phase_offset_bounds: I
    outer_one_boundary_moment_bounds: I
    outer_zero_boundary_moment_bounds: I
    joined_form_upper_bound: Q
    contact_edge_drift_upper_bound: Q
    finite_response_remainder_upper_bound: Q
    finite_increment_bounds: tuple[I, I]
    observed_increment_bounds: tuple[I, I]
    contrast_bounds: I
    observed_contrast_bounds: I
    resolution_margin: Q
    contact_work_upper_bound: Q
    joined_initial_storage_upper_bounds: tuple[Q, Q]
    identity_margins: tuple[tuple[str, Q], ...]
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    work_certified: bool
    identity_certified: bool
    comparison_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    elapsed_scaled_time: Q = _HORIZON
    elapsed_original_time: Q = _HORIZON / _ORIGINAL_CLOCK_FACTOR
    readout_error_upper_bound: Q = _READOUT_ERROR
    resolution_threshold: Q = _RESOLUTION
    attachment_work_budget: Q = Q(100)
    joined_storage_budget: Q = Q(200)
    donor_formation_time: Q = Q(200)
    donor_recovery_duration: Q = Q(2 * 10**13)
    family_order: tuple[str, str] = ("outer_one", "outer_zero")
    protocol_revision: str = "7edac2ab"
    scope: tuple[str, ...] = (
        "actual_acquisition_and_complete_single_history_observation_before_contact",
        "six_independent_original_sources_and_all_ninety_six_relative_directions",
        "external_origin_and_synchronized_primitive_preparation_supplied",
        "zero_form_totals_retained_as_correlations_not_independent_node_boxes",
        "one_fixed_finite_external_mean_response_with_four_bounded_raw_errors",
        "all_six_internal_and_nine_outer_identities_only_through_the_contact_horizon",
        "contact_work_separate_from_acquisition_observation_work_and_continuous_loss",
        "no_winding_only_closure_topological_charge_autonomous_contact_or_physical_identity",
        "no_trajectory_frozen_producer_or_protocol_scan",
    )


def _assess_organized_external_interaction(*, form_error_bounds, phase_error_bounds):
    """Rebuild the fixed contact proof from twelve original source allowances."""
    rx = _vector(form_error_bounds, "form_error_bounds", 6)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 6)
    if min(*rx, *rt) < 0 or max(*rx, *rt) > _ALLOWANCE:
        raise ValueError("original errors must belong to frozen [0, 1e-10]")
    host = _assess_organized_outer_winding(
        form_error_bounds=rx[:5], phase_error_bounds=rt[:5]
    )
    donor = _assess_constitutive_recovery(
        deformation=0, form_error_bound=rx[5], phase_error_bound=rt[5]
    )
    observation = _assess_organized_sequential_observation_protocol()
    geometry = _derive_organized_external_interaction()
    pi, gamma = pi_interval(), geometry.gamma_bounds
    g, h, r, z = (
        gamma.hi,
        _HORIZON,
        host.full_form_error_upper_bound,
        host.full_phase_error_upper_bound,
    )
    one_form = max(box.abs_max for box in host.primitive_entry.endpoint_form_bounds) + r
    one_gap = (
        max(box.abs_max for box in host.endpoint_oriented_outer_phase_gap_bounds)
        + 2 * z
    )
    inclusion = (
        ("outer_one_form", _FORM_BOUND - one_form),
        ("outer_zero_form", _FORM_BOUND - r),
        ("reprojected_fine_phase", _FINE_PHASE_BOUND - 2 * z),
        ("outer_one_unknown_origin_gap", _OUTER_ANGLE_BOUND - one_gap),
        ("outer_zero_unknown_origin_gap", _OUTER_ANGLE_BOUND - 2 * z),
    )
    donor_ok = (
        donor.recovery_certified
        and donor.endpoint_form_norm_squared_upper_bound is not None
        and donor.endpoint_phase_norm_squared_upper_bound is not None
        and donor.endpoint_form_norm_squared_upper_bound < _Q**2
        and donor.endpoint_phase_norm_squared_upper_bound < _Q**2
    )
    flags = (
        host.admitted,
        host.control.capture_certified,
        donor_ok,
        observation.admitted,
        all(value > 0 for _, value in inclusion),
    )
    source_ok = all(flags)
    b_obs = observation.observation_form_upper_bound
    carried = 2 * g * b_obs * observation.total_elapsed_scaled_time
    one_phase = tuple(
        box + I(-z - carried, z + carried)
        for box in host.primitive_entry.endpoint_phase_bounds
    )
    zero_phase = (I(-z - carried, z + carried),) * 5
    donor_phase = pi / 2 + I(-_Q, _Q)
    # The exact pi/2 translation gives sin(pi/2-u)=cos(u). Each
    # interval includes the actual observed host and independent donor error.
    one_moment = sum((cos(box + I(-_Q, _Q)) for box in one_phase), I(0))
    zero_moment = sum((cos(box + I(-_Q, _Q)) for box in zero_phase), I(0))
    b = b_obs + Q(7, 4) * g * h
    contact_drift = 7 * g * b * h
    remainder = Q(15, 16) * g * h**2 + Q(15, 4) * g**2 * b * h**2
    increments = tuple(
        -(gamma * h / 4) * moment + I(-remainder, remainder)
        for moment in (one_moment, zero_moment)
    )
    observed = tuple(
        box + I(-2 * _READOUT_ERROR, 2 * _READOUT_ERROR) for box in increments
    )
    contrast = increments[0] - increments[1]
    observed_contrast = contrast + I(-4 * _READOUT_ERROR, 4 * _READOUT_ERROR)
    resolution = observed_contrast.lo - _RESOLUTION
    work = 45 * ((b_obs + _Q) ** 2 / 2 + 2)
    donor_storage = host.internal_cycle_minimum_bounds.hi + 4 * _Q**2
    host_storage = (
        host.target_storage_bounds.hi + host.endpoint_excess_upper_bound,
        5 * host.internal_cycle_minimum_bounds.hi + 40 * _Q**2,
    )
    joined_storage = tuple(
        value + observation.total_absolute_event_work_upper_bound + donor_storage + work
        for value in host_storage
    )
    d_obs = observation.continuous_edge_drift_upper_bound
    alpha = geometry.alpha_bounds
    host_positive = alpha.lo - 2 * _FINE_PHASE_BOUND - d_obs - contact_drift
    host_acute = pi.lo / 2 - alpha.hi - 2 * _FINE_PHASE_BOUND - d_obs - contact_drift
    donor_positive = alpha.lo - 2 * _Q - contact_drift
    donor_acute = pi.lo / 2 - alpha.hi - 2 * _Q - contact_drift
    outer_acute = (
        pi.lo / 2 - _OUTER_ANGLE_BOUND - 2 * _FINE_PHASE_BOUND - d_obs - contact_drift
    )
    identity = tuple(
        entry
        for c in range(6)
        for entry in (
            (
                f"internal_positive_gap[{c}]",
                donor_positive if c == 5 else host_positive,
            ),
            (f"internal_acute_gap[{c}]", donor_acute if c == 5 else host_acute),
        )
    ) + tuple((f"outer_acute_gap[{j}]", outer_acute) for j in range(9))
    work_margins = (
        ("attachment_work", Q(100) - work),
        ("outer_one_joined_storage", Q(200) - joined_storage[0]),
        ("outer_zero_joined_storage", Q(200) - joined_storage[1]),
    )
    identity_ok = all(value > 0 for _, value in identity)
    work_ok = all(value > 0 for _, value in work_margins)
    comparison_ok = resolution > 0
    margins = (
        inclusion
        + identity
        + work_margins
        + (("observed_contrast_resolution", resolution),)
    )
    flag_names = (
        "host_acquisition_not_certified",
        "paired_host_control_not_certified",
        "external_recovery_not_certified",
        "preceding_sequential_observation_not_certified",
        "actual_hosts_outside_observation_class",
    )
    unresolved = tuple(name for name, ok in zip(flag_names, flags) if not ok) + tuple(
        name for name, value in margins if value <= 0
    )
    admitted = source_ok and identity_ok and work_ok and comparison_ok
    return _OrganizedExternalInteraction(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        geometry=geometry,
        host_acquisition=host,
        donor_recovery=donor,
        observation_protocol=observation,
        relative_preparation_dimension=host.relative_preparation_dimension
        + donor.source.preparation_dimension,
        source_inclusion_margins=inclusion,
        source_admission_flags=flags,
        observation_form_upper_bound=b_obs,
        carried_phase_drift_upper_bound=carried,
        outer_one_host_phase_offset_bounds=one_phase,
        outer_zero_host_phase_offset_bounds=zero_phase,
        donor_phase_offset_bounds=donor_phase,
        outer_one_boundary_moment_bounds=one_moment,
        outer_zero_boundary_moment_bounds=zero_moment,
        joined_form_upper_bound=b,
        contact_edge_drift_upper_bound=contact_drift,
        finite_response_remainder_upper_bound=remainder,
        finite_increment_bounds=increments,
        observed_increment_bounds=observed,
        contrast_bounds=contrast,
        observed_contrast_bounds=observed_contrast,
        resolution_margin=resolution,
        contact_work_upper_bound=work,
        joined_initial_storage_upper_bounds=joined_storage,
        identity_margins=identity,
        sufficient_margins=margins,
        source_certified=source_ok,
        work_certified=source_ok and work_ok,
        identity_certified=source_ok and identity_ok,
        comparison_certified=source_ok and comparison_ok,
        admitted=admitted,
        status=(
            "certified_organized_external_interaction" if admitted else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
