"""Own-law finite contact of actual acquired and intervened organizations.

The fixed lambda-one law retains its own source, event work, complete contact
reaction and finite remainder. Conditional bounds accept primitive residual
allowances, never cached reports; the actual assessor rebuilds all sources.
No trajectory, registration scan or earlier sine response is evaluated.
Proof: theory/nodal/ORGANIZED_ALTERNATIVE_CONTACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._organized_alternative_outer_acquisition import (
    _AlternativeOuterAcquisition,
    _AlternativeOuterBounds,
    _assess_organized_alternative_outer_acquisition,
    _derive_organized_alternative_outer_bounds,
)
from ._organized_external_interaction import (
    _OrganizedExternalInteractionGeometry,
    _derive_organized_external_interaction,
)
from ._sine_class_constitutive_comparison import (
    _current_derivative_bound,
    _current_fourier_coefficients,
    _edge_law,
)
from ._sine_constitutive_recovery import (
    _ConstitutiveRecovery,
    _assess_constitutive_recovery,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from ._sine_preparation import _prepared_form_costs

_ALLOWANCE = Q(1, 10**10)
_RECOVERED = Q(1, 10**70)
_INITIAL_FORM = Q(1, 100)
_WORD_TIME = Q(1, 5000)
_WINDOW = Q(1, 100000)
_AMPLITUDE = Q(1, 10)
_READOUT_ERROR = Q(1, 10**10)
_HORIZON = Q(1, 1000)
_ONE_THRESHOLD = Q(1, 10**8)
_ZERO_THRESHOLD = Q(3, 10**7)
_CLOCK = Q(1023, 1024)
_HARMONICS = _current_fourier_coefficients(Q(1))


def _geometry(origin: I, *, supplied: bool) -> _OrganizedExternalInteractionGeometry:
    original = _derive_organized_external_interaction()
    return replace(
        original,
        external_phase_origin_bounds=origin,
        scope=tuple(
            item
            for item in original.scope
            if item != "external_pi_over_two_origin_supplied_from_primitive_preparation"
        )
        + (
            (
                "external_origin_supplied_from_primitive_preparation"
                if supplied
                else "all_external_origins_in_circle_are_separately_supplied_preparations"
            ),
        ),
    )


@dataclass(frozen=True)
class _AlternativeContactObservation:
    relative_phase_origin: Q
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
    child_form_tau_rate_bounds: tuple[I, ...]
    child_phase_tau_rate_bounds: tuple[I, ...]
    contact_form_gradient: tuple[Q, ...]
    contact_phase_current_bounds: tuple[I, ...]
    contact_child_form_tau_rate_bounds: tuple[I, ...]
    contact_child_phase_tau_rate_bounds: tuple[I, ...]
    harmonic_moment_bounds: tuple[tuple[int, I, I], ...]
    contact_mean_current_bounds: I
    contact_current_from_moments_bounds: I
    current_reconstruction_residual_bounds: I
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    attachment_work_bounds: I
    storage_tau_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    deformation: Q = Q(1)
    scope: tuple[str, ...] = (
        "supplied_full_detached_state_and_relative_origin_not_reached_source",
        "all_144_edges_and_uniform_quarter_factor_in_both_complete_rows",
        "same_slot_harmonic_moments_exclude_supplied_origin_and_retain_associations",
        "moments_supply_only_the_instantaneous_contact_current_not_closed_dynamics",
        "no_source_event_occurrence_response_or_future_identity_certificate",
    )


def _observe_organized_alternative_contact(
    *, relative_phase_origin: object, form: object, phase_offsets: object
) -> _AlternativeContactObservation:
    """Re-admit all 109 state/origin scalars before observing the actual law."""
    beta = _exact(relative_phase_origin, "relative_phase_origin")
    values = _vector(form, "form", 54)
    offsets = _vector(phase_offsets, "phase_offsets", 54)
    geometry = _geometry(I(beta), supplied=True)
    lifted = tuple(
        geometry.alpha_bounds * (node % 9 - 4) + offset + (beta if node >= 45 else 0)
        for node, offset in enumerate(offsets)
    )
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    boundary_costs, boundary_gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.contact_edges
    )
    current, boundary_current = [I(0) for _ in values], [I(0) for _ in values]
    potentials, boundary_potentials = [], []
    boundary_start = len(geometry.internal_edges) + len(geometry.outer_edges)
    for index, (a, b, weight) in enumerate(geometry.weighted_edges):
        angle = lifted[b] - lifted[a]
        potential, derivatives = _edge_law(cos(angle), sin(angle), Q(1))
        edge_current = weight * derivatives[0]
        current[a] += edge_current
        current[b] -= edge_current
        potentials.append(weight * potential)
        if index >= boundary_start:
            boundary_current[a] += edge_current
            boundary_current[b] -= edge_current
            boundary_potentials.append(weight * potential)
    gamma, quarter = geometry.gamma_bounds, geometry.clock_factor

    def means(row: tuple | list, zero: Q | I) -> tuple:
        return tuple(sum(row[9 * c : 9 * c + 9], zero) / 9 for c in range(6))

    dx = tuple(quarter * (-g + gamma * s) for g, s in zip(gradient, current))
    dt = tuple(quarter * gamma * g for g in gradient)
    boundary_dx = tuple(
        quarter * (-g + gamma * s) for g, s in zip(boundary_gradient, boundary_current)
    )
    boundary_dt = tuple(quarter * gamma * g for g in boundary_gradient)
    moments, reconstructed = [], I(0)
    for harmonic, coefficient in _HARMONICS:
        angles = tuple(
            I(harmonic * (offsets[b] - offsets[a]))
            for a, b, _ in geometry.contact_edges
        )
        cosine = sum((cos(angle) for angle in angles), I(0)) / 9
        sine = sum((sin(angle) for angle in angles), I(0)) / 9
        moments.append((harmonic, cosine, sine))
        reconstructed += coefficient * (
            sin(I(harmonic * beta)) * cosine + cos(I(harmonic * beta)) * sine
        )
    contact_mean = -sum(boundary_current[45:], I(0)) / 9
    loss = quarter * sum((g**2 for g in gradient), Q(0))
    direct = sum((g * x - s * t for g, s, x, t in zip(gradient, current, dx, dt)), I(0))
    form_storage, phase_storage = sum(costs, Q(0)), sum(potentials, I(0))
    return _AlternativeContactObservation(
        relative_phase_origin=beta,
        geometry=geometry,
        form=values,
        phase_offsets=offsets,
        phase_lift_bounds=lifted,
        form_gradient=gradient,
        phase_current_bounds=tuple(current),
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dt,
        form_original_rate_bounds=tuple(_CLOCK * value for value in dx),
        phase_original_rate_bounds=tuple(_CLOCK * value for value in dt),
        child_form_means=means(values, Q(0)),
        child_form_tau_rate_bounds=means(dx, I(0)),
        child_phase_tau_rate_bounds=means(dt, I(0)),
        contact_form_gradient=boundary_gradient,
        contact_phase_current_bounds=tuple(boundary_current),
        contact_child_form_tau_rate_bounds=means(boundary_dx, I(0)),
        contact_child_phase_tau_rate_bounds=means(boundary_dt, I(0)),
        harmonic_moment_bounds=tuple(moments),
        contact_mean_current_bounds=contact_mean,
        contact_current_from_moments_bounds=reconstructed,
        current_reconstruction_residual_bounds=contact_mean - reconstructed,
        form_storage=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=form_storage + phase_storage,
        attachment_work_bounds=sum(boundary_costs, Q(0))
        + sum(boundary_potentials, I(0)),
        storage_tau_rate=-loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
    )


@dataclass(frozen=True)
class _AlternativeContactBounds:
    """Conditional resources rebuilt from a primitive recovered-state allowance."""

    geometry: _OrganizedExternalInteractionGeometry
    host_bounds: _AlternativeOuterBounds
    recovered_norm_bound: Q
    current_absolute_upper_bound: Q
    current_lipschitz_upper_bound: Q
    source_form_upper_bounds: tuple[Q, Q]
    source_outer_gap_upper_bounds: tuple[Q, Q]
    source_inclusion_margins: tuple[tuple[str, Q], ...]
    source_class_certified: bool
    observation_form_upper_bound: Q
    carried_phase_drift_upper_bound: Q
    word_edge_drift_upper_bound: Q
    per_block_absolute_event_work_upper_bounds: tuple[Q, ...]
    total_absolute_event_work_upper_bound: Q
    word_identity_margins: tuple[tuple[str, Q], ...]
    word_work_margins: tuple[tuple[str, Q], ...]
    word_identity_certified: bool
    word_work_certified: bool
    boundary_phase_error_upper_bound: Q
    nominal_host_phase_offset_bounds: tuple[tuple[I, ...], ...]
    fundamental_cosine_bounds: tuple[I, I]
    fundamental_sine_bounds: tuple[I, I]
    fundamental_amplitude_bounds: tuple[I, I]
    higher_harmonic_current_upper_bound: Q
    joined_form_upper_bound: Q
    contact_edge_drift_upper_bound: Q
    finite_response_remainder_upper_bound: Q
    total_increment_error_upper_bound: Q
    observed_capability_bounds: tuple[I, I]
    outer_one_upper_margin: Q
    outer_zero_lower_margin: Q
    paired_absolute_response_gap_lower_bound: Q
    contact_work_upper_bound: Q
    donor_storage_upper_bound: Q
    host_preword_storage_upper_bounds: tuple[Q, Q]
    joined_initial_storage_upper_bounds: tuple[Q, Q]
    contact_work_margins: tuple[tuple[str, Q], ...]
    contact_identity_margins: tuple[tuple[str, Q], ...]
    response_margins: tuple[tuple[str, Q], ...]
    sufficient_margins: tuple[tuple[str, Q], ...]
    contact_work_certified: bool
    contact_identity_certified: bool
    response_certified: bool
    conditional_admitted: bool
    arithmetic_method: str = INTERVAL_METHOD


def _derive_organized_alternative_contact_bounds(
    *, recovered_radius: object
) -> _AlternativeContactBounds:
    """Rebuild actual-law source bounds and all finite contact estimates.

    The radius is an admitted Euclidean bound on each recovered constituent.
    This arithmetic does not certify acquisition or consume incoming reports.
    The fixed method uses the fundamental rectangle and a uniform 9/18 tail;
    it performs no beta scan or response integration.
    """
    q = _exact(recovered_radius, "recovered_radius")
    if q < 0:
        raise ValueError("recovered_radius must be nonnegative")
    host = _derive_organized_alternative_outer_bounds(recovered_radius=q)
    pi, gamma = pi_interval(), host.geometry.gamma_bounds
    geometry = _geometry(I(-pi.hi, pi.hi), supplied=False)
    g, j, k = (
        gamma.hi,
        _current_derivative_bound(Q(1), 0),
        _current_derivative_bound(Q(1), 1),
    )
    r, z = host.full_form_error_upper_bound, host.full_phase_error_upper_bound
    source_forms = (host.primitive_node_form_radius_upper_bound + r, r)
    source_gaps = (
        max(gap.abs_max for gap in host.endpoint_oriented_outer_phase_gap_bounds),
        2 * z,
    )
    inclusion = (
        ("outer_one_initial_form", _INITIAL_FORM - source_forms[0]),
        ("outer_zero_initial_form", _INITIAL_FORM - source_forms[1]),
        ("donor_initial_form", _INITIAL_FORM - q),
    )
    source_class = all(value > 0 for _, value in inclusion)
    bo = _INITIAL_FORM + g * j * _WORD_TIME
    carried = 2 * g * bo * _WORD_TIME
    word_drift = 2 * carried
    alpha = geometry.alpha_bounds
    word_identity = (
        ("word_internal_positive_gap", alpha.lo - 2 * z - word_drift),
        ("word_internal_acute_gap", pi.lo / 2 - alpha.hi - 2 * z - word_drift),
    ) + tuple(
        (f"word_{name}_outer_acute_gap", pi.lo / 2 - angle - _AMPLITUDE - word_drift)
        for name, angle in zip(("outer_one", "outer_zero"), source_gaps)
    )
    event_work = tuple(
        18 * j * abs(multiplier) * _AMPLITUDE for multiplier in (1, -2, 1)
    )
    total_work = 5 * sum(event_work, Q(0))
    word_work = (
        ("absolute_event_work", Q(4) - max(event_work)),
        ("total_absolute_word_work", Q(40) - total_work),
    )
    phase_error = z + carried + q
    node_phase = host.primitive_node_phase_radius_upper_bound
    one_phase = tuple(4 * (c - 2) / pi + I(-node_phase, node_phase) for c in range(5))
    phase_families = (one_phase, (I(0),) * 5)
    cosines = tuple(
        sum((cos(phase + I(-phase_error, phase_error)) for phase in phases), I(0))
        for phases in phase_families
    )
    sines = tuple(
        -sum((sin(phase + I(-phase_error, phase_error)) for phase in phases), I(0))
        for phases in phase_families
    )
    amplitudes = tuple(sqrt(c**2 + s**2) for c, s in zip(cosines, sines))
    tail = Q(5, 1458)
    b = bo + Q(7, 4) * g * j * _HORIZON
    contact_drift = 7 * g * b * _HORIZON
    remainder = Q(15, 16) * g * j * _HORIZON**2 + Q(15, 4) * g**2 * k * b * _HORIZON**2
    error = remainder + 2 * _READOUT_ERROR
    factor = gamma * _HORIZON / 4
    capability = tuple(
        I(
            max(Q(0), factor.lo * max(Q(0), amplitude.lo - tail) - error),
            factor.hi * (amplitude.hi + tail) + error,
        )
        for amplitude in amplitudes
    )
    one_margin = _ONE_THRESHOLD - capability[0].hi
    zero_margin = capability[1].lo - _ZERO_THRESHOLD
    work = 45 * ((bo + q) ** 2 / 2 + 2 + Q(2, 19683))
    donor_storage = host.internal_cycle_minimum_bounds.hi + 2 * (1 + k) * q**2
    host_storage = (
        host.target_storage_bounds.hi + host.endpoint_excess_upper_bound,
        5 * host.internal_cycle_minimum_bounds.hi
        + host.control_initial_excess_upper_bound,
    )
    joined_storage = tuple(
        value + total_work + donor_storage + work for value in host_storage
    )
    work_margins = (
        ("contact_attachment_work", Q(100) - work),
        ("outer_one_joined_storage", Q(200) - joined_storage[0]),
        ("outer_zero_joined_storage", Q(200) - joined_storage[1]),
    )
    identity = tuple(
        entry
        for child in range(6)
        for entry in (
            (
                f"contact_internal_positive_gap[{child}]",
                alpha.lo
                - 2 * (q if child == 5 else z)
                - (0 if child == 5 else word_drift)
                - contact_drift,
            ),
            (
                f"contact_internal_acute_gap[{child}]",
                pi.lo / 2
                - alpha.hi
                - 2 * (q if child == 5 else z)
                - (0 if child == 5 else word_drift)
                - contact_drift,
            ),
        )
    ) + tuple(
        (
            f"contact_{name}_outer_acute_gap[{slot}]",
            pi.lo / 2 - angle - word_drift - contact_drift,
        )
        for name, angle in zip(("outer_one", "outer_zero"), source_gaps)
        for slot in range(9)
    )
    response = (
        ("outer_one_all_origin_upper_bound", one_margin),
        ("outer_zero_per_source_witness_lower_bound", zero_margin),
    )
    margins = inclusion + word_identity + word_work + work_margins + identity + response
    flags = tuple(
        all(value > 0 for _, value in group)
        for group in (word_identity, word_work, work_margins, identity, response)
    )
    return _AlternativeContactBounds(
        geometry=geometry,
        host_bounds=host,
        recovered_norm_bound=q,
        current_absolute_upper_bound=j,
        current_lipschitz_upper_bound=k,
        source_form_upper_bounds=source_forms,
        source_outer_gap_upper_bounds=source_gaps,
        source_inclusion_margins=inclusion,
        source_class_certified=source_class,
        observation_form_upper_bound=bo,
        carried_phase_drift_upper_bound=carried,
        word_edge_drift_upper_bound=word_drift,
        per_block_absolute_event_work_upper_bounds=event_work,
        total_absolute_event_work_upper_bound=total_work,
        word_identity_margins=word_identity,
        word_work_margins=word_work,
        word_identity_certified=flags[0],
        word_work_certified=flags[1],
        boundary_phase_error_upper_bound=phase_error,
        nominal_host_phase_offset_bounds=phase_families,
        fundamental_cosine_bounds=cosines,
        fundamental_sine_bounds=sines,
        fundamental_amplitude_bounds=amplitudes,
        higher_harmonic_current_upper_bound=tail,
        joined_form_upper_bound=b,
        contact_edge_drift_upper_bound=contact_drift,
        finite_response_remainder_upper_bound=remainder,
        total_increment_error_upper_bound=error,
        observed_capability_bounds=capability,
        outer_one_upper_margin=one_margin,
        outer_zero_lower_margin=zero_margin,
        paired_absolute_response_gap_lower_bound=capability[1].lo - capability[0].hi,
        contact_work_upper_bound=work,
        donor_storage_upper_bound=donor_storage,
        host_preword_storage_upper_bounds=host_storage,
        joined_initial_storage_upper_bounds=joined_storage,
        contact_work_margins=work_margins,
        contact_identity_margins=identity,
        response_margins=response,
        sufficient_margins=margins,
        contact_work_certified=flags[2],
        contact_identity_certified=flags[3],
        response_certified=flags[4],
        conditional_admitted=source_class and all(flags),
    )


@dataclass(frozen=True)
class _AlternativeContact:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    host_acquisition: _AlternativeOuterAcquisition
    donor_recovery: _ConstitutiveRecovery
    resources: _AlternativeContactBounds
    relative_preparation_dimension: int
    source_admission_flags: tuple[bool, ...]
    source_certified: bool
    word_certified: bool
    work_certified: bool
    identity_certified: bool
    response_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    deformation: Q = Q(1)
    family_order: tuple[str, str] = ("outer_one", "outer_zero")
    protocol_revision: str = "e2b2e647"
    arithmetic_method: str = INTERVAL_METHOD
    pressure_refresh: str = "continuous_current_at_actual_state"
    event_policy: str = "fifteen_supplied_word_jumps_then_one_attachment_then_no_events"
    capacity: tuple[Q, ...] = (Q(1),) * 54
    clock: str = "tau=(1023/1024)*t; uniform quarter factor in both rows"
    source_host_time: Q = Q(200)
    donor_formation_time: Q = Q(200)
    donor_recovery_duration: Q = Q(2 * 10**13)
    recovered_norm_bound: Q = _RECOVERED
    source_form_totals: tuple[Q, Q] = (Q(0), Q(0))
    source_donor_form_mean: Q = Q(0)
    joined_lifted_phase_total: str = "9*beta; beta supplied at donor preparation"
    word_window_duration: Q = _WINDOW
    word_total_duration: Q = _WORD_TIME
    word_phase_amplitude: Q = _AMPLITUDE
    word_event_multipliers: tuple[int, ...] = (1, -2, 1)
    word_marked_child_order: tuple[int, ...] = tuple(range(5))
    word_receiver_child_order: tuple[int, ...] = (1, 2, 3, 4, 0)
    word_reading_count: int = 25
    word_raw_readout_error_upper_bound: Q = Q(1, 10**15)
    elapsed_scaled_time: Q = _HORIZON
    elapsed_original_time: Q = _HORIZON / _CLOCK
    raw_readout_error_upper_bound: Q = _READOUT_ERROR
    increment_readout_error_upper_bound: Q = 2 * _READOUT_ERROR
    paired_difference_readout_error_upper_bound: Q = 4 * _READOUT_ERROR
    outer_one_upper_threshold: Q = _ONE_THRESHOLD
    outer_zero_lower_threshold: Q = _ZERO_THRESHOLD
    contact_work_budget: Q = Q(100)
    joined_storage_budget: Q = Q(200)
    scope: tuple[str, ...] = (
        "all_twelve_original_allowances_admitted_before_fresh_own_law_sources",
        "same_actual_host_residuals_through_complete_sequential_word_no_reset",
        "independent_donor_preparation_ends_recovery_at_contact_without_waiting",
        "all_relative_origins_are_separately_supplied_preparations_not_a_scan",
        "full_108_coordinates_144_edges_and_equal_opposite_boundary_reaction",
        "higher_harmonic_tail_retained_in_uniform_finite_observed_response",
        "outer_one_every_origin_upper_and_outer_zero_per_source_witness_lower",
        "witness_can_depend_on_hidden_state_and_supplies_no_alignment_selector",
        "source_identity_from_preparation_not_a_transferred_readout_inverse",
        "absolute_event_work_counted_without_loss_credit_or_old_capture_transfer",
        "all_six_internal_and_nine_host_outer_identities_only_through_contact_horizon",
        "unavailable_sufficient_bound_is_not_failed_interaction_or_obstruction",
        "no_trajectory_frozen_producer_autonomous_closure_or_physical_identity",
    )


def _assess_organized_alternative_contact(
    *, form_error_bounds: object, phase_error_bounds: object
) -> _AlternativeContact:
    """Assess the unchanged preparation/history/contact using raw allowances.

    Six componentwise form and phase budgets retain the five independent
    host sources and one independent donor. No source report or changed
    preparation, law, event word, registration, clock or budget is accepted.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 6)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 6)
    if min(*rx, *rt) < 0 or max(*rx, *rt) > _ALLOWANCE:
        raise ValueError("original preparation allowances must lie in [0,1e-10]")
    host = _assess_organized_alternative_outer_acquisition(
        form_error_bounds=rx[:5], phase_error_bounds=rt[:5]
    )
    donor = _assess_constitutive_recovery(
        deformation=Q(1), form_error_bound=rx[5], phase_error_bound=rt[5]
    )
    resources = _derive_organized_alternative_contact_bounds(
        recovered_radius=_RECOVERED
    )
    host_dimension = 8 * sum(value > 0 for value in rx[:5] + rt[:5])
    donor_dimension = 8 * (int(rx[5] > 0) + int(rt[5] > 0))
    host_ok = (
        host.deformation == 1
        and host.form_error_bounds == rx[:5]
        and host.phase_error_bounds == rt[:5]
        and host.admitted
        and host.capture_certified
        and host.outer_winding_retained
        and host.child_winding_retained
        and host.control_capture_certified
        and host.control_outer_zero_winding_retained
        and host.control_child_winding_retained
        and host.source_formation_time == 200
        and host.source_relaxation_duration == 2 * 10**13
        and host.elapsed_scaled_time == 200
        and host.recovered_child_norm_bound == _RECOVERED
        and host.relative_preparation_dimension == host_dimension
    )
    donor_ok = (
        donor.deformation == 1
        and donor.form_error_bound == rx[5]
        and donor.phase_error_bound == rt[5]
        and donor.recovery_certified
        and donor.formation_time == 100
        and donor.relaxation_duration == 10**13
        and donor.source.preparation_dimension == donor_dimension
        and donor.endpoint_form_norm_squared_upper_bound is not None
        and donor.endpoint_phase_norm_squared_upper_bound is not None
        and donor.endpoint_form_norm_squared_upper_bound < _RECOVERED**2
        and donor.endpoint_phase_norm_squared_upper_bound < _RECOVERED**2
    )
    source_flags = (host_ok, donor_ok, resources.source_class_certified)
    source_ok = all(source_flags)
    word_ok = (
        source_ok
        and resources.word_identity_certified
        and resources.word_work_certified
    )
    work_ok = word_ok and resources.contact_work_certified
    identity_ok = word_ok and resources.contact_identity_certified
    response_ok = word_ok and resources.response_certified
    unresolved = tuple(
        name
        for name, ok in zip(
            (
                "actual_own_law_host_source_not_certified",
                "actual_own_law_donor_source_not_certified",
                "actual_sources_outside_proved_form_class",
            ),
            source_flags,
        )
        if not ok
    ) + tuple(name for name, value in resources.sufficient_margins if value <= 0)
    admitted = source_ok and word_ok and work_ok and identity_ok and response_ok
    return _AlternativeContact(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        host_acquisition=host,
        donor_recovery=donor,
        resources=resources,
        relative_preparation_dimension=host.relative_preparation_dimension
        + donor.source.preparation_dimension,
        source_admission_flags=source_flags,
        source_certified=source_ok,
        word_certified=word_ok,
        work_certified=work_ok,
        identity_certified=identity_ok,
        response_certified=response_ok,
        admitted=admitted,
        status=(
            "alternative_contact_certified"
            if admitted
            else "alternative_contact_unavailable"
        ),
        unresolved_conditions=unresolved,
    )
