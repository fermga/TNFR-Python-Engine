"""Detached contact capture, weak-contact decisions and relational forecasts.

ORGANIZED_CONTACT_MAINTENANCE.md owns the fixed complete law and source
admission for the unit-contact capture. ORGANIZED_WEAK_CONTACT_MAINTENANCE.md
owns the separate weighted law, source-basin decision, marked-record inverse
and conditional registration timing. Geometry, local stability and inclusion
of an acquired source remain separate results.
No trajectory, origin scan or frozen producer is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from math import factorial

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin
from ._organized_external_interaction import _Q as _RECOVERED_RADIUS
from ._organized_external_interaction import (
    _SOURCE_FAILURE_NAMES,
    _assess_organized_external_interaction,
    _derive_organized_external_interaction,
    _OrganizedExternalInteraction,
    _OrganizedExternalInteractionGeometry,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector

_CAPTURE_THRESHOLD = Q(1, 100)
_WEAK_CONTACT = Q(1, 100000)
_BASIN_INNER = Q(5, 8)
_BASIN_OUTER = Q(627, 1000)
_RELATIONAL_READOUT_ERROR = Q(1, 10000)
_REGISTRATION_TOLERANCE = Q(1, 1000)
_CORRECTED_REGISTRATION_TUBE = Q(1, 2000)
_REGISTRATION_SINE_MINORANT = Q(1, 500)


@dataclass(frozen=True)
class _ContactStationaryMode:
    """One spatial mode, retaining its two complete form/phase tangent rows."""

    internal_mode: int
    outer_mode: str
    laplacian_bounds: I
    aligned_hessian_bounds: I
    antipodal_hessian_bounds: I
    trace_bounds: I
    aligned_determinant_bounds: I
    antipodal_determinant_bounds: I


@dataclass(frozen=True)
class _OrganizedContactMaintenanceGeometry:
    interaction_geometry: _OrganizedExternalInteractionGeometry
    antipodal_donor_phase_origin_bounds: I
    modes: tuple[_ContactStationaryMode, ...]
    aligned_relative_stable_modes: int
    antipodal_relative_stable_modes: int
    antipodal_relative_unstable_modes: int
    conserved_mean_modes: int
    near_carrier_instability_coefficient_bounds: I
    internal_phase_defect_threshold_bounds: I
    aligned_stability_certified: bool
    antipodal_instability_certified: bool
    near_carrier_obstruction_certified: bool
    scope: tuple[str, ...] = (
        "full_fifty_four_spatial_modes_and_both_complete_tangent_rows",
        "aligned_outer_origins_and_antipodal_donor_are_separate_stationary_controls",
        "two_neutral_modes_are_the_conserved_form_and_phase_lift_means",
        "negative_hessian_directions_use_joint_form_phase_instability",
        "triangle_period_obstruction_only_excludes_an_all_edge_acute_chart",
        "near_carrier_defect_bound_is_a_conditional_equilibrium_instability_theorem",
        "stationary_geometry_supplies_no_acquired_source_entry_or_identity_loss",
    )


def _derive_organized_contact_maintenance():
    """Rebuild fixed full-state modes and a conditional defect threshold."""
    original = _derive_organized_external_interaction()
    geometry = replace(
        original,
        external_phase_origin_bounds=I(0),
        scope=tuple(
            (
                "external_zero_origin_for_the_aligned_stationary_target"
                if entry
                == "external_pi_over_two_origin_supplied_from_primitive_preparation"
                else entry
            )
            for entry in original.scope
        ),
    )
    pi, gamma = pi_interval(), geometry.gamma_bounds
    gamma_squared = gamma**2
    inner_cosine = cos(geometry.alpha_bounds)
    outer_cosines = tuple(cos(2 * pi * k / 5) for k in range(1, 5))
    outer_modes = (("common", I(0), I(0)), ("hub", I(6), I(-6))) + tuple(
        (f"ring{k}", 3 - 2 * cosine, 1 - 2 * cosine)
        for k, cosine in enumerate(outer_cosines, 1)
    )
    modes = []
    for k in range(9):
        internal = I(0) if k == 0 else 2 - 2 * cos(2 * pi * k / 9)
        for label, outer, antipodal in outer_modes:
            laplacian = internal + outer
            aligned_hessian = inner_cosine * internal + outer
            antipodal_hessian = inner_cosine * internal + antipodal
            modes.append(
                _ContactStationaryMode(
                    internal_mode=k,
                    outer_mode=label,
                    laplacian_bounds=laplacian,
                    aligned_hessian_bounds=aligned_hessian,
                    antipodal_hessian_bounds=antipodal_hessian,
                    trace_bounds=-laplacian / 4,
                    aligned_determinant_bounds=gamma_squared
                    * laplacian
                    * aligned_hessian
                    / 16,
                    antipodal_determinant_bounds=gamma_squared
                    * laplacian
                    * antipodal_hessian
                    / 16,
                )
            )
    relative = tuple(
        mode for mode in modes if (mode.internal_mode, mode.outer_mode) != (0, "common")
    )
    aligned_stable = sum(
        2
        for mode in relative
        if mode.trace_bounds.hi < 0 < mode.aligned_determinant_bounds.lo
    )
    antipodal_stable = sum(
        (
            1
            if mode.antipodal_determinant_bounds.hi < 0
            else (
                2
                if mode.trace_bounds.hi < 0 < mode.antipodal_determinant_bounds.lo
                else 0
            )
        )
        for mode in relative
    )
    antipodal_unstable = sum(
        mode.antipodal_determinant_bounds.hi < 0 for mode in relative
    )
    outer_cosine = outer_cosines[0]
    defect = Q(5, 4) - 5 * outer_cosine * (1 - outer_cosine)
    return _OrganizedContactMaintenanceGeometry(
        interaction_geometry=geometry,
        antipodal_donor_phase_origin_bounds=pi,
        modes=tuple(modes),
        aligned_relative_stable_modes=aligned_stable,
        antipodal_relative_stable_modes=antipodal_stable,
        antipodal_relative_unstable_modes=antipodal_unstable,
        conserved_mean_modes=2,
        near_carrier_instability_coefficient_bounds=defect,
        internal_phase_defect_threshold_bounds=9 * defect,
        aligned_stability_certified=aligned_stable == 106,
        antipodal_instability_certified=(antipodal_stable, antipodal_unstable)
        == (97, 9),
        near_carrier_obstruction_certified=defect.lo > 0,
    )


@dataclass(frozen=True)
class _OrganizedContactMaintenance:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    external_interaction: _OrganizedExternalInteraction
    geometry: _OrganizedContactMaintenanceGeometry
    control_host_form_upper_bound: Q
    control_host_phase_offset_upper_bound: Q
    control_joined_form_upper_bound: Q
    control_joined_phase_offset_upper_bound: Q
    control_form_excess_upper_bound: Q
    control_phase_excess_upper_bound: Q
    control_initial_excess_upper_bound: Q
    control_attachment_work_upper_bound: Q
    internal_boundary_excess_bounds: I
    control_capture_barrier_lower_bound: Q
    outer_one_initial_internal_phase_defect_upper_bound: Q
    outer_one_initial_defect_margin: Q
    outer_one_initial_defect_domain_certified: bool
    control_initial_cell_margins: tuple[tuple[str, Q], ...]
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    control_capture_certified: bool
    control_identity_retention_certified: bool
    control_convergence_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    relative_phase_origin: Q = Q(0)
    capture_threshold: Q = _CAPTURE_THRESHOLD
    relative_preparation_dimension: int = 96
    protocol_revision: str = "4c4addf1"
    outer_one_maintenance_status: str = (
        "unavailable_full_source_maintenance_not_assessed"
    )
    other_origin_maintenance_status: str = "unavailable_source_inclusion_not_assessed"
    scope: tuple[str, ...] = (
        "actual_outer_zero_source_after_the_complete_single_history_observation",
        "six_independent_original_sources_and_all_ninety_six_relative_directions",
        "separately_prepared_beta_zero_contact_without_source_reset_or_extra_waiting",
        "sharper_control_storage_retains_every_continuous_phase_drift_and_fine_error",
        "fresh_c9_boundary_owner_and_new_cross_edge_barrier",
        "all_future_six_inner_one_and_nine_outer_zero_windings_if_capture_is_admitted",
        "asymptotic_convergence_on_the_conserved_mean_leaf_without_settling_deadline",
        "no_all_future_positive_individual_inner_gap_claim",
        "attachment_work_separate_from_historical_preparation_and_observation_costs",
        "legacy_pi_over_two_comparison_verdict_is_not_a_capture_premise",
        "outer_one_near_carrier_obstruction_is_not_an_actual_source_loss_verdict",
        "initial_defect_domain_inclusion_does_not_prove_it_is_retained",
        "no_fundamental_physical_identification_or_autonomous_event_selection",
        "no_trajectory_frozen_producer_replay_or_registration_scan",
    )


def _assess_organized_contact_maintenance(*, form_error_bounds, phase_error_bounds):
    """Rebuild actual sources once, then admit the separately prepared control.

    The shared primitive-only owner admits all twelve original allowances and
    rebuilds acquisition, recovery and the complete observation word. Its old
    pi/2 comparison is preserved as evidence, not consumed as a premise here.
    """
    source = _assess_organized_external_interaction(
        form_error_bounds=form_error_bounds, phase_error_bounds=phase_error_bounds
    )
    geometry = _derive_organized_contact_maintenance()
    host, observation = source.host_acquisition, source.observation_protocol
    g = geometry.interaction_geometry.gamma_bounds.hi
    duration = observation.total_elapsed_scaled_time
    q = _RECOVERED_RADIUS
    # Phase-only observation commands cancel exactly after the complete word.
    # The real motion between those events remains in these global bounds.
    host_form = host.full_form_error_upper_bound + g * duration
    host_phase = host.full_phase_error_upper_bound + 2 * g * host_form * duration
    form, phase = max(host_form, q), max(host_phase, q)
    # Each of the 144 unit edges has coordinate difference at most 2*radius.
    # The phase potential's linear terms cancel at the actual target, while
    # its global second derivative is at most one on every edge.
    form_excess, phase_excess = 288 * form**2, 288 * phase**2
    excess = form_excess + phase_excess
    work = Q(45, 2) * ((host_form + q) ** 2 + (host_phase + q) ** 2)
    internal_boundary = host.internal_boundary_excess_bounds
    barrier = min(internal_boundary.lo, Q(1))
    pi, alpha = pi_interval(), geometry.interaction_geometry.alpha_bounds
    initial = (
        ("initial_internal_positive_gap", alpha.lo - 2 * phase),
        ("initial_internal_acute_gap", pi.lo / 2 - alpha.hi - 2 * phase),
        ("initial_cross_acute_gap", pi.lo / 2 - 2 * phase),
    )
    margins = initial + (
        ("frozen_control_excess_threshold", _CAPTURE_THRESHOLD - excess),
        ("control_capture_boundary", barrier - excess),
    )
    source_ok = source.source_certified
    one_phase = (
        host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    )
    one_defect = 90 * one_phase**2 + 18 * q**2
    one_defect_margin = geometry.internal_phase_defect_threshold_bounds.lo - one_defect
    capture = (
        source_ok
        and geometry.aligned_stability_certified
        and all(value > 0 for _, value in margins)
    )
    unresolved = (
        tuple(
            name
            for name, ok in zip(_SOURCE_FAILURE_NAMES, source.source_admission_flags)
            if not ok
        )
        + (
            ()
            if geometry.aligned_stability_certified
            else ("aligned_target_not_certified",)
        )
        + tuple(name for name, value in margins if value <= 0)
    )
    return _OrganizedContactMaintenance(
        form_error_bounds=source.form_error_bounds,
        phase_error_bounds=source.phase_error_bounds,
        external_interaction=source,
        geometry=geometry,
        control_host_form_upper_bound=host_form,
        control_host_phase_offset_upper_bound=host_phase,
        control_joined_form_upper_bound=form,
        control_joined_phase_offset_upper_bound=phase,
        control_form_excess_upper_bound=form_excess,
        control_phase_excess_upper_bound=phase_excess,
        control_initial_excess_upper_bound=excess,
        control_attachment_work_upper_bound=work,
        internal_boundary_excess_bounds=internal_boundary,
        control_capture_barrier_lower_bound=barrier,
        outer_one_initial_internal_phase_defect_upper_bound=one_defect,
        outer_one_initial_defect_margin=one_defect_margin,
        outer_one_initial_defect_domain_certified=(
            source_ok
            and geometry.near_carrier_obstruction_certified
            and one_defect_margin >= 0
        ),
        control_initial_cell_margins=initial,
        sufficient_margins=margins,
        source_certified=source_ok,
        control_capture_certified=capture,
        control_identity_retention_certified=capture,
        control_convergence_certified=capture,
        admitted=capture,
        status="certified_beta_zero_control_maintenance" if capture else "unavailable",
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _OrganizedWeakContactStrip:
    """Conditional comparison arithmetic, with no source provenance.

    The caller establishes the complete weak-contact law, both initial envelopes,
    chi(0)=0, the retained cell and the complete equilibrium classification.
    """

    gamma_upper: Q
    contact_strength: Q
    shape_initial_upper: Q
    fine_initial_upper: Q
    inner_radius: Q
    outer_radius: Q
    transient_coefficient: Q
    shape_plateau: Q
    relative_form_upper: Q
    transient_displacement_upper: Q
    permanent_torque_error_upper: Q
    sine_lower: Q | None
    restoring_torque_lower: Q | None
    margins: tuple[tuple[str, Q], ...]
    certified: bool
    unavailable_reasons: tuple[str, ...]


def _bound_organized_weak_contact_strip(
    *,
    gamma_upper,
    contact_strength,
    shape_initial_upper,
    fine_initial_upper,
    inner_radius,
    outer_radius,
):
    """Bound a whole inward strip under the section 8 comparison premises.

    All parameters are exact or represented reals. Invalid scalar domains reject;
    valid bounds that do not satisfy the sufficient theorem return unavailable.
    No passed margin authenticates an initial state or supplies an occurrence law.
    """
    g, eta, e0, f0, a, b = (
        _exact(value, name)
        for name, value in (
            ("gamma_upper", gamma_upper),
            ("contact_strength", contact_strength),
            ("shape_initial_upper", shape_initial_upper),
            ("fine_initial_upper", fine_initial_upper),
            ("inner_radius", inner_radius),
            ("outer_radius", outer_radius),
        )
    )
    if g <= 0 or eta <= 0 or min(e0, f0) < 0 or not 0 < a < b:
        raise ValueError(
            "positive coupling/strength, nonnegative envelopes and 0<a<b required"
        )
    transient = e0 + 160 * f0
    plateau = 1200000 * eta**6
    chi = g**2 * (transient + plateau + 200 * eta**4 + Q(4, 15) * f0)
    impulse = 300 * eta * transient + Q(320, 19) * eta * f0
    permanent = 5 * eta * plateau
    pi = pi_interval()
    # Non-strict domain bounds precede the strict first-exit margins.
    reasons = []
    if eta > _WEAK_CONTACT:
        reasons.append("contact_strength_outside_proved_weak_domain")
    if g > Q(1, 3000):
        reasons.append("gamma_upper_outside_comparison_domain")
    margins = (
        ("shape_decay", Q(9, 1600) - 30 * eta**2 - Q(1, 200)),
        ("relative_form_damping", eta - g**2 / 20),
        ("fine_forcing", 3 - Q(8, 3) - 8 * eta**2),
        ("positive_corrected_strip", a - chi),
        ("classified_limit_separation", pi.lo / 5 - b - chi),
        ("transient_cannot_cross_strip", b - a - impulse),
    )
    sine_lower = restoring = None
    if a > chi and b + chi < pi.lo / 5:
        sine_lower = sin(I(5 * (a - chi), 5 * (b + chi))).lo
        restoring = 400 * eta**5 * sine_lower
        margins += (
            ("positive_strip_sine", sine_lower),
            ("permanent_torque_cannot_reverse", restoring - permanent),
        )
    else:
        reasons.append("corrected_strip_outside_positive_sine_chamber")
    reasons.extend(name for name, margin in margins if margin <= 0)
    return _OrganizedWeakContactStrip(
        gamma_upper=g,
        contact_strength=eta,
        shape_initial_upper=e0,
        fine_initial_upper=f0,
        inner_radius=a,
        outer_radius=b,
        transient_coefficient=transient,
        shape_plateau=plateau,
        relative_form_upper=chi,
        transient_displacement_upper=impulse,
        permanent_torque_error_upper=permanent,
        sine_lower=sine_lower,
        restoring_torque_lower=restoring,
        margins=margins,
        certified=not reasons,
        unavailable_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class _OrganizedWeakContactWitness:
    """One exact origin 2*pi*center_index/5+offset within the supplied lift."""

    center_index: int
    offset: Q
    origin_bounds: I
    registration_label: int
    limiting_relative_origin_turns: Q


@dataclass(frozen=True)
class _OrganizedWeakContactBasin:
    """A conditional decision about every source in one supplied origin interval.

    The reader supports a strict principal lift (-pi,pi); it never independently
    wraps endpoints. Mixed outcomes require explicit different certified witnesses.
    Source admission rebuilds a mathematical protocol family, not a live history.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    donor_origin_bounds: tuple[Q, Q]
    source: _OrganizedExternalInteraction | None
    strip: _OrganizedWeakContactStrip | None
    weak_weighted_edges: tuple[tuple[int, int, Q], ...]
    weak_weighted_degrees: tuple[Q, ...]
    source_margins: tuple[tuple[str, Q], ...]
    initial_storage_excess_upper: Q | None
    attachment_work_upper: Q | None
    source_shape_initial_upper: Q | None
    source_fine_initial_upper: Q | None
    candidate_offset_bounds: tuple[tuple[int, I], ...]
    witnesses: tuple[_OrganizedWeakContactWitness, ...]
    chart_certified: bool
    source_certified: bool
    strip_certified: bool
    registration_label: int | None
    limiting_relative_origin_turns: Q | None
    status: str
    unavailable_reasons: tuple[str, ...]
    contact_strength: Q = _WEAK_CONTACT
    clock_factor: Q = Q(1, 4)
    held_capacities: tuple[Q, ...] = (Q(1),) * 54
    initial_relative_form: Q = Q(0)
    scope: tuple[str, ...] = (
        "unchanged_outer_one_acquisition_observation_and_independent_donor",
        "all_twelve_original_allowances_and_ninety_six_relative_directions",
        "same_weighted_support_in_both_complete_quarter_rate_rows",
        "donor_origin_supplied_at_primitive_preparation_without_reset",
        "exact_chi_zero_and_y_initial_equals_minus_supplied_beta",
        "conditional_source_family_not_authenticated_live_membership",
        "strict_principal_lift_no_independent_endpoint_wrapping",
        "mixed_witnesses_prove_nonunique_outcomes_not_exact_saddle_location",
        "no_trajectory_producer_event_guard_search_or_physical_validation",
    )


def _assess_organized_weak_contact_basin(
    *, form_error_bounds, phase_error_bounds, donor_origin_bounds
):
    """Decide a registration under fixed eta and the frozen inward strip.

    The actual twelve source allowances are re-admitted before source rebuilding.
    Only fresh preparation, acquisition and observation premises are consumed from
    the legacy source owner; its pi/2 comparison and unit-contact verdicts do not
    decide this weak-law outcome. The high-level API accepts no claimed source,
    envelope, strip verdict or initial mean correlation.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 6)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 6)
    lower, upper = _vector(donor_origin_bounds, "donor_origin_bounds", 2)
    if min(*rx, *rt) < 0 or max(*rx, *rt) > Q(1, 10**10):
        raise ValueError("original errors must belong to frozen [0, 1e-10]")
    if lower > upper:
        raise ValueError("donor_origin_bounds must be an ordered lifted interval")
    pi = pi_interval()
    if not -pi.lo < lower <= upper < pi.lo:
        return _decide_admitted_weak_contact_basin(rx, rt, (lower, upper), None)
    source = _assess_organized_external_interaction(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    return _decide_admitted_weak_contact_basin(rx, rt, (lower, upper), source)


def _decide_admitted_weak_contact_basin(rx, rt, origin_bounds, source):
    """Compose a source rebuilt in the current primitive-only reader call.

    This internal kernel is not a report-admission API. Its callers have admitted
    all primitives and rebuilt this source themselves; ``None`` is used only to
    return an unsupported-chart result without constructing a source.
    """
    lower, upper = origin_bounds
    pi = pi_interval()
    chart = -pi.lo < lower <= upper < pi.lo
    fields = dict(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        donor_origin_bounds=(lower, upper),
        source=None,
        strip=None,
        weak_weighted_edges=(),
        weak_weighted_degrees=(),
        source_margins=(),
        initial_storage_excess_upper=None,
        attachment_work_upper=None,
        source_shape_initial_upper=None,
        source_fine_initial_upper=None,
        candidate_offset_bounds=(),
        witnesses=(),
        chart_certified=chart,
        source_certified=False,
        strip_certified=False,
        registration_label=None,
        limiting_relative_origin_turns=None,
    )
    if not chart:
        return _OrganizedWeakContactBasin(
            **fields,
            status="unavailable_chart",
            unavailable_reasons=("origin_lift_not_certified_inside_minus_pi_pi",),
        )
    assert source is not None
    host, observation, geometry = (
        source.host_acquisition,
        source.observation_protocol,
        source.geometry,
    )
    eta, g, q = _WEAK_CONTACT, Q(1, 3000), _RECOVERED_RADIUS
    duration = observation.total_elapsed_scaled_time
    form = Q(1, 100) + g * duration
    form_drift = (2 * form + g) * duration
    phase_drift = 2 * g * form * duration
    host_work = 360 * form * form_drift + 180 * phase_drift
    attachment = 45 * eta * ((form + q) ** 2 / 2 + 2)
    excess = Q(11, 1000) + host_work + 4 * q**2 + attachment
    phase_fine, form_fine = 2 * (Q(1, 10**6) + phase_drift), 2 * form
    fine_initial = 27 * ((phase_fine + g * form_fine) ** 2 + (g * form_fine) ** 2)
    shape_initial = Q(1, 6) + 7 * eta + g / 40
    alpha = geometry.alpha_bounds
    weak_edges = (
        geometry.internal_edges
        + geometry.outer_edges
        + tuple((a, b, eta) for a, b, _ in geometry.contact_edges)
    )
    weak_degrees = tuple(
        sum((weight for a, b, weight in weak_edges if node in (a, b)), Q(0))
        for node in range(54)
    )
    # All lower-source properties are freshly rebuilt, not incoming flags.
    # Added weak-law cell and storage bounds are derived independently here.
    margins = (
        ("gamma_majorant", g - geometry.gamma_bounds.hi),
        ("host_acquired_excess", Q(11, 1000) - host.endpoint_excess_upper_bound),
        ("host_internal_positive", alpha.lo - 2 * Q(1, 10**6) - 2 * phase_drift),
        (
            "host_internal_acute",
            pi.lo / 2 - alpha.hi - 2 * Q(1, 10**6) - 2 * phase_drift,
        ),
        ("host_rim_acute", pi.lo / 2 - Q(7, 5) - 2 * Q(1, 10**6) - 2 * phase_drift),
        ("donor_internal_positive", alpha.lo - 2 * q),
        ("donor_internal_acute", pi.lo / 2 - alpha.hi - 2 * q),
        ("weak_storage_sublevel", Q(3, 250) - excess),
        ("old_cell_barrier", host.boundary_energy_lower_bound - excess),
        ("source_shape_envelope", Q(1, 5) - shape_initial),
        ("source_fine_envelope", Q(1, 10**8) - fine_initial),
        ("host_shape_norm", Q(1, 6) ** 2 - Q(2, 75)),
        ("host_form_norm", Q(1, 40) ** 2 - 5 * form**2),
    )
    reasons = [
        name
        for name, ok in zip(_SOURCE_FAILURE_NAMES, source.source_admission_flags)
        if not ok
    ]
    if not source.source_certified:
        reasons.append("original_source_not_certified")
    if not (
        source.relative_preparation_dimension == 96
        and geometry.node_order == tuple(range(54))
        and geometry.held_capacities == (Q(1),) * 54
        and geometry.clock_factor == Q(1, 4)
        and len(geometry.internal_edges + geometry.outer_edges) == 99
        and len(geometry.contact_edges) == 45
        and all(weight == 1 for _, _, weight in geometry.weighted_edges)
        and weak_degrees == (4 + eta,) * 45 + (2 + 5 * eta,) * 9
        and duration == Q(1, 5000)
        and observation.form_upper_bound == Q(1, 100)
        and observation.fine_phase_upper_bound == Q(1, 10**6)
        and observation.child_origin_gap_upper_bound == Q(7, 5)
        and observation.event_multipliers == (1, -2, 1)
        and observation.marked_child_order == tuple(range(5))
    ):
        reasons.append("fixed_source_support_history_or_clock_not_matched")
    reasons.extend(name for name, margin in margins if margin <= 0)
    source_ok = not reasons
    strip = _bound_organized_weak_contact_strip(
        gamma_upper=g,
        contact_strength=eta,
        shape_initial_upper=Q(1, 5),
        fine_initial_upper=Q(1, 10**8),
        inner_radius=_BASIN_INNER,
        outer_radius=_BASIN_OUTER,
    )
    fields.update(
        source=source,
        strip=strip,
        weak_weighted_edges=weak_edges,
        weak_weighted_degrees=weak_degrees,
        source_margins=margins,
        initial_storage_excess_upper=excess,
        attachment_work_upper=attachment,
        source_shape_initial_upper=shape_initial,
        source_fine_initial_upper=fine_initial,
        source_certified=source_ok,
        strip_certified=strip.certified,
    )
    if not source_ok:
        return _OrganizedWeakContactBasin(
            **fields, status="unavailable_source", unavailable_reasons=tuple(reasons)
        )
    if not strip.certified:
        return _OrganizedWeakContactBasin(
            **fields,
            status="unavailable_strip",
            unavailable_reasons=strip.unavailable_reasons,
        )
    origin = I(lower, upper)
    candidates, witnesses, containing = [], [], []
    for k in range(-2, 3):
        center = 2 * k * pi / 5
        relative = origin - center
        candidates.append((k, relative))
        if relative.abs_max <= strip.inner_radius:
            containing.append(k)
        # A fixed finite list of exact mathematical origins, not a trajectory
        # or saddle search. Rounding may abstain, never invent membership.
        for offset in (Q(0), strip.inner_radius, -strip.inner_radius):
            witness = center + offset
            if lower <= witness.lo and witness.hi <= upper:
                witnesses.append(
                    _OrganizedWeakContactWitness(
                        center_index=k,
                        offset=offset,
                        origin_bounds=witness,
                        registration_label=k % 5,
                        limiting_relative_origin_turns=Q(-k, 5),
                    )
                )
    fields.update(candidate_offset_bounds=tuple(candidates), witnesses=tuple(witnesses))
    if len(containing) == 1:
        k = containing[0]
        fields.update(registration_label=k % 5, limiting_relative_origin_turns=Q(-k, 5))
        return _OrganizedWeakContactBasin(
            **fields, status="certified_unique_registration", unavailable_reasons=()
        )
    if len({w.registration_label for w in witnesses}) >= 2:
        return _OrganizedWeakContactBasin(
            **fields,
            status="certified_mixed_registrations",
            unavailable_reasons=(
                "different_certified_origins_have_different_destinations",
            ),
        )
    return _OrganizedWeakContactBasin(
        **fields,
        status="unavailable_insufficient_certificate",
        unavailable_reasons=("interval_not_contained_in_one_certified_source_basin",),
    )


@dataclass(frozen=True)
class _OrganizedWeakContactRecord:
    """Conditional prediction from one marked relational phasor record.

    The two primitive quadratures observe exp(i*(theta[45]-theta[18])) at the
    post-observation, pre-contact source. The inferred origin is circular, in a
    certified principal lift; it is not an unwrapped conserved phase mean.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    quadratures: tuple[Q, Q]
    raw_rectangle: tuple[tuple[Q, Q], tuple[Q, Q]]
    unit_circle_norm_squared_bounds: tuple[Q, Q]
    unit_circle_compatible: bool
    quadrature_bounds: tuple[I, I] | None
    argument_bounds: I | None
    host_port_offset_bounds: I | None
    donor_port_residual_bounds: I | None
    donor_origin_bounds: I | None
    source: _OrganizedExternalInteraction | None
    basin: _OrganizedWeakContactBasin | None
    source_certified: bool
    argument_certified: bool
    origin_lift_certified: bool
    registration_label: int | None
    limiting_relative_origin_turns: Q | None
    status: str
    unavailable_reasons: tuple[str, ...]
    readout_error: Q = _RELATIONAL_READOUT_ERROR
    marked_nodes: tuple[int, int] = (45, 18)
    scope: tuple[str, ...] = (
        "one_precontact_relational_record_without_state_or_source_change",
        "two_raw_quadratures_with_fixed_componentwise_possibly_correlated_error",
        "all_original_source_allowances_and_complete_observation_history",
        "common_slot_zero_carrier_cancels_but_host_endpoint_offset_is_retained",
        "independent_donor_residual_excludes_legacy_supplied_pi_over_two_origin",
        "principal_circular_origin_not_unwrapped_conserved_phase_mean",
        "whole_inverse_enclosure_must_have_one_certified_registration",
        "mixed_outer_interval_witnesses_do_not_prove_same_record_mixed_outcomes",
        "unit_circle_compatibility_does_not_authenticate_source_or_live_history",
        "no_supplied_beta_sensor_calibration_event_or_trajectory_execution",
    )


def _assess_organized_weak_contact_record(
    *, form_error_bounds, phase_error_bounds, quadratures
):
    """Decode one raw marked record and compose a fresh weak-basin decision.

    All fourteen primitive scalars are admitted before model checks or source
    rebuilding. The caller supplies neither donor origin nor uncertainty policy,
    source report or basin verdict. Impossible records, argument cuts, unresolved
    lifts and insufficient basin evidence retain distinct non-label outcomes.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 6)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 6)
    real, imaginary = _vector(quadratures, "quadratures", 2)
    if min(*rx, *rt) < 0 or max(*rx, *rt) > Q(1, 10**10):
        raise ValueError("original errors must belong to frozen [0, 1e-10]")
    epsilon = _RELATIONAL_READOUT_ERROR
    raw = tuple((value - epsilon, value + epsilon) for value in (real, imaginary))
    # Exact bounds precede outward dyadic rounding. The squared norm is
    # continuous on the connected rectangle, so these extrema characterize
    # intersection with the unit circle, including a single tangent point.
    norm_lower = sum(
        (Q(0) if lower <= 0 <= upper else min(lower**2, upper**2))
        for lower, upper in raw
    )
    norm_upper = sum(max(lower**2, upper**2) for lower, upper in raw)
    compatible = norm_lower <= 1 <= norm_upper
    fields = dict(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        quadratures=(real, imaginary),
        raw_rectangle=raw,
        unit_circle_norm_squared_bounds=(norm_lower, norm_upper),
        unit_circle_compatible=compatible,
        quadrature_bounds=None,
        argument_bounds=None,
        host_port_offset_bounds=None,
        donor_port_residual_bounds=None,
        donor_origin_bounds=None,
        source=None,
        basin=None,
        source_certified=False,
        argument_certified=False,
        origin_lift_certified=False,
        registration_label=None,
        limiting_relative_origin_turns=None,
    )
    if not compatible:
        return _OrganizedWeakContactRecord(
            **fields,
            status="incompatible_observation",
            unavailable_reasons=("readout_rectangle_misses_unit_circle",),
        )
    boxes = tuple(I(max(Q(-1), lower), min(Q(1), upper)) for lower, upper in raw)
    fields.update(quadrature_bounds=boxes)
    try:
        angle = arg(*boxes)
    except ValueError:
        return _OrganizedWeakContactRecord(
            **fields,
            status="unavailable_argument",
            unavailable_reasons=("readout_rectangle_has_no_certified_principal_arg",),
        )
    fields.update(argument_bounds=angle, argument_certified=True)
    source = _assess_organized_external_interaction(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    host = source.host_acquisition
    radius = host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    host_port = host.primitive_entry.endpoint_phase_bounds[2] + I(-radius, radius)
    donor_residual = I(-_RECOVERED_RADIUS, _RECOVERED_RADIUS)
    # theta45-theta18 = beta + donor residual - host child-two offset.
    # Both slot-zero carriers are the same exact angle and cancel identically.
    origin = angle + host_port - donor_residual
    basin = _decide_admitted_weak_contact_basin(rx, rt, (origin.lo, origin.hi), source)
    fields.update(
        source=source,
        basin=basin,
        host_port_offset_bounds=host_port,
        donor_port_residual_bounds=donor_residual,
        donor_origin_bounds=origin,
        source_certified=basin.source_certified,
        origin_lift_certified=basin.chart_certified,
    )
    if not basin.chart_certified:
        return _OrganizedWeakContactRecord(
            **fields,
            status="unavailable_lift",
            unavailable_reasons=("inferred_origin_has_no_certified_principal_lift",),
        )
    if not basin.source_certified:
        return _OrganizedWeakContactRecord(
            **fields,
            status="unavailable_source",
            unavailable_reasons=basin.unavailable_reasons,
        )
    if basin.status != "certified_unique_registration":
        return _OrganizedWeakContactRecord(
            **fields,
            status="unavailable_registration",
            unavailable_reasons=(
                "outer_inverse_interval_has_no_single_certified_registration",
            ),
        )
    fields.update(
        registration_label=basin.registration_label,
        limiting_relative_origin_turns=basin.limiting_relative_origin_turns,
    )
    return _OrganizedWeakContactRecord(
        **fields,
        status="certified_observed_registration",
        unavailable_reasons=(),
    )


@dataclass(frozen=True)
class _OrganizedWeakContactTiming:
    """Independent early exclusion and permanent late collective retention.

    Both deadlines use structural contact time starting at attachment. The
    tolerance concerns circular host-minus-donor registration, not complete
    state equilibrium, exact finite settlement or a laboratory clock.
    """

    record: _OrganizedWeakContactRecord
    gamma_bounds: I | None
    selected_origin_center_bounds: I | None
    initial_distance_lower: Q | None
    early_deadline: Q | None
    early_displacement_upper: Q | None
    early_margin: Q | None
    late_deadline: Q | None
    burn_time: Q | None
    burn_exponent_lower: Q | None
    decay_upper: Q | None
    transient_torque_upper: Q | None
    permanent_torque_upper: Q | None
    relative_form_upper: Q | None
    annulus_angle_bounds: I | None
    endpoint_sine_bounds: tuple[I, ...]
    normalized_inward_speed_lower: Q | None
    travel_time_upper: Q | None
    retained_distance_upper: Q | None
    late_margins: tuple[tuple[str, Q], ...]
    early_certified: bool
    late_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    tolerance: Q = _REGISTRATION_TOLERANCE
    corrected_tube_radius: Q = _CORRECTED_REGISTRATION_TUBE
    restoring_sine_minorant: Q = _REGISTRATION_SINE_MINORANT
    clock: str = "structural_tau_elapsed_since_original_attachment"
    scope: tuple[str, ...] = (
        "same_raw_fourteen_primitive_record_and_one_fresh_source_rebuild",
        "actual_gamma_enclosure_and_unchanged_complete_weak_contact_rows",
        "whole_origin_interval_and_signed_selected_center_without_truth_input",
        "early_exclusion_for_every_time_from_zero_through_early_deadline",
        "late_inclusion_for_every_time_at_or_after_late_deadline",
        "persistent_relative_form_correction_retained_in_both_claims",
        "independent_early_and_late_availability_without_trajectory_inference",
        "collective_registration_tolerance_not_full_state_equilibrium",
        "no_exponential_at_enormous_deadline_or_scientific_response_execution",
    )


def _assess_organized_weak_contact_timing(
    *, form_error_bounds, phase_error_bounds, quadratures
):
    """Predict conditional time windows from the unchanged primitive record.

    The shared record reader admits every primitive, checks compatibility and
    rebuilds the source once. There are no supplied truth, report, family,
    deadline or tolerance inputs. Early exclusion may remain unavailable even
    when late retention is proved, for example for records near the destination.
    """
    record = _assess_organized_weak_contact_record(
        form_error_bounds=form_error_bounds,
        phase_error_bounds=phase_error_bounds,
        quadratures=quadratures,
    )
    return _time_admitted_weak_contact_record(record)


def _time_admitted_weak_contact_record(record):
    """Timing composition for the record rebuilt within the current call.

    This internal kernel is not an incoming-report admission API. The only
    runtime caller supplies its freshly rebuilt record; constructed controls
    may separately establish the same whole-interval premises.
    """
    fields = dict(
        record=record,
        gamma_bounds=None,
        selected_origin_center_bounds=None,
        initial_distance_lower=None,
        early_deadline=None,
        early_displacement_upper=None,
        early_margin=None,
        late_deadline=None,
        burn_time=None,
        burn_exponent_lower=None,
        decay_upper=None,
        transient_torque_upper=None,
        permanent_torque_upper=None,
        relative_form_upper=None,
        annulus_angle_bounds=None,
        endpoint_sine_bounds=(),
        normalized_inward_speed_lower=None,
        travel_time_upper=None,
        retained_distance_upper=None,
        late_margins=(),
        early_certified=False,
        late_certified=False,
    )
    if record.status != "certified_observed_registration":
        return _OrganizedWeakContactTiming(
            **fields,
            status="unavailable_record",
            unavailable_reasons=("upstream_record_not_certified", record.status)
            + record.unavailable_reasons,
        )
    gamma = record.source.geometry.gamma_bounds
    g_lo, g_hi = gamma.lo, gamma.hi
    strip = record.basin.strip
    eta, eta5 = record.basin.contact_strength, record.basin.contact_strength**5
    correction = strip.relative_form_upper
    permanent = strip.permanent_torque_error_upper
    pi = pi_interval()
    # The limiting turns include the signed principal center index. Using the
    # nonnegative label alone would choose the wrong lift for labels three/four.
    center = -2 * pi * record.limiting_relative_origin_turns
    offset = record.donor_origin_bounds - center
    initial_distance = (
        Q(0) if offset.contains(0) else min(abs(offset.lo), abs(offset.hi))
    )
    early = 1 / (1000 * g_hi**2 * eta5)
    late = 4 / (g_lo**2 * eta5)
    burn = 1 / (g_lo**2 * eta)
    early_displacement = (
        correction
        + strip.transient_displacement_upper
        + Q(3, 10) * g_hi**2 * (1000 * eta5 + permanent) * early
    )
    early_margin = initial_distance - _REGISTRATION_TOLERANCE - early_displacement
    # lambda >= g_lo^2/200, rho >= lambda. At the frozen burn time both
    # exponents are at least 500. The positive degree-20 Taylor term of exp(z)
    # bounds exp(-z) strictly by this exact rational for every z >= 500.
    burn_exponent = g_lo**2 * burn / 200
    decay = Q(factorial(20), 500**20)
    transient = (
        5 * eta * strip.transient_coefficient + Q(4, 3) * eta * strip.fine_initial_upper
    ) * decay
    h, b, sigma = (
        _CORRECTED_REGISTRATION_TUBE,
        strip.outer_radius,
        _REGISTRATION_SINE_MINORANT,
    )
    lower_angle, upper_angle = 5 * (h - correction), 5 * (b + correction)
    angles = I(lower_angle, upper_angle)
    endpoints = ()
    # Sine is concave on [0,pi], so the minimum over this full annulus is
    # attained at an endpoint. A midpoint-Lipschitz enclosure of the wide
    # interval would lose the needed positive lower bound.
    if 0 < lower_angle <= upper_angle < pi.lo:
        endpoints = sin(I(lower_angle)), sin(I(upper_angle))
    inward = Q(3, 10) * (400 * sigma - permanent / eta5 - transient / eta5)
    travel = 5 * b / (g_lo**2 * eta5)
    retained = h + correction
    margins = (
        ("persistent_correction_budget", Q(1, 10**6) - correction),
        ("restored_collective_tolerance", _REGISTRATION_TOLERANCE - retained),
        ("positive_annulus_angle", lower_angle),
        ("annulus_below_pi", pi.lo - upper_angle),
        ("decayed_transient_torque", eta5 / 10 - transient),
        ("inward_speed", inward - Q(1, 5)),
        ("arrival_before_late_deadline", late - burn - travel),
    )
    if endpoints:
        margins += (("whole_annulus_sine", min(box.lo for box in endpoints) - sigma),)
    reasons = []
    if burn_exponent < 500:
        reasons.append("burn_exponent_below_positive_taylor_bound")
    if not endpoints:
        reasons.append("annulus_outside_positive_sine_chamber")
    reasons.extend(name for name, margin in margins if margin <= 0)
    late_ok = not reasons
    early_ok = early_margin > 0
    if not early_ok:
        reasons.append("initial_interval_too_close_for_early_exclusion")
    fields.update(
        gamma_bounds=gamma,
        selected_origin_center_bounds=center,
        initial_distance_lower=initial_distance,
        early_deadline=early,
        early_displacement_upper=early_displacement,
        early_margin=early_margin,
        late_deadline=late,
        burn_time=burn,
        burn_exponent_lower=burn_exponent,
        decay_upper=decay,
        transient_torque_upper=transient,
        permanent_torque_upper=permanent,
        relative_form_upper=correction,
        annulus_angle_bounds=angles,
        endpoint_sine_bounds=endpoints,
        normalized_inward_speed_lower=inward,
        travel_time_upper=travel,
        retained_distance_upper=retained,
        late_margins=margins,
        early_certified=early_ok,
        late_certified=late_ok,
    )
    status = {
        (True, True): "certified_early_exclusion_and_late_retention",
        (False, True): "certified_late_retention_only",
        (True, False): "certified_early_exclusion_only",
        (False, False): "unavailable_timing_bounds",
    }[early_ok, late_ok]
    return _OrganizedWeakContactTiming(
        **fields, status=status, unavailable_reasons=tuple(reasons)
    )
