"""Detached stationary geometry and one actual continued-contact capture.

ORGANIZED_CONTACT_MAINTENANCE.md owns the fixed complete law and source
admission. Geometry, local stability and inclusion of an acquired source are
separate results. No trajectory, origin scan or frozen producer is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._organized_external_interaction import (
    _OrganizedExternalInteraction,
    _OrganizedExternalInteractionGeometry,
    _assess_organized_external_interaction,
    _derive_organized_external_interaction,
)
from ._organized_external_interaction import _Q as _RECOVERED_RADIUS

_CAPTURE_THRESHOLD = Q(1, 100)


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
    inner_cosine = cos(geometry.alpha_bounds)
    outer_modes = (("common", I(0), I(0)), ("hub", I(6), I(-6))) + tuple(
        (f"ring{k}", 3 - 2 * cos(2 * pi * k / 5), 1 - 2 * cos(2 * pi * k / 5))
        for k in range(1, 5)
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
                    aligned_determinant_bounds=gamma**2
                    * laplacian
                    * aligned_hessian
                    / 16,
                    antipodal_determinant_bounds=gamma**2
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
    outer_cosine = cos(2 * pi / 5)
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
    flag_names = (
        "host_acquisition_not_certified",
        "paired_host_control_not_certified",
        "external_recovery_not_certified",
        "preceding_sequential_observation_not_certified",
        "actual_hosts_outside_observation_class",
    )
    unresolved = (
        tuple(
            name
            for name, ok in zip(flag_names, source.source_admission_flags)
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
