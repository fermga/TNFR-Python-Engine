"""Detached full-state capture of the actual attached five-C9 source.

ORGANIZED_SOURCE_CAPTURE.md owns the frozen source, acute-chart boundary
proof and disconnected control. No trajectory, support event or source
reset is performed, and no incoming report replaces primitive admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval
from ._organized_source_port import (
    _OrganizedSourceAttachment,
    _OrganizedSourceGeometry,
    _assess_organized_source_attachment,
    _derive_organized_source_port,
)
from ._repeated_organized_composition import _vector

_TARGET_ORIGIN = Q(1, 40)


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
