"""Full-state capture of three acquired C9 objects under two fixed contacts.

ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md owns the conditional proof. This
detached certificate rebuilds original preparation and recovery; it neither
integrates a response nor resets a reached state to its registered target.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_network_capture import (
    _CELL_GAP,
    _TARGET_GAP,
    _derive_organized_triangle_capture,
)
from ._organized_network_compatibility import _EQUAL_CONTROL as _EQUAL
from ._organized_network_compatibility import _UNEQUAL_CONTROL as _UNEQUAL
from ._organized_network_compatibility import _mixings, _OrganizedTriangle
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)


@dataclass(frozen=True)
class _AcquiredOrganizedTriangleCapture:
    """Conditional full-law certificate; unsupported sources remain unavailable.

    Tiny source/work corrections are exact rationals kept separately from
    interval-rounded nominal quantities. Only a certified source makes these
    conditional bounds statements about the actual acquired trajectories.
    """

    recovery: _ConstitutiveRecovery
    geometry: _OrganizedTriangle
    target_root_bounds: I | None
    target_root_endpoint_sign_bounds: tuple[I, I] | None
    target_root_derivative_lower_bound: Q | None
    target_phase_origins_bounds: tuple[I, ...]
    target_contact_gaps_bounds: tuple[I, ...]
    target_fine_gap_bounds: tuple[I, ...]
    target_hessian_lower_bound: Q
    laplacian_gap_lower_bound: Q
    edge_effective_resistances: tuple[Q, ...]
    bregman_coefficient_bounds: I
    nominal_excess_bounds: I
    source_excess_upper_bound: Q
    initial_excess_upper_bound: Q
    boundary_energy_lower_bound: Q
    capture_margin: Q
    attachment_work_bounds: I
    attachment_residual_work_bounds: tuple[Q, Q]
    joint_error_squared_upper_bound: Q | None
    source_certified: bool
    target_certified: bool
    barrier_certified: bool
    capture_certified: bool
    winding_retained: bool
    converges_to_target: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    cell_edge_gap_bound: Q = _CELL_GAP
    target_edge_gap_bound: Q = _TARGET_GAP
    target_windings: tuple[int, ...] = (1, 1, 1)
    scope: tuple[str, ...] = (
        "three_actual_independently_recovered_class_one_C9_sources",
        "fixed_equal_or_unequal_oriented_contacts_and_full_54_coordinate_flow",
        "held_unit_capacity_degree_four_sine_law_and_structural_clock",
        "registered_lifts_and_conserved_global_zero_means",
        "target_acute_but_trapping_cell_need_not_be_acute_or_convex_in_energy",
        "supplied_positive_attachment_work_is_separate_from_continuous_loss",
        "no_reset_trajectory_campaign_contact_selection_or_physical_identification",
    )


def _assess_acquired_organized_triangle(
    *, mixing, form_error_bound, phase_error_bound
) -> _AcquiredOrganizedTriangleCapture:
    """Admit primitives, rebuild source and certify both fixed contact controls.

    Other mixtures require a separate target/source theorem, rather than a
    rounded target, transferred pair constants or an incoming cached report.
    """
    weights = _mixings(mixing)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    if weights not in (_EQUAL, _UNEQUAL):
        raise ValueError("capture requires one of the two fixed contact controls")
    recovery = _assess_constitutive_recovery(
        deformation=Q(0), form_error_bound=rx, phase_error_bound=rt
    )
    capture_geometry = _derive_organized_triangle_capture(mixing=weights)
    geometry = capture_geometry.geometry
    excess = capture_geometry.nominal_excess_bounds
    barrier = capture_geometry.boundary_energy_lower_bound
    target_ok = capture_geometry.target_certified
    barrier_ok = capture_geometry.barrier_certified
    alpha = 2 * pi_interval() / 9
    epsilon = recovery.endpoint_radius
    source_excess = 24 * epsilon**2
    initial = excess.hi + source_excess
    margin = barrier - initial
    source_ok = recovery.recovery_certified
    entry_ok = alpha.hi + 2 * epsilon < _CELL_GAP
    unresolved = []
    for condition, admitted in (
        ("original_source_recovery_not_certified", source_ok),
        ("joined_target_not_certified", target_ok),
        ("full_state_boundary_not_certified", barrier_ok),
        ("source_not_inside_registered_cell", entry_ok),
        ("source_energy_not_below_boundary", margin > 0),
    ):
        if not admitted:
            unresolved.append(condition)
    capture = not unresolved
    work_residual = (Q(0), 12 * epsilon**2)
    return _AcquiredOrganizedTriangleCapture(
        recovery=recovery,
        geometry=geometry,
        target_root_bounds=capture_geometry.target_root_bounds,
        target_root_endpoint_sign_bounds=capture_geometry.target_root_endpoint_sign_bounds,
        target_root_derivative_lower_bound=capture_geometry.target_root_derivative_lower_bound,
        target_phase_origins_bounds=capture_geometry.target_phase_origins_bounds,
        target_contact_gaps_bounds=capture_geometry.target_contact_gaps_bounds,
        target_fine_gap_bounds=capture_geometry.target_fine_gap_bounds,
        target_hessian_lower_bound=capture_geometry.target_hessian_lower_bound,
        laplacian_gap_lower_bound=capture_geometry.laplacian_gap_lower_bound,
        edge_effective_resistances=capture_geometry.edge_effective_resistances,
        bregman_coefficient_bounds=capture_geometry.bregman_coefficient_bounds,
        nominal_excess_bounds=excess,
        source_excess_upper_bound=source_excess,
        initial_excess_upper_bound=initial,
        boundary_energy_lower_bound=barrier,
        capture_margin=margin,
        attachment_work_bounds=geometry.aligned_attachment_work_bounds
        + I(*work_residual),
        attachment_residual_work_bounds=work_residual,
        joint_error_squared_upper_bound=(
            initial / capture_geometry.joint_coercivity_lower_bound if capture else None
        ),
        source_certified=source_ok,
        target_certified=target_ok,
        barrier_certified=barrier_ok,
        capture_certified=capture,
        winding_retained=capture,
        converges_to_target=capture,
        status=(
            "certified_acquired_organized_triangle_capture"
            if capture
            else "unavailable"
        ),
        unresolved_conditions=tuple(unresolved),
    )
