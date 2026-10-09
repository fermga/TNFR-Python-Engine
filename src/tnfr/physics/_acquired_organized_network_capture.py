"""Full-state capture of three acquired C9 objects under two fixed contacts.

ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md owns the conditional proof. This
detached certificate rebuilds original preparation and recovery; it neither
integrates a response nor resets a reached state to its registered target.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._exact_linear_algebra import exact_matrix_inverse
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_network_compatibility import (
    _derive_organized_triangle,
    _mixings,
    _OrganizedTriangle,
)
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)

_EQUAL = (Q(1, 2),) * 3
_UNEQUAL = (Q(1, 4), Q(1, 2), Q(3, 4))
_TARGET_GAP = Q(22, 25)
_CELL_GAP = Q(2)
_BREGMAN = Q(3, 20)
_RESISTANCE = Q(3, 5)
_LAPLACIAN_GAP = Q(2, 5)


def _edge_resistances(geometry):
    """Exact grounded solve on the actual full support, without projection."""
    laplacian = [[Q(0) for _ in range(27)] for _ in range(27)]
    for a, b, weight in geometry.weighted_edges:
        laplacian[a][a] += weight
        laplacian[b][b] += weight
        laplacian[a][b] -= weight
        laplacian[b][a] -= weight
    inverse = exact_matrix_inverse(tuple(tuple(row[:26]) for row in laplacian[:26]))

    def entry(a, b):
        return inverse[a][b] if a < 26 and b < 26 else Q(0)

    return tuple(
        entry(a, a) + entry(b, b) - 2 * entry(a, b)
        for a, b, _ in geometry.weighted_edges
    )


def _target(geometry):
    """Certify the implicit target; bracket endpoints are never target values."""
    alpha = 2 * pi_interval() / 9
    if geometry.mixing == _EQUAL:
        return None, None, None, (I(0),) * 3, (I(0),) * 3, I(0), True
    high = geometry.contacts[0].amplitude_bounds
    low = geometry.contacts[1].amplitude_bounds
    phase = geometry.contacts[0].phase_origin_shift_bounds

    def current_difference(a):
        return high * sin(I(a)) - low * sin(pi_interval() / 3 - 2 * a)

    left, right = Q(347, 1000), Q(348, 1000)
    original = I(left, right)
    derivative = (
        high * cos(original) + 2 * low * cos(pi_interval() / 3 - 2 * original)
    ).lo
    bracket_ok = current_difference(left).hi < 0 < current_difference(right).lo
    if bracket_ok and derivative > 0:
        for _ in range(32):
            middle = (left + right) / 2
            value = current_difference(middle)
            if value.hi < 0:
                left = middle
            elif value.lo > 0:
                right = middle
            else:
                break  # Retain the proved bracket, never choose an unresolved sign.
    signs = (current_difference(left), current_difference(right))
    a = I(left, right)
    b = pi_interval() / 3 - 2 * a
    gaps = (a - phase, b - alpha / 2, a - alpha + phase)
    raw = (I(0), gaps[0], gaps[0] + gaps[1])
    mean = sum(raw, I(0)) / 3
    origins = tuple(value - mean for value in raw)
    excess = 9 * (2 * high * cos(a) + low * cos(b) - Q(3, 2) * (1 + cos(alpha)))
    certified = bracket_ok and derivative > 0 and signs[0].hi < 0 < signs[1].lo
    return a, signs, derivative, origins, gaps, excess, certified


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
    geometry = _derive_organized_triangle(mixing=weights)
    root, signs, derivative, origins, gaps, excess, root_ok = _target(geometry)
    alpha = 2 * pi_interval() / 9
    gap_by_contact = dict(zip(geometry.contact_order, gaps))
    fine_gaps = tuple(
        (
            alpha
            if a // 9 == b // 9
            else (gap_by_contact[(a // 9, b // 9)] + ((b - a) % 9) * alpha)
        )
        for a, b, _ in geometry.weighted_edges
    )
    target_ok = root_ok and max(value.abs_max for value in fine_gaps) < _TARGET_GAP
    resistances = _edge_resistances(geometry)
    width = _CELL_GAP - _TARGET_GAP
    coefficient = (
        cos(I(_TARGET_GAP)) - cos(I(_CELL_GAP)) - width * sin(I(_TARGET_GAP))
    ) / width**2
    barrier = _BREGMAN * width**2 / _RESISTANCE
    barrier_ok = (
        coefficient.lo > _BREGMAN
        and 0 < min(resistances) <= max(resistances) < _RESISTANCE
        and _CELL_GAP + _TARGET_GAP < pi_interval().lo
    )
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
        target_root_bounds=root,
        target_root_endpoint_sign_bounds=signs,
        target_root_derivative_lower_bound=derivative,
        target_phase_origins_bounds=origins,
        target_contact_gaps_bounds=gaps,
        target_fine_gap_bounds=fine_gaps,
        target_hessian_lower_bound=cos(I(_TARGET_GAP)).lo * _LAPLACIAN_GAP,
        laplacian_gap_lower_bound=_LAPLACIAN_GAP,
        edge_effective_resistances=resistances,
        bregman_coefficient_bounds=coefficient,
        nominal_excess_bounds=excess,
        source_excess_upper_bound=source_excess,
        initial_excess_upper_bound=initial,
        boundary_energy_lower_bound=barrier,
        capture_margin=margin,
        attachment_work_bounds=geometry.aligned_attachment_work_bounds
        + I(*work_residual),
        attachment_residual_work_bounds=work_residual,
        joint_error_squared_upper_bound=(
            initial / (_BREGMAN * _LAPLACIAN_GAP) if capture else None
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
