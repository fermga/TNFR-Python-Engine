"""Joint C9 acquisition from the fixed ramps on preexisting fine support.

ORGANIZED_JOINT_ACQUISITION.md owns this complete-law certificate. The
structural deadline and whole-network preparation budget are fixed before
assessment; no isolated acquisition or recovery verdict is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval, sqrt
from ._organized_network_capture import (
    _derive_organized_triangle_capture,
    _OrganizedTriangleCapture,
)
from ._organized_network_compatibility import _EQUAL_CONTROL, _UNEQUAL_CONTROL, _mixings
from ._sine_preparation import _prepared_transit_radii
from .relational_sine_formed_classes import _SOURCE_SCALE
from .reversible_eigenmode_reference import _negative_exp_bounds

_TIME = Q(200)
_BUDGET = Q(10**9)


@dataclass(frozen=True)
class _OrganizedJointAcquisition:
    """Fresh source, full-law transit and independently rebuilt capture geometry.

    Norm bounds concern the actual endpoint. Budget policy, initial winding,
    endpoint entry and formation are separate conditions. An unavailable
    sufficient bound is not a nonformation theorem.
    """

    capture_geometry: _OrganizedTriangleCapture
    form_error_bound: Q
    phase_error_bound: Q
    relative_preparation_dimension: int
    nominal_initial_forms: tuple[Q, ...]
    nominal_initial_internal_form_storage: Q
    nominal_initial_contact_form_storage: Q
    nominal_initial_form_storage: Q
    initial_form_gradient_l1: Q
    initial_storage_correction_upper_bound: Q
    initial_storage_upper_bound: Q
    source_budget_margin: Q
    initial_form_error_norm_upper_bound: Q
    initial_phase_error_norm_upper_bound: Q
    scaled_initial_norm_upper_bound: Q
    semigroup_gap_lower_bound: Q
    exponential_decay_bounds: I
    forcing_norm_bounds: I
    proxy_target_distance_bounds: I
    endpoint_scaled_form_norm_upper_bound: Q
    endpoint_form_norm_upper_bound: Q
    endpoint_aligned_phase_norm_upper_bound: Q
    endpoint_linear_phase_storage_upper_bound: Q
    endpoint_form_storage_upper_bound: Q
    endpoint_quadratic_phase_storage_upper_bound: Q
    endpoint_excess_storage_upper_bound: Q
    initial_zero_winding_margin: Q
    endpoint_acute_margin: Q
    endpoint_cell_margin: Q
    capture_margin: Q
    source_budget_certified: bool
    initial_zero_winding_certified: bool
    entry_certified: bool
    formation_certified: bool
    winding_retained: bool
    converges_to_target: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    acquisition_time: Q = _TIME
    original_acquisition_time: Q = _TIME * Q(1024, 1023)
    common_source_storage_budget: Q = _BUDGET
    nominal_source_scale: Q = _SOURCE_SCALE
    nominal_initial_phases: tuple[Q, ...] = (Q(0),) * 27
    held_capacities: tuple[Q, ...] = (Q(1),) * 27
    nominal_initial_windings: tuple[int, ...] = (0, 0, 0)
    target_windings: tuple[int, ...] = (1, 1, 1)
    joined_mean_leaf_dimension: int = 52
    scope: tuple[str, ...] = (
        "same_primitive_ramps_freshly_admitted_on_both_fixed_connected_supports",
        "complete_degree_four_reciprocal_sine_law_and_held_unit_capacity",
        "all_initial_contact_storage_included_no_later_input_or_event",
        "separate_initial_component_error_sums_zero_only_global_means_conserved",
        "all_54_coordinates_retained_through_full_law_semigroup_bounds",
        "phase_passage_before_deadline_allowed_target_cell_entered_only_at_deadline",
        "linear_phase_storage_term_retained_for_moving_component_means",
        "no_isolated_capture_recovery_or_tiny_endpoint_radius_transferred",
        "budget_policy_separate_from_mathematical_formation_and_capture",
        "no_trajectory_support_birth_autonomous_preparation_or_physical_identification",
    )


def _assess_organized_joint_acquisition(
    *, mixing, form_error_bound, phase_error_bound
) -> _OrganizedJointAcquisition:
    """Assess the frozen time and budget from primitive preparation allowances."""
    weights = _mixings(mixing)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    if weights not in (_EQUAL_CONTROL, _UNEQUAL_CONTROL):
        raise ValueError("joint acquisition requires one of the two fixed controls")
    capture = _derive_organized_triangle_capture(mixing=weights)
    geometry = capture.geometry
    forms = tuple(_SOURCE_SCALE * (node % 9 - 4) for node in geometry.nodes)
    gradient = [Q(0)] * 27
    internal, contact = Q(0), Q(0)
    for a, b, weight in geometry.weighted_edges:
        difference = forms[a] - forms[b]
        gradient[a] += weight * difference
        gradient[b] -= weight * difference
        cost = weight * difference**2 / 2
        if a // 9 == b // 9:
            internal += cost
        else:
            contact += cost
    gradient_l1 = sum(abs(value) for value in gradient)
    correction = gradient_l1 * rx + 108 * (rx**2 + rt**2)
    source_storage = internal + contact + correction
    source_margin = _BUDGET - source_storage
    pi = pi_interval()
    gamma = geometry.gamma_bounds
    alpha = 2 * pi / 9
    norm = sqrt(I(27))
    ex, et = (norm * rx).hi, (norm * rt).hi
    initial_norm = gamma.hi * (sqrt(I(sum(value**2 for value in forms))).hi + ex)
    gap = capture.laplacian_gap_lower_bound * geometry.clock_factor
    decay = I(*_negative_exp_bounds(gap * _TIME))
    scaled_form, phase_remainder = _prepared_transit_radii(
        time=_TIME,
        decay_upper=decay.hi,
        gap_lower=gap,
        initial_norm_upper=initial_norm,
        feedback_upper=(gamma**2).hi,
        forcing_upper=norm.hi,
    )
    form_radius = (1 / gamma).hi * scaled_form
    proxy_distance = sqrt(I(180)) * abs(gamma * _SOURCE_SCALE - alpha)
    phase_radius = proxy_distance.hi + phase_remainder + gamma.hi * ex + et
    linear = capture.aligned_gradient_norm_bounds.hi * phase_radius
    form_storage = 4 * form_radius**2
    quadratic = 4 * phase_radius**2
    excess = capture.nominal_excess_bounds.hi + form_storage + linear + quadratic
    initial_margin = pi.lo - 2 * rt
    edge_gap_upper = alpha.hi + 2 * phase_radius
    acute_margin = pi.lo / 2 - edge_gap_upper
    cell_margin = capture.cell_edge_gap_bound - edge_gap_upper
    capture_margin = capture.boundary_energy_lower_bound - excess
    budget_ok = source_margin > 0
    initial_ok = initial_margin > 0
    entry = (
        capture.target_certified
        and capture.barrier_certified
        and acute_margin > 0
        and cell_margin > 0
        and capture_margin > 0
    )
    formation = initial_ok and entry
    unresolved = tuple(
        reason
        for passed, reason in (
            (budget_ok, "connected_preparation_budget_not_certified"),
            (initial_ok, "initial_zero_winding_not_certified"),
            (capture.target_certified, "joined_target_not_certified"),
            (capture.barrier_certified, "full_state_boundary_not_certified"),
            (acute_margin > 0, "endpoint_acute_winding_not_certified"),
            (cell_margin > 0, "endpoint_cell_entry_not_certified"),
            (capture_margin > 0, "endpoint_energy_entry_not_certified"),
        )
        if not passed
    )
    return _OrganizedJointAcquisition(
        capture_geometry=capture,
        form_error_bound=rx,
        phase_error_bound=rt,
        relative_preparation_dimension=24 * (int(rx > 0) + int(rt > 0)),
        nominal_initial_forms=forms,
        nominal_initial_internal_form_storage=internal,
        nominal_initial_contact_form_storage=contact,
        nominal_initial_form_storage=internal + contact,
        initial_form_gradient_l1=gradient_l1,
        initial_storage_correction_upper_bound=correction,
        initial_storage_upper_bound=source_storage,
        source_budget_margin=source_margin,
        initial_form_error_norm_upper_bound=ex,
        initial_phase_error_norm_upper_bound=et,
        scaled_initial_norm_upper_bound=initial_norm,
        semigroup_gap_lower_bound=gap,
        exponential_decay_bounds=decay,
        forcing_norm_bounds=norm,
        proxy_target_distance_bounds=proxy_distance,
        endpoint_scaled_form_norm_upper_bound=scaled_form,
        endpoint_form_norm_upper_bound=form_radius,
        endpoint_aligned_phase_norm_upper_bound=phase_radius,
        endpoint_linear_phase_storage_upper_bound=linear,
        endpoint_form_storage_upper_bound=form_storage,
        endpoint_quadratic_phase_storage_upper_bound=quadratic,
        endpoint_excess_storage_upper_bound=excess,
        initial_zero_winding_margin=initial_margin,
        endpoint_acute_margin=acute_margin,
        endpoint_cell_margin=cell_margin,
        capture_margin=capture_margin,
        source_budget_certified=budget_ok,
        initial_zero_winding_certified=initial_ok,
        entry_certified=entry,
        formation_certified=formation,
        winding_retained=entry,
        converges_to_target=entry,
        status=(
            "certified_organized_joint_acquisition" if not unresolved else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
