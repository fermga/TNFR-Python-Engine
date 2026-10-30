"""Source-specific receiver delay under the complete connected sine law.

ORGANIZED_RECEIVER_MECHANISM.md owns the proof. Two original ramps and a
locally flat receiver are freshly admitted; no previous acquisition verdict
or numerical trajectory is consumed. Eventual joined capture stays open.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ._organized_network_capture import (
    _derive_organized_triangle_capture,
    _OrganizedTriangleCapture,
)
from ._organized_network_compatibility import _EQUAL_CONTROL
from ._sine_preparation import _prepared_form_costs, _prepared_transit_radii
from .relational_sine_formed_classes import _SOURCE_SCALE
from .reversible_eigenmode_reference import _negative_exp_bounds

_LAYER_TIME = Q(200)


@dataclass(frozen=True)
class _OrganizedReceiverMechanism:
    """Fresh source bounds and necessary delay, without a formation verdict.

    The exclusion endpoint is open and analytically derived, not a reserved
    response deadline. Donor winding at the layer endpoint supplies no later
    donor retention. Current orientation concerns the mixed coordinate.
    """

    capture_geometry: _OrganizedTriangleCapture
    form_error_bound: Q
    phase_error_bound: Q
    relative_preparation_dimension: int
    nominal_initial_forms: tuple[Q, ...]
    nominal_initial_form_gradient: tuple[Q, ...]
    nominal_initial_phase_rate_bounds: tuple[I, ...]
    nominal_initial_internal_form_storage: Q
    nominal_initial_contact_form_storage: Q
    nominal_initial_form_storage: Q
    initial_form_gradient_l1: Q
    initial_storage_correction_upper_bound: Q
    initial_storage_upper_bound: Q
    scaled_initial_norm_upper_bound: Q
    semigroup_gap_lower_bound: Q
    exponential_decay_bounds: I
    forcing_norm_bounds: I
    slow_clock_factor_bounds: I
    reference_phase_bounds: tuple[I, ...]
    proxy_reference_distance_bounds: I
    layer_scaled_form_norm_upper_bound: Q
    layer_form_norm_upper_bound: Q
    layer_phase_error_norm_upper_bound: Q
    layer_donor_acute_margin: Q
    layer_receiver_zero_winding_margin: Q
    early_receiver_edge_upper_bound: Q
    early_receiver_zero_winding_margin: Q
    late_receiver_edge_offset_upper_bound: Q
    late_receiver_edge_growth_upper_bound: Q
    receiver_exclusion_time_lower_bound: Q
    receiver_exclusion_until: Q | None
    reference_current_bounds: tuple[I, ...]
    layer_current_error_upper_bound: Q
    layer_receiver_oriented_current_lower_bound: Q
    initial_zero_winding_certified: bool
    layer_windings: tuple[int, int, int] | None
    early_receiver_zero_winding_certified: bool
    receiver_delay_certified: bool
    receiver_oriented_current_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    layer_time: Q = _LAYER_TIME
    original_layer_time: Q = _LAYER_TIME * Q(1024, 1023)
    nominal_source_scale: Q = _SOURCE_SCALE
    nominal_initial_phases: tuple[Q, ...] = (Q(0),) * 27
    nominal_initial_windings: tuple[int, ...] = (0, 0, 0)
    held_capacities: tuple[Q, ...] = (Q(1),) * 27
    heat_shape_upper_bound: Q = Q(3, 4)
    linear_receiver_heat_edge_upper_bound: Q = Q(4)
    receiver_exclusion_start: Q = Q(0)
    eventual_receiver_acquisition: str = "unresolved"
    eventual_donor_retention: str = "unresolved"
    joined_capture: str = "unresolved"
    scope: tuple[str, ...] = (
        "two_original_donor_ramps_and_nominally_flat_receiver_on_equal_support",
        "complete_degree_four_law_held_unit_capacity_structural_clock",
        "all_initial_internal_and_contact_storage_retained",
        "initial_component_error_sums_zero_only_global_means_conserved",
        "commuting_heat_bound_and_full_nonlinear_history_exclude_early_slip",
        "delay_is_a_necessary_condition_not_an_eventual_obstruction",
        "oriented_current_is_mixed_coordinate_motion_not_receiver_formation",
        "donor_endpoint_winding_is_not_a_future_retention_certificate",
        "shared_target_and_barrier_supply_no_source_entry_verdict",
        "no_trajectory_campaign_source_retuning_or_physical_identification",
    )


def _assess_organized_receiver_mechanism(
    *, form_error_bound, phase_error_bound
) -> _OrganizedReceiverMechanism:
    """Rebuild the fixed-source mechanism from admitted primitive allowances."""
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    capture = _derive_organized_triangle_capture(mixing=_EQUAL_CONTROL)
    geometry = capture.geometry
    forms = tuple(
        _SOURCE_SCALE * (node % 9 - 4) if node < 18 else Q(0) for node in geometry.nodes
    )
    costs, gradient = _prepared_form_costs(
        form=forms, weighted_edges=geometry.weighted_edges
    )
    internal = sum(
        (
            cost
            for (a, b, _), cost in zip(geometry.weighted_edges, costs)
            if a // 9 == b // 9
        ),
        Q(0),
    )
    contact = sum(costs, Q(0)) - internal
    gradient_l1 = sum(abs(value) for value in gradient)
    correction = gradient_l1 * rx + 108 * (rx**2 + rt**2)
    pi = pi_interval()
    gamma = geometry.gamma_bounds
    epsilon = gamma**2
    alpha = 2 * pi / 9
    norm, root_two = sqrt(I(27)), sqrt(I(2))
    ex, et = (norm * rx).hi, (norm * rt).hi
    initial_norm = gamma.hi * (sqrt(I(sum(value**2 for value in forms))).hi + ex)
    gap = capture.laplacian_gap_lower_bound * geometry.clock_factor
    decay = I(*_negative_exp_bounds(gap * _LAYER_TIME))
    scaled_form, phase_remainder = _prepared_transit_radii(
        time=_LAYER_TIME,
        decay_upper=decay.hi,
        gap_lower=gap,
        initial_norm_upper=initial_norm,
        feedback_upper=epsilon.hi,
        forcing_upper=norm.hi,
    )
    proxy = sqrt(I(120)) * abs(gamma * _SOURCE_SCALE - alpha)
    phase_radius = proxy.hi + phase_remainder + gamma.hi * ex + et
    form_radius = (1 / gamma).hi * scaled_form
    donor_margin = pi.lo / 2 - alpha.hi - root_two.hi * phase_radius
    receiver_margin = pi.lo - root_two.hi * phase_radius
    # The strict rational inequality bounds the maximum heat-shape factor
    # (1-exp(-3*t/4))*exp(-t/10) by 3/4 for every nonnegative t.
    if not 4 * 20**15 < 17**17:
        raise ArithmeticError("receiver heat-shape bound failed")
    early_edge = (
        4 * gamma.hi * _SOURCE_SCALE
        + 2 * rt
        + 4 * gamma.hi * rx
        + 4 * epsilon.hi * _LAYER_TIME
    )
    early_margin = pi.lo - early_edge
    offset = 2 * (rt + gamma.hi * rx) + root_two.hi * scaled_form
    growth = 2 * epsilon.hi
    delay = (pi.lo - offset) / growth
    initial_ok = pi.lo - 2 * rt > 0
    layer_ok = donor_margin > 0 and receiver_margin > 0
    early_ok = early_margin > 0
    delay_ok = initial_ok and early_ok and delay > _LAYER_TIME
    phases = tuple(alpha * (node % 9 - 4) if node < 18 else I(0) for node in range(27))
    sines = tuple(sin(alpha * (j - 4)) for j in range(9))
    current = (
        tuple(sin(alpha) / 2 - value for value in sines)
        + tuple(-sin(alpha) / 2 - value for value in sines)
        + tuple((1 + cos(alpha)) * value for value in sines)
    )
    current_error = 4 * root_two.hi * phase_radius
    oriented = min(
        (-current[18 + j].hi if j < 4 else current[18 + j].lo) - current_error
        for j in range(9)
        if j != 4
    )
    orientation_ok = oriented > 0
    unresolved = tuple(
        reason
        for passed, reason in (
            (initial_ok, "initial_zero_winding_not_certified"),
            (layer_ok, "layer_donor_one_receiver_zero_not_certified"),
            (early_ok, "early_receiver_zero_winding_not_certified"),
            (delay_ok, "receiver_delay_not_certified"),
            (orientation_ok, "receiver_current_orientation_not_certified"),
        )
        if not passed
    )
    return _OrganizedReceiverMechanism(
        capture_geometry=capture,
        form_error_bound=rx,
        phase_error_bound=rt,
        relative_preparation_dimension=24 * (int(rx > 0) + int(rt > 0)),
        nominal_initial_forms=forms,
        nominal_initial_form_gradient=gradient,
        nominal_initial_phase_rate_bounds=tuple(
            gamma * value / 4 for value in gradient
        ),
        nominal_initial_internal_form_storage=internal,
        nominal_initial_contact_form_storage=contact,
        nominal_initial_form_storage=internal + contact,
        initial_form_gradient_l1=gradient_l1,
        initial_storage_correction_upper_bound=correction,
        initial_storage_upper_bound=internal + contact + correction,
        scaled_initial_norm_upper_bound=initial_norm,
        semigroup_gap_lower_bound=gap,
        exponential_decay_bounds=decay,
        forcing_norm_bounds=norm,
        slow_clock_factor_bounds=epsilon,
        reference_phase_bounds=phases,
        proxy_reference_distance_bounds=proxy,
        layer_scaled_form_norm_upper_bound=scaled_form,
        layer_form_norm_upper_bound=form_radius,
        layer_phase_error_norm_upper_bound=phase_radius,
        layer_donor_acute_margin=donor_margin,
        layer_receiver_zero_winding_margin=receiver_margin,
        early_receiver_edge_upper_bound=early_edge,
        early_receiver_zero_winding_margin=early_margin,
        late_receiver_edge_offset_upper_bound=offset,
        late_receiver_edge_growth_upper_bound=growth,
        receiver_exclusion_time_lower_bound=delay,
        receiver_exclusion_until=delay if delay_ok else None,
        reference_current_bounds=current,
        layer_current_error_upper_bound=current_error,
        layer_receiver_oriented_current_lower_bound=oriented,
        initial_zero_winding_certified=initial_ok,
        layer_windings=(1, 1, 0) if layer_ok else None,
        early_receiver_zero_winding_certified=early_ok,
        receiver_delay_certified=delay_ok,
        receiver_oriented_current_certified=orientation_ok,
        status=(
            "certified_receiver_delay_and_current" if not unresolved else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
