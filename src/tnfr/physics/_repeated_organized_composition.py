"""Detached two-stage composition of four registered winding-one C9s.

REPEATED_ORGANIZED_COMPOSITION.md owns the conditional theorem. The fixed
36-node law, static observations and analytic response bounds install no
trajectory, support event, preparation or operator selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real as _exact
from .._exact_time import exp_unit_bounds
from ..mathematics._comparison_flow import scalar_affine_upper
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ._sine_constitutive_inheritance import _assess_acquired_constitutive_inheritance
from ._sine_preparation import _prepared_form_costs
from .relational_observations import _ordered

_SIZE = 36
_HORIZON = Q(1)
_DELTA = Q(1, 20)
_RADIUS = Q(1001, 100000)
_PREPARATION_RADIUS = Q(1, 100)
_RESOLUTION = Q(1, 10**7)
_INTERFACES = ("matching", "symmetric")


def _interface(value):
    if type(value) is not str or value not in _INTERFACES:
        raise ValueError("interface must be 'matching' or 'symmetric'")
    return value


def _vector(values, label, size):
    raw = _ordered(values, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} requires {size} registered coordinates")
    return tuple(_exact(value, f"{label}[{i}]") for i, value in enumerate(raw))


def _radii(form_radius, phase_radius):
    rx, rt = _exact(form_radius, "form_radius"), _exact(phase_radius, "phase_radius")
    if min(rx, rt) < 0:
        raise ValueError("initial residual sup radii must be nonnegative")
    return rx, rt


@dataclass(frozen=True)
class _RepeatedCompositionGeometry:
    """Fixed support and exact-family chart, without source admission."""

    interface: str
    internal_edges: tuple[tuple[int, int, Q], ...]
    parent_edges: tuple[tuple[int, int, Q], ...]
    outer_edges: tuple[tuple[int, int, Q], ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    weighted_degrees: tuple[Q, ...]
    support_degrees: tuple[int, ...]
    alpha_bounds: I
    gamma_bounds: I
    kappa_bounds: I
    form_chart_scale_bounds: I
    effective_gamma_bounds: I
    clock_factor: Q = Q(1, 4)
    node_order: tuple[int, ...] = tuple(range(_SIZE))
    node_coordinates: str = "node=18*parent+9*child+j; parent,child in0,1; j in0..8"
    clock: str = "tau=(1023/1024)*t; sigma=tau/4"
    scope: tuple[str, ...] = (
        "held_unit_capacity_weighted_sine_law_without_forcing_or_events",
        "first_grouping_four_registered_C9s_second_grouping_two_matching_pairs",
        "outer_conductance_one_per_node_not_unique_neighbor_normalization",
        "exact_rigid_family_chart_does_not_admit_general_mean_only_closure",
    )


def _derive_repeated_composition(*, interface):
    name = _interface(interface)
    internal = tuple(
        (9 * child + j, 9 * child + (j + 1) % 9, Q(1))
        for child in range(4)
        for j in range(9)
    )
    parents = tuple(
        (18 * parent + j, 18 * parent + 9 + j, Q(1))
        for parent in range(2)
        for j in range(9)
    )
    offsets = (0,) if name == "matching" else (-1, 0, 1)
    outer = tuple(
        (9 * child + j, 18 + 9 * child + (j + offset) % 9, Q(1, len(offsets)))
        for child in range(2)
        for j in range(9)
        for offset in offsets
    )
    edges = internal + parents + outer
    alpha, gamma = 2 * pi_interval() / 9, Q(1, 1023) / pi_interval()
    kappa = I(1) if name == "matching" else (1 + 2 * cos(alpha)) / 3
    scale = sqrt(kappa)
    return _RepeatedCompositionGeometry(
        interface=name,
        internal_edges=internal,
        parent_edges=parents,
        outer_edges=outer,
        weighted_edges=edges,
        weighted_degrees=tuple(
            sum((w for a, b, w in edges if node in (a, b)), Q(0))
            for node in range(_SIZE)
        ),
        support_degrees=tuple(
            sum(node in (a, b) for a, b, _ in edges) for node in range(_SIZE)
        ),
        alpha_bounds=alpha,
        gamma_bounds=gamma,
        kappa_bounds=kappa,
        form_chart_scale_bounds=scale,
        effective_gamma_bounds=gamma * scale,
    )


@dataclass(frozen=True)
class _RepeatedCompositionObservation:
    """Actual fine rows and both projections from one retained complete state."""

    geometry: _RepeatedCompositionGeometry
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    phase_lift_bounds: tuple[I, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    child_form_means: tuple[Q, ...]
    child_phase_offset_means: tuple[Q, ...]
    parent_form_means: tuple[Q, Q]
    parent_phase_offset_means: tuple[Q, Q]
    child_form_rate_bounds: tuple[I, ...]
    child_phase_rate_bounds: tuple[I, ...]
    parent_form_rate_bounds: tuple[I, I]
    parent_phase_rate_bounds: tuple[I, I]
    parent_form_residual_sup: Q
    parent_phase_residual_sup: Q
    outer_mean_current_bounds: I
    rigid_outer_current_bounds: I
    projected_current_defect_bounds: I
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    internal_storage_bounds: I
    first_attachment_storage_bounds: I
    outer_attachment_work_bounds: I
    nominal_outer_work_bounds: I
    outer_work_residual_bound: Q
    outer_work_residual_bounds: tuple[Q, Q]
    dissipation: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    scope: tuple[str, ...] = (
        "exact_detached_signed_forms_and_radian_offsets_not_graph_recapture",
        "phase_lift_is_alpha_times_j_minus_four_plus_admitted_offset",
        "actual_full_weighted_rows_before_both_mean_projections",
        "parent_phase_uses_declared_lifts_not_the_vanishing_fine_resultant",
        "static_attachment_work_does_not_select_an_event_or_supply_source_history",
        "projection_defect_is_retained_not_replaced_by_an_exact_parent_law",
    )


def _observe_repeated_composition(*, interface, form, phase_offsets):
    """Observe complete exact detached state without projecting it to a source."""
    name = _interface(interface)
    values = _vector(form, "form", _SIZE)
    offsets = _vector(phase_offsets, "phase_offsets", _SIZE)
    geometry = _derive_repeated_composition(interface=name)
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )

    @lru_cache(maxsize=128)
    def trigonometry(gap):
        # At most 108 edge gaps and one parent gap, all admitted intervals.
        # Reuse arithmetic only within this observation, never state or verdicts.
        return sin(gap), cos(gap)

    current, potential = [I(0) for _ in range(_SIZE)], []
    edge_currents = []
    for a, b, weight in geometry.weighted_edges:
        gap = geometry.alpha_bounds * ((b % 9) - (a % 9)) + offsets[b] - offsets[a]
        sine, cosine = trigonometry(gap)
        edge_current = weight * sine
        current[a] += edge_current
        current[b] -= edge_current
        edge_currents.append(edge_current)
        potential.append(weight * (1 - cosine))
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    dx = tuple(k * (-q + gamma * s) for q, s in zip(gradient, current))
    dy = tuple(k * gamma * q for q in gradient)

    def means(vector, size):
        return tuple(
            sum(vector[start : start + size]) / size for start in range(0, 36, size)
        )

    def rate_means(vector, size):
        return tuple(
            sum(vector[start : start + size], I(0)) / size
            for start in range(0, 36, size)
        )

    xmeans, tmeans = means(values, 18), means(offsets, 18)
    rx = max(abs(value - xmeans[i // 18]) for i, value in enumerate(values))
    rt = max(abs(value - tmeans[i // 18]) for i, value in enumerate(offsets))
    outer_start = len(geometry.internal_edges) + len(geometry.parent_edges)
    total_costs = tuple(I(x) + y for x, y in zip(costs, potential))
    outer_current = sum(edge_currents[outer_start:], I(0)) / 18
    mean_sine, mean_cosine = trigonometry(I(tmeans[1] - tmeans[0]))
    rigid_current = geometry.kappa_bounds * mean_sine
    nominal_work = 9 * (xmeans[1] - xmeans[0]) ** 2 + 18 * (
        1 - geometry.kappa_bounds * mean_cosine
    )
    loss = k * sum((q**2 for q in gradient), Q(0))
    direct = sum((q * a - s * b for q, s, a, b in zip(gradient, current, dx, dy)), I(0))
    return _RepeatedCompositionObservation(
        geometry=geometry,
        form=values,
        phase_offsets=offsets,
        phase_lift_bounds=tuple(
            geometry.alpha_bounds * (i % 9 - 4) + value
            for i, value in enumerate(offsets)
        ),
        form_gradient=gradient,
        phase_current_bounds=tuple(current),
        form_rate_bounds=dx,
        phase_rate_bounds=dy,
        child_form_means=means(values, 9),
        child_phase_offset_means=means(offsets, 9),
        parent_form_means=xmeans,
        parent_phase_offset_means=tmeans,
        child_form_rate_bounds=rate_means(dx, 9),
        child_phase_rate_bounds=rate_means(dy, 9),
        parent_form_rate_bounds=rate_means(dx, 18),
        parent_phase_rate_bounds=rate_means(dy, 18),
        parent_form_residual_sup=rx,
        parent_phase_residual_sup=rt,
        outer_mean_current_bounds=outer_current,
        rigid_outer_current_bounds=rigid_current,
        projected_current_defect_bounds=outer_current - rigid_current,
        form_storage=sum(costs, Q(0)),
        phase_storage_bounds=sum(potential, I(0)),
        storage_bounds=sum(total_costs, I(0)),
        internal_storage_bounds=sum(total_costs[:36], I(0)),
        first_attachment_storage_bounds=sum(total_costs[36:outer_start], I(0)),
        outer_attachment_work_bounds=sum(total_costs[outer_start:], I(0)),
        nominal_outer_work_bounds=nominal_work,
        outer_work_residual_bound=36 * (rx**2 + rt**2),
        outer_work_residual_bounds=(-36 * rt**2, 36 * (rx**2 + rt**2)),
        dissipation=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
    )


@dataclass(frozen=True)
class _RepeatedCompositionAdmission:
    """Conditional finite theorem on a declared parent-mean-zero neighborhood.

    Candidate numerical bounds retain their premises when admission fails;
    only the status flags make a certified source-domain statement. Neither
    response error describes a measured trajectory or an inferred preparation.
    """

    form_radius: Q
    phase_radius: Q
    geometry: tuple[_RepeatedCompositionGeometry, ...]
    form_bootstrap_upper: Q
    phase_bootstrap_upper: Q
    form_bootstrap_margin: Q
    phase_bootstrap_margin: Q
    child_lower_gap_margin: Q
    child_acute_margin: Q
    child_branch_margin: Q
    parent_identity_margin: Q
    outer_current_defect_bound: Q
    phase_drift_bound: Q
    nonlinear_reduction_error_bound: Q
    explicit_response_error_bound: Q
    nominal_response_bounds: tuple[I, I]
    nominal_contrast_bounds: I
    contrast_margin: Q
    contrast_resolution_margin: Q
    resolution_margin: Q
    outer_attachment_nominal_work_bounds: tuple[I, I]
    outer_attachment_work_residual_bound: Q
    outer_attachment_work_residual_bounds: tuple[Q, Q]
    bootstrap_certified: bool
    identity_certified: bool
    response_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    horizon: Q = _HORIZON
    initial_parent_forms: tuple[Q, Q] = (Q(0), Q(0))
    initial_parent_phase_origins: tuple[Q, Q] = (-_DELTA / 2, _DELTA / 2)
    transverse_radius: Q = _RADIUS
    original_neighborhood_radius: Q = _PREPARATION_RADIUS
    response_resolution: Q = _RESOLUTION
    clock: str = "tau=(1023/1024)*t; horizon=1 in tau"
    source_premises: tuple[str, ...] = (
        "independent_form_and_phase_residuals_have_zero_sum_in_each_18_node_parent",
        "declared_parent_means_are_exact_not_inferred_from_an_interval_snapshot",
        "relative_parent_origin_is_supplied_preparation_information",
        "initial_residual_radii_bound_all_36_coordinates_in_sup_norm",
    )
    scope: tuple[str, ...] = (
        "fixed_two_control_analytic_admission_without_trajectory_or_parameter_scan",
        "bootstrap_retains_all_parent_mean_zero_fine_transverse_directions",
        "nonlinear_reduction_error_bounds_parent_X0_and_relative_phase_one_norm",
        "explicit_response_error_is_separate_from_nonlinear_reduction_error",
        "child_winding_one_and_parent_internal_registration_retained_for_the_prefix",
        "parent_identity_is_a_declared_registration_domain_not_physical_binding",
        "unavailable_bound_is_not_observed_dynamical_failure",
        "no_autonomous_source_attachment_or_second_level_formation_claim",
    )


def _assess_repeated_composition(*, form_radius, phase_radius):
    """Admit the fixed finite interaction from declared initial sup radii."""
    rx, rt = _radii(form_radius, phase_radius)
    geometry = tuple(
        _derive_repeated_composition(interface=name) for name in _INTERFACES
    )
    gamma, radius = geometry[0].gamma_bounds, _RADIUS
    form_upper = rx + gamma.hi * (2 * radius + radius**2 / 2)
    phase_upper = rt + 2 * gamma.hi * radius
    form_margin, phase_margin = radius - form_upper, radius - phase_upper
    child_lower = geometry[0].alpha_bounds.lo - 2 * radius
    child_acute = pi_interval().lo / 2 - geometry[0].alpha_bounds.hi - 2 * radius
    child_branch = pi_interval().lo - geometry[0].alpha_bounds.hi - 2 * radius
    parent_margin = Q(1, 4) - 2 * radius
    phase_drift = gamma.hi**2 / 8
    current_defect = 2 * radius**2
    reduction = scalar_affine_upper(0, gamma.hi, _HORIZON, gamma.hi * radius**2 / 2)
    error = gamma.hi * (phase_drift + current_defect) / 4
    relaxation = 1 - 1 / I(*exp_unit_bounds(Q(1, 2)))
    response = tuple(
        gamma * item.kappa_bounds * sin(I(_DELTA)) * relaxation / 2 for item in geometry
    )
    contrast = response[0] - response[1]
    contrast_margin = contrast.lo - 2 * error
    contrast_resolution_margin = contrast_margin - _RESOLUTION
    resolution_margin = _RESOLUTION - max(error, reduction)
    bootstrap = min(form_margin, phase_margin) > 0
    identity = (
        bootstrap and min(child_lower, child_acute, child_branch, parent_margin) > 0
    )
    resolved = identity and min(contrast_resolution_margin, resolution_margin) > 0
    reasons = tuple(
        message
        for valid, message in (
            (bootstrap, "transverse_bootstrap_not_certified"),
            (identity, "finite_identity_domain_not_certified"),
            (resolved, "finite_parent_response_resolution_not_certified"),
        )
        if not valid
    )
    return _RepeatedCompositionAdmission(
        form_radius=rx,
        phase_radius=rt,
        geometry=geometry,
        form_bootstrap_upper=form_upper,
        phase_bootstrap_upper=phase_upper,
        form_bootstrap_margin=form_margin,
        phase_bootstrap_margin=phase_margin,
        child_lower_gap_margin=child_lower,
        child_acute_margin=child_acute,
        child_branch_margin=child_branch,
        parent_identity_margin=parent_margin,
        outer_current_defect_bound=current_defect,
        phase_drift_bound=phase_drift,
        nonlinear_reduction_error_bound=reduction,
        explicit_response_error_bound=error,
        nominal_response_bounds=response,
        nominal_contrast_bounds=contrast,
        contrast_margin=contrast_margin,
        contrast_resolution_margin=contrast_resolution_margin,
        resolution_margin=resolution_margin,
        outer_attachment_nominal_work_bounds=tuple(
            18 * (1 - item.kappa_bounds * cos(I(_DELTA))) for item in geometry
        ),
        outer_attachment_work_residual_bound=36 * (rx**2 + rt**2),
        outer_attachment_work_residual_bounds=(-36 * rt**2, 36 * (rx**2 + rt**2)),
        bootstrap_certified=bootstrap,
        identity_certified=identity,
        response_certified=resolved,
        status=(
            "certified_repeated_organized_composition" if resolved else "unavailable"
        ),
        unresolved_conditions=reasons,
    )


@dataclass(frozen=True)
class _AcquiredRepeatedComposition:
    """Two fresh first-stage sources and a separately admitted second attachment."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    parent_sources: tuple
    incoming_residual_sup_bound: Q
    admission: _RepeatedCompositionAdmission
    sources_certified: bool
    admitted: bool
    status: str
    scope: tuple[str, ...] = (
        "two_independent_original_matching_interface_parent_families_rebuilt",
        "first_parent_attachment_and_its_external_work_are_retained",
        "parentwise_common_phase_origins_are_new_supplied_relative_registration",
        "common_phase_translation_before_second_attachment_preserves_internal_storage",
        "second_attachment_work_is_new_not_funded_by_previous_dissipation",
        "new_degree_four_finite_identity_admission_not_transferred_from_degree_three",
        "no_state_reset_autonomous_attachment_or_new_parent_formation_proof",
    )


def _assess_acquired_repeated_composition(*, form_error_bounds, phase_error_bounds):
    """Carry two independently prepared parents, with declared new phase origins."""
    rx = _vector(form_error_bounds, "form_error_bounds", 2)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 2)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    parents = tuple(
        _assess_acquired_constitutive_inheritance(
            interface="matching", deformation=0, form_error_bound=a, phase_error_bound=b
        )
        for a, b in zip(rx, rt)
    )
    incoming = max(parent.entire_future_norm_upper_bound for parent in parents)
    admission = _assess_repeated_composition(
        form_radius=incoming, phase_radius=incoming
    )
    source_ok = all(parent.identity_certified for parent in parents)
    admitted = source_ok and admission.response_certified
    return _AcquiredRepeatedComposition(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        parent_sources=parents,
        incoming_residual_sup_bound=incoming,
        admission=admission,
        sources_certified=source_ok,
        admitted=admitted,
        status="certified_acquired_repeated_composition" if admitted else "unavailable",
    )
