"""Detached outer winding acquisition by five actual recovered C9 sources.

ORGANIZED_OUTER_WINDING.md owns the frozen preparation and product-cell
proof. The primitive C5 calculation is rebuilt on its admitted five-node
domain; a separate full-fine comparison retains all ninety coordinates.
No trajectory, support event, source reset or frozen producer is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, cos, pi_interval, sqrt
from ._repeated_organized_composition import _bounded_sine_edge_fields, _vector
from ._sine_constitutive_recovery import (
    _ConstitutiveRecovery,
    _assess_constitutive_recovery,
)
from ._sine_preparation import _prepared_form_costs, _prepared_heat_error_radii
from .relational_sine_comparison import bound_relational_sine_exchange
from .relational_sine_entry import SinePreparedEntry, certify_sine_prepared_entry

_RECOVERED_RADIUS = Q(1, 10**70)
_PREPARATION_ALLOWANCE = Q(1, 10**10)
_HORIZON = Q(200)
_BUDGET = Q(2 * 10**9)
_ORIGINAL_CLOCK_FACTOR = Q(1023, 1024)
_FORM_ORIGINS = tuple(Q(4092 * (c - 2)) for c in range(5))
_PRIMITIVE_EDGES = ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
_PRIMITIVE_TURNS = (0, -1, 0, 0, 0)


@dataclass(frozen=True)
class _OrganizedOuterWindingGeometry:
    internal_edges: tuple[tuple[int, int, Q], ...]
    outer_edges: tuple[tuple[int, int, Q], ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    internal_cycles: tuple[tuple[int, ...], ...]
    outer_cycles: tuple[tuple[int, ...], ...]
    alpha_bounds: I
    gamma_bounds: I
    outer_target_angle_bounds: I
    node_order: tuple[int, ...] = tuple(range(45))
    support_degrees: tuple[int, ...] = (4,) * 45
    weighted_degrees: tuple[Q, ...] = (Q(4),) * 45
    held_capacities: tuple[Q, ...] = (Q(1),) * 45
    clock_factor: Q = Q(1, 4)
    clock: str = "tau=(1023/1024)*t; primitive_sigma=tau/2"
    phase_domain: str = "regular"
    scope: tuple[str, ...] = (
        "supplied_unit_C9_cartesian_C5_support_with_all_ninety_edges",
        "uniform_quarter_factor_in_both_complete_rows_before_and_after_attachment",
        "independent_fine_coordinates_with_held_unit_capacity_and_no_inputs",
        "oriented_internal_j_to_j_plus_one_and_outer_c_to_c_plus_one_cycles",
        "not_a_bypass_of_the_generic_thirty_two_node_geometry_budget",
    )


def _derive_organized_outer_winding() -> _OrganizedOuterWindingGeometry:
    """Construct the fixed product incidence, without a state or verdict."""
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    outer = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    pi = pi_interval()
    return _OrganizedOuterWindingGeometry(
        internal_edges=internal,
        outer_edges=outer,
        weighted_edges=internal + outer,
        internal_cycles=tuple(tuple(9 * c + j for j in range(9)) for c in range(5)),
        outer_cycles=tuple(tuple(9 * c + j for c in range(5)) for j in range(9)),
        alpha_bounds=2 * pi / 9,
        gamma_bounds=Q(1, 1023) / pi,
        outer_target_angle_bounds=2 * pi / 5,
    )


@dataclass(frozen=True)
class _OrganizedOuterWindingObservation:
    geometry: _OrganizedOuterWindingGeometry
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
    child_phase_offset_means: tuple[Q, ...]
    child_form_tau_rate_bounds: tuple[I, ...]
    child_phase_tau_rate_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    storage_tau_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I


def _observe_organized_outer_winding(*, form, phase_offsets):
    """Observe full rows from admitted signed forms and lifted carrier offsets."""
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    geometry = _derive_organized_outer_winding()
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    current, potentials, _ = _bounded_sine_edge_fields(
        phase_offsets=offsets,
        weighted_edges=geometry.weighted_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    dx = tuple(k * (-g + gamma * s) for g, s in zip(gradient, current))
    dt = tuple(k * gamma * g for g in gradient)
    loss = k * sum((g**2 for g in gradient), Q(0))
    direct = sum((g * x - s * t for g, s, x, t in zip(gradient, current, dx, dt)), I(0))

    def means(row, zero):
        return tuple(sum(row[9 * c : 9 * c + 9], zero) / 9 for c in range(5))

    form_storage = sum(costs, Q(0))
    phase_storage = sum(potentials, I(0))
    return _OrganizedOuterWindingObservation(
        geometry=geometry,
        form=values,
        phase_offsets=offsets,
        phase_lift_bounds=tuple(
            geometry.alpha_bounds * (i % 9 - 4) + offset
            for i, offset in enumerate(offsets)
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
        form_storage=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=form_storage + phase_storage,
        storage_tau_rate=-loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
    )


def _primitive_outer_source():
    """Fresh primitive C5 source; no earlier report or evidence file is read."""
    import networkx as nx

    graph = nx.cycle_graph(5)
    for c in range(5):
        graph.nodes[c].update(EPI=_FORM_ORIGINS[c], theta=Q(0), nu_f=Q(1))
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(
        1,
        epi_weight=Q(1023, 1024),
        phase_weight=Q(1, 1024),
        phase_domain="regular",
    )
    return bound_relational_sine_exchange(graph, reference_model=model)


@dataclass(frozen=True)
class _OrganizedOuterWindingControl:
    initial_excess_upper_bound: Q
    boundary_energy_lower_bound: Q
    capture_margin: Q
    source_certified: bool
    conditional_capture_certified: bool
    capture_certified: bool
    outer_zero_winding_retained: bool
    child_winding_retained: bool
    converges_to_target: bool
    form_origins: tuple[Q, ...] = (Q(0),) * 5
    phase_origins: tuple[Q, ...] = (Q(0),) * 5
    target_child_windings: tuple[int, ...] = (1,) * 5
    target_outer_windings: tuple[int, ...] = (0,) * 9
    scope: tuple[str, ...] = (
        "same_original_source_members_with_only_primitive_common_form_ramp_removed",
        "same_fine_residuals_support_capacities_clocks_and_phase_origins",
        "actual_residual_family_is_not_assumed_stationary",
        "complete_product_storage_trap_preserves_outer_zero_for_all_future_times",
    )


@dataclass(frozen=True)
class _OrganizedOuterWinding:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    child_sources: tuple[_ConstitutiveRecovery, ...]
    geometry: _OrganizedOuterWindingGeometry
    primitive_entry: SinePreparedEntry
    control: _OrganizedOuterWindingControl
    relative_preparation_dimension: int
    original_preparation_storage_upper_bound: Q
    recovered_source_flags: tuple[bool, ...]
    initial_full_form_error_upper_bound: Q
    initial_full_phase_error_upper_bound: Q
    full_form_error_upper_bound: Q
    full_phase_error_upper_bound: Q
    fine_error_growth_upper_bound: Q
    fine_error_convolution_factor: Q
    form_laplacian_gap_lower_bound: Q
    target_storage_bounds: I
    internal_cycle_minimum_bounds: I
    outer_cycle_minimum_bounds: I
    internal_positive_face_storage_bounds: I
    internal_negative_face_storage_bounds: I
    outer_positive_face_storage_bounds: I
    internal_boundary_excess_bounds: I
    outer_boundary_excess_bounds: I
    boundary_energy_lower_bound: Q
    signed_face_excess_lower_bounds: tuple[Q | None, ...]
    signed_face_feasible: tuple[bool, ...]
    endpoint_nominal_excess_upper_bound: Q
    endpoint_residual_excess_upper_bound: Q
    endpoint_excess_upper_bound: Q
    capture_margin: Q
    initial_internal_phase_gap_bounds: I
    transit_internal_phase_gap_bounds: I
    initial_outer_phase_gap_bounds: I
    endpoint_oriented_outer_phase_gap_bounds: tuple[I, ...]
    attachment_nominal_work: Q
    attachment_residual_work_upper_bound: Q
    attachment_work_upper_bound: Q
    attachment_work_margin: Q
    initial_storage_upper_bound: Q
    initial_storage_margin: Q
    source_certified: bool
    primitive_law_and_clock_certified: bool
    fine_error_certified: bool
    barrier_certified: bool
    attachment_budget_certified: bool
    initial_outer_zero_winding_certified: bool
    child_winding_retained_during_transit: bool
    conditional_outer_entry_certified: bool
    outer_entry_certified: bool
    conditional_capture_certified: bool
    capture_certified: bool
    outer_winding_retained: bool
    child_winding_retained: bool
    converges_to_target: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    recovered_child_norm_bound: Q = _RECOVERED_RADIUS
    source_formation_time: Q = Q(200)
    source_relaxation_duration: Q = Q(2 * 10**13)
    source_handoff_time: Q = Q(200 + 2 * 10**13)
    elapsed_scaled_time: Q = _HORIZON
    elapsed_original_time: Q = _HORIZON / _ORIGINAL_CLOCK_FACTOR
    primitive_scaled_time: Q = _HORIZON / 2
    initial_child_form_origins: tuple[Q, ...] = _FORM_ORIGINS
    initial_child_phase_origins: tuple[Q, ...] = (Q(0),) * 5
    target_child_windings: tuple[int, ...] = (1,) * 5
    target_outer_windings: tuple[int, ...] = (1,) * 9
    target_form: tuple[Q, ...] = (Q(0),) * 45
    target_phase_turns: tuple[Q, ...] = tuple(
        Q(j - 4, 9) + Q(c - 2, 5) for c in range(5) for j in range(9)
    )
    budget: Q = _BUDGET
    signed_internal_boundary_faces: int = 90
    signed_outer_boundary_faces: int = 90
    infeasible_outer_boundary_faces: int = 45
    protocol_revision: str = "82825de5"
    scope: tuple[str, ...] = (
        "all_ten_original_primitives_admitted_before_five_independent_recoveries",
        "supplied_common_form_shifts_at_primitive_preparation_not_reached_state_reset",
        "quarter_factor_isolated_clock_conjugacy_transforms_both_rows",
        "fresh_admitted_primitive_C5_law_and_tau_to_sigma_clock_map",
        "full_ninety_coordinate_error_bound_through_nonacute_outer_transit",
        "five_internal_and_nine_outer_marked_cycles_observed_separately",
        "all_product_boundary_faces_with_negative_outer_faces_proved_empty",
        "complete_storage_includes_five_internal_and_nine_outer_cycle_terms",
        "strict_product_capture_retains_all_windings_and_gives_asymptotic_convergence",
        "positive_attachment_work_is_supplied_without_credit_for_previous_loss",
        "unavailable_is_not_failed_formation_or_a_nonentry_obstruction",
        "no_trajectory_frozen_producer_event_selection_or_physical_identity",
    )


def _assess_organized_outer_winding(*, form_error_bounds, phase_error_bounds):
    """Assess the frozen proof after admitting ten allowances in [0, 1e-10].

    Smaller allowances select subsets of the frozen source family. Larger
    allowances are outside this protocol even if isolated recovery could
    certify them; reject every unsupported primitive before source rebuilding.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 5)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 5)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    if max(*rx, *rt) > _PREPARATION_ALLOWANCE:
        raise ValueError("original preparation errors must not exceed frozen 1e-10")
    children = tuple(
        _assess_constitutive_recovery(
            deformation=0, form_error_bound=a, phase_error_bound=b
        )
        for a, b in zip(rx, rt)
    )
    geometry = _derive_organized_outer_winding()
    gamma, pi = geometry.gamma_bounds, pi_interval()
    q = _RECOVERED_RADIUS
    source_flags = tuple(
        child.recovery_certified
        and child.endpoint_form_norm_squared_upper_bound is not None
        and child.endpoint_phase_norm_squared_upper_bound is not None
        and child.endpoint_form_norm_squared_upper_bound < q**2
        and child.endpoint_phase_norm_squared_upper_bound < q**2
        for child in children
    )
    source_ok = all(source_flags)

    # This shared consumer re-admits primitive state, law, capacity and Gamma,
    # rebuilding all derived fields. No incoming C5 verdict is consumed.
    primitive = certify_sine_prepared_entry(
        _primitive_outer_source(),
        scaled_time=_HORIZON / 2,
        edge_turn_offsets=_PRIMITIVE_TURNS,
    )
    law_ok = (
        primitive.geometry.nodes == tuple(range(5))
        and primitive.geometry.edges == _PRIMITIVE_EDGES
        and primitive.source.epi == _FORM_ORIGINS
        and primitive.source.phase == (Q(0),) * 5
        and primitive.source.capacity == (Q(1),) * 5
        and primitive.reference_model.effective_weights == (Q(1023, 1024), Q(1, 1024))
        and primitive.reference_model.storage_scale == 1
        and primitive.reference_model.phase_domain == "regular"
        and primitive.scaled_time == _HORIZON / 2
        and primitive.horizon == _HORIZON / (2 * _ORIGINAL_CLOCK_FACTOR)
        and primitive.capture.edge_turn_offsets == _PRIMITIVE_TURNS
    )
    # A=L/4, ||L||<=8 and gap(L)>1/3. The heat-integral phase
    # estimate has growth exp(2 gamma^2 tau)<2. Keep q exact, below
    # the interval grid, and use sqrt(5)<3 only as a norm bound.
    fine_ok = (
        (2 - 2 * cos(geometry.alpha_bounds)).lo > Q(1, 3)
        and (2 - 2 * cos(geometry.outer_target_angle_bounds)).lo > Q(1, 3)
        and 2 * _HORIZON * gamma.hi**2 < Q(1, 2)
    )
    full_x, full_t = _prepared_heat_error_radii(
        initial_form_error=3 * q,
        initial_phase_error=3 * q,
        gamma_upper=gamma.hi,
        phase_growth_upper=Q(2),
        form_convolution_factor=Q(24),
    )
    alpha, beta = geometry.alpha_bounds, geometry.outer_target_angle_bounds
    v9, v5 = 9 * (1 - cos(alpha)), 5 * (1 - cos(beta))
    b9 = 1 + 8 * (1 - cos(3 * pi / 16))
    negative_b9 = 1 + 8 * (1 - cos(5 * pi / 16))
    b5 = 1 + 4 * (1 - cos(3 * pi / 8))
    internal_excess, outer_excess = b9 - v9, b5 - v5
    boundary = min(internal_excess.lo, outer_excess.lo)
    barrier_ok = (
        boundary > 0
        and negative_b9.lo > b9.hi
        and 0 < alpha.lo <= alpha.hi < pi.lo / 2
        and 0 < beta.lo <= beta.hi < pi.lo / 2
    )
    # Face order is negative then positive for each oriented internal edge,
    # followed by each oriented outer edge. Outer negative faces are empty:
    # their other four acute gaps cannot sum to 5*pi/2.
    faces = tuple(
        value
        for _ in geometry.internal_edges
        for value in ((negative_b9 - v9).lo, internal_excess.lo)
    ) + tuple(value for _ in geometry.outer_edges for value in (None, outer_excess.lo))
    feasible = tuple(value is not None for value in faces)
    nominal_excess = 9 * (primitive.capture.storage_upper_bound - v5.lo)
    node_form = max(box.abs_max for box in primitive.endpoint_form_bounds)
    nominal_form_norm = 3 * sqrt(I(5)).hi * node_form
    # Along the carrier internal sine currents cancel, so ||S||<=2sqrt45<14.
    # Global Hessian norm <=8 bounds the quadratic remainder off the carrier.
    residual_excess = (
        8 * nominal_form_norm * full_x + 4 * full_x**2 + 14 * full_t + 4 * full_t**2
    )
    endpoint_excess = nominal_excess + residual_excess
    margin = boundary - endpoint_excess
    initial_inner = alpha + I(-2 * q, 2 * q)
    transit_inner = alpha + I(-2 * full_t, 2 * full_t)
    initial_outer = I(-2 * q, 2 * q)
    primitive_gaps = primitive.capture.phase_edge_gap_bounds
    # Canonical 0->4 has turn -1; the marked closing 4->0 is its negative.
    oriented = (
        primitive_gaps[0],
        primitive_gaps[2],
        primitive_gaps[3],
        primitive_gaps[4],
        -primitive_gaps[1],
    )
    outer_gaps = tuple(gap + I(-2 * full_t, 2 * full_t) for gap in oriented)
    initial_zero = initial_outer.abs_max < pi.lo / 2
    child_transit = 0 < transit_inner.lo <= transit_inner.hi < pi.lo / 2
    outer_cell = all(0 < gap.lo <= gap.hi < pi.lo / 2 for gap in outer_gaps)
    nominal_work = 9 * sum(
        ((_FORM_ORIGINS[b] - _FORM_ORIGINS[a]) ** 2 / 2 for a, b in _PRIMITIVE_EDGES),
        Q(0),
    )
    residual_work = 20 * q**2
    work = nominal_work + residual_work
    initial_storage = 5 * v9.hi + nominal_work + 40 * q**2
    work_margin, initial_margin = _BUDGET - work, _BUDGET - initial_storage
    budget_ok = work_margin > 0 and initial_margin > 0
    conditional_entry = (
        law_ok and fine_ok and initial_zero and child_transit and outer_cell
    )
    entry = source_ok and budget_ok and conditional_entry
    conditional_capture = conditional_entry and barrier_ok and margin > 0
    capture = source_ok and budget_ok and conditional_capture
    control_excess = 40 * q**2
    control_boundary = min(internal_excess.lo, Q(1))
    control_margin = control_boundary - control_excess
    control_candidate = (
        barrier_ok
        and 0 < initial_inner.lo <= initial_inner.hi < pi.lo / 2
        and initial_zero
        and control_margin > 0
    )
    control_ok = source_ok and control_candidate
    control = _OrganizedOuterWindingControl(
        initial_excess_upper_bound=control_excess,
        boundary_energy_lower_bound=control_boundary,
        capture_margin=control_margin,
        source_certified=source_ok,
        conditional_capture_certified=control_candidate,
        capture_certified=control_ok,
        outer_zero_winding_retained=control_ok,
        child_winding_retained=control_ok,
        converges_to_target=control_ok,
    )
    checks = (
        (source_ok, "original_child_recoveries_not_certified"),
        (law_ok, "primitive_C5_complete_law_or_clock_not_certified"),
        (fine_ok, "full_fine_heat_comparison_not_certified"),
        (budget_ok, "attachment_work_or_initial_storage_budget_not_certified"),
        (initial_zero, "initial_outer_zero_winding_not_certified"),
        (child_transit, "child_winding_retention_during_transit_not_certified"),
        (outer_cell, "nine_outer_cycle_entry_not_certified"),
        (barrier_ok, "complete_product_boundary_not_certified"),
        (margin > 0, "complete_fine_endpoint_capture_margin_not_positive"),
        (control_ok, "paired_zero_form_control_not_certified"),
    )
    unresolved = tuple(label for passed, label in checks if not passed)
    admitted = not unresolved
    return _OrganizedOuterWinding(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        child_sources=children,
        geometry=geometry,
        primitive_entry=primitive,
        control=control,
        relative_preparation_dimension=sum(
            c.source.preparation_dimension for c in children
        ),
        original_preparation_storage_upper_bound=sum(
            (c.source.initial_storage_upper_bounds[0] for c in children), Q(0)
        ),
        recovered_source_flags=source_flags,
        initial_full_form_error_upper_bound=3 * q,
        initial_full_phase_error_upper_bound=3 * q,
        full_form_error_upper_bound=full_x,
        full_phase_error_upper_bound=full_t,
        fine_error_growth_upper_bound=Q(2),
        fine_error_convolution_factor=Q(24),
        form_laplacian_gap_lower_bound=Q(1, 3),
        target_storage_bounds=5 * v9 + 9 * v5,
        internal_cycle_minimum_bounds=v9,
        outer_cycle_minimum_bounds=v5,
        internal_positive_face_storage_bounds=b9,
        internal_negative_face_storage_bounds=negative_b9,
        outer_positive_face_storage_bounds=b5,
        internal_boundary_excess_bounds=internal_excess,
        outer_boundary_excess_bounds=outer_excess,
        boundary_energy_lower_bound=boundary,
        signed_face_excess_lower_bounds=faces,
        signed_face_feasible=feasible,
        endpoint_nominal_excess_upper_bound=nominal_excess,
        endpoint_residual_excess_upper_bound=residual_excess,
        endpoint_excess_upper_bound=endpoint_excess,
        capture_margin=margin,
        initial_internal_phase_gap_bounds=initial_inner,
        transit_internal_phase_gap_bounds=transit_inner,
        initial_outer_phase_gap_bounds=initial_outer,
        endpoint_oriented_outer_phase_gap_bounds=outer_gaps * 9,
        attachment_nominal_work=nominal_work,
        attachment_residual_work_upper_bound=residual_work,
        attachment_work_upper_bound=work,
        attachment_work_margin=work_margin,
        initial_storage_upper_bound=initial_storage,
        initial_storage_margin=initial_margin,
        source_certified=source_ok,
        primitive_law_and_clock_certified=law_ok,
        fine_error_certified=fine_ok,
        barrier_certified=barrier_ok,
        attachment_budget_certified=budget_ok,
        initial_outer_zero_winding_certified=source_ok and initial_zero,
        child_winding_retained_during_transit=source_ok and fine_ok and child_transit,
        conditional_outer_entry_certified=conditional_entry,
        outer_entry_certified=entry,
        conditional_capture_certified=conditional_capture,
        capture_certified=capture,
        outer_winding_retained=capture,
        child_winding_retained=capture,
        converges_to_target=capture,
        admitted=admitted,
        status="certified_organized_outer_winding" if admitted else "unavailable",
        unresolved_conditions=unresolved,
    )
