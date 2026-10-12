"""Own-law admission of the original five-child outer-winding preparation.

The fixed endpoint law is the matched ninth-harmonic potential at lambda=1.
This detached owner rebuilds five actual isolated sources and bounds their
complete product flow; it consumes no sine acquisition or capture verdict.
Proof: theory/nodal/ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from .._exact_time import exact_or_represented_real as _exact
from ._organized_outer_winding import (
    _OrganizedOuterWindingGeometry,
    _derive_organized_outer_winding,
)
from ._sine_class_constitutive_comparison import _current_derivative_bound, _edge_law
from ._sine_constitutive_recovery import (
    _ConstitutiveRecovery,
    _assess_constitutive_recovery,
)
from ._sine_preparation import (
    _prepared_coordinate_vector,
    _prepared_form_costs,
    _prepared_heat_error_radii,
    _prepared_transit_radii,
)

_ERROR = Q(1, 10**10)
_RADIUS = Q(1, 10**70)
_HORIZON = Q(200)
_BUDGET = Q(2 * 10**9)
_SHIFTS = tuple(Q(4092 * (c - 2)) for c in range(5))
_CLOCK = Q(1023, 1024)
_DEFORMATION = Q(1)


def _potential(angle: I) -> I:
    """Enclose the declared potential using its shared canonical edge owner."""
    return _edge_law(cos(angle), sin(angle), _DEFORMATION)[0]


@dataclass(frozen=True)
class _AlternativeOuterObservation:
    geometry: _OrganizedOuterWindingGeometry
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    storage_tau_rate: Q
    balance_residual_bounds: I
    deformation: Q = _DEFORMATION


def _observe_organized_alternative_outer(
    *, form: object, phase_offsets: object
) -> _AlternativeOuterObservation:
    """Observe both complete rows on the admitted fixed product support.

    Offsets are real lifts from the internal C9 profile. They are supplied
    state, not a reached-source claim, circular lifts or collective closure.
    """
    values = _prepared_coordinate_vector(form, "form", 45)
    offsets = _prepared_coordinate_vector(phase_offsets, "phase_offsets", 45)
    geometry = _derive_organized_outer_winding()
    costs, gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    current, potentials = [I(0) for _ in values], []
    for a, b, weight in geometry.weighted_edges:
        angle = geometry.alpha_bounds * (b % 9 - a % 9) + offsets[b] - offsets[a]
        potential, derivatives = _edge_law(cos(angle), sin(angle), _DEFORMATION)
        current[a] += weight * derivatives[0]
        current[b] -= weight * derivatives[0]
        potentials.append(weight * potential)
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    dx = tuple(k * (-a + gamma * b) for a, b in zip(gradient, current))
    dt = tuple(k * gamma * a for a in gradient)
    loss = k * sum((a**2 for a in gradient), Q(0))
    direct = sum((a * u - b * v for a, b, u, v in zip(gradient, current, dx, dt)), I(0))
    form_storage = sum(costs, Q(0))
    phase_storage = sum(potentials, I(0))
    return _AlternativeOuterObservation(
        geometry=geometry,
        form=values,
        phase_offsets=offsets,
        form_gradient=gradient,
        phase_current_bounds=tuple(current),
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dt,
        form_original_rate_bounds=tuple(_CLOCK * value for value in dx),
        phase_original_rate_bounds=tuple(_CLOCK * value for value in dt),
        form_storage=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=form_storage + phase_storage,
        storage_tau_rate=-loss,
        balance_residual_bounds=direct + loss,
    )


@dataclass(frozen=True)
class _AlternativeOuterBounds:
    """Conditional full-product bounds with no actual source certificate."""

    geometry: _OrganizedOuterWindingGeometry
    recovered_child_norm_bound: Q
    current_absolute_upper_bound: Q
    current_lipschitz_upper_bound: Q
    primitive_initial_scaled_norm_bounds: I
    primitive_forcing_norm_bounds: I
    primitive_decay_upper_bound: Q
    primitive_scaled_form_radius_upper_bound: Q
    primitive_phase_radius_upper_bound: Q
    primitive_node_form_radius_upper_bound: Q
    primitive_node_phase_radius_upper_bound: Q
    primitive_edge_form_radius_upper_bound: Q
    primitive_oriented_phase_gap_bounds: tuple[I, ...]
    primitive_endpoint_storage_upper_bound: Q
    full_form_error_upper_bound: Q
    full_phase_error_upper_bound: Q
    acute_convexity_split_margin: Q
    target_internal_stiffness_bounds: I
    target_outer_stiffness_bounds: I
    internal_cycle_minimum_bounds: I
    outer_cycle_minimum_bounds: I
    target_storage_bounds: I
    internal_positive_face_storage_bounds: I
    internal_negative_face_storage_bounds: I
    outer_positive_face_storage_bounds: I
    internal_boundary_excess_bounds: I
    outer_boundary_excess_bounds: I
    signed_face_excess_lower_bounds: tuple[Q | None, ...]
    signed_face_feasible: tuple[bool, ...]
    boundary_energy_lower_bound: Q
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
    control_initial_excess_upper_bound: Q
    control_boundary_energy_lower_bound: Q
    control_capture_margin: Q
    primitive_law_and_clock_certified: bool
    fine_error_certified: bool
    acute_convexity_certified: bool
    target_regular_certified: bool
    barrier_certified: bool
    attachment_budget_certified: bool
    initial_outer_zero_winding_certified: bool
    child_transit_cell_certified: bool
    outer_cell_certified: bool
    conditional_outer_entry_certified: bool
    conditional_capture_certified: bool
    control_conditional_capture_certified: bool
    deformation: Q = _DEFORMATION
    elapsed_scaled_time: Q = _HORIZON
    primitive_scaled_time: Q = _HORIZON / 2
    primitive_gap_lower_bound: Q = Q(1, 2)
    full_laplacian_gap_lower_bound: Q = Q(1, 3)
    fine_phase_growth_upper_bound: Q = Q(2)
    arithmetic_method: str = INTERVAL_METHOD


def _derive_organized_alternative_outer_bounds(
    *, recovered_radius: object
) -> _AlternativeOuterBounds:
    """Derive fixed own-law bounds after exact admission of the residual radius.

    The chosen coarse exponential and direct edge Cauchy bounds are fixed
    before assessment. No source, trajectory or earlier verdict is consumed.
    """
    q = _exact(recovered_radius, "recovered_radius")
    if q < 0:
        raise ValueError("recovered_radius must be nonnegative")
    geometry = _derive_organized_outer_winding()
    pi, gamma = pi_interval(), geometry.gamma_bounds
    alpha, beta = geometry.alpha_bounds, geometry.outer_target_angle_bounds
    current = _current_derivative_bound(_DEFORMATION, 0)
    stiffness = _current_derivative_bound(_DEFORMATION, 1)
    initial = sqrt(I(320)) / pi
    forcing = sqrt(I(10)) * current
    decay = (1 + Q(50, 16)) ** -16
    a, p = _prepared_transit_radii(
        time=_HORIZON / 2,
        decay_upper=decay,
        gap_lower=Q(1, 2),
        initial_norm_upper=initial.hi,
        feedback_upper=(gamma**2).hi,
        forcing_upper=forcing.hi,
    )
    edge_form = a * (1023 * pi).hi
    node_form = edge_form * (1 / sqrt(I(2))).hi
    node_phase = p * (1 / sqrt(I(2))).hi
    gaps = (4 / pi + I(-p, p),) * 4 + (2 * pi - 16 / pi + I(-p, p),)
    primitive_acute = all(0 < gap.lo <= gap.hi < pi.lo / 2 for gap in gaps)

    # Convexity on the closed acute interval bounds the supremum by endpoint
    # values. If that domain is unavailable, retain a full interval bound.
    def potential_upper(gap: I) -> Q:
        if primitive_acute:
            return max(_potential(I(gap.lo)).hi, _potential(I(gap.hi)).hi)
        return _potential(gap).hi

    u5 = Q(5, 2) * edge_form**2 + sum(map(potential_upper, gaps), Q(0))
    full_x, full_t = _prepared_heat_error_radii(
        initial_form_error=3 * q,
        initial_phase_error=3 * q,
        gamma_upper=gamma.hi,
        phase_growth_upper=Q(2),
        form_convolution_factor=24 * stiffness,
    )
    fine_ok = (
        (2 - 2 * cos(alpha)).lo > Q(1, 3)
        and (2 - 2 * cos(beta)).lo > Q(1, 3)
        and 2 * _HORIZON * stiffness * gamma.hi**2 < Q(1, 2)
    )
    split_margin = cos(4 * pi / 9).lo - Q(2, 243)
    convex = split_margin > 0
    inner_stiffness = cos(alpha)
    outer_stiffness = _edge_law(cos(beta), sin(beta), _DEFORMATION)[1][1]
    target_ok = convex and inner_stiffness.lo > 0 and outer_stiffness.lo > 0
    # The deformation vanishes exactly at alpha; do not round that identity.
    v9, v5 = 9 * (1 - cos(alpha)), 5 * _potential(beta)
    face = _potential(pi / 2)
    b9 = face + 8 * _potential(3 * pi / 16)
    negative_b9 = face + 8 * _potential(5 * pi / 16)
    b5 = face + 4 * _potential(3 * pi / 8)
    inner_excess, outer_excess = b9 - v9, b5 - v5
    boundary = min(inner_excess.lo, outer_excess.lo)
    barrier_ok = convex and target_ok and boundary > 0 and negative_b9.lo > b9.hi
    faces = tuple(
        value
        for _ in geometry.internal_edges
        for value in ((negative_b9 - v9).lo, inner_excess.lo)
    ) + tuple(value for _ in geometry.outer_edges for value in (None, outer_excess.lo))
    nominal_excess = 9 * (u5 - v5.lo)
    residual_excess = (
        8 * sqrt(I(45)).hi * node_form * full_x
        + 4 * full_x**2
        + 14 * current * full_t
        + 4 * stiffness * full_t**2
    )
    endpoint_excess = nominal_excess + residual_excess
    initial_inner = alpha + I(-2 * q, 2 * q)
    transit_inner = alpha + I(-2 * full_t, 2 * full_t)
    initial_outer = I(-2 * q, 2 * q)
    outer_gaps = tuple(gap + I(-2 * full_t, 2 * full_t) for gap in gaps)
    initial_zero = initial_outer.abs_max < pi.lo / 2
    child_cell = 0 < transit_inner.lo <= transit_inner.hi < pi.lo / 2
    outer_cell = all(0 < gap.lo <= gap.hi < pi.lo / 2 for gap in outer_gaps)
    nominal_work = 90 * Q(4092) ** 2
    residual_work = 10 * (1 + stiffness) * q**2
    work = nominal_work + residual_work
    initial_storage = 5 * v9.hi + nominal_work + 20 * (1 + stiffness) * q**2
    work_margin, initial_margin = _BUDGET - work, _BUDGET - initial_storage
    budget_ok = work_margin > 0 and initial_margin > 0
    law_ok = (
        geometry.node_order == tuple(range(45))
        and geometry.held_capacities == (Q(1),) * 45
        and geometry.clock_factor == Q(1, 4)
        and geometry.phase_domain == "regular"
        and _HORIZON / 2 == 100
        and _DEFORMATION == 1
    )
    entry = law_ok and fine_ok and initial_zero and child_cell and outer_cell
    margin = boundary - endpoint_excess
    capture = entry and barrier_ok and margin > 0
    control_excess = 20 * (1 + stiffness) * q**2
    control_boundary = min(inner_excess.lo, face.lo)
    control_margin = control_boundary - control_excess
    control_capture = (
        law_ok
        and barrier_ok
        and 0 < initial_inner.lo <= initial_inner.hi < pi.lo / 2
        and initial_zero
        and control_margin > 0
    )
    return _AlternativeOuterBounds(
        geometry=geometry,
        recovered_child_norm_bound=q,
        current_absolute_upper_bound=current,
        current_lipschitz_upper_bound=stiffness,
        primitive_initial_scaled_norm_bounds=initial,
        primitive_forcing_norm_bounds=forcing,
        primitive_decay_upper_bound=decay,
        primitive_scaled_form_radius_upper_bound=a,
        primitive_phase_radius_upper_bound=p,
        primitive_node_form_radius_upper_bound=node_form,
        primitive_node_phase_radius_upper_bound=node_phase,
        primitive_edge_form_radius_upper_bound=edge_form,
        primitive_oriented_phase_gap_bounds=gaps,
        primitive_endpoint_storage_upper_bound=u5,
        full_form_error_upper_bound=full_x,
        full_phase_error_upper_bound=full_t,
        acute_convexity_split_margin=split_margin,
        target_internal_stiffness_bounds=inner_stiffness,
        target_outer_stiffness_bounds=outer_stiffness,
        internal_cycle_minimum_bounds=v9,
        outer_cycle_minimum_bounds=v5,
        target_storage_bounds=5 * v9 + 9 * v5,
        internal_positive_face_storage_bounds=b9,
        internal_negative_face_storage_bounds=negative_b9,
        outer_positive_face_storage_bounds=b5,
        internal_boundary_excess_bounds=inner_excess,
        outer_boundary_excess_bounds=outer_excess,
        signed_face_excess_lower_bounds=faces,
        signed_face_feasible=tuple(value is not None for value in faces),
        boundary_energy_lower_bound=boundary,
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
        control_initial_excess_upper_bound=control_excess,
        control_boundary_energy_lower_bound=control_boundary,
        control_capture_margin=control_margin,
        primitive_law_and_clock_certified=law_ok,
        fine_error_certified=fine_ok,
        acute_convexity_certified=convex,
        target_regular_certified=target_ok,
        barrier_certified=barrier_ok,
        attachment_budget_certified=budget_ok,
        initial_outer_zero_winding_certified=initial_zero,
        child_transit_cell_certified=child_cell,
        outer_cell_certified=outer_cell,
        conditional_outer_entry_certified=entry,
        conditional_capture_certified=capture,
        control_conditional_capture_certified=control_capture,
    )


@dataclass(frozen=True)
class _AlternativeOuterAcquisition:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    child_sources: tuple[_ConstitutiveRecovery, ...]
    resources: _AlternativeOuterBounds
    recovered_source_flags: tuple[bool, ...]
    relative_preparation_dimension: int
    original_preparation_storage_upper_bound: Q
    source_certified: bool
    initial_outer_zero_winding_certified: bool
    child_winding_retained_during_transit: bool
    outer_entry_certified: bool
    capture_certified: bool
    outer_winding_retained: bool
    child_winding_retained: bool
    converges_to_target: bool
    control_capture_certified: bool
    control_outer_zero_winding_retained: bool
    control_child_winding_retained: bool
    control_converges_to_target: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    deformation: Q = _DEFORMATION
    source_formation_time: Q = Q(200)
    source_relaxation_duration: Q = Q(2 * 10**13)
    source_handoff_time: Q = Q(200 + 2 * 10**13)
    elapsed_scaled_time: Q = _HORIZON
    elapsed_original_time: Q = _HORIZON / _CLOCK
    primitive_scaled_time: Q = _HORIZON / 2
    recovered_child_norm_bound: Q = _RADIUS
    initial_child_form_origins: tuple[Q, ...] = _SHIFTS
    initial_child_phase_origins: tuple[Q, ...] = (Q(0),) * 5
    target_child_windings: tuple[int, ...] = (1,) * 5
    target_outer_windings: tuple[int, ...] = (1,) * 9
    target_form: tuple[Q, ...] = (Q(0),) * 45
    target_phase_turns: tuple[Q, ...] = tuple(
        Q(j - 4, 9) + Q(c - 2, 5) for c in range(5) for j in range(9)
    )
    control_form_origins: tuple[Q, ...] = (Q(0),) * 5
    control_target_outer_windings: tuple[int, ...] = (0,) * 9
    budget: Q = _BUDGET
    protocol_revision: str = "99bf595c"
    arithmetic_method: str = INTERVAL_METHOD
    pressure_refresh: str = "continuous_current_at_actual_state"
    event_policy: str = "one_supplied_attachment_then_no_events_or_inputs"
    scope: tuple[str, ...] = (
        "five_fresh_lambda_one_recoveries_from_all_ten_original_allowances",
        "both_isolated_rows_transform_to_the_uniform_quarter_factor_clock",
        "primitive_common_form_shifts_are_preparation_not_reached_state_resets",
        "own_current_regular_bounds_and_potential_in_every_product_term",
        "full_ninety_coordinate_heat_comparison_and_all_180_signed_faces",
        "declared_45_outer_edge_attachment_keeps_coordinates_and_supplies_work",
        "paired_control_keeps_actual_recovered_residuals_without_form_ramp",
        "unavailable_is_not_failed_formation_or_a_nonentry_obstruction",
        "no_sine_verdict_trajectory_frozen_producer_or_later_contact_certificate",
        "no_autonomous_event_selection_or_physical_identification",
    )


def _assess_organized_alternative_outer_acquisition(
    *, form_error_bounds: object, phase_error_bounds: object
) -> _AlternativeOuterAcquisition:
    """Admit the fixed endpoint-law source once from its original primitives.

    All ten componentwise allowances must lie in [0,1e-10] before any source
    rebuilding. The reached allowance q bounds each child's Euclidean norm.
    Reports, retuned clocks, laws, errors, source resets or longer horizons
    cannot be substituted for this fixed protocol.
    """
    rx = _prepared_coordinate_vector(form_error_bounds, "form_error_bounds", 5)
    rt = _prepared_coordinate_vector(phase_error_bounds, "phase_error_bounds", 5)
    if min(*rx, *rt) < 0 or max(*rx, *rt) > _ERROR:
        raise ValueError("original preparation allowances must lie in [0,1e-10]")
    children = tuple(
        _assess_constitutive_recovery(
            deformation=_DEFORMATION, form_error_bound=a, phase_error_bound=b
        )
        for a, b in zip(rx, rt)
    )
    resources = _derive_organized_alternative_outer_bounds(recovered_radius=_RADIUS)
    flags = tuple(
        child.deformation == _DEFORMATION
        and child.recovery_certified
        and child.endpoint_form_norm_squared_upper_bound is not None
        and child.endpoint_phase_norm_squared_upper_bound is not None
        and child.endpoint_form_norm_squared_upper_bound < _RADIUS**2
        and child.endpoint_phase_norm_squared_upper_bound < _RADIUS**2
        and child.formation_time == 100
        and child.relaxation_duration == 10**13
        for child in children
    )
    source_ok = all(flags)
    control_ok = source_ok and resources.control_conditional_capture_certified
    entry = (
        source_ok
        and resources.attachment_budget_certified
        and resources.conditional_outer_entry_certified
    )
    capture = (
        source_ok
        and resources.attachment_budget_certified
        and resources.conditional_capture_certified
    )
    checks = (
        (source_ok, "original_own_law_child_recoveries_not_certified"),
        (
            resources.primitive_law_and_clock_certified,
            "complete_law_or_clock_not_certified",
        ),
        (resources.fine_error_certified, "full_fine_heat_comparison_not_certified"),
        (resources.acute_convexity_certified, "own_law_acute_convexity_not_certified"),
        (resources.target_regular_certified, "own_law_target_regularity_not_certified"),
        (
            resources.attachment_budget_certified,
            "attachment_or_initial_storage_budget_not_certified",
        ),
        (
            resources.initial_outer_zero_winding_certified,
            "initial_outer_zero_winding_not_certified",
        ),
        (resources.child_transit_cell_certified, "child_winding_transit_not_certified"),
        (resources.outer_cell_certified, "nine_outer_cycle_entry_not_certified"),
        (
            resources.barrier_certified,
            "complete_own_law_product_boundary_not_certified",
        ),
        (
            resources.capture_margin > 0,
            "complete_fine_endpoint_capture_margin_not_positive",
        ),
        (control_ok, "paired_zero_form_control_not_certified"),
    )
    unresolved = tuple(label for passed, label in checks if not passed)
    return _AlternativeOuterAcquisition(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        child_sources=children,
        resources=resources,
        recovered_source_flags=flags,
        relative_preparation_dimension=sum(
            c.source.preparation_dimension for c in children
        ),
        original_preparation_storage_upper_bound=sum(
            (c.source.initial_storage_upper_bounds[0] for c in children), Q(0)
        ),
        source_certified=source_ok,
        initial_outer_zero_winding_certified=source_ok
        and resources.initial_outer_zero_winding_certified,
        child_winding_retained_during_transit=source_ok
        and resources.fine_error_certified
        and resources.child_transit_cell_certified,
        outer_entry_certified=entry,
        capture_certified=capture,
        outer_winding_retained=capture,
        child_winding_retained=capture,
        converges_to_target=capture,
        control_capture_certified=control_ok,
        control_outer_zero_winding_retained=control_ok,
        control_child_winding_retained=control_ok,
        control_converges_to_target=control_ok,
        admitted=not unresolved,
        status=(
            "certified_alternative_outer_acquisition"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
