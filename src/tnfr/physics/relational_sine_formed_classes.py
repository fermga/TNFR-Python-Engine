"""Two symmetry-inequivalent formed winding classes under one supplied C9 law.

The exact phase-flat nominal sources have full relative preparation sets on
the same conserved-mean leaf. Weighted full-law transit bounds and local
storage barriers certify formation and recovery without running a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, sin, sqrt
from ._sine_preparation import (
    _prepared_duhamel_bounds,
    _prepared_transit_radii,
    _sine_domain,
    _sine_preparation_from_rows,
)
from .phase_cycle_geometry import _derive
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

__all__ = (
    "SineFormedClassPair",
    "assess_sine_formed_class_pair",
    "SineFormedClassResponse",
    "assess_sine_formed_class_response",
)

_NODES = tuple(range(9))
_EDGES = tuple(sorted(((0, 8), *((i, i + 1) for i in range(8)))))
_CAPACITY = (Q(1),) * 9
_PHASE = (Q(0),) * 9
_WINDINGS = (1, 2)
_SOURCE_SCALE = Q(2046, 9) * Q(355, 113) ** 2
_GAP = Q(1, 5)
_SOURCE_BUDGET = Q(10**9)


def _verify_cycle_gap():
    """Check the fixed mean-free bound L/2 >= (1/5)P exactly."""
    laplacian = tuple(
        tuple(Q(2 if i == j else -int((i - j) % 9 in (1, 8))) for j in _NODES)
        for i in _NODES
    )
    shifted = tuple(
        tuple(laplacian[i][j] / 2 - _GAP * (Q(int(i == j)) - Q(1, 9)) for j in _NODES)
        for i in _NODES
    )
    if not exact_symmetric_semidefinite(shifted):
        raise ArithmeticError("the fixed C9 mean-free semigroup bound failed")


def _fresh_class_preparations(form_error_bound, phase_error_bound):
    """Build both fixed sources from one fresh, exact C9 law domain.

    Private callers admit nonnegative rational error budgets before this
    boundary. No report or cached domain supplies its premises. The returned
    records enclose the original residual families; each public certificate
    additionally retains their exact zero-mean constraints.
    """
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    forms = tuple(tuple(k * _SOURCE_SCALE * (i - 4) for i in _NODES) for k in _WINDINGS)
    domain = _sine_domain(_derive(_NODES, _EDGES), model, _CAPACITY)
    preparations = tuple(
        _sine_preparation_from_rows(
            domain,
            form=row,
            phase=_PHASE,
            form_errors=(form_error_bound,) * 9,
            phase_errors=(phase_error_bound,) * 9,
        )
        for row in forms
    )
    _verify_cycle_gap()
    return domain, preparations


@dataclass(frozen=True)
class SineFormedClassPair:
    """Conditional pair formation and recovery on a common zero-mean leaf.

    Every class uses its own residuals with the same component budgets and
    exact zero sums. Positive budgets give an open relative preparation
    interior of dimension sixteen, not an eighteen-dimensional ambient box.
    The source costs need not be equal; each entire family must fit the common
    declared budget. The symmetry comparison is the exact winding magnitude
    under cycle automorphisms, global origins and joint sign reversal.

    All class-indexed tuples follow ``class_order``. Endpoint bounds enclose
    actual complete-law states, not independent products of observed boxes.
    An unavailable certificate does not prove failed formation or instability.
    """

    scaled_time: Q
    form_error_bound: Q
    phase_error_bound: Q
    radius: Q
    original_time: Q
    reference_model: RelationalExchangeModel
    preparation_dimension: int
    full_relative_preparation: bool
    initial_forms_by_class: tuple[tuple[Q, ...], ...]
    initial_form_storage_by_class: tuple[Q, Q]
    initial_storage_upper_bounds: tuple[Q, Q]
    source_budget_margin_bounds: tuple[I, I]
    source_budget_certified_by_class: tuple[bool, bool]
    gamma_bounds: I
    inverse_gamma_bounds: I
    eta_bounds: I
    forcing_norm_bounds: I
    nominal_scaled_initial_norm_upper_bounds: tuple[Q, Q]
    scaled_initial_error_norm_upper_bounds: tuple[Q, Q]
    scaled_initial_norm_upper_bounds: tuple[Q, Q]
    initial_phase_error_norm_upper_bound: Q
    exponential_decay_bounds: I
    scaled_form_radius_upper_bounds: tuple[Q, Q]
    endpoint_form_norm_upper_bounds: tuple[Q, Q]
    endpoint_phase_error_norm_upper_bounds: tuple[Q, Q]
    proxy_target_distance_bounds: tuple[I, I]
    endpoint_relative_norm_squared_upper_bounds: tuple[Q, Q]
    endpoint_excess_storage_upper_bounds: tuple[Q, Q]
    target_phase_storage_bounds_by_class: tuple[I, I]
    initial_zero_winding_margin_bounds: I
    initial_zero_winding_certified: bool
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    acute_radius_certified: bool
    coercivity_lower_bound: Q | None
    barrier_lower_bound: Q | None
    endpoint_radius_margin_bounds: tuple[I, I]
    endpoint_radius_certified_by_class: tuple[bool, bool]
    storage_barrier_margin_bounds: tuple[I, I] | None
    storage_barrier_certified_by_class: tuple[bool, bool]
    formation_certified_by_class: tuple[bool, bool]
    symmetry_inequivalent: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    class_order: tuple[str, str] = ("winding_one", "winding_two")
    target_windings: tuple[int, int] = _WINDINGS
    symmetry_winding_orbits: tuple[tuple[int, int], ...] = ((-1, 1), (-2, 2))
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    initial_phases: tuple[Q, ...] = _PHASE
    target_phase_turns_by_class: tuple[tuple[Q, ...], ...] = tuple(
        tuple(Q(k * (i - 4), 9) for i in _NODES) for k in _WINDINGS
    )
    held_capacities: tuple[Q, ...] = _CAPACITY
    metric_weights: tuple[Q, ...] = (Q(2),) * 9
    nominal_source_scale: Q = _SOURCE_SCALE
    common_source_storage_budget: Q = _SOURCE_BUDGET
    conserved_form_mean: Q = Q(0)
    conserved_lifted_phase_mean: Q = Q(0)
    relative_mean_leaf_dimension: int = 16
    residual_constraints: tuple[str, str] = (
        "sum_form_residuals_equals_zero",
        "sum_lifted_phase_residuals_equals_zero",
    )
    lambda_lower_bound: Q = _GAP
    laplacian_upper_bound: Q = Q(4)
    form_loss: Q = Q(1023, 1024)
    exchange_weight: Q = Q(1, 1024)
    phase_exchange_beta: Q = Q(1)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_simple_unit_C9_same_complete_positive_loss_law_for_both_classes",
        "phase_flat_nominal_sources_with_all_node_mean_zero_form_and_phase_errors",
        "same_exact_zero_conserved_form_and_lifted_phase_origins",
        "positive_budgets_have_sixteen_dimensional_relative_interior_not_ambient_box",
        "unequal_nominal_costs_each_full_source_family_fits_one_common_storage_budget",
        "one_fresh_shared_preparation_domain_no_cached_certificate_consumption",
        "full_nonlinear_transit_and_initial_errors_retained_in_weighted_Duhamel_bounds",
        "strict_initial_zero_winding_endpoint_radius_acute_chart_and_storage_barriers",
        "conditional_recovery_to_target_and_uniform_form_on_each_fixed_mean_leaf",
        "D9_global_origins_and_joint_sign_preserve_winding_magnitude",
        "unavailable_is_not_failed_formation_or_instability",
        "no_receiver_response_or_spectral_measurement_prediction",
        "no_trajectory_solver_events_resets_or_parameter_search",
        "no_autonomous_preparation_support_origin_law_selection_or_physical_identity",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-formed-class-pair.v1", "report": _project(self)}


def assess_sine_formed_class_pair(
    *, scaled_time, form_error_bound, phase_error_bound, radius
) -> SineFormedClassPair:
    """Assess the two fixed phase-flat C9 source families without evolving them.

    All four arguments are required. Time and residual budgets are nonnegative,
    radius is positive, and the exponential budget requires ``scaled_time/5 <=
    4096``. Shared scalar admission precedes source construction and arithmetic.
    Exact rational primitives remain exact. Nonpositive sufficient margins
    return unavailable; malformed primitives reject.

    Residuals in each class satisfy their componentwise budgets AND exact zero
    sums separately for form and lifted phase. Thus both families have the same
    conserved zero origins. The fixed nominal slope is
    ``(2046/9)*(355/113)**2``; its first and second multiples select candidate
    windings one and two under one unchanged positive-loss sine law.
    """
    time = exact_or_represented_real(scaled_time, "scaled_time")
    rx = exact_or_represented_real(form_error_bound, "form_error_bound")
    rt = exact_or_represented_real(phase_error_bound, "phase_error_bound")
    r = exact_or_represented_real(radius, "radius")
    if time < 0 or _GAP * time > _MAX_RATIONAL_EXPONENT:
        raise ValueError("scaled_time requires 0 <= scaled_time/5 <= 4096")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    if r <= 0:
        raise ValueError("radius must be strictly positive")

    domain, preparations = _fresh_class_preparations(rx, rt)
    model, forms = domain.model, tuple(p.form for p in preparations)
    pi = domain.pi
    gamma, inverse_gamma, eta = domain.alpha, domain.inverse_alpha, domain.eta
    decay = I(*_negative_exp_bounds(_GAP * time))
    inverse_root_two = (1 / sqrt(I(2))).hi
    phase_error_initial = (domain.forcing * rt).hi
    slope_error = abs(gamma * _SOURCE_SCALE - 2 * pi / 9)
    distances = tuple(sqrt(I(60)) * k * slope_error for k in _WINDINGS)
    scaled_radii, form_norms, phase_errors, norms, energies = [], [], [], [], []
    for preparation, distance in zip(preparations, distances):
        scaled_radius, phase_radius = _prepared_transit_radii(
            time=time,
            decay_upper=decay.hi,
            gap_lower=_GAP,
            initial_norm_upper=preparation.initial_norm.hi,
            feedback_upper=eta.hi,
            forcing_upper=domain.forcing.hi,
        )
        form_norm = inverse_gamma.hi * scaled_radius * inverse_root_two
        phase_error = (
            phase_radius + preparation.initial_error_norm + phase_error_initial
        ) * inverse_root_two
        norm_squared = form_norm**2 + (distance.hi + phase_error) ** 2
        scaled_radii.append(scaled_radius)
        form_norms.append(form_norm)
        phase_errors.append(phase_error)
        norms.append(norm_squared)
        energies.append(2 * norm_squared)

    nominal_costs = tuple(36 * k**2 * _SOURCE_SCALE**2 for k in _WINDINGS)
    source_costs = tuple(
        cost + 32 * k * _SOURCE_SCALE * rx + 18 * (rx**2 + rt**2)
        for k, cost in zip(_WINDINGS, nominal_costs)
    )
    budget_margins = tuple(I(_SOURCE_BUDGET - cost) for cost in source_costs)
    budget_flags = tuple(margin.lo > 0 for margin in budget_margins)
    initial_margin = pi / 2 - 2 * rt
    initial_zero = initial_margin.lo > 0
    radius_angle = 4 * pi / 9 + sqrt(I(2)) * r
    acute_margin = pi / 2 - radius_angle
    cosine = cos(radius_angle).lo if acute_margin.lo > 0 else None
    acute = cosine is not None and cosine > 0
    coercivity = _GAP * cosine if acute else None
    barrier = (I(coercivity) * r**2).lo if coercivity is not None else None
    radius_margins = tuple(I(r**2 - value) for value in norms)
    radius_flags = tuple(margin.lo > 0 for margin in radius_margins)
    energy_margins = (
        tuple(I(barrier - value) for value in energies) if barrier is not None else None
    )
    energy_flags = tuple(
        energy_margins is not None and energy_margins[i].lo > 0 for i in (0, 1)
    )
    formation = tuple(
        initial_zero and acute and radius_flags[i] and energy_flags[i] for i in (0, 1)
    )
    inequivalent = abs(_WINDINGS[0]) != abs(_WINDINGS[1])
    reasons = tuple(
        reason
        for condition, reason in (
            (initial_zero, "whole_initial_set_zero_winding_not_certified"),
            (acute, "strict_acute_radius_not_certified"),
            (inequivalent, "distinct_symmetry_classes_not_certified"),
        )
        if not condition
    ) + tuple(
        f"{name}:{reason}"
        for i, name in enumerate(("winding_one", "winding_two"))
        for condition, reason in (
            (budget_flags[i], "strict_source_storage_budget_not_certified"),
            (radius_flags[i], "strict_endpoint_radius_not_certified"),
            (energy_flags[i], "strict_excess_storage_barrier_not_certified"),
        )
        if not condition
    )
    return SineFormedClassPair(
        scaled_time=time,
        form_error_bound=rx,
        phase_error_bound=rt,
        radius=r,
        original_time=time / domain.e,
        reference_model=model,
        preparation_dimension=8 * int(rx > 0) + 8 * int(rt > 0),
        full_relative_preparation=rx > 0 and rt > 0,
        initial_forms_by_class=forms,
        initial_form_storage_by_class=nominal_costs,
        initial_storage_upper_bounds=source_costs,
        source_budget_margin_bounds=budget_margins,
        source_budget_certified_by_class=budget_flags,
        gamma_bounds=gamma,
        inverse_gamma_bounds=inverse_gamma,
        eta_bounds=eta,
        forcing_norm_bounds=domain.forcing,
        nominal_scaled_initial_norm_upper_bounds=tuple(
            p.nominal_initial_norm.hi for p in preparations
        ),
        scaled_initial_error_norm_upper_bounds=tuple(
            p.initial_error_norm for p in preparations
        ),
        scaled_initial_norm_upper_bounds=tuple(p.initial_norm.hi for p in preparations),
        initial_phase_error_norm_upper_bound=phase_error_initial,
        exponential_decay_bounds=decay,
        scaled_form_radius_upper_bounds=tuple(scaled_radii),
        endpoint_form_norm_upper_bounds=tuple(form_norms),
        endpoint_phase_error_norm_upper_bounds=tuple(phase_errors),
        proxy_target_distance_bounds=distances,
        endpoint_relative_norm_squared_upper_bounds=tuple(norms),
        endpoint_excess_storage_upper_bounds=tuple(energies),
        target_phase_storage_bounds_by_class=tuple(
            9 * (1 - cos(2 * k * pi / 9)) for k in _WINDINGS
        ),
        initial_zero_winding_margin_bounds=initial_margin,
        initial_zero_winding_certified=initial_zero,
        radius_angle_bounds=radius_angle,
        acute_radius_margin_bounds=acute_margin,
        acute_radius_certified=acute,
        coercivity_lower_bound=coercivity,
        barrier_lower_bound=barrier,
        endpoint_radius_margin_bounds=radius_margins,
        endpoint_radius_certified_by_class=radius_flags,
        storage_barrier_margin_bounds=energy_margins,
        storage_barrier_certified_by_class=energy_flags,
        formation_certified_by_class=formation,
        symmetry_inequivalent=inequivalent,
        status="certified_two_formed_classes" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class SineFormedClassResponse:
    """One supplied mean-preserving phase probe of two actually formed classes.

    Class tuples follow the nested formation certificate's class order. The
    continued states retain the original preparation errors and nonlinear
    history; no ideal-state reset supplies the response. A single common heat
    factor is retained in the contrast, so subtracting the marginal recorded
    intervals generally gives a weaker enclosure. Probe work is a separate
    hybrid storage change, not continuous loss or an event-selection law.

    ``warmup_scaled_form_remainder_upper_bounds`` and
    ``warmup_phase_error_upper_bounds`` use the weighted norm ``M=2*I``.
    The latter measures phase distance from the nominal proxy ``v=gamma*x0``,
    not from the target. ``warmup_form_norm_upper_bounds`` is the Euclidean
    form norm; ``warmup_target_phase_radius_upper_bounds`` is the Euclidean
    phase distance from the target, including the proxy-to-target distance.
    """

    formation_time: Q
    probe_time: Q
    probe_duration: Q
    phase_increment: Q
    form_error_bound: Q
    phase_error_bound: Q
    readout_error_bound: Q
    radius: Q
    formation_certificate: SineFormedClassPair
    probe_original_time: Q
    readout_original_time: Q
    warmup_decay_bounds: I
    warmup_scaled_form_remainder_upper_bounds: tuple[Q, Q]
    warmup_phase_error_upper_bounds: tuple[Q, Q]
    warmup_form_norm_upper_bounds: tuple[Q, Q]
    warmup_target_phase_radius_upper_bounds: tuple[Q, Q]
    heat_response_factor_bounds: I
    ideal_readout_bounds_by_class: tuple[I, I]
    response_error_upper_bounds: tuple[Q, Q]
    recorded_readout_bounds_by_class: tuple[I, I]
    recorded_contrast_bounds: I
    response_certified: bool
    post_probe_phase_radius_upper_bounds: tuple[Q, Q]
    post_probe_relative_norm_squared_upper_bounds: tuple[Q, Q]
    post_probe_excess_storage_upper_bounds: tuple[Q, Q]
    acute_radius_margin_bounds_by_class: tuple[I, I]
    coercivity_lower_bounds_by_class: tuple[Q | None, Q | None]
    barrier_lower_bounds_by_class: tuple[Q | None, Q | None]
    post_probe_radius_margin_bounds: tuple[I, I]
    post_probe_storage_margin_bounds: tuple[I | None, I | None]
    recovery_certified_by_class: tuple[bool, bool]
    nominal_probe_work_bounds_by_class: tuple[I, I]
    probe_work_error_upper_bounds: tuple[Q, Q]
    probe_work_bounds_by_class: tuple[I, I]
    status: str
    unavailable_reasons: tuple[str, ...]
    probe_vector: tuple[Q, ...] = (Q(8, 9),) + (Q(-1, 9),) * 8
    receiver_node: int = 0
    readout: str = "signed_form_at_node_zero"
    contrast_order: tuple[str, str] = ("winding_two", "winding_one")
    probe_law: str = (
        "theta_plus=theta_minus+phase_increment*probe_vector;x_plus=x_minus"
    )
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_primitive_formation_prerequisite_no_incoming_report_or_cached_verdict",
        "same_complete_positive_loss_C9_law_and_zero_mean_preparation_families",
        "actual_formed_states_continue_to_probe_time_without_reset",
        "one_supplied_mean_preserving_phase_jump_common_to_both_classes",
        "one_correlated_heat_factor_for_both_ideal_readouts",
        "full_nonlinear_warmup_and_probe_remainders_retained",
        "independent_bounded_signed_form_readout_errors",
        "probe_work_separate_from_continuous_storage_loss_no_passivity_claim",
        "strict_post_probe_storage_and_radius_barriers_imply_conditional_recovery",
        "unavailable_is_not_response_equality_or_failed_recovery",
        "no_trajectory_solver_fitted_response_or_parameter_search",
        "no_autonomous_probe_selection_or_physical_measurement_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-formed-class-response.v1",
            "report": _project(self),
        }


def assess_sine_formed_class_response(
    *,
    formation_time,
    probe_time,
    probe_duration,
    phase_increment,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
    radius,
) -> SineFormedClassResponse:
    """Bound a common supplied phase probe and recovery of two formed C9 classes.

    All arguments are required represented reals. Times, increment and error
    budgets are nonnegative; radius is positive; probe time cannot precede
    formation time. Rational exponential work requires ``probe_time/5 <=4096``
    and ``2*probe_duration <=4096``. Malformed primitives reject before source
    construction. Nonpositive sufficient margins produce an unavailable report.

    The actual preparation residuals obey the pair assessor's componentwise
    budgets and exact zero sums. The event adds ``delta*(e0-1/9)`` to the lifted
    phases and leaves forms unchanged. The readout is the signed form at node
    zero after the supplied duration. Its strict contrast compares class two
    minus class one, retaining their shared heat response factor. Work is
    supplied at the event; subsequent recovery uses the unchanged loss law.
    """
    tf = exact_or_represented_real(formation_time, "formation_time")
    tp = exact_or_represented_real(probe_time, "probe_time")
    h = exact_or_represented_real(probe_duration, "probe_duration")
    delta = exact_or_represented_real(phase_increment, "phase_increment")
    rx = exact_or_represented_real(form_error_bound, "form_error_bound")
    rt = exact_or_represented_real(phase_error_bound, "phase_error_bound")
    noise = exact_or_represented_real(readout_error_bound, "readout_error_bound")
    r = exact_or_represented_real(radius, "radius")
    if tf < 0 or tp < tf or _GAP * tp > _MAX_RATIONAL_EXPONENT:
        raise ValueError("times require 0 <= formation_time <= probe_time <= 20480")
    if h < 0 or 2 * h > _MAX_RATIONAL_EXPONENT:
        raise ValueError("probe_duration requires 0 <= 2*probe_duration <=4096")
    if delta < 0 or min(rx, rt, noise) < 0:
        raise ValueError("phase increment and error bounds must be nonnegative")
    if r <= 0:
        raise ValueError("radius must be strictly positive")

    formation = assess_sine_formed_class_pair(
        scaled_time=tf, form_error_bound=rx, phase_error_bound=rt, radius=r
    )
    domain, preparations = _fresh_class_preparations(rx, rt)
    pi, gamma, eta = domain.pi, domain.alpha, domain.eta
    root_two, inverse_root_two = sqrt(I(2)), 1 / sqrt(I(2))
    q_norm = sqrt(I(Q(8, 9)))
    alpha = 2 * pi / 9
    distances = tuple(
        sqrt(I(60)) * k * abs(gamma * _SOURCE_SCALE - alpha) for k in _WINDINGS
    )
    decay = I(*_negative_exp_bounds(_GAP * tp))
    initial_phase_error = (domain.forcing * rt).hi
    remainders, phase_errors, form_norms, phase_radii = [], [], [], []
    for preparation, distance in zip(preparations, distances):
        remainder, phase_error = _prepared_duhamel_bounds(
            time=tp,
            decay_upper=decay.hi,
            gap_lower=_GAP,
            rate_upper=Q(2),
            forcing_upper=domain.forcing.hi,
            initial_norm_upper=preparation.nominal_initial_norm.hi,
            scaled_form_error_upper=preparation.initial_error_norm,
            phase_error_upper=initial_phase_error,
            feedback_upper=eta.hi,
        )
        poisson_norm = 2 * root_two.hi * distance.hi / _GAP
        form_norm = (
            domain.inverse_alpha.hi
            * (eta.hi * poisson_norm + remainder)
            * inverse_root_two.hi
        )
        remainders.append(remainder)
        phase_errors.append(phase_error)
        form_norms.append(form_norm)
        phase_radii.append(distance.hi + phase_error * inverse_root_two.hi)

    heat_decay = I(*_negative_exp_bounds(2 * h))
    heat = I(max(Q(0), (1 - heat_decay.hi) / 2), min(h, Q(8, 9)))
    sine = I(0) if delta == 0 else sin(I(delta))
    cosines = tuple(cos(k * alpha) for k in _WINDINGS)
    ideals = tuple(-gamma * cosine * sine * heat for cosine in cosines)
    errors = tuple(
        form_norm
        + 2 * gamma.hi * h * phase_radius
        + 2 * gamma.hi**2 * h**2 * form_norm
        + 2 * gamma.hi**3 * h**3
        for form_norm, phase_radius in zip(form_norms, phase_radii)
    )
    recorded = tuple(
        ideal + I(-error - noise, error + noise) for ideal, error in zip(ideals, errors)
    )
    contrast_error = sum(errors, Q(0)) + 2 * noise
    contrast = gamma * (cosines[0] - cosines[1]) * sine * heat + I(
        -contrast_error, contrast_error
    )
    response = contrast.lo > 0

    post_phase = tuple(value + delta * q_norm.hi for value in phase_radii)
    post_norms = tuple(x**2 + theta**2 for x, theta in zip(form_norms, post_phase))
    # A nonpositive lower angle uses the global Hessian bound. Otherwise it
    # belongs to [0,k*alpha] within the positive-cosine chart of each target.
    upper_cosines = tuple(
        cos(I(max(Q(0), (k * alpha - root_two * theta).lo))).hi
        for k, theta in zip(_WINDINGS, post_phase)
    )
    post_energy = tuple(
        2 * x**2 + 2 * upper_cosine * theta**2
        for x, theta, upper_cosine in zip(form_norms, post_phase, upper_cosines)
    )
    acute_margins = tuple(pi / 2 - (k * alpha + root_two * r) for k in _WINDINGS)
    coercivities = tuple(
        _GAP * cos(k * alpha + root_two * r).lo if margin.lo > 0 else None
        for k, margin in zip(_WINDINGS, acute_margins)
    )
    barriers = tuple(
        (I(value) * r**2).lo if value is not None and value > 0 else None
        for value in coercivities
    )
    radius_margins = tuple(I(r**2 - norm) for norm in post_norms)
    storage_margins = tuple(
        I(barrier - energy) if barrier is not None else None
        for barrier, energy in zip(barriers, post_energy)
    )
    recovery = tuple(
        acute_margins[i].lo > 0
        and radius_margins[i].lo > 0
        and storage_margins[i] is not None
        and storage_margins[i].lo > 0
        for i in (0, 1)
    )
    half_sine = I(0) if delta == 0 else sin(I(delta) / 2)
    one_minus_cosine = 2 * half_sine**2
    nominal_work = tuple(2 * cosine * one_minus_cosine for cosine in cosines)
    work_errors = tuple(4 * delta * q_norm.hi * value for value in phase_radii)
    works = tuple(
        nominal + I(-error, error) for nominal, error in zip(nominal_work, work_errors)
    )
    reasons = tuple(
        reason
        for condition, reason in (
            (
                formation.status == "certified_two_formed_classes",
                "formation_prerequisite_unavailable",
            ),
            (response, "recorded_contrast_not_strictly_positive"),
        )
        if not condition
    ) + tuple(
        f"{name}:{reason}"
        for i, name in enumerate(("winding_one", "winding_two"))
        for condition, reason in (
            (acute_margins[i].lo > 0, "strict_acute_radius_not_certified"),
            (radius_margins[i].lo > 0, "strict_post_probe_radius_not_certified"),
            (
                storage_margins[i] is not None and storage_margins[i].lo > 0,
                "strict_post_probe_storage_barrier_not_certified",
            ),
        )
        if not condition
    )
    return SineFormedClassResponse(
        formation_time=tf,
        probe_time=tp,
        probe_duration=h,
        phase_increment=delta,
        form_error_bound=rx,
        phase_error_bound=rt,
        readout_error_bound=noise,
        radius=r,
        formation_certificate=formation,
        probe_original_time=tp / domain.e,
        readout_original_time=(tp + h) / domain.e,
        warmup_decay_bounds=decay,
        warmup_scaled_form_remainder_upper_bounds=tuple(remainders),
        warmup_phase_error_upper_bounds=tuple(phase_errors),
        warmup_form_norm_upper_bounds=tuple(form_norms),
        warmup_target_phase_radius_upper_bounds=tuple(phase_radii),
        heat_response_factor_bounds=heat,
        ideal_readout_bounds_by_class=ideals,
        response_error_upper_bounds=errors,
        recorded_readout_bounds_by_class=recorded,
        recorded_contrast_bounds=contrast,
        response_certified=response,
        post_probe_phase_radius_upper_bounds=post_phase,
        post_probe_relative_norm_squared_upper_bounds=post_norms,
        post_probe_excess_storage_upper_bounds=post_energy,
        acute_radius_margin_bounds_by_class=acute_margins,
        coercivity_lower_bounds_by_class=coercivities,
        barrier_lower_bounds_by_class=barriers,
        post_probe_radius_margin_bounds=radius_margins,
        post_probe_storage_margin_bounds=storage_margins,
        recovery_certified_by_class=recovery,
        nominal_probe_work_bounds_by_class=nominal_work,
        probe_work_error_upper_bounds=work_errors,
        probe_work_bounds_by_class=works,
        status="certified_formed_class_response" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
