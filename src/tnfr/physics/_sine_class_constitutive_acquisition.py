"""Own-law C9 acquisition for the matched ninth-harmonic potential family.

Proof: theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md
#organized-pattern-same-source-acquisition. The independently supplied ramp
families are unchanged; forcing, source storage and capture are bounded under
V_lambda(delta)=1-cos(delta)+lambda*(1-cos(9*delta))**2/(6*9**4).
This certificate uses the shared semigroup arithmetic, never a sine-domain
report, a trajectory, or a sine acquisition verdict. Later dwell, contact and
interaction conclusions are outside its scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._sine_class_constitutive_comparison import _current_derivative_bound, _deformation
from ._sine_preparation import _prepared_transit_radii
from .relational_sine_formed_classes import (
    _CAPACITY,
    _EDGES,
    _GAP,
    _NODES,
    _PHASE,
    _SOURCE_BUDGET,
    _SOURCE_SCALE,
    _WINDINGS,
    _verify_cycle_gap,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds


@dataclass(frozen=True)
class _ConstitutiveFormedClassPair:
    """Conditional acquisition/capture of both isolated fixed-source families.

    Residuals satisfy the component budgets and separate exact zero sums. The
    eighteen coordinates remain present; positive budgets give a sixteen-
    dimensional relative source interior. Bounds are pointwise in deformation.
    Uniformity over [0,1] requires the theorem's separate monotonicity argument.

    Formation flags retain geometric entry and own-law capture premises;
    source storage policy has separate flags. The overall status requires
    both. Missing or nonpositive sufficient margins mean unavailable, not
    failed acquisition, instability or an obstruction to another proof.
    """

    deformation: Q
    scaled_time: Q
    form_error_bound: Q
    phase_error_bound: Q
    radius: Q
    original_time: Q
    preparation_dimension: int
    full_relative_preparation: bool
    initial_forms_by_class: tuple[tuple[Q, ...], ...]
    initial_form_storage_by_class: tuple[Q, Q]
    initial_potential_deformation_upper_bound: Q
    initial_storage_upper_bounds: tuple[Q, Q]
    source_budget_margin_bounds: tuple[I, I]
    source_budget_certified_by_class: tuple[bool, bool]
    gamma_bounds: I
    inverse_gamma_bounds: I
    eta_bounds: I
    metric_unit_norm_bounds: I
    current_absolute_upper_bound: Q
    phase_hessian_upper_bound: Q
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
    endpoint_target_phase_norm_upper_bounds: tuple[Q, Q]
    endpoint_relative_norm_squared_upper_bounds: tuple[Q, Q]
    endpoint_form_storage_upper_bounds: tuple[Q, Q]
    endpoint_phase_excess_storage_upper_bounds: tuple[Q, Q]
    endpoint_excess_storage_upper_bounds: tuple[Q, Q]
    target_phase_storage_bounds_by_class: tuple[I, I]
    initial_zero_winding_margin_bounds: I
    initial_zero_winding_certified: bool
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    acute_radius_certified: bool
    phase_hessian_lower_bound_candidate: Q | None
    phase_hessian_lower_bound: Q | None
    positive_phase_hessian_certified: bool
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
    law: str = "normalized_matched_ninth_harmonic_reciprocal_exchange"
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_isolated_unit_C9_same_original_rational_sources_for_every_deformation",
        "own_potential_current_and_both_reciprocal_rows_in_the_declared_clock",
        "fresh_primitive_admission_no_sine_domain_preparation_or_verdict_consumption",
        "componentwise_form_and_lifted_phase_errors_with_separate_exact_zero_sums",
        "metric_source_error_norms_do_not_use_the_enlarged_current_forcing_bound",
        "own_law_source_storage_includes_the_exact_quartic_potential_correction",
        "full_family_semigroup_transit_then_strict_own_Hessian_and_storage_capture",
        "source_budget_flags_are_separate_from_geometric_formation_flags",
        "nonpositive_sufficient_margin_is_unavailable_not_failed_acquisition",
        "no_source_search_trajectory_or_sine_specific_nonlinear_remainder_transfer",
        "no_later_dwell_tiny_handoff_contact_work_or_finite_interaction_certificate",
        "no_autonomous_source_selection_or_physical_identification",
    )


def _assess_constitutive_class_pair(
    *, deformation, scaled_time, form_error_bound, phase_error_bound, radius
) -> _ConstitutiveFormedClassPair:
    """Bound the original two source families under their declared own law.

    All five primitives are required. Deformation and error bounds are finite
    nonnegative represented reals; radius is positive. Structural time obeys
    0 <= scaled_time/5 <= 4096, the shared exact exponential work budget.
    No upper deformation cap is imposed: unavailable stiffness is reported
    rather than borrowing a sine capture certificate. Every raw scalar is
    admitted before geometry, interval materialization or exponentiation.
    """
    coefficient = _deformation(deformation)
    time = _exact(scaled_time, "scaled_time")
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    r = _exact(radius, "radius")
    if time < 0 or _GAP * time > _MAX_RATIONAL_EXPONENT:
        raise ValueError("scaled_time requires 0 <= scaled_time/5 <= 4096")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    if r <= 0:
        raise ValueError("radius must be strictly positive")

    _verify_cycle_gap()
    forms = tuple(tuple(k * _SOURCE_SCALE * (i - 4) for i in _NODES) for k in _WINDINGS)
    pi = pi_interval()
    gamma, inverse_gamma = Q(1, 1023) / pi, 1023 * pi
    eta = Q(1, 1023**2) / pi**2
    metric_norm = sqrt(I(18))
    current_upper = _current_derivative_bound(coefficient, 0)
    hessian_upper = _current_derivative_bound(coefficient, 1)
    forcing = metric_norm * current_upper
    nominal_norms = tuple(
        gamma * sqrt(I(sum((2 * value**2 for value in row), Q(0)))) for row in forms
    )
    # These are source-coordinate norms, not nonlinear-current bounds.
    form_error_initial = (gamma * sqrt(I(18 * rx**2))).hi
    phase_error_initial = (metric_norm * rt).hi
    initial_norms = tuple(value.hi + form_error_initial for value in nominal_norms)
    decay = I(*_negative_exp_bounds(_GAP * time))
    inverse_root_two = (1 / sqrt(I(2))).hi
    slope_error = abs(gamma * _SOURCE_SCALE - 2 * pi / 9)
    distances = tuple(sqrt(I(60)) * k * slope_error for k in _WINDINGS)

    scaled_radii, form_norms, phase_errors, phase_norms = [], [], [], []
    norms, form_energies, phase_energies, energies = [], [], [], []
    for initial, distance in zip(initial_norms, distances):
        scaled_radius, phase_radius = _prepared_transit_radii(
            time=time,
            decay_upper=decay.hi,
            gap_lower=_GAP,
            initial_norm_upper=initial,
            feedback_upper=eta.hi,
            forcing_upper=forcing.hi,
        )
        form_norm = inverse_gamma.hi * scaled_radius * inverse_root_two
        phase_error = (
            phase_radius + form_error_initial + phase_error_initial
        ) * inverse_root_two
        target_phase_norm = distance.hi + phase_error
        form_energy = 2 * form_norm**2
        phase_energy = 2 * hessian_upper * target_phase_norm**2
        scaled_radii.append(scaled_radius)
        form_norms.append(form_norm)
        phase_errors.append(phase_error)
        phase_norms.append(target_phase_norm)
        norms.append(form_norm**2 + target_phase_norm**2)
        form_energies.append(form_energy)
        phase_energies.append(phase_energy)
        energies.append(form_energy + phase_energy)

    nominal_costs = tuple(36 * k**2 * _SOURCE_SCALE**2 for k in _WINDINGS)
    # Nine edges and |delta|<=2*rt give 9*lambda*(2*rt)**4/24.
    # Preserve this possibly subgrid contribution exactly before adding it.
    source_correction = 6 * coefficient * rt**4
    source_costs = tuple(
        cost + 32 * k * _SOURCE_SCALE * rx + 18 * (rx**2 + rt**2) + source_correction
        for k, cost in zip(_WINDINGS, nominal_costs)
    )
    budget_margins = tuple(I(_SOURCE_BUDGET - cost) for cost in source_costs)
    budget_flags = tuple(value.lo > 0 for value in budget_margins)
    initial_margin = pi / 2 - 2 * rt
    initial_zero = initial_margin.lo > 0
    radius_angle = 4 * pi / 9 + sqrt(I(2)) * r
    acute_margin = pi / 2 - radius_angle
    acute = acute_margin.lo > 0
    hessian_candidate = cos(radius_angle).lo - (hessian_upper - 1) if acute else None
    positive_hessian = hessian_candidate is not None and hessian_candidate > 0
    hessian_lower = hessian_candidate if positive_hessian else None
    coercivity = _GAP * hessian_lower if hessian_lower is not None else None
    barrier = (I(coercivity) * r**2).lo if coercivity is not None else None
    radius_margins = tuple(I(r**2 - value) for value in norms)
    radius_flags = tuple(value.lo > 0 for value in radius_margins)
    energy_margins = (
        tuple(I(barrier - value) for value in energies) if barrier is not None else None
    )
    energy_flags = tuple(
        energy_margins is not None and energy_margins[i].lo > 0 for i in (0, 1)
    )
    formation = tuple(
        initial_zero
        and acute
        and positive_hessian
        and radius_flags[i]
        and energy_flags[i]
        for i in (0, 1)
    )
    inequivalent = abs(_WINDINGS[0]) != abs(_WINDINGS[1])
    reasons = tuple(
        reason
        for condition, reason in (
            (initial_zero, "whole_initial_set_zero_winding_not_certified"),
            (acute, "strict_acute_radius_not_certified"),
            (positive_hessian, "strict_own_law_phase_hessian_not_certified"),
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
    return _ConstitutiveFormedClassPair(
        deformation=coefficient,
        scaled_time=time,
        form_error_bound=rx,
        phase_error_bound=rt,
        radius=r,
        original_time=time / Q(1023, 1024),
        preparation_dimension=8 * int(rx > 0) + 8 * int(rt > 0),
        full_relative_preparation=rx > 0 and rt > 0,
        initial_forms_by_class=forms,
        initial_form_storage_by_class=nominal_costs,
        initial_potential_deformation_upper_bound=source_correction,
        initial_storage_upper_bounds=source_costs,
        source_budget_margin_bounds=budget_margins,
        source_budget_certified_by_class=budget_flags,
        gamma_bounds=gamma,
        inverse_gamma_bounds=inverse_gamma,
        eta_bounds=eta,
        metric_unit_norm_bounds=metric_norm,
        current_absolute_upper_bound=current_upper,
        phase_hessian_upper_bound=hessian_upper,
        forcing_norm_bounds=forcing,
        nominal_scaled_initial_norm_upper_bounds=tuple(
            value.hi for value in nominal_norms
        ),
        scaled_initial_error_norm_upper_bounds=(form_error_initial,) * 2,
        scaled_initial_norm_upper_bounds=initial_norms,
        initial_phase_error_norm_upper_bound=phase_error_initial,
        exponential_decay_bounds=decay,
        scaled_form_radius_upper_bounds=tuple(scaled_radii),
        endpoint_form_norm_upper_bounds=tuple(form_norms),
        endpoint_phase_error_norm_upper_bounds=tuple(phase_errors),
        proxy_target_distance_bounds=distances,
        endpoint_target_phase_norm_upper_bounds=tuple(phase_norms),
        endpoint_relative_norm_squared_upper_bounds=tuple(norms),
        endpoint_form_storage_upper_bounds=tuple(form_energies),
        endpoint_phase_excess_storage_upper_bounds=tuple(phase_energies),
        endpoint_excess_storage_upper_bounds=tuple(energies),
        target_phase_storage_bounds_by_class=tuple(
            9 * (1 - cos(2 * k * pi / 9)) for k in _WINDINGS
        ),
        initial_zero_winding_margin_bounds=initial_margin,
        initial_zero_winding_certified=initial_zero,
        radius_angle_bounds=radius_angle,
        acute_radius_margin_bounds=acute_margin,
        acute_radius_certified=acute,
        phase_hessian_lower_bound_candidate=hessian_candidate,
        phase_hessian_lower_bound=hessian_lower,
        positive_phase_hessian_certified=positive_hessian,
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
