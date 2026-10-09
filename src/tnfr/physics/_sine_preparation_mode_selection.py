"""Matched-observation source families selecting distinct isolated C9 modes.

Proof: theory/nodal/SINE_PREPARATION_MODE_SELECTION.md. The fixed positive-loss
sine law is unchanged. A freshly verified rational reflection pairs complete
source families with identical initial phases, means and storage. Transit and
capture use the original continuous target lifts, not their circular images.
This private certificate computes no trajectory and supplies no later dwell,
contact, interaction or physical-identification conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._cycle_algebra import dirichlet_energy, dot, laplacian_matrix
from ._sine_preparation import _prepared_transit_radii
from .phase_cycle_geometry import _derive, reconstruct_phase_cycle_state
from .relational_sine_formed_classes import (
    _CAPACITY,
    _EDGES,
    _GAP,
    _NODES,
    _PHASE,
    _SOURCE_SCALE,
    _WINDINGS,
    _verify_cycle_gap,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

_LIFT_SCALE = 512
_STORAGE_BUDGET = Q(10**15)


@dataclass(frozen=True)
class _PairedPreparation:
    """Exact construction; residuals are paired, not independent box corners."""

    normalized_initial_forms_by_class: tuple[tuple[Q, ...], ...]
    initial_forms_by_class: tuple[tuple[Q, ...], ...]
    source_laplacian: tuple[tuple[Q, ...], ...]
    reflection_direction: tuple[Q, ...]
    reflection_denominator: Q
    source_reflection: tuple[tuple[Q, ...], ...]
    reflection_row_absolute_sums: tuple[Q, ...]
    reflection_maximum_row_absolute_sum: Q
    common_normalized_form_quadratic: Q
    common_initial_form_storage: Q
    target_phase_turns_by_class: tuple[tuple[Q, ...], ...]
    target_principal_edge_turns_by_class: tuple[tuple[Q, ...], ...]
    target_sine_balance_certified_by_class: tuple[bool, bool]
    target_windings: tuple[int, int]
    source_pairing_certified: bool


def _paired_source_construction() -> _PairedPreparation:
    """Rebuild the fixed mean-preserving L-isometry and target identities."""
    ramp = tuple(Q(j - 4) for j in _NODES)
    dipole = tuple(Q(int(j == 3) - int(j == 4)) for j in _NODES)
    coefficient = Q(1, _LIFT_SCALE)
    normalized = (
        tuple((9 * _LIFT_SCALE + coefficient) * h + r for h, r in zip(dipole, ramp)),
        tuple(
            (9 * _LIFT_SCALE - coefficient) * h + 2 * r for h, r in zip(dipole, ramp)
        ),
    )
    forms = tuple(tuple(_SOURCE_SCALE * value for value in row) for row in normalized)
    laplacian = tuple(tuple(2 * value for value in row) for row in laplacian_matrix(9))
    direction = tuple(a - b for a, b in zip(*normalized))
    covector = tuple(dot(row, direction) for row in laplacian)
    denominator = dot(direction, covector)
    if denominator <= 0:
        raise ArithmeticError("the fixed source reflection has no positive denominator")
    identity = tuple(tuple(Q(int(i == j)) for j in _NODES) for i in _NODES)
    reflection = tuple(
        tuple(
            identity[i][j] - 2 * direction[i] * covector[j] / denominator
            for j in _NODES
        )
        for i in _NODES
    )
    transposed = tuple(zip(*reflection))
    quadratics = tuple(
        dot(row, tuple(dot(line, row) for line in laplacian)) for row in normalized
    )
    checks = (
        all(sum(row, Q(0)) == 0 for row in normalized),
        quadratics[0] == quadratics[1],
        tuple(dot(row, normalized[0]) for row in reflection) == normalized[1],
        exact_matrix_product(reflection, reflection) == identity,
        exact_matrix_product(transposed, exact_matrix_product(laplacian, reflection))
        == laplacian,
        all(sum(row, Q(0)) == 1 for row in reflection),
        all(sum(column, Q(0)) == 1 for column in transposed),
    )
    if not all(checks):
        raise ArithmeticError(
            "the fixed source pairing failed exact means/storage identities"
        )
    targets = tuple(
        tuple(_LIFT_SCALE * h + Q(k, 9) * r for h, r in zip(dipole, ramp))
        for k in _WINDINGS
    )
    principal = tuple(
        tuple((row[j] - row[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in _EDGES)
        for row in targets
    )
    geometry = _derive(_NODES, _EDGES)
    states = tuple(
        reconstruct_phase_cycle_state(geometry, edge_turns=row) for row in principal
    )
    balanced = tuple(
        state.sine_balance_status == "proved_by_odd_cancellation" for state in states
    )
    windings = tuple(
        sum(((row[(i + 1) % 9] - row[i] + Q(1, 2)) % 1 - Q(1, 2) for i in _NODES), Q(0))
        for row in targets
    )
    if (
        not all(balanced)
        or windings != _WINDINGS
        or any(sum(row, Q(0)) for row in targets)
    ):
        raise ArithmeticError(
            "the fixed target lifts failed equilibrium/mean/winding identities"
        )
    row_sums = tuple(sum(map(abs, row), Q(0)) for row in reflection)
    return _PairedPreparation(
        normalized_initial_forms_by_class=normalized,
        initial_forms_by_class=forms,
        source_laplacian=laplacian,
        reflection_direction=direction,
        reflection_denominator=denominator,
        source_reflection=reflection,
        reflection_row_absolute_sums=row_sums,
        reflection_maximum_row_absolute_sum=max(row_sums),
        common_normalized_form_quadratic=quadratics[0],
        common_initial_form_storage=dirichlet_energy(forms[0]),
        target_phase_turns_by_class=targets,
        target_principal_edge_turns_by_class=principal,
        target_sine_balance_certified_by_class=balanced,
        target_windings=tuple(int(value) for value in windings),
        source_pairing_certified=all(checks),
    )


@dataclass(frozen=True)
class _PreparationModeSelection:
    """Conditional mode distinction for one exactly paired source family.

    Each latent pair (xi,zeta) has the declared component bounds and separate
    exact zero sums. Branches use (m*a+xi,zeta) and (m*b+T*xi,zeta).
    Marginal row budgets below are sufficient enclosures, not replacements for
    this coupling. All sixteen relative directions remain when both budgets
    are positive. Nonpositive sufficient margins mean unavailable, not failed
    formation or a claim that the selected observations determine a mode.
    """

    scaled_time: Q
    form_error_bound: Q
    phase_error_bound: Q
    radius: Q
    original_time: Q
    source: _PairedPreparation
    preparation_dimension: int
    full_relative_preparation: bool
    form_error_bounds_by_class: tuple[tuple[Q, ...], ...]
    common_initial_storage_upper_bound: Q
    source_budget_margin_bounds: I
    source_budget_certified: bool
    gamma_bounds: I
    inverse_gamma_bounds: I
    eta_bounds: I
    forcing_norm_bounds: I
    nominal_scaled_initial_norm_upper_bounds: tuple[Q, Q]
    scaled_initial_error_norm_upper_bounds: tuple[Q, Q]
    scaled_initial_norm_upper_bounds: tuple[Q, Q]
    initial_phase_error_norm_upper_bound: Q
    exponential_decay_bounds: I
    proxy_target_error_bounds_by_class: tuple[tuple[I, ...], ...]
    proxy_target_distance_bounds: tuple[I, I]
    scaled_form_radius_upper_bounds: tuple[Q, Q]
    endpoint_form_norm_upper_bounds: tuple[Q, Q]
    endpoint_phase_error_norm_upper_bounds: tuple[Q, Q]
    endpoint_target_phase_norm_upper_bounds: tuple[Q, Q]
    endpoint_relative_norm_squared_upper_bounds: tuple[Q, Q]
    endpoint_excess_storage_upper_bounds: tuple[Q, Q]
    target_phase_storage_bounds_by_class: tuple[I, I]
    initial_zero_winding_margin_bounds: I
    initial_zero_winding_certified: bool
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    acute_radius_certified: bool
    phase_hessian_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    barrier_lower_bound: Q | None
    endpoint_radius_margin_bounds: tuple[I, I]
    endpoint_radius_certified_by_class: tuple[bool, bool]
    storage_barrier_margin_bounds: tuple[I, I] | None
    storage_barrier_certified_by_class: tuple[bool, bool]
    formation_certified_by_class: tuple[bool, bool]
    symmetry_inequivalent: bool
    matched_observation_mode_distinction_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    class_order: tuple[str, str] = ("winding_one", "winding_two")
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    held_capacities: tuple[Q, ...] = _CAPACITY
    initial_phases: tuple[Q, ...] = _PHASE
    metric_weights: tuple[Q, ...] = (Q(2),) * 9
    nominal_source_scale: Q = _SOURCE_SCALE
    target_lift_scale: int = _LIFT_SCALE
    common_source_storage_budget: Q = _STORAGE_BUDGET
    conserved_form_mean: Q = Q(0)
    conserved_lifted_phase_mean: Q = Q(0)
    latent_residual_constraints: tuple[str, ...] = (
        "xi_and_zeta_have_separate_exact_zero_sums",
        "branch_one_form_residual_is_xi_branch_two_is_source_reflection_times_xi",
        "the_same_zeta_is_the_lifted_phase_residual_in_both_branches",
    )
    matched_initial_observations: tuple[str, ...] = (
        "complete_initial_phase_row",
        "conserved_form_mean",
        "conserved_lifted_phase_mean",
        "complete_initial_storage",
    )
    lambda_lower_bound: Q = _GAP
    laplacian_upper_bound: Q = Q(4)
    form_loss: Q = Q(1023, 1024)
    exchange_weight: Q = Q(1, 1024)
    phase_exchange_beta: Q = Q(1)
    deformation: Q = Q(0)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_C9_complete_positive_loss_sine_law_and_unit_capacities",
        "source_pairing_and_means_storage_identity_rebuilt_from_exact_primitives",
        "every_latent_realization_has_equal_observations_across_its_two_branches",
        "storage_may_vary_between_different_latent_realizations",
        "correlated_source_family_is_not_the_product_of_marginal_error_boxes",
        "unwrapped_target_lifts_for_transit_wrapped_edge_turns_for_equilibrium",
        "full_family_semigroup_transit_then_strict_radius_and_storage_capture",
        "source_budget_and_capture_flags_remain_separate",
        "no_trajectory_source_search_cached_verdict_or_old_assessor_evaluation",
        "no_later_dwell_contact_response_or_physical_identification_transfer",
    )


def _assess_preparation_mode_selection(
    *, scaled_time, form_error_bound, phase_error_bound, radius
) -> _PreparationModeSelection:
    """Re-admit budgets and bound both branches of the fixed paired family.

    Time and residual bounds are nonnegative; radius is positive. The shared
    exponential resource limit is scaled_time/5 <= 4096. The source lift scale
    512 and storage budget 10**15 are fixed construction/policy constants.
    All raw scalars are admitted before source or interval computations.
    """
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

    source = _paired_source_construction()
    _verify_cycle_gap()
    pi = pi_interval()
    gamma, inverse_gamma = Q(1, 1023) / pi, 1023 * pi
    eta, forcing = Q(1, 1023**2) / pi**2, sqrt(I(18))
    error_rows = (
        (rx,) * 9,
        tuple(rx * value for value in source.reflection_row_absolute_sums),
    )
    nominal_norms = tuple(
        gamma * sqrt(I(sum((2 * value**2 for value in row), Q(0))))
        for row in source.initial_forms_by_class
    )
    form_errors = tuple(
        (gamma * sqrt(I(sum((2 * value**2 for value in row), Q(0))))).hi
        for row in error_rows
    )
    phase_error = (forcing * rt).hi
    initial_norms = tuple(
        value.hi + error for value, error in zip(nominal_norms, form_errors)
    )
    decay = I(*_negative_exp_bounds(_GAP * time))
    # These compare actual continuous representatives. Wrapping here would
    # erase the preparation mechanism even though the target circle is unchanged.
    proxy_errors = tuple(
        tuple(gamma * value - 2 * pi * turn for value, turn in zip(form, target))
        for form, target in zip(
            source.initial_forms_by_class, source.target_phase_turns_by_class
        )
    )
    distances = tuple(
        sqrt(sum((value**2 for value in row), I(0))) for row in proxy_errors
    )
    inverse_root_two = (1 / sqrt(I(2))).hi
    scaled_radii, form_norms, phase_errors, phase_norms, norms, energies = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for initial, source_error, distance in zip(initial_norms, form_errors, distances):
        scaled_radius, phase_radius = _prepared_transit_radii(
            time=time,
            decay_upper=decay.hi,
            gap_lower=_GAP,
            initial_norm_upper=initial,
            feedback_upper=eta.hi,
            forcing_upper=forcing.hi,
        )
        form_norm = inverse_gamma.hi * scaled_radius * inverse_root_two
        phase_remainder = (phase_radius + source_error + phase_error) * inverse_root_two
        target_phase_norm = distance.hi + phase_remainder
        norm = form_norm**2 + target_phase_norm**2
        scaled_radii.append(scaled_radius)
        form_norms.append(form_norm)
        phase_errors.append(phase_remainder)
        phase_norms.append(target_phase_norm)
        norms.append(norm)
        energies.append(2 * norm)

    # Compute the complete source bound once. The exact L-isometry transfers
    # its form energy to branch two; zeta is shared, so its potential is equal.
    first_form = source.initial_forms_by_class[0]
    source_cost = (
        sum(
            ((abs(first_form[j] - first_form[i]) + 2 * rx) ** 2 / 2 for i, j in _EDGES),
            Q(0),
        )
        + 18 * rt**2
    )
    budget_margin = I(_STORAGE_BUDGET - source_cost)
    budget_flag = budget_margin.lo > 0
    initial_margin = pi / 2 - 2 * rt
    initial_zero = initial_margin.lo > 0
    radius_angle = 4 * pi / 9 + sqrt(I(2)) * r
    acute_margin = pi / 2 - radius_angle
    hessian = cos(radius_angle).lo if acute_margin.lo > 0 else None
    acute = hessian is not None and hessian > 0
    hessian = hessian if acute else None
    coercivity = _GAP * hessian if hessian is not None else None
    barrier = (I(coercivity) * r**2).lo if coercivity is not None else None
    radius_margins = tuple(I(r**2 - value) for value in norms)
    radius_flags = tuple(value.lo > 0 for value in radius_margins)
    storage_margins = (
        tuple(I(barrier - value) for value in energies) if barrier is not None else None
    )
    storage_flags = tuple(
        storage_margins is not None and storage_margins[i].lo > 0 for i in (0, 1)
    )
    formation = tuple(
        initial_zero and acute and radius_flags[i] and storage_flags[i] for i in (0, 1)
    )
    inequivalent = abs(source.target_windings[0]) != abs(source.target_windings[1])
    certified = (
        source.source_pairing_certified
        and budget_flag
        and inequivalent
        and all(formation)
    )
    reasons = tuple(
        reason
        for condition, reason in (
            (budget_flag, "strict_common_source_storage_budget_not_certified"),
            (initial_zero, "whole_initial_set_zero_winding_not_certified"),
            (acute, "strict_acute_radius_not_certified"),
            (inequivalent, "distinct_symmetry_classes_not_certified"),
        )
        if not condition
    ) + tuple(
        f"{name}:{reason}"
        for i, name in enumerate(("winding_one", "winding_two"))
        for condition, reason in (
            (radius_flags[i], "strict_endpoint_radius_not_certified"),
            (storage_flags[i], "strict_excess_storage_barrier_not_certified"),
        )
        if not condition
    )
    return _PreparationModeSelection(
        scaled_time=time,
        form_error_bound=rx,
        phase_error_bound=rt,
        radius=r,
        original_time=time / Q(1023, 1024),
        source=source,
        preparation_dimension=8 * int(rx > 0) + 8 * int(rt > 0),
        full_relative_preparation=rx > 0 and rt > 0,
        form_error_bounds_by_class=error_rows,
        common_initial_storage_upper_bound=source_cost,
        source_budget_margin_bounds=budget_margin,
        source_budget_certified=budget_flag,
        gamma_bounds=gamma,
        inverse_gamma_bounds=inverse_gamma,
        eta_bounds=eta,
        forcing_norm_bounds=forcing,
        nominal_scaled_initial_norm_upper_bounds=tuple(
            value.hi for value in nominal_norms
        ),
        scaled_initial_error_norm_upper_bounds=form_errors,
        scaled_initial_norm_upper_bounds=initial_norms,
        initial_phase_error_norm_upper_bound=phase_error,
        exponential_decay_bounds=decay,
        proxy_target_error_bounds_by_class=proxy_errors,
        proxy_target_distance_bounds=distances,
        scaled_form_radius_upper_bounds=tuple(scaled_radii),
        endpoint_form_norm_upper_bounds=tuple(form_norms),
        endpoint_phase_error_norm_upper_bounds=tuple(phase_errors),
        endpoint_target_phase_norm_upper_bounds=tuple(phase_norms),
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
        phase_hessian_lower_bound=hessian,
        coercivity_lower_bound=coercivity,
        barrier_lower_bound=barrier,
        endpoint_radius_margin_bounds=radius_margins,
        endpoint_radius_certified_by_class=radius_flags,
        storage_barrier_margin_bounds=storage_margins,
        storage_barrier_certified_by_class=storage_flags,
        formation_certified_by_class=formation,
        symmetry_inequivalent=inequivalent,
        matched_observation_mode_distinction_certified=certified,
        status=(
            "certified_matched_observation_mode_selection"
            if certified
            else "unavailable"
        ),
        unavailable_reasons=reasons,
    )
