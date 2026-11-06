"""Detached certified cubic observation of a prepared mobile encounter.

The frozen protocol is LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-protocol.
This integrates an analytic actual-tangent response, not a nonlinear trajectory.
Identity, numerical completion and the two response verdicts remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._gauss_legendre import composite_gauss_remainder, gauss_legendre_rule
from ..mathematics._rational_interval import I, pi_interval, sqrt
from ._localized_mobile_interaction import (
    _bound_localized_mobile_interaction,
    _LocalizedMobileInteractionBound,
)
from ._localized_mobile_spectral import _prepare_mobile_spectral_kernel


@dataclass(frozen=True)
class _LocalizedMobileResponse:
    """One endpoint observation with explicit unavailable and null boundaries."""

    budget: _LocalizedMobileInteractionBound
    quadrature_order: int
    panel_limit: int
    panels: int
    numerical_error_target: Q
    derivative_growth_upper_bound: Q
    integrand_derivative_upper_bound: Q
    quadrature_error_upper_bound: Q
    spectral_modes: tuple[int, ...] = ()
    spectral_quadrature: tuple[tuple[I, ...], ...] = ()
    arithmetic_error_upper_bound: Q | None = None
    quadrature_norm_bounds: I | None = None
    numerical_error_upper_bound: Q | None = None
    cubic_norm_bounds: I | None = None
    nominal_defect_norm_bounds: I | None = None
    uniform_defect_norm_bounds: I | None = None
    integrand_evaluations: int = 0
    numerical_certified: bool = False
    nominal_interaction_certified: bool = False
    uniform_interaction_certified: bool = False
    status: str = "unavailable"
    unavailable_reasons: tuple[str, ...] = ()
    method: str = "actual_tangent_spectral_gauss_dyadic128_v1"
    scope: tuple[str, ...] = (
        "same_prepared_mobile_pair_positive_loss_complete_law_and_structural_clock",
        "joint_source_is_exact_sum_of_the_same_two_primitive_single_sources",
        "actual_damped_tangent_cubic_Duhamel_integral_not_nonlinear_trajectory",
        "finite_Fourier_support_is_exact_for_the_cubic_proxy_not_a_nonlinear_truncation",
        "deterministic_panel_selection_uses_only_prior_derivative_and_error_bounds",
        "full_state_superposition_defect_is_not_intensity_interference",
        "nominal_and_uniform_shared_source_verdicts_have_separate_error_budgets",
        "no_new_organization_formation_or_physical_identification",
    )


def _spectral_norm_bounds(boxes):
    """Parseval norm of both real channels, retaining every complex component."""
    squared = sum((coordinate**2 for row in boxes for coordinate in row), I(0))
    return sqrt(squared)


def _expand_norm(bounds, allowance):
    return I(max(Q(0), bounds.lo - allowance), bounds.hi + allowance)


def _evaluate_localized_mobile_response(
    *,
    reference_model,
    node_count,
    fejer_order,
    amplitude,
    initial_state_error_bound,
    localization_radius,
    horizon,
    quadrature_order=12,
    max_panels=65536,
    progress=None,
):
    """Enclose the frozen type of cubic endpoint from freshly admitted inputs.

    The error target is a**3/1000. Panels double from one until the proven
    Gauss truncation error is at most one tenth of that target, before any
    integrand value is read. Interval node, weight, source and propagator
    uncertainty is retained separately in the spectral integral boxes.
    Failed enclosure precision never becomes an observed zero response.

    The callable progress hook receives completed-panel metadata only. A hook
    failure aborts evaluation; an evidence runner must retain that attempt.
    """
    budget = _bound_localized_mobile_interaction(
        reference_model=reference_model,
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=amplitude,
        initial_state_error_bound=initial_state_error_bound,
        localization_radius=localization_radius,
        horizon=horizon,
    )
    nodes, weights = gauss_legendre_rule(quadrature_order)
    if type(max_panels) is not int:
        raise TypeError("max_panels must be an ordinary Python integer")
    if not 1 <= max_panels <= 65536:
        raise ValueError("max_panels requires 1 <= max_panels <= 65536")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    a = budget.amplitude
    target = a**3 / 1000
    pi = pi_interval()
    sigma = pi.hi * budget.form_loss_coefficient / budget.exchange_coefficient
    qmax = 2 * pi.hi * Q(fejer_order - 1, node_count)
    growth = (1 + sigma) * (4 + 6 * qmax)
    derivative = 8 * a**3 * growth ** (2 * quadrature_order)
    panels = 1
    while True:
        remainder = composite_gauss_remainder(
            order=quadrature_order,
            panels=panels,
            interval_length=horizon,
            derivative_bound=derivative,
        )
        if remainder <= target / 10 or 2 * panels > max_panels:
            break
        panels *= 2
    common = dict(
        budget=budget,
        quadrature_order=quadrature_order,
        panel_limit=max_panels,
        panels=panels,
        numerical_error_target=target,
        derivative_growth_upper_bound=growth,
        integrand_derivative_upper_bound=derivative,
        quadrature_error_upper_bound=remainder,
    )
    if not budget.cubic_remainder_certified:
        return _LocalizedMobileResponse(
            **common, unavailable_reasons=("analytic_response_budget_not_certified",)
        )
    if remainder > target / 10:
        return _LocalizedMobileResponse(
            **common, unavailable_reasons=("quadrature_panel_budget_exhausted",)
        )
    metadata, evaluate = _prepare_mobile_spectral_kernel(
        reference_model=reference_model,
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=a,
        horizon=horizon,
    )
    totals = [[I(0) for _ in range(4)] for _ in metadata.modes]
    half_step = Q(horizon, 2 * panels)
    for panel in range(panels):
        center = (2 * panel + 1) * half_step
        for node, weight in zip(nodes, weights):
            sample = evaluate(I(center) + half_step * node)
            factor = half_step * weight
            for target_row, source_row in zip(totals, sample):
                for coordinate in range(4):
                    target_row[coordinate] += factor * source_row[coordinate]
        if progress is not None:
            progress({"completed_panels": panel + 1, "panels": panels})
    boxes = tuple(tuple(row) for row in totals)
    # Exact accumulation precedes square-root enclosure, avoiding a rounded
    # square for every tiny component radius on the absolute dyadic grid.
    squared_radius = sum(
        (coordinate.radius**2 for row in boxes for coordinate in row), Q(0)
    )
    arithmetic_error = sqrt(I(squared_radius)).hi
    norm_bounds = _spectral_norm_bounds(boxes)
    numeric_error = remainder + max(arithmetic_error, norm_bounds.radius)
    cubic_norm = _expand_norm(norm_bounds, remainder)
    nominal_norm = _expand_norm(cubic_norm, budget.nominal_cubic_remainder_upper_bound)
    uniform_norm = _expand_norm(
        nominal_norm, budget.shared_preparation_sensitivity_upper_bound
    )
    numerical = numeric_error <= target
    identity = budget.prepared_identity_certified
    reasons = []
    if not numerical:
        reasons.append("interval_arithmetic_budget_exhausted")
    if not identity:
        reasons.append("prepared_identity_not_certified")
    return _LocalizedMobileResponse(
        **common,
        spectral_modes=metadata.modes,
        spectral_quadrature=boxes,
        arithmetic_error_upper_bound=arithmetic_error,
        quadrature_norm_bounds=norm_bounds,
        numerical_error_upper_bound=numeric_error,
        cubic_norm_bounds=cubic_norm,
        nominal_defect_norm_bounds=nominal_norm,
        uniform_defect_norm_bounds=uniform_norm,
        integrand_evaluations=panels * quadrature_order,
        numerical_certified=numerical,
        nominal_interaction_certified=numerical and identity and nominal_norm.lo > 0,
        uniform_interaction_certified=numerical and identity and uniform_norm.lo > 0,
        status="evaluated" if numerical else "unavailable",
        unavailable_reasons=tuple(reasons),
    )
