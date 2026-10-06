"""Exact full-state sensitivity metric near the conservative C5 saddle.

The metric bounds differences of the same declared nonlinear nodal law while
both trajectories remain in an explicitly supplied primitive-phase tube.
It is an analysis norm, not physical storage or an installed forecast method.
Every form/phase mode, including the common origins, is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, sqrt
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)
from .relational_sine_resonance import SineC5LeafSaddle, assess_sine_c5_leaf_saddle

__all__ = ("SineSaddleSensitivity", "assess_sine_saddle_sensitivity")


def _transpose(matrix):
    return tuple(zip(*matrix))


def _block_diagonal(left, right):
    rows, columns = len(left), len(right)
    return tuple(tuple(row) + (Q(0),) * columns for row in left) + tuple(
        (Q(0),) * rows + tuple(row) for row in right
    )


def _congruence(matrix, basis):
    return exact_matrix_product(_transpose(basis), exact_matrix_product(matrix, basis))


@dataclass(frozen=True)
class SineSaddleSensitivity:
    """Conditional two-sided growth bound in a positive full20 metric.

    If the complete compared trajectories remain in the declared phase tube,
    their W-distance grows by at most exp(growth_rate*abs(tau)). The report
    does not establish this whole-time premise for the captured source.
    """

    source: SineExchangeComparison
    saddle: SineC5LeafSaddle
    phase_radius: Q
    degree_weights: tuple[Q, ...]
    relative_basis: tuple[tuple[Q, ...], ...]
    coordinate_basis: tuple[tuple[Q, ...], ...]
    inverse_coordinate_basis: tuple[tuple[Q, ...], ...]
    relative_inverse_mobility: tuple[tuple[Q, ...], ...]
    relative_mobility: tuple[tuple[Q, ...], ...]
    relative_form_metric: tuple[tuple[Q, ...], ...]
    relative_phase_hessian: tuple[tuple[Q, ...], ...]
    relative_mechanical_mass: tuple[tuple[Q, ...], ...]
    relative_phase_metric: tuple[tuple[Q, ...], ...]
    phase_mass_slack: tuple[tuple[Q, ...], ...]
    normalized_laplacian_slack: tuple[tuple[Q, ...], ...]
    full_metric: tuple[tuple[Q, ...], ...]
    inverse_full_metric: tuple[tuple[Q, ...], ...]
    forward_tangent_slack: tuple[tuple[Q, ...], ...]
    reverse_tangent_slack: tuple[tuple[Q, ...], ...]
    tangent_growth_rate_upper_bound: Q
    perturbation_coefficient_upper_bound: Q
    nonlinear_growth_rate_upper_bound: Q
    infinity_to_metric_upper_bound: Q
    metric_to_coordinate_upper_bounds: tuple[Q, ...]
    metric_to_infinity_upper_bound: Q
    metric_positive_definite: bool
    both_tangent_directions_certified: bool
    nonlinear_tube_implication_certified: bool
    captured_source_flow_bound_certified: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    conditional_premises: tuple[str, ...] = (
        "both_compared_trajectories_follow_the_same_complete_unit_conservative_sine_law",
        "every_primitive_phase_of_both_trajectories_stays_within_phase_radius_of_exact_saddle",
        "the_phase_premise_holds_throughout_the_entire_time_interval_including_its_convex_segments",
        "support_unit_capacities_and_clock_stay_fixed_without_forcing_or_events",
    )
    scope: tuple[str, ...] = (
        "source_anchors_support_and_law_not_a_validated_source_trajectory",
        "all_twenty_form_phase_coordinates_and_both_common_origins_are_retained",
        "relative_coordinates_remove_origins_only_temporarily_before_exact_full_reconstruction",
        "positive_analysis_metric_is_not_a_new_energy_law_or_constitutive_parameter",
        "both_forward_and_reverse_tangent_growth_are_checked_by_exact_rational_LMI",
        "nonlinear_extension_uses_phase_curvature_changes_not_a_tangent_extrapolation",
        "form_amplitudes_do_not_enter_the_phase_only_Jacobian_variation_bound",
        "conversion_constants_do_not_license_reboxing_or_repeated_condition_number_factors",
        "no_ellipsoidal_integrator_forecast_horizon_retention_or_physical_claim_is_supplied",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        self.saddle.to_dict()
        return {"schema": "tnfr.sine-saddle-sensitivity.v1", "report": _project(self)}


def assess_sine_saddle_sensitivity(
    source, *, cycle, phase_radius=Q(1, 1000)
) -> SineSaddleSensitivity:
    """Build and check one full-state metric; no trajectory is evaluated.

    Phase radius is a nonnegative exact or represented real radius in radians.
    The declared phase tube is centered at the mathematical saddle, rather
    than a rounded capture. A certified result is a conditional theorem, not
    evidence that the source stays in that tube or that a forecast can finish.
    """
    radius = exact_or_represented_real(phase_radius, "phase_radius")
    if radius < 0:
        raise ValueError("phase_radius must be nonnegative")
    saddle = assess_sine_c5_leaf_saddle(source, cycle=cycle)
    source = saddle.source
    size = len(source.nodes)
    degrees = tuple(map(Q, source.degrees))
    relative = tuple(
        tuple(
            -degrees[column] / degrees[-1] if row == size - 1 else Q(row == column)
            for column in range(size - 1)
        )
        for row in range(size)
    )
    basis = tuple((Q(1),) + row for row in relative)
    inverse_basis = exact_matrix_inverse(basis)
    degree_metric = tuple(
        tuple(degrees[i] if i == j else Q(0) for j in range(size)) for i in range(size)
    )
    gram = _congruence(degree_metric, relative)
    mobility = exact_matrix_inverse(gram)
    form = _congruence(saddle.form_laplacian, relative)
    hessian = _congruence(saddle.phase_hessian, relative)
    mass = exact_matrix_inverse(
        exact_matrix_product(exact_matrix_product(mobility, form), mobility)
    )
    dimension = size - 1
    phase_metric = tuple(
        tuple(hessian[i][j] + mass[i][j] / 6 for j in range(dimension))
        for i in range(dimension)
    )
    phase_slack = tuple(
        tuple(hessian[i][j] + mass[i][j] / 12 for j in range(dimension))
        for i in range(dimension)
    )
    gdg = exact_matrix_product(exact_matrix_product(form, mobility), form)
    laplacian_slack = tuple(
        tuple(2 * form[i][j] - gdg[i][j] for j in range(dimension))
        for i in range(dimension)
    )
    full_form_metric = _congruence(_block_diagonal(((Q(1),),), form), inverse_basis)
    full_phase_metric = _congruence(
        _block_diagonal(((Q(1),),), phase_metric), inverse_basis
    )
    metric = _block_diagonal(full_form_metric, full_phase_metric)
    inverse_metric = _block_diagonal(
        exact_matrix_inverse(full_form_metric), exact_matrix_inverse(full_phase_metric)
    )
    wj = exact_matrix_product(metric, saddle.full_tangent_generator)
    symmetric = tuple(
        tuple(wj[i][j] + wj[j][i] for j in range(2 * size)) for i in range(2 * size)
    )
    rate = Q(1, 3)
    slacks = tuple(
        tuple(
            tuple(
                2 * rate * metric[i][j] + sign * symmetric[i][j]
                for j in range(2 * size)
            )
            for i in range(2 * size)
        )
        for sign in (-1, 1)
    )
    positive = (
        exact_symmetric_semidefinite(form, strict=True)
        and exact_symmetric_semidefinite(mass, strict=True)
        and exact_symmetric_semidefinite(metric, strict=True)
    )
    tangent = positive and all(
        exact_symmetric_semidefinite(matrix) for matrix in slacks
    )
    weights = tuple(value / sum(degrees) for value in degrees)
    origins = inverse_basis[0] == weights
    variation = (
        tangent
        and origins
        and exact_symmetric_semidefinite(phase_slack)
        and exact_symmetric_semidefinite(laplacian_slack)
        and 16**2 * 3 < 28**2
    )
    # ||error||_W <= sqrt(sum|W_ij|)||error||_infinity, and each
    # |error_i| <= sqrt((W^-1)_ii)||error||_W by the exact dual norm.
    into_metric = sqrt(I(sum((abs(value) for row in metric for value in row), Q(0)))).hi
    into_coordinates = tuple(sqrt(I(inverse_metric[i][i])).hi for i in range(2 * size))
    reasons = tuple(
        reason
        for passed, reason in (
            (positive, "full_metric_positive_definiteness_not_certified"),
            (tangent, "both_exact_tangent_growth_inequalities_not_certified"),
            (origins, "full_common_origin_reconstruction_not_certified"),
            (variation, "nonlinear_phase_tube_extension_not_certified"),
        )
        if not passed
    )
    return SineSaddleSensitivity(
        source=source,
        saddle=saddle,
        phase_radius=radius,
        degree_weights=weights,
        relative_basis=relative,
        coordinate_basis=basis,
        inverse_coordinate_basis=inverse_basis,
        relative_inverse_mobility=gram,
        relative_mobility=mobility,
        relative_form_metric=form,
        relative_phase_hessian=hessian,
        relative_mechanical_mass=mass,
        relative_phase_metric=phase_metric,
        phase_mass_slack=phase_slack,
        normalized_laplacian_slack=laplacian_slack,
        full_metric=metric,
        inverse_full_metric=inverse_metric,
        forward_tangent_slack=slacks[0],
        reverse_tangent_slack=slacks[1],
        tangent_growth_rate_upper_bound=rate,
        perturbation_coefficient_upper_bound=Q(28),
        nonlinear_growth_rate_upper_bound=rate + 28 * radius,
        infinity_to_metric_upper_bound=into_metric,
        metric_to_coordinate_upper_bounds=into_coordinates,
        metric_to_infinity_upper_bound=max(into_coordinates),
        metric_positive_definite=positive,
        both_tangent_directions_certified=tangent,
        nonlinear_tube_implication_certified=variation,
        captured_source_flow_bound_certified=False,
        status="certified" if variation else "unavailable",
        reasons=reasons,
    )
