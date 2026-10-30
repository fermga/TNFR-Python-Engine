"""Finite phase-informed forecasts for the complete quarter-sine law.

At fixed gamma=1/2 and zero initial form, an exact coupled tangent response
is available from a heat semigroup. A separate uniform cubic remainder
retains the actual nonlinear sine law and the complete phase feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._comparison_flow import (
    MetzlerExponentialBounds,
    _admit_metzler_exponential,
    metzler_exponential_bounds,
)
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ._sine_contact_information import (
    _admit_contact_coordinate_information,
    _prepared_coordinate_forecast,
    _signed_heat_row_bounds,
)


@dataclass(frozen=True)
class _PreparedSinePhaseInformation:
    """Actual nonlinear forecast with distinct information and model allowances.

    Full vectors follow nodes. Hidden prior/posterior vectors follow the
    full-order complement of retained_nodes. Retained phases and measured
    values follow their own explicitly ordered node lists. Every phase is
    relative to phase_anchor in the declared initial chart; an arbitrary
    common circular rotation is absent from both dynamics and observation.

    linear_information_error_bounds concerns the tangent only. Subtracting
    the uniform nonlinear allowance yields information_error_lower_bound
    for the actual response. No exact nonlinear box extrema are claimed.
    The rational forecast includes both the nonlinear and computational
    allowances, then the independent future reading error. A positive
    sufficient bound or an unresolved verdict does not prove failure.
    """

    nodes: tuple
    old_edges: tuple[tuple, ...]
    old_conductances: tuple[Q, ...]
    capacity: tuple[Q, ...]
    contact: tuple
    contact_weight: Q
    retained_nodes: tuple
    observation_nodes: tuple
    phase_anchor: object
    retained_relative_phase: tuple[Q, ...]
    hidden_prior_relative_phase_bounds: tuple[tuple[Q, Q], ...]
    measured_nodes: tuple
    phase_measurement_values: tuple[Q, ...]
    phase_measurement_error_bounds: tuple[Q, ...]
    elapsed_time: Q
    mean_kind: str
    reading_error_bound: Q
    tolerance: Q
    hidden_nodes: tuple
    index_old_edges: tuple[tuple[int, int], ...]
    index_contact: tuple[int, int]
    retained_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    observation_indices: tuple[int, ...]
    measured_indices: tuple[int, ...]
    observation_weights: tuple[Q, ...]
    off_generator: tuple[tuple[Q, ...], ...]
    on_generator: tuple[tuple[Q, ...], ...]
    heat_time: Q
    off_kernel: MetzlerExponentialBounds | None
    on_kernel: MetzlerExponentialBounds | None
    tangent_response_coefficient_bounds: tuple[tuple[Q, Q], ...]
    hidden_posterior_relative_phase_bounds: tuple[tuple[Q, Q], ...]
    posterior_center: tuple[Q, ...]
    posterior_radii: tuple[Q, ...]
    initial_phase_upper_bound: Q
    off_mobility_degree: Q
    on_mobility_degree: Q
    off_phase_growth_bound: Q
    on_phase_growth_bound: Q
    off_nonlinear_error_upper_bound: Q
    on_nonlinear_error_upper_bound: Q
    nonlinear_response_error_upper_bound: Q
    linear_information_error_bounds: tuple[Q, Q]
    information_error_lower_bound: Q
    computational_center_error_upper_bound: Q
    predicted_response: Q
    forecast_error_upper_bound: Q
    forecast_response_bounds: tuple[Q, Q]
    status: str
    response_sign: str
    measurement_cost: int
    gamma: Q = Q(1, 2)
    phase_chart_radius: Q = Q(1, 4)
    scope: tuple[str, ...] = (
        "complete_held_weighted_quarter_sine_law_at_gamma_one_half",
        "all_initial_forms_exactly_zero_with_both_form_and_phase_feedback_retained",
        "matched_common_initial_phase_on_off_branches_differ_by_one_additive_contact",
        "independently_supplied_initial_relative_chart_about_a_retained_zero_anchor",
        "common_circular_rotation_is_free_no_absolute_phase_reconstruction",
        "initial_measurement_arcs_lie_inside_chart_before_posterior_intersection",
        "unit_cost_per_individually_measured_hidden_relative_phase_coordinate",
        "actual_arithmetic_or_inverse_capacity_form_mean_on_original_constituent",
        "coupled_tangent_phase_to_form_response_uses_signed_heat_semigroup_product",
        "original_prior_phase_ceiling_bounds_complete_nonlinear_cubic_remainder",
        "actual_information_lower_bound_subtracts_uniform_nonlinear_response_error",
        "no_exact_nonlinear_box_extrema_or_automatic_linear_verdict_transfer",
        "computational_center_error_and_future_reading_error_remain_separate",
        "fixed_horizon_forecast_prepares_full_support_and_evolves_zero_coordinates",
        "no_trajectory_event_autonomous_reduction_speedup_or_physical_admission",
    )


def _signed_phase_to_form_bounds(generator, weights, heat_time, kernel):
    """Project s*G*exp(s*G) with signed exact rows, after primitive admission."""
    row = tuple(
        heat_time * value for value in exact_matrix_product((weights,), generator)[0]
    )
    return _signed_heat_row_bounds(row, kernel)


def _prepared_sine_phase_information(
    *,
    nodes,
    old_edges,
    old_conductances,
    capacity,
    contact,
    contact_weight,
    retained_nodes,
    observation_nodes,
    phase_anchor,
    retained_relative_phase,
    hidden_prior_relative_phase_bounds,
    measured_nodes,
    phase_measurement_values,
    phase_measurement_error_bounds,
    elapsed_time,
    mean_kind,
    reading_error_bound,
    tolerance,
) -> _PreparedSinePhaseInformation:
    """Predict a fixed finite nonlinear contact response from relative phases.

    The full held law is x'=D*(-L*x+S(theta)/2)/4, theta'=D*L*x/8,
    with all initial forms zero and no later input or event. Both branches
    share the full initial phase. R is proper, contains the contact endpoints
    and original observation, and has independently supplied relative phases.
    The named anchor lies in R and has exactly zero relative phase.

    Each known or prior relative phase lies in [-1/4,1/4] radians. A measured
    hidden phase has a circular arc |phi-value|<=error wholly inside this
    same initial chart: abs(value)+error<=1/4. These explicit chart premises
    make ordinary real intersections unambiguous; no wrapping, branch choice
    or absolute lift recovery is inferred. They impose no later chart claim.

    For G=-D*L/4 and s=h/2, the exact coupled tangent phase-to-form map is
    s*G*exp(s*G). If k=max(nu_i*degree_i), z=k*h/4<=1/2 and
    C=1/(1-z*z/2), the complete nonlinear form error in each branch is at
    most h*k*delta**3*C**4/6. Delta bounds the original prior and retained
    phases, before measurement clipping. This remainder uses both full rows,
    the heat maximum principle and |sin(u)-u|<=|u|**3/6.

    All raw information, chart conditions and both branch work domains are
    admitted before either exponential. Exact zero time/contact shortcuts
    follow admission and certify the matched response itself. Outside those
    identities the two uniform nonlinear branch allowances are summed.
    """
    prepared = _admit_contact_coordinate_information(
        nodes=nodes,
        old_edges=old_edges,
        old_conductances=old_conductances,
        capacity=capacity,
        contact=contact,
        contact_weight=contact_weight,
        retained_nodes=retained_nodes,
        observation_nodes=observation_nodes,
        retained_coordinates=retained_relative_phase,
        hidden_prior_bounds=hidden_prior_relative_phase_bounds,
        measured_nodes=measured_nodes,
        measurement_values=phase_measurement_values,
        measurement_error_bounds=phase_measurement_error_bounds,
        elapsed_time=elapsed_time,
        mean_kind=mean_kind,
        reading_error_bound=reading_error_bound,
        tolerance=tolerance,
    )
    if phase_anchor not in prepared.retained_nodes:
        raise ValueError("phase_anchor must belong to retained_nodes")
    anchor_slot = prepared.retained_nodes.index(phase_anchor)
    if prepared.retained_coordinates[anchor_slot] != 0:
        raise ValueError("the retained anchor must have exactly zero relative phase")
    initial_ceiling = max(
        *(abs(value) for value in prepared.retained_coordinates),
        *(abs(value) for pair in prepared.hidden_prior_bounds for value in pair),
    )
    if initial_ceiling > Q(1, 4):
        raise ValueError("all original relative phase priors must lie in [-1/4,1/4]")
    if any(
        abs(value) + error > Q(1, 4)
        for value, error in zip(
            prepared.measurement_values, prepared.measurement_error_bounds
        )
    ):
        raise ValueError("each initial circular measurement arc must lie in [-1/4,1/4]")
    size, time = len(prepared.nodes), prepared.elapsed_time
    heat_time = time / 2
    matrices = (prepared.off_generator, prepared.on_generator)
    degrees = tuple(4 * max(-matrix[i][i] for i in range(size)) for matrix in matrices)
    exponents = tuple(degree * time / 4 for degree in degrees)
    if any(exponent > Q(1, 2) for exponent in exponents):
        raise ValueError("phase information requires each mobility_degree*h/4<=1/2")
    for matrix in matrices:
        _admit_metzler_exponential(matrix, heat_time)
    growth = tuple(1 / (1 - exponent**2 / 2) for exponent in exponents)
    remainders = tuple(
        time * degree * initial_ceiling**3 * factor**4 / 6
        for degree, factor in zip(degrees, growth)
    )
    if not prepared.contact_weight or not time:
        off_kernel = on_kernel = None
        coefficients = ((Q(0), Q(0)),) * size
        nonlinear_error = Q(0)
    else:
        off_kernel = metzler_exponential_bounds(matrices[0], heat_time)
        on_kernel = metzler_exponential_bounds(matrices[1], heat_time)
        off_bounds, on_bounds = (
            _signed_phase_to_form_bounds(
                matrix, prepared.observation_weights, heat_time, kernel
            )
            for matrix, kernel in zip(matrices, (off_kernel, on_kernel))
        )
        coefficients = tuple(
            (on_lower - off_upper, on_upper - off_lower)
            for (off_lower, off_upper), (on_lower, on_upper) in zip(
                off_bounds, on_bounds
            )
        )
        nonlinear_error = sum(remainders, Q(0))
    budget = _prepared_coordinate_forecast(
        coefficient_bounds=coefficients,
        center=prepared.posterior_center,
        radii=prepared.posterior_radii,
        model_error_bound=nonlinear_error,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
    )
    return _PreparedSinePhaseInformation(
        nodes=prepared.nodes,
        old_edges=prepared.old_edges,
        old_conductances=prepared.old_conductances,
        capacity=prepared.capacity,
        contact=prepared.contact,
        contact_weight=prepared.contact_weight,
        retained_nodes=prepared.retained_nodes,
        observation_nodes=prepared.observation_nodes,
        phase_anchor=phase_anchor,
        retained_relative_phase=prepared.retained_coordinates,
        hidden_prior_relative_phase_bounds=prepared.hidden_prior_bounds,
        measured_nodes=prepared.measured_nodes,
        phase_measurement_values=prepared.measurement_values,
        phase_measurement_error_bounds=prepared.measurement_error_bounds,
        elapsed_time=prepared.elapsed_time,
        mean_kind=prepared.mean_kind,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
        hidden_nodes=prepared.hidden_nodes,
        index_old_edges=prepared.index_old_edges,
        index_contact=prepared.index_contact,
        retained_indices=prepared.retained_indices,
        hidden_indices=prepared.hidden_indices,
        observation_indices=prepared.observation_indices,
        measured_indices=prepared.measured_indices,
        observation_weights=prepared.observation_weights,
        off_generator=prepared.off_generator,
        on_generator=prepared.on_generator,
        heat_time=heat_time,
        off_kernel=off_kernel,
        on_kernel=on_kernel,
        tangent_response_coefficient_bounds=coefficients,
        hidden_posterior_relative_phase_bounds=prepared.hidden_posterior_bounds,
        posterior_center=prepared.posterior_center,
        posterior_radii=prepared.posterior_radii,
        initial_phase_upper_bound=initial_ceiling,
        off_mobility_degree=degrees[0],
        on_mobility_degree=degrees[1],
        off_phase_growth_bound=growth[0],
        on_phase_growth_bound=growth[1],
        off_nonlinear_error_upper_bound=remainders[0],
        on_nonlinear_error_upper_bound=remainders[1],
        nonlinear_response_error_upper_bound=nonlinear_error,
        linear_information_error_bounds=budget.linear_information_error_bounds,
        information_error_lower_bound=budget.information_error_lower_bound,
        computational_center_error_upper_bound=budget.computational_center_error_upper_bound,
        predicted_response=budget.predicted_response,
        forecast_error_upper_bound=budget.forecast_error_upper_bound,
        forecast_response_bounds=budget.forecast_response_bounds,
        status=budget.status,
        response_sign=budget.response_sign,
        measurement_cost=len(prepared.measured_nodes),
    )
