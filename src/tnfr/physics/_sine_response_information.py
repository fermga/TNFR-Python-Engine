"""Necessary and sufficient coordinate information for a finite heat response.

This is the exact gamma=0 member of the held weighted quarter-sine law.
Initial coordinate measurements constrain a box before the response; no
future response is used as a measured input. No trajectory is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._comparison_flow import (
    MetzlerExponentialBounds,
    _admit_metzler_exponential,
    metzler_exponential_bounds,
)
from ._sine_contact_information import (
    _admit_contact_coordinate_information,
    _prepared_coordinate_forecast,
)


@dataclass(frozen=True)
class _PreparedSineResponseInformation:
    """One fixed-horizon forecast and its conditional information limitation.

    Coefficients and full vectors follow nodes. Prior/posterior hidden bounds
    follow the full-order complement of retained_nodes; measurement vectors
    follow measured_nodes. Retained initial forms follow retained_nodes.

    information_error_bounds encloses the exact minimax error for the true
    response over the posterior box. Its lower endpoint can prove that no
    predictor using precisely these data attains the tolerance. The actual
    rational predicted_response has the additional computational center
    allowance. forecast_error_upper_bound also includes the independently
    supplied future reading error, and response_sign concerns that complete
    forecast interval. An unresolved result is neither failure nor success.
    """

    nodes: tuple
    old_edges: tuple[tuple, ...]
    old_conductances: tuple[Q, ...]
    capacity: tuple[Q, ...]
    contact: tuple
    contact_weight: Q
    retained_nodes: tuple
    observation_nodes: tuple
    retained_initial_form: tuple[Q, ...]
    hidden_prior_form_bounds: tuple[tuple[Q, Q], ...]
    measured_nodes: tuple
    measurement_values: tuple[Q, ...]
    measurement_error_bounds: tuple[Q, ...]
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
    off_kernel: MetzlerExponentialBounds | None
    on_kernel: MetzlerExponentialBounds | None
    response_coefficient_bounds: tuple[tuple[Q, Q], ...]
    hidden_posterior_form_bounds: tuple[tuple[Q, Q], ...]
    posterior_center: tuple[Q, ...]
    posterior_radii: tuple[Q, ...]
    information_error_bounds: tuple[Q, Q]
    computational_center_error_upper_bound: Q
    predicted_response: Q
    forecast_error_upper_bound: Q
    forecast_response_bounds: tuple[Q, Q]
    status: str
    response_sign: str
    measurement_cost: int
    scope: tuple[str, ...] = (
        "exact_gamma_zero_held_positive_capacity_weighted_quarter_law",
        "common_initial_form_and_clock_on_off_branches_differ_only_by_additive_contact",
        "no_later_input_event_or_coefficient_change_and_no_phase_data_consumed",
        "original_constituent_arithmetic_or_actual_inverse_capacity_form_mean",
        "independent_initial_coordinate_measurements_intersect_hidden_prior_boxes",
        "unit_cost_per_distinct_measured_hidden_coordinate_no_response_encoded_measurement",
        "exact_conditional_box_minimax_radius_has_attainable_opposite_corners",
        "coefficient_enclosures_bound_information_radius_without_midpoint_rank_claims",
        "rational_forecast_adds_computational_center_and_future_reading_allowances",
        "fixed_horizon_scalar_forecast_evolves_zero_coordinates_but_prepares_full_matrices",
        "no_autonomous_reduced_flow_speedup_nonlinear_gamma_transfer_or_physical_admission",
    )


def _prepared_sine_response_information(
    *,
    nodes,
    old_edges,
    old_conductances,
    capacity,
    contact,
    contact_weight,
    retained_nodes,
    observation_nodes,
    retained_initial_form,
    hidden_prior_form_bounds,
    measured_nodes,
    measurement_values,
    measurement_error_bounds,
    elapsed_time,
    mean_kind,
    reading_error_bound,
    tolerance,
) -> _PreparedSineResponseInformation:
    """Forecast one contact response from a declared initial information budget.

    The complete law is x'=-D*L*x/4, theta'=0, with held positive capacities,
    symmetric nonnegative conductances and no later forcing or events. Old
    support is simple; the additive contact may overlap an old edge. R is a
    nonempty proper region containing both contact endpoints and the original
    nonempty observation. The prior hidden box and every coordinate reading
    are independently available before the response. All exact or represented
    scalars retain their supplied values; no response or cached report is input.

    For a=ell*(exp(h*G_on)-exp(h*G_off)), posterior center m and radii r,
    the exact minimax error is rho=sum(abs(a_j)*r_j). Both extreme corners
    are admissible, and a*m attains this radius. Coefficient intervals give
    rational lower/upper bounds for rho. Their midpoints define the actual
    forecast, with center allowance sum(coefficient_radius_j*abs(m_j)).

    Every primitive, posterior intersection and both matrix work domains is
    admitted before either exponential. Numerical policies are n<=24,
    diagonal_shift*h<=1 and h*norm(shifted_generator)<=32. Exact zero contact
    or time uses an identical-branch identity only after this admission.
    Outside those identities, subtracting fixed-order branch enclosures need
    not resolve arbitrarily tiny contact responses. No missing observation,
    nonlinear transfer, exact finite-time rank or computational gain is inferred.
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
        retained_coordinates=retained_initial_form,
        hidden_prior_bounds=hidden_prior_form_bounds,
        measured_nodes=measured_nodes,
        measurement_values=measurement_values,
        measurement_error_bounds=measurement_error_bounds,
        elapsed_time=elapsed_time,
        mean_kind=mean_kind,
        reading_error_bound=reading_error_bound,
        tolerance=tolerance,
    )
    size = len(prepared.nodes)
    weight, time = prepared.contact_weight, prepared.elapsed_time
    off, on = prepared.off_generator, prepared.on_generator
    observation_weights = prepared.observation_weights
    # Both branch policies precede the first kernel, including zero shortcuts.
    _admit_metzler_exponential(off, time)
    _admit_metzler_exponential(on, time)
    if not weight or not time:
        off_kernel = on_kernel = None
        coefficients = ((Q(0), Q(0)),) * size
    else:
        off_kernel = metzler_exponential_bounds(off, time)
        on_kernel = metzler_exponential_bounds(on, time)
        coefficients = tuple(
            (
                sum(
                    (
                        v * (on_kernel.lower[i][j] - off_kernel.upper[i][j])
                        for i, v in enumerate(observation_weights)
                    ),
                    Q(0),
                ),
                sum(
                    (
                        v * (on_kernel.upper[i][j] - off_kernel.lower[i][j])
                        for i, v in enumerate(observation_weights)
                    ),
                    Q(0),
                ),
            )
            for j in range(size)
        )
    budget = _prepared_coordinate_forecast(
        coefficient_bounds=coefficients,
        center=prepared.posterior_center,
        radii=prepared.posterior_radii,
        model_error_bound=Q(0),
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
    )
    return _PreparedSineResponseInformation(
        nodes=prepared.nodes,
        old_edges=prepared.old_edges,
        old_conductances=prepared.old_conductances,
        capacity=prepared.capacity,
        contact=prepared.contact,
        contact_weight=prepared.contact_weight,
        retained_nodes=prepared.retained_nodes,
        observation_nodes=prepared.observation_nodes,
        retained_initial_form=prepared.retained_coordinates,
        hidden_prior_form_bounds=prepared.hidden_prior_bounds,
        measured_nodes=prepared.measured_nodes,
        measurement_values=prepared.measurement_values,
        measurement_error_bounds=prepared.measurement_error_bounds,
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
        observation_weights=observation_weights,
        off_generator=off,
        on_generator=on,
        off_kernel=off_kernel,
        on_kernel=on_kernel,
        response_coefficient_bounds=coefficients,
        hidden_posterior_form_bounds=prepared.hidden_posterior_bounds,
        posterior_center=prepared.posterior_center,
        posterior_radii=prepared.posterior_radii,
        information_error_bounds=budget.linear_information_error_bounds,
        computational_center_error_upper_bound=budget.computational_center_error_upper_bound,
        predicted_response=budget.predicted_response,
        forecast_error_upper_bound=budget.forecast_error_upper_bound,
        forecast_response_bounds=budget.forecast_response_bounds,
        status=budget.status,
        response_sign=budget.response_sign,
        measurement_cost=len(prepared.measured_nodes),
    )
