"""History forecasts with one unknown held coupling shared by both branches.

The critical history provides a reference forecast, not a general-coupling
propagator. Exact matched matrix polynomials bound finite coupling variation
before source absolute values, with separate truncation and nonlinear errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import (
    _admit_metzler_exponential,
    _prepared_exponential_series_tail_upper,
)
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ._sine_contact_information import (
    _admit_contact_support_information,
    _prepared_coordinate_forecast,
)
from ._sine_history_information import (
    _prepared_sine_history_information,
    _PreparedSineHistoryInformation,
)
from ._sine_joint_information import (
    _admit_affine_initial_information,
    _prepared_sine_tube,
)
from ._sine_preparation import _prepared_coordinate_vector

_COUPLING_POLYNOMIAL_DEGREE = 16


@dataclass(frozen=True)
class _PreparedSineHistoryCouplingInformation:
    """One forecast for every admitted source and every common held gamma.

    critical_reference is freshly rebuilt from normalized primitive inputs.
    Its kernels, reached-state enclosures and dynamic bounds refer only to
    gamma=1/2. They are not substituted for unknown-coupling reached states.
    The separate tubes and nonlinear errors here bound the entire supplied
    coupling interval under the actual complete law.

    parameter_center_coefficients and parameter_generator_coefficients are
    exact degree-16 polynomials in e=gamma-1/2, after all spatial degrees and
    original source columns have been contracted. The latter follows the
    original uncertainty_labels, one polynomial per column. Coefficients are
    ordered by ascending e power, including zeros. Their constant term is a
    truncated reference; the point forecast uses the complete critical heat
    enclosure instead. Polynomial variation and omitted-series allowance
    jointly bound a response difference, not a conditional minimax equality.

    linear_information_error_bounds describe the reference tangent source.
    Coupling variation, general-law nonlinear error, critical computational
    center error and readout error separately enter the final forecast. No
    physical acquisition, automatic switch or reduced dynamics is asserted.
    """

    critical_reference: _PreparedSineHistoryInformation
    gamma_bounds: tuple[Q, Q]
    coupling_deviation_upper_bound: Q
    polynomial_degree: int
    parameter_center_coefficients: tuple[Q, ...]
    parameter_generator_coefficients: tuple[tuple[Q, ...], ...]
    parameter_polynomial_error_upper_bound: Q
    parameter_tail_error_upper_bound: Q
    parameter_response_error_upper_bound: Q
    initial_state_upper_bound: Q
    series_norm_step: Q
    series_tail_upper_bound: Q
    preparation_mobility_degree: Q
    off_history_mobility_degree: Q
    on_history_mobility_degree: Q
    preparation_phase_growth_bound: Q
    preparation_form_tube_upper_bound: Q
    preparation_phase_tube_upper_bound: Q
    preparation_form_error_upper_bound: Q
    preparation_phase_error_upper_bound: Q
    off_phase_growth_bound: Q
    on_phase_growth_bound: Q
    off_phase_tube_upper_bound: Q
    on_phase_tube_upper_bound: Q
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
    reference_gamma: Q = Q(1, 2)
    gamma_information_policy: str = (
        "unknown_common_held_gamma_for_both_stages_and_branches"
    )
    scope: tuple[str, ...] = (
        "complete_held_weighted_quarter_sine_law_with_one_unknown_common_gamma",
        "independent_coupling_interval_contains_critical_reference_one_half",
        "same_gamma_and_original_affine_parameters_through_both_stages",
        "unchanged_full_state_at_prescribed_contact_without_reset_or_forcing",
        "critical_reference_rebuilt_from_normalized_primitive_inputs",
        "critical_reached_enclosures_are_not_unknown_coupling_reached_states",
        "ordered_full_tangent_product_without_noncritical_nilpotent_extension",
        "matched_signed_graph_and_source_contraction_before_parameter_absolute_values",
        "degree16_centered_parameter_polynomial_and_exact_uniform_series_tail",
        "fresh_general_coupling_nonlinear_error_from_original_state_and_total_time",
        "one_point_forecast_covers_all_couplings_and_all_initial_sources",
        "parameter_error_is_a_sufficient_bound_not_a_minimax_equality",
        "no_generic_signed_solver_trajectory_or_retained_producer_replay",
        "no_autonomous_event_information_acquisition_or_computation_speedup_claim",
    )


def _prepared_divided_matrix_powers(matrix, duration, degree):
    """Build exact divided powers after complete consumer admission."""
    size = len(matrix)
    powers = [tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))]
    for order in range(1, degree + 1):
        product = exact_matrix_product(powers[-1], matrix)
        powers.append(
            tuple(tuple(value * duration / order for value in row) for row in product)
        )
    return tuple(powers)


def _prepared_coupling_channel_rows(degree):
    """First rows of M_gamma powers, as exact polynomials in gamma-1/2."""
    zero = (Q(0),) * (degree + 1)
    first, second = (Q(1),) + zero[1:], zero
    rows = [(first, second)]
    for _ in range(degree):
        gamma_first = tuple(
            value / 2 + (first[j - 1] if j else 0) for j, value in enumerate(first)
        )
        gamma_second = tuple(
            value / 2 + (second[j - 1] if j else 0) for j, value in enumerate(second)
        )
        first = tuple(a - b for a, b in zip(first, gamma_second))
        second = gamma_first
        rows.append((first, second))
    return tuple(rows)


def _prepared_matched_coupling_polynomials(
    *,
    off_generator,
    on_generator,
    observation_weights,
    preparation_time,
    elapsed_time,
    form_columns,
    phase_columns,
):
    """Contract the actual ordered two-stage tangent through total degree 16.

    For total degree p and response degree j, the spatial row is
    ell*(Gon**j-Goff**j)*Goff**(p-j)*h**j*tau**(p-j)/(j!*(p-j)!).
    Its channel factor is the first row of M_gamma**p, since the same gamma
    is held in both stages. No nodal-generator commutation is assumed. This
    helper has no channel-reflection or altered-handoff interpretation.
    """
    degree = _COUPLING_POLYNOMIAL_DEGREE
    size, count = len(off_generator), len(form_columns[0])
    coefficients = [[Q(0)] * (degree + 1) for _ in range(count)]
    prefix = _prepared_divided_matrix_powers(off_generator, preparation_time, degree)
    off = _prepared_divided_matrix_powers(off_generator, elapsed_time, degree)
    on = _prepared_divided_matrix_powers(on_generator, elapsed_time, degree)
    matched_rows = tuple(
        exact_matrix_product(
            (observation_weights,),
            tuple(
                tuple(a - b for a, b in zip(on_row, off_row))
                for on_row, off_row in zip(on_matrix, off_matrix)
            ),
        )[0]
        for on_matrix, off_matrix in zip(on, off)
    )
    channel_rows = _prepared_coupling_channel_rows(degree)
    for total_degree in range(1, degree + 1):
        row = [Q(0)] * size
        for response_degree in range(1, total_degree + 1):
            term = exact_matrix_product(
                (matched_rows[response_degree],), prefix[total_degree - response_degree]
            )[0]
            row = [a + b for a, b in zip(row, term)]
        form = exact_matrix_product((tuple(row),), form_columns)[0]
        phase = exact_matrix_product((tuple(row),), phase_columns)[0]
        form_polynomial, phase_polynomial = channel_rows[total_degree]
        for column, (x, phi) in enumerate(zip(form, phase)):
            for power, (a, b) in enumerate(zip(form_polynomial, phase_polynomial)):
                coefficients[column][power] += a * x + b * phi
    return tuple(tuple(row) for row in coefficients)


def _prepared_sine_history_coupling_information(
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
    coordinate_labels,
    initial_center,
    initial_generators,
    uncertainty_labels,
    preparation_relation_cost,
    preparation_time,
    elapsed_time,
    mean_kind,
    reading_error_bound,
    tolerance,
    gamma_bounds,
) -> _PreparedSineHistoryCouplingInformation:
    """Certify one critical-center forecast for an unknown coupling interval.

    The complete actual law is x'=D*(-L*x+gamma*S(theta))/4,
    theta'=gamma*D*L*x/4. One fixed unknown gamma is common to the unchanged
    old-support preparation and both contact branches. Its nonnegative
    ordered interval must contain 1/2. The full original affine source,
    uncertain retained coordinates and original phase origin are preserved.

    Write e=gamma-1/2 and eta=max(abs(e)). The exact degree-16 matched tangent
    polynomial is contracted with the original center and each generator
    before summing absolute nonconstant coefficients times eta**power. For
    k=max mobility degree and T=tau+h, z=(1+gamma_max)*k*T/2<=1 bounds the
    complete signed tangent norm. The two matched tails at gamma and 1/2
    add at most 4*max(B0,P0)*R17(z), with the shared scalar geometric tail.

    Fresh general-law nonlinear comparison uses zg=gamma_max*k*T/2<=1/2
    and both dynamic rows over the entire history. The critical reference
    only supplies its full tangent forecast and computational enclosure.
    All primitives and every domain are admitted before either matrix work
    or that fresh reference call. Zero coupling deviation, response time or
    contact gives exact zero parameter loss after admission. A zero contact
    or response duration also gives zero matched nonlinear allowance, while
    prefix and individual branch bounds retain their separate meanings.
    """
    prepared = _admit_contact_support_information(
        nodes=nodes,
        old_edges=old_edges,
        old_conductances=old_conductances,
        capacity=capacity,
        contact=contact,
        contact_weight=contact_weight,
        retained_nodes=retained_nodes,
        observation_nodes=observation_nodes,
        elapsed_time=elapsed_time,
        mean_kind=mean_kind,
        reading_error_bound=reading_error_bound,
        tolerance=tolerance,
    )
    initial = _admit_affine_initial_information(
        prepared,
        phase_anchor=phase_anchor,
        coordinate_labels=coordinate_labels,
        initial_center=initial_center,
        initial_generators=initial_generators,
        uncertainty_labels=uncertainty_labels,
        preparation_relation_cost=preparation_relation_cost,
        require_known_retained=False,
    )
    gamma = _prepared_coordinate_vector(gamma_bounds, "gamma_bounds", 2)
    if not 0 <= gamma[0] <= Q(1, 2) <= gamma[1]:
        raise ValueError("gamma_bounds must be nonnegative, ordered and contain 1/2")
    deviation = max(Q(1, 2) - gamma[0], gamma[1] - Q(1, 2))
    prefix_time = exact_or_represented_real(preparation_time, "preparation_time")
    if prefix_time < 0:
        raise ValueError("preparation_time must be nonnegative")
    time, total_time = prepared.elapsed_time, prefix_time + prepared.elapsed_time
    size, dimension = len(prepared.nodes), len(initial.uncertainty_labels)
    matrices = (prepared.off_generator, prepared.on_generator)
    degrees = tuple(4 * max(-matrix[i][i] for i in range(size)) for matrix in matrices)
    history_degrees = tuple(max(degrees[0], degree) for degree in degrees)
    prefix_tube = _prepared_sine_tube(
        prefix_time, degrees[0], initial.form_ceiling, initial.phase_ceiling, gamma[1]
    )
    history_tubes = tuple(
        _prepared_sine_tube(
            total_time, degree, initial.form_ceiling, initial.phase_ceiling, gamma[1]
        )
        for degree in history_degrees
    )
    norm_step = (1 + gamma[1]) * max(history_degrees) * total_time / 2
    if norm_step > 1:
        raise ValueError("coupling polynomial requires (1+gamma_max)*k*T/2<=1")
    _admit_metzler_exponential(matrices[0], prefix_time / 2)
    for matrix in matrices:
        _admit_metzler_exponential(matrix, time / 2)

    # Normalize one-shot inputs before the fresh critical consumer re-admits
    # them. No externally supplied report or cached verdict is consumed.
    critical = _prepared_sine_history_information(
        nodes=prepared.nodes,
        old_edges=prepared.old_edges,
        old_conductances=prepared.old_conductances,
        capacity=prepared.capacity,
        contact=prepared.contact,
        contact_weight=prepared.contact_weight,
        retained_nodes=prepared.retained_nodes,
        observation_nodes=prepared.observation_nodes,
        phase_anchor=phase_anchor,
        coordinate_labels=initial.labels,
        initial_center=initial.center,
        initial_generators=initial.generators,
        uncertainty_labels=initial.uncertainty_labels,
        preparation_relation_cost=preparation_relation_cost,
        preparation_time=prefix_time,
        elapsed_time=time,
        mean_kind=prepared.mean_kind,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
    )
    degree = _COUPLING_POLYNOMIAL_DEGREE
    state_bound = max(initial.form_ceiling, initial.phase_ceiling)
    series_tail = _prepared_exponential_series_tail_upper(norm_step, degree)
    if not prepared.contact_weight or not time:
        coefficients = ((Q(0),) * (degree + 1),) * (dimension + 1)
        polynomial_error = tail_error = nonlinear_error = Q(0)
    else:
        coefficients = _prepared_matched_coupling_polynomials(
            off_generator=matrices[0],
            on_generator=matrices[1],
            observation_weights=prepared.observation_weights,
            preparation_time=prefix_time,
            elapsed_time=time,
            form_columns=tuple(
                (initial.center[i], *initial.generators[i])
                for i in initial.form_indices
            ),
            phase_columns=tuple(
                (initial.center[i], *initial.generators[i])
                for i in initial.phase_indices
            ),
        )
        polynomial_error = sum(
            (
                deviation**power * sum((abs(row[power]) for row in coefficients), Q(0))
                for power in range(1, degree + 1)
            ),
            Q(0),
        )
        tail_error = 4 * state_bound * series_tail if deviation else Q(0)
        nonlinear_error = sum((tube.form_error for tube in history_tubes), Q(0))
    parameter_error = polynomial_error + tail_error
    budget = _prepared_coordinate_forecast(
        coefficient_bounds=(critical.tangent_center_bounds,)
        + critical.tangent_generator_coefficient_bounds,
        center=(Q(1),) + (Q(0),) * dimension,
        radii=(Q(0),) + (Q(1),) * dimension,
        model_error_bound=parameter_error + nonlinear_error,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
    )
    return _PreparedSineHistoryCouplingInformation(
        critical_reference=critical,
        gamma_bounds=gamma,
        coupling_deviation_upper_bound=deviation,
        polynomial_degree=degree,
        parameter_center_coefficients=coefficients[0],
        parameter_generator_coefficients=coefficients[1:],
        parameter_polynomial_error_upper_bound=polynomial_error,
        parameter_tail_error_upper_bound=tail_error,
        parameter_response_error_upper_bound=parameter_error,
        initial_state_upper_bound=state_bound,
        series_norm_step=norm_step,
        series_tail_upper_bound=series_tail,
        preparation_mobility_degree=degrees[0],
        off_history_mobility_degree=history_degrees[0],
        on_history_mobility_degree=history_degrees[1],
        preparation_phase_growth_bound=prefix_tube.growth,
        preparation_form_tube_upper_bound=prefix_tube.form_tube,
        preparation_phase_tube_upper_bound=prefix_tube.phase_tube,
        preparation_form_error_upper_bound=prefix_tube.form_error,
        preparation_phase_error_upper_bound=prefix_tube.phase_error,
        off_phase_growth_bound=history_tubes[0].growth,
        on_phase_growth_bound=history_tubes[1].growth,
        off_phase_tube_upper_bound=history_tubes[0].phase_tube,
        on_phase_tube_upper_bound=history_tubes[1].phase_tube,
        off_nonlinear_error_upper_bound=history_tubes[0].form_error,
        on_nonlinear_error_upper_bound=history_tubes[1].form_error,
        nonlinear_response_error_upper_bound=nonlinear_error,
        linear_information_error_bounds=budget.linear_information_error_bounds,
        information_error_lower_bound=budget.information_error_lower_bound,
        computational_center_error_upper_bound=budget.computational_center_error_upper_bound,
        predicted_response=budget.predicted_response,
        forecast_error_upper_bound=budget.forecast_error_upper_bound,
        forecast_response_bounds=budget.forecast_response_bounds,
        status=budget.status,
        response_sign=budget.response_sign,
    )
