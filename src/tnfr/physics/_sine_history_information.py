"""Contact forecasts preserving a declared unforced preparation history.

The source affine parameters pass through both complete critical tangent
stages. Uniform nonlinear allowances refer to that same original source.
Reached coordinate intervals are outer enclosures, not exact projections or
independent replacement parameters at the prescribed contact time.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import (
    MetzlerExponentialBounds,
    _admit_metzler_exponential,
    metzler_exponential_bounds,
)
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ._sine_contact_information import (
    _admit_contact_support_information,
    _interval_row_exact_columns,
    _interval_row_heat_bounds,
    _prepared_coordinate_forecast,
    _signed_heat_row_bounds,
)
from ._sine_joint_information import (
    _admit_affine_initial_information,
    _critical_affine_columns,
    _prepared_critical_sine_tube,
    _PreparedSineJointInformation,
)


@dataclass(frozen=True, kw_only=True)
class _PreparedSineHistoryInformation(_PreparedSineJointInformation):
    """Original-source prediction through an unforced preparation and contact.

    Original center, loadings and coordinate_marginal_bounds follow
    coordinate_labels. Retained coordinates may be uncertain. The phase
    origin is fixed at preparation time, without resetting any reached row.

    preparation_tangent_center_bounds and preparation_tangent_generator_bounds
    enclose the complete reached tangent map in the same row/column ordering.
    preparation_marginal_bounds are outer nonlinear coordinate enclosures;
    they are neither exact reached projections nor an alternative source set.
    No nonlinear joint image or independent hybrid membership is computed.

    off/on_mobility_degree describe the response supports. Their history
    counterparts bound both stages. off/on growth, phase tubes and nonlinear
    errors concern total_time and the original source. The prefix has its own
    separately reported tubes and form/phase errors. Coefficient intervals
    enclose the composed tangent response before latent absolute values.

    Inherited uncertainty_dimension counts latent columns, possibly redundant.
    preparation_relation_cost is conditional metadata, not authenticated cost,
    physical rank, new measurements, entropy reduction or identity formation.
    """

    preparation_time: Q
    total_time: Q
    preparation_heat_time: Q
    preparation_kernel: MetzlerExponentialBounds
    preparation_mobility_degree: Q
    off_history_mobility_degree: Q
    on_history_mobility_degree: Q
    preparation_phase_growth_bound: Q
    preparation_form_tube_upper_bound: Q
    preparation_phase_tube_upper_bound: Q
    preparation_form_error_upper_bound: Q
    preparation_phase_error_upper_bound: Q
    preparation_tangent_center_bounds: tuple[tuple[Q, Q], ...]
    preparation_tangent_generator_bounds: tuple[tuple[tuple[Q, Q], ...], ...]
    preparation_marginal_bounds: tuple[tuple[Q, Q], ...]
    scope: tuple[str, ...] = (
        "complete_held_weighted_quarter_sine_law_at_gamma_one_half",
        "common_unforced_old_support_preparation_before_one_additive_contact",
        "continuous_whole_state_at_prescribed_switch_without_resets_or_forcing",
        "same_original_affine_parameters_retained_through_both_stages",
        "initial_and_reached_retained_coordinates_may_be_uncertain",
        "retained_phase_anchor_zero_only_at_original_preparation_time",
        "fixed_original_phase_origin_and_common_circular_rotation_free",
        "exact_source_column_association_before_interval_projection",
        "piecewise_full_law_remainder_uses_original_form_and_phase_ceilings",
        "prefix_tangent_and_nonlinear_coordinate_bounds_are_outer_enclosures",
        "no_exact_reached_marginal_or_channel_product_membership_computation",
        "actual_information_lower_bound_subtracts_uniform_nonlinear_error",
        "no_nonlinear_affine_corner_extrema_or_tangent_verdict_transfer",
        "conditional_relation_cost_is_not_authenticated_acquisition_or_rank",
        "full_support_preprocessing_and_zero_evolved_coordinates",
        "no_trajectory_event_autonomous_reduction_speedup_or_physical_admission",
    )


def _composed_affine_response_bounds(
    *,
    preparation_generator,
    preparation_heat_time,
    preparation_kernel,
    response_generators,
    response_heat_time,
    response_kernels,
    observation_weights,
    form_columns,
    phase_columns,
):
    """Pull the matched response back to its original affine parameters.

    With a=ell*(Hon-Hoff) and b=s*ell*(Gon*Hon-Goff*Hoff), the rows are
    rx=a*Hp and rq=(r*a*Gpre+b)*Hp. They act on x0 and x0+phi0. Gpre
    commutes with its own Hp; no commutation with either response law is used.
    """
    heat_rows = tuple(
        _signed_heat_row_bounds(observation_weights, kernel)
        for kernel in response_kernels
    )
    phase_rows = tuple(
        _signed_heat_row_bounds(
            tuple(
                response_heat_time * value
                for value in exact_matrix_product((observation_weights,), generator)[0]
            ),
            kernel,
        )
        for generator, kernel in zip(response_generators, response_kernels)
    )
    a, b = (
        tuple(
            (on_lo - off_hi, on_hi - off_lo)
            for (off_lo, off_hi), (on_lo, on_hi) in zip(off, on)
        )
        for off, on in (heat_rows, phase_rows)
    )
    a_generator = _interval_row_exact_columns(a, preparation_generator)
    q_row = tuple(
        (preparation_heat_time * lo + blo, preparation_heat_time * hi + bhi)
        for (lo, hi), (blo, bhi) in zip(a_generator, b)
    )
    rx = _interval_row_heat_bounds(a, preparation_kernel)
    rq = _interval_row_heat_bounds(q_row, preparation_kernel)
    joint_columns = tuple(
        tuple(x + phase for x, phase in zip(xrow, prow))
        for xrow, prow in zip(form_columns, phase_columns)
    )
    x_bounds = _interval_row_exact_columns(rx, form_columns)
    q_bounds = _interval_row_exact_columns(rq, joint_columns)
    return tuple(
        (xlo + qlo, xhi + qhi) for (xlo, xhi), (qlo, qhi) in zip(x_bounds, q_bounds)
    )


def _prepared_sine_history_information(
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
) -> _PreparedSineHistoryInformation:
    """Predict a later original contact response from a declared full history.

    The complete law is x'=D*(-L*x+S(theta)/2)/4, theta'=D*L*x/8.
    It evolves on old support for preparation_time, then either continues or
    adds the supplied contact for elapsed_time. Capacities remain positive and
    held. The prescribed switch installs no automatic event or state reset.

    The original affine image m+V*u has independent u in [-1,1]. Labels name
    every node/channel once and every latent column explicitly. Unlike the
    direct joint consumer, retained rows may have nonzero loadings. Only the
    original retained phase anchor must have zero center and loadings. Phase
    marginals initially lie in [-1/4,1/4] relative to that fixed origin; later
    phases need not keep the anchor zero or remain in this initial chart.

    For each complete branch use k=max(kpre,kresponse), T=tau+h,
    z=k*T/4<=1/2, C=1/(1-z*z/2). Original marginal ceilings B0,P0 give
    form error T*k*(P0+z*B0)**3*C**4/6. Piecewise heat contractivity and both
    evolving rows justify this bound without independently restarting the
    reached coordinates. The prefix has its own form error and phase error
    bounded by zpre times that form error.

    All raw information, charts, time and all branch/work domains precede
    any kernel or zero-response shortcut. A zero contact or response time
    gives exactly zero matched response while preserving computed nonzero
    preparation-state enclosures. Full-support preprocessing is not an
    autonomous reduced flow or a computational speedup claim.
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
    prefix_time = exact_or_represented_real(preparation_time, "preparation_time")
    if prefix_time < 0:
        raise ValueError("preparation_time must be nonnegative")
    time = prepared.elapsed_time
    total_time = prefix_time + time
    prefix_heat_time, heat_time = prefix_time / 2, time / 2
    size = len(prepared.nodes)
    dimension = len(initial.uncertainty_labels)
    matrices = (prepared.off_generator, prepared.on_generator)
    degrees = tuple(4 * max(-matrix[i][i] for i in range(size)) for matrix in matrices)
    history_degrees = tuple(max(degrees[0], degree) for degree in degrees)
    prefix_tube = _prepared_critical_sine_tube(
        prefix_time, degrees[0], initial.form_ceiling, initial.phase_ceiling
    )
    history_tubes = tuple(
        _prepared_critical_sine_tube(
            total_time, degree, initial.form_ceiling, initial.phase_ceiling
        )
        for degree in history_degrees
    )
    _admit_metzler_exponential(matrices[0], prefix_heat_time)
    for matrix in matrices:
        _admit_metzler_exponential(matrix, heat_time)

    # Prefix enclosures are computed even when the future contrast is zero.
    prefix_kernel = metzler_exponential_bounds(matrices[0], prefix_heat_time)
    form_columns = tuple(
        (initial.center[i], *initial.generators[i]) for i in initial.form_indices
    )
    phase_columns = tuple(
        (initial.center[i], *initial.generators[i]) for i in initial.phase_indices
    )
    form_factors, phase_factors = _critical_affine_columns(
        matrices[0], prefix_heat_time, form_columns, phase_columns
    )
    prefix_columns = [None] * (2 * size)
    prefix_marginals = [None] * (2 * size)
    for indices, factors, error in (
        (initial.form_indices, form_factors, prefix_tube.form_error),
        (initial.phase_indices, phase_factors, prefix_tube.phase_error),
    ):
        for node_index, coordinate_index in enumerate(indices):
            row = tuple(
                zip(prefix_kernel.lower[node_index], prefix_kernel.upper[node_index])
            )
            bounds = _interval_row_exact_columns(row, factors)
            prefix_columns[coordinate_index] = bounds
            radius = sum((max(abs(lo), abs(hi)) for lo, hi in bounds[1:]), Q(0))
            prefix_marginals[coordinate_index] = (
                bounds[0][0] - radius - error,
                bounds[0][1] + radius + error,
            )

    if not prepared.contact_weight or not time:
        off_kernel = on_kernel = None
        coefficients = ((Q(0), Q(0)),) * (dimension + 1)
        nonlinear_error = Q(0)
    else:
        off_kernel = metzler_exponential_bounds(matrices[0], heat_time)
        on_kernel = metzler_exponential_bounds(matrices[1], heat_time)
        coefficients = _composed_affine_response_bounds(
            preparation_generator=matrices[0],
            preparation_heat_time=prefix_heat_time,
            preparation_kernel=prefix_kernel,
            response_generators=matrices,
            response_heat_time=heat_time,
            response_kernels=(off_kernel, on_kernel),
            observation_weights=prepared.observation_weights,
            form_columns=form_columns,
            phase_columns=phase_columns,
        )
        nonlinear_error = sum((tube.form_error for tube in history_tubes), Q(0))
    budget = _prepared_coordinate_forecast(
        coefficient_bounds=coefficients,
        center=(Q(1),) + (Q(0),) * dimension,
        radii=(Q(0),) + (Q(1),) * dimension,
        model_error_bound=nonlinear_error,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
    )
    return _PreparedSineHistoryInformation(
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
        total_time=total_time,
        preparation_heat_time=prefix_heat_time,
        elapsed_time=time,
        mean_kind=prepared.mean_kind,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
        hidden_nodes=prepared.hidden_nodes,
        index_old_edges=prepared.index_old_edges,
        index_contact=prepared.index_contact,
        retained_indices=prepared.retained_indices,
        hidden_indices=prepared.hidden_indices,
        observation_indices=prepared.observation_indices,
        observation_weights=prepared.observation_weights,
        form_coordinate_indices=initial.form_indices,
        phase_coordinate_indices=initial.phase_indices,
        coordinate_marginal_bounds=initial.marginals,
        off_generator=matrices[0],
        on_generator=matrices[1],
        heat_time=heat_time,
        preparation_kernel=prefix_kernel,
        off_kernel=off_kernel,
        on_kernel=on_kernel,
        preparation_tangent_center_bounds=tuple(row[0] for row in prefix_columns),
        preparation_tangent_generator_bounds=tuple(row[1:] for row in prefix_columns),
        preparation_marginal_bounds=tuple(prefix_marginals),
        tangent_center_bounds=coefficients[0],
        tangent_generator_coefficient_bounds=coefficients[1:],
        initial_form_upper_bound=initial.form_ceiling,
        initial_phase_upper_bound=initial.phase_ceiling,
        preparation_mobility_degree=degrees[0],
        off_mobility_degree=degrees[0],
        on_mobility_degree=degrees[1],
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
        static_coordinate_count=2 * size,
        uncertainty_dimension=dimension,
    )
