"""Finite contact forecasts from supplied joint form and phase preparations.

An affine image of an independent unit box retains deterministic preparation
relations. The critical coupled tangent is projected through that image before
interval arithmetic, with a separate full-law nonlinear error allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._comparison_flow import (
    MAX_COMPARISON_DIMENSION,
    MetzlerExponentialBounds,
    _admit_metzler_exponential,
    metzler_exponential_bounds,
)
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ._sine_contact_information import (
    _admit_contact_support_information,
    _interval_row_exact_columns,
    _prepared_coordinate_forecast,
    _signed_heat_row_bounds,
)
from ._sine_preparation import _prepared_coordinate_vector
from .relational_observations import _ordered


@dataclass(frozen=True)
class _PreparedSineJointInformation:
    """Conditional affine-preparation forecast under the complete sine law.

    Center, generator rows and marginal bounds follow coordinate_labels;
    columns follow uncertainty_labels. These labels describe independent
    unit-box parameters, not acquired observations or physical state rank.
    uncertainty_dimension counts those columns, including redundant ones;
    it is not the rank of the affine image.
    The supplied integer preparation_relation_cost is declared metadata and
    authenticates no preparation or measurement process.

    The tangent information radius is exact for its affine source when its
    coefficients are exact. Its reported enclosure and uniform nonlinear
    allowance bound actual unavoidable error and the returned point forecast
    separately. Nonlinear extrema need not occur at affine-box corners.
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
    coordinate_labels: tuple[tuple, ...]
    initial_center: tuple[Q, ...]
    initial_generators: tuple[tuple[Q, ...], ...]
    uncertainty_labels: tuple
    preparation_relation_cost: int
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
    observation_weights: tuple[Q, ...]
    form_coordinate_indices: tuple[int, ...]
    phase_coordinate_indices: tuple[int, ...]
    coordinate_marginal_bounds: tuple[tuple[Q, Q], ...]
    off_generator: tuple[tuple[Q, ...], ...]
    on_generator: tuple[tuple[Q, ...], ...]
    heat_time: Q
    off_kernel: MetzlerExponentialBounds | None
    on_kernel: MetzlerExponentialBounds | None
    tangent_center_bounds: tuple[Q, Q]
    tangent_generator_coefficient_bounds: tuple[tuple[Q, Q], ...]
    initial_form_upper_bound: Q
    initial_phase_upper_bound: Q
    off_mobility_degree: Q
    on_mobility_degree: Q
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
    static_coordinate_count: int
    uncertainty_dimension: int
    measurement_cost: int = 0
    evolved_coordinate_count: int = 0
    gamma: Q = Q(1, 2)
    phase_chart_radius: Q = Q(1, 4)
    scope: tuple[str, ...] = (
        "complete_held_weighted_quarter_sine_law_at_gamma_one_half",
        "matched_common_affine_initial_state_on_off_one_additive_contact",
        "supplied_deterministic_joint_preparation_is_not_a_covariance_model",
        "explicit_node_channel_rows_and_independent_unit_box_columns",
        "all_retained_coordinates_known_and_retained_phase_anchor_zero",
        "initial_relative_phase_marginals_admitted_in_one_common_chart",
        "common_circular_rotation_free_no_later_chart_or_fixed_anchor_claim",
        "exact_joint_affine_contraction_precedes_heat_interval_projection",
        "nonzero_initial_form_enters_the_complete_nonlinear_phase_tube",
        "actual_information_lower_bound_subtracts_uniform_nonlinear_error",
        "no_exact_nonlinear_affine_corner_extrema_or_tangent_verdict_transfer",
        "conditional_relation_cost_is_not_authenticated_acquisition_or_rank",
        "no_individually_acquired_coordinates_or_latent_measurement_invention",
        "full_support_preprocessing_and_zero_evolved_coordinates",
        "no_trajectory_event_autonomous_reduction_speedup_or_physical_admission",
    )


@dataclass(frozen=True)
class _AffineInitialInformation:
    labels: tuple[tuple, ...]
    center: tuple[Q, ...]
    generators: tuple[tuple[Q, ...], ...]
    uncertainty_labels: tuple
    form_indices: tuple[int, ...]
    phase_indices: tuple[int, ...]
    marginals: tuple[tuple[Q, Q], ...]
    form_ceiling: Q
    phase_ceiling: Q


def _admit_affine_initial_information(
    prepared,
    *,
    phase_anchor,
    coordinate_labels,
    initial_center,
    initial_generators,
    uncertainty_labels,
    preparation_relation_cost,
    require_known_retained,
):
    """Admit original affine coordinates against freshly admitted support.

    The consumer selects whether retained coordinates must be known. A
    retained anchor with zero center and loadings fixes only the original
    phase origin; this helper asserts no later phase or coordinate reset.
    """
    size = len(prepared.nodes)
    labels = _ordered(coordinate_labels, "coordinate_labels", limit=2 * size + 1)
    labels = tuple(_ordered(label, "coordinate label", limit=3) for label in labels)
    if (
        len(labels) != 2 * size
        or any(len(label) != 2 for label in labels)
        or any(
            node not in prepared.nodes
            or type(channel) is not str
            or channel not in ("form", "relative_phase")
            for node, channel in labels
        )
        or len(set(labels)) != 2 * size
    ):
        raise ValueError("coordinate_labels must name each node and channel once")
    uncertainty_labels = _ordered(
        uncertainty_labels,
        "uncertainty_labels",
        limit=2 * MAX_COMPARISON_DIMENSION + 1,
    )
    dimension = len(uncertainty_labels)
    if (
        dimension > 2 * MAX_COMPARISON_DIMENSION
        or len(set(uncertainty_labels)) != dimension
    ):
        raise ValueError("uncertainty_labels must be distinct with at most 48 entries")
    center = _prepared_coordinate_vector(initial_center, "initial_center", 2 * size)
    rows = _ordered(initial_generators, "initial_generators", limit=2 * size + 1)
    if len(rows) != 2 * size:
        raise ValueError("initial_generators requires one row per coordinate label")
    generators = tuple(
        _prepared_coordinate_vector(row, f"initial_generators[{i}]", dimension)
        for i, row in enumerate(rows)
    )
    if type(preparation_relation_cost) is not int or preparation_relation_cost < 0:
        raise ValueError("preparation_relation_cost must be a nonnegative integer")
    if phase_anchor not in prepared.retained_nodes:
        raise ValueError("phase_anchor must belong to retained_nodes")
    positions = {label: i for i, label in enumerate(labels)}
    form_indices = tuple(positions[(node, "form")] for node in prepared.nodes)
    phase_indices = tuple(
        positions[(node, "relative_phase")] for node in prepared.nodes
    )
    if require_known_retained and any(
        any(generators[positions[(node, channel)]])
        for node in prepared.retained_nodes
        for channel in ("form", "relative_phase")
    ):
        raise ValueError("all retained form and phase generator rows must be zero")
    anchor_index = positions[(phase_anchor, "relative_phase")]
    if center[anchor_index] != 0 or any(generators[anchor_index]):
        raise ValueError("the retained anchor must have exactly zero relative phase")
    radii = tuple(sum(map(abs, row), Q(0)) for row in generators)
    marginals = tuple(
        (value - radius, value + radius) for value, radius in zip(center, radii)
    )
    form_ceiling = max(abs(center[i]) + radii[i] for i in form_indices)
    phase_ceiling = max(abs(center[i]) + radii[i] for i in phase_indices)
    if phase_ceiling > Q(1, 4):
        raise ValueError("all affine relative phase marginals must lie in [-1/4,1/4]")

    return _AffineInitialInformation(
        labels,
        center,
        generators,
        uncertainty_labels,
        form_indices,
        phase_indices,
        marginals,
        form_ceiling,
        phase_ceiling,
    )


@dataclass(frozen=True)
class _CriticalSineTube:
    exponent: Q
    growth: Q
    phase_tube: Q
    form_tube: Q
    form_error: Q
    phase_error: Q


def _prepared_critical_sine_tube(time, degree, form_ceiling, phase_ceiling):
    """Bound a held or piecewise-held critical law from its original source.

    All arguments are admitted nonnegative exact quantities. For a piecewise
    law degree bounds every stage and time is the entire elapsed duration.
    No reached coordinate is restarted as an independent initial parameter.
    """
    return _prepared_sine_tube(time, degree, form_ceiling, phase_ceiling, Q(1, 2))


def _prepared_sine_tube(time, degree, form_ceiling, phase_ceiling, gamma):
    """Bound a complete held or piecewise-held sine law at admitted gamma.

    Arguments are prepared nonnegative exact quantities. gamma and degree
    may be upper bounds shared by every stage. Form heat contractivity and
    both evolving rows give the phase comparison; the global sine residual
    supplies the form error and its integrated phase counterpart. Source
    ceilings always refer to the original state, without a handoff reset.
    """
    exponent = gamma * degree * time / 2
    if exponent > Q(1, 2):
        raise ValueError("sine tube requires gamma*mobility_degree*time/2<=1/2")
    growth = 1 / (1 - exponent**2 / 2)
    source = phase_ceiling + exponent * form_ceiling
    phase_tube = source * growth
    form_tube = form_ceiling + exponent * phase_tube
    form_error = time * gamma * degree * source**3 * growth**4 / 3
    return _CriticalSineTube(
        exponent, growth, phase_tube, form_tube, form_error, exponent * form_error
    )


def _critical_affine_columns(generator, heat_time, form_columns, phase_columns):
    """Form both exact critical factors before the common heat action."""
    joint = tuple(
        tuple(x + phase for x, phase in zip(xrow, prow))
        for xrow, prow in zip(form_columns, phase_columns)
    )
    derivative = exact_matrix_product(generator, joint)
    form = tuple(
        tuple(x + heat_time * dx for x, dx in zip(xrow, drow))
        for xrow, drow in zip(form_columns, derivative)
    )
    phase = tuple(
        tuple(value - heat_time * dx for value, dx in zip(prow, drow))
        for prow, drow in zip(phase_columns, derivative)
    )
    return form, phase


def _critical_affine_branch_bounds(
    generator, weights, heat_time, kernel, form_columns, phase_columns
):
    """Project exact joint columns before enclosing the critical heat action.

    The first column carries the affine center and the rest independent unit
    parameters. Combining form and phase before enclosure preserves supplied
    cancellations in x+s*G*(x+phi), including x+phi=0 directions.
    """
    columns, _ = _critical_affine_columns(
        generator, heat_time, form_columns, phase_columns
    )
    heat_row = _signed_heat_row_bounds(weights, kernel)
    return _interval_row_exact_columns(heat_row, columns)


def _prepared_sine_joint_information(
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
    elapsed_time,
    mean_kind,
    reading_error_bound,
    tolerance,
) -> _PreparedSineJointInformation:
    """Predict the original matched response from a supplied joint affine set.

    The complete law is x'=D*(-L*x+S(theta)/2)/4, theta'=D*L*x/8.
    Each initial state is m+V*u with independently varying u_j in [-1,1].
    coordinate_labels contains every (node, 'form') and
    (node, 'relative_phase') exactly once, in the order of m and V's rows.
    uncertainty_labels names V's columns; zero through 48 are supported.
    Neither a column nor its name is a measured state coordinate.

    All retained rows have zero generator loadings. The retained phase anchor
    also has zero center. Every phase marginal lies in [-1/4,1/4] radians,
    specifying an initial relative chart and permitting any common rotation.
    These are supplied preparation premises, not inferred acquisitions or
    future invariants. preparation_relation_cost is a nonnegative integer
    declared by that independent preparation protocol; it affects no bound.

    At s=h/2, the exact critical tangent is H*(x+s*G*(x+phi)),
    H=exp(s*G), G=-D*L/4. The affine center and each generator are combined
    exactly before interval projection. If B0 and P0 bound the original
    form and phase marginals, each branch with k=max(nu_i*degree_i),
    z=k*h/4<=1/2 and C=1/(1-z*z/2) has form remainder at most
    h*k*(P0+z*B0)**3*C**4/6. This includes full phase feedback.

    All primitive information, charts and both branch domains precede either
    heat kernel and exact zero shortcuts. The work limits bound computation,
    not the mathematical range of the conditional comparison theorem.
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
        require_known_retained=True,
    )
    size = len(prepared.nodes)
    labels = initial.labels
    center = initial.center
    generators = initial.generators
    uncertainty_labels = initial.uncertainty_labels
    dimension = len(uncertainty_labels)
    form_indices = initial.form_indices
    phase_indices = initial.phase_indices
    marginals = initial.marginals
    form_ceiling = initial.form_ceiling
    phase_ceiling = initial.phase_ceiling

    time = prepared.elapsed_time
    heat_time = time / 2
    matrices = (prepared.off_generator, prepared.on_generator)
    degrees = tuple(4 * max(-matrix[i][i] for i in range(size)) for matrix in matrices)
    exponents = tuple(degree * time / 4 for degree in degrees)
    if any(exponent > Q(1, 2) for exponent in exponents):
        raise ValueError("joint information requires each mobility_degree*h/4<=1/2")
    for matrix in matrices:
        _admit_metzler_exponential(matrix, heat_time)
    tubes = tuple(
        _prepared_critical_sine_tube(time, degree, form_ceiling, phase_ceiling)
        for degree in degrees
    )
    growth = tuple(tube.growth for tube in tubes)
    phase_tubes = tuple(tube.phase_tube for tube in tubes)
    remainders = tuple(tube.form_error for tube in tubes)
    if not prepared.contact_weight or not time:
        off_kernel = on_kernel = None
        coefficients = ((Q(0), Q(0)),) * (dimension + 1)
        nonlinear_error = Q(0)
    else:
        form_columns = tuple((center[i], *generators[i]) for i in form_indices)
        phase_columns = tuple((center[i], *generators[i]) for i in phase_indices)
        off_kernel = metzler_exponential_bounds(matrices[0], heat_time)
        on_kernel = metzler_exponential_bounds(matrices[1], heat_time)
        off_bounds, on_bounds = (
            _critical_affine_branch_bounds(
                matrix,
                prepared.observation_weights,
                heat_time,
                kernel,
                form_columns,
                phase_columns,
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
    # The first scalar carries a known affine offset; remaining scalars are
    # mathematical unit-box parameters, never invented coordinate readings.
    budget = _prepared_coordinate_forecast(
        coefficient_bounds=coefficients,
        center=(Q(1),) + (Q(0),) * dimension,
        radii=(Q(0),) + (Q(1),) * dimension,
        model_error_bound=nonlinear_error,
        reading_error_bound=prepared.reading_error_bound,
        tolerance=prepared.tolerance,
    )
    return _PreparedSineJointInformation(
        nodes=prepared.nodes,
        old_edges=prepared.old_edges,
        old_conductances=prepared.old_conductances,
        capacity=prepared.capacity,
        contact=prepared.contact,
        contact_weight=prepared.contact_weight,
        retained_nodes=prepared.retained_nodes,
        observation_nodes=prepared.observation_nodes,
        phase_anchor=phase_anchor,
        coordinate_labels=labels,
        initial_center=center,
        initial_generators=generators,
        uncertainty_labels=uncertainty_labels,
        preparation_relation_cost=preparation_relation_cost,
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
        form_coordinate_indices=form_indices,
        phase_coordinate_indices=phase_indices,
        coordinate_marginal_bounds=marginals,
        off_generator=matrices[0],
        on_generator=matrices[1],
        heat_time=heat_time,
        off_kernel=off_kernel,
        on_kernel=on_kernel,
        tangent_center_bounds=coefficients[0],
        tangent_generator_coefficient_bounds=coefficients[1:],
        initial_form_upper_bound=form_ceiling,
        initial_phase_upper_bound=phase_ceiling,
        off_mobility_degree=degrees[0],
        on_mobility_degree=degrees[1],
        off_phase_growth_bound=growth[0],
        on_phase_growth_bound=growth[1],
        off_phase_tube_upper_bound=phase_tubes[0],
        on_phase_tube_upper_bound=phase_tubes[1],
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
        static_coordinate_count=2 * size,
        uncertainty_dimension=dimension,
    )
