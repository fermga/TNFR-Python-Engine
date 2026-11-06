"""Primitive-support refinement of the conditional two-contact sine bound.

This detached reader retains actual weighted paths, marked endpoint incidence
and a fixed observation. It supplies no trajectory, event or mean evolution
law. Both complete quarter-sine rows use the same conductance weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import (
    MAX_COMPARISON_DIMENSION,
    MAX_NONNEGATIVE_EXPONENTIAL_STEP,
    NonnegativeExponentialUpper,
    nonnegative_exponential_upper,
)
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._sine_preparation import (
    _prepared_coordinate_vector,
    _prepared_two_contact_composition,
    _PreparedTwoContactComposition,
)
from .relational_observations import _ordered
from .relational_sine_support import _support_topology
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT
from .support_transport import _support_components


@dataclass(frozen=True)
class _PreparedSupportContactComposition:
    """Nodal mixed bounds and two fixed normalized observation projections.

    Each nodal sensitivity bounds ``abs(partial12 x_i)+abs(partial12 theta_i)``.
    Its maximum is not the previous sum of coordinate maximum norms. The
    freshly rebuilt ``universal`` report retains that distinct norm and its
    admitted scalar/error data. True zero flags certify the corresponding
    all-node or observed contrast under the premises; False is inconclusive.
    Positive-support components are ordered index sets, including isolates.
    All observation weights align with ``nodes`` and vanish outside the region.
    """

    nodes: tuple
    old_edges: tuple[tuple, ...]
    old_conductances: tuple[Q, ...]
    capacity: tuple[Q, ...]
    first_contact: tuple
    second_contact: tuple
    observation_nodes: tuple
    old_edge_chart_margins: tuple[Q, ...] | None
    index_old_edges: tuple[tuple[int, int], ...]
    index_first_contact: tuple[int, int]
    index_second_contact: tuple[int, int]
    observation_indices: tuple[int, ...]
    joined_adjacency: tuple[tuple[Q, ...], ...]
    weighted_degrees: tuple[Q, ...]
    positive_support_components: tuple[tuple[int, ...], ...]
    arithmetic_observation_weights: tuple[Q, ...]
    inverse_capacity_observation_weights: tuple[Q, ...]
    comparison_matrix: tuple[tuple[Q, ...], ...]
    first_contact_derivative_matrix: tuple[tuple[Q, ...], ...]
    second_contact_derivative_matrix: tuple[tuple[Q, ...], ...]
    first_forcing_constant: tuple[Q, ...]
    second_forcing_constant: tuple[Q, ...]
    first_forcing_linear: tuple[Q, ...]
    second_forcing_linear: tuple[Q, ...]
    kernel: NonnegativeExponentialUpper
    first_sensitivity_slope_upper_bounds: tuple[Q, ...]
    second_sensitivity_slope_upper_bounds: tuple[Q, ...]
    mixed_contact_forcing_upper_bounds: tuple[Q, ...]
    mixed_hessian_upper_bounds: tuple[Q, ...]
    raw_nodal_mixed_sensitivity_upper_bounds: tuple[Q, ...]
    nodal_mixed_sensitivity_upper_bounds: tuple[Q, ...]
    nodal_mixed_defect_upper_bounds: tuple[Q, ...]
    arithmetic_defect_upper_bound: Q
    inverse_capacity_defect_upper_bound: Q
    exact_true_defect_bounds: tuple[Q, Q]
    true_defect_bounds: I
    exact_observed_defect_bounds: tuple[Q, Q]
    observed_defect_bounds: I
    inverse_capacity_exact_true_defect_bounds: tuple[Q, Q]
    inverse_capacity_true_defect_bounds: I
    inverse_capacity_exact_observed_defect_bounds: tuple[Q, Q] | None
    inverse_capacity_observed_defect_bounds: I | None
    exact_mixed_zero: bool
    arithmetic_exact_zero: bool
    inverse_capacity_exact_zero: bool
    independent_contact_blocks: bool
    retention_margins: tuple[Q, ...] | None
    retention_status: str
    universal: _PreparedTwoContactComposition
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_initial_state_clock_and_held_law_on_four_contact_branches",
        "same_weighted_support_in_both_quarter_sine_rows_no_input_or_event",
        "global_initial_absolute_form_bound_and_arbitrary_common_phase_lift",
        "primitive_weighted_support_and_capacities_rebuild_all_consumed_fields",
        "coincident_marked_contacts_are_separate_additive_parameter_terms",
        "zero_conductance_is_absent_from_both_rows_positive_paths_remain_exact",
        "nodal_absolute_form_plus_phase_sensitivity_is_not_sum_of_maxima",
        "positive_matrix_comparison_retains_contact_derivatives_and_sine_curvature",
        "fixed_arithmetic_and_inverse_capacity_observations_project_separately",
        "four_reading_errors_have_independent_observation_budgets",
        "exact_contact_product_and_endpoint_pairs_precede_dyadic_display",
        "positive_upper_bound_is_inconclusive_about_actual_nonzero_influence",
        "chart_margins_require_declared_initial_edge_lift_association",
        "no_native_bare_phase_transfer_mean_closure_finite_speed_or_physical_claim",
    )


def _support_matrix_vector(matrix, vector):
    """Apply the shared exact product to already admitted complete rows."""
    return tuple(
        row[0] for row in exact_matrix_product(matrix, tuple((v,) for v in vector))
    )


def _prepared_support_contact_composition(
    *,
    nodes,
    old_edges,
    old_conductances,
    capacity,
    first_contact,
    second_contact,
    observation_nodes,
    first_contact_weight_bounds,
    second_contact_weight_bounds,
    gamma_bounds,
    initial_form_upper_bound,
    elapsed_time,
    reading_error_bounds,
    inverse_capacity_reading_error_bounds=None,
    old_edge_chart_margins=None,
) -> _PreparedSupportContactComposition:
    """Bound the finite four-branch mixed contrast using its actual support.

    Callers establish identical complete initial form/phase states and the
    held law x'=D*(-L*x+gamma*S(theta))/4, theta'=gamma*D*L*x/4 across the
    contact rectangle. Every initial absolute form is at most B0. Both rows
    use the admitted weights; capacities are positive, gamma nonnegative,
    and there are no later inputs, events or moving contacts. Scalar data
    do not authenticate the initial state or these dynamical premises.

    At the upper contact corner form A=(1+g)*D*(diag(degree)+adjacency)/4.
    For contact j, qj_i=abs(partial_j x_i)+abs(partial_j theta_i) satisfies
    qj'<=A*qj+f0j+t*f1j. A positive exponential upper matrix H gives
    qj(s)<=s*Uj with Uj=H*(f0j+h*f1j/2). For the mixed derivative, the
    endpoint Jacobians Bj and sine Hessian give
    R=H*(h**2*(B1*U2+B2*U1)/2+h**3*Hess(U1,U2)/3).
    Cap each R_i by the previous universal coefficient, multiply the exact
    upper contact product, then project with the actual two normalized
    observation weight vectors. No sign or equality of those means follows.

    Nodes are 2..24 ordered unique hashable labels. Old support is simple,
    but either marked additive contact may coincide with the other or an
    old edge. Declared zero old weights remain associated while contributing
    to neither dynamics nor positive support. Requested chart margins have
    one entry per old edge, including an empty tuple on empty old support.
    Their caller-associated initial collars are independent of composition
    accuracy. Separate contact blocks imply exact zero; a positive allowance
    is inconclusive and can overestimate cancellations in communicating blocks.

    Before exponential, inverse or interval work admit every primitive and
    both numerical policies: h*norm_infinity(A)<=32 and g*M*K*h/2<=4096.
    They bound implementation work, not the theorem's finite-time scope.
    Exact endpoint pairs retain arbitrarily small path/contact weights;
    displayed intervals can retain a dyadic rounding floor.
    """
    nodes = _ordered(nodes, "nodes", limit=MAX_COMPARISON_DIMENSION + 1)
    size = len(nodes)
    if not 2 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("support requires between 2 and 24 nodes")
    max_edges = size * (size - 1) // 2
    old_edges = _ordered(old_edges, "old_edges", limit=max_edges + 1)
    if len(old_edges) > max_edges:
        raise ValueError("old support exceeds the simple-edge budget")
    old_edges = tuple(_ordered(edge, "old edge", limit=3) for edge in old_edges)
    index_edges, _ = _support_topology(nodes, old_edges, require_connected=False)
    contacts = tuple(
        _ordered(raw, label, limit=3)
        for raw, label in (
            (first_contact, "first_contact"),
            (second_contact, "second_contact"),
        )
    )
    contact_indices = tuple(
        _support_topology(nodes, (contact,), require_connected=False)[0][0]
        for contact in contacts
    )
    observation_nodes = _ordered(observation_nodes, "observation_nodes", limit=size + 1)
    positions = {node: i for i, node in enumerate(nodes)}
    if (
        not 1 <= len(observation_nodes) <= size
        or len(set(observation_nodes)) != len(observation_nodes)
        or any(node not in positions for node in observation_nodes)
    ):
        raise ValueError(
            "observation_nodes must be a nonempty distinct subset of nodes"
        )
    observed = tuple(positions[node] for node in observation_nodes)
    conductances = _prepared_coordinate_vector(
        old_conductances, "old_conductances", len(old_edges)
    )
    capacities = _prepared_coordinate_vector(capacity, "capacity", size)
    first, second, gamma = (
        _prepared_coordinate_vector(raw, label, 2)
        for raw, label in (
            (first_contact_weight_bounds, "first_contact_weight_bounds"),
            (second_contact_weight_bounds, "second_contact_weight_bounds"),
            (gamma_bounds, "gamma_bounds"),
        )
    )
    initial, time = (
        exact_or_represented_real(raw, label)
        for raw, label in (
            (initial_form_upper_bound, "initial_form_upper_bound"),
            (elapsed_time, "elapsed_time"),
        )
    )
    errors = _prepared_coordinate_vector(
        reading_error_bounds, "reading_error_bounds", 4
    )
    weighted_errors = (
        None
        if inverse_capacity_reading_error_bounds is None
        else _prepared_coordinate_vector(
            inverse_capacity_reading_error_bounds,
            "inverse_capacity_reading_error_bounds",
            4,
        )
    )
    margins = (
        None
        if old_edge_chart_margins is None
        else _prepared_coordinate_vector(
            old_edge_chart_margins, "old_edge_chart_margins", len(old_edges)
        )
    )
    if any(value < 0 for value in conductances):
        raise ValueError("old_conductances must be nonnegative")
    if any(value <= 0 for value in capacities):
        raise ValueError("capacity must be strictly positive")
    for bounds, label in (
        (first, "first_contact_weight_bounds"),
        (second, "second_contact_weight_bounds"),
        (gamma, "gamma_bounds"),
    ):
        if bounds[0] < 0 or bounds[0] > bounds[1]:
            raise ValueError(f"{label} requires ordered nonnegative endpoints")
    if min(initial, time, *errors) < 0:
        raise ValueError(
            "initial form bound, horizon and reading errors must be nonnegative"
        )
    if weighted_errors is not None and min(weighted_errors) < 0:
        raise ValueError("inverse_capacity_reading_error_bounds must be nonnegative")
    if margins is not None and any(value < 0 for value in margins):
        raise ValueError("old_edge_chart_margins must be nonnegative")

    adjacency = [[Q(0)] * size for _ in nodes]
    weighted_edges = tuple(zip(index_edges, conductances)) + tuple(
        zip(contact_indices, (first[1], second[1]))
    )
    for (left, right), weight in weighted_edges:
        adjacency[left][right] += weight
        adjacency[right][left] += weight
    adjacency = tuple(tuple(row) for row in adjacency)
    degrees = tuple(sum(row, Q(0)) for row in adjacency)
    maximum, degree, g = max(capacities), max(degrees), gamma[1]
    mobility_degree = maximum * degree
    comparison = tuple(
        tuple(
            (1 + g) * capacities[i] * (value + (degrees[i] if i == j else 0)) / 4
            for j, value in enumerate(row)
        )
        for i, row in enumerate(adjacency)
    )
    if (
        time * max(sum(row, Q(0)) for row in comparison)
        > MAX_NONNEGATIVE_EXPONENTIAL_STEP
    ):
        raise ValueError("support comparison requires h*norm(A)<=32")
    if g * mobility_degree * time / 2 > _MAX_RATIONAL_EXPONENT:
        raise ValueError("support comparison requires g*M*K*h/2<=4096")

    # Only admitted rational derivations precede this shared numerical reader.
    universal = _prepared_two_contact_composition(
        first_contact_weight_bounds=first,
        second_contact_weight_bounds=second,
        gamma_bounds=gamma,
        capacity_bounds=(min(capacities), maximum),
        initial_form_upper_bound=initial,
        weighted_degree_upper_bound=degree,
        elapsed_time=time,
        reading_error_bounds=errors,
        inverse_capacity_reading_error_bounds=weighted_errors,
        old_edge_chart_margins=margins if margins else None,
    )
    kernel = nonnegative_exponential_upper(comparison, time)
    derivative_matrices = tuple(
        tuple(
            tuple(
                (1 + g) * capacities[i] / 4 if i in contact and j in contact else Q(0)
                for j in range(size)
            )
            for i in range(size)
        )
        for contact in contact_indices
    )
    constants = tuple(
        tuple(
            capacities[i] * ((1 + g) * initial / 2 + g / 4) if i in contact else Q(0)
            for i in range(size)
        )
        for contact in contact_indices
    )
    linears = tuple(
        tuple(
            capacities[i] * (1 + g) * mobility_degree * g / 8 if i in contact else Q(0)
            for i in range(size)
        )
        for contact in contact_indices
    )
    slopes = tuple(
        _support_matrix_vector(
            kernel.upper, tuple(a + time * b / 2 for a, b in zip(constant, linear))
        )
        for constant, linear in zip(constants, linears)
    )
    first_contact_forcing = _support_matrix_vector(derivative_matrices[0], slopes[1])
    second_contact_forcing = _support_matrix_vector(derivative_matrices[1], slopes[0])
    contact_forcing = tuple(
        a + b for a, b in zip(first_contact_forcing, second_contact_forcing)
    )
    hessian = tuple(
        g
        * capacities[i]
        / 4
        * sum(
            (
                weight * (slopes[0][i] + slopes[0][j]) * (slopes[1][i] + slopes[1][j])
                for j, weight in enumerate(row)
            ),
            Q(0),
        )
        for i, row in enumerate(adjacency)
    )
    raw_mixed = _support_matrix_vector(
        kernel.upper,
        tuple(
            time**2 * a / 2 + time**3 * b / 3 for a, b in zip(contact_forcing, hessian)
        ),
    )
    mixed = tuple(
        min(value, universal.mixed_sensitivity_upper_bound) for value in raw_mixed
    )
    defects = tuple(
        value * universal.contact_weight_product_upper_bound for value in mixed
    )
    arithmetic_weights = tuple(
        Q(1, len(observed)) if i in observed else Q(0) for i in range(size)
    )
    inverse_mass = sum((1 / capacities[i] for i in observed), Q(0))
    inverse_weights = tuple(
        1 / (capacities[i] * inverse_mass) if i in observed else Q(0)
        for i in range(size)
    )
    arithmetic_bound = sum(
        (weight * value for weight, value in zip(arithmetic_weights, defects)), Q(0)
    )
    inverse_bound = sum(
        (weight * value for weight, value in zip(inverse_weights, defects)), Q(0)
    )
    exact_true = (-arithmetic_bound, arithmetic_bound)
    exact_observed = (
        -arithmetic_bound - universal.reading_error_upper_bound,
        arithmetic_bound + universal.reading_error_upper_bound,
    )
    inverse_true = (-inverse_bound, inverse_bound)
    weighted_error = universal.inverse_capacity_reading_error_upper_bound
    inverse_observed = (
        None
        if weighted_error is None
        else (-inverse_bound - weighted_error, inverse_bound + weighted_error)
    )
    components = _support_components(
        tuple(
            tuple(j for j, weight in enumerate(row) if weight > 0) for row in adjacency
        )
    )
    first_blocks = {
        block
        for block, component in enumerate(components)
        if any(i in component for i in contact_indices[0])
    }
    second_blocks = {
        block
        for block, component in enumerate(components)
        if any(i in component for i in contact_indices[1])
    }
    retention_margins = () if margins == () else universal.retention_margins
    retention = "certified" if margins == () else universal.retention_status
    return _PreparedSupportContactComposition(
        nodes=nodes,
        old_edges=old_edges,
        old_conductances=conductances,
        capacity=capacities,
        first_contact=contacts[0],
        second_contact=contacts[1],
        observation_nodes=observation_nodes,
        old_edge_chart_margins=margins,
        index_old_edges=index_edges,
        index_first_contact=contact_indices[0],
        index_second_contact=contact_indices[1],
        observation_indices=observed,
        joined_adjacency=adjacency,
        weighted_degrees=degrees,
        positive_support_components=components,
        arithmetic_observation_weights=arithmetic_weights,
        inverse_capacity_observation_weights=inverse_weights,
        comparison_matrix=comparison,
        first_contact_derivative_matrix=derivative_matrices[0],
        second_contact_derivative_matrix=derivative_matrices[1],
        first_forcing_constant=constants[0],
        second_forcing_constant=constants[1],
        first_forcing_linear=linears[0],
        second_forcing_linear=linears[1],
        kernel=kernel,
        first_sensitivity_slope_upper_bounds=slopes[0],
        second_sensitivity_slope_upper_bounds=slopes[1],
        mixed_contact_forcing_upper_bounds=contact_forcing,
        mixed_hessian_upper_bounds=hessian,
        raw_nodal_mixed_sensitivity_upper_bounds=raw_mixed,
        nodal_mixed_sensitivity_upper_bounds=mixed,
        nodal_mixed_defect_upper_bounds=defects,
        arithmetic_defect_upper_bound=arithmetic_bound,
        inverse_capacity_defect_upper_bound=inverse_bound,
        exact_true_defect_bounds=exact_true,
        true_defect_bounds=I(*exact_true),
        exact_observed_defect_bounds=exact_observed,
        observed_defect_bounds=I(*exact_observed),
        inverse_capacity_exact_true_defect_bounds=inverse_true,
        inverse_capacity_true_defect_bounds=I(*inverse_true),
        inverse_capacity_exact_observed_defect_bounds=inverse_observed,
        inverse_capacity_observed_defect_bounds=(
            None if inverse_observed is None else I(*inverse_observed)
        ),
        exact_mixed_zero=not any(defects),
        arithmetic_exact_zero=arithmetic_bound == 0,
        inverse_capacity_exact_zero=inverse_bound == 0,
        independent_contact_blocks=first_blocks.isdisjoint(second_blocks),
        retention_margins=retention_margins,
        retention_status=retention,
        universal=universal,
    )
