"""A smaller held-boundary sine predictor with separated finite error budgets.

The full law keeps every positive physical capacity. Only the supplied
approximation freezes hidden reference coordinates; its evolving state has
two coordinates per retained node. No trajectory or contact event is run.
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
from ..mathematics._rational_interval import I
from ._sine_flow import _held_quarter_sine_field
from ._sine_preparation import _prepared_coordinate_vector
from .relational_observations import _ordered
from .relational_sine_support import _support_topology


@dataclass(frozen=True)
class _RetainedBoundaryPreparation:
    nodes: tuple
    edges: tuple[tuple, ...]
    conductances: tuple[Q, ...]
    capacity: tuple[Q, ...]
    gamma: Q
    retained_nodes: tuple
    observation_nodes: tuple
    retained_initial_form: tuple[Q, ...]
    retained_initial_phase: tuple[Q, ...]
    hidden_reference_form: tuple[Q, ...]
    hidden_reference_phase: tuple[Q, ...]
    hidden_nodes: tuple
    index_edges: tuple[tuple[int, int], ...]
    retained_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    observation_indices: tuple[int, ...]
    reference_form: tuple[Q, ...]
    reference_phase: tuple[Q, ...]
    reduced_initial_state: tuple[Q, ...]
    adjacency: tuple[tuple[Q, ...], ...]
    weighted_degrees: tuple[Q, ...]
    comparison_matrix: tuple[tuple[Q, ...], ...]
    arithmetic_observation_weights: tuple[Q, ...]
    inverse_capacity_observation_weights: tuple[Q, ...]

    def make_reduced_field(self):
        """Re-admit stored primitives and compile only the retained rows.

        No cached indices, matrices, rates or verdicts are consumed. The
        resulting callable snapshots immutable coefficients and constants;
        it accepts retained forms followed by retained phase lifts.
        """
        fresh = _admit_retained_boundary(
            nodes=self.nodes,
            edges=self.edges,
            conductances=self.conductances,
            capacity=self.capacity,
            gamma=self.gamma,
            retained_nodes=self.retained_nodes,
            observation_nodes=self.observation_nodes,
            retained_initial_form=self.retained_initial_form,
            retained_initial_phase=self.retained_initial_phase,
            hidden_reference_form=self.hidden_reference_form,
            hidden_reference_phase=self.hidden_reference_phase,
        )
        return _field_from_preparation(fresh)


@dataclass(frozen=True)
class _PreparedSineRetainedBoundary(_RetainedBoundaryPreparation):
    """Conditional exact reduced-dynamics error, without a numerical solution.

    All nodal vectors follow full node order. Hidden reference/radius vectors
    follow the full-order complement of the declared retained node sequence.
    Initial rate bounds enclose the actual two rows before absolute values,
    retaining cancellation. Numerical enclosures can overestimate a zero.

    Reading error is an independently supplied total allowance for comparing
    the chosen full and reduced observations. This theorem compares exact
    flows and includes no implicit numerical solver error. Future computed
    readings require their own justified error within that allowance. True
    zero flags certify zero under the premises; positive bounds and a failed
    tolerance certificate are inconclusive about actual prediction failure.
    """

    hidden_form_error_bounds: tuple[Q, ...]
    hidden_phase_error_bounds: tuple[Q, ...]
    elapsed_time: Q
    reading_error_bound: Q
    tolerance: Q
    inverse_capacity_reading_error_bound: Q | None
    inverse_capacity_tolerance: Q | None
    kernel: NonnegativeExponentialUpper
    retained_kernel: NonnegativeExponentialUpper
    initial_form_rate_bounds: tuple[I, ...]
    initial_phase_rate_bounds: tuple[I, ...]
    initial_rate_upper_bounds: tuple[Q, ...]
    retained_movement_rate_upper_bounds: tuple[Q, ...]
    initial_discrepancy_upper_bounds: tuple[Q, ...]
    hidden_drift_forcing: tuple[Q, ...]
    hidden_feedback_forcing: tuple[Q, ...]
    initialization_error_upper_bounds: tuple[Q, ...]
    hidden_drift_error_upper_bounds: tuple[Q, ...]
    boundary_feedback_error_upper_bounds: tuple[Q, ...]
    nodal_error_upper_bounds: tuple[Q, ...]
    arithmetic_initialization_error_upper_bound: Q
    arithmetic_hidden_drift_error_upper_bound: Q
    arithmetic_boundary_feedback_error_upper_bound: Q
    arithmetic_error_upper_bound: Q
    arithmetic_observed_error_upper_bound: Q
    tolerance_certified: bool
    inverse_capacity_initialization_error_upper_bound: Q
    inverse_capacity_hidden_drift_error_upper_bound: Q
    inverse_capacity_boundary_feedback_error_upper_bound: Q
    inverse_capacity_error_upper_bound: Q
    inverse_capacity_observed_error_upper_bound: Q | None
    inverse_capacity_tolerance_certified: bool | None
    exact_observation_zero: bool
    inverse_capacity_exact_observation_zero: bool
    scope: tuple[str, ...] = (
        "held_positive_capacity_weighted_quarter_sine_law_in_both_full_rows",
        "no_later_input_event_contact_motion_or_coefficient_change",
        "proper_retained_region_contains_the_unchanged_observation",
        "retained_initial_state_matches_full_state_hidden_errors_are_independent",
        "hidden_phase_errors_are_absolute_continuous_lift_errors_not_wrapped_distances",
        "reduced_field_evolves_only_retained_forms_and_phases_with_fixed_hidden_references",
        "full_support_is_static_preparation_information_not_additional_evolved_state",
        "omitted_initialization_hidden_drift_and_retained_feedback_remain_separate",
        "initial_rate_enclosures_preserve_full_row_cancellation_before_absolute_values",
        "arithmetic_and_inverse_capacity_observations_use_their_actual_normalized_weights",
        "observation_errors_and_tolerances_are_independently_supplied_before_response",
        "no_implicit_solver_error_chart_retention_mean_closure_or_physical_identification",
        "contact_response_error_sums_branch_budgets_without_assumed_cancellation",
    )


def _admit_retained_boundary(
    *,
    nodes,
    edges,
    conductances,
    capacity,
    gamma,
    retained_nodes,
    observation_nodes,
    retained_initial_form,
    retained_initial_phase,
    hidden_reference_form,
    hidden_reference_phase,
):
    """Normalize primitive associations without sine or exponential work."""
    nodes = _ordered(nodes, "nodes", limit=MAX_COMPARISON_DIMENSION + 1)
    size = len(nodes)
    if not 2 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("retained boundary support requires between 2 and 24 nodes")
    max_edges = size * (size - 1) // 2
    edges = _ordered(edges, "edges", limit=max_edges + 1)
    if len(edges) > max_edges:
        raise ValueError("support exceeds its simple-edge budget")
    edges = tuple(_ordered(edge, "edge", limit=3) for edge in edges)
    index_edges, _ = _support_topology(nodes, edges, require_connected=False)
    conductances = _prepared_coordinate_vector(conductances, "conductances", len(edges))
    capacity = _prepared_coordinate_vector(capacity, "capacity", size)
    gamma = exact_or_represented_real(gamma, "gamma")
    if gamma < 0 or any(weight < 0 for weight in conductances):
        raise ValueError("gamma and conductances must be nonnegative")
    if any(value <= 0 for value in capacity):
        raise ValueError("full physical capacities must be strictly positive")
    positions = {node: i for i, node in enumerate(nodes)}
    retained_nodes = _ordered(retained_nodes, "retained_nodes", limit=size + 1)
    if (
        not 1 <= len(retained_nodes) < size
        or len(set(retained_nodes)) != len(retained_nodes)
        or any(node not in positions for node in retained_nodes)
    ):
        raise ValueError("retained_nodes must be a nonempty proper distinct subset")
    retained = tuple(positions[node] for node in retained_nodes)
    hidden = tuple(i for i in range(size) if i not in retained)
    observation_nodes = _ordered(observation_nodes, "observation_nodes", limit=size + 1)
    if (
        not observation_nodes
        or len(observation_nodes) > len(retained)
        or len(set(observation_nodes)) != len(observation_nodes)
        or any(node not in retained_nodes for node in observation_nodes)
    ):
        raise ValueError(
            "observation_nodes must be a nonempty distinct subset of retained_nodes"
        )
    observed = tuple(positions[node] for node in observation_nodes)
    retained_form, retained_phase, hidden_form, hidden_phase = (
        _prepared_coordinate_vector(raw, label, length)
        for raw, label, length in (
            (retained_initial_form, "retained_initial_form", len(retained)),
            (retained_initial_phase, "retained_initial_phase", len(retained)),
            (hidden_reference_form, "hidden_reference_form", len(hidden)),
            (hidden_reference_phase, "hidden_reference_phase", len(hidden)),
        )
    )
    reference_form, reference_phase = [Q(0)] * size, [Q(0)] * size
    for indices, forms, phases in (
        (retained, retained_form, retained_phase),
        (hidden, hidden_form, hidden_phase),
    ):
        for i, form, phase in zip(indices, forms, phases):
            reference_form[i], reference_phase[i] = form, phase
    adjacency = [[Q(0)] * size for _ in nodes]
    for (i, j), weight in zip(index_edges, conductances):
        adjacency[i][j] = adjacency[j][i] = weight
    adjacency = tuple(tuple(row) for row in adjacency)
    degrees = tuple(sum(row, Q(0)) for row in adjacency)
    comparison = tuple(
        tuple(
            (1 + gamma) * capacity[i] * (value + (degrees[i] if i == j else 0)) / 4
            for j, value in enumerate(row)
        )
        for i, row in enumerate(adjacency)
    )
    arithmetic_weights = tuple(
        Q(1, len(observed)) if i in observed else Q(0) for i in range(size)
    )
    inverse_mass = sum((1 / capacity[i] for i in observed), Q(0))
    inverse_weights = tuple(
        1 / (capacity[i] * inverse_mass) if i in observed else Q(0) for i in range(size)
    )
    return _RetainedBoundaryPreparation(
        nodes=nodes,
        edges=edges,
        conductances=conductances,
        capacity=capacity,
        gamma=gamma,
        retained_nodes=retained_nodes,
        observation_nodes=observation_nodes,
        retained_initial_form=retained_form,
        retained_initial_phase=retained_phase,
        hidden_reference_form=hidden_form,
        hidden_reference_phase=hidden_phase,
        hidden_nodes=tuple(nodes[i] for i in hidden),
        index_edges=index_edges,
        retained_indices=retained,
        hidden_indices=hidden,
        observation_indices=observed,
        reference_form=tuple(reference_form),
        reference_phase=tuple(reference_phase),
        reduced_initial_state=retained_form + retained_phase,
        adjacency=adjacency,
        weighted_degrees=degrees,
        comparison_matrix=comparison,
        arithmetic_observation_weights=arithmetic_weights,
        inverse_capacity_observation_weights=inverse_weights,
    )


def _field_from_preparation(preparation, *, full=False):
    """Compile from freshly admitted primitive rows, never an incoming report."""
    return _held_quarter_sine_field(
        capacity=preparation.capacity,
        gamma=preparation.gamma,
        weighted_edges=tuple(
            (i, j, weight)
            for (i, j), weight in zip(preparation.index_edges, preparation.conductances)
        ),
        retained_indices=None if full else preparation.retained_indices,
        hidden_form=() if full else preparation.hidden_reference_form,
        hidden_phase=() if full else preparation.hidden_reference_phase,
    )


def _boundary_matrix_vector(matrix, vector):
    return tuple(
        row[0]
        for row in exact_matrix_product(matrix, tuple((value,) for value in vector))
    )


def _prepared_sine_retained_boundary(
    *,
    nodes,
    edges,
    conductances,
    capacity,
    gamma,
    retained_nodes,
    observation_nodes,
    retained_initial_form,
    retained_initial_phase,
    hidden_reference_form,
    hidden_reference_phase,
    hidden_form_error_bounds,
    hidden_phase_error_bounds,
    elapsed_time,
    reading_error_bound,
    tolerance,
    inverse_capacity_reading_error_bound=None,
    inverse_capacity_tolerance=None,
) -> _PreparedSineRetainedBoundary:
    """Certify a smaller clamped-boundary predictor for one fixed support.

    The complete law is x'=D*(-Lx+gamma*S(theta))/4 and theta'=gamma*D*Lx/4
    on fixed symmetric nonnegative conductances and positive held capacities.
    Both full and reduced preparations agree exactly on retained coordinates.
    Actual hidden coordinates differ from their fixed reference by separately
    supplied absolute form and continuous-phase-lift radii. They may have any
    signs or correlations within those bounds. The observed constituent must
    be wholly retained; no substitution of a port reading is performed.

    For A=(1+gamma)*D*(diag(degree)+W)/4, let P>=exp(hA) and
    K>=exp(h*A_RR). Full initial row enclosures give c_i>=|Fx_i|+|Ftheta_i|.
    Retained movement from its initial value is at most t*K*c_R, so the
    omitted full hidden rows are at most c_H+t*A_HR*K*c_R. The embedded
    predictor error is bounded by the separately reported vectors
    P*q0, h*P*embed_H(c_H), and h**2*P*embed_H(A_HR*K*c_R)/2.
    Their two normalized form projections and each requested observation
    uncertainty are compared with independently supplied tolerances.

    All primitives and h*norm_infinity(A)<=32 precede sine and exponential
    work. The same cap admits A_RR automatically. No legacy scalar growth
    certificate, phase chart, full response, solver or cached report is used.
    For a matched on/off response, sum the two branch allowances and reading
    errors; the theorem assumes no cancellation of approximation errors.
    """
    p = _admit_retained_boundary(
        nodes=nodes,
        edges=edges,
        conductances=conductances,
        capacity=capacity,
        gamma=gamma,
        retained_nodes=retained_nodes,
        observation_nodes=observation_nodes,
        retained_initial_form=retained_initial_form,
        retained_initial_phase=retained_initial_phase,
        hidden_reference_form=hidden_reference_form,
        hidden_reference_phase=hidden_reference_phase,
    )
    rx, rt = (
        _prepared_coordinate_vector(raw, label, len(p.hidden_indices))
        for raw, label in (
            (hidden_form_error_bounds, "hidden_form_error_bounds"),
            (hidden_phase_error_bounds, "hidden_phase_error_bounds"),
        )
    )
    time, error, tolerance = (
        exact_or_represented_real(raw, label)
        for raw, label in (
            (elapsed_time, "elapsed_time"),
            (reading_error_bound, "reading_error_bound"),
            (tolerance, "tolerance"),
        )
    )
    weighted_error = (
        None
        if inverse_capacity_reading_error_bound is None
        else exact_or_represented_real(
            inverse_capacity_reading_error_bound, "inverse_capacity_reading_error_bound"
        )
    )
    weighted_tolerance = (
        None
        if inverse_capacity_tolerance is None
        else exact_or_represented_real(
            inverse_capacity_tolerance, "inverse_capacity_tolerance"
        )
    )
    if min(time, error, tolerance, *rx, *rt) < 0:
        raise ValueError(
            "horizon, coordinate radii, reading error and tolerance must be nonnegative"
        )
    if any(
        value is not None and value < 0
        for value in (weighted_error, weighted_tolerance)
    ):
        raise ValueError("inverse-capacity error and tolerance must be nonnegative")
    if (
        time * max(sum(row, Q(0)) for row in p.comparison_matrix)
        > MAX_NONNEGATIVE_EXPONENTIAL_STEP
    ):
        raise ValueError("retained boundary comparison requires h*norm(A)<=32")

    size, retained, hidden = len(p.nodes), p.retained_indices, p.hidden_indices
    initial_rates = _field_from_preparation(p, full=True)(
        p.reference_form + p.reference_phase
    )
    form_rates, phase_rates = initial_rates[:size], initial_rates[size:]
    rates = tuple(
        x.abs_max + theta.abs_max for x, theta in zip(form_rates, phase_rates)
    )
    kernel = nonnegative_exponential_upper(p.comparison_matrix, time)
    retained_matrix = tuple(
        tuple(p.comparison_matrix[i][j] for j in retained) for i in retained
    )
    retained_kernel = nonnegative_exponential_upper(retained_matrix, time)
    movement = _boundary_matrix_vector(
        retained_kernel.upper, tuple(rates[i] for i in retained)
    )
    discrepancy = [Q(0)] * size
    for i, form_error, phase_error in zip(hidden, rx, rt):
        discrepancy[i] = form_error + phase_error
    discrepancy = tuple(discrepancy)
    drift = tuple(rates[i] if i in hidden else Q(0) for i in range(size))
    feedback = tuple(
        (
            sum(
                (
                    p.comparison_matrix[i][j] * value
                    for j, value in zip(retained, movement)
                ),
                Q(0),
            )
            if i in hidden
            else Q(0)
        )
        for i in range(size)
    )
    initialization_error = _boundary_matrix_vector(kernel.upper, discrepancy)
    drift_error = _boundary_matrix_vector(
        kernel.upper, tuple(time * value for value in drift)
    )
    feedback_error = _boundary_matrix_vector(
        kernel.upper, tuple(time**2 * value / 2 for value in feedback)
    )
    nodal = tuple(
        a + b + c for a, b, c in zip(initialization_error, drift_error, feedback_error)
    )

    def project(weights):
        return tuple(
            sum((weight * value for weight, value in zip(weights, vector)), Q(0))
            for vector in (initialization_error, drift_error, feedback_error)
        )

    arithmetic = project(p.arithmetic_observation_weights)
    inverse = project(p.inverse_capacity_observation_weights)
    arithmetic_bound, inverse_bound = sum(arithmetic, Q(0)), sum(inverse, Q(0))
    observed_error = arithmetic_bound + error
    inverse_observed = (
        None if weighted_error is None else inverse_bound + weighted_error
    )
    return _PreparedSineRetainedBoundary(
        **vars(p),
        hidden_form_error_bounds=rx,
        hidden_phase_error_bounds=rt,
        elapsed_time=time,
        reading_error_bound=error,
        tolerance=tolerance,
        inverse_capacity_reading_error_bound=weighted_error,
        inverse_capacity_tolerance=weighted_tolerance,
        kernel=kernel,
        retained_kernel=retained_kernel,
        initial_form_rate_bounds=form_rates,
        initial_phase_rate_bounds=phase_rates,
        initial_rate_upper_bounds=rates,
        retained_movement_rate_upper_bounds=movement,
        initial_discrepancy_upper_bounds=discrepancy,
        hidden_drift_forcing=drift,
        hidden_feedback_forcing=feedback,
        initialization_error_upper_bounds=initialization_error,
        hidden_drift_error_upper_bounds=drift_error,
        boundary_feedback_error_upper_bounds=feedback_error,
        nodal_error_upper_bounds=nodal,
        arithmetic_initialization_error_upper_bound=arithmetic[0],
        arithmetic_hidden_drift_error_upper_bound=arithmetic[1],
        arithmetic_boundary_feedback_error_upper_bound=arithmetic[2],
        arithmetic_error_upper_bound=arithmetic_bound,
        arithmetic_observed_error_upper_bound=observed_error,
        tolerance_certified=observed_error <= tolerance,
        inverse_capacity_initialization_error_upper_bound=inverse[0],
        inverse_capacity_hidden_drift_error_upper_bound=inverse[1],
        inverse_capacity_boundary_feedback_error_upper_bound=inverse[2],
        inverse_capacity_error_upper_bound=inverse_bound,
        inverse_capacity_observed_error_upper_bound=inverse_observed,
        inverse_capacity_tolerance_certified=(
            None
            if inverse_observed is None or weighted_tolerance is None
            else inverse_observed <= weighted_tolerance
        ),
        exact_observation_zero=arithmetic_bound == 0,
        inverse_capacity_exact_observation_zero=inverse_bound == 0,
    )
