"""Shared primitive coordinate information for held quarter-law contacts.

Coordinate meaning and complete dynamic claims belong to each consumer.
This preparation admits support, associations and independent initial boxes;
it evaluates no field, exponential, future response or measurement source.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import MAX_COMPARISON_DIMENSION
from ._sine_preparation import _prepared_coordinate_vector
from .relational_observations import _ordered
from .relational_sine_support import _support_topology


@dataclass(frozen=True)
class _ContactCoordinateInformation:
    nodes: tuple
    old_edges: tuple[tuple, ...]
    old_conductances: tuple[Q, ...]
    capacity: tuple[Q, ...]
    contact: tuple
    contact_weight: Q
    retained_nodes: tuple
    observation_nodes: tuple
    retained_coordinates: tuple[Q, ...]
    hidden_prior_bounds: tuple[tuple[Q, Q], ...]
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
    hidden_posterior_bounds: tuple[tuple[Q, Q], ...]
    posterior_center: tuple[Q, ...]
    posterior_radii: tuple[Q, ...]


@dataclass(frozen=True)
class _ContactSupportInformation:
    """Primitive matched contact support without invented state coordinates."""

    nodes: tuple
    old_edges: tuple[tuple, ...]
    old_conductances: tuple[Q, ...]
    capacity: tuple[Q, ...]
    contact: tuple
    contact_weight: Q
    retained_nodes: tuple
    observation_nodes: tuple
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
    off_generator: tuple[tuple[Q, ...], ...]
    on_generator: tuple[tuple[Q, ...], ...]


def _admit_contact_support_information(
    *,
    nodes,
    old_edges,
    old_conductances,
    capacity,
    contact,
    contact_weight,
    retained_nodes,
    observation_nodes,
    elapsed_time,
    mean_kind,
    reading_error_bound,
    tolerance,
):
    """Admit support, receiving capacities and original normalized observation.

    The exact matrices are G=-D*L/4. No initial information, phase chart,
    complete law, exponential domain or acquisition evidence is inferred.
    """
    nodes = _ordered(nodes, "nodes", limit=MAX_COMPARISON_DIMENSION + 1)
    size = len(nodes)
    if not 2 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("response support requires between 2 and 24 nodes")
    max_edges = size * (size - 1) // 2
    old_edges = _ordered(old_edges, "old_edges", limit=max_edges + 1)
    if len(old_edges) > max_edges:
        raise ValueError("old support exceeds its simple-edge budget")
    old_edges = tuple(_ordered(edge, "old edge", limit=3) for edge in old_edges)
    index_edges, _ = _support_topology(nodes, old_edges, require_connected=False)
    contact = _ordered(contact, "contact", limit=3)
    index_contact = _support_topology(nodes, (contact,), require_connected=False)[0][0]
    conductances = _prepared_coordinate_vector(
        old_conductances, "old_conductances", len(old_edges)
    )
    capacities = _prepared_coordinate_vector(capacity, "capacity", size)
    weight, time, reading_error, target = (
        exact_or_represented_real(raw, label)
        for raw, label in (
            (contact_weight, "contact_weight"),
            (elapsed_time, "elapsed_time"),
            (reading_error_bound, "reading_error_bound"),
            (tolerance, "tolerance"),
        )
    )
    if (
        any(value < 0 for value in conductances)
        or min(weight, time, reading_error, target) < 0
    ):
        raise ValueError(
            "conductances, time, reading error and tolerance must be nonnegative"
        )
    if any(value <= 0 for value in capacities):
        raise ValueError("physical capacities must be strictly positive")
    if type(mean_kind) is not str or mean_kind not in (
        "arithmetic",
        "inverse_capacity",
    ):
        raise ValueError("mean_kind must be arithmetic or inverse_capacity")

    positions = {node: i for i, node in enumerate(nodes)}
    retained_nodes = _ordered(retained_nodes, "retained_nodes", limit=size + 1)
    if (
        not 1 <= len(retained_nodes) < size
        or len(set(retained_nodes)) != len(retained_nodes)
        or any(node not in positions for node in retained_nodes)
        or any(node not in retained_nodes for node in contact)
    ):
        raise ValueError(
            "retained_nodes must be proper, distinct and contain the contact"
        )
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
    adjacency = [[Q(0)] * size for _ in nodes]
    for (i, j), conductance in zip(index_edges, conductances):
        adjacency[i][j] = adjacency[j][i] = conductance

    def generator(rows):
        degrees = tuple(sum(row, Q(0)) for row in rows)
        return tuple(
            tuple(
                capacities[i] * (entry - (degrees[i] if i == j else 0)) / 4
                for j, entry in enumerate(row)
            )
            for i, row in enumerate(rows)
        )

    off = generator(adjacency)
    i, j = index_contact
    adjacency[i][j] += weight
    adjacency[j][i] += weight
    on = generator(adjacency)
    raw_weights = tuple(
        (
            (Q(1) if mean_kind == "arithmetic" else 1 / capacities[i])
            if i in observed
            else Q(0)
        )
        for i in range(size)
    )
    mass = sum(raw_weights, Q(0))
    observation_weights = tuple(value / mass for value in raw_weights)
    return _ContactSupportInformation(
        nodes=nodes,
        old_edges=old_edges,
        old_conductances=conductances,
        capacity=capacities,
        contact=contact,
        contact_weight=weight,
        retained_nodes=retained_nodes,
        observation_nodes=observation_nodes,
        elapsed_time=time,
        mean_kind=mean_kind,
        reading_error_bound=reading_error,
        tolerance=target,
        hidden_nodes=tuple(nodes[i] for i in hidden),
        index_old_edges=index_edges,
        index_contact=index_contact,
        retained_indices=retained,
        hidden_indices=hidden,
        observation_indices=observed,
        observation_weights=observation_weights,
        off_generator=off,
        on_generator=on,
    )


def _admit_contact_coordinate_information(
    *,
    nodes,
    old_edges,
    old_conductances,
    capacity,
    contact,
    contact_weight,
    retained_nodes,
    observation_nodes,
    retained_coordinates,
    hidden_prior_bounds,
    measured_nodes,
    measurement_values,
    measurement_error_bounds,
    elapsed_time,
    mean_kind,
    reading_error_bound,
    tolerance,
):
    """Admit actual support and independent coordinate information, without work.

    The exact matrices are G=-D*L/4 on the off and additive-contact supports.
    Consumers establish the coordinate interpretation, any chart, complete law
    and numerical domains before using them. This helper does not require an
    invented form or phase state, and incoming cached reports are not accepted.
    """
    support = _admit_contact_support_information(
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
    nodes, size = support.nodes, len(support.nodes)
    positions = {node: i for i, node in enumerate(nodes)}
    retained, hidden = support.retained_indices, support.hidden_indices
    retained_values = _prepared_coordinate_vector(
        retained_coordinates, "retained_coordinates", len(retained)
    )
    prior = _ordered(hidden_prior_bounds, "hidden_prior_bounds", limit=len(hidden) + 1)
    if len(prior) != len(hidden):
        raise ValueError(
            "hidden prior bounds must follow the full-order hidden complement"
        )
    prior = tuple(
        _prepared_coordinate_vector(pair, f"hidden_prior_bounds[{i}]", 2)
        for i, pair in enumerate(prior)
    )
    if any(lower > upper for lower, upper in prior):
        raise ValueError("hidden prior bounds must be ordered")
    measured_nodes = _ordered(measured_nodes, "measured_nodes", limit=len(hidden) + 1)
    if (
        len(measured_nodes) > len(hidden)
        or len(set(measured_nodes)) != len(measured_nodes)
        or any(
            node not in positions or positions[node] not in hidden
            for node in measured_nodes
        )
    ):
        raise ValueError("measured_nodes must be a distinct subset of hidden nodes")
    measured = tuple(positions[node] for node in measured_nodes)
    values = _prepared_coordinate_vector(
        measurement_values, "measurement_values", len(measured)
    )
    errors = _prepared_coordinate_vector(
        measurement_error_bounds, "measurement_error_bounds", len(measured)
    )
    if any(error < 0 for error in errors):
        raise ValueError("initial measurement error bounds must be nonnegative")
    posterior = list(prior)
    hidden_positions = {index: j for j, index in enumerate(hidden)}
    for index, value, error in zip(measured, values, errors):
        j = hidden_positions[index]
        lower, upper = posterior[j]
        posterior[j] = (max(lower, value - error), min(upper, value + error))
        if posterior[j][0] > posterior[j][1]:
            raise ValueError(
                "initial measurement contradicts its hidden prior interval"
            )
    posterior = tuple(posterior)

    centers, radii = [Q(0)] * size, [Q(0)] * size
    for index, value in zip(retained, retained_values):
        centers[index] = value
    for index, (lower, upper) in zip(hidden, posterior):
        centers[index], radii[index] = (lower + upper) / 2, (upper - lower) / 2
    centers, radii = tuple(centers), tuple(radii)
    return _ContactCoordinateInformation(
        nodes=support.nodes,
        old_edges=support.old_edges,
        old_conductances=support.old_conductances,
        capacity=support.capacity,
        contact=support.contact,
        contact_weight=support.contact_weight,
        retained_nodes=support.retained_nodes,
        observation_nodes=support.observation_nodes,
        retained_coordinates=retained_values,
        hidden_prior_bounds=prior,
        measured_nodes=measured_nodes,
        measurement_values=values,
        measurement_error_bounds=errors,
        elapsed_time=support.elapsed_time,
        mean_kind=support.mean_kind,
        reading_error_bound=support.reading_error_bound,
        tolerance=support.tolerance,
        hidden_nodes=support.hidden_nodes,
        index_old_edges=support.index_old_edges,
        index_contact=support.index_contact,
        retained_indices=retained,
        hidden_indices=hidden,
        observation_indices=support.observation_indices,
        measured_indices=measured,
        observation_weights=support.observation_weights,
        off_generator=support.off_generator,
        on_generator=support.on_generator,
        hidden_posterior_bounds=posterior,
        posterior_center=centers,
        posterior_radii=radii,
    )


@dataclass(frozen=True)
class _CoordinateForecast:
    """Box-linear information and a separately supplied uniform model error."""

    linear_information_error_bounds: tuple[Q, Q]
    information_error_lower_bound: Q
    computational_center_error_upper_bound: Q
    predicted_response: Q
    forecast_error_upper_bound: Q
    forecast_response_bounds: tuple[Q, Q]
    status: str
    response_sign: str


def _prepared_coordinate_forecast(
    *,
    coefficient_bounds,
    center,
    radii,
    model_error_bound,
    reading_error_bound,
    tolerance,
):
    """Apply interval coefficient arithmetic after full primitive admission.

    Coefficients enclose one fixed linear functional on the admitted box.
    A separately proved uniform response difference at most model_error_bound
    transfers its minimax lower bound and the midpoint's upper allowance.
    This establishes no model error or exact nonlinear corner extrema itself.
    """
    if not (len(coefficient_bounds) == len(center) == len(radii)):
        raise ValueError("coefficient bounds, center and radii must have equal lengths")
    intrinsic_lower = intrinsic_upper = computational = forecast = Q(0)
    for (lower, upper), value, radius in zip(coefficient_bounds, center, radii):
        if lower > upper or radius < 0:
            raise ValueError("prepared coefficient intervals and radii must be ordered")
        minimum = Q(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
        intrinsic_lower += minimum * radius
        intrinsic_upper += max(abs(lower), abs(upper)) * radius
        computational += (upper - lower) * abs(value) / 2
        forecast += (lower + upper) * value / 2
    if min(model_error_bound, reading_error_bound, tolerance) < 0:
        raise ValueError("prepared error bounds and tolerance must be nonnegative")
    unavoidable = max(Q(0), intrinsic_lower - model_error_bound)
    allowance = (
        intrinsic_upper + model_error_bound + computational + reading_error_bound
    )
    forecast_bounds = (forecast - allowance, forecast + allowance)
    status = (
        "insufficient_information"
        if unavoidable > tolerance
        else "certified" if allowance <= tolerance else "unresolved"
    )
    sign = (
        "positive"
        if forecast_bounds[0] > 0
        else (
            "negative"
            if forecast_bounds[1] < 0
            else "zero" if forecast_bounds == (0, 0) else "unresolved"
        )
    )
    return _CoordinateForecast(
        (intrinsic_lower, intrinsic_upper),
        unavoidable,
        computational,
        forecast,
        allowance,
        forecast_bounds,
        status,
        sign,
    )


def _signed_heat_row_bounds(row, kernel):
    """Enclose an exact signed row times a freshly prepared heat kernel."""
    size = len(row)
    return tuple(
        (
            sum(
                (
                    value * (kernel.lower[i][j] if value >= 0 else kernel.upper[i][j])
                    for i, value in enumerate(row)
                ),
                Q(0),
            ),
            sum(
                (
                    value * (kernel.upper[i][j] if value >= 0 else kernel.lower[i][j])
                    for i, value in enumerate(row)
                ),
                Q(0),
            ),
        )
        for j in range(size)
    )


def _interval_row_exact_columns(row, columns):
    """Contract a prepared interval row with exact signed column vectors.

    Keep all supplied column associations until this projection. No interval
    endpoint is a new independent physical preparation parameter.
    """
    return tuple(
        (
            sum(
                (
                    value * (lower if value >= 0 else upper)
                    for (lower, upper), value in zip(row, column)
                ),
                Q(0),
            ),
            sum(
                (
                    value * (upper if value >= 0 else lower)
                    for (lower, upper), value in zip(row, column)
                ),
                Q(0),
            ),
        )
        for column in zip(*columns)
    )


def _interval_row_heat_bounds(row, kernel):
    """Enclose a prepared interval row times a nonnegative heat kernel.

    These intervals represent computational enclosures. Multiplication does
    not assert independence of their endpoints or discard source parameters.
    """
    size = len(row)
    return tuple(
        (
            sum(
                (
                    lower * (kernel.lower[i][j] if lower >= 0 else kernel.upper[i][j])
                    for i, (lower, _) in enumerate(row)
                ),
                Q(0),
            ),
            sum(
                (
                    upper * (kernel.upper[i][j] if upper >= 0 else kernel.lower[i][j])
                    for i, (_, upper) in enumerate(row)
                ),
                Q(0),
            ),
        )
        for j in range(size)
    )
