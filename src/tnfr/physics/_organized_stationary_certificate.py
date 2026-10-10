"""Source-free root inclusion for a fixed, connected unit sine support.

A rationalized numerical inverse is only a proposed preconditioner. Exact
residual and contraction bounds admit it independently. The caller owns its
search protocol, marked winding branches, complete temporal law and source.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import I, INTERVAL_METHOD
from ._repeated_organized_composition import _bounded_sine_edge_fields
from ._sine_preparation import _prepared_coordinate_vector
from .relational_observations import _ordered

_MAX_NODES = 54
_LOWER_WEIGHT_BITS = 32


def _admit_support(edges, size):
    raw = _ordered(edges, "unit edges", limit=size * (size - 1) // 2 + 1)
    if len(raw) > size * (size - 1) // 2:
        raise ValueError("unit support exceeds the simple graph edge budget")
    normalized, seen = [], set()
    neighbors = [set() for _ in range(size)]
    for raw_edge in raw:
        edge = _ordered(raw_edge, "unit edge", limit=3)
        if len(edge) != 2:
            raise ValueError("unit edges require exactly two endpoints")
        if any(type(node) is not int for node in edge):
            raise TypeError("edge endpoints must be exact integers, not booleans")
        left, right = edge
        if not (0 <= left < size and 0 <= right < size) or left == right:
            raise ValueError("unit edge endpoints must be distinct registered nodes")
        identity = tuple(sorted(edge))
        if identity in seen:
            raise ValueError("unit support cannot repeat an undirected edge")
        seen.add(identity)
        normalized.append(edge)
        neighbors[left].add(right)
        neighbors[right].add(left)
    reached, pending = {0}, [0]
    while pending:
        for neighbor in neighbors[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if len(reached) != size:
        raise ValueError("unit support must be connected")
    return tuple(normalized), max(map(len, neighbors))


def _grounded_laplacian(edges, weights, size, zero):
    """Assemble the phase Hessian with the first-node gauge removed."""
    rows = [[zero for _ in range(size - 1)] for _ in range(size - 1)]
    for (left, right), weight in zip(edges, weights):
        if left:
            rows[left - 1][left - 1] += weight
        if right:
            rows[right - 1][right - 1] += weight
        if left and right:
            rows[left - 1][right - 1] -= weight
            rows[right - 1][left - 1] -= weight
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class _OrganizedStationaryCertificate:
    phase_center: tuple[Q, ...]
    edges: tuple[tuple[int, int], ...]
    radius: Q
    inverse: tuple[tuple[Q, ...], ...]
    maximum_degree: int
    phase_current_bounds: tuple[I, ...]
    edge_hessian_bounds: tuple[I, ...]
    grounded_hessian_bounds: tuple[tuple[I, ...], ...]
    midpoint_grounded_hessian: tuple[tuple[Q, ...], ...]
    lower_edge_hessian: tuple[Q, ...]
    lower_grounded_hessian: tuple[tuple[Q, ...], ...]
    inverse_norm: Q
    eta: Q
    q0: Q
    qbox: Q
    image_radius_bound: Q
    contraction_certified: bool
    root_inclusion_certified: bool
    relative_hessian_positive: bool
    stable_stationary_certified: bool
    arithmetic_method: str = INTERVAL_METHOD
    lower_weight_bits: int = _LOWER_WEIGHT_BITS
    scope: tuple[str, ...] = (
        "first_phase_is_zero_gauge_with_all_other_phases_free",
        "unique_stationary_sine_root_in_the_declared_phase_box_if_included",
        "strict_grounded_hessian_positivity_is_a_sufficient_relative_condition",
        "failed_positivity_is_unavailable_not_an_instability_certificate",
        "complete_temporal_law_and_marked_windings_are_caller_premises",
        "no_source_inclusion_trajectory_event_or_physical_identification",
    )


def _certify_organized_stationary(*, phase_center, edges, radius, inverse):
    """Certify a proposed root box without searching or trusting its inverse.

    Support has 2..54 nodes and unique undirected unit edges. The first phase
    is exactly zero; every other coordinate ranges independently over center
    +/- radius. ``inverse`` is an arbitrary proposed inverse of the grounded
    phase Hessian, admitted and checked rather than assumed correct.

    For F=-S on free nodes, T(y)=y-R F(y) contracts with bound qbox and maps
    the box into its interior if eta+qbox*radius<radius. Invertibility of R
    follows from q0<1. Root inclusion and positivity remain separate results.
    """
    raw_center = _ordered(phase_center, "phase center", limit=_MAX_NODES + 1)
    size = len(raw_center)
    if not 2 <= size <= _MAX_NODES:
        raise ValueError("phase center requires 2..54 registered coordinates")
    center = _prepared_coordinate_vector(raw_center, "phase center", size)
    if center[0] != 0:
        raise ValueError("the first phase center must be exactly zero")
    admitted_radius = exact_or_represented_real(radius, "inclusion radius")
    if admitted_radius <= 0:
        raise ValueError("inclusion radius must be strictly positive")
    support, degree = _admit_support(edges, size)
    raw_inverse = _ordered(inverse, "inverse", limit=size)
    if len(raw_inverse) != size - 1:
        raise ValueError("inverse requires one row per free phase")
    proposed_inverse = tuple(
        _prepared_coordinate_vector(row, f"inverse[{index}]", size - 1)
        for index, row in enumerate(raw_inverse)
    )

    currents, potentials, _ = _bounded_sine_edge_fields(
        phase_offsets=center,
        weighted_edges=tuple((left, right, Q(1)) for left, right in support),
        alpha_bounds=I(0),
    )
    edge_hessian = tuple(1 - value for value in potentials)
    hessian = _grounded_laplacian(support, edge_hessian, size, I(0))
    midpoint = tuple(tuple(value.midpoint for value in row) for row in hessian)
    hessian_row_radii = tuple(sum(value.radius for value in row) for row in hessian)
    residuals = tuple(-value for value in currents[1:])
    norm = max(sum(abs(value) for value in row) for row in proposed_inverse)
    eta = max(
        abs(sum(value * residual.midpoint for value, residual in zip(row, residuals)))
        + sum(abs(value) * residual.radius for value, residual in zip(row, residuals))
        for row in proposed_inverse
    )
    product = exact_matrix_product(proposed_inverse, midpoint)
    q0 = max(
        sum(abs(Q(i == j) - value) for j, value in enumerate(row))
        + sum(
            abs(value) * row_radius
            for value, row_radius in zip(proposed_inverse[i], hessian_row_radii)
        )
        for i, row in enumerate(product)
    )
    # |cos(delta_y)-cos(delta_center)| <= 2*r; a Laplacian row
    # has absolute sum at most twice the sum of incident weight changes.
    qbox = q0 + 4 * degree * admitted_radius * norm
    image_radius = eta + qbox * admitted_radius
    contraction = qbox < 1
    inclusion = contraction and image_radius < admitted_radius

    scale = 1 << _LOWER_WEIGHT_BITS
    lower_weights = []
    for (left, right), value in zip(support, edge_hessian):
        moving_endpoints = int(left != 0) + int(right != 0)
        lower = max(Q(-1), value.lo - moving_endpoints * admitted_radius)
        scaled = lower * scale
        lower_weights.append(Q(scaled.numerator // scaled.denominator, scale))
    lower_weights = tuple(lower_weights)
    lower_hessian = _grounded_laplacian(support, lower_weights, size, Q(0))
    positive = exact_symmetric_semidefinite(lower_hessian, strict=True)
    return _OrganizedStationaryCertificate(
        phase_center=center,
        edges=support,
        radius=admitted_radius,
        inverse=proposed_inverse,
        maximum_degree=degree,
        phase_current_bounds=currents,
        edge_hessian_bounds=edge_hessian,
        grounded_hessian_bounds=hessian,
        midpoint_grounded_hessian=midpoint,
        lower_edge_hessian=lower_weights,
        lower_grounded_hessian=lower_hessian,
        inverse_norm=norm,
        eta=eta,
        q0=q0,
        qbox=qbox,
        image_radius_bound=image_radius,
        contraction_certified=contraction,
        root_inclusion_certified=inclusion,
        relative_hessian_positive=positive,
        stable_stationary_certified=inclusion and positive,
    )
