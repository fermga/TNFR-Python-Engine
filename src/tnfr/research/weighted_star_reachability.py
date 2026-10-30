"""Exact relative-state quadratic obstructions for one retained contact.

This is the complete eight-dimensional mean-fixed five-block family. It is
not an eight-output reduction for arbitrary fine states. No trajectory,
operator, producer, autonomous selector or changed constitutive law is run.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._exact_linear_algebra import (
    _rational_vector,
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import I, pi_interval, sin
from ..mathematics.krylov import exact_rank
from .weighted_star_constitutive_handoff import (
    compare_weighted_star_constitutive_handoff,
)
from .weighted_star_contact import COLLECTIVE_BLOCKS

WITNESS_COEFFICIENTS = (Q(3, 4), Q(-41, 900), Q(43, 54000), Q(-1, 270000))


def _transpose(matrix):
    return tuple(zip(*matrix))


def _combine(left, right, factor=Q(1)):
    return tuple(
        tuple(a + factor * b for a, b in zip(row, other))
        for row, other in zip(left, right)
    )


def _quadratic(matrix, vector):
    return sum(
        (
            vector[i] * value * vector[j]
            for i, row in enumerate(matrix)
            for j, value in enumerate(row)
        ),
        Q(0),
    )


def _lyapunov(generator, metric):
    product = exact_matrix_product(metric, generator)
    return _combine(product, _transpose(product))


@dataclass(frozen=True)
class WeightedStarRelativeSystem:
    """Rebuilt exact matrices; generator_pi is pi times the real-time field."""

    preparation_record_sha256: str
    generator_pi: tuple
    observation: tuple
    initial_state: tuple
    mean_weights: tuple
    conserved_origins: tuple
    mean_fixed_injection: tuple
    energy_metric: tuple
    initial_tangent_storage: Q
    invariant_basis: tuple
    invariant_dimension: int
    lyapunov_rank: int
    representative_mobility: tuple
    bare_neighbor_rows: tuple


@dataclass(frozen=True)
class WeightedStarQuadraticObstruction:
    """Exact tangent witness and its independently computed sine derivative."""

    system: WeightedStarRelativeSystem
    coefficients: tuple
    invariant: tuple
    domination_margin: tuple
    domination_certified: bool
    initial_invariant: Q
    threshold_margin: Q
    tangent_exclusion_certified: bool
    sine_initial_drift_coefficient: Q
    sine_initial_derivative: I
    nonlinear_invariance_excluded: bool


def build_weighted_star_relative_system(preparation):
    """Re-admit the fixed preparation and characterize every quadratic invariant.

    Coordinates are differences from block A=(0,1,2,3), in the order
    (H=4,U=5,V=(6,7,8),D=9), first form, then continued real phase.
    The separately conserved weighted means reconstruct the missing origins.
    """
    handoff = compare_weighted_star_constitutive_handoff(preparation)
    source = handoff.prefixes[-1].source
    blocks = COLLECTIVE_BLOCKS
    node_block = tuple(
        next(k for k, block in enumerate(blocks) if i in block) for i in range(10)
    )
    representatives = tuple(block[0] for block in blocks)
    laplacians = [[[Q(0) for _ in blocks] for _ in blocks] for _ in range(2)]
    neighbors = [[0 for _ in blocks] for _ in blocks]
    for (i, j), weight in zip(source.edges, source.conductances):
        a, b = node_block[i], node_block[j]
        if a != b:
            for matrix, value in zip(laplacians, (weight, Q(1))):
                matrix[a][a] += value
                matrix[b][b] += value
                matrix[a][b] -= value
                matrix[b][a] -= value
        if i in representatives:
            neighbors[node_block[i]][b] += 1
        if j in representatives:
            neighbors[node_block[j]][a] += 1
    multiplicities = tuple(map(len, blocks))
    mobility = tuple(source.capacity[i] / source.degrees[i] for i in representatives)
    # Coarse Laplacians sum all fine incidences; divide by block multiplicity
    # before applying the actual representative mobility.
    rows = tuple(
        tuple(
            tuple(mobility[i] * matrix[i][j] / multiplicities[i] for j in range(5))
            for i in range(5)
        )
        for matrix in laplacians
    )
    reduced = tuple(
        tuple(
            tuple(matrix[i][j] - matrix[0][j] for j in range(1, 5)) for i in range(1, 5)
        )
        for matrix in rows
    )
    generator = tuple(
        tuple(
            (
                -reduced[1][i][j]
                if channel == 0 and other == 1
                else reduced[0][i][j] if channel == 1 and other == 0 else Q(0)
            )
            for other in range(2)
            for j in range(4)
        )
        for channel in range(2)
        for i in range(4)
    )
    energy = tuple(
        tuple(
            laplacians[channel][i + 1][j + 1] if channel == other else Q(0)
            for other in range(2)
            for j in range(4)
        )
        for channel in range(2)
        for i in range(4)
    )
    masses = multiplicities[1:]
    observation = tuple(
        tuple(
            (
                (Q(masses[i]) if i == j else Q(0)) - Q(masses[i] * masses[j], 10)
                if channel == other
                else Q(0)
            )
            for other in range(2)
            for j in range(4)
        )
        for channel in range(2)
        for i in range(4)
    )
    weights = tuple(
        sum((Q(source.degrees[i]) / source.capacity[i] for i in block), Q(0))
        for block in blocks
    )
    total = sum(weights, Q(0))
    injection = tuple(
        tuple(-weights[j + 1] / total + int(i == j + 1) for j in range(4))
        for i in range(5)
    )
    origins = tuple(
        sum((weights[i] * values[node] for i, node in enumerate(representatives)), Q(0))
        / total
        for values in (source.epi, source.phase)
    )
    initial = tuple(
        values[node] - values[representatives[0]]
        for values in (source.epi, source.phase)
        for node in representatives[1:]
    )
    tangent_storage = _quadratic(energy, initial) / 2
    hub_edges = {frozenset(edge) for edge in source.edges}
    if tangent_storage >= Q(1, 2) or any(
        frozenset((4, node)) not in hub_edges for node in source.nodes if node != 4
    ):
        raise ValueError("tangent circular-observation short-arc premises failed")
    negative_square = tuple(
        tuple(-value for value in row)
        for row in exact_matrix_product(generator, generator)
    )
    basis = [energy]
    for _ in range(3):
        basis.append(exact_matrix_product(basis[-1], negative_square))
    upper = tuple((i, j) for i in range(8) for j in range(i, 8))
    lyapunov_columns = []
    for a, b in upper:
        unit = tuple(
            tuple(Q(int((i, j) in ((a, b), (b, a)))) for j in range(8))
            for i in range(8)
        )
        image = _lyapunov(generator, unit)
        lyapunov_columns.append(tuple(image[i][j] for i, j in upper))
    rank = exact_rank(_transpose(tuple(lyapunov_columns)))
    basis_rank = exact_rank(
        tuple(tuple(matrix[i][j] for i, j in upper) for matrix in basis)
    )
    if (
        not exact_symmetric_semidefinite(energy, strict=True)
        or any(matrix != _transpose(matrix) for matrix in basis)
        or any(
            any(value for row in _lyapunov(generator, matrix) for value in row)
            for matrix in basis
        )
        or basis_rank != 4
        or rank != len(upper) - basis_rank
    ):
        raise ValueError("retained source quadratic invariant characterization failed")
    return WeightedStarRelativeSystem(
        preparation_record_sha256=handoff.preparation_record_sha256,
        generator_pi=generator,
        observation=observation,
        initial_state=initial,
        mean_weights=weights,
        conserved_origins=origins,
        mean_fixed_injection=injection,
        energy_metric=energy,
        initial_tangent_storage=tangent_storage,
        invariant_basis=tuple(basis),
        invariant_dimension=basis_rank,
        lyapunov_rank=rank,
        representative_mobility=mobility,
        bare_neighbor_rows=tuple(tuple(row) for row in neighbors),
    )


def _sine_remainder(system, state):
    phases = (Q(0),) + tuple(state[4:])
    rates = tuple(
        mobility
        * sum(
            (
                count * (sin(I(phases[j] - phases[i])) - (phases[j] - phases[i]))
                for j, count in enumerate(row)
            ),
            I(0),
        )
        / pi_interval()
        for i, (mobility, row) in enumerate(
            zip(system.representative_mobility, system.bare_neighbor_rows)
        )
    )
    return tuple(rate - rates[0] for rate in rates[1:]) + (I(0),) * 4


def assess_weighted_star_quadratic_obstruction(
    preparation, *, coefficients=WITNESS_COEFFICIENTS
):
    """Verify one rational polynomial witness without numerical optimization.

    Coefficients multiply E(-A^2)^k, k=0,...,3. An unsuccessful candidate is
    not evidence of reachability or absence of other quadratic witnesses.
    The sine derivative is at the unchanged nominal initial state only.
    """
    coefficients = _rational_vector(coefficients, "invariant coefficients")
    if len(coefficients) != 4:
        raise ValueError("exactly four polynomial invariant coefficients are required")
    system = build_weighted_star_relative_system(preparation)
    invariant = tuple(
        tuple(
            sum(
                (
                    coefficient * basis[i][j]
                    for coefficient, basis in zip(coefficients, system.invariant_basis)
                ),
                Q(0),
            )
            for j in range(8)
        )
        for i in range(8)
    )
    margin = _combine(system.observation, invariant, Q(-1))
    dominated = exact_symmetric_semidefinite(margin)
    initial = _quadratic(invariant, system.initial_state)
    # At this source every nonzero bare gap is +/-alpha. Factor the shared
    # sine-minus-tangent term before interval evaluation, preserving exact
    # cancellations in the invariant's source coefficient.
    alpha = system.initial_state[5]
    phases = (Q(0),) + system.initial_state[4:]
    current_factors = tuple(
        mobility
        * sum(
            (count * (phases[j] - phases[i]) / alpha for j, count in enumerate(row)),
            Q(0),
        )
        for i, (mobility, row) in enumerate(
            zip(system.representative_mobility, system.bare_neighbor_rows)
        )
    )
    direction = (
        tuple(value - current_factors[0] for value in current_factors[1:]) + (Q(0),) * 4
    )
    coefficient = 2 * sum(
        (
            system.initial_state[i] * invariant[i][j] * direction[j]
            for i in range(8)
            for j in range(8)
        ),
        Q(0),
    )
    derivative = coefficient * (sin(I(alpha)) - alpha) / pi_interval()
    return WeightedStarQuadraticObstruction(
        system=system,
        coefficients=coefficients,
        invariant=invariant,
        domination_margin=margin,
        domination_certified=dominated,
        initial_invariant=initial,
        threshold_margin=initial - Q(1, 4),
        tangent_exclusion_certified=dominated and initial > Q(1, 4),
        sine_initial_drift_coefficient=coefficient,
        sine_initial_derivative=derivative,
        nonlinear_invariance_excluded=not derivative.contains(0),
    )
