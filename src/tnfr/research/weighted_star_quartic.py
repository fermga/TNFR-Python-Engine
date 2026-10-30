"""Exact quartic transfer test for the retained weighted-star sine source.

The fixed descending-lexicographic gauge removes the ten free coefficients.
This detached calculation runs neither events nor trajectories. SymPy is an
optional candidate-solving backend; exact rational identities and a modular
rank certificate independently check its answer before any source evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import lcm, prod

from ..mathematics._exact_linear_algebra import _rational_vector, exact_matrix_inverse
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ..mathematics.krylov import exact_rank
from .weighted_star_reachability import (
    WeightedStarQuadraticObstruction,
    assess_weighted_star_quadratic_obstruction,
)

_DIMENSION = 8
_RANK_PRIME = 65521


def _monomials(degree, dimension=_DIMENSION):
    if dimension == 1:
        yield (degree,)
        return
    for power in range(degree, -1, -1):
        for tail in _monomials(degree - power, dimension - 1):
            yield (power,) + tail


def _add(polynomial, exponent, coefficient):
    if coefficient:
        value = polynomial.get(exponent, Q(0)) + coefficient
        if value:
            polynomial[exponent] = value
        else:
            polynomial.pop(exponent, None)


def _freeze(polynomial):
    return tuple(sorted(polynomial.items(), reverse=True))


def _multiply(left, right):
    result = {}
    for a, c in left.items():
        for b, d in right.items():
            _add(result, tuple(x + y for x, y in zip(a, b)), c * d)
    return result


def _variable(index):
    return {tuple(int(i == index) for i in range(_DIMENSION)): Q(1)}


def _phase_difference(first, second):
    result = {}
    for block, sign in ((first, -1), (second, 1)):
        if block:
            _add(result, next(iter(_variable(block + 3))), Q(sign))
    return result


def _cubic_field(system):
    # These are pi times the cubic rates. Phase channels have no nonlinear
    # term; bare neighbor counts, not conductances, set the form channel.
    absolute = []
    for i, (mobility, neighbors) in enumerate(
        zip(system.representative_mobility, system.bare_neighbor_rows)
    ):
        polynomial = {}
        for j, count in enumerate(neighbors):
            gap = _phase_difference(i, j)
            cube = _multiply(_multiply(gap, gap), gap)
            for exponent, value in cube.items():
                _add(polynomial, exponent, -mobility * count * value / 6)
        absolute.append(polynomial)
    relative = []
    for row in absolute[1:]:
        polynomial = row.copy()
        for exponent, value in absolute[0].items():
            _add(polynomial, exponent, -value)
        relative.append(_freeze(polynomial))
    return tuple(relative) + ((),) * 4


def _quadratic_polynomial(matrix):
    result = {}
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            exponent = [0] * _DIMENSION
            exponent[i] += 1
            exponent[j] += 1
            _add(result, tuple(exponent), value)
    return result


def _directional_derivative(polynomial, field):
    """Differentiate sparse polynomials without using the assembled matrix."""
    result = {}
    for exponent, coefficient in polynomial.items():
        for index, power in enumerate(exponent):
            if power:
                reduced = list(exponent)
                reduced[index] -= 1
                for direction, value in field[index]:
                    target = tuple(a + b for a, b in zip(reduced, direction))
                    _add(result, target, coefficient * power * value)
    return result


def _linear_field(generator):
    return tuple(
        tuple((next(iter(_variable(j))), value) for j, value in enumerate(row) if value)
        for row in generator
    )


def _evaluate(polynomial, state):
    return sum(
        (
            coefficient * prod(x**n for x, n in zip(state, exponent))
            for exponent, coefficient in polynomial
        ),
        Q(0),
    )


def _modular_pivots(matrix):
    """A full-row-rank image over this prime proves rational full row rank.

    Denominators must be units in the field. No claim is made from a smaller
    modular rank, since reduction modulo a prime can lose rational rank.
    """
    prime = _RANK_PRIME
    rows = []
    for row in matrix:
        if any(value.denominator % prime == 0 for value in row):
            raise ValueError("rank-certificate denominator is not invertible")
        rows.append(
            [
                (value.numerator % prime) * pow(value.denominator, -1, prime) % prime
                for value in row
            ]
        )
    pivots = []
    for column in range(len(rows[0])):
        index = len(pivots)
        pivot = next((i for i in range(index, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[index], rows[pivot] = rows[pivot], rows[index]
        inverse = pow(rows[index][column], -1, prime)
        rows[index] = [value * inverse % prime for value in rows[index]]
        for i in range(index + 1, len(rows)):
            multiplier = rows[i][column]
            if multiplier:
                rows[i] = [
                    (a - multiplier * b) % prime for a, b in zip(rows[i], rows[index])
                ]
        pivots.append(column)
        if len(pivots) == len(rows):
            break
    return tuple(pivots)


@dataclass(frozen=True)
class WeightedStarQuarticSystem:
    """Admitted source, frozen parity sector and exact homological equation."""

    quadratic: WeightedStarQuadraticObstruction
    unknown_monomials: tuple
    equation_monomials: tuple
    cubic_field_pi: tuple
    matrix: tuple
    right_hand_side: tuple
    rank_prime: int
    certified_pivots: tuple
    gauge_basis: tuple
    gauge_dimension: int


@dataclass(frozen=True)
class WeightedStarQuarticTransfer:
    """Verified fixed-gauge correction; solvability alone is not transfer."""

    system: WeightedStarQuarticSystem
    coefficients: tuple
    polynomial: tuple
    free_columns: tuple
    initial_correction: Q
    initial_corrected_witness: Q
    initial_correction_gradient: tuple
    initial_remainder_gradient_coefficient: Q
    initial_corrected_derivative: I
    domain: WeightedStarQuarticDomain


@dataclass(frozen=True)
class WeightedStarQuarticDomain:
    """Conserved-storage bounds, not a trajectory or a reachability verdict."""

    initial_sine_storage: I
    phase_coercivity: Q
    ellipsoid_metric: tuple
    coordinate_absolute_bounds: tuple
    correction_absolute_bound: Q
    corrected_derivative_absolute_bound: Q
    initial_observation_lower_bound: Q
    absolute_bound_retains_threshold: bool
    same_storage_target_correction: I
    same_storage_target_observation: I


def _source_domain(system, polynomial, initial_corrected):
    relative = system.quadratic.system
    state = relative.initial_state
    energy = relative.energy_metric
    phases = (Q(0),) + state[4:]
    multiplicities = (4, 1, 1, 3, 1)
    bare_edges = tuple(
        (i, j, multiplicities[i] * count)
        for i, row in enumerate(relative.bare_neighbor_rows)
        for j, count in enumerate(row)
        if i < j and count
    )
    storage = I(
        sum(
            (state[i] * energy[i][j] * state[j] for i in range(4) for j in range(4)),
            Q(0),
        )
        / 2
    )
    storage += sum(
        (count * (1 - cos(I(phases[j] - phases[i]))) for i, j, count in bare_edges),
        I(0),
    )
    inverse_phase = exact_matrix_inverse(tuple(row[4:] for row in energy[4:]))
    directions = {
        (i, j): tuple(Q(int(j == k + 1) - int(i == k + 1)) for k in range(4))
        for i, j, _ in bare_edges
    }
    resistances = tuple(
        sum(
            (
                direction[i] * inverse_phase[i][j] * direction[j]
                for i in range(4)
                for j in range(4)
            ),
            Q(0),
        )
        for direction in directions.values()
    )
    if (
        storage.hi >= Q(21, 50)
        or max(resistances) > Q(1, 3)
        or any(
            abs(phases[j] - phases[i]) >= pi_interval().lo / 3 for i, j, _ in bare_edges
        )
    ):
        raise ValueError("retained short-arc source-domain premises failed")
    # H<1/2 prevents any bare edge reaching pi/3. The initial 9/10
    # coercivity gives delta^2<14/45, then Taylor gives >97/100.
    # Repeating that bound yields delta^2<28/97<3/10, hence 39/40.
    beta = Q(39, 40)
    metric = tuple(
        tuple(value * (beta if i >= 4 else 1) for value in row)
        for i, row in enumerate(energy)
    )
    inverse = exact_matrix_inverse(metric)
    radius = tuple(sqrt(I(2 * storage.hi * inverse[i][i])).hi for i in range(8))
    gaps = {}
    for (i, j), direction in directions.items():
        squared = sum(
            (
                direction[a] * inverse[a + 4][b + 4] * direction[b]
                for a in range(4)
                for b in range(4)
            ),
            Q(0),
        )
        gaps[i, j] = sqrt(I(2 * storage.hi * squared)).hi
    correction_bound = sum(
        (
            abs(value) * prod(b**n for b, n in zip(radius, exponent))
            for exponent, value in polynomial
        ),
        Q(0),
    )
    gradient_bounds = []
    for axis in range(8):
        total = Q(0)
        for exponent, value in polynomial:
            if exponent[axis]:
                powers = list(exponent)
                powers[axis] -= 1
                total += (
                    abs(value)
                    * exponent[axis]
                    * prod(b**n for b, n in zip(radius, powers))
                )
        gradient_bounds.append(total)
    remainders = []
    for power, divisor in ((3, 6), (5, 120)):
        absolute = tuple(
            mobility
            * sum(
                (
                    count * gaps[tuple(sorted((i, j)))] ** power / divisor
                    for j, count in enumerate(row)
                    if i != j and count
                ),
                Q(0),
            )
            for i, (mobility, row) in enumerate(
                zip(relative.representative_mobility, relative.bare_neighbor_rows)
            )
        )
        remainders.append(
            tuple(value + absolute[0] for value in absolute[1:]) + (Q(0),) * 4
        )
    r3_bound, fifth_bound = remainders
    residual_bound = (
        2
        * sum(
            (
                radius[i] * abs(system.quadratic.invariant[i][j]) * fifth_bound[j]
                for i in range(8)
                for j in range(8)
            ),
            Q(0),
        )
        + sum((gradient_bounds[i] * r3_bound[i] for i in range(8)), Q(0))
    ) / pi_interval().lo
    lower = initial_corrected - correction_bound
    # A detached same-H/means target: only relative x_H is nonzero,
    # x_H^2=2H/Ex_HH. Its negative correction independently rules out
    # making this symmetric |P4| transfer informative by tighter boxes alone.
    target_square = 2 * storage / energy[0][0]
    target_correction = dict(polynomial).get((4,) + (0,) * 7, Q(0)) * target_square**2
    target_observation = relative.observation[0][0] * target_square
    return WeightedStarQuarticDomain(
        storage,
        beta,
        metric,
        radius,
        correction_bound,
        residual_bound,
        lower,
        lower > Q(1, 4),
        target_correction,
        target_observation,
    )


def build_weighted_star_quartic_system(preparation):
    """Re-admit the unchanged source before assembling any cached calculation."""
    quadratic = assess_weighted_star_quadratic_obstruction(preparation)
    system = quadratic.system
    unknowns = tuple(e for e in _monomials(4) if sum(e[:4]) % 2 == 0)
    equations = tuple(e for e in _monomials(4) if sum(e[:4]) % 2 == 1)
    row_indices = {exponent: index for index, exponent in enumerate(equations)}
    matrix = [[Q(0) for _ in unknowns] for _ in equations]
    for column, exponent in enumerate(unknowns):
        for i, power in enumerate(exponent):
            if power:
                for j, value in enumerate(system.generator_pi[i]):
                    if value:
                        target = list(exponent)
                        target[i] -= 1
                        target[j] += 1
                        matrix[row_indices[tuple(target)]][column] += power * value
    matrix = tuple(tuple(row) for row in matrix)
    cubic = _cubic_field(system)
    derivative = _directional_derivative(
        _quadratic_polynomial(quadratic.invariant), cubic
    )
    if any(exponent not in row_indices for exponent in derivative):
        raise ValueError("cubic drift escaped the frozen odd-form sector")
    rhs = tuple(-derivative.get(exponent, Q(0)) for exponent in equations)
    pivots = _modular_pivots(matrix)
    if pivots != tuple(range(160)):
        raise ValueError("fixed first-160-column rational rank certificate failed")
    quadratics = tuple(_quadratic_polynomial(value) for value in system.invariant_basis)
    gauges = tuple(
        _freeze(_multiply(quadratics[i], quadratics[j]))
        for i in range(4)
        for j in range(i, 4)
    )
    gauge_rank = exact_rank(
        tuple(
            tuple(dict(polynomial).get(exponent, Q(0)) for exponent in unknowns)
            for polynomial in gauges
        )
    )
    linear = _linear_field(system.generator_pi)
    if gauge_rank != 10 or any(
        _directional_derivative(dict(polynomial), linear) for polynomial in gauges
    ):
        raise ValueError("quartic invariant products failed to span the kernel")
    return WeightedStarQuarticSystem(
        quadratic,
        unknowns,
        equations,
        cubic,
        matrix,
        rhs,
        _RANK_PRIME,
        pivots,
        gauges,
        gauge_rank,
    )


@lru_cache(maxsize=1)
def _solve_fixed_system(matrix, rhs):
    """Generate a candidate with optional SymPy; caller verifies independently.

    Cache only immutable admitted algebra, never a source-dependent verdict.
    Scale the RHS globally before clearing matrix rows, limiting integer growth.
    """
    try:
        from sympy import ZZ
        from sympy.polys.matrices import DomainMatrix
    except ImportError as exc:
        raise ImportError(
            "the optional sympy backend is required for the quartic solve"
        ) from exc
    scale = lcm(*(value.denominator for value in rhs))
    integers = []
    for row, value in zip(matrix, rhs):
        denominator = lcm(*(item.denominator for item in row))
        integers.append(
            [int(item * denominator) for item in row]
            + [int(value * scale * denominator)]
        )
    reduced, denominator, pivots = (
        DomainMatrix.from_list(integers, ZZ).to_sparse().rref_den(method="FF")
    )
    if pivots != tuple(range(160)):
        raise ValueError("exact candidate solve disagrees with certified pivots")
    entries = reduced.to_dok()
    coefficients = [Q(0)] * 170
    for row, column in enumerate(pivots):
        coefficients[column] = Q(
            int(entries.get((row, 170), 0)), int(denominator) * scale
        )
    return tuple(coefficients)


def _verify_correction(system, coefficients):
    coefficients = _rational_vector(coefficients, "quartic coefficients")
    if len(coefficients) != 170:
        raise ValueError("exactly 170 quartic coefficients are required")
    free = tuple(i for i in range(170) if i not in system.certified_pivots)
    if any(coefficients[i] for i in free):
        raise ValueError("quartic coefficients violate the fixed free-zero gauge")
    polynomial = _freeze(
        {
            exponent: value
            for exponent, value in zip(system.unknown_monomials, coefficients)
            if value
        }
    )
    image = _directional_derivative(
        dict(polynomial), _linear_field(system.quadratic.system.generator_pi)
    )
    drift = _directional_derivative(
        _quadratic_polynomial(system.quadratic.invariant), system.cubic_field_pi
    )
    if image != {exponent: -value for exponent, value in drift.items()}:
        raise ValueError("quartic correction fails the exact polynomial identity")
    correction = _evaluate(polynomial, system.quadratic.system.initial_state)
    source = system.quadratic.system.initial_state
    gradient = tuple(
        _evaluate(
            _freeze(
                _directional_derivative(
                    dict(polynomial),
                    tuple((((0,) * 8, Q(1)),) if i == axis else () for i in range(8)),
                )
            ),
            source,
        )
        for axis in range(8)
    )
    alpha = source[5]
    phases = (Q(0),) + source[4:]
    raw = tuple(
        mobility
        * sum(
            (count * (phases[j] - phases[i]) / alpha for j, count in enumerate(row)),
            Q(0),
        )
        for i, (mobility, row) in enumerate(
            zip(
                system.quadratic.system.representative_mobility,
                system.quadratic.system.bare_neighbor_rows,
            )
        )
    )
    direction = tuple(value - raw[0] for value in raw[1:]) + (Q(0),) * 4
    ell = sum((a * b for a, b in zip(gradient, direction)), Q(0))
    k = system.quadratic.sine_initial_drift_coefficient
    lead = sum(
        (
            gradient[i] * value * source[j]
            for i, row in enumerate(system.quadratic.system.generator_pi)
            for j, value in enumerate(row)
        ),
        Q(0),
    )
    if lead != k * alpha**3 / 6:
        raise ValueError(
            "source quartic cancellation does not match the declared clock"
        )
    remainder = sin(I(alpha)) - alpha
    derivative = (k * (remainder + alpha**3 / 6) + ell * remainder) / pi_interval()
    return WeightedStarQuarticTransfer(
        system,
        coefficients,
        polynomial,
        free,
        correction,
        system.quadratic.initial_invariant + correction,
        gradient,
        ell,
        derivative,
        _source_domain(
            system, polynomial, system.quadratic.initial_invariant + correction
        ),
    )


def assess_weighted_star_quartic_transfer(preparation, *, coefficients=None):
    """Solve or verify the prescribed correction after fresh source admission.

    Supplied coefficients are untrusted candidates, never accepted verdicts.
    Backend unavailability or failure raises; neither is a mathematical no-go.
    """
    system = build_weighted_star_quartic_system(preparation)
    if coefficients is None:
        coefficients = _solve_fixed_system(system.matrix, system.right_hand_side)
    return _verify_correction(system, coefficients)
