"""Conservative full-boundary storage bounds for an acute sine sector.

Inputs have already passed the public recovery owner's support and sector
admission. Every signed edge face receives a convex dual lower bound, including
possibly empty faces. Rational least-norm points choose supporting planes only;
they need not be feasible, critical, or a supplied equilibrium target.
"""

from __future__ import annotations

from fractions import Fraction as Q

from ..mathematics._exact_linear_algebra import exact_matrix_inverse
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from .phase_cycle_geometry import PhaseCycleGeometry


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Q(0))


def _sector_boundary_bounds(
    geometry: PhaseCycleGeometry, cycle_periods: tuple[int, ...]
) -> tuple[Q, ...]:
    """Bound ``sum(1-cos(2*pi*t))`` on every closed acute-sector face.

    Face order is edge zero at -1/4, edge zero at +1/4, then successive
    edges. Taking the minimum covers the entire relative boundary whenever
    the open sector is nonempty. An empty face needs no feasibility verdict:
    its lower bound is vacuously valid and can only weaken the final result.

    With the fixed edge removed, write the cycle constraints ``B*t=d``.
    For any cube point u and any rational multiplier lambda, convexity gives
    the bound

    ``1 + lambda*d + sum(f(u)-f'(u)*u)
       - sum(abs(f'(u)-B.T*lambda))/4``.

    All trigonometry and arithmetic enclosing derivatives use outward
    rational intervals. The least-norm construction and multiplier selection
    are deterministic hints, not optimization or equilibrium certificates.
    """
    edge_count = len(geometry.edges)
    rank = geometry.cycle_rank
    if not rank:
        return (Q(1),) * (2 * edge_count)

    columns = tuple(
        tuple(Q(row[edge]) for row in geometry.cycle_rows) for edge in range(edge_count)
    )
    gram = tuple(
        tuple(
            sum((column[i] * column[j] for column in columns), Q(0))
            for j in range(rank)
        )
        for i in range(rank)
    )
    inverse = exact_matrix_inverse(gram)
    two_pi = 2 * pi_interval()
    quarter = Q(1, 4)
    tangents: dict[Q, tuple[I, I]] = {
        Q(0): (I(0), I(0)),
        quarter: (I(1), two_pi),
    }

    def tangent(point):
        magnitude = abs(point)
        if magnitude not in tangents:
            angle = two_pi * magnitude
            tangents[magnitude] = (1 - cos(angle), two_pi * sin(angle))
        value, derivative = tangents[magnitude]
        return value, -derivative if point < 0 else derivative

    result = []
    for edge, fixed_column in enumerate(columns):
        # Removing one edge leaves the cycle rows independent: a cycle-space
        # vector supported on one nonloop edge would have nonzero divergence.
        # Thus the rank-one inverse update always has a positive denominator.
        image = tuple(_dot(row, fixed_column) for row in inverse)
        denominator = 1 - _dot(fixed_column, image)
        if denominator <= 0:
            raise ValueError("canonical cycle rows lost rank after deleting one edge")
        reduced_inverse = tuple(
            tuple(
                inverse[i][j] + image[i] * image[j] / denominator for j in range(rank)
            )
            for i in range(rank)
        )
        free_columns = columns[:edge] + columns[edge + 1 :]
        for sign in (-1, 1):
            rhs = tuple(
                Q(period) - sign * quarter * entry
                for period, entry in zip(cycle_periods, fixed_column)
            )
            lift = tuple(_dot(row, rhs) for row in reduced_inverse)
            points = tuple(
                min(quarter, max(-quarter, _dot(column, lift)))
                for column in free_columns
            )
            planes = tuple(tangent(point) for point in points)
            projected_gradient = tuple(
                sum(
                    (
                        column[i] * plane[1].midpoint
                        for column, plane in zip(free_columns, planes)
                    ),
                    Q(0),
                )
                for i in range(rank)
            )
            multipliers = tuple(
                _dot(row, projected_gradient) for row in reduced_inverse
            )
            lower = I(1) + _dot(multipliers, rhs)
            for column, point, (value, derivative) in zip(free_columns, points, planes):
                residual = derivative - _dot(column, multipliers)
                lower += value - derivative * point - quarter * residual.abs_max
            # The fixed edge alone contributes exactly one; all other edge
            # potentials are nonnegative on the complete closed cube.
            result.append(max(Q(1), lower.lo))
    return tuple(result)
