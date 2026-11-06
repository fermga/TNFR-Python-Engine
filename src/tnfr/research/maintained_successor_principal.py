"""Exact stationary-core observations on the proposed maintained UM support.

True twin equality, leaf stationarity and the three-block chain argument
reduce any stationary PSD candidate to six relative coordinates. The further
four-coordinate map aligns the two odd-degree middle vertices. Neither map
is a dynamical closure, a root certificate or a global exclusion theorem.
Every candidate retains all fifty-three principal branch inequalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from .maintained_successor_chamber import predict_successor_chamber
from .maintained_successor_stationary import SELECTED_SHIFTS, SELECTED_SUPPORT


def _core_embedding():
    # Coordinates (parent, sibling, A2, B0, B1, B2), with A0 fixed at zero.
    matrix = [[Q(0)] * 6 for _ in range(21)]
    offsets = [Q(0)] * 21
    for node, index in ((2, 0), (3, 1), (4, 2), (5, 2), (20, 0)):
        matrix[node][index] = Q(1)
    for node in (6, 7):
        matrix[node][2], offsets[node] = Q(2, 3), Q(2, 3)
    for node in (8, 9):
        matrix[node][2], offsets[node] = Q(1, 3), Q(4, 3)
    for first, index in ((10, 3), (12, 4), (14, 5)):
        for node in (first, first + 1):
            matrix[node][index] = Q(1)
    for node in (16, 17):
        matrix[node][3], matrix[node][5] = Q(1, 3), Q(2, 3)
        offsets[node] = Q(2, 3)
    for node in (18, 19):
        matrix[node][3], matrix[node][5] = Q(2, 3), Q(1, 3)
        offsets[node] = Q(4, 3)
    return tuple(map(tuple, matrix)), tuple(offsets)


CORE_MATRIX, CORE_PI_OFFSETS = _core_embedding()
TWIN_PAIRS = (
    (0, 1),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
)


@dataclass(frozen=True)
class PrincipalCore:
    """Whole-box observations; enclosing zero is not a stationarity proof."""

    parameters: tuple[I, ...]
    phases: tuple[I, ...]
    edge_differences: tuple[I, ...]
    principal_margins: tuple[I, ...]
    chain_gaps: tuple[I, I]
    chain_acute_margins: tuple[I, I]
    potential: I
    gradient: tuple[I, ...]
    hessian: tuple[tuple[I, ...], ...]
    core_gradient: tuple[I, ...]
    core_hessian: tuple[tuple[I, ...], ...]
    transverse_eigenvalues: tuple[I, ...]


@dataclass(frozen=True)
class SuccessorCoreEvaluation:
    """Fresh primitive-source association and detached candidate observations."""

    chamber: object
    candidate: PrincipalCore


@dataclass(frozen=True)
class PrincipalReduction:
    """Four-coordinate minimum search domain, with no minimum verdict.

    Parameters are (A half-span, A midpoint, B half-span, B midpoint), with
    parent phase zero. Positive resultants justify the aligned middle phases;
    the complete principal branches and strictly acute chain gaps are admitted.
    ``potential`` is the full bare potential V=51+F, not F alone. Its gradient
    and Hessian equal those of F. The scalar and fine-edge potentials describe
    the same map; neither their values nor these derivatives certify a minimum.
    """

    parameters: tuple[I, ...]
    resultants: tuple[I, I]
    middle_phases: tuple[I, I]
    potential: I
    gradient: tuple[I, ...]
    hessian: tuple[tuple[I, ...], ...]
    core: PrincipalCore


@dataclass(frozen=True)
class SuccessorReductionEvaluation:
    chamber: object
    reduction: PrincipalReduction


def _box(values, size, radius):
    if type(values) not in (tuple, list) or len(values) != size:
        raise ValueError(f"candidate requires {size} coordinates")
    radius = exact_or_represented_real(radius, "candidate radius")
    if radius < 0:
        raise ValueError("candidate radius must be nonnegative")
    return tuple(
        (
            value
            if isinstance(value, I)
            else I(exact_or_represented_real(value, "coordinate"))
        )
        + I(-radius, radius)
        for value in values
    )


def _admit_model(edges, shifts):
    if (
        type(edges) is not tuple
        or any(
            type(edge) is not tuple or any(type(i) is not int for i in edge)
            for edge in edges
        )
        or edges != SELECTED_SUPPORT
    ):
        raise ValueError("principal reduction requires the complete selected support")
    if (
        type(shifts) is not tuple
        or any(type(k) is not int for k in shifts)
        or shifts != SELECTED_SHIFTS
    ):
        raise ValueError("principal reduction requires the complete selected branches")


def _evaluate_core(coordinates, *, edges, shifts, radius=0):
    """Rebuild every fine observation from admitted coordinates and branches.

    Principal cuts reject. Nonacute chain gaps remain visible through their
    signed margins; this evaluator never upgrades them to PSD or minimum data.
    """
    _admit_model(edges, shifts)
    values = _box(coordinates, 6, radius)
    pi = pi_interval()
    phases = tuple(
        sum((a * x for a, x in zip(row, values, strict=True)), I(0)) + c * pi
        for row, c in zip(CORE_MATRIX, CORE_PI_OFFSETS, strict=True)
    )
    coefficients = tuple(
        tuple(y - x for x, y in zip(CORE_MATRIX[i], CORE_MATRIX[j], strict=True))
        for i, j in edges
    )
    # Form differences directly from the affine rows, preserving common-node
    # cancellation instead of subtracting independently enclosed phase boxes.
    differences = tuple(
        sum((a * x for a, x in zip(row, values, strict=True)), I(0))
        + (CORE_PI_OFFSETS[j] - CORE_PI_OFFSETS[i] - 2 * k) * pi
        for (i, j), k, row in zip(edges, shifts, coefficients, strict=True)
    )
    margins = tuple(pi - abs(value) for value in differences)
    if any(value.lo <= 0 for value in margins):
        raise ValueError("candidate has unresolved or nonpositive principal margin")
    gradient = [I(0)] * 21
    hessian = [[I(0)] * 21 for _ in range(21)]
    core_gradient = [I(0)] * 6
    core_hessian = [[I(0)] * 6 for _ in range(6)]
    potential = I(0)
    for (i, j), row, difference in zip(edges, coefficients, differences, strict=True):
        current, curvature = sin(difference), cos(difference)
        potential += 1 - curvature
        gradient[i] -= current
        gradient[j] += current
        hessian[i][i] += curvature
        hessian[j][j] += curvature
        hessian[i][j] -= curvature
        hessian[j][i] -= curvature
        for left in range(6):
            core_gradient[left] += row[left] * current
            for right in range(6):
                core_hessian[left][right] += row[left] * row[right] * curvature
    gaps = ((2 * pi - values[2]) / 3, (2 * pi + values[3] - values[5]) / 3)
    return PrincipalCore(
        values,
        phases,
        differences,
        margins,
        gaps,
        tuple(pi / 2 - value for value in gaps),
        potential,
        tuple(gradient),
        tuple(map(tuple, hessian)),
        tuple(core_gradient),
        tuple(map(tuple, core_hessian)),
        tuple(hessian[i][i] - hessian[i][j] for i, j in TWIN_PAIRS),
    )


def evaluate_successor_core(graph, coordinates, *, radius=0):
    """Observe a detached core candidate after rebuilding the selected reset."""
    chamber = predict_successor_chamber(graph)
    candidate = _evaluate_core(
        coordinates, edges=chamber.edges, shifts=chamber.ideal.shifts, radius=radius
    )
    return SuccessorCoreEvaluation(chamber, candidate)


def _evaluate_reduction(coordinates, *, edges, shifts, radius=0):
    _admit_model(edges, shifts)
    values = _box(coordinates, 4, radius)
    a, m, b, n = values
    pi = pi_interval()
    gaps = ((2 * pi - 2 * a) / 3, (2 * pi - 2 * b) / 3)
    if any(gap.lo <= 0 or (pi / 2 - gap).lo <= 0 for gap in gaps):
        raise ValueError("minimum reduction requires strictly acute positive chains")
    resultants, middle = [], []
    for half_span, midpoint in ((a, m), (b, n)):
        real = 1 + 4 * cos(half_span) * cos(midpoint)
        imaginary = 4 * cos(half_span) * sin(midpoint)
        magnitude = sqrt(real**2 + imaginary**2)
        if magnitude.lo <= 0:
            raise ValueError("middle resultant is zero or unresolved")
        resultants.append(magnitude)
        middle.append(arg(real, imaginary))
    origin = m - a
    core = _evaluate_core(
        (
            -origin,
            middle[0] - origin,
            2 * a,
            n - b - origin,
            middle[1] - origin,
            n + b - origin,
        ),
        edges=edges,
        shifts=shifts,
    )
    potential = (
        51
        - 4 * cos(a) * cos(m)
        - resultants[0]
        - 12 * cos(gaps[0])
        - 4 * cos(b) * cos(n)
        - 2 * resultants[1]
        - 12 * cos(gaps[1])
        - 4 * cos(m - a - n + b)
    )
    # The aligned middle-coordinate gradients vanish identically on this map.
    # Their positive diagonal Hessian block is diag(R_A, 2 R_B); Q and B1
    # share no edge. Its exact Schur complement retains their relaxation.
    external, eliminated = (0, 2, 3, 5), (1, 4)
    divisors = (resultants[0], 2 * resultants[1])
    schur = tuple(
        tuple(
            core.core_hessian[i][j]
            - sum(
                (
                    core.core_hessian[i][r] * core.core_hessian[r][j] / divisor
                    for r, divisor in zip(eliminated, divisors, strict=True)
                ),
                I(0),
            )
            for j in external
        )
        for i in external
    )
    linear = ((1, -1, 0, 0), (2, 0, 0, 0), (1, -1, -1, 1), (1, -1, 1, 1))
    gradient = tuple(
        sum(
            (
                linear[k][i] * core.core_gradient[node]
                for k, node in enumerate(external)
            ),
            I(0),
        )
        for i in range(4)
    )
    hessian = tuple(
        tuple(
            sum(
                (
                    linear[k][i] * schur[k][column] * linear[column][j]
                    for k in range(4)
                    for column in range(4)
                ),
                I(0),
            )
            for j in range(4)
        )
        for i in range(4)
    )
    return PrincipalReduction(
        values, tuple(resultants), tuple(middle), potential, gradient, hessian, core
    )


def evaluate_successor_reduction(graph, coordinates, *, radius=0):
    """Observe the four-coordinate candidate without searching or certifying it."""
    chamber = predict_successor_chamber(graph)
    reduction = _evaluate_reduction(
        coordinates, edges=chamber.edges, shifts=chamber.ideal.shifts, radius=radius
    )
    return SuccessorReductionEvaluation(chamber, reduction)
