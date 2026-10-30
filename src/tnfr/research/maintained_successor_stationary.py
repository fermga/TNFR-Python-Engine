"""Closed-acute stationary obstruction for the proposed maintained successor.

A five-coordinate affine face is a computational witness, not a dynamical
reduction. A fixed rational center and preconditioner are independently checked
by outward interval contraction. The resulting full twenty-one-node KKT
witness excludes every stationary point in the declared closed acute chamber.
No operator, continuation or numerical root search is performed here.

The ordinary stationary kernel cannot certify this witness: its full gradient
is deliberately nonzero. The contracted equation is the gradient restricted to
an affine boundary face, with exact pi-dependent edge constraints retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from .maintained_successor_chamber import predict_successor_chamber

# A numerical static calculation proposed these dyadic data before the exact
# checker. Their provenance is not a premise: the residual and inverse defect
# below independently admit them. No observed UM response enters this witness.
_SCALE = 1 << 40
FACE_CENTER = tuple(
    Q(value, _SCALE)
    for value in (
        1082587343106,
        1211889604568,
        1438506301057,
        113581483716,
        891083831083,
    )
)
FACE_INVERSE = tuple(
    tuple(Q(value, _SCALE) for value in row)
    for row in (
        (287272327980, 150759190402, -50578160076, 149046851150, 36046380832),
        (150759190402, 571512378342, -101438144287, 78219098823, 18916974111),
        (-50578160076, -101438144287, 110330151560, -26241704341, -6346450536),
        (149046851150, 78219098823, -26241704341, 210301239846, -15973893331),
        (36046380832, 18916974111, -6346450536, -15973893331, 157291692605),
    )
)
FACE_RADIUS = Q(1, 1 << 24)


def _face_embedding():
    """Return theta=A*y+c*pi for y=(parent, sibling, A-gap, B-origin, B-gap)."""
    matrix = [[0] * 5 for _ in range(21)]
    offsets = [Q(0) for _ in range(21)]
    matrix[2][0] = matrix[20][0] = matrix[3][1] = 1
    for first, gaps in ((4, 3), (6, 2), (8, 1)):
        for node in (first, first + 1):
            matrix[node][2], offsets[node] = -gaps, Q(2)
    for first, offset in ((10, 0), (12, 0), (14, Q(1, 2)), (16, 1), (18, Q(3, 2))):
        for node in (first, first + 1):
            matrix[node][3], offsets[node] = 1, Q(offset)
    matrix[12][4] = matrix[13][4] = 1
    return tuple(map(tuple, matrix)), tuple(offsets)


PHASE_MATRIX, PHASE_PI_COEFFICIENTS = _face_embedding()


def _selected_support():
    """Symbolic actual support; checked against a freshly rebuilt UM result."""
    edges = []
    for origin in (0, 10):
        for pair in range(5):
            for member in range(2):
                left = origin + 2 * pair + member
                for next_member in range(2):
                    right = origin + 2 * ((pair + 1) % 5) + next_member
                    edges.append(tuple(sorted((left, right))))
    edges.extend(
        (
            (0, 1),
            (0, 10),
            (0, 11),
            (1, 10),
            (1, 11),
            (2, 3),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 20),
        )
    )
    return tuple(sorted(edges))


SELECTED_SUPPORT = _selected_support()
SELECTED_SHIFTS = tuple(
    int(
        (left, right)
        in {
            (0, 8),
            (0, 9),
            (1, 8),
            (1, 9),
            (10, 18),
            (10, 19),
            (11, 18),
            (11, 19),
        }
    )
    for left, right in SELECTED_SUPPORT
)


@dataclass(frozen=True)
class AffineFaceCertificate:
    """Checked restricted root and full-state KKT data, not an equilibrium.

    ``full_gradient_bounds`` are direct bounds over the entire parameter box.
    Exact vanishing of the other entries follows only at its included root,
    from the affine equations and the analytical cancellation identities.
    Active gaps are exact pi-affine identities; their acute margins are zero.
    """

    center: tuple[Q, ...]
    radius: Q
    inverse: tuple[tuple[Q, ...], ...]
    parameters: tuple[I, ...]
    phases: tuple[I, ...]
    edge_coefficients: tuple[tuple[int, ...], ...]
    edge_pi_coefficients: tuple[Q, ...]
    edge_differences: tuple[I, ...]
    active_edge_indices: tuple[int, ...]
    acute_margins: tuple[I, ...]
    minimum_nonactive_margin: Q
    full_gradient_bounds: tuple[I, ...]
    multiplier: I
    residual: tuple[I, ...]
    jacobian_bounds: tuple[tuple[I, ...], ...]
    preconditioned_residual: tuple[I, ...]
    contraction_matrix: tuple[tuple[I, ...], ...]
    eta: Q
    q: Q
    image_radius_bound: Q
    root_inclusion_certified: bool
    kkt_certified: bool
    closed_stationary_excluded: bool


@dataclass(frozen=True)
class SuccessorStationaryCertificate:
    """Primitive-source association and a conditional mathematical exclusion."""

    chamber: object
    face: AffineFaceCertificate
    source_form_storage: I
    conditional_source_form_storage: I


def _vector(values, label, size):
    if type(values) not in (tuple, list) or len(values) != size:
        raise ValueError(f"{label} requires {size} coordinates")
    return tuple(
        exact_or_represented_real(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def _affine_values(matrix, offsets, parameters):
    p = pi_interval()
    return tuple(
        sum((coefficient * value for coefficient, value in zip(row, parameters)), I(0))
        + offset * p
        for row, offset in zip(matrix, offsets, strict=True)
    )


def _gradient_and_hessian(coefficients, differences):
    gradient = [I(0) for _ in range(5)]
    hessian = [[I(0) for _ in range(5)] for _ in range(5)]
    for row, difference in zip(coefficients, differences, strict=True):
        sine, cosine = sin(difference), cos(difference)
        for i, left in enumerate(row):
            gradient[i] += left * sine
            for j, right in enumerate(row):
                hessian[i][j] += left * right * cosine
    return tuple(gradient), tuple(map(tuple, hessian))


def _certify_face(
    *, edges, shifts, center=FACE_CENTER, radius=FACE_RADIUS, inverse=FACE_INVERSE
):
    """Verify a supplied rational witness on precisely the selected support.

    Wrong support/branches and invalid scalar domains reject. A finite but poor
    proposed center or inverse yields failed inclusion; a positive result does
    not follow from numerical solver status or a cached certificate.
    """
    if (
        type(edges) is not tuple
        or any(
            type(edge) is not tuple or any(type(i) is not int for i in edge)
            for edge in edges
        )
        or edges != SELECTED_SUPPORT
    ):
        raise ValueError("face proof requires the complete selected support")
    if (
        type(shifts) is not tuple
        or any(type(shift) is not int for shift in shifts)
        or shifts != SELECTED_SHIFTS
    ):
        raise ValueError("face proof requires the complete selected branch vector")
    center = _vector(center, "face center", 5)
    radius = exact_or_represented_real(radius, "face radius")
    if radius <= 0:
        raise ValueError("face radius must be strictly positive")
    if type(inverse) not in (tuple, list) or len(inverse) != 5:
        raise ValueError("face inverse requires five rows")
    inverse = tuple(
        _vector(row, f"face inverse[{i}]", 5) for i, row in enumerate(inverse)
    )
    coefficients = tuple(
        tuple(right - left for left, right in zip(PHASE_MATRIX[i], PHASE_MATRIX[j]))
        for i, j in edges
    )
    offsets = tuple(
        PHASE_PI_COEFFICIENTS[j] - PHASE_PI_COEFFICIENTS[i] - 2 * shift
        for (i, j), shift in zip(edges, shifts, strict=True)
    )
    parameters = tuple(I(value - radius, value + radius) for value in center)
    phases = _affine_values(PHASE_MATRIX, PHASE_PI_COEFFICIENTS, parameters)
    differences = _affine_values(coefficients, offsets, parameters)
    center_differences = _affine_values(coefficients, offsets, tuple(map(I, center)))
    residual, _ = _gradient_and_hessian(coefficients, center_differences)
    _, jacobian = _gradient_and_hessian(coefficients, differences)
    preconditioned = tuple(
        sum((value * component for value, component in zip(row, residual)), I(0))
        for row in inverse
    )
    contraction = tuple(
        tuple(
            I(int(i == j))
            - sum((inverse[i][k] * jacobian[k][j] for k in range(5)), I(0))
            for j in range(5)
        )
        for i in range(5)
    )
    eta = max(value.abs_max for value in preconditioned)
    q = max(sum(value.abs_max for value in row) for row in contraction)
    image_radius = eta + q * radius
    inclusion = q < 1 and image_radius < radius
    active = tuple(
        index
        for index, (row, offset) in enumerate(zip(coefficients, offsets, strict=True))
        if not any(row) and abs(offset) == Q(1, 2)
    )
    margins = tuple(
        I(0) if index in active else pi_interval() / 2 - abs(value)
        for index, value in enumerate(differences)
    )
    minimum_margin = min(
        value.lo for index, value in enumerate(margins) if index not in active
    )
    gradient = [I(0) for _ in range(21)]
    for (left, right), difference in zip(edges, differences, strict=True):
        value = sin(difference)
        gradient[left] -= value
        gradient[right] += value
    parent, _, _, origin, gap = parameters
    multiplier = 2 * cos(gap) - 2 + cos(parent - origin)
    kkt = inclusion and len(active) == 12 and minimum_margin > 0 and multiplier.lo > 0
    return AffineFaceCertificate(
        center,
        radius,
        inverse,
        parameters,
        phases,
        coefficients,
        offsets,
        differences,
        active,
        margins,
        minimum_margin,
        tuple(gradient),
        multiplier,
        residual,
        jacobian,
        preconditioned,
        contraction,
        eta,
        q,
        image_radius,
        inclusion,
        kkt,
        kkt,
    )


def certify_successor_stationary(graph):
    """Rebuild state, ideal reset and whole chamber before admitting the proof.

    The ideal and conditional numerical-envelope resets have the same branches
    and positive law parameters. Their common bare phase support therefore has
    this same obstruction. The envelope is not an observed execution error.
    """
    chamber = predict_successor_chamber(graph)
    if chamber.ideal.shifts != chamber.conditional.shifts:
        raise ValueError("the conditional reset has a different phase chamber")
    face = _certify_face(edges=chamber.edges, shifts=chamber.ideal.shifts)
    epi = chamber.reset.epi

    def storage(conductances):
        return sum(
            (
                weight * (epi[left] - epi[right]) ** 2 / 2
                for (left, right), weight in zip(
                    chamber.edges, conductances, strict=True
                )
            ),
            I(0),
        )

    return SuccessorStationaryCertificate(
        chamber,
        face,
        storage(chamber.conductances),
        storage(chamber.conditional_conductances),
    )
