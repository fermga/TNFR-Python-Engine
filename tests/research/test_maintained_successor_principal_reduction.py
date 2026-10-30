"""Static algebra controls for the successor's full principal chamber.

These checks support necessary stationary restrictions, not a classification
from finitely many candidates. Fixed dyadic turns below test the full fine
pullback; exact signed matrices retain singular and negative-curvature cases.
The retained source is read once. No operator, root search or flow is run.
"""

from collections import Counter
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

import mpmath
import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions import Coupling, SelfOrganization
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research.artifact_io import file_receipt
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_successor_chamber import predict_successor_chamber

ROOT = Path(__file__).resolve().parents[2]
LABELS = (0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 1)
MULTIPLICITIES = {
    (0, 1): 2,
    (0, 2): 2,
    (0, 5): 4,
    (0, 6): 4,
    (1, 2): 1,
    (1, 3): 2,
    (1, 6): 2,
    (1, 7): 2,
    (1, 8): 2,
    (2, 3): 2,
    (3, 4): 4,
    (4, 5): 4,
    (6, 7): 4,
    (6, 10): 4,
    (7, 8): 4,
    (8, 9): 4,
    (9, 10): 4,
}
CHAINS = ((3, 4, 5, 0), (8, 9, 10, 6))
CORE = (0, 1, 2, 3, 6, 7, 8)
INTERIOR = (4, 5, 9, 10)


def _laplacian(size, weighted_edges, zero=Q(0)):
    matrix = [[zero for _ in range(size)] for _ in range(size)]
    for left, right, weight in weighted_edges:
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(map(tuple, matrix))


def _block(matrix, rows, columns):
    return tuple(tuple(matrix[i][j] for j in columns) for i in rows)


def _quadratic(matrix, vector):
    return sum(
        vector[i] * entry * vector[j]
        for i, row in enumerate(matrix)
        for j, entry in enumerate(row)
    )


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("principal-reduction controls must not execute an event or flow")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_parent_thol_execution, "execute_network_operator_stage"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (Coupling, "_execute"),
            (SelfOrganization, "_prepare_execution"),
            (SelfOrganization, "_prepare_bifurcation"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def chamber():
    paths = tuple(
        sorted(
            path
            for name in ("maintained_parent_thol", "local_capacity_history")
            for path in (ROOT / "docs/assets" / name).iterdir()
            if path.is_file()
        )
    )
    receipts = tuple(file_receipt(path) for path in paths)
    source = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    )
    signature = _signature(source.graph)
    yield predict_successor_chamber(source.graph)
    assert _signature(source.graph) == signature
    assert tuple(file_receipt(path) for path in paths) == receipts


def test_full_support_derives_the_eleven_group_multiplicities(chamber):
    actual = Counter()
    collapsed = []
    for left, right in chamber.edges:
        a, b = LABELS[left], LABELS[right]
        if a == b:
            collapsed.append((left, right))
        else:
            actual[tuple(sorted((a, b)))] += 1
    assert len(chamber.edges) == 53 and chamber.cycle_rank == 33
    assert actual == MULTIPLICITIES
    assert collapsed == [(0, 1), (2, 20)]
    assert sum(actual.values()) == 51
    shifts = {}
    for (left, right), shift in zip(chamber.edges, chamber.ideal.shifts):
        a, b = LABELS[left], LABELS[right]
        if a != b:
            edge, oriented_shift = tuple(sorted((a, b))), shift if a < b else -shift
            assert shifts.setdefault(edge, oriented_shift) == oriented_shift
    assert {edge for edge, shift in shifts.items() if shift} == {(0, 5), (6, 10)}
    assert all(
        shift == int(edge in {(0, 5), (6, 10)}) for edge, shift in shifts.items()
    )


def test_full_potential_gradient_and_hessian_pull_back_without_pair_dynamics(chamber):
    mp = mpmath.mp.clone()
    mp.dps = 90
    # Fixed auxiliary phases, not a root proposal or an observed reset.
    turns = tuple(map(Q, (0,))) + (
        Q(1, 5),
        Q(7, 32),
        Q(2, 5),
        Q(3, 5),
        Q(4, 5),
        Q(1, 32),
        Q(7, 32),
        Q(13, 32),
        Q(19, 32),
        Q(25, 32),
    )
    reduced = tuple(2 * mp.pi * _mp(mp, value) for value in turns)
    phases = tuple(reduced[label] for label in LABELS)
    fine_gradient = [mp.mpf(0)] * 21
    fine_hessian_edges, fine_potential = [], mp.mpf(0)
    for (left, right), shift in zip(chamber.edges, chamber.ideal.shifts):
        delta = phases[right] - phases[left]
        assert abs(delta - 2 * mp.pi * shift) < mp.pi
        current, cosine = mp.sin(delta), mp.cos(delta)
        fine_gradient[left] -= current
        fine_gradient[right] += current
        fine_hessian_edges.append((left, right, cosine))
        fine_potential += 1 - cosine
    fine_hessian = _laplacian(21, fine_hessian_edges, mp.mpf(0))
    reduced_gradient, reduced_potential = [mp.mpf(0)] * 11, mp.mpf(0)
    reduced_hessian_edges = []
    for (left, right), count in MULTIPLICITIES.items():
        delta = reduced[right] - reduced[left]
        current = count * mp.sin(delta)
        reduced_gradient[left] -= current
        reduced_gradient[right] += current
        reduced_potential += count * (1 - mp.cos(delta))
        reduced_hessian_edges.append((left, right, count * mp.cos(delta)))
    reduced_hessian = _laplacian(11, reduced_hessian_edges, mp.mpf(0))
    tolerance = mp.mpf("1e-80")
    assert abs(fine_potential - reduced_potential) < tolerance
    for a in range(11):
        assert (
            abs(
                sum(fine_gradient[i] for i in range(21) if LABELS[i] == a)
                - reduced_gradient[a]
            )
            < tolerance
        )
        for b in range(11):
            pulled = sum(
                fine_hessian[i][j]
                for i in range(21)
                for j in range(21)
                if LABELS[i] == a and LABELS[j] == b
            )
            assert abs(pulled - reduced_hessian[a][b]) < tolerance
    assert phases[2] != phases[3]


@pytest.mark.parametrize(
    "cosines", ((Q(1, 4), Q(3, 4)), (Q(1), Q(1, 8)), (Q(3, 8), Q(5, 8)))
)
def test_positive_three_edge_chains_have_exact_seven_core_schur_complement(cosines):
    chain_edges = {
        tuple(sorted((left, right))): 4 * cosine
        for chain, cosine in zip(CHAINS, cosines)
        for left, right in zip(chain, chain[1:])
    }
    weights = {
        edge: chain_edges.get(edge, Q(count * (1 + index % 3), 8))
        for index, (edge, count) in enumerate(MULTIPLICITIES.items())
    }
    matrix = _laplacian(11, ((i, j, weight) for (i, j), weight in weights.items()))
    interior = _block(matrix, INTERIOR, INTERIOR)
    assert exact_symmetric_semidefinite(interior, strict=True)
    inverse = exact_matrix_inverse(interior)
    cross = _block(matrix, CORE, INTERIOR)
    correction = exact_matrix_product(
        exact_matrix_product(cross, inverse), _block(matrix, INTERIOR, CORE)
    )
    core = _block(matrix, CORE, CORE)
    schur = tuple(
        tuple(a - b for a, b in zip(row, correction[i])) for i, row in enumerate(core)
    )
    positions = {node: index for index, node in enumerate(CORE)}
    expected_edges = [
        (positions[i], positions[j], weight)
        for (i, j), weight in weights.items()
        if (i, j) not in chain_edges
    ]
    expected_edges.extend(
        (positions[chain[0]], positions[chain[-1]], 4 * cosine / 3)
        for chain, cosine in zip(CHAINS, cosines)
    )
    assert schur == _laplacian(7, expected_edges)
    # Independent energy minimization gives linear interpolation along each
    # equal-conductance chain, not deletion of its endpoint curvature.
    vector = [Q(index - 3, 8) for index in range(11)]
    for chain in CHAINS:
        for index, node in enumerate(chain[1:-1], start=1):
            vector[node] = (
                (3 - index) * vector[chain[0]] + index * vector[chain[-1]]
            ) / 3
    assert _quadratic(matrix, vector) == _quadratic(
        schur, tuple(vector[i] for i in CORE)
    )


@pytest.mark.parametrize("signs", tuple(product((-1, 1), repeat=3)))
def test_stationary_chain_equal_sines_exclude_every_negative_cosine_case(signs):
    full = _laplacian(4, ((i, i + 1, Q(3 * sign, 2)) for i, sign in enumerate(signs)))
    interior = _block(full, (1, 2), (1, 2))
    assert exact_symmetric_semidefinite(interior) == (signs == (1, 1, 1))
    if signs != (1, 1, 1):
        assert any(
            _quadratic(interior, vector) < 0
            for vector in product(range(-2, 3), repeat=2)
        )


def test_zero_chain_is_kept_as_a_singular_zero_row_case():
    zero = ((Q(0), Q(0)), (Q(0), Q(0)))
    assert exact_symmetric_semidefinite(zero)
    assert not exact_symmetric_semidefinite(zero, strict=True)
    with pytest.raises(ValueError, match="singular"):
        exact_matrix_inverse(zero)
    # A PSD zero diagonal requires an entire zero row, independently of the
    # other diagonal. A nonzero coupling cannot be hidden by eliminating it.
    for diagonal in (Q(0), Q(1), Q(128)):
        assert not exact_symmetric_semidefinite(((Q(0), Q(1, 8)), (Q(1, 8), diagonal)))
    # With zero cosines and equal stationary sines, all three principal gaps
    # are the same signed quarter turn; both signs remain in the algebra.
    for signs in product((-1, 1), repeat=3):
        stationary = signs[0] == signs[1] == signs[2]
        assert stationary == (
            tuple(Q(sign, 4) for sign in signs) in ((Q(1, 4),) * 3, (Q(-1, 4),) * 3)
        )


def test_live_support_has_only_the_declared_twin_pairs_and_leaf(chamber):
    neighbors = [set() for _ in range(21)]
    for left, right in chamber.edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    twins = {
        (i, j)
        for i in range(21)
        for j in range(i + 1, 21)
        if neighbors[i] - {j} == neighbors[j] - {i}
    }
    assert twins == {(i, i + 1) for i in (0, 4, 6, 8, 10, 12, 14, 16, 18)}
    assert neighbors[20] == {2}
    assert (2, 3) not in twins
    assert {i for i, row in enumerate(neighbors) if len(row) % 2} == {
        0,
        1,
        3,
        10,
        11,
        12,
        13,
        14,
        15,
        20,
    }
    # Zero PSD rows imply quarter-turn incident gaps. An odd number of their
    # signed unit sines cannot satisfy the nodal stationarity equation.
    for degree in {len(row) for row in neighbors if len(row) % 2}:
        assert all(sum(signs) != 0 for signs in product((-1, 1), repeat=degree))


def test_zero_resultant_antipodal_twins_need_different_edge_branches():
    turns = (Q(1, 4), Q(-1, 4), Q(0), Q(1, 2))
    # Twins 0,1 have opposite quadrature neighbors and zero resultants.
    edges = ((0, 2), (1, 2), (0, 3), (1, 3))
    gaps = tuple(turns[j] - turns[i] for i, j in edges)
    cardinal_sine = {Q(1, 4): Q(1), Q(3, 4): Q(-1)}
    currents = [Q(0)] * 4
    for (left, right), gap in zip(edges, gaps):
        current = cardinal_sine[gap % 1]
        currents[left] += current
        currents[right] -= current
    assert currents == [0] * 4
    # Derive both components of the common neighbor resultant at the exact
    # cardinal turns; the zero-resultant example is not merely a label.
    cardinal_cosine = {Q(0): Q(1), Q(1, 4): Q(0), Q(1, 2): Q(-1), Q(3, 4): Q(0)}
    full_sine = {Q(0): Q(0), Q(1, 2): Q(0), **cardinal_sine}
    assert sum(cardinal_cosine[turns[i] % 1] for i in (2, 3)) == 0
    assert sum(full_sine[turns[i] % 1] for i in (2, 3)) == 0
    hessian = _laplacian(
        4, ((i, j, cardinal_cosine[(turns[j] - turns[i]) % 1]) for i, j in edges)
    )
    assert exact_symmetric_semidefinite(hessian)
    assert not exact_symmetric_semidefinite(hessian, strict=True)
    branches = tuple(
        (gap + Q(1, 2)).numerator // (gap + Q(1, 2)).denominator for gap in gaps
    )
    assert branches == (0, 0, 0, 1)
    assert all(abs(gap - branch) < Q(1, 2) for gap, branch in zip(gaps, branches))
    # In the same-branch intersection, only the common midpoint neighbor is
    # perpendicular to both twins; the antipodal neighbor is outside it.
    lower, upper = max(turns[:2]) - Q(1, 2), min(turns[:2]) + Q(1, 2)
    assert lower < turns[2] < upper
    assert not lower < turns[3] < upper
    assert tuple(
        value for value in (Q(0), Q(1, 2), Q(-1, 2)) if lower < value < upper
    ) == (Q(0),)


def test_adjacent_twin_stationary_split_has_a_negative_fine_direction():
    # Triangle phases (+pi/3,-pi/3,pi) have zero sine current and cosine
    # -1/2 on every edge. The antisymmetric twin direction remains negative.
    turns = (Q(1, 6), Q(-1, 6), Q(1, 2))
    sine_coefficients = {Q(1, 3): Q(1), Q(2, 3): Q(-1)}
    currents = [Q(0)] * 3
    for left, right in ((0, 1), (0, 2), (1, 2)):
        coefficient = sine_coefficients[(turns[right] - turns[left]) % 1]
        currents[left] += coefficient
        currents[right] -= coefficient
    assert currents == [0] * 3  # Coefficients of sqrt(3)/2 vanish exactly.
    matrix = _laplacian(3, ((0, 1, Q(-1, 2)), (0, 2, Q(-1, 2)), (1, 2, Q(-1, 2))))
    assert _quadratic(matrix, (Q(1), Q(-1), Q(0))) == -3
    assert not exact_symmetric_semidefinite(matrix)


def test_one_negative_edge_does_not_imply_a_negative_full_direction():
    # This is a signed-Laplacian countercontrol, not a proposed stationary
    # phase assignment. Positive parallel routes can dominate a negative edge.
    matrix = _laplacian(3, ((0, 1, Q(-1, 4)), (0, 2, Q(1)), (1, 2, Q(1))))
    assert exact_symmetric_semidefinite(matrix)
    assert exact_symmetric_semidefinite(_block(matrix, (1, 2), (1, 2)), strict=True)


def test_actual_weighted_mean_centering_preserves_every_curvature_direction(chamber):
    degrees = [0] * 21
    for i, j in chamber.edges:
        degrees[i] += 1
        degrees[j] += 1
    # The ideal capacity blend is exact rational arithmetic even though its
    # carrier type is an interval. No midpoint replaces an uncertain value.
    assert all(value.lo == value.hi > 0 for value in chamber.reset.capacity_after)
    rho = tuple(
        Q(degree) / capacity.lo
        for degree, capacity in zip(degrees, chamber.reset.capacity_after)
    )
    vector = tuple(Q((i * 7) % 13 - 6, 16) for i in range(21))
    offset = sum(weight * value for weight, value in zip(rho, vector)) / sum(rho)
    relative = tuple(value - offset for value in vector)
    assert sum(weight * value for weight, value in zip(rho, relative)) == 0
    assert sum(relative) != 0
    # Signed curvature is allowed: this identity is not limited to minima.
    weights = tuple(Q(-1 if index % 3 == 0 else 1, 2) for index in range(53))
    matrix = _laplacian(
        21, ((i, j, value) for (i, j), value in zip(chamber.edges, weights))
    )
    assert all(sum(row) == 0 for row in matrix)
    assert _quadratic(matrix, vector) == _quadratic(matrix, relative)


def test_synchronized_zero_winding_is_a_minimum_in_a_different_branch_class(chamber):
    phase = _laplacian(21, ((i, j, Q(1)) for i, j in chamber.edges))
    # Positive interval lower conductances supply a valid lower form Hessian
    # for every actual admitted conductance, without treating midpoints as law.
    form = _laplacian(
        21,
        (
            (i, j, value.lo)
            for (i, j), value in zip(chamber.edges, chamber.conductances)
        ),
    )
    for matrix in (phase, form):
        assert exact_symmetric_semidefinite(matrix)
        assert not exact_symmetric_semidefinite(matrix, strict=True)
        assert exact_symmetric_semidefinite(
            _block(matrix, range(1, 21), range(1, 21)), strict=True
        )
    assert any(chamber.ideal.shifts)
    # Equal lifts have zero sine current, but retained nonzero branches would
    # give gaps of +/-2*pi, outside this study's strict principal chamber.
    assert any(abs(-2 * branch) >= 1 for branch in chamber.ideal.shifts)
