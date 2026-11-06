"""Static controls for frozen contact stationary and energy geometry.

SINE_CONTACT_STATIONARY_GEOMETRY section 1 fixes the 20-node, 53-edge
support and every marked branch before these checks. No event, root search
or trajectory is run. Auxiliary conductances cycle through 3/8,...,7/8;
they exercise the same stationary reduction but are not regenerated UM data.
Algebra probes use 100-digit arithmetic, dyadic member offsets, and the two
declared reflected receiver angles pi/3 and 5*pi/12. They test identities,
not a finite substitute for the chamber-wide analytical exclusion.

Section 8 separately fixes the nonacute minimum question and its auxiliary
quadrature, split-twin, receiver and signed-chain controls before evaluation.
They retain degeneracies and never treat a semidefinite Hessian as a minimum.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath
import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I, cos, sin, sqrt
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange

_MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
_RECEIVERS = (10, 11, 12, 13, 18, 19)
_TOLERANCE = Q(1, 2**46)


def _marked_edges():
    """Retain the protocol's orientations, rather than sorted-edge branches."""
    result = []
    for offset in (0, 10):
        for pair in range(5):
            following = (pair + 1) % 5
            for member in (0, 1):
                for neighbor in (0, 1):
                    result.append(
                        (
                            offset + 2 * pair + member,
                            offset + 2 * following + neighbor,
                            -int(pair == 4),
                        )
                    )
    result.append((0, 1, 0))
    result.extend(
        (left, right, int(right >= 18)) for left in (0, 1) for right in _RECEIVERS
    )
    return tuple(result)


_MARKED = _marked_edges()
_EDGES = tuple((left, right) for left, right, _ in _MARKED)
_BRANCHES = tuple(branch for _, _, branch in _MARKED)


def _graph(phases, *, nonconstant_form=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    for index, (left, right) in enumerate(_EDGES):
        graph.add_edge(left, right, weight=Q(3 + index % 5, 8))
    for node, phase in enumerate(phases):
        graph.nodes[node].update(
            EPI=Q(5, 16) + (Q(node, 128) if nonconstant_form else 0),
            theta=float(phase),
            nu_f=Q(1, 2) if node in (0, 1) else Q(1),
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _currents(mp, phases, edges=_EDGES):
    result = [mp.mpf(0) for _ in phases]
    for left, right in edges:
        current = mp.sin(phases[right] - phases[left])
        result[left] += current
        result[right] -= current
    return tuple(result)


def _phase_potential(mp, phases, edges):
    return sum(1 - mp.cos(phases[right] - phases[left]) for left, right in edges)


def _phase_hessian(mp, phases, edges):
    matrix = [[mp.mpf(0) for _ in phases] for _ in phases]
    for left, right in edges:
        weight = mp.cos(phases[right] - phases[left])
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(tuple(row) for row in matrix)


def _quadratic(matrix, vector):
    return sum(
        left * entry * right
        for left, row in zip(vector, matrix)
        for entry, right in zip(row, vector)
    )


def _cycle_row(cycle):
    positions = {edge: index for index, edge in enumerate(_EDGES)}
    row = [0] * len(_EDGES)
    for left, right in zip(cycle, cycle[1:] + cycle[:1]):
        if (left, right) in positions:
            row[positions[left, right]] += 1
        else:
            row[positions[right, left]] -= 1
    return tuple(row)


def _laplacian(graph, *, weighted):
    matrix = [[Q(0) for _ in graph] for _ in graph]
    for left, right, data in graph.edges(data=True):
        weight = data["weight"] if weighted else Q(1)
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(tuple(row) for row in matrix)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module", autouse=True)
def no_response_or_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("the stationary geometry protocol permits no evolution")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


def test_full_support_has_34_independent_marked_cycles():
    graph = _graph((0,) * 20)
    assert len(graph) == 20
    assert len(_EDGES) == graph.number_of_edges() == 53
    assert nx.is_connected(graph)
    assert tuple(dict(graph.degree()).values()) == tuple(
        11 if node in (0, 1) else 6 if node in _RECEIVERS else 4 for node in range(20)
    )
    incidence = tuple(
        tuple(int(node == right) - int(node == left) for left, right in _EDGES)
        for node in graph
    )
    assert exact_rank(incidence) == 19
    rows = tuple(_cycle_row(cycle) for cycle in nx.cycle_basis(graph, root=0))
    assert len(rows) == exact_rank(rows) == 34
    for row in rows:
        assert all(sum(a * b for a, b in zip(row, node)) == 0 for node in incidence)

    # Two old winding observations leave 32 independent cycle directions.
    old_cycles = tuple(
        _cycle_row(list(range(offset, offset + 10, 2))) for offset in (0, 10)
    )
    assert exact_rank(old_cycles) == 2
    assert exact_rank(rows + old_cycles) == 34


def test_exact_turn_branches_retain_the_complete_acute_reference():
    turns = tuple(Q((node % 10) // 2, 5) for node in range(20))
    gaps = tuple(turns[right] - turns[left] - branch for left, right, branch in _MARKED)
    assert all(-Q(1, 4) < gap < Q(1, 4) for gap in gaps)
    graph = _graph((0,) * 20)
    cycles = tuple(nx.cycle_basis(graph, root=0))
    for cycle in cycles:
        row = _cycle_row(cycle)
        winding = sum(value * gap for value, gap in zip(row, gaps))
        assert winding.denominator == 1
        assert winding == -sum(value * branch for value, branch in zip(row, _BRANCHES))
    for offset in (0, 10):
        old = _cycle_row(list(range(offset, offset + 10, 2)))
        assert sum(value * gap for value, gap in zip(old, gaps)) == 1
    contact_triangle = _cycle_row([0, 10, 12])
    assert sum(value * gap for value, gap in zip(contact_triangle, gaps)) == 0


def test_full_fine_twin_difference_has_strict_acute_monotonicity(mp):
    phases = tuple(
        2 * mp.pi * ((node % 10) // 2) / 5 + _mp(mp, Q(node % 7 - 3, 512))
        for node in range(20)
    )
    graph = _graph(phases)
    report = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    captured = tuple(_mp(mp, phase) for phase in report.phase)
    currents = _currents(mp, captured)
    for index, current in enumerate(currents):
        assert _mp(mp, report.relative_resultant[index][1].lo) <= current
        assert current <= _mp(mp, report.relative_resultant[index][1].hi)

    for left in range(0, 20, 2):
        right = left + 1
        external = set(graph[left]) - {right}
        assert external == set(graph[right]) - {left}
        mean = (captured[left] + captured[right]) / 2
        split = (captured[left] - captured[right]) / 2
        factor = sum(mp.cos(captured[node] - mean) for node in external)
        if graph.has_edge(left, right):
            factor += 2 * mp.cos(split)
        expected = -2 * mp.sin(split) * factor
        assert factor > 0
        assert abs(currents[left] - currents[right] - expected) < mp.mpf("1e-90")
        assert (captured[left] - captured[right]) * (
            currents[left] - currents[right]
        ) < 0
    # This concerns bare phase stationarity; arbitrary conductance rows above
    # are retained and are not assumed to define a dynamical pair quotient.


def test_acute_gradient_monotonicity_and_phase_reflection_are_full_fine(mp):
    reference = tuple(2 * mp.pi * ((node % 10) // 2) / 5 for node in range(20))
    first = tuple(
        value + _mp(mp, Q(node % 7 - 3, 512)) for node, value in enumerate(reference)
    )
    second = tuple(
        value + _mp(mp, Q(node % 5 - 2, 1024)) for node, value in enumerate(reference)
    )
    for phases in (first, second):
        assert all(
            abs(phases[right] - phases[left] - 2 * mp.pi * branch) < mp.pi / 2
            for left, right, branch in _MARKED
        )
    change = tuple(after - before for before, after in zip(first, second))
    first_current, second_current = _currents(mp, first), _currents(mp, second)
    nodal = sum(
        delta * (after - before)
        for delta, before, after in zip(change, first_current, second_current)
    )
    edge = -sum(
        (change[right] - change[left])
        * (mp.sin(second[right] - second[left]) - mp.sin(first[right] - first[left]))
        for left, right in _EDGES
    )
    assert abs(nodal - edge) < mp.mpf("1e-90")
    assert nodal < 0

    reflection = tuple(
        (node // 10) * 10 + 2 * (-((node % 10) // 2) % 5) + node % 2
        for node in range(20)
    )
    reflected = tuple(
        -first[reflection[node]] + (2 * mp.pi if (node % 10) // 2 else 0)
        for node in range(20)
    )
    assert all(
        abs(reflected[right] - reflected[left] - 2 * mp.pi * branch) < mp.pi / 2
        for left, right, branch in _MARKED
    )
    reflected_current = _currents(mp, reflected)
    assert all(
        abs(value + first_current[reflection[node]]) < mp.mpf("1e-90")
        for node, value in enumerate(reflected_current)
    )


@pytest.mark.parametrize("receiver_turn", (Q(1, 6), Q(5, 24)))
def test_independent_fine_rows_recover_the_impossible_receiver_balance(
    mp, receiver_turn
):
    angle = 2 * mp.pi * _mp(mp, receiver_turn)
    return_gap = (2 * mp.pi - 2 * angle) / 3
    pair_phases = tuple(2 * mp.pi * pair / 5 for pair in range(5)) + (
        mp.mpf(0),
        angle,
        angle + return_gap,
        angle + 2 * return_gap,
        2 * mp.pi - angle,
    )
    phases = tuple(value for value in pair_phases for _ in (0, 1))
    assert mp.pi / 4 < angle < mp.pi / 2
    assert 0 < return_gap < mp.pi / 2
    assert all(
        abs(phases[right] - phases[left] - 2 * mp.pi * branch) < mp.pi / 2
        for left, right, branch in _MARKED
    )
    currents = _currents(mp, phases)
    receiver_defect = 2 * (mp.sin(return_gap) - 2 * mp.sin(angle))
    for node, current in enumerate(currents):
        expected = (
            receiver_defect
            if node in (12, 13)
            else -receiver_defect if node in (18, 19) else 0
        )
        assert abs(current - expected) < mp.mpf("1e-90")
    # Strict acute winding requires angle > pi/4; its required current
    # exceeds the maximum return-path sine. This exact bound has no probe fit.
    assert sqrt(I(2)).lo - 1 > Q(2, 5)
    assert receiver_defect < -2 * (mp.sqrt(2) - 1)
    report = bound_relational_sine_support_exchange(
        _graph(phases), reference_model=_MODEL
    )
    expected_rate = receiver_defect / (6 * mp.pi)
    assert abs(_mp(mp, report.form_rates[12].midpoint) - expected_rate) < _mp(
        mp, _TOLERANCE
    )
    assert report.form_rates[12].hi < 0
    assert report.phase_rate_numerators() == (Q(0),) * 20


def test_zero_winding_equilibrium_exists_on_same_weighted_support():
    graph = _graph((0,) * 20)
    report = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    assert report.form_gradient == (Q(0),) * 20
    assert report.form_rates == report.phase_rates == (I(0),) * 20
    assert report.storage == I(0)
    for weighted in (False, True):
        laplacian = _laplacian(graph, weighted=weighted)
        assert exact_rank(laplacian) == 19
        assert exact_symmetric_semidefinite(
            tuple(row[1:] for row in laplacian[1:]), strict=True
        )
    # The constant phase has zero cycle windings and fails the original
    # closing-edge branches, although its own zero-branch chamber is acute.
    assert any(abs(Q(branch)) >= Q(1, 4) for branch in _BRANCHES)

    nonconstant = _graph((0,) * 20, nonconstant_form=True)
    changed = bound_relational_sine_support_exchange(
        nonconstant, reference_model=_MODEL
    )
    assert changed.form_rates == (I(0),) * 20
    assert any(value != 0 for value in changed.phase_rate_numerators())
    quadratic = sum(
        value * gradient for value, gradient in zip(changed.epi, changed.form_gradient)
    )
    independent = sum(
        weight * (changed.epi[left] - changed.epi[right]) ** 2
        for (left, right), weight in zip(changed.edges, changed.conductances)
    )
    assert quadratic == independent == 2 * changed.form_storage > 0


def test_disconnected_winding_twists_are_stationary_but_contact_changes_rows(mp):
    phases = tuple(2 * mp.pi * ((node % 10) // 2) / 5 for node in range(20))
    old_edges = _EDGES[:40]
    disconnected = nx.Graph()
    disconnected.add_nodes_from(range(20))
    disconnected.add_edges_from(old_edges)
    assert nx.number_connected_components(disconnected) == 2
    old = _currents(mp, phases, old_edges)
    assert max(map(abs, old)) < mp.mpf("1e-90")
    assert mp.cos(2 * mp.pi / 5) > 0
    joined = _currents(mp, phases)
    strength = 2 * mp.sin(2 * mp.pi / 5)
    for node, value in enumerate(joined):
        expected = (
            -strength if node in (12, 13) else strength if node in (18, 19) else 0
        )
        assert abs(value - expected) < mp.mpf("1e-90")


def test_quadrature_zero_resultant_fiber_is_not_a_local_minimum(mp):
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    phases = tuple(node * mp.pi / 2 for node in range(4))
    assert max(map(abs, _currents(mp, phases, edges))) < mp.mpf("1e-90")
    hessian = _phase_hessian(mp, phases, edges)
    assert max(abs(value) for row in hessian for value in row) < mp.mpf("1e-90")
    # Exact quarter-turn cosines are zero. The admitted algebraic Hessian
    # is PSD, but supplies no strict-curvature or local-minimum certificate.
    exact_hessian = tuple((Q(0),) * 4 for _ in range(4))
    assert exact_symmetric_semidefinite(exact_hessian)
    assert not exact_symmetric_semidefinite(exact_hessian, strict=True)

    increment = mp.mpf(1) / 64
    fiber = (increment,) + phases[1:]
    baseline = _phase_potential(mp, phases, edges)
    assert abs(_phase_potential(mp, fiber, edges) - baseline) < mp.mpf("1e-90")
    fiber_currents = _currents(mp, fiber, edges)
    assert abs(fiber_currents[1] - (1 - mp.cos(increment))) < mp.mpf("1e-90")
    assert fiber_currents[1] > 0
    lowered = (increment, mp.pi / 2 + increment / 2) + phases[2:]
    difference = _phase_potential(mp, lowered, edges) - baseline
    identity = 2 * mp.sin(increment / 2) * (mp.cos(increment / 2) - 1)
    assert abs(difference - identity) < mp.mpf("1e-90")
    assert difference < 0
    branches = (0, 0, 0, -1)
    assert all(
        abs(lowered[right] - lowered[left] - 2 * mp.pi * branch) < mp.pi
        for (left, right), branch in zip(edges, branches)
    )


def test_quadrature_antipodal_twins_need_different_fine_branch_assignments(mp):
    # The same quadrature C4, relabeled as two false twins and their common
    # neighbors. This auxiliary stationary PSD state is not in the contact
    # blocks' equal-branch class; it is not an excluded generic possibility.
    turns = (Q(1, 4), Q(0), -Q(1, 4), Q(1, 2))
    edges = ((0, 1), (2, 1), (0, 3), (2, 3))
    phases = tuple(2 * mp.pi * _mp(mp, turn) for turn in turns)
    assert max(map(abs, _currents(mp, phases, edges))) < mp.mpf("1e-90")
    raw_gaps = tuple(turns[right] - turns[left] for left, right in edges)
    branches = tuple(
        (gap + Q(1, 2)).numerator // (gap + Q(1, 2)).denominator for gap in raw_gaps
    )
    principal = tuple(gap - branch for gap, branch in zip(raw_gaps, branches))
    assert all(abs(gap) < Q(1, 2) for gap in principal)
    assert branches[:2] == (0, 0)
    assert branches[2] != branches[3]
    lower = max(turns[0], turns[2]) - Q(1, 2)
    upper = min(turns[0], turns[2]) + Q(1, 2)
    assert lower < turns[1] < upper
    assert not lower < turns[3] < upper


def test_stationary_split_adjacent_twins_have_negative_diagonal_curvature(mp):
    split = mp.pi / 3
    phases = (split, -split, mp.pi)
    edges = ((0, 1), (0, 2), (1, 2))
    assert max(map(abs, _currents(mp, phases, edges))) < mp.mpf("1e-90")
    external_projection = mp.cos(phases[2])
    assert abs(external_projection + 2 * mp.cos(split)) < mp.mpf("1e-90")
    hessian = _phase_hessian(mp, phases, edges)
    exact = tuple(
        tuple(Q(-1) if row == column else Q(1, 2) for column in range(3))
        for row in range(3)
    )
    for actual, expected in zip(hessian, exact):
        assert all(
            abs(value - _mp(mp, wanted)) < mp.mpf("1e-90")
            for value, wanted in zip(actual, expected)
        )
    assert not exact_symmetric_semidefinite(exact)
    assert _quadratic(exact, (Q(1), Q(-1), Q(0))) == -3


def test_full_receiver_restriction_preserves_phase_potential_gradient_and_curvature(mp):
    receiver_turns = (Q(0), Q(1, 16), Q(1, 6), Q(3, 8), Q(5, 8), Q(7, 8))
    receiver = tuple(2 * mp.pi * _mp(mp, turn) for turn in receiver_turns)
    edges = ((0, 1), (0, 2), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1))
    phases = tuple(
        (
            receiver[0] + 2 * mp.pi * (node // 2) / 5
            if node < 10
            else receiver[(node - 10) // 2 + 1]
        )
        for node in range(20)
    )
    assert all(
        abs(phases[right] - phases[left] - 2 * mp.pi * branch) < mp.pi
        for left, right, branch in _MARKED
    )
    donor_constant = _phase_potential(mp, phases, _EDGES[:20])
    full_potential = _phase_potential(mp, phases, _EDGES)
    reduced_potential = _phase_potential(mp, receiver, edges)
    assert abs(full_potential - donor_constant - 4 * reduced_potential) < mp.mpf(
        "1e-90"
    )
    labels = tuple(0 if node < 10 else (node - 10) // 2 + 1 for node in range(20))
    full_current, reduced_current = _currents(mp, phases), _currents(
        mp, receiver, edges
    )
    for label, current in enumerate(reduced_current):
        projected = sum(
            full_current[node] for node in range(20) if labels[node] == label
        )
        assert abs(projected - 4 * current) < mp.mpf("1e-90")

    full_hessian = _phase_hessian(mp, phases, _EDGES)
    reduced_hessian = _phase_hessian(mp, receiver, edges)
    for left in range(6):
        for right in range(6):
            projected = sum(
                full_hessian[i][j]
                for i in range(20)
                for j in range(20)
                if labels[i] == left and labels[j] == right
            )
            assert abs(projected - 4 * reduced_hessian[left][right]) < mp.mpf("1e-90")

    increment = mp.mpf(1) / 64
    rotated = tuple(
        value + increment if node < 10 else value for node, value in enumerate(phases)
    )
    rotated_receiver = (receiver[0] + increment,) + receiver[1:]
    assert abs(
        _phase_potential(mp, rotated, _EDGES)
        - full_potential
        - 4 * (_phase_potential(mp, rotated_receiver, edges) - reduced_potential)
    ) < mp.mpf("1e-90")
    report = bound_relational_sine_support_exchange(
        _graph(phases), reference_model=_MODEL
    )
    captured_potential = _phase_potential(
        mp, tuple(_mp(mp, phase) for phase in report.phase), _EDGES
    )
    assert (
        _mp(mp, report.phase_storage.lo)
        <= captured_potential
        <= _mp(mp, report.phase_storage.hi)
    )


@pytest.mark.parametrize("signs", tuple(product((-1, 1), repeat=3)))
def test_stationary_chain_signed_curvature_requires_all_positive_cosines(signs):
    # Equal stationary sine values give cosines c*signs with common c>0.
    # Divide out c, assemble the path Hessian, and retain both free interior
    # directions. Nonnegative node diagonals alone are not sufficient.
    graph = nx.path_graph(4)
    for edge, sign in zip(graph.edges(), signs):
        graph.edges[edge]["weight"] = Q(sign)
    full = _laplacian(graph, weighted=True)
    interior = tuple(tuple(full[left][right] for right in (1, 2)) for left in (1, 2))
    expected = (
        (Q(signs[0] + signs[1]), Q(-signs[1])),
        (Q(-signs[1]), Q(signs[1] + signs[2])),
    )
    assert interior == expected
    if signs == (1, 1, 1):
        assert exact_symmetric_semidefinite(interior, strict=True)
    else:
        assert not exact_symmetric_semidefinite(interior)
        assert any(
            _quadratic(interior, tuple(map(Q, vector))) < 0
            for vector in product(range(-2, 3), repeat=2)
        )


def test_zero_chain_curvature_is_retained_and_endpoint_bounds_are_incompatible():
    zero = ((Q(0), Q(0)), (Q(0), Q(0)))
    assert exact_symmetric_semidefinite(zero)
    assert not exact_symmetric_semidefinite(zero, strict=True)
    # If |sin a|<=1/2, an obtuse principal branch has cos a<=-sqrt(3)/2.
    # Even the largest possible return cosine cannot compensate its diagonal.
    maximum_obtuse_diagonal = 1 - sqrt(I(3))
    assert maximum_obtuse_diagonal.hi < 0
    maximum_endpoint_turn = Q(1, 12)
    maximum_return_gap_turn = Q(1, 4)
    maximum_endpoint_span = 2 * maximum_endpoint_turn
    minimum_winding_span = 1 - 3 * maximum_return_gap_turn
    assert maximum_endpoint_span < minimum_winding_span


def test_fixed_algebraic_saddle_is_stationary_in_the_original_marked_class(mp):
    # Section 15.1 fixes this algebraic construction before evaluation.
    # The reader captures rounded phases, so its tiny residual is bounded
    # separately rather than claiming an exact represented stationary point.
    cosine = (1 + mp.sqrt(17)) / 8
    assert abs(4 * cosine**2 - cosine - 1) < mp.mpf("1e-90")
    parameter = mp.acos(cosine)
    gap = 2 * parameter
    endpoint = mp.pi - 3 * parameter
    receiver = (
        mp.mpf(0),
        endpoint,
        endpoint + gap,
        endpoint + 2 * gap,
        2 * mp.pi - endpoint,
    )
    phases = tuple(
        2 * mp.pi * (node // 2) / 5 if node < 10 else receiver[(node - 10) // 2]
        for node in range(20)
    )
    assert mp.pi / 2 < gap < mp.pi
    assert all(
        abs(phases[right] - phases[left] - 2 * mp.pi * branch) < mp.pi
        for left, right, branch in _MARKED
    )
    assert abs(mp.sin(gap) - 2 * mp.sin(endpoint)) < mp.mpf("1e-90")
    assert max(map(abs, _currents(mp, phases))) < mp.mpf("1e-90")
    hessian = _phase_hessian(mp, phases, _EDGES)
    exact_diagonal_bounds = (sqrt(I(17)) - 7) / 4
    assert exact_diagonal_bounds.hi < 0
    expected_diagonal = (mp.sqrt(17) - 7) / 4
    assert abs(hessian[14][14] - 4 * mp.cos(gap)) < mp.mpf("1e-90")
    assert abs(hessian[14][14] - expected_diagonal) < mp.mpf("1e-90")
    report = bound_relational_sine_support_exchange(
        _graph(phases), reference_model=_MODEL
    )
    assert report.form_gradient == (Q(0),) * 20
    assert report.phase_rate_numerators() == (Q(0),) * 20
    rho = tuple(
        degree / capacity for degree, capacity in zip(report.degrees, report.capacity)
    )
    shift = rho[14] / sum(rho)
    relative_direction = tuple(Q(int(node == 14)) - shift for node in range(20))
    assert sum(weight * value for weight, value in zip(rho, relative_direction)) == 0
    curvature = _quadratic(
        hessian, tuple(_mp(mp, value) for value in relative_direction)
    )
    assert abs(curvature - expected_diagonal) < mp.mpf("1e-90")
    assert curvature < 0

    captured = tuple(_mp(mp, phase) for phase in report.phase)
    captured_currents = _currents(mp, captured)
    for current, capacity, degree, bounds in zip(
        captured_currents, report.capacity, report.degrees, report.form_rates
    ):
        expected = _mp(mp, capacity) * current / (mp.pi * degree)
        assert abs(expected) < _mp(mp, _TOLERANCE)
        assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)
        assert abs(_mp(mp, bounds.midpoint) - expected) < _mp(mp, _TOLERANCE)


@pytest.fixture(scope="module")
def confinement_reference(mp):
    """Section 17's nominal AL forms on the unchanged auxiliary support."""
    phases = tuple(2 * mp.pi * ((node % 10) // 2) / 5 for node in range(20))
    graph = _graph(phases)
    for node in (0, 1):
        graph.nodes[node]["EPI"] += Q(1, 1024)
    report = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    rho = tuple(
        degree / capacity for degree, capacity in zip(report.degrees, report.capacity)
    )
    mean = sum(weight * value for weight, value in zip(rho, report.epi)) / sum(rho)
    direction = tuple(value - mean for value in report.epi)
    return phases, graph, report, rho, mean, direction


def test_contact_form_storage_is_positive_and_coercive_on_the_weighted_leaf(
    confinement_reference,
):
    _, graph, report, rho, mean, direction = confinement_reference
    assert sum(report.epi[:10]) / 10 - sum(report.epi[10:]) / 10 == Q(1, 5120)
    assert sum(weight * value for weight, value in zip(rho, direction)) == 0
    assert mean != sum(report.epi) / len(report.epi)
    arithmetic_center = tuple(value - sum(report.epi) / 20 for value in report.epi)
    assert sum(weight * value for weight, value in zip(rho, arithmetic_center)) != 0
    independent_storage = sum(
        weight * (report.epi[right] - report.epi[left]) ** 2 / 2
        for (left, right), weight in zip(report.edges, report.conductances)
    )
    assert independent_storage == report.form_storage > 0
    laplacian = _laplacian(graph, weighted=True)
    assert _quadratic(laplacian, direction) / 2 == independent_storage

    # Restrict with exact columns e_j-(rho_j/rho_0)e_0. This is the actual
    # conserved-mean tangent space, not the arithmetic mean-zero slice.
    basis = tuple(
        tuple(
            Q(node == column) - (rho[column] / rho[0] if node == 0 else 0)
            for column in range(1, 20)
        )
        for node in range(20)
    )
    assert exact_rank(basis) == 19
    assert all(
        sum(rho[node] * basis[node][column] for node in range(20)) == 0
        for column in range(19)
    )
    restricted = exact_matrix_product(
        tuple(zip(*basis)), exact_matrix_product(laplacian, basis)
    )
    assert exact_symmetric_semidefinite(restricted, strict=True)

    assert nx.shortest_path_length(graph, 4, 14) == nx.diameter(graph) == 4
    minimum_weight = min(report.conductances)
    squared_range_bound = 2 * nx.diameter(graph) * report.form_storage / minimum_weight
    assert max(value**2 for value in direction) <= squared_range_bound
    # The graph-path Cauchy bound uses positive conductance and a fixed mean.
    for left in graph:
        for right in graph:
            path = nx.shortest_path(graph, left, right)
            differences = tuple(
                report.epi[head] - report.epi[tail]
                for tail, head in zip(path, path[1:])
            )
            assert (report.epi[right] - report.epi[left]) ** 2 <= len(
                differences
            ) * sum(value**2 for value in differences)


def test_frozen_phase_comparison_path_has_strictly_decreasing_storage():
    alpha_cosine = (sqrt(I(5)) - 1) / 4
    alpha_sine = sqrt(10 + 2 * sqrt(I(5))) / 4
    interval = I(0, Q(1, 64))
    derivative = 24 * alpha_cosine * sin(interval) - 8 * alpha_sine * cos(interval)
    assert derivative.hi < 0
    # Independently bound the entire derivative without a trajectory or
    # treating the three sampled points as a monotonicity certificate.
    coarse_upper = Q(24, 64) - 8 * Q(19, 20) * (1 - Q(1, 2 * 64**2))
    assert derivative.hi < coarse_upper < 0


@pytest.mark.parametrize(
    "uniform_form", (False, True), ids=("post-al-forms", "uniform-auxiliary")
)
@pytest.mark.parametrize("shift", (Q(0), Q(1, 128), Q(1, 64)))
def test_static_radial_form_lift_preserves_storage_and_both_weighted_means(
    mp, confinement_reference, uniform_form, shift
):
    phases, _, reference, rho, nonconstant_mean, direction = confinement_reference
    phase_increment = tuple(
        -shift if node in (12, 13) else shift if node in (18, 19) else Q(0)
        for node in range(20)
    )
    mean_increment = sum(
        weight * value for weight, value in zip(rho, phase_increment)
    ) / sum(rho)
    assert mean_increment == 0
    moved_phases = tuple(
        value + _mp(mp, change - mean_increment)
        for value, change in zip(phases, phase_increment)
    )
    assert all(
        abs(moved_phases[right] - moved_phases[left] - 2 * mp.pi * branch) < mp.pi
        for left, right, branch in _MARKED
    )
    initial_potential = _phase_potential(mp, phases, _EDGES)
    moved_potential = _phase_potential(mp, moved_phases, _EDGES)
    alpha, parameter = 2 * mp.pi / 5, _mp(mp, shift)
    independently_reduced_change = 24 * mp.cos(alpha) * (
        1 - mp.cos(parameter)
    ) - 8 * mp.sin(alpha) * mp.sin(parameter)
    assert abs(
        moved_potential - initial_potential - independently_reduced_change
    ) < mp.mpf("1e-90")
    assert moved_potential <= initial_potential
    mean = Q(5, 16) if uniform_form else nonconstant_mean
    source_form_storage = Q(0) if uniform_form else reference.form_storage
    target_energy = initial_potential + _mp(mp, source_form_storage)
    scale = mp.sqrt((target_energy - moved_potential) / _mp(mp, reference.form_storage))
    form = tuple(_mp(mp, mean) + scale * _mp(mp, value) for value in direction)
    lifted_storage = sum(
        _mp(mp, weight) * (form[right] - form[left]) ** 2 / 2
        for (left, right), weight in zip(reference.edges, reference.conductances)
    )
    assert abs(lifted_storage + moved_potential - target_energy) < mp.mpf("1e-90")
    assert abs(
        sum(
            _mp(mp, weight) * (value - _mp(mp, mean))
            for weight, value in zip(rho, form)
        )
    ) < mp.mpf("1e-90")
    assert abs(
        sum(
            _mp(mp, weight) * (after - before)
            for weight, before, after in zip(rho, phases, moved_phases)
        )
    ) < mp.mpf("1e-90")
    if shift == 0:
        expected = (Q(5, 16),) * 20 if uniform_form else reference.epi
        assert all(
            abs(actual - _mp(mp, wanted)) < mp.mpf("1e-90")
            for actual, wanted in zip(form, expected)
        )
    elif uniform_form:
        assert scale > 0

    # Re-admit the represented configuration independently. Its actual
    # captured H, the ideal path identity and enclosure width stay separate.
    graph = _graph(moved_phases)
    for node, value in enumerate(form):
        graph.nodes[node]["EPI"] = float(value)
    captured = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    captured_energy = _mp(mp, captured.form_storage) + _phase_potential(
        mp, tuple(_mp(mp, value) for value in captured.phase), _EDGES
    )
    assert (
        _mp(mp, captured.storage.lo) <= captured_energy <= _mp(mp, captured.storage.hi)
    )
    assert abs(_mp(mp, captured.storage.midpoint) - captured_energy) < _mp(
        mp, _TOLERANCE
    )
    assert abs(captured_energy - target_energy) < _mp(mp, _TOLERANCE)
    if not uniform_form and shift == 0:
        # This particular comparison path is not the conservative flow:
        # it fixes the donor phase while that admitted field has nonzero rate.
        assert phase_increment[0] == 0
        assert captured.phase_rate_numerators()[0] > 0


def test_rigid_receiver_rotation_separates_mixed_cut_from_constituent_windings(
    mp, confinement_reference
):
    _, _, reference, rho, _, _ = confinement_reference
    original = tuple(Q((node % 10) // 2, 5) for node in range(20))
    old_rows = tuple(
        _cycle_row(list(range(offset, offset + 10, 2))) for offset in (0, 10)
    )
    mixed = _cycle_row([0, 12, 10])
    baseline_potential = _phase_potential(
        mp, tuple(2 * mp.pi * _mp(mp, value) for value in original), _EDGES
    )
    for rotation in (Q(0), Q(1, 4), Q(3, 10), Q(2, 5)):
        correction = rotation * sum(rho[10:]) / sum(rho)
        turns = tuple(
            value + (rotation if node >= 10 else 0) - correction
            for node, value in enumerate(original)
        )
        assert sum(weight * after for weight, after in zip(rho, turns)) == sum(
            weight * before for weight, before in zip(rho, original)
        )
        declared_gaps = tuple(
            turns[right] - turns[left] - branch for left, right, branch in _MARKED
        )
        assert declared_gaps[:40] == tuple(
            original[right] - original[left] - branch
            for left, right, branch in _MARKED[:40]
        )
        for row in old_rows:
            assert (
                sum(coefficient * gap for coefficient, gap in zip(row, declared_gaps))
                == 1
            )
        cuts = tuple(
            index for index, gap in enumerate(declared_gaps) if abs(gap) == Q(1, 2)
        )
        if rotation < Q(3, 10):
            assert all(abs(gap) < Q(1, 2) for gap in declared_gaps)
        elif rotation == Q(3, 10):
            assert len(cuts) == 4
            assert {_EDGES[index] for index in cuts} == {
                (left, right) for left in (0, 1) for right in (12, 13)
            }
            assert any(mixed[index] for index in cuts)
        else:
            assert not cuts
            assert sum(abs(gap) > Q(1, 2) for gap in declared_gaps) == 4
        # A cycle with a cut edge has no principal-winding readout at that
        # endpoint; choosing a wrap convention would not repair availability.
        if not any(mixed[index] for index in cuts):
            principal = tuple(
                gap - ((gap + Q(1, 2)).numerator // (gap + Q(1, 2)).denominator)
                for gap in declared_gaps
            )
            expected_winding = 0 if rotation < Q(3, 10) else -1
            assert (
                sum(coefficient * gap for coefficient, gap in zip(mixed, principal))
                == expected_winding
            )
        potential = _phase_potential(
            mp, tuple(2 * mp.pi * _mp(mp, value) for value in turns), _EDGES
        )
        if rotation:
            assert potential > baseline_potential
    # This rigid rotation is a cycle-information control, not the theorem's
    # constant-energy path or a claim about the actual contacted response.
