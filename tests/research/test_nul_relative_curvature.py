"""Full relative phase curvature for the fixed causal cross-law construction.

The original preparation and pure NUL image come from the preceding controls.
All time-dependent statements use analytic whole-window bounds; no trajectory,
retained response or sampled eigenvalue is evaluated. Matrix controls retain
all twenty independent phase variations, including variations breaking twins.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.research import test_nul_cross_law_observation as cross_law
from tests.research.test_nul_causal_class_exit import (
    CRITICAL_EDGES,
    HORIZON,
    PERMUTATION,
)
from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage

PARAMETER = I(0, 3)
DISPLACEMENT_CEILING = Q(1, 2**18)
GOOD_FLOOR = Q(1, 6)
NEGATIVE_CEILING = 10 * DISPLACEMENT_CEILING
branches = cross_law.branches


@pytest.fixture(scope="module", autouse=True)
def no_event_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("curvature controls must not execute or prepare a runtime event")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "execute_pointwise_stage",
            "execute_operator_major_stage",
            "_propose_pointwise",
        ):
            patch.setattr(network_stage, name, forbidden)
        yield


def _rational_matrix(matrix):
    return tuple(
        tuple(Q(int(value.p), int(value.q)) for value in matrix.row(row))
        for row in range(matrix.rows)
    )


@pytest.fixture(scope="module")
def geometry(branches):
    s = pytest.importorskip("sympy")
    graph = branches[0]
    edges = tuple(graph.edges)
    incidence = s.zeros(20, 45)
    for column, (left, right) in enumerate(edges):
        incidence[left, column], incidence[right, column] = 1, -1
    good_columns = [i for i, edge in enumerate(edges) if edge not in CRITICAL_EDGES]
    critical_columns = [i for i, edge in enumerate(edges) if edge in CRITICAL_EDGES]
    good_incidence = incidence[:, good_columns]
    critical_incidence = incidence[:, critical_columns]
    return (
        s,
        graph,
        edges,
        incidence,
        good_incidence * good_incidence.T,
        critical_incidence * critical_incidence.T,
    )


def test_canonical_unweighted_hessian_and_edge_sign_are_derived_from_potential(
    geometry,
):
    s, graph, edges, incidence, _, _ = geometry
    gap, epsilon, displacement = s.symbols("gap epsilon displacement", real=True)
    left, right = s.symbols("left right", real=True)
    potential = (
        1 - s.cos(gap) + epsilon * (s.Rational(2, 3) - s.cos(gap) + s.cos(gap) ** 3 / 3)
    )
    stiffness = s.cos(gap) * (1 + 3 * epsilon * s.sin(gap) ** 2)
    assert s.trigsimp(s.diff(potential, gap, 2) - stiffness) == 0
    directional = s.diff(
        potential.subs(gap, gap + displacement * (right - left)), displacement, 2
    ).subs(displacement, 0)
    assert s.trigsimp(directional - stiffness * (right - left) ** 2) == 0
    # The positive factor fixes the sign for every real gap and epsilon in [0,3].
    multiplier = 1 + 3 * PARAMETER * I(-1, 1) ** 2
    assert multiplier == I(1, 10)
    assert stiffness.subs(gap, 0) == 1
    assert stiffness.subs(gap, s.pi / 2) == 0
    assert stiffness.subs(gap, s.pi) == -1
    variations = s.Matrix(s.symbols("variation0:20", real=True))
    coefficients = s.symbols("stiffness0:45", real=True)
    hessian = incidence * s.diag(*coefficients) * incidence.T
    expected = sum(
        coefficient * (variations[i] - variations[j]) ** 2
        for coefficient, (i, j) in zip(coefficients, edges)
    )
    assert s.expand(variations.dot(hessian * variations) - expected) == 0
    assert hessian * s.ones(20, 1) == s.zeros(20, 1)
    weights = s.diag(*(s.Rational(graph.edges[edge]["weight"]) for edge in edges))
    assert incidence * weights * s.diag(*coefficients) * incidence.T != hessian


def test_actual_good_graph_is_connected_on_all_twenty_phase_coordinates(geometry):
    s, graph, _, _, good, critical = geometry
    retained = graph.copy()
    retained.remove_edges_from(CRITICAL_EDGES)
    assert tuple(retained) == tuple(range(20))
    assert graph.number_of_edges() == 45 and retained.number_of_edges() == 43
    assert nx.is_connected(retained)
    assert good * s.ones(20, 1) == critical * s.ones(20, 1) == s.zeros(20, 1)
    assert exact_symmetric_semidefinite(_rational_matrix(good))
    assert not exact_symmetric_semidefinite(_rational_matrix(good), strict=True)
    assert exact_symmetric_semidefinite(_rational_matrix(good[1:, 1:]), strict=True)


def test_critical_paths_dominate_exactly_without_restricting_twin_variations(geometry):
    s, graph, _, _, good, critical = geometry
    variation = s.Matrix(s.symbols("variation0:20", real=True))
    path_bound = 0
    critical_sum = 0
    square_remainders = 0
    for endpoint in (2, 3):
        assert graph.has_edge(1, 0) and graph.has_edge(0, endpoint)
        assert (0, endpoint) not in CRITICAL_EDGES
        first = variation[1] - variation[0]
        second = variation[0] - variation[endpoint]
        path_bound += 2 * (first**2 + second**2)
        critical_sum += (variation[1] - variation[endpoint]) ** 2
        square_remainders += (first - second) ** 2
    assert s.expand(path_bound - critical_sum - square_remainders) == 0
    assert s.expand(variation.dot(critical * variation) - critical_sum) == 0
    assert exact_symmetric_semidefinite(_rational_matrix(4 * good - critical))
    twin_breaking = s.zeros(20, 1)
    twin_breaking[2], twin_breaking[3] = 1, -1
    assert twin_breaking[PERMUTATION[2]] != twin_breaking[2]
    assert twin_breaking.dot(critical * twin_breaking) == 2
    assert twin_breaking.dot(good * twin_breaking) > 0
    assert twin_breaking.dot((4 * good - critical) * twin_breaking) > 0


def test_rooted_good_tree_controls_the_full_relative_norm(geometry):
    s, graph, _, _, good, _ = geometry
    parents = {
        **dict.fromkeys((1, 2, 3, 8, 9, 10, 11), 0),
        **dict.fromkeys((4, 5), 2),
        **dict.fromkeys((6, 7), 8),
        **dict.fromkeys((12, 13, 18, 19), 10),
        **dict.fromkeys((14, 15), 12),
        **dict.fromkeys((16, 17), 18),
    }
    tree = nx.Graph((parent, child) for child, parent in parents.items())
    assert set(tree) == set(graph) and nx.is_tree(tree)
    assert all(graph.has_edge(*edge) for edge in tree.edges)
    assert all(tuple(sorted(edge)) not in CRITICAL_EDGES for edge in tree.edges)
    distances = nx.single_source_shortest_path_length(tree, 0)
    assert sum(distances.values()) == 35
    assert {node for node, distance in distances.items() if distance == 3} == {
        14,
        15,
        16,
        17,
    }
    root_form = s.zeros(20)
    path_form = s.zeros(20)
    for node in range(1, 20):
        path = nx.shortest_path(tree, 0, node)
        endpoint = s.zeros(20, 1)
        endpoint[node], endpoint[0] = 1, -1
        root_form += endpoint * endpoint.T
        path_laplacian = s.zeros(20)
        for left, right in zip(path, path[1:]):
            edge = s.zeros(20, 1)
            edge[left], edge[right] = 1, -1
            path_laplacian += edge * edge.T
        cauchy = distances[node] * path_laplacian - endpoint * endpoint.T
        assert exact_symmetric_semidefinite(_rational_matrix(cauchy))
        path_form += distances[node] * path_laplacian
    relative = s.eye(20) - s.ones(20) / 20
    assert exact_symmetric_semidefinite(_rational_matrix(path_form - root_form))
    assert exact_symmetric_semidefinite(_rational_matrix(35 * good - path_form))
    assert exact_symmetric_semidefinite(_rational_matrix(root_form - relative))
    assert exact_symmetric_semidefinite(_rational_matrix(35 * good - relative))


@pytest.mark.parametrize("branch", (0, 1))
def test_whole_window_stiffness_bounds_cover_all_edges_and_every_parameter(
    branches, branch
):
    graph = branches[branch]
    pi = pi_interval()
    forms = tuple(Q(graph.nodes[node]["EPI"]) for node in graph)
    amplitude = max(forms) - min(forms)
    assert 0 <= amplitude < Q(1, 16)
    assert all(0 < Q(graph.nodes[node]["nu_f"]) <= 1 for node in graph)
    assert all(0 < Q(data["weight"]) <= 1 for _, _, data in graph.edges(data=True))
    # The previously proved global acceleration bound uses |j_epsilon| <= 4.
    # This displacement encloses every t in [0,T], not just its endpoint.
    displacement = 2 * (amplitude * HORIZON + 4 * HORIZON**2 / pi.lo) / pi.lo
    assert displacement < HORIZON / 24 + Q(4, 2**30) < DISPLACEMENT_CEILING
    assert Q(1, 4) - DISPLACEMENT_CEILING > Q(1, 5)
    assert sin(I(Q(1, 5))).lo > GOOD_FLOOR
    good_count, critical_count = 0, 0
    for edge in graph.edges:
        initial = cross_law._gap(graph, edge)
        whole_window = I(initial.lo - displacement, initial.hi + displacement)
        cosine = cos(whole_window)
        stiffness = cosine * (1 + 3 * PARAMETER * sin(whole_window) ** 2)
        assert whole_window.abs_max < pi.lo
        if edge in CRITICAL_EDGES:
            assert initial.hi < pi.lo / 2
            assert cosine.lo > -DISPLACEMENT_CEILING
            assert stiffness.lo > -NEGATIVE_CEILING
            critical_count += 1
        else:
            assert pi.lo / 2 - initial.abs_max > Q(1, 4)
            assert pi.lo / 2 - whole_window.abs_max > Q(1, 5)
            assert cosine.lo > GOOD_FLOOR
            assert stiffness.lo > GOOD_FLOOR
            good_count += 1
    assert (good_count, critical_count) == (43, 2)


def test_endpoint_negative_critical_curvature_coexists_with_uniform_full_bound(
    branches,
):
    before, after, _ = branches
    pi = pi_interval()
    remainder = 8 * HORIZON**2 / pi.lo**2
    assert remainder < Q(4, 2**30)
    for graph, crosses in ((before, False), (after, True)):
        numerators = cross_law._primitive_rows(graph, Q(0))[2]
        for edge in CRITICAL_EDGES:
            left, right = edge
            rate = I(numerators[right] - numerators[left]) / pi
            endpoint = (
                cross_law._gap(graph, edge) + rate * HORIZON + I(-remainder, remainder)
            )
            stiffness = cos(endpoint) * (1 + 3 * PARAMETER * sin(endpoint) ** 2)
            assert 0 < endpoint.lo <= endpoint.hi < pi.lo
            if crosses:
                assert endpoint.lo - pi.hi / 2 > Q(12, 2**30)
                assert -NEGATIVE_CEILING < stiffness.lo <= stiffness.hi < 0
            else:
                assert pi.lo / 2 - endpoint.hi > Q(4, 2**30)
                assert stiffness.lo > 0


def test_uniform_full_matrix_bound_has_only_the_common_phase_kernel(geometry):
    s, _, _, _, good, critical = geometry
    coefficient = GOOD_FLOOR - 4 * NEGATIVE_CEILING
    assert coefficient > Q(1, 7)
    lower = s.Rational(GOOD_FLOOR) * good - s.Rational(NEGATIVE_CEILING) * critical
    relative = s.eye(20) - s.ones(20) / 20
    for comparison in (
        lower,
        lower - good / 7,
        lower - relative / 245,
    ):
        assert comparison * s.ones(20, 1) == s.zeros(20, 1)
        assert exact_symmetric_semidefinite(_rational_matrix(comparison))
        assert not exact_symmetric_semidefinite(
            _rational_matrix(comparison), strict=True
        )
        assert exact_symmetric_semidefinite(
            _rational_matrix(comparison[1:, 1:]), strict=True
        )
    # Strict inequalities apply to nonconstant variations. The common phase
    # direction has zero curvature exactly, so strict positivity on R^20 fails.


def test_an_uncontrolled_negative_edge_can_destroy_relative_curvature(geometry):
    s, _, edges, incidence, good, _ = geometry
    column = edges.index((1, 2))
    edge = incidence[:, column]
    unsafe = good - 6 * edge * edge.T
    variation = s.zeros(20, 1)
    variation[1] = 1
    assert variation.dot(good * variation) == 5
    assert variation.dot(unsafe * variation) == -1
    assert not exact_symmetric_semidefinite(_rational_matrix(unsafe))
    # This deliberately violates the proved small-negative-stiffness bound.
    # Connectivity alone cannot justify ignoring an arbitrary negative edge.
