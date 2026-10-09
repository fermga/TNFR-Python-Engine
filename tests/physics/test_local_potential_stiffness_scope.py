"""One-hop structural conditions constrain, but do not select, a potential.

These are exact algebraic controls on the retained unit triangular prism.
Q denotes the constant Hessian of U, so the previously selected form
3 sum(dy wedge dv) gives acceleration -Q*y/3. No potential, phase lift,
clock, trajectory or runtime update is installed by these tests.
"""

from itertools import permutations

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    Q_MODE,
    P,
    _exact_generator,
    _graph,
)
from tnfr.physics.support_transport import _laplacian, observe_support_transport
from tnfr.physics.symmetry_sectors import automorphism_permutations

s = pytest.importorskip("sympy")


@pytest.fixture(scope="module")
def prism():
    graph = _graph()
    snapshot = observe_support_transport(graph)
    index = {node: i for i, node in enumerate(NODES)}
    group = automorphism_permutations(graph, weight="weight", cap=12)
    images = tuple(tuple(index[g[node]] for node in NODES) for g in group)
    edges = tuple(
        sorted(
            tuple(sorted((index[left], index[right]))) for left, right in graph.edges
        )
    )
    triangle = tuple(edge for edge in edges if NODES[edge[0]][0] == NODES[edge[1]][0])
    matching = tuple(edge for edge in edges if edge not in triangle)

    def block_laplacian(edge_set):
        conductance = tuple(
            (i, j, weight)
            for i, j, weight in snapshot.conductance
            if tuple(sorted((i, j))) in edge_set
        )
        columns = tuple(
            _laplacian(conductance, tuple(s.Rational(i == j) for i in range(6)))
            for j in range(6)
        )
        return s.Matrix(columns).T

    return (
        graph,
        images,
        edges,
        triangle,
        matching,
        block_laplacian(triangle),
        block_laplacian(matching),
        -s.Matrix(_exact_generator(snapshot)),
    )


def test_all_automorphisms_leave_exactly_two_one_hop_hessian_coefficients(prism):
    graph, images, edges, triangle, matching, tri, match, _ = prism
    index = {node: i for i, node in enumerate(NODES)}
    expected_group = {
        tuple(index[(layer ^ flip, permutation[i])] for layer, i in NODES)
        for flip in (0, 1)
        for permutation in permutations(range(3))
    }
    assert len(images) == 12 and set(images) == expected_group
    assert {degree for _, degree in graph.degree()} == {3}
    # Triangle edges belong to 3-cycles; matching edges do not. The group
    # is vertex-transitive but is not edge-transitive despite equal degrees.
    for family in (triangle, matching):
        i, j = family[0]
        orbit = {tuple(sorted((image[i], image[j]))) for image in images}
        assert orbit == set(family)
    assert (len(triangle), len(matching)) == (6, 3)

    # Every symmetric one-hop Q with Q*1=0 has exactly this edge form;
    # off-diagonal entries fix its weights and row sums fix its diagonal.
    coefficients = s.symbols("c0:9", real=True)
    q = s.zeros(6)
    for (i, j), coefficient in zip(edges, coefficients, strict=True):
        incidence = s.eye(6)[:, i] - s.eye(6)[:, j]
        q += coefficient * incidence * incidence.T
    assert q == q.T and q * s.ones(6, 1) == s.zeros(6, 1)
    constraints = [
        q[image[i], image[j]] - q[i, j]
        for image in images
        for i in range(6)
        for j in range(6)
    ]
    equations, zero = s.linear_eq_to_matrix(constraints, coefficients)
    assert zero == s.zeros(len(constraints), 1)
    assert len(coefficients) - equations.rank() == 2
    assert s.Matrix.hstack(tri.reshape(36, 1), match.reshape(36, 1)).rank() == 2
    # Two independent members of the two-dimensional solution space span
    # every admissible constant Hessian, including indefinite choices.
    for matrix in (tri, match):
        for image in images:
            assert matrix.extract(image, image) == matrix


def test_full_mode_decomposition_exposes_the_unfixed_edge_ratio(prism):
    _, _, _, _, _, tri, match, _ = prism
    a, b = s.symbols("a b", real=True)
    q = a * tri + b * match
    mean_difference = s.Matrix((1, 1, 1, -1, -1, -1))
    even_p, even_q = s.Matrix(P * 2), s.Matrix(Q_MODE * 2)
    odd_p = s.Matrix(P + tuple(-value for value in P))
    odd_q = s.Matrix(Q_MODE + tuple(-value for value in Q_MODE))
    basis = s.Matrix.hstack(s.ones(6, 1), mean_difference, even_p, even_q, odd_p, odd_q)
    eigenvalues = s.diag(0, 2 * b, 3 * a, 3 * a, 3 * a + 2 * b, 3 * a + 2 * b)
    assert basis.rank() == 6
    assert (q * basis - basis * eigenvalues).applyfunc(s.expand) == s.zeros(6)
    # This full basis proves Q>=0 iff a,b>=0 and positive centered Q iff
    # a,b>0. Positivity alone leaves their ratio arbitrary.
    assert all(value != 0 for value in (basis.T * basis).diagonal())
    assert basis.T * basis == s.diag(6, 6, 4, 12, 4, 12)
    original = even_p / 4
    assert (q * original - 3 * a * original).applyfunc(s.expand) == s.zeros(6, 1)
    assert s.diff(q * original, b) == s.zeros(6, 1)
    assert s.diff(q * mean_difference, b) == 2 * mean_difference
    # Under the selected kinetic term, squared modal frequencies are the
    # positive eigenvalues divided by three, not first-order decay rates.
    assert s.simplify((3 * a / 3) / (2 * b / 3)) == 3 * a / (2 * b)


def test_matching_the_supplied_unit_transport_is_an_additional_relation(prism):
    graph, _, _, _, _, tri, match, lap = prism
    assert tri + match == 3 * lap
    a, b, sigma = s.symbols("a b sigma", real=True)
    q = a * tri + b * match
    # U=y^T Q y/2 and H=3||v||^2/2+U yield -Q*y/3. Requiring this
    # acceleration to equal -sigma*L*y on every centered vector selects
    # equal edge coefficients. This premise is not supplied by symmetry.
    project = s.eye(6) - s.ones(6) / 6
    residual = (q / 3 - sigma * lap) * project
    assert s.solve(list(residual), (a, b)) == {a: sigma, b: sigma}
    assert s.solve(list(residual), (a, b, sigma)) == {a: sigma, b: sigma}
    # Even the original form fixes only a=sigma, leaving the matching
    # stiffness b invisible. No new amplitude preparation is introduced.
    original = s.Matrix(P * 2) / 4
    actual = s.Matrix([data["EPI"] for _, data in graph.nodes(data=True)])
    assert actual == s.ones(6, 1) / 2 + original
    assert s.solve(list(residual * original), (a, b)) == {a: sigma}


def test_nonlinear_one_hop_potentials_share_the_matched_consensus_hessian(prism):
    _, images, edges, _, _, tri, match, lap = prism
    x = s.Matrix(s.symbols("x0:6", real=True))
    quadratic = (x.T * (tri + match) * x)[0] / 2
    quartic = sum((x[i] - x[j]) ** 4 for i, j in edges) / 4
    # Declared dimensionless polynomial countercontrols: choosing the
    # quartic term physically would need a form scale and a new premise.
    # They are not installed or fitted candidate maintenance mechanisms.
    potentials = (quadratic, quadratic + quartic)
    gradients = tuple(
        s.Matrix([s.diff(value, item) for item in x]) for value in potentials
    )
    hessians = tuple(gradient.jacobian(x) for gradient in gradients)
    consensus = {item: 0 for item in x}
    assert hessians[0] == 3 * lap
    assert hessians[1].subs(consensus) == hessians[0]
    assert gradients[1] != gradients[0]
    shift = s.symbols("shift", real=True)
    for potential, gradient, hessian in zip(
        potentials, gradients, hessians, strict=True
    ):
        assert (
            s.expand(potential.xreplace({item: item + shift for item in x}) - potential)
            == 0
        )
        assert s.expand(sum(gradient)) == 0
        for i in range(6):
            for j in range(6):
                if i != j and tuple(sorted((i, j))) not in edges:
                    assert hessian[i, j] == 0
        for image in images:
            relabeling = {x[i]: x[image[i]] for i in range(6)}
            assert s.expand(potential.xreplace(relabeling) - potential) == 0
    # Their different nonlinear restoring response is already visible at
    # the retained form. Equal linearization is therefore not a full law.
    original = dict(zip(x, s.Matrix(P * 2) / 4, strict=True))
    assert (gradients[1] - gradients[0]).subs(original) != s.zeros(6, 1)
