"""Exact static equilibrium controls and separate materialized tangent wiring.

No trajectory or eigenvalue tolerance proves an equilibrium here. Exact
quadratic-form controls support the Hessian signatures; the native twenty-row
checks only verify that the existing engine observes the same local law.
"""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import relational_reflected_equilibria as owner
from tnfr.physics import relational_reflected_transit as transit

MODEL = RelationalExchangeModel(1, phase_domain="regular")
BRANCHES = (("consensus", 1),) + tuple(
    (family, orientation)
    for family in ("aligned_twist", "aligned_saddle", "opposite_twist")
    for orientation in (1, -1)
)
EDGES = tuple(
    (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
) + ((0, 5), (1, 6))


@pytest.fixture(scope="module", autouse=True)
def no_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("analytic equilibrium classification must not evolve a state")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(transit, "certify_relational_reflected_transit", forbidden)
        patch.setattr(transit, "validated_taylor_step", forbidden)
        patch.setattr(_validated_taylor, "validated_taylor_step", forbidden)
        yield


@pytest.fixture(scope="module")
def certificates(no_trajectory):
    return {
        branch: owner.certify_relational_reflected_equilibrium(
            branch[0], model=MODEL, orientation=branch[1]
        )
        for branch in BRANCHES
    }


def _phases(report):
    _, _, _, _, a, b, capital_a, capital_b = report.coordinates
    return (a, -a, -b, I(0), b, capital_a, -capital_a, -capital_b, I(0), capital_b)


def _laplacian(size, edges, weights):
    rows = [[Q(0)] * size for _ in range(size)]
    for (left, right), weight in zip(edges, weights):
        rows[left][left] += weight
        rows[right][right] += weight
        rows[left][right] -= weight
        rows[right][left] -= weight
    return tuple(tuple(row) for row in rows)


def _transpose(matrix):
    return tuple(tuple(row) for row in zip(*matrix))


def _congruence(matrix, columns):
    basis = _transpose(columns)
    return exact_matrix_product(_transpose(basis), exact_matrix_product(matrix, basis))


def _quadratic(matrix, vector):
    return sum(
        (
            left * value * right
            for left, row in zip(vector, matrix)
            for value, right in zip(row, vector)
        ),
        Q(0),
    )


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


@pytest.mark.parametrize("branch", BRANCHES)
def test_named_branches_satisfy_independent_full_graph_phase_equations(
    certificates, branch
):
    report = certificates[branch]
    phases = _phases(report)
    graph = nx.Graph(EDGES)
    assert len(report.coordinates) == len(report.field_rate_enclosures) == 8
    assert all(value == I(0) for value in report.coordinates[:4])
    assert min(report.domain_lower_bounds) > 0
    assert all(value.contains(0) for value in report.field_rate_enclosures)
    assert all(value.width < Q(1, 10**18) for value in report.field_rate_enclosures)
    for node in range(10):
        differences = tuple(phases[other] - phases[node] for other in graph[node])
        imaginary = sum((sin(gap) for gap in differences), I(0))
        real = sum((cos(gap) for gap in differences), I(0))
        assert imaginary.contains(0) and imaginary.width < Q(1, 10**18)
        assert real.lo > 0
    independent_storage = sum((1 - cos(phases[j] - phases[i]) for i, j in EDGES), I(0))
    assert (report.storage - independent_storage).contains(0)


def test_opposite_branch_isolates_an_exact_cubic_root_and_its_angle(certificates):
    report = certificates[("opposite_twist", 1)]
    root = report.root_cosine
    assert root is not None

    def polynomial(value):
        return 16 * value**3 - 8 * value + 1

    assert Q(1, 8) < root.lo <= root.hi < Q(1, 7)
    assert polynomial(root.lo) >= 0 >= polynomial(root.hi)
    assert (48 * I(Q(1, 8), Q(1, 7)) ** 2 - 8).hi < 0
    assert root.width < Q(1, 10**25)
    _, _, _, _, a, b, capital_a, capital_b = report.coordinates
    assert (cos(b) - root).contains(0)
    assert (a - 2 * b).contains(0)
    assert (capital_a + a).contains(0) and (capital_b + b).contains(0)
    assert (2 * sin(2 * a) + sin(a / 2)).contains(0)
    assert report.cycle_port_cosine == report.bridge_cosine
    assert report.cycle_path_cosine == root


def test_seven_distinct_lift_branches_keep_winding_and_the_seven_beta_separator(
    certificates,
):
    assert len(certificates) == 7
    coordinates = tuple(report.coordinates for report in certificates.values())
    assert all(
        any(
            left.hi < right.lo or right.hi < left.lo
            for left, right in zip(first, second)
        )
        for index, first in enumerate(coordinates)
        for second in coordinates[index + 1 :]
    )
    assert certificates[("consensus", 1)].storage.contains(0)
    assert certificates[("consensus", 1)].winding == (0, 0)
    for sign in (1, -1):
        aligned = certificates[("aligned_twist", sign)]
        saddle = certificates[("aligned_saddle", sign)]
        opposite = certificates[("opposite_twist", sign)]
        assert aligned.winding == saddle.winding == (sign, sign)
        assert opposite.winding == (sign, -sign)
        assert (aligned.storage - 10 * (1 - cos(2 * pi_interval() / 5))).contains(0)
        assert aligned.storage.hi < 7
        assert saddle.storage.contains(7)
        # Despite zero collective mean, these minima require storage above 7.
        assert opposite.storage.lo > 7
        assert (opposite.coordinates[4] + opposite.coordinates[6]).contains(0)


@pytest.mark.parametrize(
    "branch", tuple(branch for branch in BRANCHES if branch[0] != "aligned_saddle")
)
def test_minima_have_exact_positive_full_network_quotient_stiffness(
    certificates, branch
):
    report = certificates[branch]
    # Rational edge lower bounds give a smaller quadratic form than the true K.
    weights = (report.cycle_port_cosine.lo,) + (report.cycle_path_cosine.lo,) * 4
    weights = weights * 2 + (report.bridge_cosine.lo,) * 2
    assert min(weights) > 0
    stiffness = _laplacian(10, EDGES, weights)
    assert all(sum(row) == 0 for row in stiffness)
    grounded = tuple(row[:-1] for row in stiffness[:-1])
    assert exact_symmetric_semidefinite(grounded, strict=True)
    assert report.phase_hessian_inertia == (9, 0, 1)
    assert report.quotient_modes == (18, 0, 0)
    assert report.stability == "locally_exponentially_stable_modulo_offsets"


def test_saddle_signature_has_an_exact_full_network_congruence_and_negative_witness(
    certificates,
):
    ring_edges = EDGES[:5]
    weights = (Q(-1, 2),) + (Q(1, 2),) * 4
    ring = _laplacian(5, ring_edges, weights)
    stiffness = _laplacian(10, EDGES, weights * 2 + (Q(1), Q(1)))
    identity = tuple(tuple(Q(i == j) for i in range(5)) for j in range(5))
    even = tuple(vector + vector for vector in identity)
    odd = tuple(vector + tuple(-value for value in vector) for vector in identity)
    blocks = _congruence(stiffness, even + odd)
    assert all(blocks[i][j] == 0 for i in range(5) for j in range(5, 10))
    assert tuple(row[:5] for row in blocks[:5]) == tuple(
        tuple(2 * value for value in row) for row in ring
    )
    odd_block = tuple(row[5:] for row in blocks[5:])
    assert exact_symmetric_semidefinite(odd_block, strict=True)
    witness = tuple(map(Q, (1, -1, Q(-1, 2), 0, Q(1, 2))))
    assert _quadratic(ring, witness) == Q(-3, 2)
    assert _quadratic(stiffness, witness + witness) == -3
    # delta=v0-v1 vanishes on this three-dimensional positive subspace.
    positive = tuple(
        tuple(map(Q, row))
        for row in ((1, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 1))
    )
    assert exact_symmetric_semidefinite(_congruence(ring, positive), strict=True)
    assert all(sum(row) == 0 for row in ring)
    # Three positive, one negative and the constant null vector exhaust dim 5.
    for sign in (1, -1):
        report = certificates[("aligned_saddle", sign)]
        assert report.cycle_port_cosine == I(Q(-1, 2))
        assert report.cycle_path_cosine == I(Q(1, 2))
        assert report.bridge_cosine == I(1)
        assert report.phase_hessian_inertia == (8, 1, 1)
        assert report.quotient_modes == (17, 1, 0)
        assert report.stability == "hyperbolic_saddle_modulo_offsets"


@pytest.mark.parametrize("branch", BRANCHES)
def test_twenty_coordinate_native_tangent_wiring_is_a_separate_numerical_check(
    certificates, branch
):
    report = certificates[branch]
    phases = tuple(float(value.midpoint) for value in _phases(report))
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(EDGES)
    graph.graph.update(_t=9, payload={"history": [1, 2]})
    for i in graph:
        graph.nodes[i].update(EPI=-0.75, theta=phases[i], nu_f=1, delta_nfr=123)
    before = _snapshot(graph)
    tangent = relational.evaluate_relational_uniform_tangent(graph, model=MODEL)
    assert _snapshot(graph) == before
    adjacency = nx.to_numpy_array(graph, nodelist=tuple(range(10)))
    degree = adjacency.sum(axis=1)
    support = np.diag(degree) - adjacency
    cosines = adjacency * np.cos(np.subtract.outer(phases, phases))
    stiffness = np.diag(cosines.sum(axis=1)) - cosines
    inverse_metric = np.diag(1 / (math.pi * cosines.sum(axis=1)))
    e, w = MODEL.effective_weights
    expected = np.block(
        [
            [-e * np.diag(1 / degree) @ support, -w * inverse_metric @ stiffness],
            [(w / MODEL.storage_scale) * inverse_metric @ support, np.zeros((10, 10))],
        ]
    )
    np.testing.assert_allclose(tangent.generator, expected, rtol=2e-14, atol=2e-15)
    assert max(map(abs, tangent.field.form_rate + tangent.field.phase_rate)) < 2e-15
    # These eigenvalues check adapter wiring, not the analytic verdict above.
    eigenvalues = np.linalg.eigvals(np.asarray(tangent.generator))
    assert np.count_nonzero(eigenvalues.real > 1e-8) == report.quotient_modes[1]
    assert np.count_nonzero(eigenvalues.real < -1e-8) == report.quotient_modes[0]
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-8) == 2


def test_storage_scale_and_positive_damping_leave_named_equilibria_unchanged(
    certificates,
):
    model = RelationalExchangeModel(
        3, epi_weight=3, phase_weight=1, phase_domain="regular"
    )
    report = owner.certify_relational_reflected_equilibrium(
        "aligned_saddle", model=model
    )
    base = certificates[("aligned_saddle", 1)]
    assert report.model is model
    assert report.coordinates == base.coordinates
    assert report.storage.contains(21)
    assert (report.storage - 3 * base.storage).contains(0)
    assert report.quotient_modes == base.quotient_modes


@pytest.mark.parametrize(
    "family", (None, True, 1, "twist", "consensus ", ("consensus",))
)
def test_invalid_family_is_rejected_without_guessing(family):
    with pytest.raises(ValueError, match="family"):
        owner.certify_relational_reflected_equilibrium(family, model=MODEL)


@pytest.mark.parametrize("orientation", (True, False, 0, 2, 1.0, Q(1), "1", None))
def test_orientation_requires_an_exact_signed_integer(orientation):
    with pytest.raises(ValueError, match="orientation"):
        owner.certify_relational_reflected_equilibrium(
            "aligned_twist", model=MODEL, orientation=orientation
        )


def test_consensus_does_not_duplicate_an_orientation_independent_state():
    with pytest.raises(ValueError, match="orientation-independent"):
        owner.certify_relational_reflected_equilibrium(
            "consensus", model=MODEL, orientation=-1
        )


@pytest.mark.parametrize(
    "model",
    (
        None,
        {},
        RelationalExchangeModel(1),
        RelationalExchangeModel(1, phase_domain="positive_resultant"),
    ),
)
def test_regular_law_is_explicit_and_not_silently_substituted(model):
    with pytest.raises(ValueError, match="regular"):
        owner.certify_relational_reflected_equilibrium("consensus", model=model)


def test_zero_damping_cannot_inherit_the_exponential_recovery_certificate():
    model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    with pytest.raises(ValueError, match="positive epi_weight"):
        owner.certify_relational_reflected_equilibrium("consensus", model=model)


def test_export_retains_exact_root_coordinates_and_the_limited_claim(certificates):
    report = certificates[("opposite_twist", -1)]
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-reflected-equilibrium.v1"
    projected = payload["report"]
    assert Q(**projected["root_cosine"]["lo"]) == report.root_cosine.lo
    assert Q(**projected["coordinates"][4]["hi"]) == report.coordinates[4].hi
    assert projected["phase_hessian_inertia"] == [9, 0, 1]
    assert projected["quotient_modes"] == [18, 0, 0]
    assert (
        "complete_equilibrium_list_only_in_inherited_reflection_lift"
        in projected["scope"]
    )
    assert (
        "no_formation_global_regular_continuation_or_physical_identification"
        in projected["scope"]
    )
