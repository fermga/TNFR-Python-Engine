"""Exact diffusion matching does not select an inertial constitutive law.

The fixed unit prism supplies its existing exact EPI transport. The auxiliary
second-order equations here are conditional algebraic comparisons, not native
phase laws, trajectories or new production dynamics. K denotes acceleration
stiffness, so the selected degree-three canonical chart has Hess(U)=3*K.
"""

import pytest

from tests.physics._internal_mode_fixture import P, _exact_generator, _graph
from tnfr.physics.support_transport import observe_support_transport


@pytest.fixture(scope="module")
def transport():
    s = pytest.importorskip("sympy")
    graph = _graph()
    snapshot = observe_support_transport(graph)
    lap = -s.Matrix(_exact_generator(snapshot))
    project = s.eye(6) - s.ones(6) / 6
    assert lap == lap.T and lap * project == project * lap == lap
    assert lap.eigenvals() == {0: 1, s.Rational(2, 3): 1, 1: 2, s.Rational(5, 3): 2}
    return s, graph, lap, project


def test_exact_invariant_diffusion_graph_fixes_stiffness_only_after_damping(transport):
    s, graph, lap, project = transport
    e, gamma, rate, eigenvalue = s.symbols("e gamma rate eigenvalue", positive=True)
    stiffness = gamma * e * lap - e**2 * lap**2
    generator = (
        s.zeros(6).row_join(s.eye(6)).col_join((-stiffness).row_join(-gamma * s.eye(6)))
    )
    # The graph of v=-e*L*y is invariant exactly when the lower block agrees.
    # With K*1=0, its condition on centered y fixes all of K, but not gamma.
    lift = project.col_join(-e * lap)
    arbitrary_stiffness = s.Matrix(6, 6, s.symbols("k0:36"))
    arbitrary_generator = (
        s.zeros(6)
        .row_join(s.eye(6))
        .col_join((-arbitrary_stiffness).row_join(-gamma * s.eye(6)))
    )
    arbitrary_residual = (arbitrary_generator * lift - lift * (-e * lap)).applyfunc(
        s.expand
    )
    assert arbitrary_residual == s.zeros(6).col_join(
        (stiffness - arbitrary_stiffness * project).applyfunc(s.expand)
    )
    assert (generator * lift - lift * (-e * lap)).applyfunc(s.expand) == s.zeros(12, 6)
    assert stiffness * project == stiffness
    assert stiffness.diff(gamma) == e * lap != s.zeros(6)
    polynomial = rate**2 + gamma * rate + gamma * e * eigenvalue - e**2 * eigenvalue**2
    assert (
        s.expand(polynomial - (rate + e * eigenvalue) * (rate + gamma - e * eigenvalue))
        == 0
    )
    # Thus diffusion rate -e*lambda and the other rate -(gamma-e*lambda)
    # coexist. Selecting this invariant graph also specifies initial velocity.
    assert (0, 0) in graph and (1, 1) not in graph.neighbors((0, 0))
    assert lap[0, 4] == 0 and (lap**2)[0, 4] == s.Rational(2, 9)
    assert stiffness[0, 4] == -2 * e**2 / 9
    # No choice of scalar damping removes this distance-two acceleration.


def test_leading_overdamped_match_has_nonzero_exact_tangency_residual(transport):
    s, graph, lap, project = transport
    e, gamma = s.symbols("e gamma", positive=True)
    leading_stiffness = gamma * e * lap
    lift = project.col_join(-e * lap)
    generator = (
        s.zeros(6)
        .row_join(s.eye(6))
        .col_join((-leading_stiffness).row_join(-gamma * s.eye(6)))
    )
    residual = (generator * lift - lift * (-e * lap)).applyfunc(s.expand)
    assert residual == s.zeros(6).col_join(-(e**2) * lap**2)
    original = s.Matrix(P * 2) / 4
    actual_form = s.Matrix([data["EPI"] for _, data in graph.nodes(data=True)])
    assert project * actual_form == original
    assert lap * original == original and original.dot(original) == s.Rational(1, 4)
    assert residual[6:, :] * original == -(e**2) * original != s.zeros(6, 1)
    # Dividing by gamma makes a singular perturbation with tau=1/gamma;
    # discarding tau*v_dot is a leading balance, not exact finite-tau equality.
    tau = s.symbols("tau", positive=True)
    velocity = -e * lap * original
    required_acceleration = -e * lap * velocity
    assert (
        tau * required_acceleration + velocity + e * lap * original
        == tau * e**2 * original
    )


def test_positive_embedding_family_still_dissipates_its_hamiltonian(transport):
    s, _, lap, _ = transport
    e, excess = s.symbols("e excess", positive=True)
    lambda_max = max(lap.eigenvals())
    gamma = e * lambda_max + excess
    stiffness = gamma * e * lap - e**2 * lap**2
    eigenvalues = lap.eigenvals()
    assert (
        sum(multiplicity for value, multiplicity in eigenvalues.items() if value > 0)
        == 5
    )
    for value in eigenvalues:
        if value > 0:
            assert (e * value * (excess + e * (lambda_max - value))).is_positive
    # Every excess>0 gives positive centered stiffness. Damping remains free.
    y = s.Matrix(s.symbols("y0:6", real=True))
    v = s.Matrix(s.symbols("v0:6", real=True))
    energy = 3 * (v.dot(v) + y.dot(stiffness * y)) / 2
    gradient_y = s.Matrix([s.diff(energy, item) for item in y])
    gradient_v = s.Matrix([s.diff(energy, item) for item in v])
    acceleration = -stiffness * y - gamma * v
    energy_rate = s.expand(gradient_y.dot(v) + gradient_v.dot(acceleration))
    assert s.expand(energy_rate + 3 * gamma * v.dot(v)) == 0
    # A periodic solution must have v=0 throughout from this strict balance;
    # positive centered stiffness then forces y=0. This is not maintenance.


def test_conservative_exact_embedding_requires_negative_centered_stiffness(transport):
    s, _, lap, project = transport
    e = s.symbols("e", positive=True)
    stiffness = -(e**2) * lap**2
    assert stiffness * project == stiffness
    for value in lap.eigenvals():
        if value > 0:
            assert (-(e**2) * value**2).is_negative
    # The exact gamma=0 embedding has roots +/- e*lambda: diffusion is its
    # selected stable graph, paired with an unstable graph, not a positive wave.
    rate, eigenvalue = s.symbols("rate eigenvalue", real=True)
    assert (
        s.expand((rate + e * eigenvalue) * (rate - e * eigenvalue))
        == rate**2 - e**2 * eigenvalue**2
    )
    original = s.Matrix(P * 2) / 4
    assert s.expand(3 * original.dot(stiffness * original) / 2) == -3 * e**2 / 8
    embedded_velocity = -e * lap * original
    assert (
        s.expand(
            3
            * (
                embedded_velocity.dot(embedded_velocity)
                + original.dot(stiffness * original)
            )
            / 2
        )
        == 0
    )
    assert lap.rank() == 5 and (lap**2).rank() == 5
    # Positive conservative stiffness cannot preserve pure diffusion on an
    # open centered chart; agreement only at consensus is a weaker statement.
