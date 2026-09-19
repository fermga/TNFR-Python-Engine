"""Derived regional form angles on two explicitly supplied directed triangles.

Only pure-EPI outgoing diffusion is used, with fixed common capacity. The
regional complex coordinates encode real scalar form; they are not written
into primitive nodal phase. No new law, clock, event or engine path is added.
"""

import networkx as nx
import numpy as np
import pytest

from tests.physics._internal_mode_fixture import NODES, Q_MODE, P
from tnfr.physics.directed_diffusion import (
    directed_cayley_adjacency,
    directed_rw_laplacian,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator


@pytest.fixture(scope="module")
def model():
    s = pytest.importorskip("sympy")
    shift_array = directed_cayley_adjacency(3, {1})
    shift = s.Matrix(shift_array.astype(int))
    adjacency = s.BlockMatrix([[shift, s.eye(3)], [s.eye(3), shift]]).as_explicit()
    generator = adjacency / 2 - s.eye(6)
    basis = s.Matrix.hstack(s.Matrix(P) / s.sqrt(2), s.Matrix(Q_MODE) / s.sqrt(6))
    lift = s.diag(basis, basis)
    reduced = s.simplify(lift.T * generator * lift)
    return s, adjacency, generator, basis, lift, reduced


def test_declared_graph_and_both_production_readers_bind_the_exact_reduction(model):
    s, adjacency, generator, basis, lift, reduced = model
    graph = nx.DiGraph()
    graph.add_nodes_from(NODES)
    for i, node in enumerate(NODES):
        graph.nodes[node].update(EPI=0.5, theta=0.0, nu_f=1.0)
        for j, other in enumerate(NODES):
            if adjacency[i, j]:
                graph.add_edge(node, other, weight=1.0)
    assert graph.number_of_edges() == 12
    assert tuple(dict(graph.out_degree()).values()) == (2,) * 6
    nodes, laplacian = structural_diffusion_operator(graph)
    assert tuple(nodes) == NODES
    outgoing = directed_rw_laplacian(np.asarray(adjacency, dtype=float))
    np.testing.assert_array_equal(laplacian, outgoing)
    # All matrix coefficients are dyadic here; only the modal basis has surds.
    assert -s.Matrix(laplacian).applyfunc(s.Rational) == generator
    assert basis.T * basis == s.eye(2)
    assert s.simplify(generator * lift - lift * reduced) == s.zeros(6, 4)
    assert s.simplify(lift.T * generator - reduced * lift.T) == s.zeros(4, 6)
    means = s.diag(s.ones(3, 1), s.ones(3, 1))
    mean_generator = s.Matrix([[-1, 1], [1, -1]]) / 2
    assert generator * means == means * mean_generator
    assert s.simplify(lift.T * means) == s.zeros(4, 2)
    assert means.row_join(lift).rank() == 6


def test_joint_cartesian_law_and_complete_loss_follow_from_support(model):
    s, _, _, _, _, reduced = model
    nu = s.symbols("nu", positive=True)
    u0, v0, u1, v1 = s.symbols("u0 v0 u1 v1", real=True)
    state = s.Matrix([u0, v0, u1, v1])
    rate = nu * reduced * state
    diagonal = s.Matrix([[-5, s.sqrt(3)], [-s.sqrt(3), -5]]) / 4
    assert (
        reduced
        == s.BlockMatrix(
            [[diagonal, s.eye(2) / 2], [s.eye(2) / 2, diagonal]]
        ).as_explicit()
    )
    total = state.dot(state)
    gap = (u0 - u1) ** 2 + (v0 - v1) ** 2
    assert s.expand(2 * state.dot(rate) + 3 * nu * total / 2 + nu * gap) == 0
    # A rotating, coupled form still loses contrast. No amplitude maintenance
    # is inferred from the skew part, phase locking, or an observable period.


def test_polar_rate_is_derived_with_an_amplitude_ratio_not_a_fitted_phase_gain(model):
    s, _, _, _, _, reduced = model
    nu, r0, r1 = s.symbols("nu r0 r1", positive=True)
    phi0, phi1 = s.symbols("phi0 phi1", real=True)
    state = s.Matrix(
        [r0 * s.cos(phi0), r0 * s.sin(phi0), r1 * s.cos(phi1), r1 * s.sin(phi1)]
    )
    rate = nu * reduced * state
    expected_radial = (
        -5 * nu * r0 / 4 + nu * r1 * s.cos(phi1 - phi0) / 2,
        -5 * nu * r1 / 4 + nu * r0 * s.cos(phi1 - phi0) / 2,
    )
    expected_angular = (
        -s.sqrt(3) * nu / 4 + nu * r1 * s.sin(phi1 - phi0) / (2 * r0),
        -s.sqrt(3) * nu / 4 + nu * r0 * s.sin(phi0 - phi1) / (2 * r1),
    )
    for a, radius in enumerate((r0, r1)):
        offset = 2 * a
        form = state[offset : offset + 2, 0]
        velocity = rate[offset : offset + 2, 0]
        radial = form.dot(velocity) / radius
        angular = s.det(s.Matrix.hstack(form, velocity)) / radius**2
        assert s.trigsimp(radial - expected_radial[a]) == 0
        assert s.trigsimp(angular - expected_angular[a]) == 0
    delta_rate = expected_angular[1] - expected_angular[0]
    assert (
        s.trigsimp(delta_rate + nu * (r0 / r1 + r1 / r0) * s.sin(phi1 - phi0) / 2) == 0
    )
    # Equal amplitudes are an invariant special family, not a general closure.
    assert s.simplify((expected_radial[0] - expected_radial[1]).subs(r1, r0)) == 0
    assert s.trigsimp(delta_rate.subs(r1, r0) + nu * s.sin(phi1 - phi0)) == 0


def test_equal_observed_angles_can_have_different_future_phase_rates(model):
    s, _, _, _, _, reduced = model
    # These rows have equal regional phases (0, pi/2) and equal regional means.
    # Arbitrarily small common scaling places the fine EPI in its allowed band.
    first = s.Matrix([1, 0, 0, 1]) / 16
    second = s.Matrix([1, 0, 0, 2]) / 16

    def angular_at_first(state):
        velocity = reduced * state
        return s.det(s.Matrix.hstack(state[:2, 0], velocity[:2, 0])) / state[:2, 0].dot(
            state[:2, 0]
        )

    assert s.simplify(angular_at_first(first) - (s.Rational(1, 2) - s.sqrt(3) / 4)) == 0
    assert s.simplify(angular_at_first(second) - (1 - s.sqrt(3) / 4)) == 0
    assert s.simplify(angular_at_first(second) - angular_at_first(first)) == s.Rational(
        1, 2
    )
    # Hence the two angles alone cannot replace the complete regional form.
    # The finite Cartesian law remains defined if an amplitude reaches zero.
    zero = s.Matrix([0, 0, 1, 0])
    assert reduced * zero == s.Matrix(
        [s.Rational(1, 2), 0, -s.Rational(5, 4), -s.sqrt(3) / 4]
    )
