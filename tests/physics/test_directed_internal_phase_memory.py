"""Derived angular readout and hidden memory of one fixed directed C3.

The complete fine law here is the existing outgoing pure-EPI diffusion with
fixed unit conductance and common positive capacity. Its contrast angle is an
observation of form, not the independently stored primitive phase. Exact
matrix identities establish rotation and decay without a numerical trajectory,
an imposed oscillator, a fitted clock, or a sustaining source. The reversible
``epi_memory`` graph adapter is deliberately not applied to directed support.
"""

from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.mathematics.cayley import cayley_laplacian
from tnfr.physics.directed_diffusion import (
    directed_cayley_adjacency,
    directed_rw_laplacian,
)


def _system(s, connection):
    """Identify the materialized outgoing operator with its exact unit model."""
    adjacency = directed_cayley_adjacency(3, connection)
    represented = directed_rw_laplacian(adjacency)
    laplacian = s.Matrix(
        [[s.Rational(Fraction(float(value))) for value in row] for row in represented]
    )
    assert laplacian == s.Matrix(cayley_laplacian(3, connection))
    basis = s.Matrix.hstack(
        s.Matrix([1, -1, 0]) / s.sqrt(2),
        s.Matrix([1, 1, -2]) / s.sqrt(6),
    )
    generator = -laplacian
    reduced = (basis.T * generator * basis).applyfunc(s.simplify)
    return generator, basis, reduced


@pytest.mark.parametrize("connection,orientation", [({1}, -1), ({2}, 1), ({1, 2}, 0)])
def test_actual_outgoing_generator_fixes_rotation_sign_and_reciprocal_control(
    connection, orientation
):
    s = pytest.importorskip("sympy")
    generator, basis, reduced = _system(s, connection)
    quarter_turn = s.Matrix([[0, -1], [1, 0]])
    expected = -s.Rational(3, 2) * s.eye(2)
    expected += orientation * s.sqrt(3) * quarter_turn / 2

    assert basis.T * basis == s.eye(2)
    assert basis.T * s.ones(3, 1) == s.zeros(2, 1)
    assert generator * s.ones(3, 1) == s.zeros(3, 1)
    assert s.ones(1, 3) * generator == s.zeros(1, 3)
    assert reduced == expected
    assert (generator * basis - basis * reduced).applyfunc(s.simplify) == s.zeros(3, 2)
    assert (basis.T * generator - reduced * basis.T).applyfunc(s.simplify) == s.zeros(
        2, 3
    )


def test_derived_angle_turns_while_exact_form_norm_decays():
    s = pytest.importorskip("sympy")
    _, _, reduced = _system(s, {1})
    capacity = s.Symbol("nu", positive=True)
    first, second = s.symbols("first second", real=True)
    state = s.Matrix([first, second])
    rate = capacity * reduced * state
    norm_squared = state.dot(state)
    norm_rate = s.expand(2 * state.dot(rate))
    oriented_rate = s.expand(first * rate[1] - second * rate[0])

    assert s.expand(norm_rate + 3 * capacity * norm_squared) == 0
    assert s.simplify(oriented_rate + s.sqrt(3) * capacity * norm_squared / 2) == 0
    # Only on nonzero contrast: arg(z)'=cross(z,z')/|z|^2. No angle is
    # assigned at consensus, where the Cartesian form rate is exactly zero.
    assert rate.subs({first: 0, second: 0}) == s.zeros(2, 1)
    radius_rate = -3 * capacity / 2
    angular_speed = s.sqrt(3) * capacity / 2
    full_turn_time = 2 * s.pi / angular_speed
    assert s.simplify(s.exp(radius_rate * full_turn_time)) == s.exp(
        -2 * s.sqrt(3) * s.pi
    )
    # Capacity rescales the time parameter, not this strictly contracting
    # amplitude ratio per turn. The result supplies no persistent identity.
    assert s.simplify(radius_rate / angular_speed) == -s.sqrt(3)


def test_elimination_derives_signed_memory_initial_source_and_second_order_row():
    s = pytest.importorskip("sympy")
    _, _, reduced = _system(s, {1})
    capacity = s.Symbol("nu", positive=True)
    time = s.Symbol("t", nonnegative=True)
    first, second = s.symbols("first second", real=True)
    a, b = 3 * capacity / 2, s.sqrt(3) * capacity / 2
    generator = capacity * reduced
    decay_operator = -generator
    project, lift = s.Matrix([[1, 0]]), s.Matrix([1, 0])
    hidden = s.eye(2) - lift * project
    hidden_semigroup = s.diag(1, s.exp(-a * time))
    assert hidden * decay_operator * hidden == s.diag(0, a)

    kernel = (
        project * decay_operator * hidden_semigroup * hidden * decay_operator * lift
    )[0]
    source = (
        -project
        * decay_operator
        * hidden_semigroup
        * hidden
        * s.Matrix([first, second])
    )[0]
    assert s.simplify(kernel + b**2 * s.exp(-a * time)) == 0
    assert s.simplify(source - b * second * s.exp(-a * time)) == 0
    # These are the algebraic elimination identities underlying the existing
    # memory owner. The negative kernel is not its reversible Gram-positive
    # specialization, and the initial hidden-state term cannot be omitted.
    assert (project * decay_operator * lift)[0] == a
    row = project * generator
    assert (row * generator + 2 * a * row + (a**2 + b**2) * project).applyfunc(
        s.simplify
    ) == s.zeros(1, 2)
    assert s.simplify(2 * a) == 3 * capacity
    assert s.simplify(a**2 + b**2) == 3 * capacity**2

    # Verify the full source/convolution identity for arbitrary visible form,
    # not just a selected exponential solution or numerically sampled path.
    tau = s.Symbol("tau", nonnegative=True)
    visible = s.Function("visible")
    initial_hidden = s.Symbol("initial_hidden", real=True)
    convolution = s.Integral(s.exp(-a * (time - tau)) * visible(tau), (tau, 0, time))
    reconstructed_hidden = s.exp(-a * time) * initial_hidden - b * convolution
    assert (
        s.simplify(
            s.diff(reconstructed_hidden, time)
            + a * reconstructed_hidden
            + b * visible(time)
        )
        == 0
    )
    assert reconstructed_hidden.subs(time, 0).doit() == initial_hidden
    visible_rate = -a * visible(time) + b * reconstructed_hidden
    assert (
        s.expand(
            visible_rate
            + a * visible(time)
            - b * s.exp(-a * time) * initial_hidden
            + b**2 * convolution
        )
        == 0
    )


def test_fresh_engine_pressure_retains_hidden_form_information_without_primitive_phase():
    s = pytest.importorskip("sympy")
    generator, basis, _ = _system(s, {1})
    states = (
        (Fraction(5, 8), Fraction(3, 8), Fraction(1, 2)),
        (Fraction(11, 16), Fraction(7, 16), Fraction(3, 8)),
    )
    visible, visible_rates, hidden = [], [], []
    for initial in states:
        graph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        graph.graph.update(
            DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
            GAMMA={"type": "none"},
            use_extended_dynamics=False,
        )
        for node, value in zip(graph, initial, strict=True):
            graph.nodes[node].update(EPI=float(value), nu_f=1.0, theta=0.0)
        default_compute_delta_nfr(graph)
        pressure = s.Matrix(
            [
                s.Rational(Fraction(get_attr(graph.nodes[node], ALIAS_DNFR, None)))
                for node in graph
            ]
        )
        exact_initial = s.Matrix(initial)
        assert all(0 < value < 1 for value in initial)
        assert pressure == generator * exact_initial
        assert tuple(
            get_attr(graph.nodes[node], ALIAS_EPI, None) for node in graph
        ) == tuple(map(float, initial))
        assert all(
            get_attr(graph.nodes[node], ALIAS_THETA, None) == 0 for node in graph
        )
        coordinates = (basis.T * exact_initial).applyfunc(s.simplify)
        rate = (basis.T * pressure).applyfunc(s.simplify)
        visible.append(coordinates[0])
        hidden.append(coordinates[1])
        visible_rates.append(rate[0])

    assert visible == [s.sqrt(2) / 8] * 2
    assert hidden == [0, s.sqrt(6) / 16]
    assert visible_rates == [-3 * s.sqrt(2) / 16, -3 * s.sqrt(2) / 32]
    # A scalar instantaneous observation cannot predict both derivatives.
    # Keeping the second form coordinate, or its exact memory and initial
    # source, resolves this obstruction without changing stored phase.
