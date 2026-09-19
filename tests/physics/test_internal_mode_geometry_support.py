"""Declared prism geometry can do energy work without maintaining internal form.

These are exact continuous-model identities and one detached coefficient jet.
The analytic history is supplied as a counterexample to an unqualified
asymptotic claim; it is not a derived conductance law or an engine trajectory.
Primitive phase, support, the internal basis and metric lengths remain fixed.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import (
    GRAM,
    LIFT,
    NODES,
    PROJECTION,
    _apply,
    _graph,
    _inner,
)
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_derivative,
)


def _symbolic_prism():
    symbolic = pytest.importorskip("sympy")
    a, b, nu, e = symbolic.symbols("a b nu e", positive=True)
    graph = _graph()
    nx.set_edge_attributes(graph, 1, "length")
    adjacency = symbolic.Matrix(
        [
            [a if i[0] == j[0] else b if graph.has_edge(i, j) else 0 for j in NODES]
            for i in NODES
        ]
    )
    # A fiber is K3, so its diagonal is zero rather than another internal edge.
    for i in range(6):
        adjacency[i, i] = 0
    degree = 2 * a + b
    fine = nu * e * (adjacency / degree - symbolic.eye(6))
    induced = (
        nu
        * e
        / degree
        * symbolic.Matrix(
            [
                [-(3 * a + b), 0, b, 0],
                [0, -(3 * a + b), 0, b],
                [b, 0, -(3 * a + b), 0],
                [0, b, 0, -(3 * a + b)],
            ]
        )
    )
    return symbolic, (a, b, nu, e), adjacency, fine, induced


def test_two_conductances_induce_the_normalized_modal_law_on_actual_prism():
    s, (a, b, nu, e), adjacency, fine, induced = _symbolic_prism()
    assert adjacency * s.ones(6, 1) == (2 * a + b) * s.ones(6, 1)
    lift, projection = s.Matrix(LIFT), s.Matrix(PROJECTION)
    assert s.simplify(fine * lift - lift * induced) == s.zeros(6, 4)
    assert s.simplify(projection * fine - induced * projection) == s.zeros(4, 6)
    # The coefficient includes row normalization; 3a is genuine internal loss.
    assert s.simplify(induced[0, 0] + induced[0, 2]) == -3 * a * nu * e / (2 * a + b)


def test_fixed_internal_norm_strictly_decays_for_every_positive_geometry():
    s, (a, b, nu, e), _, _, induced = _symbolic_prism()
    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    metric = s.diag(*GRAM, *GRAM)
    squared = (state.T * metric * state)[0]
    difference = state[:2, :] - state[2:, :]
    difference_squared = (difference.T * s.diag(*GRAM) * difference)[0]
    derivative = 2 * (state.T * metric * induced * state)[0]
    expected = -2 * nu * e / (2 * a + b) * (3 * a * squared + b * difference_squared)
    assert s.simplify(derivative - expected) == 0
    assert (-6 * nu * e * a / (2 * a + b)).is_negative
    # squared>0 for every nonzero state because G=diag(2,6)>0; the other
    # squared term is nonnegative. No a' or b' appears in this fixed norm.
    assert all(value > 0 for value in metric.diagonal())


def test_symmetric_and_antisymmetric_modes_have_distinct_positive_decay_rates():
    s, (a, b, nu, e), _, _, induced = _symbolic_prism()
    identity = s.eye(2)
    transform = (
        s.BlockMatrix([[identity, identity], [identity, -identity]]).as_explicit() / 2
    )
    inverse = 2 * transform
    plus = 3 * a * nu * e / (2 * a + b)
    minus = (3 * a + 2 * b) * nu * e / (2 * a + b)
    assert transform * inverse == s.eye(4)
    expected = s.diag(-plus, -plus, -minus, -minus)
    assert s.simplify(transform * induced * inverse - expected) == s.zeros(4)
    assert plus.is_positive and minus.is_positive
    assert s.simplify(minus - plus) == 2 * b * nu * e / (2 * a + b)
    # z_plus=(z0+z1)/2 and z_minus=(z0-z1)/2 use the same fixed G.
    assert inverse.T * s.diag(*GRAM, *GRAM) * inverse == 2 * s.diag(*GRAM, *GRAM)


def test_cumulative_exposure_formula_separates_decay_from_pointwise_positivity():
    s = pytest.importorskip("sympy")
    time = s.Symbol("time", nonnegative=True)
    plus_exposure = s.Function("Lambda_plus")(time)
    minus_exposure = s.Function("Lambda_minus")(time)
    initial_plus, initial_minus = s.symbols("initial_plus initial_minus", real=True)
    plus = initial_plus * s.exp(-plus_exposure)
    minus = initial_minus * s.exp(-minus_exposure)
    # For locally integrable supplied rates, Lambda(t)=integral_0^t rate;
    # Lambda(0)=0 is needed for the displayed initial coefficients.
    assert s.diff(plus, time) == -s.diff(plus_exposure, time) * plus
    assert s.diff(minus, time) == -s.diff(minus_exposure, time) * minus
    lp, lm = s.symbols("lp lm", nonnegative=True)
    energy_plus, energy_minus = s.symbols("energy_plus energy_minus", positive=True)
    norm_squared = 2 * (energy_plus * s.exp(-2 * lp) + energy_minus * s.exp(-2 * lm))
    assert s.limit(s.limit(norm_squared, lp, s.oo), lm, s.oo) == 0
    finite_plus_limit = s.limit(norm_squared, lm, s.oo)
    assert finite_plus_limit == 2 * energy_plus * s.exp(-2 * lp)
    assert finite_plus_limit.is_positive
    # Each initially nonzero component vanishes iff its own cumulative
    # exposure diverges. Zero components impose no exposure requirement.
    assert norm_squared.subs(energy_plus, 0) == 2 * energy_minus * s.exp(-2 * lm)


def test_supplied_finite_exposure_history_leaves_a_nonzero_symmetric_amplitude():
    s = pytest.importorskip("sympy")
    time = s.Symbol("time", nonnegative=True)
    a = s.exp(-time)  # Supplied history, not an inferred constitutive rule.
    # b=nu=e=1; a remains positive at every finite time but has no lower bound.
    plus_rate = 3 * a / (2 * a + 1)
    minus_rate = (3 * a + 2) / (2 * a + 1)
    plus_exposure = s.Rational(3, 2) * s.log(3 / (1 + 2 * a))
    minus_exposure = 2 * time - s.log(3 / (1 + 2 * a)) / 2
    assert s.simplify(s.diff(plus_exposure, time) - plus_rate) == 0
    assert s.simplify(s.diff(minus_exposure, time) - minus_rate) == 0
    assert plus_exposure.subs(time, 0) == minus_exposure.subs(time, 0) == 0
    assert s.limit(plus_exposure, time, s.oo) == 3 * s.log(3) / 2
    assert s.limit(minus_exposure, time, s.oo) == s.oo
    amplitude_factor = s.exp(-plus_exposure)
    assert amplitude_factor.subs(time, 0) == 1
    assert s.limit(amplitude_factor, time, s.oo) == s.Pow(3, -s.Rational(3, 2))
    assert s.limit(amplitude_factor**2, time, s.oo) == s.Rational(1, 27)
    # Residual form comes from asymptotically vanishing internal conductance;
    # it is not active restoration or an autonomous persistent NFR theorem.


def test_internal_geometry_work_increases_energy_while_internal_form_decays():
    graph = _graph()
    nx.set_edge_attributes(graph, 1, "length")
    original = deepcopy(graph)
    observed = observe_support_transport(graph)
    # Bind a detached exact pure-EPI model. This does not refresh or write
    # live stored pressure and is not a binary64 execution certificate.
    source = replace(observed, stored_pressure=observed.epi_gradient)
    rates = tuple(
        Q(3) if source.nodes[i][0] == source.nodes[j][0] else Q(0)
        for i, j, _ in source.conductance
    )
    derivative = observe_support_transport_derivative(source, conductance_rates=rates)
    state = _apply(PROJECTION, derivative.source.epi)
    rate = _apply(PROJECTION, derivative.source.rate)
    squared = _inner(state[:2], state[:2]) + _inner(state[2:], state[2:])
    squared_rate = 2 * (_inner(state[:2], rate[:2]) + _inner(state[2:], rate[2:]))
    assert state == (Q(1, 4), 0, Q(1, 4), 0)
    assert squared == Q(1, 4) and squared_rate == Q(-1, 2)
    assert derivative.source.dirichlet_energy == Q(3, 8)
    assert derivative.nodal_work == Q(-3, 4)
    assert derivative.conductance_work == Q(9, 8)
    assert derivative.energy_rate == Q(3, 8)
    assert derivative.geometry_gradient_rate == observed.epi_gradient
    assert derivative.epi_gradient_rate == (0,) * 6
    assert nx.utils.graphs_equal(graph, original)
    assert set(nx.get_edge_attributes(graph, "length").values()) == {1}
