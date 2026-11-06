"""Independent complete-law and fixed-preparation storage-family controls."""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_resonance import (
    assess_bridge_storage_family,
    assess_sine_bridge_channels,
)

COEFFICIENTS = (Q(0), Q(4, 9), Q(1))
TARGET = tuple(Q(i % 6, 6) for i in range(12))


@pytest.fixture(scope="module")
def symbolic_law():
    s = pytest.importorskip("sympy")
    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    epsilon = s.Symbol("epsilon", nonnegative=True)
    x = s.symbols("x0:12", real=True)
    theta = s.symbols("theta0:12", real=True)
    delta = s.Symbol("delta", real=True)
    potential = (
        1
        - s.cos(delta)
        + epsilon * (s.Rational(2, 3) - s.cos(delta) + s.cos(delta) ** 3 / 3)
    )
    current = s.sin(delta) + epsilon * s.sin(delta) ** 3
    form_gradient = s.Matrix([sum(x[i] - x[j] for j in graph[i]) for i in graph])
    currents = s.Matrix(
        [sum(current.subs(delta, theta[j] - theta[i]) for j in graph[i]) for i in graph]
    )
    mobility = s.diag(*(s.Rational(1, graph.degree[i]) for i in graph))
    field = (mobility * currents).col_join(mobility * form_gradient)
    critical = dict(zip(x, (0,) * 12))
    critical.update({theta[i]: 2 * s.pi * s.Rational(i % 6, 6) for i in graph})
    tangent = field.jacobian((*x, *theta)).subs(critical).applyfunc(s.simplify)
    laplacian = form_gradient.jacobian(x)
    hessian = (-currents.jacobian(theta)).subs(critical).applyfunc(s.simplify)
    zero = s.zeros(12)
    metric = laplacian.row_join(zero).col_join(zero.row_join(hessian))
    return {
        "s": s,
        "graph": graph,
        "epsilon": epsilon,
        "x": x,
        "theta": theta,
        "delta": delta,
        "potential": potential,
        "current": current,
        "field": field,
        "critical": critical,
        "tangent": tangent,
        "laplacian": laplacian,
        "hessian": hessian,
        "metric": metric,
    }


@pytest.fixture(scope="module")
def source():
    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    return bound_relational_sine_exchange(graph, reference_model=model)


def _report(source, epsilon, *, reverse_cycles=False):
    left, right = tuple(range(6)), tuple(range(6, 12))
    if reverse_cycles:
        left = (left[0], *reversed(left[1:]))
        right = (right[0], *reversed(right[1:]))
    return assess_bridge_storage_family(
        source,
        left_cycle=left,
        right_cycle=right,
        target_phase_turns=TARGET,
        epsilon=epsilon,
    )


def test_symbolic_complete_family_keeps_criticality_and_a_strict_local_energy_minimum(
    symbolic_law,
):
    law = symbolic_law
    s, epsilon, delta = law["s"], law["epsilon"], law["delta"]
    potential, current = law["potential"], law["current"]
    assert s.trigsimp(s.diff(potential, delta) - current) == 0
    assert potential.subs(delta, 0) == 0
    assert s.diff(potential, delta, 2).subs(delta, 0) == 1
    assert (
        s.trigsimp(
            s.diff(potential, delta, 2)
            - s.cos(delta)
            - 3 * epsilon * s.sin(delta) ** 2 * s.cos(delta)
        )
        == 0
    )
    assert s.trigsimp(current + current.subs(delta, -delta)) == 0
    assert law["field"].subs(law["critical"]) == s.zeros(24, 1)
    curvature = s.simplify(s.diff(current, delta).subs(delta, s.pi / 3))
    assert curvature == s.Rational(1, 2) + 9 * epsilon / 8
    assert curvature.is_positive
    bridge_laplacian = s.zeros(12)
    bridge_laplacian[0, 0] = bridge_laplacian[6, 6] = 1
    bridge_laplacian[0, 6] = bridge_laplacian[6, 0] = -1
    expected_hessian = (
        curvature * (law["laplacian"] - bridge_laplacian) + bridge_laplacian
    )
    assert (law["hessian"] - expected_hessian).applyfunc(s.simplify) == s.zeros(12)
    assert law["hessian"] * s.ones(12, 1) == s.zeros(12, 1)
    # Positive edge stiffness on connected support gives positivity modulo the
    # single phase origin. This exact cofactor also excludes an extra zero mode.
    assert s.simplify(law["hessian"][:11, :11].det() - 36 * curvature**10) == 0
    conserved = law["tangent"].T * law["metric"] + law["metric"] * law["tangent"]
    assert conserved.applyfunc(s.simplify) == s.zeros(24)
    assert s.simplify(12 * potential.subs(delta, s.pi / 3)) == 6 + 5 * epsilon / 2
    # These are local Hessian and conservative identities. A lower energy at
    # consensus still prevents using target-relative energy globally.
    assert potential.subs(delta, 0) < potential.subs(delta, s.pi / 3)


@pytest.mark.parametrize("epsilon", COEFFICIENTS)
def test_full_rows_reproduce_family_jets_for_the_same_nodal_preparation(
    source, symbolic_law, epsilon
):
    law, report = symbolic_law, _report(source, epsilon)
    s = law["s"]
    parameter = s.Rational(epsilon.numerator, epsilon.denominator)
    tangent = law["tangent"].subs(law["epsilon"], parameter)
    metric = law["metric"].subs(law["epsilon"], parameter)
    assert s.Matrix(report.full_tangent_generator) == tangent
    assert s.Matrix(report.full_energy_metric) == metric
    assert s.Matrix(report.form_laplacian) == law["laplacian"]
    assert s.Matrix(report.phase_hessian) == law["hessian"].subs(
        law["epsilon"], parameter
    )
    preparation = s.zeros(24, 2)
    observation = s.zeros(2, 24)
    for i in range(12):
        # Piecewise constant opposite ring offsets create only the bridge gap.
        # Its stiffness is one for every epsilon, so this physical preparation
        # and its energy normalization are unchanged across the entire family.
        value = s.Rational(-1 if i < 6 else 1, 2)
        preparation[i, 0] = preparation[12 + i, 1] = value
    observation[0, 0], observation[0, 6] = -1, 1
    observation[1, 12], observation[1, 18] = -1, 1
    assert s.Matrix(report.nodal_bridge_preparation) == preparation
    assert s.Matrix(report.bridge_observation_rows) == observation
    assert observation * preparation == s.eye(2)
    assert preparation.T * metric * preparation == s.eye(2)
    first = observation * tangent * preparation
    second = observation * tangent**2 * preparation
    assert first == s.Matrix([[0, -s.Rational(2, 3)], [s.Rational(2, 3), 0]])
    assert second == s.diag(-s.Rational(2, 3) - parameter / 2, -s.Rational(8, 9))
    assert s.Matrix(report.first_jet) == first
    assert s.Matrix(report.second_jet) == second
    assert report.channel_difference == Q(2, 9) - epsilon / 2
    assert report.target_storage == 6 + 5 * epsilon / 2
    assert report.source.phase == (0,) * 12  # Capture is not relabeled as target.
    assert report.local_phase_radius_turns == Q(1, 24)
    curvature_floor = (s.sqrt(6) - s.sqrt(2)) / 4
    assert report.local_curvature_lower_bounds.contains(
        Q(str(s.N(curvature_floor, 80)))
    )
    assert report.local_excess_storage_threshold_bounds.contains(
        Q(str(s.N(curvature_floor * s.pi**2 / 288, 80)))
    )


def test_tangent_covariance_at_crossing_does_not_make_the_nonlinear_law_symmetric(
    source, symbolic_law
):
    law, report = symbolic_law, _report(source, Q(4, 9))
    s = law["s"]
    identity, zero = s.eye(12), s.zeros(12)
    rotation = zero.row_join(-identity).col_join(identity.row_join(zero))
    tangent = s.Matrix(report.full_tangent_generator)
    assert report.phase_hessian == report.form_laplacian
    assert tangent * rotation == rotation * tangent
    assert report.channel_difference == 0
    # The adjacent phase perturbation has a nonzero second variation in the
    # full form row, while every phase row stays linear in form. Equality of
    # the tangent Hessians does not extend to a nonlinear channel rotation.
    second_variation = s.diff(law["field"][0], law["theta"][1], 2)
    second_variation = s.simplify(
        second_variation.subs(law["critical"]).subs(law["epsilon"], s.Rational(4, 9))
    )
    assert second_variation == -2 * s.sqrt(3) / 9
    assert s.diff(law["field"][12], law["x"][1], 2) == 0


def test_opposite_winding_retains_full_criticality_and_tangent(source, symbolic_law):
    law = symbolic_law
    s = law["s"]
    critical = dict(law["critical"])
    for i in range(6, 12):
        critical[law["theta"][i]] = -critical[law["theta"][i]]
    assert law["field"].subs(critical).applyfunc(s.simplify) == s.zeros(24, 1)
    tangent = law["field"].jacobian((*law["x"], *law["theta"]))
    tangent = tangent.subs(critical).subs(law["epsilon"], 1).applyfunc(s.simplify)
    report = assess_bridge_storage_family(
        source,
        left_cycle=tuple(range(6)),
        right_cycle=tuple(range(6, 12)),
        target_phase_turns=TARGET[:6] + tuple(-turn for turn in TARGET[6:]),
        epsilon=1,
    )
    baseline = _report(source, Q(1))
    assert s.Matrix(report.full_tangent_generator) == tangent
    assert report.full_tangent_generator == baseline.full_tangent_generator
    assert report.target_storage == baseline.target_storage
    assert report.channel_difference == baseline.channel_difference


def test_cycle_orientation_and_zero_parameter_retain_the_existing_observation(source):
    for epsilon in COEFFICIENTS:
        normal = _report(source, epsilon)
        reversed_cycles = _report(source, epsilon, reverse_cycles=True)
        for name in (
            "phase_hessian",
            "full_tangent_generator",
            "nodal_bridge_preparation",
            "first_jet",
            "second_jet",
        ):
            assert getattr(normal, name) == getattr(reversed_cycles, name)
    base = _report(source, Q(0))
    original = assess_sine_bridge_channels(
        source, target_phase_turns=TARGET, bridge=(0, 6)
    )
    for i in range(2):
        for j in range(2):
            assert original.first_jet_bounds[i][j].contains(base.first_jet[i][j])
        assert original.second_jet_diagonal_bounds[i].contains(base.second_jet[i][i])
