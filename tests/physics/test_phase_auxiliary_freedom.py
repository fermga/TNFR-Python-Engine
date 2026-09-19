"""Structural restrictions do not select the auxiliary joint potential.

V and V+Q are logical alternatives, not installed laws or fitted parameters.
Their alignment properties do not establish a stable full nodal pattern.
Exact phase derivatives remain separate from captured binary64 pressure.
"""

from fractions import Fraction as F
from math import pi

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    _exact_generator,
    _graph,
    _prepared_phase_geometry,
    _prepared_phase_metric,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing


def _costs(s, graph, phases):
    # One contribution per undirected edge of this simple support. Nothing
    # here selects a potential for runtime use or changes its phase channel.
    terms = tuple(1 - s.cos(phases[j] - phases[i]) for i, j in graph.edges)
    return sum(terms), sum(term**2 for term in terms)


def test_edge_local_alternatives_share_symmetries_and_the_consensus_hessian():
    s, graph, _, _, _, _, _, _ = _prepared_phase_geometry()
    theta = s.symbols("theta0:6", real=True)
    phases = dict(zip(NODES, theta, strict=True))
    v, q = _costs(s, graph, phases)
    assert q != 0
    rotation = s.Symbol("rotation", real=True)
    turns = s.symbols("turn0:6", integer=True)
    rotated = {node: phase + rotation for node, phase in phases.items()}
    periodic = {
        node: phase + 2 * s.pi * turn
        for node, phase, turn in zip(NODES, theta, turns, strict=True)
    }
    permutation = dict(zip(NODES, (4, 1, 5, 0, 3, 2), strict=True))
    relabeled = nx.relabel_nodes(graph, permutation)
    relabeled_phases = {permutation[node]: phase for node, phase in phases.items()}
    for other in (
        _costs(s, graph, rotated),
        _costs(s, graph, periodic),
        _costs(s, relabeled, relabeled_phases),
    ):
        assert all(s.simplify(a - b) == 0 for a, b in zip((v, q), other, strict=True))
    consensus = dict.fromkeys(theta, 0)
    laplacian = s.Matrix(
        6,
        6,
        lambda i, j: 3 if i == j else -int(graph.has_edge(NODES[i], NODES[j])),
    )
    for potential in (v, v + q):
        assert potential.subs(consensus) == 0
        assert s.Matrix(
            [s.diff(potential, angle).subs(consensus) for angle in theta]
        ) == (s.zeros(6, 1))
        assert s.hessian(potential, theta).subs(consensus) == laplacian
        for i in range(6):
            for j in range(i + 1, 6):
                if not graph.has_edge(NODES[i], NODES[j]):
                    assert s.diff(potential, theta[i], theta[j]) == 0
    # Matching the Hessian fixes the same quadratic normalization, not the
    # higher-order edge function. Both candidates use only phase differences.


def test_both_candidates_keep_nonnegativity_zeros_and_strict_u3_curvature():
    s = pytest.importorskip("sympy")
    delta = s.Symbol("delta", real=True)
    a = s.Symbol("a", real=True)
    edge_v = 1 - s.cos(delta)
    edge_q = edge_v**2
    # With a=1-cos(delta) in [0,2], a <= a+a^2 <= 3a and their zero sets agree.
    assert s.factor((a + a**2) - a) == a**2
    assert s.expand(3 * a - (a + a**2) - a * (2 - a)) == 0
    domain = s.Interval(0, 2)
    assert s.solveset(a, a, domain=domain) == s.solveset(a + a**2, a, domain=domain)
    assert s.solveset(a + a**2, a, domain=domain) == s.FiniteSet(0)
    c = s.Symbol("c", real=True)
    curvature = s.expand_trig(s.diff(edge_v + edge_q, delta, 2))
    curvature = s.expand(curvature.subs(s.sin(delta) ** 2, 1 - s.cos(delta) ** 2))
    curvature = s.expand(curvature.subs(s.cos(delta), c))
    assert s.expand(curvature - (c + 2 * (1 - c) * (1 + 2 * c))) == 0
    assert s.minimum(curvature, c, s.Interval(0, 1)) == 1
    assert s.diff(edge_v, delta, 2) == s.cos(delta)
    # Strict U3 gives c>0, so both graph Hessians have positive edge weights
    # and only the common-rotation kernel on connected support. These are
    # properties of the auxiliary costs, not stability of the joint system.


def test_same_prepared_pressure_has_different_conditional_phase_and_pressure_rates():
    s, _, phases, metric, phase_source, source_jacobian = _prepared_phase_metric()
    graph = _graph((F(1, 16), 0, F(1, 16), 0))
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": 0.5,
        "phase": 0.25,
        "vf": 0.25,
        "topo": 0.0,
    }
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = (0.0, pi / 3, pi / 6)[i]
    captured = capture_non_epi_forcing(graph)
    for node, pressure in zip(NODES, captured.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(pressure)
    captured = capture_non_epi_forcing(graph)
    assert captured.stored_pressure_residual == (0,) * 6
    assert captured.snapshot.rate == captured.full_kernel_pressure
    e, w = s.Rational(1, 2), s.Rational(1, 4)
    theta = s.symbols("theta0:6", real=True)
    v, q = _costs(s, graph, dict(zip(NODES, theta, strict=True)))
    at_phase = dict(zip(theta, phases, strict=True))
    direction = s.Matrix((1, -1, 0) * 2)
    resultant = 1 + s.sqrt(3)
    assert s.simplify(v.subs(at_phase)) == 5 - 2 * s.sqrt(3)
    assert s.simplify(q.subs(at_phase)) == s.Rational(15, 2) - 4 * s.sqrt(3)
    gradient_q = s.Matrix(
        [s.simplify(s.diff(q, angle).subs(at_phase)) for angle in theta]
    )
    assert gradient_q == -direction
    fine = s.Matrix(_exact_generator(captured.snapshot))
    degree = 3 * s.eye(6)
    stiffness = -degree * fine
    x = s.Matrix(s.symbols("x0:6", real=True))
    at_x = dict(zip(x, map(s.Rational, captured.snapshot.epi), strict=True))
    common_energy = (
        e * (x.T * stiffness * x)[0] / 2 - (x.T * degree * w * phase_source)[0]
    )
    mobility = metric.inv()
    phase_rates = []
    pressure_rates = []
    for potential in (v, v + q):
        # Psi is independent of x, so both choices give the same nodal row.
        energy = common_energy + potential.subs(at_phase)
        pressure = -s.Matrix([s.diff(energy, xi) for xi in x]) / 3
        pressure = s.simplify(pressure.subs(at_x))
        assert pressure == direction / 96
        gradient = s.Matrix(
            [s.simplify(s.diff(potential, angle).subs(at_phase)) for angle in theta]
        )
        phase_rate = mobility * (
            w * source_jacobian.T * degree * x.subs(at_x) - gradient
        )
        phase_rates.append(phase_rate)
        # Held W, capacity and channel mix: p'=e*B_W*p+w*Dg*theta'.
        pressure_rates.append(e * fine * pressure + w * source_jacobian * phase_rate)
    assert s.simplify(
        phase_rates[1] - phase_rates[0] - direction / (3 * resultant)
    ) == (s.zeros(6, 1))
    assert s.simplify(
        pressure_rates[1] - pressure_rates[0] + direction / (12 * s.pi * resultant)
    ) == s.zeros(6, 1)
    # Exact ideal phases and captured binary64 pressure are distinct inputs;
    # retain the current realization defect instead of calling the derivative
    # above a derivative of the runtime trigonometric kernel.
    residual = tuple(
        actual - F(int(sign), 96)
        for actual, sign in zip(captured.full_kernel_pressure, direction, strict=True)
    )
    assert any(residual) and max(map(abs, residual)) < F(1, 2**49)
    repeated = capture_non_epi_forcing(graph)
    assert repeated == captured
    # Neither logical choice is installed. Equal current rates and equal
    # local consensus Hessians do not select a future constitutive response.
