"""Full phase-metric differential and its work scope on the retained prism.

The source is g_i=(Arg(S_i)-theta_i)/pi on a regular fixed branch. Its
positive metric H makes H*g the negative derivative of the edge cosine
potential. Differentiating H is essential: a nonsymmetric source Jacobian
is not a new phase law or an independent source of sustaining work. These
exact symbolic controls neither prescribe phase velocity nor integrate a
trajectory. EPI source work in the joint state is a different one-form.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    _metric_differential,
    _nonrepeated_phase_geometry,
    _phase_support,
    _prepared_phase_geometry,
)


def _potential(s, graph, phases):
    nodes = tuple(graph)
    return sum(
        1 - s.cos(phases[nodes.index(i)] - phases[nodes.index(j)])
        for i, j in graph.edges()
    )


@pytest.mark.parametrize("preparation", ["repeated", "nonrepeated"])
def test_full_source_jacobian_skew_is_accounted_for_by_metric_variation(preparation):
    if preparation == "repeated":
        s, graph, rows, phases, *_ = _prepared_phase_geometry()
    else:
        s, graph, rows, phases = _nonrepeated_phase_geometry()
    metric, derivative, source, jacobian = _metric_differential(s, rows, phases)
    theta = s.symbols("theta0:6", real=True)
    potential = _potential(s, graph, theta)
    gradient = s.Matrix([s.diff(potential, angle) for angle in theta])
    point = dict(zip(theta, phases, strict=True))
    hessian = gradient.jacobian(theta).subs(point).applyfunc(s.simplify)
    gradient = gradient.subs(point).applyfunc(s.simplify)

    assert (metric * source + gradient).applyfunc(s.simplify) == s.zeros(6, 1)
    weighted_jacobian = metric * jacobian
    metric_change = s.diag(*source) * derivative
    assert (weighted_jacobian + metric_change + hessian).applyfunc(
        s.simplify
    ) == s.zeros(6)
    weighted_skew = (weighted_jacobian - weighted_jacobian.T).applyfunc(s.simplify)
    correction_skew = (metric_change - metric_change.T).applyfunc(s.simplify)
    assert weighted_skew != s.zeros(6)
    assert (weighted_skew + correction_skew).applyfunc(s.simplify) == s.zeros(6)

    # The full chain rule holds for every supplied velocity, including its
    # common component. It does not select that velocity or its orientation.
    omega = s.Matrix(s.symbols("omega0:6", real=True))
    response = weighted_jacobian * omega + s.diag(*source) * derivative * omega
    assert (response + hessian * omega).applyfunc(
        lambda value: s.simplify(s.expand(value))
    ) == s.zeros(6, 1)


def test_weighted_source_one_form_is_exact_for_arbitrary_regular_phase_paths():
    s = pytest.importorskip("sympy")
    graph, rows = _phase_support()
    theta = s.symbols("theta0:6", real=True)
    omega = s.Matrix(s.symbols("omega0:6", real=True))
    potential = _potential(s, graph, theta)
    gradient = s.Matrix([s.diff(potential, angle) for angle in theta])
    # On every regular chart, H_i*g_i=r_i*sin(delta_i). Express that
    # product directly through the actual neighbor phasors; this identity
    # requires no velocity model and no choice of angle at a zero resultant.
    weighted_source = s.Matrix(
        [
            sum(s.sin(theta[j]) for j in row) * s.cos(theta[i])
            - sum(s.cos(theta[j]) for j in row) * s.sin(theta[i])
            for i, row in enumerate(rows)
        ]
    )
    assert (weighted_source + gradient).applyfunc(s.trigsimp) == s.zeros(6, 1)
    assert s.trigsimp(weighted_source.dot(omega) + gradient.dot(omega)) == 0
    exterior_derivative = weighted_source.jacobian(theta)
    assert (exterior_derivative - exterior_derivative.T).applyfunc(
        s.trigsimp
    ) == s.zeros(6)
    # Consequently integral g^T H dtheta=-Delta V on regular paths and is
    # zero on a closed phase loop. This is not integral w*g^T D dx, and
    # does not set the EPI work of a coupled trajectory to zero.


def test_zero_displacement_extension_and_equilibrium_have_no_metric_skew_term():
    s = pytest.importorskip("sympy")
    delta = s.Symbol("delta", real=True)
    sinc = s.sin(delta) / delta
    assert s.limit(sinc, delta, 0) == 1
    assert s.limit(s.diff(sinc, delta), delta, 0) == 0
    graph, rows = _phase_support()
    metric, derivative, source, jacobian = _metric_differential(s, rows, (0,) * 6)
    theta = s.symbols("theta0:6", real=True)
    potential = _potential(s, graph, theta)
    hessian = s.hessian(potential, theta).subs(dict.fromkeys(theta, 0))

    assert metric == 3 * s.pi * s.eye(6)
    assert derivative == s.zeros(6)
    assert source == s.zeros(6, 1)
    assert metric * jacobian == -hessian
    assert metric * jacobian == jacobian.T * metric
    # More generally at any regular zero-source state, diag(g)*DH vanishes;
    # the identity gives H*Dg=-Hess(V). It is not a claim that every
    # nonzero-source Jacobian is asymmetric or that H defines a phase law.
