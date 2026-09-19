"""Exact source-response directionality on one reciprocal prism.

The nonlinear phase source can have a nonreversible derivative even though
the support is undirected. These are regular-chart derivative identities,
not a phase evolution law, a complex-frequency claim, or sustaining work.
The retained angles have irrational cosine data: symbolic expressions use
the shared owner's cosine-sum formula without passing rounded Gram data to
its exact-rational input boundary. No trajectory or parameter scan is run.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    _nonrepeated_phase_geometry,
    _prepared_phase_geometry,
    _symbolic_phase_response,
)


def _retained(s):
    _, graph, rows, phases = _nonrepeated_phase_geometry()
    response, squared = _symbolic_phase_response(s, phases, rows)
    return graph, rows, phases, response, squared


def test_repeated_preparation_has_positive_detailed_balance_on_reciprocal_support():
    s, _, rows, phases, _, cosine, _, prepared_response = _prepared_phase_geometry()
    response, squared = _symbolic_phase_response(s, phases, rows)
    resultant = 1 + s.sqrt(3)
    metric = s.diag(*cosine)

    assert all(s.simplify(value - resultant**2) == 0 for value in squared)
    assert (response - prepared_response).applyfunc(s.simplify) == s.zeros(6)
    assert (response * s.ones(6, 1)) == s.ones(6, 1)
    assert (metric * response - response.T * metric).applyfunc(s.simplify) == s.zeros(6)
    stationary = s.Matrix(cosine) / (2 * resultant)
    assert all(value.is_positive for value in stationary)
    assert s.simplify(sum(stationary)) == 1
    assert (response.T * stationary - stationary).applyfunc(s.simplify) == s.zeros(6, 1)


def test_retained_nonrepeated_source_jacobian_breaks_cycle_balance_exactly():
    s = pytest.importorskip("sympy")
    _, rows, phases, response, squared = _retained(s)
    a, b = (3 - s.sqrt(3)) / 4, (s.sqrt(3) - 1) / 2
    expected = s.Matrix(
        [
            [0, a, b, a, 0, 0],
            [a, 0, b, 0, a, 0],
            [s.Rational(2, 7), s.Rational(5, 14), 0, 0, 0, s.Rational(5, 14)],
            [s.Rational(2, 7), 0, 0, 0, s.Rational(5, 14), s.Rational(5, 14)],
            [0, s.Rational(5, 14), 0, s.Rational(2, 7), 0, s.Rational(5, 14)],
            [0, 0, b, a, a, 0],
        ]
    )
    assert response == expected
    assert squared == (4 + 2 * s.sqrt(3),) * 2 + (7,) * 3 + (4 + 2 * s.sqrt(3),)
    assert (response * s.ones(6, 1)).applyfunc(s.simplify) == s.ones(6, 1)
    assert all(response[i, j].is_positive for i, row in enumerate(rows) for j in row)
    assert max(phases) - min(phases) == s.pi / 3 < s.pi / 2

    # Independently differentiate Arg(S_i)=atan(Im(S_i)/Re(S_i)) in the
    # positive-real chart. This binds the cosine construction to the actual
    # canonical source g_i=(Arg(S_i)-theta_i)/pi, not an invented transition.
    theta = s.symbols("theta0:6", real=True)
    source = s.Matrix(
        [
            (
                s.atan(
                    sum(s.sin(theta[j]) for j in row)
                    / sum(s.cos(theta[j]) for j in row)
                )
                - theta[i]
            )
            / s.pi
            for i, row in enumerate(rows)
        ]
    )
    point = dict(zip(theta, phases, strict=True))
    assert all(sum(s.cos(phases[j]) for j in row).is_positive for row in rows)
    jacobian = source.jacobian(theta).subs(point).applyfunc(s.simplify)
    assert (s.pi * jacobian - response + s.eye(6)).applyfunc(s.simplify) == s.zeros(6)

    forward = response[0, 1] * response[1, 2] * response[2, 0]
    reverse = response[0, 2] * response[2, 1] * response[1, 0]
    assert s.simplify(forward / reverse) == s.Rational(4, 5)
    # Detailed-balance weights would cancel around this cycle. The ratio
    # rules out EVERY positive diagonal symmetrizer, not only uniform weights.
    assert s.simplify(
        jacobian[0, 1]
        * jacobian[1, 2]
        * jacobian[2, 0]
        / (jacobian[0, 2] * jacobian[2, 1] * jacobian[1, 0])
    ) == s.Rational(4, 5)


def test_reflection_reverses_source_not_response_and_mobility_derivative_matters():
    s = pytest.importorskip("sympy")
    graph, rows, phases, response, _ = _retained(s)
    reflected_response, _ = _symbolic_phase_response(
        s, tuple(-value for value in phases), rows
    )
    assert reflected_response == response
    a = s.pi / 6
    gamma = s.atan(1 / (3 * s.sqrt(3)))
    expected = s.Matrix([a, -a, gamma, gamma + a, gamma - a, -a]) / s.pi

    def source(angles):
        return s.Matrix(
            [
                (
                    s.atan(
                        sum(s.sin(angles[j]) for j in row)
                        / sum(s.cos(angles[j]) for j in row)
                    )
                    - angles[i]
                )
                / s.pi
                for i, row in enumerate(rows)
            ]
        ).applyfunc(s.simplify)

    assert (source(phases) - expected).applyfunc(s.simplify) == s.zeros(6, 1)
    assert (source(tuple(-value for value in phases)) + expected).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)

    # For V=sum_edges(1-cos gap), g=-M grad(V) on this regular chart.
    # At a noncritical point its derivative also includes -(dM) grad(V):
    # state-dependent mobility does not imply a symmetrizable source Jacobian.
    nodes = tuple(graph)
    edges = tuple((nodes.index(i), nodes.index(j)) for i, j in graph.edges())
    theta = s.symbols("theta0:6", real=True)
    potential = sum(1 - s.cos(theta[i] - theta[j]) for i, j in edges)
    gradient = s.Matrix([s.diff(potential, angle) for angle in theta])
    hessian = gradient.jacobian(theta)
    point = dict(zip(theta, phases, strict=True))
    resultant = 1 + s.sqrt(3)
    mobility0 = a / (s.pi * resultant * s.sin(a))
    assert s.simplify(gradient[0].subs(point) + resultant * s.sin(a)) == 0
    assert s.simplify(-mobility0 * gradient[0].subs(point) - expected[0]) == 0

    # One explicit cross entry exposes the nonzero derivative-of-mobility
    # correction. The phase direction j=1 changes both delta_0 and |S_0|.
    delta, radius = s.symbols("delta radius", positive=True)
    mobility = delta / (s.pi * radius * s.sin(delta))
    mobility_derivative = (
        s.diff(mobility, delta) * response[0, 1] - s.diff(mobility, radius) / 2
    ).subs({delta: a, radius: resultant})
    correction = s.simplify(-mobility_derivative * gradient[0].subs(point))
    observed = response[0, 1] / s.pi
    held_metric_term = -mobility0 * hessian[0, 1].subs(point)
    assert s.simplify(observed - held_metric_term - correction) == 0
    assert (
        s.simplify(correction - ((3 - s.sqrt(3)) / (4 * s.pi) - 1 / (6 * resultant)))
        == 0
    )
    # pi<4 supplies an exact strictly positive lower bound for this entry.
    assert s.simplify((3 - s.sqrt(3)) / 16 - 1 / (6 * resultant)).is_positive
