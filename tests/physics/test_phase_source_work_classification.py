"""Actual form work versus phase-source sensitivity on the retained prism.

The phase velocity remains a supplied tangent, never a new evolution law.
Reflection and tangent-sign controls keep the original EPI amplitude and
support. They do not tune a preparation for growth or run a trajectory.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    _exact_generator,
    _nonrepeated_phase_geometry,
    _symbolic_phase_response,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.support_transport import observe_support_transport


@pytest.fixture(scope="module")
def prepared():
    s, graph, rows, phases = _nonrepeated_phase_geometry()
    response, _ = _symbolic_phase_response(s, phases, rows)
    laplacian = -s.Matrix(_exact_generator(observe_support_transport(graph)))
    angles = s.Matrix(
        [
            s.atan(
                sum(s.sin(phases[j]) for j in row) / sum(s.cos(phases[j]) for j in row)
            )
            for row in rows
        ]
    )
    source = ((angles - s.Matrix(phases)) / s.pi).applyfunc(s.simplify)
    form = s.Matrix([s.Rational(graph.nodes[node]["EPI"]) for node in NODES])
    center = s.eye(6) - s.ones(6) / 6
    return s, graph, rows, phases, response, laplacian, source, form, center


def test_same_response_matrix_has_opposite_actual_form_work_and_mean_rate(prepared):
    s, _, rows, phases, response, lap, source, form, center = prepared
    reflected, _ = _symbolic_phase_response(s, tuple(-value for value in phases), rows)
    assert reflected == response
    y = center * form
    assert y == s.Matrix([1, -1, 0, 1, -1, 0]) / 4
    assert lap * y == y and y.dot(y) == s.Rational(1, 4)
    assert s.simplify(y.dot(source)) == s.Rational(1, 6)
    e, w = s.Rational(1, 2), s.Rational(1, 4)
    gamma = s.atan(1 / (3 * s.sqrt(3)))
    for sign, expected_rate in ((1, -s.Rational(1, 6)), (-1, -s.Rational(1, 3))):
        pressure = -e * lap * form + sign * w * source
        phase_work = 2 * sign * w * y.dot(source)
        assert s.simplify(phase_work) == sign * s.Rational(1, 12)
        assert s.simplify(2 * y.dot(pressure)) == expected_rate
        assert (
            s.simplify(sum(pressure) / 6 - sign * (3 * gamma - s.pi / 6) / (24 * s.pi))
            == 0
        )
    # In both unchanged-amplitude controls the total form rate is negative.
    # R's cycle imbalance cannot identify either the sign of phase work or
    # mean drift, and favorable source work alone does not mean maintenance.


def test_phase_velocity_selects_work_change_and_mean_acceleration_not_response_alone(
    prepared,
):
    s, _, _, _, response, lap, source, form, center = prepared
    e, w = s.Rational(1, 2), s.Rational(1, 4)
    jacobian = (response - s.eye(6)) / s.pi
    y = center * form
    pressure = -e * lap * form + w * source
    omega = s.Matrix(s.symbols("omega0:6", real=True))
    pressure_rate = -e * lap * pressure + w * jacobian * omega
    # Differentiate S'=2*y^T*p along the actual nodal x'=p, retaining mean.
    norm_acceleration = 2 * (center * pressure).dot(center * pressure) + 2 * y.dot(
        pressure_rate
    )
    fixed_phase_acceleration = norm_acceleration.subs(dict.fromkeys(omega, 0))
    phase_contribution = s.expand(norm_acceleration - fixed_phase_acceleration)
    assert s.simplify(phase_contribution - 2 * w * y.dot(jacobian * omega)) == 0
    assert s.simplify(sum(pressure_rate) / 6 - w * sum(jacobian * omega) / 6) == 0

    tangent = s.Matrix([-1, 1, 0, 0, 0, 0])
    expected = (40 - 7 * s.sqrt(3)) / (112 * s.pi)
    assert expected.is_positive
    for sign in (-1, 0, 1):
        values = dict(zip(omega, sign * tangent, strict=True))
        assert s.simplify(phase_contribution.subs(values) - sign * expected) == 0
        assert (
            s.simplify(sum(pressure_rate.subs(values)) / 6 - sign / (168 * s.pi)) == 0
        )
    assert (jacobian * s.ones(6, 1)).applyfunc(s.simplify) == s.zeros(6, 1)
    # These are all regular local tangent choices, not admitted trajectories.
    # Common rotation changes neither source, mean acceleration, nor work.


def test_detached_pressure_reader_confirms_reflected_work_without_state_mutation(
    prepared,
):
    s, original, _, phases, _, _, source, form, center = prepared
    y = tuple(Q(value) for value in center * form)
    for sign, expected in ((1, -Q(1, 6)), (-1, -Q(1, 3))):
        graph = deepcopy(original)
        graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=0.25, vf=0.25, topo=0.0)
        for node, phase in zip(NODES, phases, strict=True):
            graph.nodes[node]["theta"] = float(sign * phase)
        before = deepcopy(graph)
        captured = capture_non_epi_forcing(graph)
        assert tuple(map(float, captured.phase_gradient)) == pytest.approx(
            [float(sign * value) for value in source], abs=3e-16, rel=0
        )
        actual_rate = 2 * sum(
            (
                value * pressure
                for value, pressure in zip(
                    y, captured.full_kernel_pressure, strict=True
                )
            ),
            Q(0),
        )
        assert float(actual_rate) == pytest.approx(float(expected), abs=3e-16, rel=0)
        assert graph.graph == before.graph
        assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
        assert dict(graph.edges) == dict(before.edges)
    # Production binary64 phase evaluation is checked against, not identified
    # with, the exact-real formulas. No history or selection is manufactured.
