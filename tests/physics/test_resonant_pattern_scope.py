"""Separate a supplied periodic source from autonomous resonant dynamics.

The exact unit-prism model is evaluated symbolically, without evolving a
graph. Its phase clock is supplied explicitly. No binary64 trigonometric
identity, phase law, recurrent NFR, or finite-frequency resonance is inferred.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    P,
    _exact_generator,
    _graph,
    _phase_support,
    _prepared_phase_metric,
    _unit_regional_budget,
)
from tnfr.physics.support_transport import observe_support_transport


def test_prescribed_periodic_phase_source_matches_the_exact_fine_nodal_law():
    s = pytest.importorskip("sympy")
    angle = s.Symbol("angle", real=True)
    graph, neighbors = _phase_support()
    phases = (-angle, angle, s.Integer(0)) * 2
    resultant = 1 + 2 * s.cos(angle)
    for row in neighbors:
        assert sorted(NODES[j][1] for j in row) == [0, 1, 2]
        assert s.simplify(sum(s.cos(phases[j]) for j in row)) == resultant
        assert s.simplify(sum(s.sin(phases[j]) for j in row)) == 0
    cosine_range = s.calculus.util.function_range(
        s.cos(angle), angle, s.Interval(-s.pi / 4, s.pi / 4)
    )
    assert cosine_range == s.Interval(s.sqrt(2) / 2, 1)
    assert (1 + 2 * cosine_range.start).is_positive
    # |angle|<=A<pi/4 gives a positive real resultant and every edge gap
    # at most 2*A<pi/2. Arg(resultant)=0 on this regular strict-U3 chart.
    q = s.Matrix(P * 2)
    phase_source = s.Matrix([-phase / s.pi for phase in phases])
    assert phase_source == angle * q / s.pi

    snapshot = observe_support_transport(graph)
    generator = s.Matrix(_exact_generator(snapshot))
    assert generator * q == -q
    assert generator * s.ones(6, 1) == s.zeros(6, 1)
    e, weight, amplitude, omega = s.symbols("e weight amplitude omega", positive=True)
    time, mean = s.symbols("time mean", real=True)
    prescribed_angle = amplitude * s.cos(omega * time)
    response = (
        weight
        * amplitude
        * (e * s.cos(omega * time) + omega * s.sin(omega * time))
        / (s.pi * (e**2 + omega**2))
    )
    field = mean * s.ones(6, 1) + response * q
    forcing = weight * prescribed_angle * q / s.pi
    fine_rate = e * generator * field + forcing
    assert (s.diff(field, time) - fine_rate).applyfunc(s.simplify) == s.zeros(6, 1)
    assert s.simplify((3 * s.ones(1, 6) * forcing)[0]) == 0
    # Mean compatibility holds pointwise here. Periodic response relies on
    # the prescribed source clock, not on a law derived for primitive phase.
    # This exact regional snapshot independently anchors the factor S=4*u^2.
    rational_snapshot = observe_support_transport(_graph((Q(1, 16), 0, Q(1, 16), 0)))
    declared_forcing = tuple(Q(1, 24) * value for value in P * 2)
    assert _unit_regional_budget(rational_snapshot, declared_forcing) == (
        Q(1, 64),
        Q(1, 192),
        Q(1, 48),
    )


def test_periodic_source_work_pays_loss_and_has_no_nonzero_modal_resonance_peak():
    s = pytest.importorskip("sympy")
    e, weight, amplitude, omega = s.symbols("e weight amplitude omega", positive=True)
    clock = s.Symbol("clock", real=True)
    forcing = weight * amplitude * s.cos(clock) / s.pi
    response = (
        weight
        * amplitude
        * (e * s.cos(clock) + omega * s.sin(clock))
        / (s.pi * (e**2 + omega**2))
    )
    squared = 4 * response**2
    work = 8 * response * forcing
    loss = 2 * e * squared
    assert s.trigsimp(omega * s.diff(squared, clock) + loss - work) == 0
    cycle_work = s.simplify(s.integrate(work / omega, (clock, 0, 2 * s.pi)))
    cycle_loss = s.simplify(s.integrate(loss / omega, (clock, 0, 2 * s.pi)))
    assert (
        cycle_work
        == cycle_loss
        == (8 * e * weight**2 * amplitude**2 / (s.pi * omega * (e**2 + omega**2)))
    )
    assert cycle_work.is_positive
    response_amplitude_squared = (
        weight**2 * amplitude**2 / (s.pi**2 * (e**2 + omega**2))
    )
    assert s.diff(response_amplitude_squared, omega).is_negative
    assert s.limit(response_amplitude_squared, omega, s.oo) == 0
    # The phase amplitude and spatial driver stay fixed as omega varies.
    # This is a first-order modal response, not a claim about arbitrary
    # input/output mixtures or the separately supplied graph-wave model.


def test_common_phase_rotation_is_invisible_to_source_but_not_gradient_motion():
    s, _, _, metric, source, jacobian = _prepared_phase_metric()
    ones = s.ones(6, 1)
    assert (jacobian * ones).applyfunc(s.simplify) == s.zeros(6, 1)
    assert source != s.zeros(6, 1)
    x = s.Matrix(s.symbols("x0:6", real=True))
    weight = s.Symbol("weight", positive=True)
    # Psi=V_phi is only a declared control. More generally every rotation-
    # invariant Psi has ones.T*gradient(Psi)=0, with the same conclusion.
    gradient = -weight * jacobian.T * (3 * x) - metric * source
    assert s.simplify((ones.T * gradient)[0]) == 0
    velocity = -metric.inv() * gradient
    assert s.simplify((ones.T * metric * velocity)[0]) == 0
    omega = s.Symbol("omega", real=True)
    rotation_metric_total = s.simplify((ones.T * metric * (omega * ones))[0])
    assert s.diff(rotation_metric_total, omega).is_positive
    assert s.solve(rotation_metric_total, omega) == [0]

    # This actual-metric control excludes nonzero pure common rotation
    # under the stated invariant gradient completion. Source invariance
    # alone does not determine the otherwise unspecified primitive velocity.
