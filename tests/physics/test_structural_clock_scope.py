"""Clock observability controls on the current prism, without evolving it.

Phase paths and curve parameterizations below are explicitly supplied. A
positive scalar clock can rescale a tangent but cannot remove its transverse
error; synchronization alone does not select that scalar or supply a clock.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    P,
    _exact_generator,
    _graph,
    _phase_support,
)
from tnfr.dynamics.canonical import compute_canonical_nodal_derivative
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.gamma import kuramoto_R_psi
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.directed_diffusion import structural_time
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import observe_support_transport


def _fresh_snapshot(*, equilibrium=False):
    graph = _graph((0, 0, 0, 0)) if equilibrium else _graph()
    # Evaluate actual canonical pressure on a fixture; no EPI/phase step.
    default_compute_delta_nfr(graph)
    return graph, observe_support_transport(graph)


def test_phase_consensus_does_not_stop_nodal_change_or_capacity_exposure():
    graph, snapshot = _fresh_snapshot()
    assert kuramoto_R_psi(graph)[0] == 1.0
    assert snapshot.rate == (Q(-1, 4), Q(1, 4), Q(0)) * 2
    assert snapshot.stored_pressure == snapshot.epi_gradient
    assert (
        tuple(
            Q.from_float(
                compute_canonical_nodal_derivative(float(nu), float(p)).derivative
            )
            for nu, p in zip(snapshot.capacity, snapshot.stored_pressure, strict=True)
        )
        == snapshot.rate
    )
    assert tuple(structural_time(lambda _t: 1.0, (0.0, 0.25, 1.0))) == (0, 0.25, 1)

    s = pytest.importorskip("sympy")
    time = s.Symbol("time", nonnegative=True)
    q = s.Matrix(P * 2)
    field = s.ones(6, 1) / 2 + s.exp(-time) * q / 4
    generator = s.Matrix(_exact_generator(snapshot))
    assert (s.diff(field, time) - generator * field).applyfunc(s.simplify) == s.zeros(
        6, 1
    )
    assert tuple(field.subs(time, 0)) == snapshot.epi
    # A separately declared constant phase path is synchronized throughout.
    # This exact EPI solution does not derive any primitive phase evolution.
    assert s.diff(s.Integer(0), time) == 0
    assert s.simplify((field - s.ones(6, 1) / 2).dot(q)) == s.exp(-time)


def test_current_prism_kuramoto_order_repeats_without_orienting_the_phase_path():
    s = pytest.importorskip("sympy")
    _, neighbors = _phase_support()
    angle, clock = s.symbols("angle clock", real=True)
    amplitude = s.Symbol("amplitude", positive=True)
    phases = (-angle, angle, s.Integer(0)) * 2
    cosine_sum = s.simplify(sum(s.cos(value) for value in phases))
    sine_sum = s.simplify(sum(s.sin(value) for value in phases))
    assert cosine_sum == 2 + 4 * s.cos(angle)
    assert sine_sum == 0
    for row in neighbors:
        assert sorted(NODES[j][1] for j in row) == [0, 1, 2]
    cosine_range = s.calculus.util.function_range(
        s.cos(angle), angle, s.Interval(-s.pi / 4, s.pi / 4)
    )
    assert cosine_range.start == s.sqrt(2) / 2
    # On |angle|<pi/4 both global and neighbor resultants are positive real.
    order = cosine_sum / 6
    assert s.simplify(order.subs(angle, -angle) - order) == 0
    supplied_angle = amplitude * s.cos(clock)  # declared 0<amplitude<pi/4
    observed = order.subs(angle, supplied_angle)
    assert s.simplify(observed.subs(clock, clock + s.pi) - observed) == 0
    assert s.simplify(observed.subs(clock, -clock) - observed) == 0
    source = supplied_angle * s.Matrix(P * 2) / s.pi
    assert (source.subs(clock, clock + s.pi) + source).applyfunc(s.simplify) == s.zeros(
        6, 1
    )
    assert source.subs(clock, 0) != source.subs(clock, s.pi)
    # The positive shared metric cannot turn this lossy scalar into an
    # oriented state observation: opposite sources have identical R.


def test_positive_clock_rescales_conformant_tangent_but_not_transverse_error():
    _, snapshot = _fresh_snapshot()
    reference = derive_forced_support_balance(snapshot, epi_weight=1, forcing=(0,) * 6)
    metric = reference.metric_weights
    b = snapshot.rate
    conformant = tuple(2 * value for value in b)
    bb = dot(metric, tuple(value**2 for value in b))
    vv = dot(metric, tuple(value**2 for value in conformant))
    bv = dot(
        metric, tuple(left * right for left, right in zip(b, conformant, strict=True))
    )
    clock_scale = bv / vv
    assert clock_scale == Q(1, 2) > 0
    assert tuple(clock_scale * value for value in conformant) == b
    assert bb * vv - bv**2 == 0

    transverse = tuple(value + 1 for value in conformant)
    tt = dot(metric, tuple(value**2 for value in transverse))
    bt = dot(
        metric, tuple(left * right for left, right in zip(b, transverse, strict=True))
    )
    projected_scale = bt / tt
    assert projected_scale > 0
    assert bb * tt - bt**2 > 0
    assert any(
        actual != projected_scale * tangent
        for actual, tangent in zip(b, transverse, strict=True)
    )
    # A positive fitted scalar exists, but no scalar makes b=h*v exactly.


def test_zero_tangent_and_stationary_field_have_distinct_clock_availability():
    _, equilibrium = _fresh_snapshot(equilibrium=True)
    _, moving = _fresh_snapshot()
    assert equilibrium.rate == (0,) * 6
    assert moving.rate != equilibrium.rate
    s = pytest.importorskip("sympy")
    h = s.Symbol("h", positive=True)
    zero = s.zeros(6, 1)
    stationary = s.Matrix(equilibrium.rate)
    tangent = s.Matrix(moving.rate)

    assert stationary == h * zero  # every positive h works; none is identified
    assert s.solve(stationary - h * tangent, h) == []  # would require h=0
    assert tangent != h * zero  # nonzero field with zero tangent is impossible
