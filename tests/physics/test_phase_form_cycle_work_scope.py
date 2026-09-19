"""Exact oriented source/form work, distinct from choosing a phase clock.

These controls reuse the supplied periodic prism family with 0<A<pi/4,
unit capacity, and nondimensional structural time. They evaluate symbolic
curves and exact fine generators, not trajectories or binary64 derivatives.
Reparameterizing a curve does not authorize changing its physical speed
while retaining the same nodal capacity and pressure law.
"""

from types import SimpleNamespace

import pytest

from tests.physics._internal_mode_fixture import P, _exact_generator, _graph
from tnfr.physics.support_transport import observe_support_transport


@pytest.fixture
def loop():
    s = pytest.importorskip("sympy")
    e, weight, amplitude, omega = s.symbols("e weight amplitude omega", positive=True)
    angle, mean = s.symbols("angle mean", real=True)
    source = observe_support_transport(_graph())
    strengths = tuple(
        sum(w for i, _, w in source.conductance if i == row)
        for row in range(len(source.nodes))
    )
    assert strengths == (3,) * 6 and source.capacity == (1,) * 6
    metric = s.diag(*strengths)
    generator = s.Matrix(_exact_generator(source))
    stiffness = -metric * generator
    q = s.Matrix(P * 2)
    a = amplitude * s.cos(angle)
    u = (
        weight
        * amplitude
        * (e * s.cos(angle) + omega * s.sin(angle))
        / (s.pi * (e**2 + omega**2))
    )
    x = mean * s.ones(6, 1) + u * q
    g = a * q / s.pi
    forcing = weight * g
    model_rate = e * generator * x + forcing
    assert (model_rate - omega * x.diff(angle)).applyfunc(s.simplify) == s.zeros(6, 1)
    return SimpleNamespace(
        s=s,
        e=e,
        weight=weight,
        amplitude=amplitude,
        omega=omega,
        angle=angle,
        q=q,
        u=u,
        x=x,
        g=g,
        forcing=forcing,
        metric=metric,
        stiffness=stiffness,
        model_rate=model_rate,
    )


def test_oriented_phase_form_work_equals_speed_action_and_reciprocal_exchange(loop):
    c, s = loop, loop.s
    tangent = c.x.diff(c.angle)
    work_density = (c.forcing.T * c.metric * tangent)[0]
    reciprocal_density = -(c.weight * c.x.T * c.metric * c.g.diff(c.angle))[0]
    action_density = (c.model_rate.T * c.metric * c.model_rate)[0] / c.omega
    dirichlet = (c.x.T * c.stiffness * c.x)[0] / 2
    assert (
        s.simplify(work_density - action_density - c.e * s.diff(dirichlet, c.angle))
        == 0
    )
    expected = (
        12 * c.weight**2 * c.amplitude**2 * c.omega / (s.pi * (c.e**2 + c.omega**2))
    )
    for density in (work_density, reciprocal_density, action_density):
        assert s.simplify(s.integrate(density, (c.angle, 0, 2 * s.pi))) == expected
    assert expected.is_positive
    assert (
        s.simplify(dirichlet.subs(c.angle, 2 * s.pi) - dirichlet.subs(c.angle, 0)) == 0
    )
    # This integrates DF against dx, not the earlier amplitude-maintenance
    # quantity 2<y,F>dt. Their different normalizations have different roles.


def test_reversing_the_same_loop_fails_positive_clock_and_held_source_cannot_pay(loop):
    c, s = loop, loop.s
    reverse_x = c.x.subs(c.angle, -c.angle)
    reverse_force = c.forcing.subs(c.angle, -c.angle)
    reverse_density = (reverse_force.T * c.metric * reverse_x.diff(c.angle))[0]
    reverse_work = s.simplify(s.integrate(reverse_density, (c.angle, 0, 2 * s.pi)))
    assert reverse_work.is_negative
    # At this regular nonzero tangent the same nodal vector points against
    # the reversed traversal. No positive change of clock fixes orientation.
    tangent_at_zero = reverse_x.diff(c.angle).subs(c.angle, 0)
    rate_at_zero = c.model_rate.subs(c.angle, 0)
    assert tangent_at_zero != s.zeros(6, 1)
    assert (rate_at_zero + c.omega * tangent_at_zero).applyfunc(s.simplify) == s.zeros(
        6, 1
    )
    speed = s.Symbol("speed", real=True)
    assert s.solve(speed * tangent_at_zero[0] - rate_at_zero[0], speed) == [-c.omega]

    held = s.Symbol("held", real=True) * c.q
    held_density = (held.T * c.metric * c.x.diff(c.angle))[0]
    assert s.simplify(s.integrate(held_density, (c.angle, 0, 2 * s.pi))) == 0
    # The moving closed EPI curve has positive H-speed action. A constant
    # source, including one preserved by common phase rotation, supplies
    # zero oriented work and cannot generate that curve in this fixed model.


def test_positive_phase_parameter_change_preserves_linework_but_not_arbitrary_speed(
    loop,
):
    c, s = loop, loop.s
    sigma = s.Symbol("sigma", real=True)
    progress = sigma + s.sin(sigma) / 2
    progress_rate = s.diff(progress, sigma)
    assert s.calculus.util.function_range(
        progress_rate, sigma, s.Interval(0, 2 * s.pi)
    ) == s.Interval(s.Rational(1, 2), s.Rational(3, 2))
    assert progress.subs(sigma, 0) == 0 and progress.subs(sigma, 2 * s.pi) == 2 * s.pi
    reparam_x = c.x.subs(c.angle, progress)
    reparam_force = c.forcing.subs(c.angle, progress)
    reparam_rate = c.model_rate.subs(c.angle, progress)
    tangent = reparam_x.diff(sigma)
    clock_rate = c.omega / progress_rate
    assert (clock_rate * tangent - reparam_rate).applyfunc(s.simplify) == s.zeros(6, 1)
    # dt/dsigma=progress_rate/omega lies in [1/(2*omega),3/(2*omega)]
    # for this coordinate change. This follows from the original supplied
    # nodal solution; it is not a new law selecting a primitive phase clock.
    density = (c.forcing.T * c.metric * c.x.diff(c.angle))[0]
    pulled_density = (reparam_force.T * c.metric * tangent)[0]
    assert (
        s.simplify(pulled_density - density.subs(c.angle, progress) * progress_rate)
        == 0
    )
    primitive = s.integrate(density, c.angle)
    composed_primitive = primitive.subs(c.angle, progress)
    assert s.simplify(s.diff(composed_primitive, sigma) - pulled_density) == 0
    original_work = primitive.subs(c.angle, 2 * s.pi) - primitive.subs(c.angle, 0)
    reparam_work = composed_primitive.subs(sigma, 2 * s.pi) - composed_primitive.subs(
        sigma, 0
    )
    assert s.simplify(reparam_work - original_work) == 0
    # Holding sigma_dot=omega instead would change the fine velocity by 3/2
    # at sigma=0 while pressure/capacity remain unchanged: that is a new law.
    invalid_rate = (c.omega * tangent).subs(sigma, 0)
    correct_rate = reparam_rate.subs(sigma, 0)
    assert (invalid_rate - s.Rational(3, 2) * correct_rate).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)
    assert invalid_rate != correct_rate
