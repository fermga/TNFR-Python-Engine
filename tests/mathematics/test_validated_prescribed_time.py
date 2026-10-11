"""Independent analytic controls for a prescribed clock and state uncertainty.

No research preparation, reserved response or model-specific trajectory is
evaluated here. Time derivatives and same-time state derivatives differ.
"""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric import validated_metric_taylor_step
from tnfr.mathematics._validated_taylor import flow_jets, interval_jacobian


def _everywhere(time, state):
    return (Q(1),)


def test_time_jet_retains_nonzero_origin_and_all_polynomial_derivatives():
    rows = flow_jets(
        (I(7),),
        4,
        lambda time, state: (time**2 + 0 * state[0],),
        prescribed_time=Q(3, 2),
    )
    # Integral of (3/2+s)^2, independently expanded in powers of s.
    expected = (Q(7), Q(9, 4), Q(3, 2), Q(1, 3), Q(0))
    assert all(bound.contains(value) for bound, value in zip(rows[0], expected))
    assert rows[0][-1] == I(0)


def test_under_ordered_polynomial_retains_time_derivative_in_remainder():
    start, duration, center = Q(2), Q(1, 8), Q(3)
    step, failed, reason = validated_metric_taylor_step(
        (center,),
        Q(0),
        ((Q(1),),),
        duration,
        lambda time, state: (time**2 + 0 * state[0],),
        _everywhere,
        growth_rate=Q(0),
        order=2,
        time=start,
        time_dependent=True,
    )
    assert step is not None and failed is reason is None
    exact = center + ((start + duration) ** 3 - start**3) / 3
    assert step.endpoint[0].contains(exact)
    assert step.local_remainder_bounds[0].contains(duration**3 / 3)
    assert step.local_remainder_bounds[0].lo > 0
    assert step.prescribed_time_interval == I(start, start + duration)


def test_large_clock_derivative_does_not_create_initial_clock_uncertainty():
    start, duration, radius = Q(2), Q(1, 8), Q(1, 32)
    scale = Q(10**6)
    step, failed, reason = validated_metric_taylor_step(
        (Q(3),),
        radius,
        ((Q(1),),),
        duration,
        lambda time, state: (scale * time**2 + 0 * state[0],),
        _everywhere,
        growth_rate_bounds=(Q(0), Q(0)),
        order=4,
        time=start,
        time_dependent=True,
    )
    assert step is not None and failed is reason is None
    assert step.growth_certificate.certified
    assert step.growth_certificate.jacobian_bounds == ((I(0),),)
    assert step.growth_certificate.prescribed_time_interval == I(
        start, start + duration
    )
    assert step.propagated_initial_radius == radius
    assert radius <= step.endpoint_radius < radius + Q(1, 10**25)
    assert len(step.endpoint) == 1
    shift = scale * ((start + duration) ** 3 - start**3) / 3
    for initial in (3 - radius, Q(3), 3 + radius):
        assert step.endpoint[0].contains(initial + shift)


def test_time_varying_state_growth_covers_the_whole_common_time_interval():
    start, duration, radius = Q(3, 2), Q(1, 8), Q(1, 64)
    step, failed, reason = validated_metric_taylor_step(
        (Q(1),),
        radius,
        ((Q(1),),),
        duration,
        lambda time, state: (time * state[0],),
        _everywhere,
        growth_rate_bounds=(start + duration, start + duration),
        order=12,
        time=start,
        time_dependent=True,
    )
    assert step is not None and failed is reason is None
    assert step.growth_certificate.jacobian_bounds == ((I(start, start + duration),),)
    ctx = mp.mp.clone()
    ctx.dps = 85

    def real(value):
        return ctx.mpf(value.numerator) / value.denominator

    exponent = ((start + duration) ** 2 - start**2) / 2
    for initial in (1 - radius, Q(1), 1 + radius):
        exact = real(initial) * ctx.exp(real(exponent))
        assert real(step.endpoint[0].lo) <= exact <= real(step.endpoint[0].hi)
    assert step.growth_rate_upper_bound == start + duration


def test_non_autonomous_nonlinear_solution_retains_initial_ball():
    start, duration, radius = Q(1, 2), Q(1, 32), Q(1, 64)
    step, failed, reason = validated_metric_taylor_step(
        (Q(1),),
        radius,
        ((Q(1),),),
        duration,
        lambda time, state: (time * state[0] ** 2,),
        lambda time, state: (state[0].lo - Q(1, 2), 2 - state[0].hi),
        growth_rate=Q(3),
        order=8,
        time=start,
        time_dependent=True,
    )
    assert step is not None and failed is reason is None
    integral = ((start + duration) ** 2 - start**2) / 2
    for initial in (1 - radius, Q(1), 1 + radius):
        exact = initial / (1 - initial * integral)
        assert step.endpoint[0].contains(exact)
        assert abs(exact - step.endpoint_center[0]) <= step.endpoint_radius


def test_state_jacobian_holds_time_fixed_but_encloses_all_times():
    actual = interval_jacobian(
        (I(2, 3), I(4, 5)),
        lambda time, state: (time**2 * state[1], time * state[0] ** 2),
        prescribed_time=I(1, 2),
    )
    assert actual == ((I(0), I(1, 4)), (I(4, 12), I(0)))


def test_domain_reader_receives_the_entire_prescribed_time_interval():
    seen = []

    def domain(time, state):
        seen.append(time)
        return (2 - time.hi,)

    step, failed, reason = validated_metric_taylor_step(
        (Q(0),),
        Q(0),
        ((Q(1),),),
        Q(1, 4),
        lambda time, state: (time + 0 * state[0],),
        domain,
        growth_rate=Q(0),
        order=4,
        time=Q(15, 8),
        time_dependent=True,
    )
    assert step is None and failed is not None
    assert reason == "whole_tube_domain_not_admitted"
    assert seen == [I(Q(15, 8), Q(17, 8))]


def test_declared_switch_preserves_reached_state_and_radius():
    center, radius, duration = Q(5), Q(1, 16), Q(1, 8)
    initial_radius = radius
    for index, sign in enumerate((1, -1)):
        step, failed, reason = validated_metric_taylor_step(
            (center,),
            radius,
            ((Q(1),),),
            duration,
            lambda time, state: (sign * time + 0 * state[0],),
            _everywhere,
            growth_rate=Q(0),
            order=3,
            time=index * duration,
            time_dependent=True,
        )
        assert step is not None and failed is reason is None
        center, radius = step.endpoint_center[0], step.endpoint_radius
    assert center == 5 - Q(1, 64)
    assert radius == initial_radius
    assert step.endpoint[0] == I(center - radius, center + radius)


@pytest.mark.parametrize("flag", (None, 0, 1, "true"))
def test_prescribed_time_mode_requires_an_explicit_boolean(flag):
    with pytest.raises(TypeError, match="time_dependent"):
        validated_metric_taylor_step(
            (Q(0),),
            Q(0),
            ((Q(1),),),
            Q(1, 8),
            lambda state: state,
            lambda state: (Q(1),),
            growth_rate=Q(1),
            order=4,
            time_dependent=flag,
        )
