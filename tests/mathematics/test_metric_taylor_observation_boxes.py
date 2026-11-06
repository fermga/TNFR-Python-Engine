"""Whole-time observation controls on manufactured analytic systems only."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric import (
    metric_taylor_observation_boxes,
    validated_metric_taylor_step,
)


def _everywhere(*_):
    return (Q(1),)


def _oscillator(values):
    return (-values[1], values[0])


@pytest.fixture(scope="module")
def oscillator_step():
    # x(t)=cos(t)+sin(t)/8 has an interior maximum near t=1/8.
    step, failed, reason = validated_metric_taylor_step(
        (Q(1), Q(-1, 8)),
        Q(0),
        ((Q(1), Q(0)), (Q(0), Q(1))),
        Q(1, 4),
        _oscillator,
        _everywhere,
        growth_rate=Q(0),
        order=12,
        time=Q(3, 2),
    )
    assert step is not None and failed is reason is None
    return step


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_complete_oscillator_bins_enclose_interior_excursion(oscillator_step, mp):
    bins = metric_taylor_observation_boxes(oscillator_step, _oscillator, bin_count=4)
    assert tuple((lo, hi) for lo, hi, _ in bins) == tuple(
        (Q(i, 16), Q(i + 1, 16)) for i in range(4)
    )
    for lower, upper, box in bins:
        for elapsed in (lower, (lower + upper) / 2, upper):
            t = _mp(mp, elapsed)
            exact = (mp.cos(t) + mp.sin(t) / 8, mp.sin(t) - mp.cos(t) / 8)
            for bound, value in zip(box, exact, strict=True):
                _contains(mp, bound, value)
    threshold = Q(257, 256)
    assert oscillator_step.initial_center[0] < threshold
    assert oscillator_step.endpoint[0].hi < threshold
    middle = mp.cos(mp.mpf(1) / 8) + mp.sin(mp.mpf(1) / 8) / 8
    assert middle > _mp(mp, threshold)
    assert bins[1][2][0].hi > threshold
    assert bins[2][2][0].hi > threshold


def test_negative_growth_retains_incoming_ball_at_time_zero(mp):
    def flow(values):
        return (-values[0],)

    step, _, reason = validated_metric_taylor_step(
        (Q(0),),
        Q(1, 8),
        ((Q(4),),),
        Q(1, 4),
        flow,
        _everywhere,
        growth_rate=Q(-1),
        order=6,
    )
    assert reason is None and step.growth_factor_upper_bound < 1
    bins = metric_taylor_observation_boxes(step, flow, bin_count=4)
    assert bins[0][2][0].contains(Q(1, 16))
    assert bins[0][2][0].contains(Q(-1, 16))
    for lower, upper, box in bins:
        for elapsed in (lower, (lower + upper) / 2, upper):
            for sign in (-1, 1):
                exact = sign * mp.exp(-_mp(mp, elapsed)) / 16
                _contains(mp, box[0], exact)


def test_growing_initial_ball_and_positive_remainder_are_retained(mp):
    def flow(values):
        return (values[0] ** 2,)

    step, _, reason = validated_metric_taylor_step(
        (Q(1),),
        Q(1, 64),
        ((Q(9),),),
        Q(1, 16),
        flow,
        lambda box: (box[0].lo - Q(1, 2), 2 - box[0].hi),
        growth_rate=Q(4),
        order=2,
    )
    assert reason is None and step.local_remainder_bounds[0].lo > 0
    for lower, upper, box in metric_taylor_observation_boxes(step, flow, bin_count=3):
        for elapsed in (lower, (lower + upper) / 2, upper):
            for initial in (Q(1) - Q(1, 192), Q(1), Q(1) + Q(1, 192)):
                exact = initial / (1 - elapsed * initial)
                _contains(mp, box[0], _mp(mp, exact))


def test_prescribed_clock_uses_absolute_step_time_and_same_initial_state():
    def flow(time, values):
        return (0 * values[0] + 2 * time,)

    start, duration = Q(3), Q(1, 4)
    step, _, reason = validated_metric_taylor_step(
        (Q(5),),
        Q(1, 32),
        ((Q(4),),),
        duration,
        flow,
        _everywhere,
        growth_rate=Q(0),
        order=3,
        time=start,
        time_dependent=True,
    )
    assert reason is None
    for lower, upper, box in metric_taylor_observation_boxes(step, flow, bin_count=4):
        for elapsed in (lower, (lower + upper) / 2, upper):
            for offset in (-Q(1, 64), Q(1, 64)):
                assert box[0].contains(5 + offset + (start + elapsed) ** 2 - start**2)


def test_small_duration_does_not_divide_by_rounded_remainder_scale():
    duration = Q(1, 2**40)

    def flow(values):
        return (values[0] * 0 + 1,)

    step, _, reason = validated_metric_taylor_step(
        (Q(0),),
        Q(0),
        ((Q(1),),),
        duration,
        flow,
        _everywhere,
        growth_rate=Q(0),
        order=4,
    )
    assert reason is None and I(duration**5).contains(0)
    bins = metric_taylor_observation_boxes(step, flow, bin_count=2)
    for lower, upper, box in bins:
        assert box[0].contains(lower) and box[0].contains(upper)


def test_cached_projection_growth_and_endpoint_are_not_consumed(oscillator_step):
    step, failed, reason = validated_metric_taylor_step(
        oscillator_step.initial_center,
        Q(1, 64),
        oscillator_step.metric,
        oscillator_step.duration,
        _oscillator,
        _everywhere,
        growth_rate=Q(0),
        order=oscillator_step.order,
    )
    assert step is not None and failed is reason is None
    expected = metric_taylor_observation_boxes(step, _oscillator, bin_count=3)
    corrupted = replace(
        step,
        coordinate_projection_factors=(Q(0), Q(0)),
        growth_factor_upper_bound=Q(0),
        endpoint_center=(Q(100), Q(100)),
        endpoint=(I(100), I(100)),
        endpoint_radius=Q(0),
    )
    assert (
        metric_taylor_observation_boxes(corrupted, _oscillator, bin_count=3) == expected
    )


@pytest.mark.parametrize("count", (True, False, 0, -1, Q(2), 2.0, 4097))
def test_invalid_work_policy_rejected_before_field_evaluation(oscillator_step, count):
    def forbidden(_):
        pytest.fail("invalid observation policy must not evaluate the field")

    with pytest.raises(ValueError):
        metric_taylor_observation_boxes(oscillator_step, forbidden, bin_count=count)


@pytest.mark.parametrize(
    "field,value",
    (
        ("duration", Q(0)),
        ("duration", True),
        ("time", Q(-1)),
        ("initial_radius", Q(-1)),
        ("growth_rate_upper_bound", True),
        ("order", True),
        ("order", 17),
        ("metric", ((Q(1), Q(0)), (Q(0), Q(0)))),
        ("initial_center", (True, Q(0))),
        ("local_remainder_bounds", (I(0),)),
        ("prescribed_time_interval", I(0, 1)),
    ),
)
def test_invalid_consumed_step_premises_are_rejected(oscillator_step, field, value):
    def forbidden(_):
        pytest.fail("invalid step evidence must not evaluate the field")

    with pytest.raises((TypeError, ValueError)):
        metric_taylor_observation_boxes(
            replace(oscillator_step, **{field: value}), forbidden, bin_count=2
        )


def test_untyped_step_and_noncallable_field_are_rejected(oscillator_step):
    with pytest.raises(TypeError):
        metric_taylor_observation_boxes(object(), _oscillator, bin_count=1)
    with pytest.raises(TypeError):
        metric_taylor_observation_boxes(oscillator_step, None, bin_count=1)
