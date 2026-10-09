"""Independent complete-solution controls for direct source-box Taylor steps."""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics import _validated_taylor as owner
from tnfr.mathematics._rational_interval import I


def _smooth(_):
    return (Q(1),)


def _assert_contains_mp(interval, value):
    lower = mp.mpf(interval.lo.numerator) / interval.lo.denominator
    upper = mp.mpf(interval.hi.numerator) / interval.hi.denominator
    assert lower <= value <= upper


def test_uncertain_linear_initial_values_are_enclosed_without_comparison(monkeypatch):
    monkeypatch.setattr(
        owner,
        "comparison_flow_upper",
        lambda *args: pytest.fail("direct box used comparison propagation"),
    )
    step, failed, reason = owner.validated_box_taylor_step(
        (I(1, 2),), Q(1, 8), lambda state: (-2 * state[0],), _smooth, order=6
    )
    assert step is not None and failed is reason is None
    assert step.initial_box == (I(1, 2),)
    assert step.picard_interior_margin > 0
    assert len(step.series[0]) == 7
    with mp.workdps(90):
        for initial in (mp.mpf(1), mp.mpf(3) / 2, mp.mpf(2)):
            exact = initial * mp.exp(-mp.mpf(1) / 4)
            _assert_contains_mp(step.endpoint[0], exact)
            _assert_contains_mp(step.increment[0], exact - initial)


def test_increment_symbolically_cancels_uncertain_baseline():
    step, _, _ = owner.validated_box_taylor_step(
        (I(-100, 100),), Q(1, 8), lambda state: (0 * state[0] + 3,), _smooth, order=4
    )
    assert step is not None
    assert step.increment == (I(Q(3, 8)),)
    assert step.endpoint == (I(Q(-797, 8), Q(803, 8)),)
    assert (step.endpoint[0] - step.initial_box[0]).width == 400
    assert step.local_remainder_bounds == (I(0),)


def test_coupled_oscillator_matches_independent_trigonometric_solution():
    source = (I(1), I(Q(1, 3)))
    horizon = Q(1, 16)
    step, failed, reason = owner.validated_box_taylor_step(
        source,
        horizon,
        lambda state: (state[1], -state[0]),
        _smooth,
        order=6,
        time=Q(3),
    )
    assert step is not None and failed is reason is None
    assert step.time == 3 and step.duration == horizon
    with mp.workdps(90):
        h = mp.mpf(1) / 16
        expected = (mp.cos(h) + mp.sin(h) / 3, -mp.sin(h) + mp.cos(h) / 3)
        for i, value in enumerate(expected):
            _assert_contains_mp(step.endpoint[i], value)
            _assert_contains_mp(
                step.increment[i], value - (mp.mpf(1) if i == 0 else mp.mpf(1) / 3)
            )
    assert all(value.width < Q(1, 10**8) for value in step.endpoint)


def test_nonlinear_logistic_solution_including_whole_tube_remainder():
    step, failed, reason = owner.validated_box_taylor_step(
        (I(Q(1, 4), Q(1, 3)),),
        Q(1, 8),
        lambda state: (state[0] * (1 - state[0]),),
        _smooth,
        order=5,
    )
    assert step is not None and failed is reason is None
    with mp.workdps(90):
        for initial in (mp.mpf(1) / 4, mp.mpf(1) / 3):
            exact = 1 / (1 + (1 / initial - 1) * mp.exp(-mp.mpf(1) / 8))
            _assert_contains_mp(step.endpoint[0], exact)
            _assert_contains_mp(step.increment[0], exact - initial)
    assert step.local_remainder_bounds[0].width > 0


@pytest.mark.parametrize("dimension", (36, 64))
def test_full_source_dimension_policy_does_not_change_comparison_cap(dimension):
    source = tuple(I(i - 17, i - 16) for i in range(dimension))
    step, _, _ = owner.validated_box_taylor_step(
        source,
        Q(1, 4),
        lambda state: tuple(0 * value + i for i, value in enumerate(state)),
        _smooth,
        order=2,
    )
    assert step is not None and len(step.endpoint) == dimension
    assert all(value == I(Q(i, 4)) for i, value in enumerate(step.increment))
    assert owner.MAX_COMPARISON_DIMENSION == 24
    with pytest.raises(ValueError, match="dimension"):
        owner.validated_taylor_step(
            source, Q(1, 4), lambda state: state, _smooth, order=2
        )


def test_domain_failure_returns_source_tube_without_field_evaluation():
    source = (I(-1, 1), I(2, 3))
    step, failed, reason = owner.validated_box_taylor_step(
        source,
        Q(1, 8),
        lambda _: pytest.fail("failed domain reached flow"),
        lambda _: (Q(0),),
        order=4,
        domain_failure="source_margin_unavailable",
    )
    assert step is None and failed == source
    assert reason == "source_margin_unavailable"


def test_no_adaptive_retry_when_picard_budget_fails():
    calls = []

    def domain(tube):
        calls.append(tube)
        return (Q(1),)

    step, failed, reason = owner.validated_box_taylor_step(
        (I(1), I(-1)),
        Q(1, 2),
        lambda state: (100 * state[0], 100 * state[1]),
        domain,
        order=4,
    )
    assert step is None and failed is not None
    assert reason == "strict_Picard_inclusion_not_resolved"
    assert len(calls) == 16


@pytest.mark.parametrize(
    "overrides",
    (
        {"box": ()},
        {"box": (I(0),) * 65},
        {"duration": 0},
        {"duration": True},
        {"duration": 0.1},
        {"time": -1},
        {"time": False},
        {"order": True},
        {"order": 0},
        {"order": 17},
        {"box": (True,)},
    ),
)
def test_invalid_work_and_scalar_domains_are_rejected(overrides):
    arguments = dict(
        box=(I(0),),
        duration=Q(1, 8),
        time=Q(0),
        flow=lambda _: pytest.fail("invalid domain reached flow"),
        domain=_smooth,
        order=4,
    )
    arguments.update(overrides)
    with pytest.raises((ValueError, TypeError)):
        owner.validated_box_taylor_step(**arguments)


def test_unavailable_remainder_keeps_picard_tube(monkeypatch):
    original = owner.flow_jets

    def failure(box, order, flow):
        if order == 5:
            raise ArithmeticError("controlled derivative enclosure failure")
        return original(box, order, flow)

    monkeypatch.setattr(owner, "flow_jets", failure)
    step, failed, reason = owner.validated_box_taylor_step(
        (I(1),), Q(1, 8), lambda state: (-state[0],), _smooth, order=4
    )
    assert step is None and failed[0].contains(1)
    assert "Taylor_source_box_unavailable" in reason
    assert "controlled derivative enclosure failure" in reason
