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


@pytest.mark.parametrize(
    "stepper", (owner.validated_taylor_step, owner.validated_box_taylor_step)
)
def test_exact_clock_components_are_normalized_before_taylor_powers(stepper):
    np = pytest.importorskip("numpy")
    duration = Q(1, 2**32)
    time = Q(2**62, 3)
    arguments = dict(
        box=(I(1),), flow=lambda state: (-state[0],), domain=_smooth, order=3
    )
    step, failed, reason = stepper(
        duration=Q(np.int64(1), np.int64(2**32)),
        time=Q(np.int64(2**62), np.int64(3)),
        **arguments,
    )
    assert step is not None and failed is reason is None
    assert (step, failed, reason) == stepper(duration=duration, time=time, **arguments)
    for value in (step.duration, step.time):
        assert type(value.numerator) is type(value.denominator) is int
    # Alternating exp(-h) bounds, independent of interval jets/Picard iteration.
    lower = 1 - duration + duration**2 / 2 - duration**3 / 6
    upper = lower + duration**4 / 24
    assert step.endpoint[0].lo <= lower <= upper <= step.endpoint[0].hi


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


def _partial_arguments(**changes):
    arguments = dict(
        initial_box=(I(-1, 2),),
        tube=(I(-2, 4),),
        series=((I(-1, 2), I(3), I(-2)),),
        local_remainder_bounds=(I(Q(5, 64)),),
        full_duration=Q(1, 4),
        elapsed=Q(1, 8),
        order=2,
    )
    arguments.update(changes)
    return arguments


@pytest.mark.parametrize("elapsed", [Q(0), Q(1, 16), Q(1, 12), Q(1, 8), Q(1, 4)])
@pytest.mark.parametrize("cubic", [-5, 5])
def test_partial_time_cubic_uses_uniform_remainder_and_retains_source_correlation(
    elapsed, cubic, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("partial-time arithmetic must not produce jets or a trajectory")

    for name in ("flow_jets", "picard_tube", "validated_box_taylor_step"):
        monkeypatch.setattr(owner, name, forbidden)
    # y(t)=y0+3t-2t^2+cubic*t^3: the normalized third derivative
    # is the constant cubic on the entire tube, not just at the endpoint.
    increment, endpoint = owner.reconstruct_box_taylor_partial(
        **_partial_arguments(elapsed=elapsed, local_remainder_bounds=(I(Q(cubic, 64)),))
    )
    change = 3 * elapsed - 2 * elapsed**2 + cubic * elapsed**3
    assert increment[0].contains(change)
    for source in (-1, Q(1, 2), 2):
        assert endpoint[0].contains(source + change)
    # Uncertainty of the initial state is not added to the increment.
    assert increment[0].width < Q(1, 10**30)
    assert (endpoint[0] - I(-1, 2)).width >= 6


def test_partial_time_zero_and_full_endpoints_preserve_original_result():
    step, failed, reason = owner.validated_box_taylor_step(
        (I(1, 2),), Q(1, 8), lambda state: (-state[0],), _smooth, order=4
    )
    assert step is not None and failed is reason is None
    arguments = dict(
        initial_box=step.initial_box,
        tube=step.tube,
        series=step.series,
        local_remainder_bounds=step.local_remainder_bounds,
        full_duration=step.duration,
        order=step.order,
    )
    assert owner.reconstruct_box_taylor_partial(**arguments, elapsed=0) == (
        (I(0),),
        step.initial_box,
    )
    assert owner.reconstruct_box_taylor_partial(**arguments, elapsed=step.duration) == (
        step.increment,
        step.endpoint,
    )


def test_partial_nonlinear_solution_for_all_supplied_initial_values(monkeypatch):
    step, failed, reason = owner.validated_box_taylor_step(
        (I(Q(1, 2), 1),),
        Q(1, 16),
        lambda state: (state[0] ** 2,),
        _smooth,
        order=5,
    )
    assert step is not None and failed is reason is None
    assert step.local_remainder_bounds[0].lo > 0

    def forbidden(*args, **kwargs):
        pytest.fail("a retained step must be reconstructed without new derivatives")

    monkeypatch.setattr(owner, "flow_jets", forbidden)
    monkeypatch.setattr(owner, "picard_tube", forbidden)
    for elapsed in (Q(1, 64), Q(1, 48), Q(1, 32), Q(3, 64)):
        increment, endpoint = owner.reconstruct_box_taylor_partial(
            step.initial_box,
            step.tube,
            step.series,
            step.local_remainder_bounds,
            step.duration,
            elapsed,
            order=step.order,
        )
        # Exact rational solution of y'=y^2, independent of Taylor arithmetic.
        for initial in (Q(1, 2), Q(3, 4), Q(1)):
            exact = initial / (1 - initial * elapsed)
            assert endpoint[0].contains(exact)
            assert increment[0].contains(exact - initial)


def test_endpoint_only_error_cancellation_does_not_supply_uniform_remainder():
    # y(t)=t^2*(1-t), H=1, order1: the endpoint error is zero, but
    # y''(t)/2=1-3t ranges over [-2,1]. That uniform bound is indispensable.
    elapsed = Q(1, 2)
    endpoint_error = Q(1) ** 2 * (1 - Q(1))
    interior_error = elapsed**2 * (1 - elapsed)
    assert endpoint_error == 0 and interior_error > 0
    increment, endpoint = owner.reconstruct_box_taylor_partial(
        (I(0),),
        (I(-2, 2),),
        ((I(0), I(0)),),
        (I(-2, 1),),
        1,
        elapsed,
        order=1,
    )
    assert increment == endpoint == (I(Q(-1, 2), Q(1, 4)),)
    assert endpoint[0].contains(interior_error)


def test_partial_time_below_interval_grid_is_not_replaced_by_zero():
    elapsed = Q(1, 2**160)
    increment, endpoint = owner.reconstruct_box_taylor_partial(
        (I(0),),
        (I(-1, 2),),
        ((I(0), I(1)),),
        (I(0),),
        1,
        elapsed,
        order=1,
    )
    assert increment[0].contains(elapsed)
    assert endpoint[0].contains(elapsed)
    assert endpoint[0].hi > 0


def test_partial_time_normalizes_exact_clock_components_before_scaling():
    np = pytest.importorskip("numpy")
    source = _partial_arguments(
        full_duration=Q(np.int64(1), np.int64(2**32)),
        elapsed=Q(np.int64(1), np.int64(2**33)),
    )
    actual = owner.reconstruct_box_taylor_partial(**source)
    normalized = _partial_arguments(full_duration=Q(1, 2**32), elapsed=Q(1, 2**33))
    assert actual == owner.reconstruct_box_taylor_partial(**normalized)


def test_partial_time_consumes_ordered_evidence_generators_once():
    arguments = _partial_arguments()
    expected = owner.reconstruct_box_taylor_partial(**arguments)
    for name in ("initial_box", "tube", "local_remainder_bounds"):
        arguments[name] = iter(arguments[name])
    arguments["series"] = (iter(row) for row in arguments["series"])
    assert owner.reconstruct_box_taylor_partial(**arguments) == expected


@pytest.mark.parametrize("elapsed", [Q(0), Q(1, 8), Q(1, 4)])
def test_partial_time_rejects_disjoint_full_step_even_at_zero(elapsed):
    with pytest.raises(ArithmeticError, match="disjoint endpoint"):
        owner.reconstruct_box_taylor_partial(
            **_partial_arguments(
                initial_box=(I(0),),
                tube=(I(-1, 1),),
                series=((I(0), I(100), I(0)),),
                local_remainder_bounds=(I(0),),
                elapsed=elapsed,
            )
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"full_duration": 0},
        {"full_duration": -1},
        {"full_duration": True},
        {"full_duration": 0.25},
        {"full_duration": float("inf")},
        {"elapsed": -Q(1, 8)},
        {"elapsed": Q(1, 4) + Q(1, 2**100)},
        {"elapsed": True},
        {"elapsed": 0.125},
        {"elapsed": float("nan")},
        {"initial_box": ()},
        {"initial_box": (I(0),) * 65},
        {"initial_box": (True,)},
        {"tube": (I(0, 1),)},
        {"tube": ()},
        {"series": ((I(-1, 1), I(3), I(-2)),)},
        {"series": ((I(-1, 2), I(3)),)},
        {"series": ((I(-1, 2), True, I(-2)),)},
        {"series": {"row": (I(-1, 2), I(3), I(-2))}},
        {"local_remainder_bounds": ()},
        {"local_remainder_bounds": (False,)},
        {"order": True},
        {"order": 0},
        {"order": 17},
    ],
)
def test_partial_time_rejects_malformed_or_unsupported_domains(changes):
    with pytest.raises((ValueError, TypeError)):
        owner.reconstruct_box_taylor_partial(**_partial_arguments(**changes))
