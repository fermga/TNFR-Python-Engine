"""Static forecast controls; no reserved endpoint or source is evaluated.

The derivative controls use independently specified states. Step-loop controls
replace the integrator with a manufactured translation and test wiring only.
"""

from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath
import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_computable_prediction as owner
from tnfr.physics import _organized_electrical_actuation as pulse_owner
from tnfr.physics import _organized_finite_observation_source as source_owner
from tnfr.physics._organized_finite_observation import (
    _evaluate_organized_finite_observation_word,
)


@pytest.fixture(scope="module", autouse=True)
def no_forecast_or_frozen_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("a static consumer control attempted a reserved calculation")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_assess_organized_computable_prediction", forbidden)
        patch.setattr(owner, "validated_metric_taylor_step", forbidden)
        patch.setattr(source_owner, "_assess_organized_finite_observation", forbidden)
        patch.setattr(pulse_owner, "_assess_sine_recovery_resources", forbidden)
        patch.setattr(pulse_owner, "_derive_organized_outer_resource_bounds", forbidden)
        yield


@pytest.fixture(scope="module")
def arithmetic():
    mp = mpmath.mp.clone()
    mp.dps = 100
    return mp


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _laplacian(values):
    return tuple(
        2 * values[c] - values[(c - 1) % 5] - values[(c + 1) % 5] for c in range(5)
    )


def test_schedule_retains_every_switch_and_the_unforced_future_suffix():
    stages = owner._forecast_stages()
    assert len(stages) == 32
    assert stages[0].start == 0 and stages[0].duration == 200
    assert stages[-1].start == Q(4300003, 20000)
    assert stages[-1].duration == Q(19997, 20000)
    assert stages[-1].start + stages[-1].duration == 216
    assert all(a.start + a.duration == b.start for a, b in zip(stages, stages[1:]))
    for child in range(5):
        block = stages[1 + 6 * child : 7 + 6 * child]
        assert tuple(stage.name for stage in block) == ("plateau", "ramp") * 3
        assert tuple(stage.child for stage in block) == (child,) * 6
        assert tuple(stage.base for stage in block[::2]) == (0, Q(1, 10), -Q(1, 10))
        assert tuple(stage.increment for stage in block[1::2]) == (
            Q(1, 10),
            -Q(1, 5),
            Q(1, 10),
        )
        assert all(stage.duration == Q(1, 100000) for stage in block[::2])
        assert all(stage.duration == 1 for stage in block[1::2])
    steps = sum(-(-stage.duration // owner._MAX_STEP) for stage in stages)
    assert steps == 1743 < owner._MAX_STEPS


def test_polynomial_command_matches_the_original_word_and_is_continuous():
    stages = owner._forecast_stages()
    for stage in stages[1:-1]:
        for fraction in (Q(0), Q(1, 3), Q(1)):
            time = stage.start + fraction * stage.duration
            actual = stage.command(time)
            original = _evaluate_organized_finite_observation_word(elapsed=time - 200)
            assert all(
                bound.contains(value)
                for value, bound in zip(actual, original.phase_offset_bounds)
            )
            assert sum(actual) == 0
            assert max(abs(actual[i] - actual[(i + 1) % 5]) for i in range(5)) <= Q(
                1, 10
            )
    for left, right in zip(stages, stages[1:]):
        boundary = right.start
        assert left.command(boundary) == right.command(boundary)
    assert stages[-1].command(Q(216)) == (0,) * 5


@pytest.mark.parametrize("omit_word", (False, True))
def test_full_degree_four_rows_and_command_removal_match_independent_algebra(
    arithmetic, omit_word
):
    mp = arithmetic
    stage = next(
        value
        for value in owner._forecast_stages()
        if value.child == 2 and value.increment == -Q(1, 5)
    )
    time = stage.start + stage.duration / 3
    state = (Q(2, 7), -Q(3, 11), Q(5, 13), -Q(1, 2), Q(4, 9)) + (
        Q(1, 5),
        -Q(2, 7),
        Q(3, 8),
        Q(4, 13),
        -Q(5, 17),
    )
    flow, domain = owner._command_removed_field(stage, omit_word=omit_word)
    rates = flow(I(time), tuple(I(value) for value in state))
    command = (Q(0),) * 5 if omit_word else stage.command(time)
    y = tuple(_mp(mp, value) for value in state[:5])
    phase = tuple(_mp(mp, a + b) for a, b in zip(state[5:], command))
    gamma = 1 / (1023 * mp.pi)
    gradient = _laplacian(y)
    current = tuple(
        mp.sin(phase[(c - 1) % 5] - phase[c]) + mp.sin(phase[(c + 1) % 5] - phase[c])
        for c in range(5)
    )
    expected = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current)) + tuple(
        gamma * value / 4 for value in gradient
    )
    for bound, value in zip(rates, expected):
        _contains(mp, bound, value)
    assert domain(I(stage.start, stage.start + stage.duration), ()) == (Q(1),)
    assert domain(I(stage.start - 1, stage.start), ()) == (Q(-1),)
    assert domain(I(stage.start, stage.start + stage.duration + 1), ()) == (Q(-1),)
    assert sum(rates[:5], I(0)).contains(0)
    assert sum(rates[5:], I(0)).contains(0)

    if not omit_word:
        # Recover the original rows, including the large prescribed offset.
        # This checks both cancellation identities, not just the shifted sine.
        r = mp.mpf(1) / 3
        direction = tuple(mp.mpf(int(c == 2)) - mp.mpf(1) / 5 for c in range(5))
        column = (Q(2, 5), Q(0), -Q(1, 5), -Q(1, 5), Q(0))
        inverse = tuple(_mp(mp, column[(c - 2) % 5]) for c in range(5))
        amplitude = -mp.mpf(1) / 5
        offset = tuple(4 * amplitude * v * (6 * r - 6 * r**2) / gamma for v in inverse)
        offset_rate = tuple(4 * amplitude * v * (6 - 12 * r) / gamma for v in inverse)
        phase_rate = tuple(amplitude * k * (6 * r - 6 * r**2) for k in direction)
        forcing = tuple(a + b / 4 for a, b in zip(offset_rate, _laplacian(offset)))
        full_gradient = _laplacian(tuple(a + b for a, b in zip(y, offset)))
        waveform = _evaluate_organized_finite_observation_word(elapsed=time - 200)
        for c in range(5):
            _contains(
                mp,
                rates[c] + waveform.form_tau_rate_bounds[c],
                (-full_gradient[c] + gamma * current[c]) / 4 + forcing[c],
            )
            _contains(
                mp,
                rates[5 + c] + waveform.phase_tau_rate_bounds[c],
                gamma * full_gradient[c] / 4,
            )
            assert abs(
                expected[5 + c] + phase_rate[c] - gamma * full_gradient[c] / 4
            ) < mp.mpf("1e-95")


def test_zero_force_rows_keep_uniform_translations_and_both_global_means():
    stage = owner._forecast_stages()[0]
    flow, _ = owner._command_removed_field(stage, omit_word=False)
    stationary = tuple(I(value) for value in (Q(7, 3),) * 5 + (Q(-4, 9),) * 5)
    assert all(value.contains(0) for value in flow(I(1), stationary))
    state = tuple(I(Q(c * c - 5, 8)) for c in range(10))
    translated = tuple(
        value + (Q(3, 2) if c < 5 else -Q(7, 4)) for c, value in enumerate(state)
    )
    original, moved = flow(I(1), state), flow(I(1), translated)
    assert original == moved


def test_preparation_error_solves_the_coupled_mean_error_inequalities():
    source, _, _ = owner._source_and_suffix_bounds()
    g, horizon = Q(1, 3000), Q(216)
    form_radius, phase_radius = 3 * Q(1, 10**9), 3 * Q(2, 10**12)
    hidden = 2 * (phase_radius + g * form_radius)
    forcing = g * hidden**2 / 9
    phase_error = g * horizon * source
    assert source == horizon * forcing + g * horizon * phase_error
    assert source < Q(16, 10**25)
    assert phase_error < Q(12, 10**26)
    assert 4 * hidden**2 / 9 < Q(1, 10**17)
    assert source < owner._NUMERICAL_BUDGET < owner._READING_ERROR


def test_identity_extension_and_tail_loss_use_the_full_fine_support():
    _, margins, loss = owner._source_and_suffix_bounds()
    pi, g, duration = pi_interval(), Q(1, 3000), Q(16)
    phase = 2 * g * Q(1, 100) + 2 * g**2 * duration
    residual, amplitude, outer = Q(1, 10**6), Q(1, 10), Q(7, 5)
    assert margins == (
        2 * pi.lo / 9 - 2 * residual - 2 * phase,
        pi.lo / 2 - 2 * pi.hi / 9 - 2 * residual - 2 * phase,
        pi.lo / 2 - outer - amplitude - 2 * residual - 2 * phase,
        pi.lo / 2 - 2 * residual - 4 * phase,
    )
    assert min(margins) > Q(7, 100)
    form = Q(1, 100) + g * duration
    tail = Q(19997, 20000)
    assert loss == 45 * (8 * form) ** 2 * tail / 4
    # A static fine-node configuration exposes the degree-two substitution:
    # the magnitude-only fine proof cannot use a 180*M^2 loss-rate bound.
    state = {(c, j): (-1) ** (c + j) * form for c in range(5) for j in range(9)}
    gradient = tuple(
        4 * state[c, j]
        - state[(c - 1) % 5, j]
        - state[(c + 1) % 5, j]
        - state[c, (j - 1) % 9]
        - state[c, (j + 1) % 9]
        for c, j in state
    )
    rate = sum(value**2 for value in gradient) / 4
    assert 180 * form**2 < rate <= loss / tail


def test_interval_separation_accounts_for_both_numerical_radii_before_noise():
    left_center = (Q(0),) * 10
    right_center = (Q(3, 2),) + (Q(0),) * 9
    left = owner._endpoint_bands(left_center, Q(1))
    right = owner._endpoint_bands(right_center, Q(1))
    assert abs(left_center[0] - right_center[0]) > 1
    assert owner._interval_separation(left, right) == 0
    right_center = (Q(4),) + (Q(0),) * 9
    right = owner._endpoint_bands(right_center, Q(1))
    assert owner._interval_separation(left, right) == 2
    expanded_left = owner._endpoint_bands(left_center, Q(1), Q(1))
    expanded_right = owner._endpoint_bands(right_center, Q(1), Q(1))
    assert owner._interval_separation(expanded_left, expanded_right) == 0
    assert len(expanded_left) == 5
    assert expanded_left[0].lo == -2 and expanded_left[0].hi == 2


def _manufactured_step(center, radius, metric, duration, flow, domain, **kwargs):
    """Stub translation: it deliberately never evaluates the scientific flow."""
    endpoint = tuple(value + duration for value in center)
    return (
        SimpleNamespace(
            tube=tuple(I(value, value + duration) for value in center),
            picard_interior_margin=Q(1, 100),
            center_polynomial_bounds=tuple(I(value) for value in endpoint),
            local_remainder_bounds=(I(0),) * 10,
            growth_factor_upper_bound=Q(1),
            local_metric_error_upper_bound=Q(1, 10**20),
            endpoint_center=endpoint,
            endpoint_radius=radius + Q(1, 10**20),
        ),
        None,
        None,
    )


def test_step_wiring_splits_switches_and_carries_the_entire_reached_ball(monkeypatch):
    stages = (
        owner._ForecastStage(Q(1, 7), Q(3, 10), "synthetic_first"),
        owner._ForecastStage(Q(31, 70), Q(2, 9), "synthetic_second"),
    )
    calls = []

    def instrumented(center, radius, metric, duration, flow, domain, **kwargs):
        calls.append((center, radius, duration, kwargs))
        return _manufactured_step(
            center, radius, metric, duration, flow, domain, **kwargs
        )

    monkeypatch.setattr(owner, "validated_metric_taylor_step", instrumented)
    center, radius = (Q(0),) * 10, Q(1, 10**18)
    result = owner._forecast_leg("synthetic", stages, center, radius, omit_word=True)
    assert result.admitted and result.reason is None
    assert tuple(call[2] for call in calls) == (
        Q(1, 8),
        Q(1, 8),
        Q(1, 20),
        Q(1, 8),
        Q(7, 72),
    )
    assert calls[0][:2] == (center, radius)
    for previous, current in zip(result.steps, calls[1:]):
        assert current[:2] == (previous.endpoint_center, previous.endpoint_radius)
    assert all(call[3]["time_dependent"] is True for call in calls)
    assert all(call[3]["growth_rate"] == Q(1, 3000) for call in calls)
    assert all(call[3]["order"] == 16 for call in calls)
    assert result.end == Q(1, 7) + Q(3, 10) + Q(2, 9)
    assert result.endpoint_center == (Q(3, 10) + Q(2, 9),) * 10
    assert result.endpoint_radius == radius + len(calls) * Q(1, 10**20)


@pytest.mark.parametrize("failure", ("step_limit", "radius", "solver"))
def test_declared_failures_stop_without_a_retry_or_reset(monkeypatch, failure):
    calls = []

    def instrumented(*args, **kwargs):
        calls.append((args, kwargs))
        if failure == "solver":
            return None, (I(-1, 1),) * 10, "synthetic_tube_unavailable"
        result = _manufactured_step(*args, **kwargs)
        if failure == "radius":
            result[0].endpoint_radius = 2 * owner._NUMERICAL_BUDGET
        return result

    monkeypatch.setattr(owner, "validated_metric_taylor_step", instrumented)
    if failure == "step_limit":
        monkeypatch.setattr(owner, "_MAX_STEPS", 2)
    initial = (Q(0),) * 10
    result = owner._forecast_leg(
        "synthetic",
        (owner._ForecastStage(Q(0), Q(1), "synthetic"),),
        initial,
        Q(0),
        omit_word=False,
    )
    assert not result.admitted
    expected = {
        "step_limit": (2, "declared_step_budget_exhausted"),
        "radius": (1, "declared_numerical_radius_exceeded"),
        "solver": (1, "synthetic_tube_unavailable"),
    }
    assert (len(calls), result.reason) == expected[failure]
    if failure == "solver":
        assert result.endpoint_center == initial and result.end == 0
        assert result.steps == ()


def test_each_branch_step_limit_includes_its_carried_formation_prefix(monkeypatch):
    monkeypatch.setattr(owner, "validated_metric_taylor_step", _manufactured_step)
    result = owner._forecast_leg(
        "synthetic",
        (owner._ForecastStage(Q(3), Q(1), "synthetic"),),
        (Q(0),) * 10,
        Q(0),
        omit_word=True,
        prefix_steps=owner._MAX_STEPS - 1,
    )
    assert not result.admitted
    assert result.reason == "declared_step_budget_exhausted"
    assert len(result.steps) == 1 and result.end == Q(25, 8)


def test_noncontiguous_stages_cannot_silently_reset_the_clock(monkeypatch):
    monkeypatch.setattr(owner, "validated_metric_taylor_step", _manufactured_step)
    stages = (
        owner._ForecastStage(Q(0), Q(1, 8), "synthetic_first"),
        owner._ForecastStage(Q(1, 4), Q(1, 8), "synthetic_second"),
    )
    with pytest.raises(ArithmeticError, match="continuous time"):
        owner._forecast_leg("synthetic", stages, (Q(0),) * 10, Q(0), omit_word=True)
