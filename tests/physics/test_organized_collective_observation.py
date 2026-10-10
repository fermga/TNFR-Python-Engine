"""Circular collective means from independent marked phase constructions."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_collective_observation as owner


_FORM = tuple(Q((7 * node) % 19 - 9, 64) for node in range(54))
_PHASE = tuple(Q((11 * node) % 23 - 11, 128) for node in range(54))


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _observe(**changes):
    return owner._observe_organized_collective_observation(
        **(
            dict(
                form=_FORM,
                phase_offsets=_PHASE,
                relative_phase_origin=Q(2, 5),
            )
            | changes
        )
    )


def test_marked_signed_means_and_circular_lifts_match_independent_observation(mp):
    report = _observe()
    assert report.available
    assert report.form == _FORM
    for child, observation in enumerate(report.children):
        span = slice(9 * child, 9 * child + 9)
        assert observation.nodes == tuple(range(9 * child, 9 * child + 9))
        assert observation.form_mean == sum(_FORM[span]) / 9
        assert sum(observation.centered_form) == 0
        phase = tuple(_mp(mp, value) for value in _PHASE[span])
        origin = _mp(mp, Q(2, 5)) if child == 5 else mp.mpf(0)
        resultant = sum(mp.exp(1j * (value - phase[0])) for value in phase)
        anchor = phase[0] + mp.arg(resultant)
        residual = tuple(mp.arg(mp.exp(1j * (value - anchor))) for value in phase)
        residual_mean = sum(residual) / 9
        mean = anchor + residual_mean + origin
        _contains(mp, observation.rotated_resultant_real_bounds, resultant.real)
        _contains(mp, observation.rotated_resultant_imaginary_bounds, resultant.imag)
        _contains(mp, observation.phase_anchor_lift_bounds, anchor + origin)
        _contains(mp, observation.phase_mean_lift_bounds, mean)
        _contains(mp, observation.phase_mean_cosine_bounds, mp.cos(mean))
        _contains(mp, observation.phase_mean_sine_bounds, mp.sin(mean))
        assert abs(mean - (sum(phase) / 9 + origin)) < mp.mpf("1e-170")
        for bound, value in zip(observation.centered_phase_bounds, residual):
            _contains(mp, bound, value - residual_mean)
        assert observation.collar_cosine_lower_bound > 0
        assert observation.available and observation.unavailable_reason is None


def test_arithmetic_collar_mean_is_distinct_from_raw_argument(mp):
    phase = (Q(0),) * 8 + (Q(1),)
    report = _observe(phase_offsets=phase * 6, relative_phase_origin=0)
    expected = mp.mpf(1) / 9
    raw_arg = mp.arg(8 + mp.exp(1j))
    assert abs(raw_arg - expected) > mp.mpf("0.01")
    for child in report.children:
        _contains(mp, child.phase_mean_lift_bounds, expected)
        _contains(mp, child.phase_anchor_lift_bounds, raw_arg)
        assert child.phase_mean_lift_bounds.lo > child.phase_anchor_lift_bounds.hi


def test_independent_exact_full_turns_preserve_circular_mean_and_deviations(mp):
    turns = tuple((7 * node) % 11 - 5 for node in range(54))
    shifted = tuple(
        I(value) + 2 * turn * pi_interval() for value, turn in zip(_PHASE, turns)
    )
    report = _observe(phase_offsets=shifted, relative_phase_origin=pi_interval() / 2)
    assert report.available
    for child, observation in enumerate(report.children):
        span = slice(9 * child, 9 * child + 9)
        mean = _mp(mp, sum(_PHASE[span]) / 9)
        origin = mp.pi / 2 if child == 5 else 0
        _contains(
            mp,
            observation.phase_mean_lift_bounds,
            mean + origin + 2 * turns[9 * child] * mp.pi,
        )
        _contains(mp, observation.phase_mean_cosine_bounds, mp.cos(mean + origin))
        _contains(mp, observation.phase_mean_sine_bounds, mp.sin(mean + origin))
        for bound, value in zip(observation.centered_phase_bounds, _PHASE[span]):
            _contains(mp, bound, _mp(mp, value) - mean)


def test_global_argument_cut_does_not_remove_the_coherent_observation(mp):
    for side in (-1, 0, 1):
        offsets = tuple(
            pi_interval() + Q(side, 100) + Q(node % 9 - 4, 1000) for node in range(54)
        )
        report = _observe(phase_offsets=offsets, relative_phase_origin=0)
        assert report.available
        for child in report.children:
            mean = mp.pi + mp.mpf(side) / 100
            _contains(mp, child.phase_mean_lift_bounds, mean)
            _contains(mp, child.phase_mean_cosine_bounds, mp.cos(mean))
            _contains(mp, child.phase_mean_sine_bounds, mp.sin(mean))


@pytest.mark.parametrize("kind", ("zero", "boundary", "outside", "wide"))
def test_unavailable_phase_is_not_zero_or_a_passing_domain_flag(kind):
    pi = pi_interval()
    phases = {
        "zero": (I(0), 2 * pi / 3, -2 * pi / 3) * 3,
        "boundary": (I(0),) + (pi / 2,) * 4 + (-pi / 2,) * 4,
        "outside": (I(0),) * 7 + (I(2), I(-2)),
        "wide": (I(-4, 4),) * 9,
    }[kind]
    report = _observe(phase_offsets=phases + (Q(0),) * 45)
    assert not report.available
    unavailable = report.children[0]
    assert not unavailable.available
    assert unavailable.unavailable_reason is not None
    assert unavailable.phase_anchor_lift_bounds is None
    assert unavailable.phase_mean_lift_bounds is None
    assert unavailable.phase_mean_cosine_bounds is None
    assert unavailable.phase_mean_sine_bounds is None
    assert unavailable.centered_phase_bounds is None
    assert unavailable.collar_cosine_lower_bound is None
    assert unavailable.form_mean == sum(_FORM[:9]) / 9
    assert all(child.available for child in report.children[1:])


def test_common_donor_origin_does_not_widen_centered_phase_or_collar():
    narrow = _observe(relative_phase_origin=0)
    wide = _observe(relative_phase_origin=I(-10, 10))
    assert wide.available
    assert wide.children[:5] == narrow.children[:5]
    left, right = narrow.children[5], wide.children[5]
    assert left.centered_phase_bounds == right.centered_phase_bounds
    assert left.collar_cosine_lower_bound == right.collar_cosine_lower_bound
    assert right.phase_mean_lift_bounds.width > 20
    assert right.phase_mean_cosine_bounds == I(-1, 1)
    assert right.phase_mean_sine_bounds == I(-1, 1)


def test_local_lift_rate_is_the_arithmetic_mean_even_across_display_cut(mp):
    step = Q(1, 10**6)
    rates = tuple(Q(node % 7 - 2, 8) for node in range(54))
    offsets = tuple(pi_interval() + value for value in _PHASE)
    minus = _observe(
        phase_offsets=tuple(value - step * rate for value, rate in zip(offsets, rates))
    )
    plus = _observe(
        phase_offsets=tuple(value + step * rate for value, rate in zip(offsets, rates))
    )
    assert minus.available and plus.available
    for index, (before, after) in enumerate(zip(minus.children, plus.children)):
        measured = (after.phase_mean_lift_bounds - before.phase_mean_lift_bounds) / (
            2 * step
        )
        expected = sum(rates[9 * index : 9 * index + 9]) / 9
        _contains(mp, measured, _mp(mp, expected))
        assert measured.width < Q(1, 10**25)


@pytest.mark.parametrize("value", (True, complex(1), "0", float("nan"), float("inf")))
@pytest.mark.parametrize("field", ("form", "phase_offsets", "relative_phase_origin"))
def test_invalid_primitive_scalars_reject_before_observation(monkeypatch, field, value):
    def forbidden(**kwargs):
        pytest.fail("observation started before primitive admission completed")

    monkeypatch.setattr(owner, "_observe_child", forbidden)
    changed = value if field == "relative_phase_origin" else (Q(0),) * 53 + (value,)
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: changed})


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("value", ((0,) * 53, (0,) * 55, {0: 0}, {0}, "0"))
def test_registered_coordinate_order_and_count_are_required(field, value):
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: value})


def test_tiny_exact_signed_form_survives_and_report_is_immutable():
    tiny = -Q(1, 10**500)
    report = _observe(form=(tiny,) + (Q(0),) * 53)
    assert report.form[0] == tiny
    assert report.children[0].form_mean == tiny / 9
    assert sum(report.children[0].centered_form) == 0
    with pytest.raises(FrozenInstanceError):
        report.available = False
    with pytest.raises(FrozenInstanceError):
        report.children[0].available = False


def test_overlong_iterator_is_bounded_at_one_excess_coordinate():
    seen = []

    def unbounded():
        while True:
            seen.append(0)
            yield 0

    with pytest.raises(ValueError, match="54 registered coordinates"):
        _observe(phase_offsets=unbounded())
    assert len(seen) == 55
