"""Independent scalar integral and admission controls for decay convolution."""

from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tnfr.mathematics._comparison_flow import decay_convolution_upper
from tnfr.physics import cycle_relaxation
from tnfr.physics.reversible_eigenmode_reference import _negative_exp_bounds


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 250
    return context


def _mp(context, value):
    return context.mpf(int(value.numerator)) / int(value.denominator)


@pytest.mark.parametrize(
    "first,second,time",
    (
        (Q(0), Q(0), Q(0)),
        (Q(0), Q(0), Q(3, 7)),
        (Q(0), Q(2), Q(1, 3)),
        (Q(3, 4), Q(0), Q(7, 5)),
        (Q(2, 7), Q(2, 7), Q(3)),
        (Q(2), Q(7), Q(1, 3)),
        (Q(7), Q(2), Q(1, 3)),
        (Q(1, 3), Q(30, 3), Q(30)),
        (Q(1), Q(1) + Q(1, 2**600), Q(1)),
        (Q(1), Q(30), Q(1, 2**600)),
        (Q(1, 2**600), Q(30, 2**600), Q(2**600)),
    ),
)
def test_upper_encloses_independent_high_precision_integral(mp, first, second, time):
    bounds = (_negative_exp_bounds(first * time), _negative_exp_bounds(second * time))
    actual = decay_convolution_upper(first, second, time, *bounds)
    a, b, t = (_mp(mp, value) for value in (first, second, time))
    if first == second:
        expected = t * mp.exp(-a * t)
    else:
        separation = abs(a - b)
        expected = mp.exp(-min(a, b) * t) * -mp.expm1(-separation * t) / separation
    assert actual >= 0
    assert expected <= _mp(mp, actual)
    if time:
        assert actual > 0
        assert _mp(mp, actual) - expected < abs(expected) * mp.mpf("1e-35")
    else:
        assert actual == 0
    # Commutativity follows from reversing the integration variable. The
    # supplied endpoint pairs remain associated with their own rates.
    assert actual == decay_convolution_upper(second, first, time, bounds[1], bounds[0])


@pytest.mark.parametrize("rates", ((Q(1, 3), Q(2)), (Q(7), Q(7))))
def test_upper_matches_independent_quadrature(mp, rates):
    time = Q(3, 5)
    first, second = rates
    bounds = (_negative_exp_bounds(first * time), _negative_exp_bounds(second * time))
    actual = decay_convolution_upper(first, second, time, *bounds)
    a, b, t = (_mp(mp, value) for value in (first, second, time))
    integral = mp.quad(lambda s: mp.exp(-a * (t - s) - b * s), [0, t])
    assert integral <= _mp(mp, actual)
    assert _mp(mp, actual) - integral < mp.mpf("1e-35")


def test_nearly_equal_rates_and_tiny_time_keep_a_positive_finite_allowance():
    tiny = Q(1, 2**1200)
    near_one = (1 - tiny, Q(1))
    # These broad yet valid pairs deliberately make subtraction uninformative.
    actual = decay_convolution_upper(Q(1), Q(1) + tiny, tiny / 2, near_one, near_one)
    assert actual == tiny / 2
    assert actual.denominator == 2**1201


def test_equal_rates_and_exact_zero_decay_are_regular_branches():
    bounds = (Q(1, 3), Q(2, 5))
    assert decay_convolution_upper(Q(1), Q(1), Q(1), bounds, bounds) == Q(2, 5)
    assert decay_convolution_upper(Q(0), Q(0), Q(7, 3), (Q(1),) * 2, (Q(1),) * 2) == Q(
        7, 3
    )


def test_fixed_width_fraction_payloads_normalize_before_products():
    values = (Q(0), Q(1, 2**62), Q(2**62), (Q(1), Q(1)), (Q(1, 3), Q(2, 5)))

    def payload(value):
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    prepared = tuple(payload(value) for value in values[:3]) + tuple(
        tuple(payload(value) for value in pair) for pair in values[3:]
    )
    with np.errstate(over="raise"):
        actual = decay_convolution_upper(*prepared)
    assert actual == decay_convolution_upper(*values) == Q(2**63, 3)
    assert type(actual.numerator) is type(actual.denominator) is int


def test_cycle_compatibility_wrapper_delegates_without_changing_inputs(monkeypatch):
    arguments = (Q(1), Q(2), Q(3), (Q(0), Q(1)), (Q(0), Q(1)))
    seen = []

    def shared(*args):
        seen.append(args)
        return Q(17, 23)

    monkeypatch.setattr(cycle_relaxation, "decay_convolution_upper", shared)
    assert cycle_relaxation._duhamel_upper(*arguments) == Q(17, 23)
    assert seen == [arguments]


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("invalid", (True, 0.0, float("nan"), float("inf"), 1))
def test_unprepared_scalars_reject_even_at_zero_time(position, invalid):
    arguments = [Q(1), Q(2), Q(0), (Q(1), Q(1)), (Q(1), Q(1))]
    arguments[position] = invalid
    with pytest.raises(TypeError):
        decay_convolution_upper(*arguments)


@pytest.mark.parametrize("position", range(3))
def test_negative_rates_or_time_reject(position):
    arguments = [Q(1), Q(2), Q(1), (Q(0), Q(1)), (Q(0), Q(1))]
    arguments[position] = Q(-1, 2**600)
    with pytest.raises(ValueError, match="nonnegative"):
        decay_convolution_upper(*arguments)


@pytest.mark.parametrize(
    "invalid",
    (
        (),
        (Q(1),),
        (Q(0), Q(1), Q(1)),
        {Q(0), Q(1)},
        {0: Q(0), 1: Q(1)},
        "01",
        (True, Q(1)),
        (Q(0), 1.0),
        (Q(-1), Q(1)),
        (Q(1), Q(0)),
        (Q(0), Q(2)),
        (Q(0), Q(0)),
    ),
)
def test_invalid_prepared_exponential_pairs_reject(invalid):
    with pytest.raises((TypeError, ValueError)):
        decay_convolution_upper(Q(1), Q(2), Q(1), invalid, (Q(0), Q(1)))


@pytest.mark.parametrize(
    "arguments",
    (
        (Q(0), Q(1), Q(1), (Q(0), Q(1, 2)), (Q(0), Q(1))),
        (Q(1), Q(2), Q(0), (Q(0), Q(1)), (Q(0), Q(1, 2))),
        (Q(1), Q(2), Q(1), (Q(0), Q(1, 4)), (Q(1, 2), Q(1))),
        (Q(1), Q(1), Q(1), (Q(1, 2), Q(1)), (Q(0), Q(1, 4))),
    ),
)
def test_known_exponential_consistency_failures_are_not_certified(arguments):
    with pytest.raises(ValueError):
        decay_convolution_upper(*arguments)


def test_ordered_endpoint_iterators_are_detached():
    first, second = [Q(1, 3), Q(2, 5)], [Q(1, 8), Q(1, 6)]
    actual = decay_convolution_upper(Q(1), Q(2), Q(1), iter(first), iter(second))
    first[:] = [Q(0), Q(0)]
    second[:] = [Q(1), Q(1)]
    assert actual == Q(11, 40)
