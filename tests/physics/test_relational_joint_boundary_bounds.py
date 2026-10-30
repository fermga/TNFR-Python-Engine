"""Outward full-field evidence at the declared exact joint-pairing boundary.

The amplitude is a symbolic trigonometric real enclosed here, never a rounded
graph attribute promoted to an exact boundary. No trajectories are evaluated.
"""

from fractions import Fraction as Q
from itertools import combinations

import mpmath as mp
import pytest

from tests.physics.test_relational_joint_grouping import _support
from tnfr.mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    certified_sine_bounds,
)
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt

D = Q(1, 8)
PHASE = (Q(0), D, 2 * D, 3 * D, Q(1), Q(1), Q(2), Q(2), Q(3), Q(3))
FORM_SIGN = (1, -1, 1, -1, 0, 0, 0, 0, 0, 0)
NEAREST_GROUPS = (
    (1, 2),
    (0, 2, 3),
    (0, 1, 3),
    (1, 2),
    (5,),
    (4,),
    (7,),
    (6,),
    (9,),
    (8,),
)


def _sine(value):
    return I(*certified_sine_bounds(value))


def _chord(value):
    return 2 * (1 - I(*certified_cosine_bounds(value)))


def _distance_terms(first, second):
    """Exact linear combination of chord symbols after using 4*u_c^2=F2-F1."""
    coefficient = Q((FORM_SIGN[second] - FORM_SIGN[first]) ** 2, 4)
    terms = {D: -coefficient, 2 * D: coefficient}
    gap = abs(PHASE[second] - PHASE[first])
    if gap:
        terms[gap] = terms.get(gap, Q(0)) + 1
    return {angle: weight for angle, weight in terms.items() if weight}


def _distance_bound(first, second):
    return sum(
        (
            weight * _chord(angle)
            for angle, weight in _distance_terms(first, second).items()
        ),
        I(0),
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def boundary():
    square = (_chord(2 * D) - _chord(D)) / 4
    amplitude = sqrt(square)
    neighbors = _support()
    form_rates = tuple(
        sum((_sine(PHASE[j] - PHASE[i]) for j in row), I(0))
        / (len(row) * pi_interval())
        for i, row in enumerate(neighbors)
    )
    phase_numerators = tuple(
        Q(sum(FORM_SIGN[i] - FORM_SIGN[j] for j in row), len(row))
        for i, row in enumerate(neighbors)
    )
    assert phase_numerators == FORM_SIGN
    rates = {}
    for i, j in combinations(range(10), 2):
        form_gap = (FORM_SIGN[j] - FORM_SIGN[i]) * amplitude
        phase_rate_gap = (phase_numerators[j] - phase_numerators[i]) * amplitude
        rates[i, j] = 2 * form_gap * (form_rates[j] - form_rates[i])
        rates[i, j] += 2 * _sine(PHASE[j] - PHASE[i]) * phase_rate_gap / pi_interval()
    return square, amplitude, rates


def test_exact_ties_and_every_outsider_have_separate_algebraic_evidence(boundary):
    square, amplitude, _ = boundary
    assert 0 < square.lo < square.hi < D**2
    assert Q(1, 10) < amplitude.lo < amplitude.hi < Q(11, 100)
    outsider_count = 0
    for node, group in enumerate(NEAREST_GROUPS):
        exact_terms = _distance_terms(node, group[0])
        # Equality is an identity of chord expressions, not interval overlap.
        assert all(
            _distance_terms(node, candidate) == exact_terms for candidate in group
        )
        nearest = _distance_bound(node, group[0])
        for other in range(10):
            if other != node and other not in group:
                assert _distance_bound(node, other).lo > nearest.hi
                outsider_count += 1
    assert outsider_count == 74
    assert _distance_terms(0, 1) == _distance_terms(0, 2) == {2 * D: Q(1)}
    assert _distance_terms(4, 5) == {}


def test_both_complete_rows_resolve_one_sided_maps_and_support(boundary):
    _, amplitude, rates = boundary
    assert rates[0, 1].hi < 0
    assert rates[1, 2].lo > 0
    assert rates[2, 3].hi < 0
    assert rates[0, 2] == rates[1, 3] == I(0)

    def nearest_for(orientation, time_direction):
        result = []
        for node, group in enumerate(NEAREST_GROUPS):
            values = {
                other: orientation
                * time_direction
                * rates[tuple(sorted((node, other)))]
                for other in group
            }
            candidates = [
                other
                for other, value in values.items()
                if all(
                    value.hi < bound.lo for key, bound in values.items() if key != other
                )
            ]
            assert len(candidates) == 1
            result.append(candidates[0])
        return tuple(result)

    forward = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)
    backward = (2, 2, 1, 1, 5, 4, 7, 6, 9, 8)
    assert nearest_for(1, 1) == nearest_for(-1, -1) == forward
    assert nearest_for(1, -1) == nearest_for(-1, 1) == backward
    assert all(forward[partner] == node for node, partner in enumerate(forward))
    assert tuple(
        node for node, partner in enumerate(backward) if backward[partner] != node
    ) == (0, 3)
    neighbors = _support()
    for node, partner in enumerate(forward):
        assert partner not in neighbors[node]
        assert neighbors[node] == neighbors[partner]

    # Independent high-precision complete field, not the three simplified brackets.
    with mp.workdps(100):
        theta = list(map(_mp, PHASE))
        u = mp.sqrt(2 * (mp.cos(_mp(D)) - mp.cos(2 * _mp(D)))) / 2
        assert _mp(amplitude.lo) < u < _mp(amplitude.hi)
        form = [coefficient * u for coefficient in FORM_SIGN]
        xdot = [
            sum(mp.sin(theta[j] - theta[i]) for j in row) / (len(row) * mp.pi)
            for i, row in enumerate(neighbors)
        ]
        phasedot = [
            sum(form[i] - form[j] for j in row) / (len(row) * mp.pi)
            for i, row in enumerate(neighbors)
        ]
        for (i, j), bound in rates.items():
            expected = 2 * (form[j] - form[i]) * (xdot[j] - xdot[i])
            expected += 2 * mp.sin(theta[j] - theta[i]) * (phasedot[j] - phasedot[i])
            if bound.lo == bound.hi:
                assert abs(expected - _mp(bound.lo)) < mp.mpf("1e-85")
            else:
                assert _mp(bound.lo) <= expected <= _mp(bound.hi)


def test_rounding_the_declared_amplitude_does_not_preserve_exact_ties(boundary):
    square, amplitude, _ = boundary
    represented = Q.from_float(float(amplitude.midpoint))
    deviation = I(represented**2) - square
    # This represented preparation has a nonzero margin 4*(u_repr^2-u_c^2).
    # Its smallness cannot make it the exact tied source or supply a horizon.
    assert not deviation.contains(0)
    assert deviation.abs_max < Q(1, 10**16)
