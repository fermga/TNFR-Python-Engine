"""Static quadrature controls; no root search, record or trajectory evaluation."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import _organized_law_observation as owner
from tnfr.physics._sine_class_constitutive_comparison import _edge_law

EPSILON = Q(1, 10)


def _close(left, right):
    difference = left - right
    assert difference.contains(0)
    assert difference.width < Q(1, 10**24)


@pytest.mark.parametrize("angle", (Q(-1, 5), Q(0), Q(31, 100), Q(4, 5)))
@pytest.mark.parametrize("deformation", (Q(0), Q(1, 2), Q(1)))
def test_quadratures_match_actual_receiver_current_and_expanded_harmonics(
    angle, deformation
):
    cosine, sine = owner._frozen_edge_quadratures(angle, deformation=deformation)
    # Receiver r=c+1 sees marked-edge gap -a+q*epsilon. Its other edge
    # is unchanged and cancels in the odd/even combinations.
    currents = tuple(
        _edge_law(
            cos(level * EPSILON - angle), sin(level * EPSILON - angle), deformation
        )[1][0]
        for level in (0, 1, -1)
    )
    zero, positive, negative = currents
    _close(cosine, (positive - negative) / (2 * sin(EPSILON)))
    _close(sine, (positive + negative - 2 * zero) / (2 * (1 - cos(EPSILON))))
    harmonics = ((1, Q(1)), (9, deformation / 2187), (18, -deformation / 4374))
    expanded_cosine = sum(
        (
            weight * sin(order * EPSILON) * cos(order * angle)
            for order, weight in harmonics
        ),
        I(0),
    ) / sin(EPSILON)
    expanded_sine = sum(
        (
            weight * (1 - cos(order * EPSILON)) * sin(order * angle)
            for order, weight in harmonics
        ),
        I(0),
    ) / (1 - cos(EPSILON))
    _close(cosine, expanded_cosine)
    _close(sine, expanded_sine)


@pytest.mark.parametrize("angle", (Q(-2, 5), Q(0), Q(1, 3), I(Q(1, 8), Q(1, 7))))
def test_sine_member_retains_the_original_unit_phasor_quadratures(angle):
    cosine, sine = owner._frozen_edge_quadratures(angle, deformation=0)
    assert cosine == cos(angle) and sine == sin(angle)


def test_endpoint_signs_prove_a_circle_intersection_without_searching_for_it():
    cosine0, sine0 = owner._frozen_edge_quadratures(0, deformation=1)
    cosine1, sine1 = owner._frozen_edge_quadratures(pi_interval() / 9, deformation=1)
    assert (cosine0**2 + sine0**2 - 1).lo > 0
    assert (cosine1**2 + sine1**2 - 1).hi < 0
    decrement = (sin(9 * EPSILON) + sin(18 * EPSILON) / 2) / (2187 * sin(EPSILON))
    assert 0 < decrement.lo <= decrement.hi < Q(1, 100)
    _close(cosine1, cos(pi_interval() / 9) - decrement)
    _close(sine1, sin(pi_interval() / 9))
    # On 0<=a<=pi/9 the harmonic correction has absolute value at most
    # 3/(2*2187*sin(epsilon)); cos(a) decreases to cos(pi/9).
    lower = cos(pi_interval() / 9) - Q(3, 2) / (2187 * sin(EPSILON))
    assert lower.lo > Q(8, 9) > cos(Q(1, 2)).hi
    # Thus any unit-circle intersection has positive C and |arg|<1/2,
    # inside the acute 7/5 prior. This asserts no complete-record overlap.


def test_neighbor_current_range_covers_the_local_baseline_discrepancy():
    ninth_sine_scale = (1 - cos(9 * EPSILON)) / (2187 * (1 - cos(EPSILON)))
    assert 0 < ninth_sine_scale.lo <= ninth_sine_scale.hi < Q(1, 26)
    # The unknown collision location is never evaluated. Its sine contrast
    # correction and current deformation have these uniform amplitude bounds.
    discrepancy = 3 * ninth_sine_scale + Q(1, 1458)
    endpoint_current = sin(Q(1, 2)) + (sin(Q(9, 2)) - sin(Q(9)) / 2) / 2187
    assert discrepancy.hi < Q(1, 8) < endpoint_current.lo
    assert Q(1, 2) - Q(1, 48) - Q(1, 1458) > Q(1, 8)


@pytest.mark.parametrize("bad", (True, False, float("nan"), float("inf"), "0", None))
def test_invalid_angle_is_rejected_before_quadrature_arithmetic(bad):
    with pytest.raises((ValueError, TypeError)):
        owner._frozen_edge_quadratures(bad, deformation=1)


@pytest.mark.parametrize("bad", (True, Q(-1), Q(2), float("nan"), float("inf")))
def test_deformation_must_be_an_admitted_member_of_the_fixed_comparison(bad):
    with pytest.raises((ValueError, TypeError)):
        owner._frozen_edge_quadratures(Q(1, 5), deformation=bad)
