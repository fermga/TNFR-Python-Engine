"""Faithful moment coordinates and singular counterexamples of the full star."""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.physics.phase_response import (
    derive_phase_moment_motion,
    derive_sine_star_moment_closure,
)
from tnfr.sdk import relational_report_to_dict


def _fine_star(phasors, forms):
    """Evaluate full nodal rows first, then differentiate their observation."""
    neighbors = ((1, 2), (0,), (0,))
    phase_rates = tuple(
        forms[i] - sum((forms[j] for j in row), Q(0)) / len(row)
        for i, row in enumerate(neighbors)
    )
    form_rates = (
        sum((s for _, s in phasors), Q(0)) / 2,
        -phasors[0][1],
        -phasors[1][1],
    )
    phase_acceleration = tuple(
        form_rates[i] - sum((form_rates[j] for j in row), Q(0)) / len(row)
        for i, row in enumerate(neighbors)
    )
    rates = tuple(phase_rates[i] - phase_rates[0] for i in (1, 2))
    acceleration = tuple(phase_acceleration[i] - phase_acceleration[0] for i in (1, 2))
    motion = derive_phase_moment_motion(phasors, rates, epsilon=0)
    mdot = (
        sum(a * c - r * r * s for (c, s), r, a in zip(phasors, rates, acceleration)),
        sum(a * s + r * r * c for (c, s), r, a in zip(phasors, rates, acceleration)),
    )
    storage = sum((forms[i] - forms[0]) ** 2 / 2 for i in (1, 2)) + sum(
        1 - c for c, _ in phasors
    )
    return motion, mdot, storage


@pytest.mark.parametrize(
    "phasors,forms",
    (
        (((Q(3, 5), Q(4, 5)), (Q(4, 5), Q(3, 5))), (Q(0), Q(3, 4), Q(-1, 4))),
        (((1, 0), (0, 1)), (Q(5), Q(3), Q(-2))),
        (((-1, 0), (Q(-3, 5), Q(-4, 5))), (Q(-3), Q(7, 2), Q(-7, 2))),
        (((Q(3, 5), Q(4, 5)), (Q(3, 5), Q(-4, 5))), (Q(2), Q(2), Q(2))),
    ),
)
def test_regular_chart_reconstructs_state_and_pushes_forward_both_complete_rows(
    phasors, forms
):
    motion, mdot, storage = _fine_star(phasors, forms)
    report = derive_sine_star_moment_closure(
        motion.first_resultant, motion.first_rate_weighted_resultant
    )
    assert report.relative_mean_form == (forms[1] + forms[2]) / 2 - forms[0]
    assert report.internal_form_squared == ((forms[1] - forms[2]) / 2) ** 2
    assert report.full_storage == storage
    assert report.first_resultant_rate == motion.first_resultant_rate
    assert report.first_rate_weighted_resultant_rate == mdot
    (c, s), (d, t) = phasors
    assert report.phase_product == (c * d - s * t, c * t + s * d)
    swapped, _, _ = _fine_star(tuple(reversed(phasors)), (forms[0], forms[2], forms[1]))
    assert (
        derive_sine_star_moment_closure(
            swapped.first_resultant, swapped.first_rate_weighted_resultant
        )
        == report
    )
    shifted, _, _ = _fine_star(phasors, tuple(x + Q(137, 11) for x in forms))
    assert (
        derive_sine_star_moment_closure(
            shifted.first_resultant, shifted.first_rate_weighted_resultant
        )
        == report
    )


def test_admitted_moments_can_have_irrational_fine_lifts_and_storage_is_conserved():
    report = derive_sine_star_moment_closure((1, 0), (2, 3))
    assert report.relative_mean_form == 1
    assert report.internal_form_squared == 3
    # Leaf phasors have imaginary parts +/-sqrt(3)/2, not rational.
    with mp.workdps(80):
        c, s, mr, mi = map(mp.mpf, (1, 0, 2, 3))
        u = mp.sqrt(3)
        phasors = (mp.mpc(mp.mpf("0.5"), u / 2), mp.mpc(mp.mpf("0.5"), -u / 2))
        rates = (2 + u, 2 - u)
        assert abs(
            sum(r * z for r, z in zip(rates, phasors)) - mp.mpc(mr, mi)
        ) < mp.mpf("1e-75")

        def energy(c, s, mr, mi):
            norm = c * c + s * s
            a = (mr * c + mi * s) / norm
            b = (mi * c - mr * s) / norm
            return a * a / 4 + b * b * norm / (4 - norm) + 2 - c

        def real(q):
            return mp.mpf(q.numerator) / q.denominator

        direction = tuple(
            map(
                real,
                report.first_resultant_rate + report.first_rate_weighted_resultant_rate,
            )
        )
        derivative = mp.diff(
            lambda h: energy(*(v + h * dv for v, dv in zip((c, s, mr, mi), direction))),
            0,
        )
        assert abs(derivative) < mp.mpf("1e-75")


def test_coincident_phases_hide_internal_form_and_have_different_next_motion():
    pairs = ((1, 0), (1, 0))
    left = _fine_star(pairs, (Q(0), Q(0), Q(0)))
    right = _fine_star(pairs, (Q(0), Q(1), Q(-1)))
    assert left[0].first_resultant == right[0].first_resultant == (2, 0)
    assert (
        left[0].first_rate_weighted_resultant
        == right[0].first_rate_weighted_resultant
        == (0, 0)
    )
    assert left[1] == (0, 0)
    assert right[1] == (0, 2)


def test_antipodal_nonzero_motion_keeps_orientation_but_loses_relative_mean_form():
    phasors = ((1, 0), (-1, 0))
    left = _fine_star(phasors, (Q(-1, 2), Q(1), Q(0)))
    right = _fine_star(phasors, (Q(1, 2), Q(0), Q(-1)))
    assert left[0].first_resultant == right[0].first_resultant == (0, 0)
    assert (
        left[0].first_rate_weighted_resultant
        == right[0].first_rate_weighted_resultant
        == (1, 0)
    )
    assert left[2] == right[2] == Q(13, 4)
    assert left[1] == (0, 4)
    assert right[1] == (0, -4)


def test_antipodal_zero_motion_loses_orientation_as_well():
    forms = (Q(0),) * 3
    left = _fine_star(((1, 0), (-1, 0)), forms)
    right = _fine_star(((0, 1), (0, -1)), forms)
    assert left[0].first_resultant == right[0].first_resultant == (0, 0)
    assert (
        left[0].first_rate_weighted_resultant
        == right[0].first_rate_weighted_resultant
        == (0, 0)
    )
    assert left[1] == (0, 0)
    assert right[1] == (0, -2)


@pytest.mark.parametrize(
    "z,m",
    (
        ((0, 0), (1, 0)),
        ((2, 0), (0, 0)),
        ((2, 1), (0, 0)),
        ((True, 0), (0, 0)),
        ((1, 0), (False, 0)),
        ((float("nan"), 0), (0, 0)),
        ((1, 0), (float("inf"), 0)),
        ((1,), (0, 0)),
        ((1, 0), (0, 0, 0)),
        ({1, 0}, (0, 0)),
        ((1, 0), {0: 0, 1: 0}),
    ),
)
def test_singular_or_invalid_moment_inputs_reject(z, m):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        derive_sine_star_moment_closure(z, m)


def test_near_boundaries_are_exact_and_export_does_not_erase_small_values():
    tiny = Q(1, 10**400)
    near_zero = derive_sine_star_moment_closure((tiny, 0), (0, tiny))
    assert near_zero.internal_form_squared == tiny * tiny / (4 - tiny * tiny)
    near_coincident = derive_sine_star_moment_closure((2 - tiny, 0), (0, 1))
    assert near_coincident.internal_form_squared == 1 / (4 - (2 - tiny) ** 2)
    assert near_coincident.internal_form_squared > 10**399
    payload = relational_report_to_dict(near_zero)
    assert payload["report_type"] == "SineStarMomentClosure"
    assert payload["report"]["first_resultant"][0] == {
        "numerator": 1,
        "denominator": 10**400,
    }
    assert near_zero.to_dict()["schema"] == "tnfr.sine-star-moment-closure.v1"
