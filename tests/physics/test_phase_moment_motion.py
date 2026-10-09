"""Current, work and lost phase/rate association from complete nodal controls."""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.physics.phase_response import (
    assess_phase_moment_information,
    derive_phase_moment_motion,
)
from tnfr.sdk import relational_report_to_dict

PHASES = ((Q(3, 5), Q(4, 5)), (Q(4, 5), Q(3, 5)))


def _star_gap_rates(forms):
    neighbors = ((1, 2), (0,), (0,))
    phase_rates = tuple(
        forms[i] - sum((forms[j] for j in row), Q(0)) / len(row)
        for i, row in enumerate(neighbors)
    )
    return tuple(phase_rates[j] - phase_rates[0] for j in neighbors[0])


@pytest.mark.parametrize("epsilon", (Q(0), Q(1, 100), Q(2, 3)))
def test_full_star_distinguishes_pairing_from_separate_phase_and_form_inventories(
    epsilon,
):
    forms = ((Q(0), Q(3, 4), Q(-1, 4)), (Q(0), Q(-1, 4), Q(3, 4)))
    reports = [
        derive_phase_moment_motion(PHASES, _star_gap_rates(x), epsilon=epsilon)
        for x in forms
    ]
    left, right = reports
    assert sorted(forms[0]) == sorted(forms[1])
    assert sorted(left.gap_rates) == sorted(right.gap_rates)
    assert left.first_resultant == right.first_resultant
    assert left.third_resultant == right.third_resultant
    assert left.cubic_source_pi_numerator == right.cubic_source_pi_numerator
    assert left.cubic_storage_sum == right.cubic_storage_sum
    assert left.cubic_source_rate_pi_numerator != right.cubic_source_rate_pi_numerator
    assert left.cubic_storage_rate != right.cubic_storage_rate
    for report in reports:
        # The full unit-capacity star in tau=t/pi has theta'=K L x.
        # Differentiating its root current gives x_root'' in that clock.
        acceleration = (
            sum(
                c * (1 + 3 * epsilon * s**2) * rate
                for (c, s), rate in zip(PHASES, report.gap_rates)
            )
            / 2
        )
        work = sum(
            (s + epsilon * s**3) * rate
            for (_, s), rate in zip(PHASES, report.gap_rates)
        )
        assert report.cubic_source_rate_pi_numerator == acceleration
        assert report.cubic_storage_rate == work
    assert left.sine_source_rate_pi_numerator == Q(3, 10)
    assert right.sine_source_rate_pi_numerator == Q(2, 5)


def test_independent_angular_differentiation_and_static_owner_agree():
    phasors = PHASES + ((Q(-5, 13), Q(12, 13)),)
    rates = (Q(1, 7), Q(-3, 5), Q(9, 4))
    epsilon = Q(2, 3)
    report = derive_phase_moment_motion(phasors, rates, epsilon=epsilon)
    static = assess_phase_moment_information(phasors, phasors, epsilon=epsilon)
    assert report.cubic_source_pi_numerator == static.cubic_source_pi_numerators[0]
    assert report.cubic_storage_sum == static.cubic_storage_sums[0]
    with mp.workdps(80):

        def real(value):
            return mp.mpf(value.numerator) / value.denominator

        angles = [mp.atan2(real(s), real(c)) for c, s in phasors]

        def current(t):
            gaps = [angle + t * real(rate) for angle, rate in zip(angles, rates)]
            return sum(mp.sin(g) + real(epsilon) * mp.sin(g) ** 3 for g in gaps) / 3

        def storage(t):
            gaps = [angle + t * real(rate) for angle, rate in zip(angles, rates)]
            return sum(
                1
                - mp.cos(g)
                + real(epsilon) * (mp.mpf(2) / 3 - mp.cos(g) + mp.cos(g) ** 3 / 3)
                for g in gaps
            )

        assert abs(
            mp.diff(current, 0) - real(report.cubic_source_rate_pi_numerator)
        ) < mp.mpf("1e-70")
        assert abs(mp.diff(storage, 0) - real(report.cubic_storage_rate)) < mp.mpf(
            "1e-70"
        )


def test_joint_reordering_and_clock_scaling_preserve_the_declared_relations():
    base = derive_phase_moment_motion(PHASES, (Q(2), Q(-3)))
    reordered = derive_phase_moment_motion(iter(reversed(PHASES)), iter((Q(-3), Q(2))))
    assert reordered.first_rate_weighted_resultant == base.first_rate_weighted_resultant
    assert reordered.cubic_storage_rate == base.cubic_storage_rate
    faster = derive_phase_moment_motion(PHASES, (Q(6), Q(-9)))
    assert faster.first_resultant == base.first_resultant
    assert (
        faster.cubic_source_rate_pi_numerator == 3 * base.cubic_source_rate_pi_numerator
    )
    assert faster.cubic_storage_rate == 3 * base.cubic_storage_rate


def test_zero_resultant_and_unresolvably_small_rates_are_not_erased():
    tiny = Q(1, 10**400)
    report = derive_phase_moment_motion(((1, 0), (-1, 0)), (tiny, 0), epsilon=0)
    assert report.first_resultant == (0, 0)
    assert report.first_resultant_rate == (0, tiny)
    assert report.sine_source_rate_pi_numerator == tiny / 2
    assert report.cosine_storage_rate == 0


@pytest.mark.parametrize(
    "phasors,rates,epsilon",
    (
        (PHASES, (True, 0), 0),
        (PHASES, (float("nan"), 0), 0),
        (PHASES, (float("inf"), 0), 0),
        (PHASES, (0,), 0),
        (PHASES, (0, 1, 2), 0),
        (PHASES, {0, 1}, 0),
        (PHASES, (0, 1), True),
        (PHASES, (0, 1), -1),
        (((True, 0),), (0,), 0),
        (((0.6, 0.8),), (0,), 0),
        ((), (), 0),
    ),
)
def test_invalid_or_unassociated_inputs_are_rejected(phasors, rates, epsilon):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        derive_phase_moment_motion(phasors, rates, epsilon=epsilon)


def test_sdk_retains_exact_paired_rates_and_derived_scope():
    report = derive_phase_moment_motion(PHASES, (Q(1), Q(0)))
    payload = relational_report_to_dict(report)
    assert payload["report_type"] == "PhaseMomentMotion"
    assert payload["report"]["gap_rates"] == [
        {"numerator": 1, "denominator": 1},
        {"numerator": 0, "denominator": 1},
    ]
    assert (
        "rate_weighted_moments_are_derived_observables_not_independent_state_or_new_laws"
        in payload["report"]["scope"]
    )
