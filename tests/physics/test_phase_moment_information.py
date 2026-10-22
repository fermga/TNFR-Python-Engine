"""Exact acute moment collisions and static source-information boundaries."""

from fractions import Fraction as Q
from itertools import repeat

import mpmath as mp
import pytest

from tnfr.physics.phase_response import assess_phase_moment_information

LEFT = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),) * 3
RIGHT = ((Q(4, 5), Q(3, 5)),) * 5 + ((Q(4, 5), Q(-3, 5)),)


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def test_strictly_acute_nonzero_first_moment_does_not_determine_cubic_source():
    report = assess_phase_moment_information(LEFT, RIGHT)
    assert report.degree == 6
    assert report.first_resultants == ((Q(24, 5), Q(12, 5)),) * 2
    assert report.first_resultants_equal
    assert report.nonzero_first_resultants == (True, True)
    assert report.strictly_acute == (True, True)
    assert report.sine_source_pi_numerators == (Q(2, 5),) * 2
    assert report.cubic_source_difference_pi_numerator == Q(-14, 125)
    assert report.cubic_source_difference_bounds.hi < 0
    assert report.cosine_storage_difference == 0
    assert report.cubic_storage_difference == Q(-24, 125)
    assert report.sufficiency_obstruction


def test_third_harmonic_identities_retain_source_and_storage_information():
    epsilon = Q(3, 7)
    report = assess_phase_moment_information(LEFT, RIGHT, epsilon=epsilon)
    for index in range(2):
        real, imaginary = report.first_resultants[index]
        real3, imaginary3 = report.third_resultants[index]
        source = ((1 + 3 * epsilon / 4) * imaginary - epsilon * imaginary3 / 4) / 6
        storage = (
            (1 + 2 * epsilon / 3) * 6
            - (1 + 3 * epsilon / 4) * real
            + epsilon * real3 / 12
        )
        assert report.cubic_source_pi_numerators[index] == source
        assert report.cubic_storage_sums[index] == storage
    assert report.cubic_source_difference_pi_numerator == Q(-14, 125) * epsilon


def test_independent_angle_evaluation_matches_the_declared_static_potential_and_source():
    report = assess_phase_moment_information(LEFT, RIGHT, epsilon=Q(2, 3))
    with mp.workdps(90):
        pressures = []
        for index, phasors in enumerate((LEFT, RIGHT)):
            angles = [mp.atan2(_mp(s), _mp(c)) for c, s in phasors]
            pressure = sum(
                mp.sin(delta) + _mp(report.epsilon) * mp.sin(delta) ** 3
                for delta in angles
            ) / (6 * mp.pi)
            potential = sum(
                1
                - mp.cos(delta)
                + _mp(report.epsilon)
                * (mp.mpf(2) / 3 - mp.cos(delta) + mp.cos(delta) ** 3 / 3)
                for delta in angles
            )
            assert abs(
                pressure * mp.pi - _mp(report.cubic_source_pi_numerators[index])
            ) < mp.mpf("1e-85")
            assert abs(potential - _mp(report.cubic_storage_sums[index])) < mp.mpf(
                "1e-85"
            )
            pressures.append(pressure)
        bounds = report.cubic_source_difference_bounds
        assert _mp(bounds.lo) <= pressures[1] - pressures[0] <= _mp(bounds.hi)


def test_equal_potential_does_not_restore_information_lost_by_first_resultant():
    left = ((Q(3, 5), Q(4, 5)),) * 3 + ((Q(4, 5), Q(-3, 5)),) * 4
    right = tuple((c, -s) for c, s in left)
    report = assess_phase_moment_information(left, right)
    assert report.degree == 7
    assert report.first_resultants == ((Q(5), Q(0)),) * 2
    assert report.cubic_storage_difference == report.cosine_storage_difference == 0
    assert report.cubic_source_pi_numerators == (Q(12, 125), Q(-12, 125))
    assert report.sufficiency_obstruction
    assert report.strictly_acute == (True, True)


def test_equal_law_equal_state_and_unequal_information_are_distinct_controls():
    sine = assess_phase_moment_information(LEFT, RIGHT, epsilon=0)
    assert sine.first_resultants_equal
    assert sine.cubic_source_pi_numerators == sine.sine_source_pi_numerators
    assert sine.cubic_storage_sums == sine.cosine_storage_sums
    assert not sine.sufficiency_obstruction
    same = assess_phase_moment_information(LEFT, tuple(reversed(LEFT)))
    assert same.first_resultants_equal
    assert not same.sufficiency_obstruction
    different = assess_phase_moment_information(((1, 0),), ((-1, 0),))
    assert not different.first_resultants_equal
    assert different.cubic_source_difference_pi_numerator == 0
    assert not different.sufficiency_obstruction


@pytest.mark.parametrize("epsilon", (0, 1))
def test_collective_mean_direction_discards_current_information(epsilon):
    """A static K2,2 control; finite examples do not prove the kernel theorem."""
    center = (Q(4, 5), Q(3, 5))
    dispersion = (Q(12, 13), Q(5, 13))
    synchronized = (center, center)
    spread = tuple(
        (
            center[0] * dispersion[0] - sign * center[1] * dispersion[1],
            center[1] * dispersion[0] + sign * center[0] * dispersion[1],
        )
        for sign in (-1, 1)
    )
    report = assess_phase_moment_information(synchronized, spread, epsilon=epsilon)
    left, right = report.first_resultants
    # The resultants have the same direction, but different nonzero lengths.
    assert left[0] * right[1] == left[1] * right[0]
    assert sum(a * b for a, b in zip(left, right)) > 0
    assert sum(value**2 for value in right) < sum(value**2 for value in left)
    assert report.nonzero_first_resultants == report.strictly_acute == (True, True)
    assert not report.first_resultants_equal
    assert not report.sufficiency_obstruction  # Its premise retains length too.

    # Both pairs have zero form and unit held capacity. Evaluate every fine
    # neighbor current independently, in the same pi-scaled clock, before
    # taking either pair's mean rate. No phase row or trajectory is supplied.
    neighbors = ((2, 3), (2, 3), (0, 1), (0, 1))
    root_rates = []
    for phasors in (synchronized, spread):
        phases = ((Q(1), Q(0)),) * 2 + phasors
        rates = []
        for node, row in enumerate(neighbors):
            ci, si = phases[node]
            currents = []
            for other in row:
                cj, sj = phases[other]
                gap_sine = sj * ci - cj * si
                currents.append(gap_sine + epsilon * gap_sine**3)
            rates.append(sum(currents) / len(row))
        root_rates.append((rates[0] + rates[1]) / 2)
        assert (rates[2] + rates[3]) / 2 == -root_rates[-1]
    assert tuple(root_rates) == report.cubic_source_pi_numerators
    assert root_rates[0] != root_rates[1]
    assert root_rates[1] - root_rates[0] == report.cubic_source_difference_pi_numerator


def test_zero_resultant_is_retained_without_an_arg_or_fake_acute_admission():
    report = assess_phase_moment_information(((1, 0), (-1, 0)), ((0, 1), (0, -1)))
    assert report.first_resultants_equal
    assert report.nonzero_first_resultants == (False, False)
    assert report.strictly_acute == (False, False)
    assert not report.sufficiency_obstruction


def test_antipodal_null_control_requires_half_turn_antisymmetry_not_only_oddness():
    left = ((Q(3, 5), Q(4, 5)), (Q(-3, 5), Q(-4, 5)))
    right = ((Q(1), Q(0)), (Q(-1), Q(0)))
    report = assess_phase_moment_information(left, right, epsilon=Q(2, 3))
    assert report.first_resultants == ((0, 0),) * 2
    assert report.sine_source_pi_numerators == (0, 0)
    assert report.cubic_source_pi_numerators == (0, 0)
    assert not report.sufficiency_obstruction

    # The reader's null result concerns its specified odd-harmonic family.
    # The even potential U=sin(delta)^2/2 has unit consensus curvature,
    # but its odd current is pi-periodic rather than pi-antiperiodic.
    exact_currents = tuple(sum(c * s for c, s in row) for row in (left, right))
    assert exact_currents[0] != exact_currents[1]
    with mp.workdps(80):

        def potential(angle):
            return mp.sin(angle) ** 2 / 2

        def current(angle):
            return mp.diff(potential, angle)

        angle = mp.atan2(_mp(left[0][1]), _mp(left[0][0]))
        assert abs(current(angle) + current(-angle)) < mp.mpf("1e-75")
        assert abs(current(angle) - current(angle + mp.pi)) < mp.mpf("1e-75")
        assert mp.diff(current, 0) == 1
        for phasors, exact in zip((left, right), exact_currents):
            total = sum(current(mp.atan2(_mp(s), _mp(c))) for c, s in phasors)
            assert abs(total - _mp(exact)) < mp.mpf("1e-75")


def test_rational_obstruction_survives_below_interval_resolution():
    tiny = Q(1, 10**400)
    report = assess_phase_moment_information(LEFT, RIGHT, epsilon=tiny)
    assert report.epsilon == tiny
    assert report.cubic_source_difference_pi_numerator == Q(-14, 125) * tiny
    assert report.cubic_source_difference_bounds.contains(0)
    assert report.sufficiency_obstruction


def test_balanced_pairs_and_triads_do_not_eliminate_the_fifth_harmonic():
    """Independent potential/current and root-of-unity admission controls."""
    with mp.workdps(90):
        eta = mp.mpf(1) / 7

        def potential(angle):
            return (1 - mp.cos(angle) + eta * (1 - mp.cos(5 * angle)) / 5) / (
                1 + 5 * eta
            )

        def current(angle):
            return mp.diff(potential, angle)

        assert abs(mp.diff(current, 0) - 1) < mp.mpf("1e-85")
        for count in (2, 3):
            phases = [mp.pi / 10 + 2 * mp.pi * k / count for k in range(count)]
            assert abs(sum(mp.exp(1j * angle) for angle in phases)) < mp.mpf("1e-85")
            assert abs(sum(current(angle) for angle in phases)) < mp.mpf("1e-85")
        phases = [mp.pi / 10 + 2 * mp.pi * k / 5 for k in range(5)]
        assert abs(sum(mp.exp(1j * angle) for angle in phases)) < mp.mpf("1e-85")
        total = sum(current(angle) for angle in phases)
        assert abs(total - 5 * eta / (1 + 5 * eta)) < mp.mpf("1e-85")
        assert total > 0

        triad = [mp.pi / 6 + 2 * mp.pi * k / 3 for k in range(3)]
        assert abs(sum(mp.sin(angle) for angle in triad)) < mp.mpf("1e-85")
        assert abs(sum(mp.sin(angle) ** 3 for angle in triad) + mp.mpf(3) / 4) < mp.mpf(
            "1e-85"
        )


def test_one_shot_inputs_and_export_are_detached_from_the_declared_coordinates():
    left = [list(row) for row in LEFT]
    report = assess_phase_moment_information(iter(left), (iter(row) for row in RIGHT))
    left[0][0] = 99
    assert report.left_phasors == LEFT
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.phase-moment-information.v1"
    body = payload["report"]
    assert body["cubic_source_difference_pi_numerator"] == {
        "numerator": -14,
        "denominator": 125,
    }
    body["left_phasors"][0][0]["numerator"] = 12
    assert report.left_phasors[0] == (1, 0)


@pytest.mark.parametrize(
    "phasors",
    [
        (),
        {0: (1, 0)},
        {(1, 0)},
        "1,0",
        ((1,),),
        ((1, 0, 0),),
        ((True, 0),),
        ((float("nan"), 0),),
        ((float("inf"), 0),),
        ((1, 1j),),
        (("1", 0),),
        ((Q(1, 2), Q(1, 2)),),
        ((0.6, 0.8),),
    ],
)
def test_invalid_or_merely_approximately_unit_inputs_do_not_get_coerced(phasors):
    with pytest.raises((TypeError, ValueError)):
        assess_phase_moment_information(phasors, ((1, 0),))


@pytest.mark.parametrize("epsilon", [True, -1, "1", float("nan"), float("inf")])
def test_countermodel_coefficient_has_shared_scalar_admission(epsilon):
    with pytest.raises((TypeError, ValueError)):
        assess_phase_moment_information(LEFT, RIGHT, epsilon=epsilon)


def test_degree_and_materialization_budget_are_explicit():
    with pytest.raises(ValueError, match="equal degree"):
        assess_phase_moment_information(LEFT, RIGHT[:-1])
    with pytest.raises(ValueError):
        assess_phase_moment_information(repeat((1, 0)), ((1, 0),))
    boundary = assess_phase_moment_information(((1, 0),) * 128, ((1.0, 0.0),) * 128)
    assert boundary.degree == 128
    assert boundary.first_resultants == ((128, 0),) * 2
