"""Independent full-field controls for the local/global storage discriminator.

Only static states and analytic bounds are evaluated. No scientific trajectory,
frozen response or alternative runtime is executed by these controls.
"""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tests.physics.test_sine_formation_bridge import _mp
from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES
from tnfr.physics.relational_phase_storage import assess_saddle_storage_discriminator


def _potential(delta):
    """The independently declared smooth alternative, evaluated with mpmath."""
    cutoff = mp.sqrt(2) / 2
    y = (-mp.cos(delta) - cutoff) / (1 - cutoff)
    bump = mp.mpf(0) if y <= 0 else mp.exp(1 - 1 / y)
    return 1 - mp.cos(delta) + 2 * bump


def _energy(form, phase, potential=_potential):
    return mp.fsum(
        (form[j] - form[i]) ** 2 / 2 + potential(phase[j] - phase[i])
        for i, j in CYCLE_LEAF_EDGES
    )


def _rows(form, phase):
    """Complete original-t rows from the independently differentiated U."""
    form_rate, phase_rate = [mp.mpf(0)] * 10, [mp.mpf(0)] * 10
    for i, j in CYCLE_LEAF_EDGES:
        current = mp.diff(_potential, phase[j] - phase[i])
        form_rate[i] += current
        form_rate[j] -= current
        contrast = form[i] - form[j]
        phase_rate[i] += contrast
        phase_rate[j] -= contrast
    return tuple(
        form_rate[i] / ((3 if i < 5 else 1) * mp.pi) for i in range(10)
    ), tuple(phase_rate[i] / ((3 if i < 5 else 1) * mp.pi) for i in range(10))


@pytest.fixture(scope="module")
def report():
    # Deliberately unrelated captured coordinates: the support anchors a fresh
    # declared rational preparation, not a retrospective fit to these values.
    source = _state(
        epi=tuple(Q(i * i, 7) for i in range(10)),
        phase=tuple(Q(i - 4, 9) for i in range(10)),
    )
    return assess_saddle_storage_discriminator(
        source, cycle=range(5), initial_coordinate_radius=Q(1, 2**100)
    )


def test_alternative_normalization_parity_and_local_current_agreement(report):
    with mp.workdps(85):
        tolerance = mp.mpf("1e-78")
        assert _potential(0) == 0
        assert abs(_potential(mp.pi) - _mp(report.antipodal_storage)) < tolerance
        for delta in (mp.mpf(".3"), mp.mpf("1.2"), mp.mpf("2.8")):
            assert abs(_potential(delta) - _potential(-delta)) < tolerance
            assert abs(_potential(delta + 2 * mp.pi) - _potential(delta)) < tolerance
            assert (
                abs(mp.diff(_potential, delta) + mp.diff(_potential, -delta))
                < tolerance
            )
        for delta in (mp.mpf("-.7"), mp.mpf("1.2"), mp.mpf("2.3")):
            assert abs(_potential(delta) - (1 - mp.cos(delta))) < tolerance
            assert abs(mp.diff(_potential, delta) - mp.sin(delta)) < tolerance
        assert _potential(mp.mpf("2.8")) > 1 - mp.cos(mp.mpf("2.8"))


def test_complete_nonlocal_chamber_rows_balance_storage_on_receiver_and_contacts():
    with mp.workdps(90):
        form = tuple(mp.mpf(i * i - 2 * i) / 17 for i in range(10))
        phase = tuple(
            map(
                mp.mpf, ("0", "2.8", ".3", "-.6", ".5", "2.9", "0", ".2", "-.1", "-2.4")
            )
        )
        # Both an internal receiver edge and a private contact activate the
        # alternative: this is not a check entirely in its sine subdomain.
        assert abs(phase[1] - phase[0]) > 3 * mp.pi / 4
        assert abs(phase[5] - phase[0]) > 3 * mp.pi / 4
        form_rate, phase_rate = _rows(form, phase)
        derivative = mp.diff(
            lambda t: _energy(
                tuple(x + t * dx for x, dx in zip(form, form_rate)),
                tuple(
                    theta + t * velocity for theta, velocity in zip(phase, phase_rate)
                ),
            ),
            mp.mpf(0),
        )
        assert abs(derivative) < mp.mpf("1e-78")
        assert abs(
            mp.fsum((3 if i < 5 else 1) * form_rate[i] for i in range(10))
        ) < mp.mpf("1e-78")
        assert abs(
            mp.fsum((3 if i < 5 else 1) * phase_rate[i] for i in range(10))
        ) < mp.mpf("1e-78")
        reversed_form_rate, reversed_phase_rate = _rows(tuple(-x for x in form), phase)
        assert reversed_form_rate == form_rate
        assert reversed_phase_rate == tuple(-value for value in phase_rate)


def test_equal_first_moments_do_not_determine_the_alternative_current():
    with mp.workdps(90):
        first = (5 * mp.pi / 6, -mp.pi / 6, mp.mpf(0), mp.mpf(0))
        second = (mp.pi / 2, -mp.pi / 2, mp.mpf(0), mp.mpf(0))
        first_moment = lambda gaps: mp.fsum(mp.exp(1j * value) for value in gaps)
        tolerance = mp.mpf("1e-80")
        assert abs(first_moment(first) - 2) < tolerance
        assert abs(first_moment(second) - 2) < tolerance
        assert abs(mp.fsum(mp.sin(value) for value in first)) < tolerance
        assert abs(mp.fsum(mp.sin(value) for value in second)) < tolerance
        first_current = mp.fsum(mp.diff(_potential, value) for value in first)
        second_current = mp.fsum(mp.diff(_potential, value) for value in second)
        assert first_current > 1
        assert abs(second_current) < tolerance


def test_rational_preparation_whole_neighborhood_has_shared_energy_and_rows(report):
    assert report.discriminator_certified
    assert report.initial_winding == 1
    assert report.alternative_zero_winding_passage_excluded
    assert report.alternative_winding_invariant_both_time_directions_certified
    assert (
        report.sine_initial_storage_bounds == report.alternative_initial_storage_bounds
    )
    assert report.energy_margin_lower_bound > 0
    source = report.prepared_state
    delta = report.initial_coordinate_radius
    assert len(report.initial_edge_turn_offsets) == len(report.edge_indices) == 10
    assert len(report.initial_cycle_turn_offsets) == 5
    directions = (
        (1,) * 20,
        (-1,) * 20,
        tuple(1 if i % 2 else -1 for i in range(20)),
        (1,) * 10 + (-1,) * 10,
        tuple(1 if i in (0, 6, 12, 18) else -1 for i in range(20)),
    )
    with mp.workdps(110):
        for signs in directions:
            values = tuple(
                _mp(value + sign * delta)
                for value, sign in zip(source.epi + source.phase, signs)
            )
            form, phase = values[:10], values[10:]
            for i, j in CYCLE_LEAF_EDGES:
                gap = mp.atan2(mp.sin(phase[j] - phase[i]), mp.cos(phase[j] - phase[i]))
                assert abs(gap) < 3 * mp.pi / 4
            for (i, j), turns, bound in zip(
                report.edge_indices,
                report.initial_edge_turn_offsets,
                report.initial_edge_principal_gap_bounds,
            ):
                lifted_gap = phase[j] - phase[i] - 2 * mp.pi * turns
                assert _mp(bound.lo) <= lifted_gap <= _mp(bound.hi)
            actual = _energy(form, phase)
            baseline = _energy(form, phase, lambda gap: 1 - mp.cos(gap))
            assert abs(actual - baseline) < mp.mpf("1e-98")
            assert (
                _mp(report.alternative_initial_storage_bounds.lo)
                <= actual
                <= _mp(report.alternative_initial_storage_bounds.hi)
            )
            form_rate, _ = _rows(form, phase)
            for i in range(10):
                neighbors = tuple(
                    j if i == a else a for a, j in CYCLE_LEAF_EDGES if i in (a, j)
                )
                expected = mp.fsum(mp.sin(phase[j] - phase[i]) for j in neighbors) / (
                    len(neighbors) * mp.pi
                )
                assert abs(form_rate[i] - expected) < mp.mpf("1e-98")


def test_whole_local_agreement_accounts_for_independent_source_width(report):
    with mp.workdps(85):
        # Lipschitz2 in scaled time gives exp(6)<729. The proof covers every
        # box member; the finite corners above are only independent controls.
        initial_radius = (
            3 * _mp(report.epsilon)
            + _mp(report.preparation.preparation_error_bound)
            + _mp(report.initial_coordinate_radius)
        )
        analytic_phase_radius = mp.exp(6) * initial_radius
        assert analytic_phase_radius < _mp(report.local_phase_radius_bound)
        actual_margin = mp.pi / 12 - 2 * _mp(report.local_phase_radius_bound)
        assert actual_margin >= _mp(report.local_agreement_margin_lower_bound) > 0
    assert report.local_scaled_duration == 3
    assert report.whole_local_flow_agreement_certified
