"""Static counterexamples for a form-only charge observation on joined support.

These tests evaluate the complete quarter law, not the degree-normalized sine
comparison or a trajectory. They check the implemented observation obstruction;
they neither classify every conserved observable nor identify physical charge.
"""

from fractions import Fraction as Q
from math import pi

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics._organized_slot_registration import (
    _observe_organized_slot_registration,
)

_ZERO = (Q(0),) * 54
_SLOTS = (0,) * 5
_GAMMA = 1 / (1023 * pi)
# Only endpoint display comparisons use binary floats. This absolute allowance
# covers the represented pi/2 preparation and conversion of rational enclosures.
_FLOAT_ABSOLUTE_TOLERANCE = 1e-15
# Enclosure comparisons allow the default outward dyadic lattice's rounding;
# overlapping certified boxes need not have identical endpoints.
_ENCLOSURE_ALLOWANCE = Q(1, 10**30)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


def _observe(beta, *, form=_ZERO, phase_offsets=_ZERO):
    return _observe_organized_slot_registration(
        slot_offsets=_SLOTS,
        relative_phase_origin=beta,
        form=form,
        phase_offsets=phase_offsets,
    ).interaction


@pytest.fixture(scope="module")
def preparations():
    return _observe(0), _observe(pi / 2)


def _assert_scalar_enclosure_near(interval, expected):
    for endpoint in (interval.lo, interval.hi):
        assert float(endpoint) == pytest.approx(
            expected, rel=0, abs=_FLOAT_ABSOLUTE_TOLERANCE
        )


def _center(values):
    mean = sum(values, Q(0)) / len(values)
    return tuple(value - mean for value in values)


def test_same_complete_centered_form_has_different_current_and_form_derivative(
    preparations,
):
    aligned, quarter_turn = preparations
    assert aligned.geometry.weighted_edges == quarter_turn.geometry.weighted_edges
    assert (
        aligned.geometry.clock_factor == quarter_turn.geometry.clock_factor == Q(1, 4)
    )
    assert aligned.geometry.held_capacities == (Q(1),) * 54
    assert aligned.geometry.support_degrees == (5,) * 45 + (7,) * 9
    assert _center(aligned.form) == _center(quarter_turn.form) == _ZERO

    # Internal C9 currents cancel. Each host node then receives one contact
    # contribution sin(beta); each donor receives five opposite contributions.
    for node in range(54):
        expected_current = 1 if node < 45 else -5
        _assert_scalar_enclosure_near(aligned.phase_current_bounds[node], 0)
        _assert_scalar_enclosure_near(aligned.form_tau_rate_bounds[node], 0)
        _assert_scalar_enclosure_near(
            quarter_turn.phase_current_bounds[node], expected_current
        )
        _assert_scalar_enclosure_near(
            quarter_turn.form_tau_rate_bounds[node], expected_current * _GAMMA / 4
        )
        assert (
            quarter_turn.form_tau_rate_bounds[node].lo
            > aligned.form_tau_rate_bounds[node].hi
            if node < 45
            else quarter_turn.form_tau_rate_bounds[node].hi
            < aligned.form_tau_rate_bounds[node].lo
        )


@pytest.mark.parametrize("preparation_index", (0, 1))
def test_hidden_phase_changes_transfer_without_creating_total_form(
    preparations, preparation_index
):
    state = preparations[preparation_index]
    assert state.form_gradient == _ZERO
    assert state.phase_tau_rate_bounds == (I(0),) * 54
    total_rate = sum(state.form_tau_rate_bounds, I(0))
    assert total_rate.contains(0)
    assert total_rate.width < _ENCLOSURE_ALLOWANCE

    # Check each nine-slot boundary reaction before summing the whole graph.
    for slot in range(9):
        reaction = sum(
            (
                state.contact_form_tau_rate_bounds[9 * child + slot]
                for child in range(5)
            ),
            state.contact_form_tau_rate_bounds[45 + slot],
        )
        assert reaction.contains(0)
        assert reaction.width < _ENCLOSURE_ALLOWANCE
    if preparation_index:
        assert all(box.lo > 0 for box in state.form_tau_rate_bounds[:45])
        assert all(box.hi < 0 for box in state.form_tau_rate_bounds[45:])


@pytest.mark.parametrize("beta", (0, pi / 2))
def test_common_finite_phase_shift_preserves_the_complete_observed_field(beta):
    base = _observe(beta)
    shift = Q(11, 7)
    shifted = _observe(beta, phase_offsets=(shift,) * 54)
    for original, translated in zip(base.phase_lift_bounds, shifted.phase_lift_bounds):
        assert translated == original + shift
    for name in (
        "phase_current_bounds",
        "form_tau_rate_bounds",
        "phase_tau_rate_bounds",
        "contact_phase_current_bounds",
        "contact_form_tau_rate_bounds",
        "contact_phase_tau_rate_bounds",
    ):
        for original, translated in zip(getattr(base, name), getattr(shifted, name)):
            difference = translated - original
            assert difference.contains(0)
            assert difference.width < _ENCLOSURE_ALLOWANCE


def test_form_origin_is_not_recovered_from_relative_fields(preparations):
    _, base = preparations
    shift = Q(7, 3)
    shifted = _observe(pi / 2, form=(shift,) * 54)
    assert _center(shifted.form) == _center(base.form)
    assert sum(shifted.form) - sum(base.form) == 54 * shift
    for name in (
        "form_gradient",
        "phase_current_bounds",
        "form_tau_rate_bounds",
        "phase_tau_rate_bounds",
        "contact_form_tau_rate_bounds",
        "contact_phase_tau_rate_bounds",
        "storage_tau_rate",
    ):
        assert getattr(shifted, name) == getattr(base, name)
