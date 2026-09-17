"""Exact periodic phase areas and necessary signed nonphase compensation."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import pytest

from tnfr.physics.nodal_remainder_pressure import (
    derive_periodic_phase_source_budget, observe_periodic_phase_source_compensation,
)


def _budget(rows=None, duration=.25):
    return derive_periodic_phase_source_budget(
        phase_contributions=rows if rows is not None else ((.125,) * 6, (-.25,) * 6, (.375,) * 6),
        block_duration=duration,
    )


def _observe(reference=None, **changes):
    arguments = {"block_count": 1, "initial_mean": F(1, 2), "actual_nonphase_area": F(0)}
    arguments.update(changes)
    return observe_periodic_phase_source_compensation(reference or _budget(), **arguments)


def test_hand_period_three_mean_centered_prefix_and_amplitude():
    result = _budget()
    assert result.period == 3
    assert result.mean_sources == (F(1, 8), -F(1, 4), F(3, 8))
    assert result.mean_source == F(1, 12)
    assert result.centered_prefixes == (F(0), F(1, 24), -F(7, 24), F(0))
    assert result.phase_area_per_period == F(1, 16)
    assert result.prefix_amplitude_bound == F(7, 96)
    assert not result.phase_cycle_certified


def test_row_means_use_all_six_exact_represented_coordinates():
    source = ((.5, -.25, .125, 0., 0., 0.), (.1, .2, .3, -.1, -.2, -.3))
    result = _budget(source, .1)
    assert result.mean_sources == (F(1, 16), F(0))
    assert result.mean_source == F(1, 32)
    assert result.phase_area_per_period == F(.1) / 16
    assert result.centered_prefixes == (0, F(1, 32), 0)


@pytest.mark.parametrize("count", range(10))
def test_arbitrary_prefix_equals_independent_finite_sum(count):
    budget = _budget()
    result = _observe(budget, block_count=count)
    expected = sum((F((.125, -.25, .375)[i % 3]) / 4 for i in range(count)), F(0))
    assert result.phase_area == expected
    assert result.phase_area == result.linear_phase_area + result.periodic_phase_area
    assert result.complete_periods == count // 3
    assert result.remainder_blocks == count % 3
    assert abs(result.periodic_phase_area) <= budget.prefix_amplitude_bound
    assert result.reconstructed_mean_change == expected
    assert result.reconstructed_mean == F(1, 2) + expected


def test_huge_count_uses_exact_period_wrap_without_float_conversion():
    count = 10**120 + 1
    result = _observe(block_count=count, actual_nonphase_area=-F(count, 48))
    assert result.remainder_blocks == 2
    assert result.complete_periods == (count - 2) // 3
    assert result.linear_phase_area == F(count, 48)
    assert result.periodic_phase_area == -F(7, 96)
    assert result.phase_area == F(count, 48) - F(7, 96)
    assert result.reconstructed_mean == F(41, 96)
    assert result.mean_in_band
    assert not result.future_compensation_certified


def test_zero_average_has_bounded_phase_prefix_but_does_not_certify_nonphase_budget():
    budget = _budget(((.125,) * 6, (-.125,) * 6))
    assert budget.mean_source == budget.phase_area_per_period == 0
    assert budget.prefix_amplitude_bound == F(1, 32)
    for count in (10**100, 10**100 + 1):
        result = _observe(budget, block_count=count)
        assert result.phase_area == (F(1, 32) if count % 2 else 0)
        assert result.mean_in_band
    unbalanced = _observe(budget, block_count=10**100, actual_nonphase_area=F(1))
    assert not unbalanced.mean_in_band
    assert not unbalanced.future_compensation_certified


def test_negative_period_one_bias_requires_positive_linear_compensation():
    budget = _budget(((-.25,) * 6,))
    assert budget.centered_prefixes == (0, 0)
    assert budget.prefix_amplitude_bound == 0
    assert budget.mean_source == -F(1, 4)
    assert budget.phase_area_per_period == -F(1, 16)
    uncompensated = _observe(budget, block_count=8)
    assert uncompensated.reconstructed_mean == 0
    assert not uncompensated.mean_in_band
    count = 10**90
    compensated = _observe(budget, block_count=count, actual_nonphase_area=F(count, 16))
    assert compensated.phase_area == -F(count, 16)
    assert compensated.reconstructed_mean == F(1, 2)
    assert compensated.mean_in_band
    assert not compensated.future_compensation_certified


def test_nonzero_bias_is_not_hidden_by_period_centering():
    budget = _budget(((.125,) * 6,))
    result = _observe(budget, block_count=17)
    assert result.periodic_phase_area == 0
    assert result.phase_area == F(17, 32)
    assert result.reconstructed_mean == F(33, 32)
    assert not result.mean_in_band


def test_required_compensation_interval_has_correct_sign_and_inclusive_boundaries():
    initial = _observe(block_count=1001, actual_nonphase_area=-F(20))
    assert initial.required_compensation_lower == F(.05) - F(1, 2) - initial.phase_area
    assert initial.required_compensation_upper == F(1, 2) - initial.phase_area
    for compensation, endpoint in ((initial.required_compensation_lower, F(.05)),
                                   (initial.required_compensation_upper, F(1))):
        result = _observe(block_count=1001, actual_nonphase_area=compensation)
        assert result.mean_in_band
        assert result.reconstructed_mean == endpoint
    for compensation in (initial.required_compensation_lower - F(1, 2**100),
                         initial.required_compensation_upper + F(1, 2**100)):
        assert not _observe(block_count=1001, actual_nonphase_area=compensation).mean_in_band


def test_zero_blocks_have_zero_source_and_zero_compensation():
    result = _observe(block_count=0)
    assert result.phase_area == result.reconstructed_mean_change == 0
    assert result.reconstructed_mean == F(1, 2)
    assert result.mean_in_band
    with pytest.raises(ValueError, match="zero blocks"):
        _observe(block_count=0, actual_nonphase_area=F(1))


def test_mean_membership_is_only_necessary_for_six_coordinate_trapping():
    result = _observe(_budget(((0.,) * 6,)))
    compatible_mean_example = (-F(1, 4),) + (F(13, 20),) * 5
    assert sum(compatible_mean_example) / 6 == result.reconstructed_mean
    assert result.mean_in_band
    assert min(compatible_mean_example) < F(.05)
    assert not result.future_compensation_certified


def test_forged_derived_caches_are_ignored_and_source_fields_rebuilt():
    reference = _budget()
    forged = replace(reference, period=999, mean_sources=(), mean_source=F(0),
                     centered_prefixes=(), phase_area_per_period=F(0), prefix_amplitude_bound=F(0))
    result = _observe(forged, block_count=4)
    assert result.reference == reference
    assert result.phase_area == F(3, 32)
    with pytest.raises(TypeError):
        _observe(replace(reference, phase_contributions=((F(0),) * 6,)))
    with pytest.raises(ValueError):
        _observe(replace(reference, block_duration=0.))


@pytest.mark.parametrize("rows,error", (
    ([], TypeError), ((), ValueError), (([],), TypeError), (((),), ValueError),
    (((0.,) * 5,), ValueError), (((0.,) * 7,), ValueError),
    (((0,) * 6,), TypeError), (((True,) * 6,), TypeError),
    (((F(0),) * 6,), TypeError), (((math.nan,) * 6,), ValueError),
    (((math.inf,) * 6,), ValueError),
))
def test_invalid_source_rows_are_rejected(rows, error):
    with pytest.raises(error):
        _budget(rows)


@pytest.mark.parametrize("duration,error", (
    (0., ValueError), (-.25, ValueError), (math.nan, ValueError), (math.inf, ValueError),
    (1, TypeError), (True, TypeError), (F(1, 4), TypeError),
))
def test_invalid_duration_is_rejected(duration, error):
    with pytest.raises(error):
        _budget(duration=duration)


@pytest.mark.parametrize("changes,error", (
    ({"block_count": -1}, ValueError), ({"block_count": True}, ValueError),
    ({"block_count": 1.}, ValueError), ({"initial_mean": .5}, TypeError),
    ({"initial_mean": F(0)}, ValueError), ({"initial_mean": F(2)}, ValueError),
    ({"actual_nonphase_area": 0.}, TypeError), ({"epi_lower": 0.}, ValueError),
    ({"epi_lower": 1}, TypeError), ({"epi_upper": 1.1}, ValueError),
    ({"epi_lower": .6, "epi_upper": .5}, ValueError),
))
def test_invalid_compensation_inputs_are_rejected(changes, error):
    with pytest.raises(error):
        _observe(**changes)


def test_results_are_frozen():
    budget = _budget()
    result = _observe(budget)
    with pytest.raises(FrozenInstanceError):
        budget.mean_source = F(0)
    with pytest.raises(FrozenInstanceError):
        result.mean_in_band = False
