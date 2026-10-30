"""Mean-cylinder counterexamples retain RN ties and exact interval endpoints."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import (
    NodalRemainderState,
    _validate_nodal_remainder_state,
)
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure
from tnfr.physics.c6_carried_mean_cylinder import (
    derive_c6_carried_mean_cylinder_obstruction,
    observe_c6_carried_mean_cylinder_escape,
)
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

PHASE = tuple(
    map(
        float.fromhex,
        (
            "0x1.0b8fb3e3956cbp-55",
            "0x1.0c152382d7365p+0",
            "0x1.0c152382d7365p+1",
            "0x1.921fb54442d18p+1",
            "0x1.0c152382d7365p+2",
            "0x1.4f1a6c638d03fp+2",
        ),
    )
)
DELTA = F(1, 2**54)
GRID = F(1, 2**3222)
COMMON_CARRY = F(384307168202283423, 2**114)
POSITIVE_OFFSETS = (-4, -1, 2, -2, 8, -5)
NEGATIVE_OFFSETS = (-6, -1, 2, 2, 8, -7)


def _epi(offsets):
    exact = tuple(F(1, 2) + value * DELTA for value in offsets)
    values = tuple(map(float, exact))
    assert tuple(map(F, values)) == exact
    return values


def _mean(state):
    return sum(_validate_nodal_remainder_state(state), F(0)) / 6


def _template(offsets, lower=0.375, upper=0.625):
    return NodalRemainderState(_epi(offsets), (COMMON_CARRY,) * 6, lower, upper)


def _origin(lower=0.375, upper=0.625):
    # Retained B43 actual pre-SHA coordinates. No live word is reexecuted.
    terms = (
        (-35698968670925291, 110),
        (229688483940001907, 113),
        (-203177807149452541, 112),
        (-17846977754793875, 109),
        (-139871603228525, 101),
        (41950776505050555, 111),
    )
    return NodalRemainderState(
        _epi((-3, 0, 2, 0, 8, -5)),
        tuple(F(numerator, 2**power) for numerator, power in terms),
        lower,
        upper,
    )


def _closure(lower=0.375, upper=0.625):
    reference = derive_c6_pressure_lattice(
        phase=PHASE,
        epi_weight=CHANNEL_WEIGHT_SECONDARY,
        phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    return derive_c6_carried_closure(
        derive_c6_carried_profile(reference),
        state=_origin(lower, upper),
        timestep=0.0625,
    )


@pytest.fixture(scope="module")
def obstruction():
    return derive_c6_carried_mean_cylinder_obstruction(
        _closure(),
        positive_state=_template(POSITIVE_OFFSETS),
        negative_state=_template(NEGATIVE_OFFSETS),
    )


def test_canonical_templates_bind_both_pressure_signs_at_the_actual_mean(obstruction):
    origin = obstruction.closure.base_tube.state
    mean = _mean(origin)
    assert mean == F(1, 2) + F(3293, 3 * 2**114)
    assert (
        _mean(obstruction.positive_point.state)
        == _mean(obstruction.negative_point.state)
        == mean
    )
    assert obstruction.positive_mean_increment == F(3, 2**114)
    assert obstruction.negative_mean_increment == -F(1, 2**114)
    assert obstruction.grid_quantum == GRID
    assert 0 < obstruction.positive_point.energy <= obstruction.closure.energy_bound
    assert 0 < obstruction.negative_point.energy <= obstruction.closure.energy_bound
    assert obstruction.invariance_excluded
    assert not obstruction.saved_trajectory_escape_certified
    assert not obstruction.future_runtime_certified
    assert not obstruction.general_correlated_region_excluded
    assert origin == _origin()
    with pytest.raises(FrozenInstanceError):
        obstruction.grid_quantum = F(1)


def test_uniform_translation_window_retains_open_odd_significand_ties(obstruction):
    first = (-DELTA / 2 - COMMON_CARRY) / GRID + 1
    last = (DELTA / 2 - COMMON_CARRY) / GRID - 1
    assert first.denominator == last.denominator == 1
    assert obstruction.translation_grid_lower == int(first)
    assert obstruction.translation_grid_upper == int(last)
    origin_mean = _mean(_origin())
    assert obstruction.mean_lower == origin_mean + first * GRID
    assert obstruction.mean_upper == origin_mean + last * GRID


def _assert_boundary(boundary, lower, upper, expected_direction):
    assert boundary.direction == expected_direction
    assert boundary.mean_before == _mean(boundary.state)
    assert lower <= boundary.mean_before <= upper
    assert boundary.translation == boundary.translation_grid_index * GRID
    exact = _validate_nodal_remainder_state(boundary.state)
    assert boundary.exact_candidate == tuple(
        x + a for x, a in zip(exact, boundary.exact_increment, strict=True)
    )
    assert boundary.mean_after == sum(boundary.exact_candidate, F(0)) / 6
    assert (
        boundary.mean_after - boundary.mean_before
        == sum(boundary.exact_increment, F(0)) / 6
    )
    assert boundary.mean_pressure / 16 == boundary.mean_after - boundary.mean_before
    if expected_direction == "upper":
        assert boundary.mean_after > upper
        assert 0 <= upper - boundary.mean_before < GRID
    else:
        assert boundary.mean_after < lower
        assert 0 <= boundary.mean_before - lower < GRID
    if not boundary.band_failure:
        assert boundary.step.before == boundary.state
        assert boundary.step.after.exact_epi == boundary.exact_candidate
        assert boundary.step.nodal_balance_residual == (F(0),) * 6


@pytest.mark.parametrize(
    "left,right", ((F(0), F(0)), (-F(1, 3), F(1, 3)), (-F(7, 3), F(11, 3)))
)
def test_rational_endpoints_need_not_lie_on_the_uniform_mean_grid(
    obstruction, left, right
):
    origin_mean = _mean(_origin())
    lower, upper = origin_mean + left * GRID, origin_mean + right * GRID
    result = observe_c6_carried_mean_cylinder_escape(
        obstruction, mean_lower=lower, mean_upper=upper
    )
    _assert_boundary(result.upper_boundary, lower, upper, "upper")
    _assert_boundary(result.lower_boundary, lower, upper, "lower")
    assert result.upper_boundary.translation_grid_index == right.__floor__()
    assert result.lower_boundary.translation_grid_index == left.__ceil__()
    assert result.upper_boundary.energy == obstruction.positive_point.energy
    assert result.lower_boundary.energy == obstruction.negative_point.energy
    assert result.invariance_excluded
    assert not result.saved_trajectory_escape_certified
    assert not result.future_runtime_certified
    assert not result.general_correlated_region_excluded
    assert obstruction.closure.base_tube.state == _origin()


def test_extreme_admitted_window_has_actual_legal_witnesses_on_both_sides(obstruction):
    result = observe_c6_carried_mean_cylinder_escape(
        obstruction,
        mean_lower=obstruction.mean_lower,
        mean_upper=obstruction.mean_upper,
    )
    _assert_boundary(
        result.upper_boundary, result.mean_lower, result.mean_upper, "upper"
    )
    _assert_boundary(
        result.lower_boundary, result.mean_lower, result.mean_upper, "lower"
    )
    assert result.upper_boundary.state.remainder == (DELTA / 2 - GRID,) * 6
    assert result.lower_boundary.state.remainder == (-DELTA / 2 + GRID,) * 6


def test_declared_band_clipping_can_supply_a_closed_lower_translation_boundary():
    lower, upper = float(F(1, 2) - 7 * DELTA), float(F(1, 2) + 10 * DELTA)
    result = derive_c6_carried_mean_cylinder_obstruction(
        _closure(lower, upper),
        positive_state=_template(POSITIVE_OFFSETS, lower, upper),
        negative_state=_template(NEGATIVE_OFFSETS, lower, upper),
    )
    assert result.translation_grid_lower == int(-COMMON_CARRY / GRID)
    escape = observe_c6_carried_mean_cylinder_escape(
        result, mean_lower=result.mean_lower, mean_upper=result.mean_upper
    )
    assert escape.lower_boundary.state.remainder == (F(0),) * 6
    assert min(escape.lower_boundary.state.exact_epi) == F(lower)


@pytest.mark.parametrize(
    "positive,negative",
    (
        (NEGATIVE_OFFSETS, POSITIVE_OFFSETS),
        (POSITIVE_OFFSETS, POSITIVE_OFFSETS),
        (NEGATIVE_OFFSETS, NEGATIVE_OFFSETS),
    ),
)
def test_pressure_signs_cannot_be_interchanged_or_inferred_from_labels(
    positive, negative
):
    with pytest.raises(ValueError):
        derive_c6_carried_mean_cylinder_obstruction(
            _closure(),
            positive_state=_template(positive),
            negative_state=_template(negative),
        )


def test_template_mean_must_match_the_actual_origin_without_projection():
    positive = _template(POSITIVE_OFFSETS)
    changed = replace(
        positive, remainder=(positive.remainder[0] + GRID,) + positive.remainder[1:]
    )
    with pytest.raises(ValueError):
        derive_c6_carried_mean_cylinder_obstruction(
            _closure(),
            positive_state=changed,
            negative_state=_template(NEGATIVE_OFFSETS),
        )


def test_forged_derived_obstruction_caches_are_recomputed(obstruction):
    forged = replace(
        obstruction,
        grid_quantum=F(1),
        mean_lower=F(0),
        mean_upper=F(1),
        positive_mean_increment=-F(1),
        negative_mean_increment=F(1),
        translation_grid_lower=-1,
        translation_grid_upper=1,
    )
    mean = _mean(_origin())
    expected = observe_c6_carried_mean_cylinder_escape(
        obstruction, mean_lower=mean, mean_upper=mean
    )
    observed = observe_c6_carried_mean_cylinder_escape(
        forged, mean_lower=mean, mean_upper=mean
    )
    assert observed == expected


def test_primitive_template_tampering_is_not_hidden_by_a_valid_point_cache(obstruction):
    changed = replace(
        obstruction.positive_point, state=obstruction.negative_point.state
    )
    forged = replace(obstruction, positive_point=changed)
    mean = _mean(_origin())
    with pytest.raises(ValueError):
        observe_c6_carried_mean_cylinder_escape(
            forged, mean_lower=mean, mean_upper=mean
        )


@pytest.mark.parametrize("value", (0, 0.5, True, "1/2"))
def test_mean_interval_endpoints_must_be_exact_fractions(obstruction, value):
    with pytest.raises((TypeError, ValueError)):
        observe_c6_carried_mean_cylinder_escape(
            obstruction, mean_lower=value, mean_upper=_mean(_origin())
        )


@pytest.mark.parametrize(
    "case", ("reversed", "miss_origin", "below_window", "above_window")
)
def test_interval_admission_keeps_the_proved_translation_window(obstruction, case):
    mean = _mean(_origin())
    lower, upper = mean, mean
    if case == "reversed":
        lower += GRID
    elif case == "miss_origin":
        lower += GRID
        upper += 2 * GRID
    elif case == "below_window":
        lower = obstruction.mean_lower - GRID
    else:
        upper = obstruction.mean_upper + GRID
    with pytest.raises(ValueError):
        observe_c6_carried_mean_cylinder_escape(
            obstruction, mean_lower=lower, mean_upper=upper
        )
