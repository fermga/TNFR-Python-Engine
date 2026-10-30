"""Affine carry fibers do not repair the independent mean-cylinder candidate."""

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import (
    NodalRemainderState,
    _validate_nodal_remainder_state,
)
from tnfr.physics.c6_carried_affine_mean import (
    derive_c6_carried_affine_mean_obstruction,
    observe_c6_carried_affine_mean_escape,
)
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

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
DELTA, M = F(1, 2**54), F(1, 2**113)
COMMON_CARRY = F(384307168202283423, 2**114)
POSITIVE_OFFSETS = (-4, -1, 2, -2, 8, -5)
NEGATIVE_OFFSETS = (-6, -1, 2, 2, 8, -7)


def _epi(offsets):
    exact = tuple(F(1, 2) + value * DELTA for value in offsets)
    result = tuple(map(float, exact))
    assert tuple(map(F, result)) == exact
    return result


def _case(origin_shift=F(0)):
    # The captured B43 pre-SHA state is used as data, not reexecuted.
    terms = (
        (-35698968670925291, 110),
        (229688483940001907, 113),
        (-203177807149452541, 112),
        (-17846977754793875, 109),
        (-139871603228525, 101),
        (41950776505050555, 111),
    )
    carry = tuple(F(numerator, 2**power) for numerator, power in terms)
    origin = NodalRemainderState(
        _epi((-3, 0, 2, 0, 8, -5)),
        tuple(value + origin_shift for value in carry),
        0.375,
        0.625,
    )
    template_carry = (COMMON_CARRY + origin_shift,) * 6
    positive = NodalRemainderState(_epi(POSITIVE_OFFSETS), template_carry, 0.375, 0.625)
    negative = NodalRemainderState(_epi(NEGATIVE_OFFSETS), template_carry, 0.375, 0.625)
    lattice = derive_c6_pressure_lattice(
        phase=PHASE,
        epi_weight=CHANNEL_WEIGHT_SECONDARY,
        phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    closure = derive_c6_carried_closure(
        derive_c6_carried_profile(lattice), state=origin, timestep=0.0625
    )
    return closure, positive, negative


@pytest.fixture(scope="module")
def case():
    return _case()


@pytest.fixture(scope="module")
def obstruction(case):
    closure, positive, negative = case
    return derive_c6_carried_affine_mean_obstruction(
        closure, positive_state=positive, negative_state=negative
    )


def _mean(state):
    return sum(_validate_nodal_remainder_state(state), F(0)) / 6


def test_actual_affine_spacing_and_energy_bounds_are_exact(obstruction, case):
    closure, _positive, _negative = case
    base = obstruction.base_obstruction
    assert obstruction.coordinate_spacings == tuple(
        value * M for value in (1, 1, 2, 4, 8, 4)
    )
    assert obstruction.coordinate_residues == (F(0),) * 6
    assert obstruction.global_spacing == M and obstruction.pivot == 0
    assert obstruction.mean_quantum == M / 6
    assert obstruction.gradient_value_count == 336
    assert obstruction.carry_lower_slack == 8 * M
    assert obstruction.carry_upper_slack == 19 * M
    assert obstruction.lift_error_squared_bound == 462 * M**2
    assert (
        obstruction.positive_energy_bound == 2 * base.positive_point.energy + 924 * M**2
    )
    assert (
        obstruction.negative_energy_bound == 2 * base.negative_point.energy + 924 * M**2
    )
    assert (
        max(obstruction.positive_energy_bound, obstruction.negative_energy_bound)
        < closure.energy_bound
    )
    assert obstruction.mean_lower == base.mean_lower + 8 * M
    assert obstruction.mean_upper == base.mean_upper - 19 * M
    assert (
        obstruction.mean_lower < _mean(closure.base_tube.state) < obstruction.mean_upper
    )
    assert base.positive_mean_increment == 9 * obstruction.mean_quantum
    assert base.negative_mean_increment == -3 * obstruction.mean_quantum
    assert obstruction.invariance_excluded
    assert not obstruction.saved_trajectory_escape_certified
    assert not obstruction.future_runtime_certified
    assert not obstruction.general_correlated_region_excluded
    with pytest.raises(FrozenInstanceError):
        obstruction.global_spacing = F(1)


def test_each_gcd_is_verified_by_canonical_pressure_and_independent_bezout(obstruction):
    closure = obstruction.base_obstruction.closure
    lattice = closure.base_tube.contraction.profile.lattice
    # Moving one neighbor in a region strictly below .5 realizes every
    # tested integer gradient exactly; the full CPU owner supplies pressure.
    base = F(1, 2) - 64 * DELTA
    values = []
    for node, (lo, hi, spacing) in enumerate(
        zip(
            closure.gradient_index_lower,
            closure.gradient_index_upper,
            obstruction.coordinate_spacings,
        )
    ):
        entries = {}
        for index in range(lo, hi + 1):
            epi = [float(base)] * 6
            epi[(node - 1) % 6] = float(base + index * DELTA)
            observed = _observe_rebuilt_c6_pressure_lattice(lattice, tuple(epi))
            assert observed.gradient_indices[node] == index
            entries[index] = F(observed.pressure[node]) / 16
        quotients = tuple(value / spacing for value in entries.values())
        assert all(value.denominator == 1 for value in quotients)
        assert math.gcd(*(value.numerator for value in quotients)) == 1
        values.append(entries)
    witnesses = (
        (1, 2, -25629072196961, -300136655591087),
        (2, 21, -4379842469181, 542998862359),
        (-5, -2, -490067140128037, 40909233934201),
        (5, 16, 133733011476797, 136012562140503),
        (-43, -7, -9559246482503, -14859535507807),
        (5, 16, -48528880221559, -329345734683863),
    )
    for entries, spacing, (first, second, a, b) in zip(
        values, obstruction.coordinate_spacings, witnesses
    ):
        assert a * entries[first] + b * entries[second] == spacing


def _assert_boundary(bound, point, boundary, lower, upper, direction):
    exact = _validate_nodal_remainder_state(boundary.state)
    source = _validate_nodal_remainder_state(point.state)
    assert boundary.state.epi == point.state.epi
    assert boundary.mean_before == _mean(boundary.state)
    assert lower <= boundary.mean_before <= upper
    assert boundary.mean_translation == boundary.mean_grid_index * bound.mean_quantum
    assert boundary.carry_adjustment == tuple(x - y for x, y in zip(exact, source))
    errors = tuple(
        value - boundary.mean_translation for value in boundary.carry_adjustment
    )
    assert sum(errors, F(0)) == 0
    assert sum(value**2 for value in errors) <= bound.lift_error_squared_bound
    assert 0 <= errors[bound.pivot] < bound.carry_upper_slack
    for index, (adjustment, spacing) in enumerate(
        zip(errors, bound.coordinate_spacings)
    ):
        if index != bound.pivot:
            assert -spacing < adjustment <= 0
    for x, residue, spacing in zip(
        exact, bound.coordinate_residues, bound.coordinate_spacings
    ):
        assert ((x - residue) / spacing).denominator == 1
    assert boundary.mean_after == sum(boundary.exact_candidate, F(0)) / 6
    assert boundary.mean_after - boundary.mean_before == boundary.mean_pressure / 16
    assert boundary.exact_increment == tuple(
        F(p) / 16 for p in point.observation.pressure
    )
    assert boundary.exact_candidate == tuple(
        x + a for x, a in zip(exact, boundary.exact_increment)
    )
    for x, residue, spacing in zip(
        boundary.exact_candidate, bound.coordinate_residues, bound.coordinate_spacings
    ):
        assert ((x - residue) / spacing).denominator == 1
    if direction == "upper":
        assert 0 <= upper - boundary.mean_before < bound.mean_quantum
        assert (
            boundary.mean_after > upper
            and boundary.energy <= bound.positive_energy_bound
        )
    else:
        assert 0 <= boundary.mean_before - lower < bound.mean_quantum
        assert (
            boundary.mean_after < lower
            and boundary.energy <= bound.negative_energy_bound
        )
    if not boundary.band_failure:
        assert boundary.step.before == boundary.state
        assert (
            _validate_nodal_remainder_state(boundary.step.after)
            == boundary.exact_candidate
        )
        assert boundary.step.nodal_balance_residual == (F(0),) * 6


def test_all_48_lift_residue_classes_for_both_pressure_signs(obstruction):
    base = obstruction.base_obstruction
    mean, quantum = _mean(base.closure.base_tube.state), obstruction.mean_quantum
    adjustments = []
    for index in range(48):
        lower, upper = mean - index * quantum, mean + index * quantum
        result = observe_c6_carried_affine_mean_escape(
            obstruction, mean_lower=lower, mean_upper=upper
        )
        _assert_boundary(
            obstruction,
            base.positive_point,
            result.upper_boundary,
            lower,
            upper,
            "upper",
        )
        _assert_boundary(
            obstruction,
            base.negative_point,
            result.lower_boundary,
            lower,
            upper,
            "lower",
        )
        assert result.upper_boundary.mean_grid_index == index
        assert result.lower_boundary.mean_grid_index == -index
        adjustments.append(
            tuple(
                value - result.upper_boundary.mean_translation
                for value in result.upper_boundary.carry_adjustment
            )
        )
    assert len(set(adjustments)) == 48
    period = observe_c6_carried_affine_mean_escape(
        obstruction, mean_lower=mean, mean_upper=mean + 48 * quantum
    )
    assert (
        tuple(
            value - period.upper_boundary.mean_translation
            for value in period.upper_boundary.carry_adjustment
        )
        == adjustments[0]
    )


def test_nondyadic_interval_endpoints_keep_strict_crossings(obstruction):
    base = obstruction.base_obstruction
    mean, quantum = _mean(base.closure.base_tube.state), obstruction.mean_quantum
    lower, upper = mean - F(7, 3) * quantum, mean + F(11, 3) * quantum
    result = observe_c6_carried_affine_mean_escape(
        obstruction, mean_lower=lower, mean_upper=upper
    )
    _assert_boundary(
        obstruction, base.positive_point, result.upper_boundary, lower, upper, "upper"
    )
    _assert_boundary(
        obstruction, base.negative_point, result.lower_boundary, lower, upper, "lower"
    )
    assert (
        result.upper_boundary.mean_grid_index == 3
        and result.lower_boundary.mean_grid_index == -2
    )
    assert result.invariance_excluded
    assert not result.saved_trajectory_escape_certified
    assert not result.future_runtime_certified
    assert not result.general_correlated_region_excluded


def test_shrunk_window_endpoints_remain_legal_after_nonuniform_lifting(obstruction):
    result = observe_c6_carried_affine_mean_escape(
        obstruction,
        mean_lower=obstruction.mean_lower,
        mean_upper=obstruction.mean_upper,
    )
    base = obstruction.base_obstruction
    _assert_boundary(
        obstruction,
        base.positive_point,
        result.upper_boundary,
        result.mean_lower,
        result.mean_upper,
        "upper",
    )
    _assert_boundary(
        obstruction,
        base.negative_point,
        result.lower_boundary,
        result.mean_lower,
        result.mean_upper,
        "lower",
    )


def test_nonzero_origin_residue_is_carried_through_both_witnesses():
    closure, positive, negative = _case(M / 2)
    bound = derive_c6_carried_affine_mean_obstruction(
        closure, positive_state=positive, negative_state=negative
    )
    assert bound.coordinate_residues == (M / 2,) * 6
    mean = _mean(closure.base_tube.state)
    result = observe_c6_carried_affine_mean_escape(
        bound, mean_lower=mean, mean_upper=mean
    )
    _assert_boundary(
        bound,
        bound.base_obstruction.positive_point,
        result.upper_boundary,
        mean,
        mean,
        "upper",
    )
    _assert_boundary(
        bound,
        bound.base_obstruction.negative_point,
        result.lower_boundary,
        mean,
        mean,
        "lower",
    )
    assert closure.base_tube.state.remainder[0] == F(-35698968670925291, 2**110) + M / 2


def test_nonuniform_template_carries_are_outside_the_declared_lift_contract(case):
    closure, positive, negative = case

    def skew(state):
        return replace(
            state,
            remainder=(state.remainder[0] - M / 4, state.remainder[1] + M / 4)
            + state.remainder[2:],
        )

    with pytest.raises(ValueError, match="uniform carry"):
        derive_c6_carried_affine_mean_obstruction(
            closure,
            positive_state=skew(positive),
            negative_state=skew(negative),
        )
    assert closure.base_tube.state == case[0].base_tube.state


def test_gradient_budget_is_a_complete_enumeration_limit_not_partial_evidence(case):
    closure, positive, negative = case
    with pytest.raises(ValueError):
        derive_c6_carried_affine_mean_obstruction(
            closure,
            positive_state=positive,
            negative_state=negative,
            max_gradient_values=335,
        )
    exact = derive_c6_carried_affine_mean_obstruction(
        closure,
        positive_state=positive,
        negative_state=negative,
        max_gradient_values=336,
    )
    assert exact.gradient_value_count == exact.max_gradient_values == 336


@pytest.mark.parametrize("budget", (True, 0, -1, 4096.0, "4096"))
def test_gradient_budget_type_and_positivity_are_checked(case, budget):
    closure, positive, negative = case
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_affine_mean_obstruction(
            closure,
            positive_state=positive,
            negative_state=negative,
            max_gradient_values=budget,
        )


def test_forged_derived_fields_do_not_change_the_observed_certificate(obstruction):
    forged = replace(
        obstruction,
        coordinate_spacings=(F(1),) * 6,
        coordinate_residues=(F(1),) * 6,
        global_spacing=F(1),
        mean_quantum=F(1),
        gradient_value_count=0,
        carry_lower_slack=F(0),
        carry_upper_slack=F(0),
        lift_error_squared_bound=F(0),
        positive_energy_bound=F(0),
        negative_energy_bound=F(0),
        mean_lower=F(0),
        mean_upper=F(1),
    )
    mean = _mean(obstruction.base_obstruction.closure.base_tube.state)
    expected = observe_c6_carried_affine_mean_escape(
        obstruction, mean_lower=mean, mean_upper=mean
    )
    observed = observe_c6_carried_affine_mean_escape(
        forged, mean_lower=mean, mean_upper=mean
    )
    assert observed == expected


def test_changed_primitive_origin_mean_is_not_silently_projected(case):
    closure, positive, negative = case
    changed = replace(
        positive, remainder=(positive.remainder[0] + M,) + positive.remainder[1:]
    )
    with pytest.raises(ValueError):
        derive_c6_carried_affine_mean_obstruction(
            closure, positive_state=changed, negative_state=negative
        )


@pytest.mark.parametrize("value", (0.5, 0, True, "1/2"))
def test_mean_interval_requires_exact_fraction_endpoints(obstruction, value):
    with pytest.raises((TypeError, ValueError)):
        observe_c6_carried_affine_mean_escape(
            obstruction,
            mean_lower=value,
            mean_upper=_mean(obstruction.base_obstruction.closure.base_tube.state),
        )


def test_intervals_cannot_exceed_shrunk_window_or_miss_origin(obstruction):
    mean, quantum = (
        _mean(obstruction.base_obstruction.closure.base_tube.state),
        obstruction.mean_quantum,
    )
    for lower, upper in (
        (obstruction.mean_lower - quantum, mean),
        (mean, obstruction.mean_upper + quantum),
        (mean + quantum, mean + 2 * quantum),
        (mean + quantum, mean),
    ):
        with pytest.raises(ValueError):
            observe_c6_carried_affine_mean_escape(
                obstruction, mean_lower=lower, mean_upper=upper
            )
