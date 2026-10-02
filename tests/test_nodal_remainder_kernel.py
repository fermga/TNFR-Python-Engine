"""Independent nodal-area, representation and boundary tests for exact carry."""

import math
import sys
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

import tnfr.dynamics._euler_kernel as owner
from tnfr.dynamics._euler_kernel import (
    NODAL_REMAINDER_ABSOLUTE_BOUND,
    NODAL_REMAINDER_DENOMINATOR_BITS,
    NodalRemainderState,
    advance_nodal_remainder,
    euler_update,
    initialize_nodal_remainder,
)

F = Fraction
DELTA = F(1, 2**53)


def _advance(state, *, h=1.0, capacity=None, pressure=None):
    count = len(state.epi)
    return advance_nodal_remainder(
        state,
        timestep=h,
        capacity=(1.0,) * count if capacity is None else capacity,
        pressure=(2.0**-55,) * count if pressure is None else pressure,
    )


def test_small_exact_areas_are_fed_back_until_the_visible_state_changes():
    state = initialize_nodal_remainder((0.5,))
    legacy = 0.5
    steps = []
    quantum = F(1, 2**55)
    for ordinal in range(1, 5):
        step = _advance(state)
        state = step.after
        steps.append(step)
        legacy = euler_update(legacy, 1.0, float(quantum))
        assert state.exact_epi == (F(1, 2) + ordinal * quantum,)
        assert step.visible_increment[0] == quantum + step.carry_transfer[0]
        assert step.nodal_balance_residual == (0,)
    assert legacy == 0.5
    assert tuple(step.after.epi[0] for step in steps) == (
        0.5,
        0.5,
        float(F(1, 2) + DELTA),
        float(F(1, 2) + DELTA),
    )
    assert tuple(step.after.remainder[0] for step in steps) == (
        quantum,
        2 * quantum,
        -quantum,
        0,
    )
    assert state.remainder == (0,)
    assert state.epi != (legacy,)


def test_exact_product_area_and_partial_call_partition_give_identical_encoding():
    initial = initialize_nodal_remainder((0.5, 0.6))
    capacity, pressure = (0.3, 0.7), (0.1, -0.1)
    whole = _advance(initial, h=0.25, capacity=capacity, pressure=pressure)
    current = initial
    visible, transfer = [F(0), F(0)], [F(0), F(0)]
    for _ in range(4):
        step = _advance(current, h=0.0625, capacity=capacity, pressure=pressure)
        current = step.after
        for i in range(2):
            visible[i] += step.visible_increment[i]
            transfer[i] += step.carry_transfer[i]
    expected_area = tuple(
        F(1, 4) * F(nu) * F(p) for nu, p in zip(capacity, pressure, strict=True)
    )
    expected = tuple(
        F(value) + area for value, area in zip(initial.epi, expected_area, strict=True)
    )
    assert whole.exact_increment == expected_area
    assert current == whole.after
    assert current.exact_epi == expected
    assert current.epi == tuple(map(float, expected))
    assert tuple(visible) == whole.visible_increment
    assert tuple(transfer) == whole.carry_transfer
    assert expected_area[0] != F(0.25 * (capacity[0] * pressure[0]))


def test_rounded_duration_sum_is_not_silently_treated_as_exact_partition():
    initial = initialize_nodal_remainder((0.5,))
    first = _advance(initial, h=0.1, pressure=(0.125,))
    split = _advance(first.after, h=0.2, pressure=(0.125,))
    whole = _advance(initial, h=0.3, pressure=(0.125,))
    difference = (F(0.1) + F(0.2) - F(0.3)) * F(1, 8)
    assert difference != 0
    assert split.after.exact_epi[0] - whole.after.exact_epi[0] == difference
    assert split.after != whole.after


def test_exact_opposite_sources_remove_the_repeated_boundary_rounding_error():
    pressure = (2.0**-50, -(2.0**-50))
    state = initialize_nodal_remainder((0.5, 0.5))
    legacy = state.epi
    for _ in range(4):
        state = _advance(state, h=0.0625, pressure=pressure).after
        legacy = tuple(
            euler_update(x, 0.0625, p) for x, p in zip(legacy, pressure, strict=True)
        )
        assert sum(state.exact_epi) == 1
    expected = (F(1, 2) + F(1, 2**52), F(1, 2) - F(1, 2**52))
    assert state.exact_epi == expected
    assert state.epi == tuple(map(float, expected))
    assert state.remainder == (0, 0)
    assert sum(map(F, legacy)) / 2 - F(1, 2) == -DELTA


def test_balanced_exact_mean_can_still_have_a_terminal_visible_rounding_error():
    initial = initialize_nodal_remainder((0.5, 0.5))
    step = _advance(initial, pressure=(2.0**-54, -(2.0**-54)))
    assert sum(step.exact_increment) == 0
    assert sum(step.after.exact_epi) / 2 == F(1, 2)
    visible_mean = sum(map(F, step.after.epi)) / 2
    assert visible_mean - F(1, 2) == -F(1, 2**55)
    assert step.after.remainder == (F(1, 2**54), 0)
    assert visible_mean - F(1, 2) == -sum(step.after.remainder) / 2
    assert abs(visible_mean - F(1, 2)) <= NODAL_REMAINDER_ABSOLUTE_BOUND


def test_heterogeneous_capacity_requires_nodal_area_balance_not_pressure_balance():
    initial = initialize_nodal_remainder((0.5, 0.5))
    pressure, capacity = (0.1, -0.1), (1.0, 2.0)
    step = _advance(initial, h=0.25, capacity=capacity, pressure=pressure)
    assert sum(map(F, pressure)) == 0
    assert sum(step.exact_increment) == -F(0.1) / 4
    assert sum(step.after.exact_epi) / 2 == F(1, 2) - F(0.1) / 8
    # The fixed inverse-capacity metric cancels these two exact sources.
    weights = (F(1), F(1, 2))
    assert (
        sum(
            weight * area
            for weight, area in zip(weights, step.exact_increment, strict=True)
        )
        == 0
    )
    assert sum(
        weight * value
        for weight, value in zip(weights, step.after.exact_epi, strict=True)
    ) == F(3, 4)


def test_triple_minimum_subnormal_area_is_retained_and_can_cancel_exactly():
    smallest = math.ulp(0.0)
    initial = initialize_nodal_remainder((0.5,))
    first = _advance(initial, h=smallest, capacity=(smallest,), pressure=(smallest,))
    quantum = F(1, 2**3222)
    assert NODAL_REMAINDER_DENOMINATOR_BITS == 3222
    assert first.exact_increment == (quantum,)
    assert first.after.epi == initial.epi
    assert first.after.remainder == (quantum,)
    assert first.after.exact_epi == (F(1, 2) + quantum,)
    assert smallest * (smallest * smallest) == 0.0
    second = _advance(
        first.after, h=smallest, capacity=(smallest,), pressure=(-smallest,)
    )
    assert second.after == initial


def test_zero_step_preserves_nonzero_carry_even_when_separate_rate_would_overflow():
    before = _advance(initialize_nodal_remainder((0.5,))).after
    assert before.remainder != (0,)
    result = _advance(
        before, h=0.0, capacity=(sys.float_info.max,), pressure=(sys.float_info.max,)
    )
    assert result.after == before
    assert (
        result.exact_increment
        == result.visible_increment
        == result.carry_transfer
        == (0,)
    )


def test_zero_capacity_preserves_only_its_own_full_encoding():
    state = _advance(initialize_nodal_remainder((0.5, 0.5))).after
    result = _advance(
        state, capacity=(0.0, 1.0), pressure=(sys.float_info.max, 2.0**-55)
    )
    assert result.after.epi[0] == state.epi[0]
    assert result.after.remainder[0] == state.remainder[0]
    assert result.after.exact_epi[0] == state.exact_epi[0]
    assert result.after.exact_epi[1] == F(1, 2) + F(1, 2**54)


def test_valid_nonzero_initial_carry_remains_explicit_in_the_visible_telescope():
    state = NodalRemainderState((0.5,), (F(1, 2**55),), 0.05, 1.0)
    result = _advance(state, pressure=(-(2.0**-55),))
    assert result.after.epi == state.epi
    assert result.after.remainder == (0,)
    assert result.visible_increment == (0,)
    assert result.exact_increment == (-F(1, 2**55),)
    assert result.carry_transfer == (F(1, 2**55),)


@pytest.mark.parametrize(
    "visible,remainder",
    (
        (0.5, DELTA),
        (math.nextafter(0.5, math.inf), -DELTA / 2),
        (math.nextafter(0.5, math.inf), DELTA / 2),
    ),
)
def test_forged_encoding_cannot_use_an_unequal_or_wrong_tie_visible_value(
    visible, remainder
):
    forged = NodalRemainderState((visible,), (remainder,), 0.05, 1.0)
    with pytest.raises(ValueError, match="nearest-even"):
        _ = forged.exact_epi
    with pytest.raises(ValueError, match="nearest-even"):
        _advance(forged, h=0.0)


@pytest.mark.parametrize(
    "remainder,error",
    (
        (F(1, 3), ValueError),
        (F(1, 2**3223), ValueError),
        (0, TypeError),
        (0.0, TypeError),
        (True, TypeError),
    ),
)
def test_remainders_require_the_declared_exact_dyadic_representation(remainder, error):
    state = NodalRemainderState((0.5,), (remainder,), 0.05, 1.0)
    with pytest.raises(error):
        _advance(state)


@pytest.mark.parametrize("remainders", ((), (F(0), F(0)), [F(0)]))
def test_carry_dimensions_and_ordered_tuple_contract_are_validated(remainders):
    state = NodalRemainderState((0.5,), remainders, 0.05, 1.0)
    with pytest.raises(ValueError, match="dimensions"):
        _advance(state)


@pytest.mark.parametrize(
    "initial,lower,upper,pressure",
    (
        (1.0, 0.05, 1.0, 2.0**-54),
        (0.5, 0.5, 1.0, -(2.0**-56)),
    ),
)
def test_exact_band_breach_is_rejected_even_when_display_would_stay_at_boundary(
    initial, lower, upper, pressure
):
    state = initialize_nodal_remainder((initial,), epi_lower=lower, epi_upper=upper)
    exact_after = F(initial) + F(pressure)
    assert float(exact_after) == initial
    with pytest.raises(ValueError, match="exact nodal update"):
        _advance(state, pressure=(pressure,))
    forged = replace(state, remainder=(F(pressure),))
    with pytest.raises(ValueError, match="reconstructed exact EPI"):
        _ = forged.exact_epi


def test_out_of_band_intermediate_step_cannot_be_justified_by_later_cancellation():
    initial = initialize_nodal_remainder((0.5,), epi_lower=0.25, epi_upper=0.75)
    assert _advance(initial, pressure=(0.0,)).after == initial
    with pytest.raises(ValueError, match="exact nodal update"):
        _advance(initial, pressure=(0.5,))


def test_late_coordinate_failure_leaves_the_entire_input_state_unchanged():
    initial = _advance(initialize_nodal_remainder((0.5, 0.5))).after
    before = (initial.epi, initial.remainder, initial.exact_epi)
    with pytest.raises(ValueError, match="exact nodal update"):
        _advance(initial, pressure=(0.1, 1.0))
    assert (initial.epi, initial.remainder, initial.exact_epi) == before
    with pytest.raises(FrozenInstanceError):
        initial.epi = (0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        initial.remainder = (F(0), F(0))
    step = _advance(initial, h=0.0)
    with pytest.raises(FrozenInstanceError):
        step.after = initialize_nodal_remainder((0.5,))


@pytest.mark.parametrize(
    "values,error",
    (
        ((), ValueError),
        ([0.5], TypeError),
        ({0: 0.5}, TypeError),
        ((1,), TypeError),
        ((True,), TypeError),
        ((F(1, 2),), TypeError),
        ((math.nan,), ValueError),
        ((math.inf,), ValueError),
        ((-math.inf,), ValueError),
        ((0.0,), ValueError),
        ((-0.5,), ValueError),
        ((1.1,), ValueError),
    ),
)
def test_initial_epi_requires_nonempty_finite_represented_band_coordinates(
    values, error
):
    with pytest.raises(error):
        initialize_nodal_remainder(values)


@pytest.mark.parametrize("field", ("h", "capacity", "pressure"))
@pytest.mark.parametrize(
    "value,error",
    (
        (1, TypeError),
        (True, TypeError),
        (F(1, 2), TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
    ),
)
def test_nodal_inputs_must_already_be_finite_binary64_scalars(field, value, error):
    state = initialize_nodal_remainder((0.5,))
    with pytest.raises(error):
        _advance(state, **{field: value if field == "h" else (value,)})


@pytest.mark.parametrize(
    "field,values,error",
    (
        ("capacity", (), ValueError),
        ("pressure", (), ValueError),
        ("capacity", [1.0], TypeError),
        ("pressure", [0.0], TypeError),
        ("capacity", (1.0, 1.0), ValueError),
        ("pressure", (0.0, 0.0), ValueError),
    ),
)
def test_nodal_input_tuples_cannot_zip_truncate_or_change_dimension(
    field, values, error
):
    with pytest.raises(error):
        _advance(initialize_nodal_remainder((0.5,)), **{field: values})


@pytest.mark.parametrize("arguments", ({"h": -0.1}, {"capacity": (-0.1,)}))
def test_duration_and_capacity_must_be_nonnegative(arguments):
    with pytest.raises(ValueError, match="nonnegative"):
        _advance(initialize_nodal_remainder((0.5,)), **arguments)


@pytest.mark.parametrize(
    "lower,upper,error",
    (
        (0, 1.0, TypeError),
        (0.05, 1, TypeError),
        (0.0, 1.0, ValueError),
        (0.6, 0.4, ValueError),
        (0.05, 2.0, ValueError),
        (math.nan, 1.0, ValueError),
    ),
)
def test_bands_are_finite_positive_represented_unit_intervals(lower, upper, error):
    with pytest.raises(error):
        initialize_nodal_remainder((0.5,), epi_lower=lower, epi_upper=upper)


def test_invalid_state_type_and_changed_ieee_precondition_are_rejected(monkeypatch):
    initial = initialize_nodal_remainder((0.5,))
    with pytest.raises(TypeError, match="NodalRemainderState"):
        advance_nodal_remainder({}, timestep=1.0, capacity=(1.0,), pressure=(0.0,))
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(RuntimeError, match="IEEE binary64"):
        _advance(initial)
    with pytest.raises(RuntimeError, match="IEEE binary64"):
        initialize_nodal_remainder((0.5,))
