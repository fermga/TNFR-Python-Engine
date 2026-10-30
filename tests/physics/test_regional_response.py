"""Independent exact witnesses for a conditional regional response criterion.

These tests use declared finite matrices; they never execute a TNFR trajectory.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from tnfr.physics.regional_response import observe_regional_response

IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
PARENT_INPUT = ((F(1, 2), 0, 1), (0, F(1, 2), 0), (0, 0, 1))


def _observe(
    transition=IDENTITY,
    metric=(1, 1, 1),
    region=(0, 1),
    difference=(1, -1, 0),
    residual=None,
):
    return observe_regional_response(
        transition, metric, region, difference, residual=residual
    )


def _quadratic(matrix, vector):
    return sum(
        (
            F(x) * F(a) * F(y)
            for x, row in zip(vector, matrix)
            for a, y in zip(row, vector)
        ),
        F(0),
    )


@pytest.mark.parametrize(
    "parent,energy_change,nonincreasing,strict,norm_bound",
    [
        (0, F(-3, 4), True, True, True),
        (1, F(0), True, False, True),
        (2, F(5, 4), False, False, False),
        (-2, F(-3, 4), True, True, False),
        (-3, F(0), True, False, False),
        (-4, F(5, 4), False, False, False),
    ],
)
def test_one_map_restores_or_amplifies_according_to_parent_input(
    parent, energy_change, nonincreasing, strict, norm_bound
):
    result = _observe(PARENT_INPUT, difference=(1, -1, parent))
    assert result.self_image == (F(1, 2), F(-1, 2))
    assert result.input_image == (F(parent, 2), F(-parent, 2))
    assert result.before_squared_norm == 2
    assert result.self_squared_norm == F(1, 2)
    assert result.available_squared_norm_drop == F(3, 2)
    assert result.input_squared_norm == F(parent * parent, 2)
    assert result.input_signed_work == parent + F(parent * parent, 2)
    assert result.energy_change == energy_change
    assert result.nonincreasing is nonincreasing
    assert result.strict_decrease is strict
    assert result.norm_only_sufficient is norm_bound
    assert result.nonincreasing == (
        result.input_signed_work <= result.available_squared_norm_drop
    )


def test_nonuniform_full_metric_and_ordered_region_have_hand_centering_matrix():
    result = _observe(metric=(1, 3, 2), region=(1, 0), difference=(4, 0, 5))
    assert result.child_mean == 1 and result.parent_mean == 5
    assert result.region_indices == (1, 0)
    assert result.centered_before == (-1, 3)
    assert result.centering == ((F(-1, 4), F(1, 4), 0), (F(3, 4), F(-3, 4), 0))
    assert result.energy_matrix == (
        (F(3, 4), F(-3, 4), 0),
        (F(-3, 4), F(3, 4), 0),
        (0, 0, 0),
    )
    assert result.before_squared_norm == 12
    assert result.energy_change == 0
    assert result.nullspace_preserved


def test_mean_contrast_parent_shape_and_residual_remain_separate():
    transition = ((1, 0, 1, 0), (0, 2, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1))
    result = _observe(
        transition, metric=(1, 3, 2, 4), difference=(4, 0, 5, 2), residual=(1, -1, 0, 0)
    )
    components = dict(result.input_components)
    assert result.child_mean == 1 and result.parent_mean == 3
    assert components == {
        "global_mean": (F(-9, 4), F(3, 4)),
        "mean_contrast": (F(3, 2), F(-1, 2)),
        "parent_centered": (F(9, 4), F(-3, 4)),
        "runtime_residual": (F(3, 2), F(-1, 2)),
    }
    assert result.self_image == (F(15, 4), F(-5, 4))
    assert result.input_image == (3, -1)
    assert result.centered_after == (F(27, 4), F(-9, 4))
    assert result.before_squared_norm == 12
    assert result.self_squared_norm == F(75, 4)
    assert result.input_squared_norm == 12
    assert result.input_signed_work == 42
    assert result.energy_change == F(195, 8)
    assert not result.nonincreasing


@pytest.mark.parametrize(
    "residual,linear,quadratic,change",
    [
        ((-1, 1, 0), -2, 1, -1),
        ((1, -1, 0), 2, 1, 3),
        ((7, 7, -99), 0, 0, 0),
    ],
)
def test_signed_runtime_correction_is_not_a_nonnegative_error_penalty(
    residual, linear, quadratic, change
):
    result = _observe(difference=(1, -1, 3), residual=residual)
    assert result.ideal_energy_change == 0
    assert result.residual_linear_energy == linear
    assert result.residual_quadratic_energy == quadratic
    assert result.energy_change == change
    assert result.energy_change == (
        result.ideal_energy_change
        + result.residual_linear_energy
        + result.residual_quadratic_energy
    )


def test_quadratic_matrix_predicts_ideal_change_independently_of_residual():
    result = _observe(
        PARENT_INPUT,
        metric=(1, 3, 2),
        difference=(2, -1, 3),
        residual=(F(1, 4), F(-1, 3), 7),
    )
    assert (
        _quadratic(result.energy_matrix, result.difference)
        == result.before_squared_norm
    )
    assert (
        _quadratic(result.change_matrix, result.difference) / 2
        == result.ideal_energy_change
    )
    assert result.change_matrix == tuple(zip(*result.change_matrix))
    assert (
        result.energy_change
        == (result.input_signed_work - result.available_squared_norm_drop) / 2
    )
    assert result.energy_change == (
        result.ideal_energy_change
        + result.residual_linear_energy
        + result.residual_quadratic_energy
    )


def test_zero_initial_shape_with_incoming_parent_excludes_every_finite_gain():
    result = _observe(PARENT_INPUT, difference=(0, 0, 1))
    assert result.centered_before == (0, 0)
    assert result.before_squared_norm == 0
    assert result.centered_after == (F(1, 2), F(-1, 2))
    assert result.energy_change == F(1, 4)
    assert not result.nullspace_preserved
    assert any(any(vector) for _, vector in result.nullspace_images)
    assert not result.nonincreasing and not result.norm_only_sufficient
    # Multiplying zero input energy by any finite factor cannot bound this output.
    for finite_gain in (0, 1, 10, 10**100):
        assert 2 * result.energy_change > finite_gain * result.before_squared_norm


def test_regional_mean_leakage_is_a_distinct_nullspace_obstruction():
    transition = ((1, 0, 0), (0, F(1, 2), 0), (0, 0, 1))
    result = _observe(transition, difference=(1, 1, 0))
    assert result.before_squared_norm == 0
    assert result.centered_after == (F(1, 4), F(-1, 4))
    assert result.energy_change == F(1, 16)
    assert dict(result.input_components)["mean_contrast"] == result.centered_after
    assert not result.nullspace_preserved


def test_represented_global_constant_leak_is_kept_even_when_tiny():
    epsilon = F(1, 2**90)
    transition = ((1 + epsilon, 0, 0), (0, 1, 0), (0, 0, 1))
    result = _observe(transition, difference=(1, 1, 1))
    assert result.centered_after == (epsilon / 2, -epsilon / 2)
    assert dict(result.input_components)["global_mean"] == result.centered_after
    assert result.energy_change == epsilon**2 / 4
    assert not result.nullspace_preserved


def test_nullspace_preservation_alone_does_not_imply_contraction():
    result = _observe(((2, 0, 0), (0, 2, 0), (0, 0, 1)))
    assert result.nullspace_preserved
    assert all(not any(vector) for _, vector in result.nullspace_images)
    assert result.energy_change == 3 and not result.nonincreasing


def test_zero_and_singleton_regional_scores_cannot_show_strict_restoration():
    zero = _observe(difference=(0, 0, 0))
    assert zero.before_squared_norm == zero.energy_change == 0
    assert zero.nonincreasing and zero.norm_only_sufficient
    assert not zero.strict_decrease
    singleton = _observe(PARENT_INPUT, region=(1,), residual=(5, 6, 7))
    assert singleton.centered_before == singleton.centered_after == (0,)
    assert singleton.energy_change == 0 and singleton.nullspace_preserved
    assert singleton.nonincreasing and singleton.norm_only_sufficient
    assert not singleton.strict_decrease


def test_float_inputs_keep_their_binary64_values_and_result_is_frozen():
    result = _observe(((0.1, 0, 0), (0, 0.1, 0), (0, 0, 1)), metric=(1.0, 1.0, 1.0))
    assert result.transition[0][0] == F.from_float(0.1)
    assert result.transition[0][0] != F(1, 10)
    assert result.residual == (0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        result.energy_change = 0


@pytest.mark.parametrize(
    "region",
    [
        (),
        (0, 1, 2),
        (0, 0),
        (-1,),
        (3,),
        (True,),
        (0.0,),
        (F(0),),
    ],
)
def test_malformed_regional_index_sets_are_rejected(region):
    with pytest.raises((TypeError, ValueError)):
        _observe(region=region)


@pytest.mark.parametrize(
    "metric",
    [
        (0, 1, 1),
        (-1, 1, 1),
        (float("inf"), 1, 1),
        (1, float("nan"), 1),
        (1, 1),
        (1, 1, 1, 1),
    ],
)
def test_metric_must_be_positive_finite_and_cover_the_full_domain(metric):
    with pytest.raises((TypeError, ValueError)):
        _observe(metric=metric)


@pytest.mark.parametrize(
    "transition",
    [
        (),
        ((1, 0), (0, 1)),
        ((1, 0, 0), (0, 1), (0, 0, 1)),
        ((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, float("inf"), 0), (0, 0, 1)),
    ],
)
def test_transition_must_be_finite_and_square_on_the_same_domain(transition):
    with pytest.raises((TypeError, ValueError)):
        _observe(transition)


@pytest.mark.parametrize("keyword", ["difference", "residual"])
@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        (1, 2, 3, 4),
        (float("nan"), 0, 0),
        (0, float("inf"), 0),
    ],
)
def test_state_and_residual_require_complete_finite_vectors(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        _observe(**{keyword: value})
