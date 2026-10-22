"""Conditional interval obstructions to recovery of a capacity profile."""

from fractions import Fraction

import pytest

from tnfr.physics.capacity_feedback import derive_capacity_interval_recovery_bound

F = Fraction


def test_asymmetric_profile_has_distinct_fixed_and_rescaled_separations():
    result = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=F(1, 2))
    assert result.capacity == (1, 2, 6)
    assert result.contraction == F(1, 2)
    assert result.mean_capacity == 3
    assert result.perturbed_capacity == (2, F(5, 2), F(9, 2))
    assert result.original_interval == (1, 6)
    assert result.invariant_interval == (2, F(9, 2))
    assert result.original_ratio == 6
    assert result.admissible_ratio == F(9, 4)
    assert result.fixed_scale_separation == F(3, 2)
    assert result.rescaled_separation == F(15, 14)
    assert result.minimizing_scale == F(13, 14)
    assert result.additive_separation == F(5, 4)
    assert result.minimizing_offset == F(-1, 4)


def test_rescaled_bound_is_attained_by_an_independent_admissible_box_point():
    result = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=F(1, 2))
    candidate = (F(2), F(2), F(9, 2))
    scaled_target = (F(13, 14), F(13, 7), F(39, 7))
    errors = tuple(a - b for a, b in zip(candidate, scaled_target, strict=True))
    assert errors == (F(15, 14), F(1, 7), F(-15, 14))
    assert max(map(abs, errors)) == result.rescaled_separation
    lo, hi = result.invariant_interval
    assert all(lo <= value <= hi for value in candidate)
    # This admissible vector witnesses sharpness of the box bound only.
    # No native trajectory reaching it, or preserving the box, is asserted.


def test_additive_bound_is_attained_without_rescaling_the_target_profile():
    result = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=F(1, 2))
    candidate = (F(2), F(2), F(9, 2))
    shifted_target = (F(3, 4), F(7, 4), F(23, 4))
    errors = tuple(a - b for a, b in zip(candidate, shifted_target, strict=True))
    assert errors == (F(5, 4), F(1, 4), F(-5, 4))
    assert max(map(abs, errors)) == result.additive_separation
    assert result.additive_separation != result.rescaled_separation


def test_minimizing_additive_offset_does_not_claim_positive_capacity_admission():
    result = derive_capacity_interval_recovery_bound((1, 1, 10), contraction=1)
    assert result.mean_capacity == 4
    assert result.invariant_interval == (4, 4)
    assert result.additive_separation == F(9, 2)
    assert result.minimizing_offset == F(-3, 2)
    shifted_target = tuple(
        value + result.minimizing_offset for value in result.capacity
    )
    assert shifted_target == (F(-1, 2), F(-1, 2), F(17, 2))
    assert (
        max(abs(F(4) - value) for value in shifted_target) == result.additive_separation
    )
    # The affine-line optimum includes negative coordinates. The bound still
    # applies to positive profiles, but this shift is not a physical admission.


@pytest.mark.parametrize("scale", (F(1, 10), F(1, 2), F(13, 14), F(1), F(2)))
def test_nearest_box_point_at_other_scales_cannot_beat_the_bound(scale):
    result = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=F(1, 2))
    lower, upper = result.invariant_interval
    scaled_target = tuple(scale * value for value in result.capacity)
    # Coordinatewise interval projection minimizes the infinity distance to
    # this box for a fixed scale; these are finite algebraic controls.
    closest = tuple(min(upper, max(lower, value)) for value in scaled_target)
    distance = max(abs(a - b) for a, b in zip(closest, scaled_target, strict=True))
    assert distance >= result.rescaled_separation
    if scale == result.minimizing_scale:
        assert distance == result.rescaled_separation
    else:
        assert distance > result.rescaled_separation


def test_full_contraction_reduces_the_admissible_interval_to_the_mean():
    result = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=1)
    assert result.perturbed_capacity == (3, 3, 3)
    assert result.invariant_interval == (3, 3)
    assert result.admissible_ratio == 1
    assert result.fixed_scale_separation == 3
    assert result.rescaled_separation == F(15, 7)
    assert result.minimizing_scale == F(6, 7)
    assert result.additive_separation == F(5, 2)
    assert result.minimizing_offset == F(-1, 2)


@pytest.mark.parametrize("capacity", ((F(7, 4),), (F(7, 4),) * 3))
def test_uniform_profiles_have_no_false_recovery_obstruction(capacity):
    result = derive_capacity_interval_recovery_bound(capacity, contraction=F(3, 5))
    assert result.perturbed_capacity == capacity
    assert result.mean_capacity == F(7, 4)
    assert result.original_interval == result.invariant_interval == (F(7, 4), F(7, 4))
    assert result.original_ratio == result.admissible_ratio == 1
    assert result.fixed_scale_separation == result.rescaled_separation == 0
    assert result.minimizing_scale == 1
    assert result.additive_separation == result.minimizing_offset == 0


def test_common_capacity_scaling_changes_distances_but_not_ratios_or_best_scale():
    original = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=F(1, 2))
    multiplier = F(7, 3)
    scaled = derive_capacity_interval_recovery_bound(
        tuple(multiplier * value for value in original.capacity), contraction=F(1, 2)
    )
    assert scaled.perturbed_capacity == tuple(
        multiplier * value for value in original.perturbed_capacity
    )
    assert scaled.mean_capacity == multiplier * original.mean_capacity
    assert scaled.fixed_scale_separation == multiplier * original.fixed_scale_separation
    assert scaled.rescaled_separation == multiplier * original.rescaled_separation
    assert scaled.original_ratio == original.original_ratio
    assert scaled.admissible_ratio == original.admissible_ratio
    assert scaled.minimizing_scale == original.minimizing_scale
    assert scaled.additive_separation == multiplier * original.additive_separation
    assert scaled.minimizing_offset == multiplier * original.minimizing_offset


def test_arbitrarily_small_exact_perturbation_is_not_rounded_to_no_obstruction():
    contraction = F(1, 2**1100)
    result = derive_capacity_interval_recovery_bound((1, 2, 6), contraction=contraction)
    assert float(contraction) == 0.0
    assert result.contraction == contraction
    assert result.rescaled_separation == F(15, 7) * contraction > 0
    assert result.additive_separation == F(5, 2) * contraction > 0
    assert result.invariant_interval[0] > 1
    assert result.invariant_interval[1] < 6
    assert result.admissible_ratio < result.original_ratio


@pytest.mark.parametrize(
    "capacity,exception",
    [
        ((), ValueError),
        ((0, 1), ValueError),
        ((-1, 1), ValueError),
        ((float("nan"), 1), ValueError),
        ((float("inf"), 1), ValueError),
        ((True, 1), TypeError),
        ((complex(1, 0), 1), TypeError),
        ("1,2", TypeError),
        ({1, 2}, TypeError),
        ({0: 1, 1: 2}, TypeError),
    ],
)
def test_capacity_requires_an_ordered_nonempty_positive_finite_vector(
    capacity, exception
):
    with pytest.raises(exception):
        derive_capacity_interval_recovery_bound(capacity, contraction=F(1, 2))


@pytest.mark.parametrize(
    "contraction,exception",
    [
        (0, ValueError),
        (F(-1, 10), ValueError),
        (F(11, 10), ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.5", TypeError),
        (complex(0.5, 0), TypeError),
    ],
)
def test_contraction_requires_a_strictly_positive_unit_interval_scalar(
    contraction, exception
):
    with pytest.raises(exception):
        derive_capacity_interval_recovery_bound((1, 2, 6), contraction=contraction)
