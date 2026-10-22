"""Exact archived-vector admission without graph or producer execution."""

from fractions import Fraction

import pytest

from benchmarks.thol_retained_reset_audit import _v


def test_one_shot_vector_preserves_signed_exact_and_represented_coordinates():
    values = [-3, -0.1, "-2/7", Fraction(5, 11)]
    expected = tuple(Fraction(value) for value in values)

    assert _v(value for value in values) == expected
    assert _v(values) == expected
    assert _v(iter(())) == ()


def test_vector_materializes_the_caller_iterable_once():
    class Once:
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations != 1:
                raise AssertionError("caller iterable was materialized twice")
            yield -2
            yield "1/3"

    values = Once()
    assert _v(values) == (Fraction(-2), Fraction(1, 3))
    assert values.iterations == 1


@pytest.mark.parametrize(
    "invalid", [True, False, float("nan"), float("inf"), -float("inf"), 1j]
)
def test_one_shot_vector_keeps_invalid_coordinate_rejection(invalid):
    with pytest.raises(
        ValueError, match="coordinates must be finite exact or represented reals"
    ):
        _v(value for value in (1, invalid, 2))
