"""Independent exact contractions, uncertainty corners and shared consumers."""

from fractions import Fraction as Q
from itertools import product

import pytest

from tnfr.mathematics import _dyadic_linear as owner
from tnfr.mathematics._interval_chebyshev import interpolate_lobatto_rows, lobatto_nodes
from tnfr.mathematics._rational_interval import I


def _dot(left, right, **kwargs):
    return owner._dot_encoded_rows(
        owner._encode_admitted_row(left), owner._encode_admitted_row(right), **kwargs
    )


@pytest.mark.parametrize("numerator,denominator", [(1, 1), (3, 7), (-7, 3), (0, 5)])
def test_signed_contraction_encloses_every_exact_interval_corner(
    numerator, denominator
):
    left = (I(-3, 2), I(Q(1, 4), Q(3, 4)), I(-2, -1))
    right = (I(2, 4), I(-1, 3), I(Q(-3, 4), Q(-1, 4)))
    result = _dot(left, right, numerator=numerator, denominator=denominator)
    for a in product(*((value.lo, value.hi) for value in left)):
        for b in product(*((value.lo, value.hi) for value in right)):
            expected = Q(numerator, denominator) * sum(x * y for x, y in zip(a, b))
            assert result.contains(expected)
    assert result == _dot(right, left, numerator=numerator, denominator=denominator)


def test_signed_centers_cancel_before_the_shared_final_rounding():
    large, tiny = 2**220 + 1, Q(1, 2**110)
    left = tuple(map(I, (large, -large, tiny, -tiny)))
    right = tuple(map(I, (large, large, 3, 2)))
    assert _dot(left, right) == I(tiny)
    assert _dot(left, right, numerator=-5, denominator=7) == I(-5 * tiny / 7)


def test_canceling_midpoints_do_not_cancel_unrelated_interval_errors():
    left, right = (I(1, 2), I(-2, -1)), (I(1), I(1))
    assert _dot(left, right) == I(-1, 1)
    assert _dot((I(-1, 1),), (I(-1, 1),)) == I(-1, 1)


def test_products_below_absolute_grid_are_retained_until_the_final_projection():
    tiny = Q(1, 2**100)
    result = _dot((I(tiny),), (I(tiny),))
    assert result == I(tiny**2)
    assert result.contains(tiny**2) and result != I(0)
    # Each separate product is subgrid. Their signed sum is formed exactly.
    assert _dot((I(tiny), I(tiny)), (I(tiny), I(-tiny))) == I(0)


def test_materialized_nondyadic_values_retain_original_exact_dot():
    a, b = (Q(1, 3), Q(-2, 7)), (Q(-4, 9), Q(5, 11))
    expected = sum(first * second for first, second in zip(a, b))
    result = _dot(tuple(map(I, a)), tuple(map(I, b)))
    assert result.contains(expected)
    assert result.width < Q(1, 2**120)


def test_encoding_detaches_owned_rows_and_supports_repeated_contractions():
    values = [I(2), I(-3)]
    encoded = owner._encode_admitted_row(values)
    values[0] = I(100)
    right = owner._encode_admitted_row((I(5), I(7)))
    assert owner._dot_encoded_rows(encoded, right) == I(-11)
    assert owner._dot_encoded_rows(encoded, right, numerator=-1) == I(11)
    assert isinstance(encoded, tuple) and all(
        isinstance(value, tuple) for value in encoded
    )
    with pytest.raises(TypeError):
        encoded[0] = (0, 0, 0)


def test_empty_and_zero_owned_rows_keep_the_exact_zero():
    assert _dot((), ()) == I(0)
    assert _dot((I(0),), (I(-3, 8),)) == I(0)


@pytest.mark.parametrize(
    "value", [True, 1, Q(1, 3), 0.5, float("nan"), float("inf"), "1"]
)
def test_encoder_requires_shared_intervals_and_does_not_materialize_primitives(value):
    with pytest.raises(TypeError, match="admitted I"):
        owner._encode_admitted_row((value,))


@pytest.mark.parametrize(
    "bounds", [(True, Q(1)), (Q(0), 1.0), (Q(2), Q(1)), (Q(1, 3), Q(2, 3))]
)
def test_encoder_checks_shared_grid_before_extracting_endpoint_integers(bounds):
    forged = object.__new__(I)
    object.__setattr__(forged, "lo", bounds[0])
    object.__setattr__(forged, "hi", bounds[1])
    with pytest.raises((TypeError, ValueError)):
        owner._encode_admitted_row((forged,))


@pytest.mark.parametrize(
    "scale",
    [
        {"numerator": True},
        {"numerator": 1.0},
        {"numerator": Q(1)},
        {"denominator": False},
        {"denominator": 1.0},
        {"denominator": 0},
        {"denominator": -1},
    ],
)
def test_scale_rejects_before_consuming_encoded_entries(scale):
    # The object is not an encoded entry: an arithmetic attempt would fail
    # independently. Scale admission must decide before reaching it.
    with pytest.raises((TypeError, ValueError), match="scale|denominator"):
        owner._dot_encoded_rows((object(),), (object(),), **scale)


def test_dot_requires_equal_immutable_rows_even_for_zero_output_scale():
    encoded = owner._encode_admitted_row((I(1),))
    with pytest.raises(ValueError, match="same length"):
        owner._dot_encoded_rows(encoded, (), numerator=0)
    with pytest.raises(TypeError, match="immutable"):
        owner._dot_encoded_rows(list(encoded), encoded)


@pytest.mark.parametrize("degree", [2, 7, 96])
def test_lobatto_refactor_matches_direct_rational_center_radius_formula(degree):
    roots = lobatto_nodes(degree)
    samples = tuple(
        I(Q((index % 9) - 4, 32), Q((index % 9) - 4, 32) + Q(index + 1, 2**100))
        for index in range(degree + 1)
    )
    actual = interpolate_lobatto_rows((samples,), degree=degree)[0]
    expected = []
    for mode in range(degree + 1):
        center = radius = Q(0)
        for index, sample in enumerate(samples):
            angle = mode * index % (2 * degree)
            cosine = roots[min(angle, 2 * degree - angle)]
            weight = Q(1, 2) if index in (0, degree) else Q(1)
            center += weight * sample.midpoint * cosine.midpoint
            radius += weight * (
                abs(sample.midpoint) * cosine.radius
                + sample.radius * abs(cosine.midpoint)
                + sample.radius * cosine.radius
            )
        scale = Q(1 if mode in (0, degree) else 2, degree)
        expected.append(I(scale * (center - radius), scale * (center + radius)))
    assert actual == tuple(expected)
