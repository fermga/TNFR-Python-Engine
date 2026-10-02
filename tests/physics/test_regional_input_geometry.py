"""Exact environmental-image and protected-readout witnesses.

Declared matrices only: no kernels, trajectories, or retained artifact reads.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from tnfr.physics.regional_response import observe_regional_input_geometry

METRIC = (1, 2, 3, 5)
REGION = (0, 1, 2)
IDENTITY = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
ONE_INPUT = ((1, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
CENTER = (
    (F(5, 6), F(-1, 3), F(-1, 2)),
    (F(-1, 6), F(2, 3), F(-1, 2)),
    (F(-1, 6), F(-1, 3), F(1, 2)),
)
PROTECTED = ((0, 0, 0), (0, F(3, 5), F(-3, 5)), (0, F(-2, 5), F(2, 5)))


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def _mv(a, x):
    return tuple(_dot(row, x) for row in a)


def _product(a, b):
    return tuple(
        tuple(_dot(row, column) for column in zip(*b, strict=True)) for row in a
    )


def _observe(transition=ONE_INPUT, metric=METRIC, region=REGION):
    return observe_regional_input_geometry(transition, metric, region)


def _assert_geometry(result):
    """Verify displayed certificates without using production algebra helpers."""
    region = result.region_indices
    h = tuple(result.metric_weights[i] for i in region)
    count = len(region)
    mass = sum(h, F(0))
    center = tuple(
        tuple(F(i == j) - h[j] / mass for j in range(count)) for i in range(count)
    )
    zero = ((F(0),) * count,) * count
    p, q = result.image_projection, result.protected_projection
    assert result.shape_dimension == count - 1
    assert result.rank + result.protected_dimension == result.shape_dimension
    assert len(result.image_basis) == result.rank
    assert len(result.protected_basis) == result.protected_dimension
    assert _product(p, p) == p
    assert _product(q, q) == q
    assert _product(p, q) == _product(q, p) == zero
    assert (
        tuple(
            tuple(x + y for x, y in zip(a, b, strict=True))
            for a, b in zip(p, q, strict=True)
        )
        == center
    )
    for projection in (p, q):
        for i in range(count):
            for j in range(count):
                assert h[i] * projection[i][j] == h[j] * projection[j][i]
        assert not any(_mv(projection, (F(1),) * count))
    columns = tuple(zip(*result.input_map, strict=True))
    for column in columns:
        assert _dot(h, column) == 0
        assert _mv(p, column) == column
        assert not any(_mv(q, column))
    for index, column in zip(
        result.image_basis_indices, result.image_basis, strict=True
    ):
        assert column == columns[index]
    for j, column in enumerate(columns):
        reconstructed = tuple(
            sum(
                (
                    basis[i] * result.input_coefficients[k][j]
                    for k, basis in enumerate(result.image_basis)
                ),
                F(0),
            )
            for i in range(count)
        )
        assert reconstructed == column
    if result.rank:
        gram = tuple(
            tuple(
                _dot(tuple(x * w for x, w in zip(a, h, strict=True)), b)
                for b in result.image_basis
            )
            for a in result.image_basis
        )
        assert _product(gram, result.gram_inverse) == tuple(
            tuple(F(i == j) for j in range(result.rank)) for i in range(result.rank)
        )
    else:
        assert result.gram_inverse == result.input_coefficients == ()
    for basis, readout in zip(
        result.protected_basis, result.protected_readout_rows, strict=True
    ):
        assert _dot(h, basis) == 0
        assert _mv(q, basis) == basis
        assert readout == tuple(x * w for x, w in zip(basis, h, strict=True))
        assert all(_dot(readout, column) == 0 for column in columns)
    for basis, column in zip(result.input_basis, columns, strict=True):
        assert not any(_mv(result.centering, basis))
        assert _mv(result.centering, _mv(result.transition, basis)) == column


def test_identity_has_zero_input_image_and_full_protected_shape_space():
    result = _observe(IDENTITY)
    assert result.rank == 0 and result.protected_dimension == 2
    assert result.image_basis_indices == result.image_basis == ()
    assert result.input_map == ((0, 0),) * 3
    assert result.image_projection == ((0, 0, 0),) * 3
    assert result.protected_projection == CENTER
    _assert_geometry(result)


def test_one_parent_input_has_hand_computed_weighted_annihilator():
    result = _observe()
    assert result.rank == result.protected_dimension == 1
    assert result.input_labels == ("region_constant", "outside_3")
    assert result.input_basis == ((1, 1, 1, 0), (0, 0, 0, 1))
    assert result.input_map == ((0, F(5, 6)), (0, F(-1, 6)), (0, F(-1, 6)))
    assert result.image_basis_indices == (1,)
    assert result.gram_inverse == ((F(6, 5),),)
    assert result.input_coefficients == ((0, 1),)
    assert result.protected_projection == PROTECTED
    # Protection is a weighted relation between the two unforced child rows.
    assert _mv(result.protected_projection, (0, 3, -2)) == (0, 3, -2)
    _assert_geometry(result)


def test_two_parent_inputs_span_every_shape_direction():
    transition = (
        (1, 0, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    )
    result = _observe(transition, metric=(1, 2, 3, 5, 7))
    assert result.rank == 2 and result.protected_dimension == 0
    assert result.image_projection == CENTER
    assert result.protected_projection == ((0, 0, 0),) * 3
    assert result.protected_basis == result.protected_readout_rows == ()
    _assert_geometry(result)


def test_regional_mean_input_is_retained_separately_from_parent_coordinates():
    transition = ((2, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    result = _observe(transition)
    assert result.input_map == ((F(5, 6), 0), (F(-1, 6), 0), (F(-1, 6), 0))
    assert result.image_basis_indices == (0,)
    assert result.protected_projection == PROTECTED
    _assert_geometry(result)


def test_exact_nonzero_input_is_not_discarded_by_a_float_tolerance():
    epsilon = F(1, 2**1200)
    transition = ((1, 0, 0, epsilon), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    assert float(epsilon) == 0.0
    result = _observe(transition)
    assert result.rank == 1
    assert result.input_map[0][1] == F(5, 6) * epsilon
    assert result.protected_projection == PROTECTED
    _assert_geometry(result)


@pytest.mark.parametrize("scale", [F(-3), F(1, 7), F(17)])
def test_nonzero_input_scaling_preserves_geometry(scale):
    transition = ((1, 0, 0, scale), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    result = _observe(transition)
    assert result.rank == 1 and result.protected_projection == PROTECTED
    assert result.gram_inverse == ((F(6, 5) / scale**2,),)
    _assert_geometry(result)


def test_protected_readout_does_not_imply_invariant_self_dynamics():
    transition = ((0, 1, 0, 1), (0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1))
    result = _observe(transition)
    assert result.protected_projection == PROTECTED
    protected_state = (0, 3, -2)
    assert _mv(result.protected_projection, protected_state) == protected_state
    after = _mv(result.centering, _mv(result.transition, (*protected_state, F(0))))
    assert after == (F(19, 6), F(-11, 6), F(1, 6))
    assert any(_mv(result.image_projection, after))
    assert _mv(result.protected_projection, after) != after
    _assert_geometry(result)


def test_region_order_preserves_coordinate_meaning_and_full_metric():
    result = _observe(region=(2, 0, 1))
    assert result.region_indices == (2, 0, 1)
    assert result.metric_weights == METRIC
    assert result.input_map == ((0, F(-1, 6)), (0, F(5, 6)), (0, F(-1, 6)))
    assert result.protected_projection == tuple(
        tuple(PROTECTED[i][j] for j in (2, 0, 1)) for i in (2, 0, 1)
    )
    _assert_geometry(result)


def test_singleton_region_has_no_shape_and_no_protected_nonzero_vector():
    result = _observe(region=(2,))
    assert result.shape_dimension == result.rank == result.protected_dimension == 0
    assert result.input_labels == (
        "region_constant",
        "outside_0",
        "outside_1",
        "outside_3",
    )
    assert result.image_projection == result.protected_projection == ((0,),)
    assert result.protected_basis == result.protected_readout_rows == ()
    _assert_geometry(result)


def test_float_inputs_keep_binary64_values_and_report_is_immutable():
    result = _observe(((1, 0, 0, 0.1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    assert result.transition[0][3] == F.from_float(0.1)
    assert result.transition[0][3] != F(1, 10)
    with pytest.raises(FrozenInstanceError):
        result.rank = 0
    _assert_geometry(result)


@pytest.mark.parametrize(
    "region", [(), (0, 1, 2, 3), (0, 0), (-1,), (4,), (True,), (0.0,)]
)
def test_invalid_region_domains_are_rejected(region):
    with pytest.raises((TypeError, ValueError)):
        _observe(region=region)


@pytest.mark.parametrize(
    "metric",
    [
        (0, 2, 3, 5),
        (-1, 2, 3, 5),
        (1, float("nan"), 3, 5),
        (1, 2, float("inf"), 5),
        (1,),
        (1, 2, 3),
    ],
)
def test_metric_requires_positive_finite_full_domain(metric):
    with pytest.raises((TypeError, ValueError)):
        _observe(metric=metric)


@pytest.mark.parametrize(
    "transition",
    [
        (),
        ((1, 0), (0, 1)),
        ((1, 0, 0, 0), (0, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        ((float("nan"), 0, 0, 0), *IDENTITY[1:]),
        ((float("inf"), 0, 0, 0), *IDENTITY[1:]),
    ],
)
def test_transition_requires_finite_square_full_domain(transition):
    with pytest.raises((TypeError, ValueError)):
        _observe(transition)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("transition", {0: IDENTITY[0]}),
        ("transition", "identity"),
        ("region", {0, 1}),
        ("region", "01"),
        ("metric", {1, 2, 3, 5}),
        ("metric", "1235"),
    ],
)
def test_unordered_or_string_container_inputs_are_rejected(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        _observe(**{keyword: value})
