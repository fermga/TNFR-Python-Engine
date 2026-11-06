"""Independent polynomial identities and admission for Chebyshev intervals."""

from fractions import Fraction as Q
from itertools import product
from random import Random

import pytest

import tnfr.mathematics._interval_chebyshev as owner
from tnfr.mathematics._rational_interval import I


def _direct_product(left, right):
    """Accumulate T_j*T_k=(T_(j+k)+T_|j-k|)/2 over exact coefficients."""
    result = [Q(0)] * (len(left) + len(right) - 1)
    for j, first in enumerate(left):
        for k, second in enumerate(right):
            term = Q(first) * Q(second) / 2
            result[j + k] += term
            result[abs(j - k)] += term
    return tuple(result)


def _ordinary_convolution(left, right):
    result = [0] * (len(left) + len(right) - 1)
    for j, first in enumerate(left):
        for k, second in enumerate(right):
            result[j + k] += first * second
    return tuple(result)


def _power_coefficients(chebyshev):
    """Build T_n in the power basis by its defining three-term identity."""
    bases = [(Q(1),)]
    if len(chebyshev) > 1:
        bases.append((Q(0), Q(1)))
    for degree in range(2, len(chebyshev)):
        row = [Q(0)] + [2 * value for value in bases[-1]]
        for index, value in enumerate(bases[-2]):
            row[index] -= value
        bases.append(tuple(row))
    result = [Q(0)] * len(chebyshev)
    for coefficient, basis in zip(chebyshev, bases):
        for index, value in enumerate(basis):
            result[index] += coefficient * value
    return tuple(result)


def _assert_exact(coefficients, expected):
    assert tuple(coefficients) == tuple(I(value) for value in expected)


@pytest.mark.parametrize("j,k", [(0, 0), (0, 7), (6, 0), (1, 1), (2, 5), (7, 7)])
def test_signed_basis_products_include_constant_mode(j, k):
    left, right = [0] * (j + 1), [0] * (k + 1)
    left[j], right[k] = -3, 2
    _assert_exact(owner.multiply(left, right), _direct_product(left, right))


def test_random_integer_polynomials_match_direct_dense_product():
    rng = Random(741003)
    for _ in range(32):
        left = tuple(rng.randrange(-19, 20) for _ in range(rng.randrange(1, 13)))
        right = tuple(rng.randrange(-19, 20) for _ in range(rng.randrange(1, 13)))
        expected = _direct_product(left, right)
        _assert_exact(owner.multiply(left, right), expected)
        _assert_exact(owner.multiply(right, left), expected)


def test_interval_product_encloses_every_endpoint_combination():
    left = (I(-3, 1), I(Q(-1, 2), 2), I(-2, -1))
    right = (I(-1, 4), I(Q(1, 4), Q(3, 2)))
    result = owner.multiply(left, right)
    for first in product(*((value.lo, value.hi) for value in left)):
        for second in product(*((value.lo, value.hi) for value in right)):
            expected = _direct_product(first, second)
            assert all(bound.contains(value) for bound, value in zip(result, expected))


def test_non_dyadic_product_contains_exact_rational_result():
    left, right = (Q(1, 3), Q(-5, 7), Q(2, 9)), (Q(-7, 11), Q(3, 13))
    result = owner.multiply(left, right)
    expected = _direct_product(left, right)
    assert all(bound.contains(value) for bound, value in zip(result, expected))
    assert max(bound.width for bound in result) < Q(1, 2**115)


def test_mixed_orders_and_trailing_zero_lengths_are_preserved():
    _assert_exact(owner.add((2, -3, 0, 0), (-2,)), (0, -3, 0, 0))
    _assert_exact(owner.subtract((2,), (2, -3, 0)), (0, 3, 0))
    _assert_exact(owner.scale((1, -2, 0, 0), -2), (-2, 4, 0, 0))
    _assert_exact(owner.multiply((1, 0, 0), (2, 0)), (2, 0, 0, 0))
    _assert_exact(owner.derivative((7, 0, 0, 0)), (0, 0, 0))
    _assert_exact(owner.derivative((7,)), (0,))


def test_exact_signed_cancellation_survives_packed_arithmetic():
    _assert_exact(owner.multiply((1, 1), (1, -1)), (Q(1, 2), 0, Q(-1, 2)))
    large = 2**160 + 1
    values = (large, -large, 2, -2, 0)
    _assert_exact(owner.subtract(values, values), (0,) * len(values))
    expected = _direct_product(values, (-large, large, -1))
    _assert_exact(owner.multiply(values, (-large, large, -1)), expected)


@pytest.mark.parametrize(
    "left,right",
    [
        ((-1,), (-1,)),
        ((0, 0, 0), (-2, 3)),
        ((-9, -9, -9, -9), (7, -8, 0, -1)),
        ((-(2**130) - 1, 0, 2**131 - 1), (2**129 + 3, -(2**130), -1)),
    ],
)
def test_signed_integer_convolution_retains_negative_carrying(left, right):
    assert tuple(owner._integer_convolution(left, right)) == _ordinary_convolution(
        left, right
    )


def test_unsigned_packing_retains_internal_zero_digits_and_large_carries():
    left = (2**140 - 1, 0, 1, 2**139 + 1, 0)
    right = (2**141 + 3, 0, 2**140 - 1)
    assert tuple(owner._nonnegative_convolution(left, right)) == _ordinary_convolution(
        left, right
    )


def test_unsigned_packing_bound_at_exact_power_of_two_keeps_carry_separate():
    # The middle coefficient is exactly 2**16 and needs a third base256 byte.
    left, right = (128, 128), (256, 256)
    assert tuple(owner._nonnegative_convolution(left, right)) == (2**15, 2**16, 2**15)


def test_derivative_matches_power_basis_differentiation():
    rng = Random(91381)
    for degree in range(1, 14):
        coefficients = tuple(rng.randrange(-11, 12) for _ in range(degree + 1))
        powers = _power_coefficients(coefficients)
        expected = tuple(index * powers[index] for index in range(1, len(powers)))
        actual = owner.derivative(coefficients)
        assert all(value.lo == value.hi for value in actual)
        assert _power_coefficients(tuple(value.lo for value in actual)) == expected


def test_interval_derivative_encloses_all_coefficient_corners():
    coefficients = (I(-2, 1), I(-1, 3), I(-4, -2), I(Q(1, 4), Q(3, 4)))
    result = owner.derivative(coefficients)
    for point in product(*((value.lo, value.hi) for value in coefficients)):
        # T_1'=T_0, T_2'=4T_1, T_3'=3T_0+6T_2.
        expected = (point[1] + 3 * point[3], 4 * point[2], 6 * point[3])
        assert all(bound.contains(value) for bound, value in zip(result, expected))


def test_maximum_factor_orders_preserve_full_product_without_truncation():
    left = tuple((index % 7) - 3 for index in range(193))
    right = tuple(2 - (index % 5) for index in range(193))
    result = owner.multiply(left, right)
    assert len(result) == 385
    _assert_exact(result, _direct_product(left, right))
    _assert_exact(
        owner.derivative((0,) * 384 + (1,)),
        tuple(768 if index % 2 else 0 for index in range(384)),
    )


def test_admission_detaches_input_and_preserves_represented_reals():
    values = [Q(1, 3), 0.1, Q(-1, 2**300)]
    admitted = owner.admit_coefficients(values)
    values[0] = 9
    assert admitted[0].contains(Q(1, 3))
    assert not admitted[0].contains(9)
    assert admitted[1].contains(Q.from_float(0.1))
    assert admitted[2].contains(Q(-1, 2**300))
    assert admitted[2] != I(0)


@pytest.mark.parametrize("values", [(), [], "1", b"1", {0: 1}, {1}, iter((1,))])
def test_nonempty_ordered_indexed_sequences_are_required(values):
    with pytest.raises((TypeError, ValueError)):
        owner.admit_coefficients(values)


@pytest.mark.parametrize(
    "value", [True, float("nan"), float("inf"), -float("inf"), "1"]
)
@pytest.mark.parametrize(
    "operation", ["admit", "add", "subtract", "scale", "derivative", "multiply"]
)
def test_invalid_coefficients_reject_before_polynomial_arithmetic(value, operation):
    calls = {
        "admit": lambda: owner.admit_coefficients((value,)),
        "add": lambda: owner.add((1,), (value,)),
        "subtract": lambda: owner.subtract((value,), (1,)),
        "scale": lambda: owner.scale((value,), 2),
        "derivative": lambda: owner.derivative((value,)),
        "multiply": lambda: owner.multiply((0,), (value,)),
    }
    with pytest.raises((TypeError, ValueError)):
        calls[operation]()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_invalid_scale_factor_rejects_even_for_zero_polynomial(value):
    with pytest.raises((TypeError, ValueError)):
        owner.scale((0, 0), value)


@pytest.mark.parametrize("bounds", [(True, 1), (0, float("nan")), (2, 1)])
def test_existing_interval_endpoints_are_re_admitted(bounds):
    forged = object.__new__(I)
    object.__setattr__(forged, "lo", bounds[0])
    object.__setattr__(forged, "hi", bounds[1])
    with pytest.raises((TypeError, ValueError)):
        owner.multiply((0,), (forged,))
    with pytest.raises((TypeError, ValueError)):
        owner.scale((0,), forged)


@pytest.mark.parametrize("degree", [True, 1.0, -1, 385])
def test_degree_policy_requires_an_ordinary_bounded_integer(degree):
    with pytest.raises((TypeError, ValueError)):
        owner.admit_coefficients((1,), max_degree=degree)


def test_degree_caps_apply_even_to_zero_or_trailing_zero_inputs():
    with pytest.raises(ValueError):
        owner.admit_coefficients((0,) * 386)
    with pytest.raises(ValueError):
        owner.admit_coefficients((1, 0), max_degree=0)
    with pytest.raises(ValueError):
        owner.multiply((0,) * 194, (0,))
    with pytest.raises(ValueError):
        owner.multiply((0,), (0,) * 194)


def test_packing_work_cap_rejects_before_integer_allocation(monkeypatch):
    monkeypatch.setattr(owner, "MAX_PACKED_BITS", 16)
    with pytest.raises(ValueError, match="pack|bit|work"):
        owner._integer_convolution((2**40, 1), (-(2**40), -1))


@pytest.mark.parametrize("degree", [1, 2, 3, 6, 95, 96])
def test_lobatto_nodes_retain_exact_endpoints_symmetry_and_order(degree):
    nodes = owner.lobatto_nodes(degree)
    assert len(nodes) == degree + 1
    assert nodes[0] == I(1) and nodes[-1] == I(-1)
    assert all(left.lo > right.hi for left, right in zip(nodes, nodes[1:]))
    assert nodes == tuple(-value for value in reversed(nodes))
    assert max(value.width for value in nodes) < Q(1, 2**120)
    if degree % 2 == 0:
        assert nodes[degree // 2] == I(0)


def test_lobatto_angles_match_independent_algebraic_cosine_identities():
    fourths, sixths = owner.lobatto_nodes(4), owner.lobatto_nodes(6)
    # cos(pi/4)^2=1/2, cos(pi/6)^2=3/4, cos(pi/3)=1/2.
    assert (fourths[1] ** 2).contains(Q(1, 2))
    assert (sixths[1] ** 2).contains(Q(3, 4))
    assert sixths[2].contains(Q(1, 2))
    # The complementary sine is the reflected positive cosine on this grid.
    assert (sixths[1] ** 2 + sixths[2] ** 2).contains(1)


@pytest.mark.parametrize("degree", [1, 2, 7, 96])
def test_lobatto_constant_and_highest_mode_have_unhalved_coefficients(degree):
    rows = ((3,) * (degree + 1), tuple((-1) ** j for j in range(degree + 1)))
    constant, highest = owner.interpolate_lobatto_rows(rows, degree=degree)
    expected_constant, expected_highest = [0] * (degree + 1), [0] * (degree + 1)
    expected_constant[0], expected_highest[-1] = 3, 1
    for actual, expected in (
        (constant, expected_constant),
        (highest, expected_highest),
    ):
        assert all(bound.contains(value) for bound, value in zip(actual, expected))
        assert max(value.width for value in actual) < Q(1, 2**115)


def test_lobatto_all_small_basis_modes_match_polynomial_recurrence():
    degree = 8
    nodes = owner.lobatto_nodes(degree)
    rows = [(I(1),) * (degree + 1), nodes]
    for _ in range(2, degree + 1):
        rows.append(tuple(2 * x * a - b for x, a, b in zip(nodes, rows[-1], rows[-2])))
    result = owner.interpolate_lobatto_rows(rows, degree=degree)
    for mode, coefficients in enumerate(result):
        assert all(
            value.contains(int(index == mode))
            for index, value in enumerate(coefficients)
        )
        assert max(value.width for value in coefficients) < Q(1, 2**110)


def test_full_degree_lobatto_recovers_signed_polynomial_and_endpoints():
    nodes = owner.lobatto_nodes(96)
    # Independent power forms of T_3 and T_5, with a signed mixed polynomial.
    samples = tuple(
        2 - 3 * (4 * x**3 - 3 * x) + 5 * (16 * x**5 - 20 * x**3 + 5 * x) for x in nodes
    )
    coefficients = owner.interpolate_lobatto_rows((samples,))[0]
    expected = {0: 2, 3: -3, 5: 5}
    assert len(coefficients) == 97
    assert all(
        value.contains(expected.get(index, 0))
        for index, value in enumerate(coefficients)
    )
    assert sum(coefficients, I(0)).contains(4)
    assert sum(
        ((-1) ** index * value for index, value in enumerate(coefficients)), I(0)
    ).contains(0)
    assert max(value.width for value in coefficients) < Q(1, 2**110)


def test_lobatto_interval_transform_encloses_all_sample_corners():
    samples = (I(-3, 1), I(2, 4), I(-2, 3))
    result = owner.interpolate_lobatto_rows((samples,), degree=2)[0]
    for first, middle, final in product(*((value.lo, value.hi) for value in samples)):
        # Solve the three interpolation equations at +1,0,-1 directly.
        expected = (
            (first + 2 * middle + final) / 4,
            (first - final) / 2,
            (first - 2 * middle + final) / 4,
        )
        assert all(bound.contains(value) for bound, value in zip(result, expected))
    assert tuple(result) == (I(Q(-1, 4), 3), I(-3, Q(3, 2)), I(Q(-13, 4), 0))


def test_lobatto_signed_integer_accumulation_preserves_large_cancellation():
    large = 2**200 + 7
    coefficients = owner.interpolate_lobatto_rows(((large, 0, -large),), degree=2)[0]
    _assert_exact(coefficients, (0, large, 0))
    _assert_exact(owner.interpolate_lobatto_rows(((0, 0),), degree=1)[0], (0, 0))


def test_lobatto_retains_represented_and_subgrid_sample_uncertainty():
    tiny = Q(1, 2**300)
    first, second = owner.interpolate_lobatto_rows(((0.1, -0.1), (tiny, 0)), degree=1)
    assert first[0].contains(0)
    assert first[1].contains(Q.from_float(0.1))
    assert all(value.contains(tiny / 2) for value in second)
    assert all(value != I(0) for value in second)


def test_lobatto_samples_alone_do_not_bound_between_node_error():
    # T_6 is one at degree-three Lobatto nodes but T_6(0)=-1. Its interpolant
    # therefore cannot certify its approximation error from sampled data.
    coefficients = owner.interpolate_lobatto_rows(((1,) * 4,), degree=3)[0]
    interpolant_at_zero = sum(
        (
            value * (0 if index % 2 else (-1) ** (index // 2))
            for index, value in enumerate(coefficients)
        ),
        I(0),
    )
    assert interpolant_at_zero.contains(1)
    assert not interpolant_at_zero.contains(-1)
    assert _power_coefficients((0,) * 6 + (1,))[0] == -1


def test_lobatto_detaches_samples_and_retains_immutable_cached_nodes():
    samples = [[1, -1]]
    result = owner.interpolate_lobatto_rows(samples, degree=1)
    samples[0][0] = 999
    _assert_exact(result[0], (0, 1))
    nodes = owner.lobatto_nodes(1)
    assert isinstance(result, tuple) and isinstance(result[0], tuple)
    with pytest.raises((AttributeError, TypeError)):
        nodes[0].lo = Q(9)
    assert owner.lobatto_nodes(1) == (I(1), I(-1))


@pytest.mark.parametrize("degree", [True, 1.0, 0, -1, 97])
def test_lobatto_degree_is_admitted_before_cache_access(degree, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid degree reached cache or arithmetic")

    monkeypatch.setattr(owner, "_lobatto_nodes_admitted", forbidden)
    monkeypatch.setattr(owner, "_lobatto_weights_admitted", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.lobatto_nodes(degree)
    with pytest.raises((TypeError, ValueError)):
        owner.interpolate_lobatto_rows(((0, 0),), degree=degree)


@pytest.mark.parametrize(
    "value", [True, float("nan"), float("inf"), -float("inf"), "0"]
)
def test_lobatto_all_samples_admitted_before_any_transform(value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid last sample reached weights or arithmetic")

    monkeypatch.setattr(owner, "_lobatto_weights_admitted", forbidden)
    monkeypatch.setattr(owner, "_grid_bounds", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.interpolate_lobatto_rows(((0, 0), (0, value)), degree=1)


@pytest.mark.parametrize("bounds", [(True, 1), (0, float("nan")), (2, 1)])
def test_lobatto_re_admits_forged_interval_endpoints(bounds, monkeypatch):
    forged = object.__new__(I)
    object.__setattr__(forged, "lo", bounds[0])
    object.__setattr__(forged, "hi", bounds[1])

    def forbidden(*args, **kwargs):
        raise AssertionError("forged interval reached transform")

    monkeypatch.setattr(owner, "_lobatto_weights_admitted", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.interpolate_lobatto_rows(((0, forged),), degree=1)


@pytest.mark.parametrize(
    "samples",
    [
        (),
        "00",
        {0: (0, 0)},
        {(0, 0)},
        iter(((0, 0),)),
        ((0,),),
        ((0, 0, 0),),
        ({0: 0, 1: 0},),
        (iter((0, 0)),),
    ],
)
def test_lobatto_requires_bounded_indexed_rectangular_tables(samples):
    with pytest.raises((TypeError, ValueError)):
        owner.interpolate_lobatto_rows(samples, degree=1)


def test_lobatto_row_cap_applies_before_element_access():
    class Oversized:
        def __len__(self):
            return 3073

        def __getitem__(self, index):
            raise AssertionError("oversized table accessed an element")

    with pytest.raises(ValueError):
        owner.interpolate_lobatto_rows(Oversized(), degree=1)
    rows = owner.interpolate_lobatto_rows(((1, -1),) * 3072, degree=1)
    assert len(rows) == 3072 and all(row == (I(0), I(1)) for row in rows)
