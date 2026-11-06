"""Constructed arithmetic controls, with no source or family-response audit."""

import sys
from fractions import Fraction as Q
from itertools import product
from math import factorial, inf, nextafter

import pytest

from tnfr.research import sine_preparation_bounds as owner


def test_adjacent_hull_keeps_binade_asymmetry_and_negative_reflection():
    positive = owner.adjacent_coordinate_hull(1.0)
    negative = owner.adjacent_coordinate_hull(-1.0)
    assert positive.center == 1
    assert positive.lower == 1 - Q(1, 2**53)
    assert positive.upper == 1 + Q(1, 2**52)
    assert positive.lower_error == Q(1, 2**53)
    assert positive.upper_error == positive.radius == Q(1, 2**52)
    assert (negative.lower, negative.upper) == (-positive.upper, -positive.lower)
    assert negative.lower_error == positive.upper_error
    # The supplied family reaches actual neighbors, beyond midpoint tie cells.
    assert float(positive.lower) == nextafter(1.0, -inf)
    assert float(positive.upper) == nextafter(1.0, inf)


def test_zero_and_subnormal_hulls_keep_exact_width_below_interval_grid():
    tiny = Q(1, 2**1074)
    zero = owner.adjacent_coordinate_hull(0.0)
    smallest = owner.adjacent_coordinate_hull(nextafter(0.0, inf))
    assert (zero.center, zero.lower, zero.upper, zero.radius) == (0, -tiny, tiny, tiny)
    assert (smallest.center, smallest.lower, smallest.upper) == (tiny, 0, 2 * tiny)
    assert smallest.lower_error == smallest.upper_error == tiny


@pytest.mark.parametrize(
    "value",
    (
        True,
        1,
        Q(1),
        "1.0",
        -0.0,
        inf,
        -inf,
        float("nan"),
        sys.float_info.max,
        -sys.float_info.max,
    ),
)
def test_unadmitted_coordinate_centers_or_missing_finite_neighbors_reject(value):
    with pytest.raises((TypeError, ValueError)):
        owner.adjacent_coordinate_hull(value)


def test_ieee_precondition_is_explicit(monkeypatch):
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(ValueError, match="IEEE binary64"):
        owner.adjacent_coordinate_hull(1.0)


def test_ordered_coordinate_hulls_retain_all_supplied_positions():
    rows = owner.adjacent_coordinate_hulls(value for value in (2.0, -1.0, 0.0))
    assert tuple(row.center for row in rows) == (Q(2), Q(-1), Q(0))


@pytest.mark.parametrize("values", ((), {1.0, 2.0}, {"value": 1.0}, "1.0"))
def test_empty_or_unordered_coordinate_rows_reject(values):
    with pytest.raises((TypeError, ValueError)):
        owner.adjacent_coordinate_hulls(values)


def test_zero_comparison_preserves_constructed_dyadic_errors():
    result = owner.bound_two_block_errors(0, 0, Q(1, 8), Q(3, 8), 7, steps=3)
    assert result.endpoints == ((Q(1, 8), Q(3, 8)),) * 4
    assert result.step_size == Q(7, 3) and result.steps == 3


@pytest.mark.parametrize("A,B", ((3, 0), (0, 5)))
def test_nilpotent_comparison_matches_independent_affine_solution(A, B):
    U0, V0, T = Q(1, 8), Q(3, 8), Q(1, 2)
    result = owner.bound_two_block_errors(A, B, U0, V0, T, steps=4)
    for index, pair in enumerate(result.endpoints):
        t = index * T / 4
        assert pair == (U0 + A * t * V0, V0 + B * t * U0)


def test_positive_unequal_blocks_bound_their_known_exponential_eigenmode():
    # M=(0,3;1/3,0) has eigenpair (1,(3,1)). At T=1 its solution
    # is (3e,e). Positive Taylor terms give a strict lower bound; e<3
    # follows independently from n! >= 2**(n-1), strict for n>=4.
    result = owner.bound_two_block_errors(3, Q(1, 3), 3, 1, 1, steps=4)
    lower = sum((Q(1, factorial(n)) for n in range(13)), Q(0))
    assert 3 * lower < result.final[0] < 9
    assert lower < result.final[1] < 3
    assert all(
        earlier[0] <= later[0] and earlier[1] <= later[1]
        for earlier, later in zip(result.endpoints, result.endpoints[1:])
    )


def test_exact_requested_fixed_steps_reuse_shared_owner_without_adaptation(monkeypatch):
    calls = []
    original = owner.comparison_flow_upper

    def record(matrix, radius, duration):
        answer = original(matrix, radius, duration)
        calls.append((matrix, radius, duration, answer))
        return answer

    monkeypatch.setattr(owner, "comparison_flow_upper", record)
    result = owner.bound_two_block_errors(2, 1, 1, 2, 1, steps=3)
    assert len(calls) == 3 and len(result.endpoints) == 4
    for index, (matrix, initial, duration, answer) in enumerate(calls):
        assert matrix == ((Q(0), Q(2)), (Q(1), Q(0)))
        assert duration == Q(1, 3)
        assert initial == result.endpoints[index]
        assert answer == result.endpoints[index + 1]


def test_unsupported_step_work_rejects_before_any_kernel_call(monkeypatch):
    def forbidden(*args):
        raise AssertionError("work admission must precede propagation")

    monkeypatch.setattr(owner, "comparison_flow_upper", forbidden)
    with pytest.raises(ValueError, match="h\\*max"):
        owner.bound_two_block_errors(5, 0, 0, 0, 1, steps=4)


def test_shared_kernel_failure_is_not_retried(monkeypatch):
    calls = []

    def fail(matrix, initial, duration):
        calls.append(duration)
        if len(calls) == 2:
            raise ArithmeticError("constructed unavailable bound")
        return initial

    monkeypatch.setattr(owner, "comparison_flow_upper", fail)
    with pytest.raises(ArithmeticError, match="constructed unavailable"):
        owner.bound_two_block_errors(1, 1, 1, 1, 1, steps=4)
    assert calls == [Q(1, 4), Q(1, 4)]


@pytest.mark.parametrize("key", ("A", "B", "U0", "V0", "T"))
@pytest.mark.parametrize("value", (-1, 0.5, True))
def test_comparison_requires_nonnegative_exact_inputs(key, value):
    arguments = dict(A=1, B=1, U0=1, V0=1, T=1, steps=2)
    arguments[key] = value
    with pytest.raises((ValueError, TypeError)):
        owner.bound_two_block_errors(**arguments)


@pytest.mark.parametrize("steps", (True, 0, -1, 1.0, Q(1)))
def test_step_count_requires_positive_ordinary_integer(steps):
    with pytest.raises((ValueError, TypeError)):
        owner.bound_two_block_errors(0, 0, 1, 1, 1, steps=steps)


def test_zero_duration_still_obeys_the_fixed_call_contract():
    result = owner.bound_two_block_errors(7, 9, Q(1, 2), Q(1, 4), 0, steps=2)
    assert result.endpoints == ((Q(1, 2), Q(1, 4)),) * 3


def test_marked_squared_bounds_equal_independent_rectangle_corner_maxima():
    U, V, w, child_x, child_phase = Q(1, 5), Q(2, 7), Q(3, 2), Q(1, 13), Q(1, 17)
    connected = max(
        (tc - ta) ** 2 + w * (xc - xa) ** 2
        for xa, xc, ta, tc in product((-U, U), (-U, U), (-V, V), (-V, V))
    )
    cut = max(
        (tc - ta) ** 2 + w * (xc - xa) ** 2
        for xa, xc, ta, tc in product(
            (-U, U), (-child_x, child_x), (-V, V), (-child_phase, child_phase)
        )
    )
    assert owner.connected_marked_error_squared(U, V, w) == connected
    assert owner.cut_marked_error_squared(U, V, child_x, child_phase, w) == cut
    assert cut > owner.cut_marked_error_squared(U, V, 0, 0, w)


def test_squared_bound_keeps_subgrid_exact_rationals():
    tiny = Q(1, 2**1074)
    assert owner.connected_marked_error_squared(tiny, tiny, 2) == 12 * tiny**2
    assert owner.cut_marked_error_squared(0, 0, tiny, 2 * tiny, 2) == 6 * tiny**2


@pytest.mark.parametrize("weight", (0, -1, True, 1.0))
def test_marked_observation_weight_must_be_exact_and_positive(weight):
    with pytest.raises((ValueError, TypeError)):
        owner.connected_marked_error_squared(1, 1, weight)
    with pytest.raises((ValueError, TypeError)):
        owner.cut_marked_error_squared(1, 1, 1, 1, weight)


@pytest.mark.parametrize("position", range(4))
def test_cut_error_bounds_cannot_hide_negative_child_or_host_values(position):
    arguments = [Q(1)] * 4
    arguments[position] = Q(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        owner.cut_marked_error_squared(*arguments, 1)
