"""Exact odd-cycle critical closures and independently reduced Hessian signs."""

from fractions import Fraction as Q
from math import comb

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import phase_cycle_geometry as owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _old_c5_templates():
    # Independent compatibility oracle: the original fixed-five algorithm.
    options, currents, masks = [], [], []
    for mask in range(32):
        negative = mask.bit_count()
        for period in range(-1, 4):
            current = Q(2 * period - negative, 2 * (5 - 2 * negative))
            if abs(current) >= Q(1, 4):
                continue
            options.append(
                tuple(
                    Q(1, 2) - current if mask & (1 << j) else current for j in range(5)
                )
            )
            currents.append(current)
            masks.append(mask)
    return tuple(options), tuple(currents), tuple(masks)


def _grounded_signed_hessian(size, mask):
    # Assemble the complete edge quadratic form before removing the last
    # node's gauge. No production inertia or template fields are consumed.
    rows = [[Q(0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        j = (i + 1) % size
        sign = Q(-1 if mask & (1 << i) else 1)
        rows[i][i] += sign
        rows[j][j] += sign
        rows[i][j] -= sign
        rows[j][i] -= sign
    assert all(sum(row, Q(0)) == 0 for row in rows)
    return tuple(tuple(row[:-1]) for row in rows[:-1])


def _exact_congruence_inertia(matrix):
    """Independent rational symmetric elimination with 1x1 and 2x2 pivots."""
    work = [list(row) for row in matrix]
    positive = negative = zero = 0
    while work:
        size = len(work)
        pivot = next((i for i in range(size) if work[i][i]), None)
        if pivot is not None:
            indices = [pivot] + [i for i in range(size) if i != pivot]
            work = [[work[i][j] for j in indices] for i in indices]
            value = work[0][0]
            positive += int(value > 0)
            negative += int(value < 0)
            work = [
                [work[i][j] - work[i][0] * work[0][j] / value for j in range(1, size)]
                for i in range(1, size)
            ]
            continue
        pair = next(
            ((i, j) for i in range(size) for j in range(i + 1, size) if work[i][j]),
            None,
        )
        if pair is None:
            zero += size
            break
        indices = list(pair) + [i for i in range(size) if i not in pair]
        work = [[work[i][j] for j in indices] for i in indices]
        off_diagonal = work[0][1]
        # [[0,c],[c,0]] has one sign of each kind and inverse
        # [[0,1/c],[1/c,0]], independently of c's sign.
        positive += 1
        negative += 1
        work = [
            [
                work[i][j]
                - (work[i][0] * work[1][j] + work[i][1] * work[0][j]) / off_diagonal
                for j in range(2, size)
            ]
            for i in range(2, size)
        ]
    return positive, negative, zero


def test_c5_wrappers_preserve_every_historical_option_and_its_order():
    expected = _old_c5_templates()
    assert owner._c5_sine_templates() == expected
    assert owner._odd_cycle_sine_templates(5) == expected
    assert len(expected[0]) == 30
    for count in range(6):
        assert owner._c5_negative_phase_modes(count) == count - int(count >= 3)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_all_templates_close_are_unique_and_satisfy_the_full_cycle_current(size, mp):
    options, currents, masks = owner._odd_cycle_sine_templates(size)
    assert (
        len(options)
        == len(currents)
        == len(masks)
        == size * comb(size - 1, (size - 1) // 2)
    )
    assert len(set(tuple(value % 1 for value in row) for row in options)) == len(
        options
    )
    ordering = []
    for turns, current, mask in zip(options, currents, masks):
        period = sum(turns, Q(0))
        assert period.denominator == 1
        assert len(turns) == size
        assert abs(current) < Q(1, 4)
        ordering.append((mask, period))
        angles = tuple(2 * mp.pi * _mp(mp, value) for value in turns)
        edge_currents = tuple(mp.sin(value) for value in angles)
        # The actual nodal sine balance is each outgoing current minus
        # the preceding current, including the closing edge.
        assert max(
            abs(edge_currents[j] - edge_currents[j - 1]) for j in range(size)
        ) < mp.mpf("1e-80")
        for index, angle in enumerate(angles):
            assert (mp.cos(angle) < 0) == bool(mask & (1 << index))
    assert ordering == sorted(ordering)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_zero_current_retains_exactly_all_closing_zero_half_turn_patterns(size):
    options, currents, masks = owner._odd_cycle_sine_templates(size)
    zero_options = {
        mask: row
        for row, current, mask in zip(options, currents, masks)
        if current == 0
    }
    expected = {
        mask: tuple(Q(int(bool(mask & (1 << j))), 2) for j in range(size))
        for mask in range(1 << size)
        if mask.bit_count() % 2 == 0
    }
    assert zero_options == expected
    assert len(zero_options) == 1 << (size - 1)
    # An odd cycle of quarter turns cannot close, including either all-sine
    # extrema; a finite catalog must not silently round these into a branch.
    assert (Q(size, 4)).denominator != 1
    assert all(abs(value) != Q(1, 4) for row in options for value in row)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_hessian_inertia_matches_independent_exact_edge_congruence_for_every_mask(size):
    for mask in range(1 << size):
        actual = _exact_congruence_inertia(_grounded_signed_hessian(size, mask))
        negative = owner._odd_cycle_negative_phase_modes(size, mask.bit_count())
        assert actual == (size - 1 - negative, negative, 0)


@pytest.mark.parametrize("size", [True, False, Q(5), 5.0, "5", None])
def test_noninteger_cycle_sizes_reject_before_enumeration(size):
    with pytest.raises(TypeError):
        owner._odd_cycle_sine_templates(size)
    with pytest.raises(TypeError):
        owner._odd_cycle_negative_phase_modes(size, 0)


@pytest.mark.parametrize("size", [-3, 0, 1, 2, 4, 8])
def test_non_odd_cycle_sizes_reject(size):
    with pytest.raises(ValueError):
        owner._odd_cycle_sine_templates(size)
    with pytest.raises(ValueError):
        owner._odd_cycle_negative_phase_modes(size, 0)


@pytest.mark.parametrize("count", [True, False, Q(2), 2.0, "2", None])
def test_negative_edge_count_requires_an_ordinary_integer(count):
    with pytest.raises(TypeError):
        owner._odd_cycle_negative_phase_modes(9, count)


@pytest.mark.parametrize("count", [-1, 10])
def test_negative_edge_count_stays_within_support(count):
    with pytest.raises(ValueError):
        owner._odd_cycle_negative_phase_modes(9, count)
