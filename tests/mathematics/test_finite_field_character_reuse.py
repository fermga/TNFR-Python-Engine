"""Local character reuse keeps period arithmetic, presentations and dispatch."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tnfr.mathematics import finite_fields as ff

_TRACE_CODE = ff.FiniteField.trace.__code__


def _direct_periods(field, k):
    support = field.kth_power_set(k)
    inverse_size = 1.0 / len(support)
    return [
        inverse_size * sum(ff.additive_character(field, a, s) for s in support)
        for a in field.elements()
    ]


def _assert_identical_components(actual, expected):
    actual_bits = np.asarray(actual, dtype=np.complex128).view(np.uint64)
    expected_bits = np.asarray(expected, dtype=np.complex128).view(np.uint64)
    np.testing.assert_array_equal(actual_bits, expected_bits)


def _count_trace_calls(function):
    previous = sys.getprofile()
    calls = 0

    def observe(frame, event, _argument):
        nonlocal calls
        if event == "call" and frame.f_code is _TRACE_CODE:
            calls += 1

    try:
        sys.setprofile(observe)
        result = function()
    finally:
        sys.setprofile(previous)
    return result, calls


@pytest.mark.parametrize(
    "p,f,k,modulus",
    [
        (2, 1, 1, None),
        (7, 1, 2, None),
        (7, 1, 6, None),
        (3, 2, 2, [1, 0, 1]),
        (3, 2, 2, [2, 1, 1]),
        (2, 3, 2, [1, 0, 1, 1]),
        (2, 3, 2, [1, 1, 0, 1]),
        (5, 3, 2, None),
    ],
)
def test_period_components_are_bit_identical_to_direct_character_sums(p, f, k, modulus):
    field = ff.FiniteField(p, f, modulus=modulus)
    _assert_identical_components(
        ff.normalized_periods(field, k), _direct_periods(field, k)
    )


def test_repeated_product_traces_are_evaluated_once_per_field_element():
    field = ff.FiniteField(7, 2)
    direct, direct_calls = _count_trace_calls(lambda: _direct_periods(field, 2))
    optimized, optimized_calls = _count_trace_calls(
        lambda: ff.normalized_periods(field, 2)
    )
    assert direct_calls == field.q * len(field.kth_power_set(2)) == 1176
    assert optimized_calls == field.q == 49
    _assert_identical_components(optimized, direct)


def test_next_call_rebuilds_characters_after_in_place_presentation_change():
    field = ff.FiniteField(3, 2, modulus=[1, 0, 1])
    before = ff.normalized_periods(field, 2)
    field.modulus[:] = [2, 1, 1]
    actual = ff.normalized_periods(field, 2)
    alternate = ff.FiniteField(3, 2, modulus=[2, 1, 1])
    expected = _direct_periods(alternate, 2)
    assert not np.allclose(before, expected)
    _assert_identical_components(actual, expected)
    actual[0] = 99j
    _assert_identical_components(ff.normalized_periods(field, 2), expected)


@pytest.mark.parametrize("replacement", ["instance", "class"])
@pytest.mark.parametrize(
    "name",
    [
        "_element_argument",
        "_to_list",
        "_to_int",
        "add",
        "mul",
        "power",
        "trace",
        "elements",
        "kth_power_set",
    ],
)
def test_customized_arithmetic_and_iteration_keep_original_dispatch(
    name, replacement, monkeypatch
):
    field = ff.FiniteField(2, 2)
    original = getattr(ff.FiniteField, name)
    observed = []

    def replacement_method(self, *args, **kwargs):
        observed.append(args)
        return original(self, *args, **kwargs)

    if replacement == "class":
        monkeypatch.setattr(ff.FiniteField, name, replacement_method)
    else:
        monkeypatch.setattr(
            field,
            name,
            lambda *args, **kwargs: replacement_method(field, *args, **kwargs),
        )
    actual, trace_calls = _count_trace_calls(lambda: ff.normalized_periods(field, 1))
    assert trace_calls == 12  # Four characters over a three-element support.
    assert observed
    _assert_identical_components(actual, _direct_periods(field, 1))


def test_subclasses_retain_the_public_per_pair_evaluation_path():
    class FieldWithTraceHook(ff.FiniteField):
        def trace(self, value):
            return super().trace(value)

    field = FieldWithTraceHook(2, 2)
    actual, trace_calls = _count_trace_calls(lambda: ff.normalized_periods(field, 1))
    assert trace_calls == 12
    _assert_identical_components(actual, _direct_periods(field, 1))


def test_replaced_public_character_is_applied_to_original_pairs(monkeypatch):
    field = ff.FiniteField(2, 2)
    original = ff.additive_character
    pairs = []

    def replacement(field, a, x):
        pairs.append((a, x))
        return (1 + 0.1 * a) * original(field, a, x) + 0.01j * a

    monkeypatch.setattr(ff, "additive_character", replacement)
    actual = ff.normalized_periods(field, 1)
    assert pairs == [(a, s) for a in field.elements() for s in field.kth_power_set(1)]
    _assert_identical_components(actual, _direct_periods(field, 1))


@pytest.mark.parametrize("invalid", [True, np.bool_(False), "2", 2.0, 0, -1])
def test_local_reuse_does_not_bypass_power_set_admission(invalid):
    with pytest.raises((TypeError, ValueError)):
        ff.normalized_periods(ff.FiniteField(3, 2), invalid)
