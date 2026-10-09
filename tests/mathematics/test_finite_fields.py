r"""Tests for the R5 finite-field residue networks.

Prime fields reproduce R2 exactly (``distinct = gcd(k, p−1) + 1``); extensions
show trace collisions (``distinct ≤ gcd(k, q−1) + 1``, strict for some cases).
The trace additive character sum and the explicit additive Cayley spectrum must
agree.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import pytest

from tnfr.mathematics import finite_fields as ff
from tnfr.mathematics.finite_fields import (
    FiniteField,
    additive_character,
    cyclotomic_period_count,
    distinct_period_count,
    explicit_cayley_spectrum_count,
    gauss_period,
    presentation_isomorphism,
    prime_field_matches_cyclotomy,
)

# Distinct small extension supports (p, f, k). Powers with the same gcd(k,q-1)
# generate the same subgroup, so redundant copies are omitted. The table keeps
# characteristic two, degrees two/three and strict trace-collision controls.
EXTENSION_COUNTS = {
    (2, 2, 2): 2,
    (2, 2, 3): 2,
    (2, 3, 2): 2,
    (3, 2, 2): 3,
    (3, 2, 3): 2,
    (3, 2, 4): 2,
    (3, 3, 2): 3,
    (3, 3, 3): 2,
    (5, 2, 2): 3,
    (5, 2, 3): 3,
    (5, 2, 4): 5,
    (7, 2, 2): 3,
    (7, 2, 3): 4,
    (7, 2, 4): 3,
}


# --------------------------------------------------------------------------- #
# Field arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p,f", [(5, 1), (2, 2), (2, 3), (3, 2), (3, 3), (5, 2), (7, 2)]
)
def test_field_cardinality_and_modulus(p, f):
    F = FiniteField(p, f)
    assert F.q == p**f
    assert len(F.modulus) == f + 1 or f == 1


@pytest.mark.parametrize("p,f", [(2, 2), (3, 2), (3, 3), (5, 2)])
def test_arithmetic_ring_axioms(p, f):
    F = FiniteField(p, f)
    els = list(F.elements())
    zero, one = 0, F.one
    for a in els[:8]:
        assert F.add(a, zero) == a
        assert F.mul(a, one) == a
        for b in els[:8]:
            assert F.add(a, b) == F.add(b, a)
            assert F.mul(a, b) == F.mul(b, a)


@pytest.mark.parametrize(
    "p,scalar_type",
    [(50021, np.int32), (4294967291, np.int64), (4294967291, np.uint32)],
)
def test_prime_field_numpy_elements_do_not_overflow(p, scalar_type):
    field = FiniteField(p)
    minus_one = scalar_type(p - 1)
    with np.errstate(over="raise"):
        results = (
            field.mul(minus_one, minus_one),
            field.add(minus_one, minus_one),
            field.power(minus_one, np.int64(2)),
            field.trace(minus_one),
        )
    assert results == (1, p - 2, 1, p - 1)
    assert all(type(value) is int for value in results)


def test_extension_field_numpy_coefficients_preserve_exact_arithmetic():
    # In F_257[x]/(x²+3), x²=-3 and Tr(-1)=-2 in the base field.
    field = FiniteField(257, 2, modulus=[3, 0, 1])
    minus_one = np.int16(256)
    generator = np.int16(257)
    with np.errstate(over="raise"):
        assert field.mul(minus_one, minus_one) == 1
        assert field.mul(generator, generator) == 254
        assert field.trace(minus_one) == 255
        assert field.power(generator, np.int32(field.q - 1)) == 1


@pytest.mark.parametrize("operation", ["add", "mul", "power", "trace"])
@pytest.mark.parametrize("invalid", [True, np.bool_(False), 2.0, "2"])
def test_public_field_arithmetic_rejects_noninteger_elements(operation, invalid):
    field = FiniteField(5)
    args = (invalid,) if operation == "trace" else (invalid, 1)
    with pytest.raises(TypeError, match="integer"):
        getattr(field, operation)(*args)
    if operation in ("add", "mul"):
        with pytest.raises(TypeError, match="integer"):
            getattr(field, operation)(1, invalid)


@pytest.mark.parametrize("operation", ["add", "mul", "power", "trace"])
@pytest.mark.parametrize("invalid", [-1, 5])
def test_field_arithmetic_requires_the_declared_element_encoding(operation, invalid):
    field = FiniteField(5)
    args = (invalid,) if operation == "trace" else (invalid, 1)
    with pytest.raises(ValueError, match="q-1"):
        getattr(field, operation)(*args)


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 2.0, "2"])
def test_field_exponents_require_genuine_integers(invalid):
    field = FiniteField(5)
    with pytest.raises(TypeError, match="integer"):
        field.power(2, invalid)
    with pytest.raises(TypeError, match="integer"):
        field.kth_power_set(invalid)


def test_zero_and_negative_field_exponent_boundaries():
    field = FiniteField(5)
    assert field.power(np.int64(0), np.int64(0)) == 1
    assert field.power(np.int64(2), np.int64(4)) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        field.power(2, -1)


@pytest.mark.parametrize("p,f", [(2, 2), (3, 2), (5, 2), (2, 3), (3, 3)])
def test_multiplicative_group_is_cyclic_order_q_minus_1(p, f):
    F = FiniteField(p, f)
    # some element has multiplicative order exactly q-1 (a generator exists)
    orders = []
    for a in range(1, F.q):
        o, x = 1, a
        while x != F.one:
            x = F.mul(x, a)
            o += 1
        orders.append(o)
    assert max(orders) == F.q - 1
    assert all((F.q - 1) % o == 0 for o in orders)


@pytest.mark.parametrize("p,f,k", [(2, 2, 2), (3, 2, 2), (5, 2, 3), (7, 2, 4)])
def test_kth_power_set_size(p, f, k):
    F = FiniteField(p, f)
    assert len(F.kth_power_set(k)) == (F.q - 1) // gcd(k, F.q - 1)


# --------------------------------------------------------------------------- #
# Trace and additive characters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,f", [(2, 2), (3, 2), (3, 3), (5, 2), (7, 2)])
def test_trace_is_additive_and_in_prime_field(p, f):
    F = FiniteField(p, f)
    for a in range(min(F.q, 12)):
        assert 0 <= F.trace(a) < p
        for b in range(min(F.q, 12)):
            assert (F.trace(F.add(a, b))) % p == (F.trace(a) + F.trace(b)) % p
    assert F.trace(0) == 0


def test_additive_character_is_unimodular():
    F = FiniteField(3, 2)
    for a in range(F.q):
        for x in range(F.q):
            assert abs(abs(additive_character(F, a, x)) - 1.0) < 1e-12


def test_gauss_period_at_zero_is_set_size():
    F = FiniteField(5, 2)
    S = F.kth_power_set(2)
    assert abs(gauss_period(F, 2, 0) - len(S)) < 1e-9


# --------------------------------------------------------------------------- #
# Required test 1: prime-field regression (R2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,k", [(5, 1), (7, 2), (7, 3), (13, 4)])
def test_prime_field_reproduces_cyclotomy(p, k):
    F = FiniteField(p, 1)
    assert distinct_period_count(F, k) == cyclotomic_period_count(p, k)


def test_prime_field_comparison_wrapper():
    assert prime_field_matches_cyclotomy(7, 3)


# --------------------------------------------------------------------------- #
# Required test 2: extension-field exact small cases
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=list(EXTENSION_COUNTS))
def extension_periods(request):
    p, f, k = request.param
    field = FiniteField(p, f)
    return request.param, field, distinct_period_count(field, k)


def test_extension_field_counts_and_collision_boundary(extension_periods):
    key, field, count = extension_periods
    assert count == EXTENSION_COUNTS[key]
    bound = cyclotomic_period_count(field.q, key[2])
    assert count <= bound
    if key in {(2, 2, 3), (3, 2, 4), (5, 2, 3), (7, 2, 4)}:
        # Independent negative controls: the prime-field equality fails here.
        assert count < bound


def test_period_and_explicit_spectrum_agree(extension_periods):
    key, field, count = extension_periods
    assert count == explicit_cayley_spectrum_count(field, key[2])


# --------------------------------------------------------------------------- #
# Guards and exports
# --------------------------------------------------------------------------- #
def test_degree_four_unsupported():
    with pytest.raises(NotImplementedError):
        FiniteField(2, 4)


@pytest.mark.parametrize("bad", [(1, 1), (5, 0)])
def test_invalid_field_parameters(bad):
    with pytest.raises(ValueError):
        FiniteField(*bad)


@pytest.mark.parametrize("p", [4, 6, 9, 15, 25, 341, 561])
def test_composite_characteristic_is_rejected_before_field_construction(p):
    with pytest.raises(ValueError, match="prime"):
        FiniteField(p)


@pytest.mark.parametrize("parameter", ["p", "f"])
@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 2.0, 2.5, "2", float("nan"), float("inf")]
)
def test_field_parameters_require_genuine_integers(parameter, value):
    arguments = {"p": 3, "f": 1, parameter: value}
    with pytest.raises(TypeError, match="integer"):
        FiniteField(**arguments)


def test_numpy_integer_field_parameters_are_normalized():
    field = FiniteField(
        np.int64(3), np.int64(2), modulus=[np.int64(1), np.int64(0), np.int64(1)]
    )
    assert type(field.p) is type(field.f) is type(field.q) is int
    assert all(type(value) is int for value in field.modulus)
    for element in range(1, field.q):
        assert field.power(element, field.q - 1) == field.one


@pytest.mark.parametrize("coefficient", [True, np.bool_(False), 1.5, "1"])
def test_modulus_coefficients_cannot_be_truncated_or_boolean(coefficient):
    with pytest.raises(TypeError, match="modulus coefficient"):
        FiniteField(3, 2, modulus=[coefficient, 0, 1])


def test_characteristic_admission_uses_the_exact_fallback_without_sympy(monkeypatch):
    from tnfr.mathematics import number_theory

    monkeypatch.setattr(number_theory, "HAS_SYMPY", False)
    with pytest.raises(ValueError, match="prime"):
        FiniteField(341)
    field = FiniteField(7, 2)
    assert all(field.power(element, 48) == 1 for element in range(1, 49))


def test_invalid_alternate_modulus_is_rejected():
    with pytest.raises(ValueError):
        FiniteField(3, 2, modulus=[0, 0, 1])


def test_alternate_presentations_have_an_explicit_isomorphism():
    first = FiniteField(2, 3, modulus=[1, 0, 1, 1])
    second = FiniteField(2, 3, modulus=[1, 1, 0, 1])
    mapping = presentation_isomorphism(first, second)
    assert mapping[0] == 0
    assert mapping[first.one] == second.one
    assert len(set(mapping)) == first.q


def test_kth_power_rejects_nonpositive():
    with pytest.raises(ValueError):
        FiniteField(5, 1).kth_power_set(0)


def test_module_exports_complete():
    expected = {
        "FiniteField",
        "additive_character",
        "gauss_period",
        "normalized_periods",
        "distinct_period_count",
        "cyclotomic_period_count",
        "prime_field_matches_cyclotomy",
        "explicit_cayley_spectrum_count",
    }
    assert expected <= set(ff.__all__)
