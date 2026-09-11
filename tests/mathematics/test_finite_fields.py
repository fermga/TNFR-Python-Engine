r"""Tests for the R5 finite-field residue networks.

Prime fields reproduce R2 exactly (``distinct = gcd(k, p−1) + 1``); extensions
show trace collisions (``distinct ≤ gcd(k, q−1) + 1``, strict for some cases).
The trace additive character sum and the explicit additive Cayley spectrum must
agree.
"""

from __future__ import annotations

from math import gcd

import pytest

from tnfr.mathematics import finite_fields as ff
from tnfr.mathematics.finite_fields import (
    FiniteField,
    additive_character,
    cyclotomic_period_count,
    distinct_period_count,
    explicit_cayley_spectrum_count,
    gauss_period,
    prime_field_matches_cyclotomy,
    presentation_isomorphism,
)

PRIMES = [5, 7, 11, 13, 17]
POWERS = [1, 2, 3, 4]

# Exact measured distinct-period counts for small extensions (p, f, k).
EXTENSION_COUNTS = {
    (2, 2, 2): 2, (2, 2, 3): 2, (2, 2, 4): 2,
    (2, 3, 2): 2, (2, 3, 3): 2, (2, 3, 4): 2,
    (3, 2, 2): 3, (3, 2, 3): 2, (3, 2, 4): 2,
    (3, 3, 2): 3, (3, 3, 3): 2, (3, 3, 4): 3,
    (5, 2, 2): 3, (5, 2, 3): 3, (5, 2, 4): 5,
    (7, 2, 2): 3, (7, 2, 3): 4, (7, 2, 4): 3,
}


# --------------------------------------------------------------------------- #
# Field arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,f", [(5, 1), (2, 2), (2, 3), (3, 2), (3, 3),
                                 (5, 2), (7, 2)])
def test_field_cardinality_and_modulus(p, f):
    F = FiniteField(p, f)
    assert F.q == p ** f
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
@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("k", POWERS)
def test_prime_field_reproduces_cyclotomy(p, k):
    assert prime_field_matches_cyclotomy(p, k)
    F = FiniteField(p, 1)
    assert distinct_period_count(F, k) == cyclotomic_period_count(p, k)


# --------------------------------------------------------------------------- #
# Required test 2: extension-field exact small cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(EXTENSION_COUNTS))
def test_extension_field_exact_counts(key):
    p, f, k = key
    F = FiniteField(p, f)
    assert distinct_period_count(F, k) == EXTENSION_COUNTS[key]


@pytest.mark.parametrize("key", list(EXTENSION_COUNTS))
def test_period_and_explicit_spectrum_agree(key):
    p, f, k = key
    F = FiniteField(p, f)
    assert distinct_period_count(F, k) == explicit_cayley_spectrum_count(F, k)


@pytest.mark.parametrize("p,f,k", [(2, 2, 2), (3, 2, 2), (5, 2, 4), (7, 2, 2)])
def test_distinct_count_is_upper_bounded_by_cyclotomy(p, f, k):
    F = FiniteField(p, f)
    assert distinct_period_count(F, k) <= cyclotomic_period_count(F.q, k)


@pytest.mark.parametrize("p,f,k", [(2, 2, 3), (3, 2, 4), (5, 2, 3), (7, 2, 4)])
def test_extension_collisions_exist(p, f, k):
    # the prime-field formula does NOT transfer: strict drop for these cases.
    F = FiniteField(p, f)
    assert distinct_period_count(F, k) < cyclotomic_period_count(F.q, k)


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
