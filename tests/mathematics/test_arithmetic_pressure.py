r"""Tests for the R7 arithmetic-pressure independence and completeness audit.

Separates the three claims that are easy to conflate: primality sufficiency (each
channel is 0 iff prime, so the set is *redundant* / non-minimal for primality),
linear independence (rank 3, no affine relation, despite high correlation), and
structural completeness (unproven — the fourth-channel gate is closed by default).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from tnfr.mathematics import arithmetic_pressure as ap
from tnfr.mathematics.arithmetic_pressure import (
    FourthChannelCriteria,
    abundance_class,
    ablation_detects_primes,
    admits_fourth_channel,
    all_channels_sufficient,
    arithmetic_pressure,
    big_omega,
    channel_abundance,
    channel_correlations,
    channel_divisor,
    channel_factorization,
    channel_rank,
    channels,
    channels_nonnegative,
    completeness_proven,
    divisor_sum,
    factor_class,
    has_linear_relation,
    is_redundant_for_primality,
    minimal_channels_for_primality,
    num_divisors,
    pressure_by_class,
    pressure_zero_iff_prime,
    primes_in_range,
)
from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRParameters,
)


# --------------------------------------------------------------------------- #
# Required test 1: exact invariant values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,omega,tau,sigma", [
    (2, 1, 2, 3), (4, 2, 3, 7), (6, 2, 4, 12), (12, 3, 6, 28),
    (16, 4, 5, 31), (30, 3, 8, 72), (36, 4, 9, 91),
])
def test_exact_arithmetic_functions(n, omega, tau, sigma):
    assert big_omega(n) == omega
    assert num_divisors(n) == tau
    assert divisor_sum(n) == sigma


def test_channels_are_exact_rationals():
    # n = 12: (Omega-1, tau-2, (sigma-n-1)/n) = (2, 4, 15/12)
    c1, c2, c3 = channels(12)
    assert c1 == Fraction(2)
    assert c2 == Fraction(4)
    assert c3 == Fraction(28 - 12 - 1, 12)
    assert arithmetic_pressure(12) == c1 + c2 + c3


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 97])
def test_pressure_is_exactly_zero_on_primes(p):
    assert arithmetic_pressure(p) == 0
    assert channel_factorization(p) == 0
    assert channel_divisor(p) == 0
    assert channel_abundance(p) == 0


def test_audit_matches_canonical_realization():
    # exact agreement with ArithmeticTNFRFormalism (unit coefficients)
    params = ArithmeticTNFRParameters()
    for n in range(2, 300):
        terms = ArithmeticStructuralTerms(
            tau=num_divisors(n), sigma=divisor_sum(n), omega=big_omega(n)
        )
        ref = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
        assert abs(ref - float(arithmetic_pressure(n))) < 1e-9


# --------------------------------------------------------------------------- #
# Primality sufficiency (each channel alone is 0 iff prime)
# --------------------------------------------------------------------------- #
def test_each_channel_is_individually_sufficient():
    assert all_channels_sufficient(2, 500)


def test_all_channels_nonnegative():
    assert channels_nonnegative(2, 500)


def test_pressure_zero_iff_prime():
    assert pressure_zero_iff_prime(2, 500)


# --------------------------------------------------------------------------- #
# Required test 2: channel ablations (redundancy / non-minimality)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("keep", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)])
def test_ablation_any_subset_detects_primes(keep):
    assert ablation_detects_primes(2, 500, keep)


def test_single_channel_is_minimal_for_primality():
    assert minimal_channels_for_primality(2, 500) == 1
    assert is_redundant_for_primality(2, 500) is True


def test_empty_ablation_rejected():
    with pytest.raises(ValueError):
        ablation_detects_primes(2, 100, ())


# --------------------------------------------------------------------------- #
# Required test 3: dependence tests (linear independence vs correlation)
# --------------------------------------------------------------------------- #
def test_channels_are_linearly_independent():
    assert channel_rank(2, 500) == 3
    assert has_linear_relation(2, 500) is False


def test_channels_are_correlated_but_not_dependent():
    corr = channel_correlations(2, 500)
    # strong positive correlation off-diagonal, yet strictly below 1
    for i in range(3):
        for j in range(3):
            if i != j:
                assert 0.5 < corr[i][j] < 0.999


# --------------------------------------------------------------------------- #
# Required test 4: class-conditioned distributions
# --------------------------------------------------------------------------- #
def test_pressure_by_factor_class():
    stats = pressure_by_class(2, 500, factor_class)
    assert stats["prime"]["mean"] == 0.0
    assert stats["prime"]["max"] == 0.0
    # composites carry strictly positive pressure
    for cls in ("semiprime", "prime_power", "composite_other"):
        assert stats[cls]["min"] > 0.0


@pytest.mark.parametrize("n,cls", [
    (7, "prime"), (8, "prime_power"), (9, "prime_power"),
    (6, "semiprime"), (15, "semiprime"), (30, "composite_other"),
])
def test_factor_class(n, cls):
    assert factor_class(n) == cls


@pytest.mark.parametrize("n,cls", [
    (6, "perfect"), (28, "perfect"), (12, "abundant"), (8, "deficient"),
    (7, "deficient"),
])
def test_abundance_class(n, cls):
    assert abundance_class(n) == cls


def test_primes_in_range_basic():
    assert primes_in_range(2, 20) == {2, 3, 5, 7, 11, 13, 17, 19}


# --------------------------------------------------------------------------- #
# Structural completeness is OPEN (the fourth-channel gate is closed)
# --------------------------------------------------------------------------- #
def test_completeness_is_not_proven():
    assert completeness_proven() is False


def test_fourth_channel_gate_requires_all_criteria():
    assert admits_fourth_channel(FourthChannelCriteria()) is False
    partial = FourthChannelCriteria(
        independent_structural_meaning=True,
        not_a_function_of_existing=True,
        derived_from_model_not_accuracy=True,
    )
    assert admits_fourth_channel(partial) is False
    full = FourthChannelCriteria(True, True, True, True, True, True)
    assert admits_fourth_channel(full) is True


# --------------------------------------------------------------------------- #
# Guards and exports
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1])
def test_pressure_rejects_small_n(n):
    with pytest.raises(ValueError):
        arithmetic_pressure(n)


def test_module_exports_complete():
    expected = {
        "big_omega", "num_divisors", "divisor_sum",
        "channel_factorization", "channel_divisor", "channel_abundance",
        "channels", "arithmetic_pressure", "CHANNEL_NAMES",
        "primes_in_range", "channel_zero_set", "channel_is_sufficient",
        "all_channels_sufficient", "channels_nonnegative",
        "pressure_zero_iff_prime", "channel_matrix", "channel_rank",
        "has_linear_relation", "channel_correlations",
        "ablation_detects_primes", "minimal_channels_for_primality",
        "is_redundant_for_primality", "factor_class", "abundance_class",
        "pressure_by_class", "FourthChannelCriteria",
        "admits_fourth_channel", "completeness_proven",
    }
    assert expected <= set(ap.__all__)
