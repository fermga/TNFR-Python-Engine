r"""Tests for the R3 CRT multiscale composition of residue networks.

For coprime ``a, b`` the CRT permutation makes the **unit** power-residue Cayley
operator factor exactly:

    L_ab = I − (I − L_a) ⊗ (I − L_b)      (over ℚ, residual 0),

with parent spectrum the child eigenvalue composition ``λ + μ − λμ`` and the U5
gap bound ``λ₂(ab) ≤ min(λ₂(a), λ₂(b))``.  The unrestricted (non-unit) residue
set is the non-factorizing control: it does **not** CRT-factor.  The theorem is a
*structural* branch — it uses the known factors ``a, b`` to assemble sub-networks
and makes no factoring/complexity/crypto claim.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics import crt_multiscale as cm
from tnfr.mathematics.crt_multiscale import (
    child_parent_eigenvalue,
    composed_spectrum,
    crt_kronecker_residual,
    crt_ordering,
    full_power_residue_laplacian,
    residue_set_factors,
    spectral_gap,
    u5_spectral_gap_composition,
    unit_power_residue_laplacian,
    verify_spectrum_composition,
)
from tnfr.mathematics.number_theory import (
    power_residue_set,
    unit_power_residue_set,
)
from tnfr.physics.spectral_projectors import derived_tolerance

# Coprime moduli (mix of prime and prime-power) and powers.
COPRIME_PAIRS = [(3, 5), (5, 7), (3, 7), (4, 9), (5, 9), (7, 8), (3, 11),
                 (5, 11), (8, 9), (9, 11)]
POWERS = [1, 2, 3]
COPRIME_CASES = [(a, b, k) for (a, b) in COPRIME_PAIRS for k in POWERS]


# --------------------------------------------------------------------------- #
# unit_power_residue_set: the units-only restriction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [5, 7, 11, 13, 17, 19, 23])
@pytest.mark.parametrize("k", POWERS)
def test_unit_and_full_coincide_for_primes(p, k):
    # For prime p every nonzero residue is a unit, so the two sets coincide.
    assert unit_power_residue_set(p, k) == power_residue_set(p, k)


@pytest.mark.parametrize("m,k", [(15, 2), (35, 2), (21, 2), (33, 2)])
def test_unit_strictly_subset_for_composites(m, k):
    # For composite m the unit set drops the non-unit powers.
    u = unit_power_residue_set(m, k)
    f = power_residue_set(m, k)
    assert u < f  # proper subset


def test_unit_power_residue_set_are_units():
    import math

    m, k = 45, 3
    for r in unit_power_residue_set(m, k):
        assert math.gcd(r, m) == 1


@pytest.mark.parametrize("bad", [0, 1])
def test_unit_power_residue_set_rejects_small_modulus(bad):
    with pytest.raises(Exception):
        unit_power_residue_set(bad, 2)


def test_unit_power_residue_set_rejects_nonpositive_power():
    with pytest.raises(Exception):
        unit_power_residue_set(15, 0)


# --------------------------------------------------------------------------- #
# CRT ordering (permutation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b", COPRIME_PAIRS)
def test_crt_ordering_is_a_permutation(a, b):
    perm = crt_ordering(a, b)
    assert sorted(perm) == list(range(a * b))


@pytest.mark.parametrize("a,b", COPRIME_PAIRS)
def test_crt_ordering_respects_congruences(a, b):
    perm = crt_ordering(a, b)
    for i in range(a):
        for j in range(b):
            r = perm[i * b + j]
            assert r % a == i
            assert r % b == j


@pytest.mark.parametrize("a,b", [(4, 6), (6, 9), (10, 15)])
def test_crt_ordering_requires_coprime(a, b):
    with pytest.raises(ValueError):
        crt_ordering(a, b)


# --------------------------------------------------------------------------- #
# Required test 1: CRT permutation similarity (exact Kronecker identity)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,k", COPRIME_CASES)
def test_crt_kronecker_identity_is_exact(a, b, k):
    # L_ab = I − (I−L_a)⊗(I−L_b) up to the CRT permutation, exactly over ℚ.
    assert crt_kronecker_residual(a, b, k) == Fraction(0)


@pytest.mark.parametrize("a,b,k", COPRIME_CASES)
def test_unit_residue_set_factors_under_crt(a, b, k):
    assert residue_set_factors(a, b, k, unit=True) is True


# --------------------------------------------------------------------------- #
# Required test 2: spectrum composition
# --------------------------------------------------------------------------- #
def test_child_parent_eigenvalue_formula():
    # λ + μ − λμ = 1 − (1−λ)(1−μ).
    for lam in (0.0, 0.3, 1.0, 1.5):
        for mu in (0.0, 0.4, 1.0):
            assert child_parent_eigenvalue(lam, mu) == pytest.approx(
                1 - (1 - lam) * (1 - mu)
            )


def test_child_parent_eigenvalue_identity_element():
    # μ = 0 (the trivial mode) leaves λ unchanged: every child eigenvalue embeds.
    for lam in (0.0, 0.25, 0.7, 1.3):
        assert child_parent_eigenvalue(lam, 0.0) == pytest.approx(lam)


@pytest.mark.parametrize("a,b,k", COPRIME_CASES)
def test_spectrum_composition_matches_numerically(a, b, k):
    residual = verify_spectrum_composition(a, b, k)
    tol = derived_tolerance(
        np.array(
            [[float(x) for x in row]
             for row in unit_power_residue_laplacian(a * b, k)]
        )
    )
    assert residual <= tol


def test_composed_spectrum_cardinality():
    spec_a = [0.0, 0.5, 1.5]
    spec_b = [0.0, 1.0]
    assert len(composed_spectrum(spec_a, spec_b)) == len(spec_a) * len(spec_b)


# --------------------------------------------------------------------------- #
# Required test 3: U5 telemetry (spectral-gap bound across scales)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,k", COPRIME_CASES)
def test_u5_gap_bound_holds(a, b, k):
    gap_ab, gap_a, gap_b, bounded = u5_spectral_gap_composition(a, b, k)
    assert bounded is True
    tol = derived_tolerance(
        np.array(
            [[float(x) for x in row]
             for row in unit_power_residue_laplacian(a * b, k)]
        )
    )
    assert gap_ab <= min(gap_a, gap_b) + tol


def test_spectral_gap_positive_and_below_trivial_mode():
    # The constant mode is a zero eigenvalue; the gap is the next smallest |λ|.
    L = unit_power_residue_laplacian(7, 2)
    gap = spectral_gap(L)
    assert gap > 0.0


# --------------------------------------------------------------------------- #
# Required test 4: non-factorizing control branch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,k", [(3, 5, 2), (5, 7, 2), (4, 9, 2),
                                   (3, 8, 2), (5, 9, 2)])
def test_full_residue_set_does_not_factor(a, b, k):
    # The unrestricted (non-unit) set breaks the CRT product structure.
    assert residue_set_factors(a, b, k, unit=False) is False


@pytest.mark.parametrize("a,b,k", [(3, 5, 2), (5, 7, 2), (4, 9, 2), (5, 9, 2)])
def test_full_residue_kronecker_identity_fails(a, b, k):
    # The Kronecker identity that is exact for the unit set does NOT hold for
    # the unrestricted control operator.
    La = full_power_residue_laplacian(a, k)
    Lb = full_power_residue_laplacian(b, k)
    Lab = full_power_residue_laplacian(a * b, k)
    Ia = cm._identity(a)
    Ib = cm._identity(b)
    composed = cm._sub(
        cm._identity(a * b), cm.kron(cm._sub(Ia, La), cm._sub(Ib, Lb))
    )
    perm = crt_ordering(a, b)
    ab = a * b
    worst = max(
        abs(Lab[perm[u]][perm[v]] - composed[u][v])
        for u in range(ab)
        for v in range(ab)
    )
    assert worst > Fraction(0)


# --------------------------------------------------------------------------- #
# Structural exactness: the whole identity chain is rational (no float)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,k", [(3, 5, 2), (4, 9, 2), (5, 9, 3)])
def test_operator_entries_are_rational(a, b, k):
    L = unit_power_residue_laplacian(a * b, k)
    assert all(isinstance(x, Fraction) for row in L for x in row)


@pytest.mark.parametrize("a,b,k", [(3, 5, 2), (5, 7, 2)])
def test_kron_residual_is_a_fraction(a, b, k):
    assert isinstance(crt_kronecker_residual(a, b, k), Fraction)


def test_module_exports_complete():
    expected = {
        "unit_power_residue_laplacian",
        "full_power_residue_laplacian",
        "crt_ordering",
        "kron",
        "crt_kronecker_residual",
        "residue_set_factors",
        "child_parent_eigenvalue",
        "composed_spectrum",
        "verify_spectrum_composition",
        "spectral_gap",
        "u5_spectral_gap_composition",
    }
    assert expected <= set(cm.__all__)
