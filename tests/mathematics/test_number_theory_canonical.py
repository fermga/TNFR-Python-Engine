"""Canonical-coherence tests for the arithmetic TNFR module.

Covers (1) the canonical UNIT coefficients of the arithmetic triad (only π is a
genuine structural scale; the φ/γ/e overlay is removed), (2) the §4.1 primality
theorem and its §4.2 coefficient independence, and (3) the Fractal-Resonant Node
(NFR) read-out plus the emergent-geometry diagnostics (conservation, symplectic
substrate) integrated into the arithmetic network.
"""

from __future__ import annotations

import math

import pytest

from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRNetwork,
    ArithmeticTNFRParameters,
)


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def _terms(n: int) -> ArithmeticStructuralTerms:
    import sympy

    return ArithmeticStructuralTerms(
        tau=int(sympy.divisor_count(n)),
        sigma=int(sympy.divisor_sigma(n)),
        omega=sum(sympy.factorint(n).values()),
    )


# --- canonical unit coefficients -------------------------------------------


def test_arithmetic_parameters_are_canonical_unity():
    """Only π is structural; the triad weights are canonically unity (§4.2)."""
    p = ArithmeticTNFRParameters()
    assert p.zeta == p.eta == p.theta == 1.0
    assert p.alpha == p.beta == p.gamma == 1.0
    assert p.nu_0 == p.delta == p.epsilon == 1.0


def test_delta_nfr_is_pure_structural_excess():
    """ΔNFR(n) = (Ω−1) + (τ−2) + (σ/n − (1+1/n)) with unit weights."""
    p = ArithmeticTNFRParameters()
    n = 15  # 3×5: Ω=2, τ=4, σ=24
    t = _terms(n)
    expected = (2 - 1) + (4 - 2) + (24 / 15 - (1 + 1 / 15))
    assert ArithmeticTNFRFormalism.delta_nfr_value(n, t, p) == pytest.approx(expected)


# --- primality theorem (§4.1) and coefficient independence (§4.2) ----------


@pytest.mark.parametrize("n", [2, 3, 5, 7, 17, 19, 97, 101])
def test_primes_are_equilibria(n):
    p = ArithmeticTNFRParameters()
    assert ArithmeticTNFRFormalism.delta_nfr_value(n, _terms(n), p) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("n", [4, 6, 8, 9, 15, 30, 100, 561])
def test_composites_have_positive_pressure(n):
    p = ArithmeticTNFRParameters()
    assert ArithmeticTNFRFormalism.delta_nfr_value(n, _terms(n), p) > 0.0


def test_coefficient_independence_preserves_primality():
    """Any positive coefficients give the same ΔNFR=0 primality verdict (§4.2)."""
    canonical = ArithmeticTNFRParameters()
    scaled = ArithmeticTNFRParameters(zeta=0.5, eta=2.7, theta=3.14)
    for n in range(2, 80):
        t = _terms(n)
        d_can = ArithmeticTNFRFormalism.delta_nfr_value(n, t, canonical)
        d_scl = ArithmeticTNFRFormalism.delta_nfr_value(n, t, scaled)
        assert (abs(d_can) < 1e-12) == (abs(d_scl) < 1e-12) == _is_prime(n)


# --- Fractal-Resonant Node (NFR) read-out ----------------------------------


def test_nfr_equilibrium_set_is_exactly_the_primes():
    """Canonical payoff: the ΔNFR=0 attractors of the arithmetic NFR are primes."""
    net = ArithmeticTNFRNetwork(max_number=60)
    r = net.nfr()
    assert r["equilibrium_is_primes"] is True
    n_primes = sum(1 for n in range(2, 61) if _is_prime(n))
    assert r["equilibrium_fraction"] == pytest.approx(n_primes / 59)
    assert r["topology"] in {"radial", "annular", "multinodal"}
    assert 0.0 <= r["coherence"] <= 1.0
    assert r["n_nodes"] == 59


# --- emergent geometry: conservation + symplectic substrate ----------------


def test_arithmetic_conservation_is_finite():
    net = ArithmeticTNFRNetwork(max_number=40)
    c = net.conservation()
    assert math.isfinite(c["noether_charge"])
    assert math.isfinite(c["energy"])
    assert c["energy"] >= 0.0


def test_arithmetic_symplectic_substrate_is_valid():
    net = ArithmeticTNFRNetwork(max_number=40)
    s = net.symplectic_substrate()
    # phase space dimension is 4N for an N-node network
    assert s["phase_space_dimension"] == 4 * net.graph.number_of_nodes()
    assert s["is_valid_manifold"] is True
    assert math.isfinite(s["hamiltonian"])
    # the logn phase activates the geometric sector
    assert s["background_potential"] > 0.0
