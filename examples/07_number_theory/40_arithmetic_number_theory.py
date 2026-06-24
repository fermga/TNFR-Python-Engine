#!/usr/bin/env python3
"""
Example 40 — Arithmetic TNFR Number Theory
===========================================

Demonstrates TNFR number theory: primes as zero-pressure fixed points,
the structural triad (EPI, νf, ΔNFR) on natural numbers, and component
pressure analysis.

Physics
-------
The arithmetic ΔNFR equation

    ΔNFR(n) = ζ·(Ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))

vanishes if and only if n is prime (Theorem 4.1, TNFR_NUMBER_THEORY.md).
Each coefficient (ζ, η, θ) is WRITTEN as a (φ, γ, π, e) combination chosen to
approximate the original empirical values (audit 2026: notational, NOT derived).

Experiments
-----------
1. Structural triad survey (EPI, νf, ΔNFR) for numbers 2–50
2. Primality detection: ΔNFR = 0 ⟺ prime
3. Pressure component breakdown for selected composites
4. Local coherence landscape

References
----------
- theory/TNFR_NUMBER_THEORY.md (full derivation)
- theory/FUNDAMENTAL_THEORY.md (nodal equation)
- theory/UNIFIED_GRAMMAR_RULES.md (U1-U6 grammar)
- src/tnfr/mathematics/number_theory.py (implementation)
- AGENTS.md §"Canonical Invariants" → Invariant #1 Nodal Equation Integrity
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRParameters,
)


def _compute_terms(n: int) -> ArithmeticStructuralTerms:
    """Compute arithmetic structural terms for n using trial division."""
    # Omega(n): prime factor count WITH multiplicity
    omega = 0
    temp = n
    for p in range(2, int(math.sqrt(n)) + 1):
        while temp % p == 0:
            omega += 1
            temp //= p
    if temp > 1:
        omega += 1

    # tau(n): divisor count
    tau = 0
    for d in range(1, int(math.sqrt(n)) + 1):
        if n % d == 0:
            tau += 1
            if d != n // d:
                tau += 1

    # sigma(n): divisor sum
    sigma = 0
    for d in range(1, int(math.sqrt(n)) + 1):
        if n % d == 0:
            sigma += d
            if d != n // d:
                sigma += n // d

    return ArithmeticStructuralTerms(tau=tau, sigma=sigma, omega=omega)


# ============================================================================
# EXPERIMENT 1: Structural Triad Survey
# ============================================================================
def experiment_1_structural_triad():
    """Survey EPI, νf, ΔNFR for numbers 2–50."""
    print("=" * 72)
    print("EXPERIMENT 1: Structural Triad Survey (n = 2..50)")
    print("=" * 72)
    print()
    print("Physics: Each natural number n carries a structural triad")
    print("(EPI, νf, ΔNFR) derived from the nodal equation.")
    print("Primes are uniquely identified by ΔNFR = 0.")
    print()

    params = ArithmeticTNFRParameters()

    print(
        f"{'n':>4}  {'Prime':>5}  {'Ω':>2}  {'τ':>3}  {'σ':>5}"
        f"  {'EPI':>7}  {'νf':>7}  {'ΔNFR':>8}  {'C_loc':>6}"
    )
    print("-" * 72)

    primes_found = []
    composites_found = []

    for n in range(2, 51):
        terms = _compute_terms(n)
        epi = ArithmeticTNFRFormalism.epi_value(n, terms, params)
        nuf = ArithmeticTNFRFormalism.frequency_value(n, terms, params)
        dnfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
        cloc = ArithmeticTNFRFormalism.local_coherence(dnfr)
        is_prime = abs(dnfr) < 1e-10

        label = "YES" if is_prime else "no"
        print(
            f"{n:>4}  {label:>5}  {terms.omega:>2}  {terms.tau:>3}  "
            f"{terms.sigma:>5}  {epi:>7.3f}  {nuf:>7.4f}  "
            f"{dnfr:>8.4f}  {cloc:>6.4f}"
        )

        if is_prime:
            primes_found.append(n)
        else:
            composites_found.append(n)

    print()
    print(f"Primes detected (ΔNFR = 0): {primes_found}")
    print(f"Total: {len(primes_found)} primes, " f"{len(composites_found)} composites")

    # Validate against known primes ≤ 50
    known_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    detected = set(primes_found)
    assert (
        detected == known_primes
    ), f"Mismatch: detected={detected}, expected={known_primes}"
    print("VALIDATED: All 15 primes ≤ 50 correctly detected.")
    print()


# ============================================================================
# EXPERIMENT 2: Pressure Component Breakdown
# ============================================================================
def experiment_2_component_breakdown():
    """Decompose ΔNFR into its three independent pressure channels."""
    print("=" * 72)
    print("EXPERIMENT 2: Pressure Component Breakdown")
    print("=" * 72)
    print()
    print("Physics: ΔNFR = ζ·(Ω−1) + η·(τ−2) + θ·(σ/n − (1+1/n))")
    print("Each component vanishes independently for primes.")
    print()

    params = ArithmeticTNFRParameters()
    print(f"Pressure coefficients (notational (phi,gamma,pi,e) combos, not derived):")
    print(f"  ζ = φ×γ     = {params.zeta:.4f}")
    print(f"  η = (γ/φ)×π = {params.eta:.4f}")
    print(f"  θ = 1/φ     = {params.theta:.4f}")
    print()

    # Selection of structurally interesting numbers
    test_numbers = [
        (7, "prime"),
        (15, "semiprime 3×5"),
        (8, "prime power 2³"),
        (30, "3 distinct factors 2×3×5"),
        (12, "2²×3"),
        (60, "highly composite 2²×3×5"),
        (64, "high prime power 2⁶"),
    ]

    print(
        f"{'n':>4}  {'Type':<26}  {'P_Ω':>7}  {'P_τ':>7}  " f"{'P_σ':>7}  {'ΔNFR':>8}"
    )
    print("-" * 72)

    for n, desc in test_numbers:
        terms = _compute_terms(n)
        comps = ArithmeticTNFRFormalism.component_breakdown(n, terms, params)
        dnfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)

        print(
            f"{n:>4}  {desc:<26}  "
            f"{comps['factorization_pressure']:>7.3f}  "
            f"{comps['divisor_pressure']:>7.3f}  "
            f"{comps['sigma_pressure']:>7.3f}  "
            f"{dnfr:>8.3f}"
        )

    # Verify: all components of a prime are exactly 0
    for p in [2, 3, 5, 7, 11, 13]:
        terms = _compute_terms(p)
        comps = ArithmeticTNFRFormalism.component_breakdown(p, terms, params)
        for k, v in comps.items():
            assert abs(v) < 1e-12, f"Prime {p}: {k} = {v} ≠ 0"

    print()
    print("VALIDATED: All three pressure channels vanish for every prime.")
    print()


# ============================================================================
# EXPERIMENT 3: Coefficient Independence Verification
# ============================================================================
def experiment_3_coefficient_independence():
    """Verify that primality detection is independent of coefficient values."""
    print("=" * 72)
    print("EXPERIMENT 3: Coefficient Independence of Primality Criterion")
    print("=" * 72)
    print()
    print("Physics: ΔNFR = 0 ⟺ prime, regardless of (ζ, η, θ) > 0.")
    print("The coefficients affect the composite landscape, not the fixed points.")
    print()

    known_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

    # Test with multiple coefficient sets — all must detect the same primes
    coefficient_sets = [
        ("Canonical (φ,γ,π,e)", ArithmeticTNFRParameters()),
        ("Unit coefficients", ArithmeticTNFRParameters(zeta=1.0, eta=1.0, theta=1.0)),
        ("Scaled ×10", ArithmeticTNFRParameters(zeta=9.34, eta=11.2, theta=6.18)),
        ("Asymmetric", ArithmeticTNFRParameters(zeta=0.1, eta=5.0, theta=0.001)),
    ]

    all_pass = True
    for label, params in coefficient_sets:
        detected = set()
        for n in range(2, 51):
            terms = _compute_terms(n)
            dnfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
            if abs(dnfr) < 1e-10:
                detected.add(n)
        match = detected == known_primes
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  [{status}] {label}: {len(detected)} primes detected")

    assert all_pass, "Coefficient independence violated"
    print()
    print("VALIDATED: Primality criterion is coefficient-independent.")
    print()


# ============================================================================
# EXPERIMENT 4: Local Coherence Landscape
# ============================================================================
def experiment_4_coherence_landscape():
    """Map the local coherence C_local = 1/(1 + |ΔNFR|) across [2, 100]."""
    print("=" * 72)
    print("EXPERIMENT 4: Local Coherence Landscape (n = 2..100)")
    print("=" * 72)
    print()
    print("Physics: C_local(n) = 1/(1+|ΔNFR(n)|).")
    print("Primes have C_local = 1.0 (perfect coherence).")
    print("Composites have C_local < 1 (structural pressure).")
    print()

    params = ArithmeticTNFRParameters()

    coherence_data = []
    for n in range(2, 101):
        terms = _compute_terms(n)
        dnfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
        cloc = ArithmeticTNFRFormalism.local_coherence(dnfr)
        coherence_data.append((n, cloc, abs(dnfr) < 1e-10))

    # Statistics
    prime_coherences = [c for _, c, is_p in coherence_data if is_p]
    composite_coherences = [c for _, c, is_p in coherence_data if not is_p]

    print(f"Primes (count={len(prime_coherences)}):")
    print(f"  Mean C_local = {sum(prime_coherences)/len(prime_coherences):.6f}")
    print(f"  Min  C_local = {min(prime_coherences):.6f}")
    print(f"  Max  C_local = {max(prime_coherences):.6f}")
    print()
    print(f"Composites (count={len(composite_coherences)}):")
    print(f"  Mean C_local = {sum(composite_coherences)/len(composite_coherences):.6f}")
    print(f"  Min  C_local = {min(composite_coherences):.6f}")
    print(f"  Max  C_local = {max(composite_coherences):.6f}")
    print()

    # Find the composite with highest coherence (closest to prime)
    composites_sorted = sorted(
        [(n, c) for n, c, is_p in coherence_data if not is_p], key=lambda x: -x[1]
    )
    print("Top-5 composites by coherence (closest to prime-like):")
    for n, c in composites_sorted[:5]:
        terms = _compute_terms(n)
        print(
            f"  n={n:>3}  C_local={c:.4f}  "
            f"Ω={terms.omega}  τ={terms.tau}  σ={terms.sigma}"
        )

    # Validate all primes have C_local = 1.0
    for _, c, is_p in coherence_data:
        if is_p:
            assert abs(c - 1.0) < 1e-12, f"Prime coherence ≠ 1.0: {c}"

    print()
    print("VALIDATED: All primes have C_local = 1.0 (perfect coherence).")
    print()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print()
    print("TNFR Number Theory: Arithmetic Emergence from Structural Dynamics")
    print("=" * 72)
    print()
    print("Reference: theory/TNFR_NUMBER_THEORY.md")
    print("Nodal eq.: ∂EPI/∂t = νf · ΔNFR(t)")
    print("Primality: ΔNFR(n) = 0 ⟺ n is prime")
    print()

    experiment_1_structural_triad()
    experiment_2_component_breakdown()
    experiment_3_coefficient_independence()
    experiment_4_coherence_landscape()

    print("=" * 72)
    print("ALL EXPERIMENTS PASSED")
    print("=" * 72)
