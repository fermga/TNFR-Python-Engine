#!/usr/bin/env python3
"""
Example 94 — Generative Number Construction from Structural Atoms
=================================================================

Builds a natural number's complete TNFR structural signature
(EPI, νf, ΔNFR) from its prime constituents alone, then certifies the
construction as a grammatically well-formed canonical operator sequence.

Physics
-------
The Fundamental Theorem of Arithmetic states every n ≥ 2 factors uniquely
as n = ∏ pᵢ^aᵢ. In TNFR this is operational fractality (U5): a composite
is a NESTED structure of prime sub-EPIs.

- Primes are the **structural alphabet**: AL-emitted atoms with ΔNFR = 0
  (zero-pressure fixed points; they require no reorganization).
- Composites are built by **coupling** (UM) prime atoms and by **recursive
  self-similar echo** (REMESH) for prime powers pᵢ^aᵢ.

The arithmetic functions that drive the signature are homomorphisms over
this construction:

    Ω(m·n) = Ω(m) + Ω(n)              (completely additive)
    τ(m·n) = τ(m)·τ(n)   for gcd=1    (multiplicative)
    σ(m·n) = σ(m)·σ(n)   for gcd=1    (multiplicative)

with the closed prime-power forms

    Ω(p^a) = a,  τ(p^a) = a+1,  σ(p^a) = (p^(a+1) − 1)/(p − 1).

Therefore the entire structural signature of n is DETERMINED by its prime
atoms {(pᵢ, aᵢ)} — n never needs to be inspected as a monolithic integer.

Experiments
-----------
1. Prime alphabet: primes as AL-emitted ΔNFR = 0 atoms
2. Generative reconstruction: signature from {(pᵢ, aᵢ)} matches the direct
   (divisor-enumeration) signature, exactly
3. Compositional homomorphism laws: the generation rules (Ω additive;
   τ, σ multiplicative over coprimes)
4. Grammar certification: the construction maps to a canonical operator
   sequence that satisfies the unified grammar U1-U6

Honest scope
------------
This REFORMULATES the Fundamental Theorem of Arithmetic in TNFR operator
grammar. It does NOT derive the primes themselves — primes are taken as the
primitive structural alphabet. It does NOT generate ℕ "from nothing": the
arithmetic functions Ω, τ, σ are computed from the factorization. Generating
the DISTRIBUTION of primes from structure is the open TNFR-Riemann program
(paused at the T-HP boundary; G4 = RH remains open).

References
----------
- theory/TNFR_NUMBER_THEORY.md §3-§4 (arithmetic triad, primality)
- theory/UNIFIED_GRAMMAR_RULES.md (U1-U6, U5 multi-scale coherence)
- src/tnfr/mathematics/number_theory.py (ArithmeticTNFRFormalism)
- AGENTS.md §"Canonical Invariants" → #1 Nodal Integrity, #3 Fractality
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tnfr.mathematics.number_theory import (
    ArithmeticTNFRParameters,
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
)
from tnfr.operators.definitions import (
    Emission,
    Coupling,
    Recursivity,
    Silence,
)
from tnfr.operators.grammar_validate import validate_grammar


# ============================================================================
# Atomic structural building blocks
# ============================================================================
def factorize(n: int) -> dict[int, int]:
    """Return the prime factorization of n as {prime: exponent}."""
    factors: dict[int, int] = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors


def omega_from_factors(factors: dict[int, int]) -> int:
    """Ω(n) = Σ aᵢ — completely additive over the prime atoms."""
    return sum(factors.values())


def tau_from_factors(factors: dict[int, int]) -> int:
    """τ(n) = ∏ (aᵢ + 1) — multiplicative divisor count."""
    product = 1
    for a in factors.values():
        product *= (a + 1)
    return product


def sigma_from_factors(factors: dict[int, int]) -> int:
    """σ(n) = ∏ (pᵢ^(aᵢ+1) − 1)/(pᵢ − 1) — multiplicative divisor sum."""
    product = 1
    for p, a in factors.items():
        product *= (p ** (a + 1) - 1) // (p - 1)
    return product


def reassemble(factors: dict[int, int]) -> int:
    """Rebuild n = ∏ pᵢ^aᵢ from its structural atoms."""
    n = 1
    for p, a in factors.items():
        n *= p ** a
    return n


def signature_from_atoms(
    factors: dict[int, int], params: ArithmeticTNFRParameters
) -> tuple[int, ArithmeticStructuralTerms, float, float, float]:
    """Build the full TNFR signature from prime atoms {(pᵢ, aᵢ)} alone."""
    n = reassemble(factors)
    terms = ArithmeticStructuralTerms(
        tau=tau_from_factors(factors),
        sigma=sigma_from_factors(factors),
        omega=omega_from_factors(factors),
    )
    epi = ArithmeticTNFRFormalism.epi_value(n, terms, params)
    nuf = ArithmeticTNFRFormalism.frequency_value(n, terms, params)
    dnfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
    return n, terms, epi, nuf, dnfr


def _terms_by_enumeration(n: int) -> ArithmeticStructuralTerms:
    """Independent reference: Ω, τ, σ by direct divisor enumeration."""
    import math

    omega = 0
    temp = n
    for p in range(2, int(math.isqrt(n)) + 1):
        while temp % p == 0:
            omega += 1
            temp //= p
    if temp > 1:
        omega += 1

    tau = 0
    sigma = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            tau += 1
            sigma += d
            if d != n // d:
                tau += 1
                sigma += n // d
    return ArithmeticStructuralTerms(tau=tau, sigma=sigma, omega=omega)


def _fmt_factors(factors: dict[int, int]) -> str:
    """Human-readable factorization string, e.g. 2*3^2*13."""
    parts = []
    for p in sorted(factors):
        a = factors[p]
        parts.append(f"{p}^{a}" if a > 1 else f"{p}")
    return "*".join(parts) if parts else "1"


# ============================================================================
# EXPERIMENT 1: The prime alphabet (AL-emitted atoms)
# ============================================================================
def experiment_1_prime_alphabet():
    """Primes are the structural alphabet: AL-emitted ΔNFR = 0 atoms."""
    print("=" * 72)
    print("EXPERIMENT 1: The Prime Alphabet (AL-emitted structural atoms)")
    print("=" * 72)
    print()
    print("Physics: each prime is a generator (AL) output — a structural")
    print("atom with ΔNFR = 0. Primes are NOT built from anything smaller;")
    print("they are the irreducible letters of arithmetic construction.")
    print()

    params = ArithmeticTNFRParameters()
    alphabet = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    print(f"{'prime':>6}  {'Ω':>2}  {'τ':>2}  {'σ':>4}"
          f"  {'EPI':>7}  {'νf':>7}  {'ΔNFR':>8}")
    print("-" * 60)
    for p in alphabet:
        factors = {p: 1}
        n, terms, epi, nuf, dnfr = signature_from_atoms(factors, params)
        print(f"{p:>6}  {terms.omega:>2}  {terms.tau:>2}  {terms.sigma:>4}"
              f"  {epi:>7.4f}  {nuf:>7.4f}  {dnfr:>8.4f}")
        assert abs(dnfr) < 1e-12, f"atom {p} must have ΔNFR = 0"

    print()
    print("VALIDATED: every prime atom has ΔNFR = 0 exactly")
    print("(zero-pressure fixed point — structurally inert).")
    print()


# ============================================================================
# EXPERIMENT 2: Generative reconstruction from atoms
# ============================================================================
def experiment_2_generative_reconstruction():
    """Reconstruct the full signature from {(pᵢ, aᵢ)} and verify exactness."""
    print("=" * 72)
    print("EXPERIMENT 2: Generative Reconstruction from Prime Atoms")
    print("=" * 72)
    print()
    print("Claim: the structural signature of n is FULLY DETERMINED by its")
    print("atoms {(pᵢ, aᵢ)} via multiplicative formulas — n is never read")
    print("as a monolithic integer. We verify against an independent")
    print("divisor-enumeration reference.")
    print()

    params = ArithmeticTNFRParameters()
    targets = [3, 5, 8, 30, 234, 360, 1024]

    print(f"{'n':>5}  {'factorization':>14}  {'Ω':>2}  {'τ':>3}  {'σ':>5}"
          f"  {'ΔNFR':>9}  {'match':>5}")
    print("-" * 72)
    for n in targets:
        factors = factorize(n)
        n_rebuilt, terms_atoms, epi, nuf, dnfr = signature_from_atoms(
            factors, params
        )
        terms_ref = _terms_by_enumeration(n)

        assert n_rebuilt == n, f"reassembly failed: {n_rebuilt} != {n}"
        match = (
            terms_atoms.omega == terms_ref.omega
            and terms_atoms.tau == terms_ref.tau
            and terms_atoms.sigma == terms_ref.sigma
        )
        assert match, f"reconstruction mismatch for n={n}"

        print(f"{n:>5}  {_fmt_factors(factors):>14}  {terms_atoms.omega:>2}"
              f"  {terms_atoms.tau:>3}  {terms_atoms.sigma:>5}"
              f"  {dnfr:>9.4f}  {'OK' if match else 'FAIL':>5}")

    print()
    print("VALIDATED: (Ω, τ, σ) — and hence EPI, νf, ΔNFR — reconstruct")
    print("exactly from the prime atoms via Ω=Σaᵢ, τ=∏(aᵢ+1),")
    print("σ=∏(pᵢ^(aᵢ+1)−1)/(pᵢ−1). No divisor enumeration of n needed.")
    print()


# ============================================================================
# EXPERIMENT 3: Compositional homomorphism laws (the generation rules)
# ============================================================================
def experiment_3_homomorphism_laws():
    """The building-up rules: Ω additive, τ/σ multiplicative over coprimes."""
    print("=" * 72)
    print("EXPERIMENT 3: Compositional Homomorphism Laws")
    print("=" * 72)
    print()
    print("The generation rules under coupling (UM) of coprime structures:")
    print("  Ω(m·n) = Ω(m) + Ω(n)   (additive — emission count adds)")
    print("  τ(m·n) = τ(m)·τ(n)     (multiplicative — divisor lattice)")
    print("  σ(m·n) = σ(m)·σ(n)     (multiplicative — divisor sum)")
    print()

    import math

    coprime_pairs = [(2, 3), (4, 9), (8, 13), (9, 26), (2, 117)]

    print(f"{'m':>5}  {'n':>5}  {'gcd':>3}  {'Ω law':>16}"
          f"  {'τ law':>14}  {'σ law':>16}")
    print("-" * 72)
    for m, n in coprime_pairs:
        fm, fn, fmn = factorize(m), factorize(n), factorize(m * n)
        assert math.gcd(m, n) == 1, f"{m},{n} not coprime"

        om_add = (
            omega_from_factors(fmn)
            == omega_from_factors(fm) + omega_from_factors(fn)
        )
        tau_mul = (
            tau_from_factors(fmn)
            == tau_from_factors(fm) * tau_from_factors(fn)
        )
        sig_mul = (
            sigma_from_factors(fmn)
            == sigma_from_factors(fm) * sigma_from_factors(fn)
        )
        assert om_add and tau_mul and sig_mul, f"law broke at {m},{n}"

        om_s = (f"{omega_from_factors(fm)}+{omega_from_factors(fn)}"
                f"={omega_from_factors(fmn)}")
        tau_s = (f"{tau_from_factors(fm)}*{tau_from_factors(fn)}"
                 f"={tau_from_factors(fmn)}")
        sig_s = (f"{sigma_from_factors(fm)}*{sigma_from_factors(fn)}"
                 f"={sigma_from_factors(fmn)}")
        print(f"{m:>5}  {n:>5}  {math.gcd(m, n):>3}  {om_s:>16}"
              f"  {tau_s:>14}  {sig_s:>16}")

    print()
    print("VALIDATED: the homomorphism laws hold exactly — these ARE the")
    print("structural composition rules of the construction.")
    print()


# ============================================================================
# EXPERIMENT 4: Grammar certification (U1-U6 well-formedness)
# ============================================================================
def _build_operator_sequence(factors: dict[int, int]):
    """Map a factorization to a canonical operator construction sequence.

    Each distinct prime is emitted (AL). A prime power pᵢ^aᵢ with aᵢ>1 adds
    a recursive self-similar echo (REMESH, U5 fractality). Distinct atoms are
    bound by coupling (UM). The sequence is closed with silence (SHA).
    """
    seq = []
    primes = sorted(factors)
    for idx, p in enumerate(primes):
        seq.append(Emission())          # AL: emit the prime atom
        if factors[p] > 1:
            seq.append(Recursivity())   # REMESH: recursive power pᵢ^aᵢ
        if idx > 0:
            seq.append(Coupling())      # UM: bind atom into the composite
    seq.append(Silence())               # SHA: structural closure (U1b)
    return seq


def experiment_4_grammar_certification():
    """The construction sequence must satisfy the unified grammar U1-U6."""
    print("=" * 72)
    print("EXPERIMENT 4: Grammar Certification of the Construction")
    print("=" * 72)
    print()
    print("The atom-by-atom construction maps to a canonical operator")
    print("sequence: AL (emit prime), REMESH (recursive power, U5),")
    print("UM (couple atoms), SHA (closure). It must pass U1-U6.")
    print()

    targets = [3, 8, 30, 234, 360]
    op_short = {
        "Emission": "AL", "Recursivity": "REMESH",
        "Coupling": "UM", "Silence": "SHA",
    }

    print(f"{'n':>5}  {'factorization':>14}  {'operator sequence':<34}"
          f"  {'U1-U6':>6}")
    print("-" * 72)
    for n in targets:
        factors = factorize(n)
        seq = _build_operator_sequence(factors)
        valid = validate_grammar(seq, epi_initial=0.0)
        seq_str = "-".join(op_short[type(op).__name__] for op in seq)
        print(f"{n:>5}  {_fmt_factors(factors):>14}  {seq_str:<34}"
              f"  {'PASS' if valid else 'FAIL':>6}")
        assert valid, f"construction sequence for n={n} violates grammar"

    print()
    print("VALIDATED: every construction sequence is grammatically")
    print("well-formed — generation respects U1 (initiation/closure),")
    print("U3 (resonant coupling) and U5 (multi-scale coherence).")
    print()


def main():
    print()
    print("  TNFR Example 94: Generative Number Construction")
    print("  Numbers as nested structures of prime atoms (U5 fractality)")
    print("  =========================================================")
    print()

    experiment_1_prime_alphabet()
    experiment_2_generative_reconstruction()
    experiment_3_homomorphism_laws()
    experiment_4_grammar_certification()

    print("=" * 72)
    print("HONEST SCOPE")
    print("=" * 72)
    print()
    print("This demo REFORMULATES the Fundamental Theorem of Arithmetic in")
    print("TNFR operator grammar. The structural signature of any n is")
    print("compositionally generated from its prime atoms {(pᵢ, aᵢ)}.")
    print()
    print("It does NOT:")
    print("  - derive the primes themselves (they are the primitive alphabet)")
    print("  - generate ℕ 'from nothing' (Ω, τ, σ come from factorization)")
    print("  - generate the DISTRIBUTION of primes (= open TNFR-Riemann")
    print("    program, paused at the T-HP boundary; G4 = RH remains open)")
    print()
    print("What it DOES establish: a number's complete TNFR identity is")
    print("fixed by how its prime atoms are emitted (AL) and nested")
    print("(REMESH/UM) — the structural content of unique factorization.")
    print()


if __name__ == "__main__":
    main()
