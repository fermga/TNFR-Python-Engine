#!/usr/bin/env python3
"""
Example 147 — Numbers as Words: Arithmetic Gradings and Pressure Channels
==========================================================================

This example compares the classical free-commutative-monoid description of
positive integers with the declared arithmetic pressure field.  It does not
apply a canonical graph operator or identify integers with operator-grammar
words.

The structural starting point (Fundamental Theorem of Arithmetic)
-----------------------------------------------------------------
By the FTA, the multiplicative monoid (ℕ_{≥1}, ×) is the FREE COMMUTATIVE MONOID
on the primes. In the grammar lens this means numbers ARE words:
  * primes        = single letters (the irreducible generators),
  * 1             = the empty word (the monoid identity),
  * Ω(n)          = the word length (number of prime letters with multiplicity),
  * multiplication = concatenation of words.
This supplies an algebraic analogy with the operator grammar's syntactic
monoid (example 145), while the two monoids remain distinct objects.

The new measured content: the coherence debt splits by composition law
----------------------------------------------------------------------
The arithmetic ΔNFR (TNFR_NUMBER_THEORY.md §4) has three pressure channels:
  factorization  P_Ω(n)  = Ω(n)−1
  divisor        P_τ(n)  = τ(n)−2
  abundance      P_σ(n)  = σ(n)/n − (1+1/n)

The example uses the canonical unit coefficients ζ=η=θ=1.

These three channels are distinguished by HOW THEY COMPOSE under multiplication:
  * Ω is COMPLETELY ADDITIVE: Ω(mn) = Ω(m)+Ω(n) for all m,n. So the factorization
    channel is a monoid homomorphism — the FREE-MONOID (word-length) backbone.
  * τ, σ are MULTIPLICATIVE (τ(mn)=τ(m)τ(n) for coprime m,n, similarly σ): the
    divisor/abundance channels carry the DIVISOR-LATTICE geometry.

Two arithmetic gradings and the dual-lever analogy
--------------------------------------------------
The free commutative monoid on primes has two natural additive gradings.  They
can be compared with the two channels of the TNFR nodal equation, but this is
an algebraic correspondence rather than an identity of dynamics:
  * COUNT  Ω(n) = Σ e_p          → the ΔNFR factorization pressure channel,
  * SIZE   log n = Σ e_p·log p   → the νf capacity (example 94: a prime atom
                                    carries νf = log p).
Both are monoid homomorphisms (ℕ,×) → (ℝ,+). Ω asks how many prime letters; log
asks how large the word is.

Doctrine compliance
-------------------
The arithmetic ΔNFR is the declared per-node primality field
(``ArithmeticTNFRFormalism``), read at the nodal level with unit coefficients.
The additivity and multiplicativity facts are exact properties of Ω, τ and σ.

Three measured results
----------------------
M1 NUMBERS ARE WORDS; THE DEBT SPLITS BY COMPOSITION LAW. Ω is additive on every
   pair (the free-monoid word length); τ,σ are multiplicative on coprime pairs
   (the divisor lattice). The factorization channel composes additively with one
   quantum: P_Ω(mn) = P_Ω(m) + P_Ω(n) + ζ (residual 0, exact), while the divisor
   channel does NOT compose additively. So ΔNFR = one ADDITIVE channel (Ω) + two
   MULTIPLICATIVE channels (τ, σ).

M2 MULTIPLYING BY A PRIME ADDS ONE UNIT TO THE FACTORIZATION CHANNEL. Building
   1→2→6→30→210 one prime at a time raises that channel by exactly one. The
   additive channel ALONE detects primality: Ω(n)=1 ⟺ n prime (0 mismatches in
   [2,80]) — the §4 theorem is 3× redundant (each channel detects primality) but
   only the Ω channel is the clean free-monoid backbone; primes are the single
   letters (Ω=1), 1 is the empty word (Ω=0).

M3 THE TWO GRADINGS GIVE A RESTRICTED DUAL-LEVER ANALOGY. Both Ω (count, compared
   with pressure) and log (size, compared with capacity in example 94) are exact
   additive monoid homomorphisms. This comparison does not identify the
   arithmetic monoid with TNFR operator dynamics.

Honest scope
------------
Ω additive, τ/σ multiplicative, primes = irreducible generators of (ℕ,×), and
the FTA free-monoid structure are all CLASSICAL facts. The NEW content is the
TNFR-lens reading: the three ΔNFR pressure channels split by composition law (one
additive and two multiplicative), while count Ω and log-size provide a useful
analogy for pressure and capacity. It restates classical multiplicative number
theory through a declared comparison; it is not a proof about operator grammar,
new number theory, or an open problem.

References
----------
- theory/TNFR_NUMBER_THEORY.md §4-§8 (primality field, dual-lever decomposition)
- src/tnfr/mathematics/number_theory.py (ArithmeticTNFRFormalism)
- examples/07_number_theory/94_generative_number_construction.py (νf = log p atoms)
- examples/07_number_theory/146_primality_grammatical_inertness.py (the kernel)
- examples/08_emergent_geometry/130_operators_break_substrate_charges.py (dual-lever)
- examples/08_emergent_geometry/145_syntactic_monoid_starfree.py (the monoid)
- AGENTS.md "Operator-Tetrad Synergies" (dual-lever capacity νf vs pressure ΔNFR)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import sympy as sp

from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRParameters,
)

PARAMS = ArithmeticTNFRParameters()
F = ArithmeticTNFRFormalism


def arithmetic_terms(n):
    """Canonical structural terms (Ω with multiplicity, τ, σ)."""
    factorisation = sp.factorint(n)
    return ArithmeticStructuralTerms(
        tau=int(sp.divisor_count(n)),
        sigma=int(sp.divisor_sigma(n)),
        omega=int(sum(factorisation.values())),
    )


def factorization_pressure(terms):
    return PARAMS.zeta * (terms.omega - 1)


def divisor_pressure(terms):
    return PARAMS.eta * (terms.tau - 2)


def delta_nfr(n):
    return F.delta_nfr_value(n, arithmetic_terms(n), PARAMS)


_PAIRS = [(3, 5), (4, 9), (6, 35), (8, 27), (2, 3), (10, 21), (7, 11), (12, 25)]


def experiment_1_composition_law():
    print("=" * 72)
    print("M1: numbers are words; the coherence debt splits by composition law")
    print("=" * 72)
    print(
        f"  constants: zeta={PARAMS.zeta:.4f}  eta={PARAMS.eta:.4f}  "
        f"theta={PARAMS.theta:.4f}"
    )
    add_ok = mul_ok = 0
    coprime_pairs = 0
    for m, n in _PAIRS:
        tm, tn, tmn = (
            arithmetic_terms(m),
            arithmetic_terms(n),
            arithmetic_terms(m * n),
        )
        omega_additive = tmn.omega == tm.omega + tn.omega
        add_ok += int(omega_additive)
        if math.gcd(m, n) == 1:
            coprime_pairs += 1
            if tmn.tau == tm.tau * tn.tau and tmn.sigma == tm.sigma * tn.sigma:
                mul_ok += 1
    print(f"  Omega additive (free-monoid word length): {add_ok}/{len(_PAIRS)} pairs")
    print(
        f"  tau & sigma multiplicative (divisor lattice): "
        f"{mul_ok}/{coprime_pairs} coprime pairs"
    )
    print("  factorization channel composes additively with one quantum zeta:")
    for m, n in [(3, 5), (6, 35), (4, 9)]:
        tm, tn, tmn = (
            arithmetic_terms(m),
            arithmetic_terms(n),
            arithmetic_terms(m * n),
        )
        predicted = (
            factorization_pressure(tm) + factorization_pressure(tn) + PARAMS.zeta
        )
        resid_add = abs(factorization_pressure(tmn) - predicted)
        resid_div = abs(
            divisor_pressure(tmn) - (divisor_pressure(tm) + divisor_pressure(tn))
        )
        print(
            f"    {m}x{n}: P_Om(mn)=P_Om(m)+P_Om(n)+zeta residual={resid_add:.1e}"
            f"   | divisor additive-residual={resid_div:.3f} (not additive)"
        )
    print("  -> dNFR = one ADDITIVE channel (Omega) + two MULTIPLICATIVE (tau,sigma)")


def experiment_2_unit_factor_increment():
    print()
    print("=" * 72)
    print("M2: multiplying by a prime adds one factorization unit; the additive")
    print("    channel alone bears primality (Omega=1 <=> prime)")
    print("=" * 72)
    print("  building 2*3*5*7 one prime at a time:")
    acc = 1
    for p in (2, 3, 5, 7):
        acc *= p
        t = arithmetic_terms(acc)
        d = F.delta_nfr_value(acc, t, PARAMS)
        c = F.local_coherence(d)
        print(
            f"    x{p} -> n={acc:4d}  Omega={t.omega}  "
            f"P_Om={factorization_pressure(t):.4f}  dNFR={d:7.4f}  C={c:.4f}"
        )
    mism = sum(
        1
        for n in range(2, 81)
        if (arithmetic_terms(n).omega == 1) != bool(sp.isprime(n))
    )
    print(f"  Omega(n)=1 <=> n prime: mismatches {mism}/79 (the additive channel")
    print("    alone detects primality; the section-4 theorem is 3x redundant but")
    print("    only Omega is the clean free-monoid backbone)")
    print("  -> each prime multiplication adds one unit to P_Omega.")


def experiment_3_dual_lever_gradings():
    print()
    print("=" * 72)
    print("M3: two additive gradings give a restricted dual-lever analogy")
    print("=" * 72)
    add_omega = add_log = 0
    for m, n in _PAIRS:
        if (
            arithmetic_terms(m * n).omega
            == arithmetic_terms(m).omega + arithmetic_terms(n).omega
        ):
            add_omega += 1
        if abs(math.log(m * n) - (math.log(m) + math.log(n))) <= 1e-9:
            add_log += 1
    print(
        f"  COUNT grading  Omega  additive (-> dNFR pressure channel): "
        f"{add_omega}/{len(_PAIRS)}"
    )
    print(
        f"  SIZE  grading  log    additive (-> nu_f capacity, ex 94):  "
        f"{add_log}/{len(_PAIRS)}"
    )
    print("  both are monoid homomorphisms (N,x) -> (R,+):")
    print("    Omega = how MANY prime letters  (the pressure arm of the lever)")
    print("    log   = how BIG the word is      (the capacity arm of the lever)")
    print("  sample primes (single letters, Omega=1, size log p):")
    for p in (2, 3, 5, 7, 11):
        print(f"    p={p:2d}  Omega=1  log p={math.log(p):.4f}")
    print("  -> this is an algebraic correspondence with pressure and capacity;")
    print("     it is not an identity with canonical operator dynamics.")


def main():
    print()
    print("#" * 72)
    print("# Example 147 - Numbers as Words: the Dual-Lever as Monoid Gradings")
    print("#" * 72)
    print()
    experiment_1_composition_law()
    experiment_2_unit_factor_increment()
    experiment_3_dual_lever_gradings()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  By the FTA, numbers are words in the free commutative monoid on")
    print("  primes: primes = letters, 1 = empty word, Omega = word length,")
    print("  multiplication = concatenation. The coherence debt dNFR splits by")
    print("  composition law -- the factorization channel is ADDITIVE (the free-")
    print("  monoid backbone, +zeta per prime), the divisor/abundance channels")
    print("  are MULTIPLICATIVE (the divisor lattice). The dual-lever (pressure")
    print("  dNFR vs capacity nu_f) is a restricted analogy with the two additive")
    print("  gradings, count Omega and log-size. This restates classical multiplicative")
    print("  number theory through the lens; no new number theory, no open")
    print("  problem closed.")
    print()


if __name__ == "__main__":
    main()
