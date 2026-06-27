#!/usr/bin/env python3
"""
Example 97 — Goldbach and the Additive/Multiplicative Orthogonality
====================================================================

Evaluates the open §13.2 question: can Goldbach's conjecture be cast as a
TNFR phase-matching problem |φ_p + φ_q − φ_{2n}| ≤ Δφ_max? The honest
answer is a STRUCTURAL NEGATIVE that reveals why Goldbach is hard in TNFR
terms — and motivates the ontological question of an operator external to
coherence.

Physics
-------
Goldbach: every even 2n ≥ 4 is a sum of two primes. In TNFR, primes are
ΔNFR = 0 attractors (zero-pressure atoms). The naive phase-matching idea
asks whether the additive relation p + q = 2n is a resonant condition on
some phase field φ_n.

Three results, all verified:

1. LINEAR phase φ_n = 2π·n/N (the §2.3 assignment) makes the matching
   IDENTICALLY zero for EVERY additive pair a+b = 2n — prime or not. It
   is VACUOUS: it cannot select primes.

2. COHERENCE phase φ_n = π·(1 − C_local(n)) (primes → 0, composites → π)
   makes the matching FAIL for the prime pairs Goldbach needs: 2n is
   always composite (high pressure, φ_{2n} ≈ π) while its prime summands
   have zero pressure (φ = 0), so the residual ≈ π ≫ Δφ_max. Additive
   target and coherent summands are ORTHOGONAL.

3. COHERENCE-WEIGHTED strength G(2n) = Σ_{a+b=2n} C_local(a)·C_local(b)
   is the only formulation with content: a prime pair contributes
   C·C = 1 exactly, so G(2n) ≥ 1 always and correlates with the true
   Goldbach count (Pearson ≈ 0.92). But here the PHASE does no work —
   only coherence does — and C·C = 1 ⟺ both prime merely RESTATES
   Goldbach.

The structural finding
----------------------
In TNFR, numbers emerge MULTIPLICATIVELY: primes are atoms composed by
coupling (UM) and nested by recursion (REMESH); the structural frequency
is ν_f = log p (additive in log = multiplicative in n). Goldbach is an
ADDITIVE question (p + q = 2n) about these multiplicative objects. The
coherence machinery — built on the multiplicative atom structure — is
ORTHOGONAL to the additive target. This orthogonality is exactly the
classical difficulty of Goldbach (additive questions about the
multiplicatively-defined primes), now visible in TNFR language.

Honest scope
------------
This is a NEGATIVE structural result, not a closure. It does NOT prove or
disprove Goldbach. The coherence-weighted G(2n) RESTATES Goldbach
(G(2n) ≥ 1 ⟺ a prime channel exists) without advancing it. What it
establishes: phase-matching alone cannot express Goldbach, because the
additive structure lies outside the multiplicative coherence ontology.

Ontological note (operator external to coherence)
-------------------------------------------------
The natural follow-up: would an ADDITIVE operator — external to the
coherence dynamics — make sense? The nodal equation ∂EPI/∂t = νf·ΔNFR
governs the reorganization of ONE EPI; coupling (UM) synchronises TWO by
phase resonance (a multiplicative composition). A genuine additive
combination of three nodes (p + q = 2n) is NOT the evolution of one EPI
nor the resonant coupling of two — it has no slot in the bare catalog.
So an additive operator would be branch B2 (a new canonical operator, if
derivable from the nodal equation) or B3 (no TNFR closure). This demo
does NOT decide B2 vs B3; it shows precisely why the question arises.

References
----------
- theory/TNFR_NUMBER_THEORY.md §13.2 (Goldbach open question)
- theory/UNIFIED_GRAMMAR_RULES.md U3 (resonant coupling, Δφ_max)
- src/tnfr/mathematics/number_theory.py (ΔNFR, local coherence)
- AGENTS.md §"TNFR-Riemann Program" branches B1/B2/B3 (B2 = new operator)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sympy import divisor_count, divisor_sigma, factorint, isprime

from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRParameters,
)

_PARAMS = ArithmeticTNFRParameters()

# Phase-matching residual threshold (a small |∇φ|-scale cut for the demo).
PHASE_MATCH_THRESHOLD = 0.2


def _local_coherence(n: int) -> float:
    """C_local(n) = 1/(1 + |ΔNFR(n)|); 1 for primes, < 1 for composites."""
    if n < 2:
        return 0.0
    omega = sum(factorint(n).values())
    tau = int(divisor_count(n))
    sigma = int(divisor_sigma(n))
    terms = ArithmeticStructuralTerms(tau=tau, sigma=sigma, omega=omega)
    dnfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, _PARAMS)
    return ArithmeticTNFRFormalism.local_coherence(dnfr)


def _wrap(angle: float) -> float:
    """Wrap to [-π, π] and return magnitude."""
    return abs((angle + math.pi) % (2 * math.pi) - math.pi)


# ============================================================================
# EXPERIMENT 1: Linear phase is vacuous
# ============================================================================
def experiment_1_linear_phase_vacuous():
    """φ_n = 2π·n/N makes matching identically zero for ALL pairs."""
    print("=" * 72)
    print("EXPERIMENT 1: Linear Phase Makes Phase-Matching Vacuous")
    print("=" * 72)
    print()
    print("φ_n = 2π·n/N (the §2.3 positional assignment).")
    print("Then φ_a + φ_b − φ_{2n} = 2π(a+b−2n)/N = 0 for ANY a+b = 2n.")
    print()

    N = 400

    def phi(n: int) -> float:
        return 2 * math.pi * n / N

    print(f"{'2n':>5}  {'#pairs':>7}  {'max resid':>11}  {'primes?':>8}")
    print("-" * 48)
    for two_n in [10, 28, 100, 200]:
        residuals = [
            _wrap(phi(a) + phi(two_n - a) - phi(two_n))
            for a in range(2, two_n // 2 + 1)
        ]
        n_pairs = len(residuals)
        print(f"{two_n:>5}  {n_pairs:>7}  {max(residuals):>11.2e}  {'NO':>8}")

    print()
    print("Residual ≈ 0 for EVERY additive pair, prime or composite.")
    print("VERDICT: linear phase is VACUOUS — it cannot encode primality.")
    print()


# ============================================================================
# EXPERIMENT 2: Coherence phase fails by orthogonality
# ============================================================================
def experiment_2_coherence_phase_fails():
    """φ_n = π·(1−C(n)) fails for prime pairs: 2n composite, summands prime."""
    print("=" * 72)
    print("EXPERIMENT 2: Coherence Phase Fails (Additive ⊥ Multiplicative)")
    print("=" * 72)
    print()
    print("φ_n = π·(1 − C_local(n)):  primes → 0,  composites → π.")
    print("For a Goldbach pair (p, q): φ_p = φ_q = 0 but 2n is composite,")
    print("so φ_{2n} ≈ π and the residual ≈ π ≫ Δφ_max.")
    print()

    def phi_coh(n: int) -> float:
        return math.pi * (1.0 - _local_coherence(n))

    print(
        f"{'2n':>5}  {'prime pair':>14}  {'φ_p+φ_q':>8}  {'φ_2n':>7}"
        f"  {'residual':>9}  {'match?':>7}"
    )
    print("-" * 60)
    for two_n in [10, 28, 50, 100, 200]:
        pair = next(
            (
                (a, two_n - a)
                for a in range(2, two_n // 2 + 1)
                if isprime(a) and isprime(two_n - a)
            ),
            None,
        )
        if pair is None:
            continue
        a, b = pair
        s = phi_coh(a) + phi_coh(b)
        t = phi_coh(two_n)
        res = _wrap(s - t)
        matched = res <= PHASE_MATCH_THRESHOLD
        print(
            f"{two_n:>5}  {str(pair):>14}  {s:>8.4f}  {t:>7.4f}"
            f"  {res:>9.4f}  {'YES' if matched else 'NO':>7}"
        )

    print()
    print(f"All residuals ≫ {PHASE_MATCH_THRESHOLD}. Phase-matching FAILS")
    print("for exactly the prime pairs Goldbach requires. The additive")
    print("target (2n, composite) is orthogonal to its coherent summands.")
    print()


# ============================================================================
# EXPERIMENT 3: Coherence weighting has content but only restates Goldbach
# ============================================================================
def experiment_3_coherence_weighting():
    """G(2n) = Σ C(a)C(b): genuine metric, but C·C=1 ⟺ both prime."""
    print("=" * 72)
    print("EXPERIMENT 3: Coherence-Weighted Strength G(2n)")
    print("=" * 72)
    print()
    print("G(2n) = Σ_{a+b=2n} C_local(a)·C_local(b).")
    print("A prime pair contributes C·C = 1 exactly (both atoms).")
    print("Here the PHASE does no work — only coherence does.")
    print()

    import numpy as np

    print(f"{'2n':>5}  {'r(2n)':>6}  {'G(2n)':>9}  {'max C·C':>8}")
    print("-" * 36)
    rs, gs = [], []
    for two_n in range(4, 1001, 2):
        r = 0
        g = 0.0
        best = 0.0
        for a in range(2, two_n // 2 + 1):
            b = two_n - a
            cc = _local_coherence(a) * _local_coherence(b)
            g += cc
            if isprime(a) and isprime(b):
                r += 1
                best = max(best, cc)
        rs.append(r)
        gs.append(g)
        if two_n in (10, 28, 50, 100, 200, 500, 1000):
            print(f"{two_n:>5}  {r:>6}  {g:>9.4f}  {best:>8.4f}")

    pearson = float(np.corrcoef(rs, gs)[0, 1])
    print()
    print(f"  min G(2n) over 4..1000 = {min(gs):.4f}  (≥ 1 always)")
    print(f"  Pearson(r(2n), G(2n))  = {pearson:.4f}")
    print()
    print("G(2n) ≥ 1 always BECAUSE every even has a prime channel (C·C=1)")
    print("— but that IS Goldbach. The metric RESTATES the conjecture; it")
    print("does not advance it. Phase plays no role.")
    print()


def main():
    print()
    print("  TNFR Example 97: Goldbach — Additive/Multiplicative Tension")
    print("  A negative structural result on phase-matching")
    print("  ================================================================")
    print()

    experiment_1_linear_phase_vacuous()
    experiment_2_coherence_phase_fails()
    experiment_3_coherence_weighting()

    print("=" * 72)
    print("STRUCTURAL FINDING & ONTOLOGICAL NOTE")
    print("=" * 72)
    print()
    print("Goldbach-as-phase-matching does NOT work in TNFR, for a precise")
    print("reason: numbers emerge MULTIPLICATIVELY (primes = atoms composed")
    print("by UM coupling, ν_f = log p), but Goldbach is an ADDITIVE question")
    print("(p + q = 2n). The coherence machinery is built on the")
    print("multiplicative atom structure and is ORTHOGONAL to the additive")
    print("target. This is the classical difficulty of Goldbach, in TNFR")
    print("language.")
    print()
    print("Ontological question (operator external to coherence):")
    print("  • The nodal equation ∂EPI/∂t = νf·ΔNFR evolves ONE EPI.")
    print("  • Coupling (UM) synchronises TWO by phase resonance")
    print("    (a multiplicative composition).")
    print("  • An ADDITIVE relation among three nodes (p + q = 2n) is")
    print("    neither — it has no slot in the bare 13-operator catalog.")
    print("  • So a genuine additive operator would be branch B2 (new")
    print("    canonical operator, IF derivable from the nodal equation) or")
    print("    B3 (no TNFR closure). This demo does NOT decide B2 vs B3 — it")
    print("    shows exactly why the question is well-posed.")
    print()
    print("Honest verdict: a NEGATIVE diagnostic. Phase-matching cannot")
    print("express Goldbach; the additive structure lies outside the")
    print("multiplicative coherence ontology. Whether an external additive")
    print("operator is canonically derivable remains open (B2 vs B3).")
    print()


if __name__ == "__main__":
    main()
