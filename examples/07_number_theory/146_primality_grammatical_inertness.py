#!/usr/bin/env python3
"""
Example 146 — Primality as Grammatical Inertness: the Dual-Lever and the U2
Convergence Target Read on the Arithmetic Nodes
==============================================================================

This bridges two threads that had never been connected: the operator-GRAMMAR
thread (examples 139-145, which characterized the unified grammar U1-U6 as a
formal language, its automaton, its dual-lever role classes, and its star-free
syntactic monoid) and the NUMBER-THEORY thread (examples 40, 100-102, which
established primality as the structural equilibrium ΔNFR = 0). The user's
intuition is that the dynamics implied by the grammar is a lens onto the other
modules; here that lens falls on primality.

The single bridge is the nodal equation itself
----------------------------------------------
Every operator ("word" in the grammar) acts on form through ONE rule:

    ∂EPI/∂t = νf · ΔNFR

The dual-lever (examples 37, 130): each operator acts via the CAPACITY lever νf
(how fast the node reorganizes) or the PRESSURE lever ΔNFR (the structural
forcing). On an arithmetic node the pressure is the canonical primality field
(TNFR_NUMBER_THEORY.md §4):

    ΔNFR(n) = ζ·(Ω−1) + η·(τ−2) + θ·(σ/n − (1+1/n)),   n prime ⟺ ΔNFR(n)=0

with Ω = number of prime factors with multiplicity, τ = divisor count, σ =
divisor sum, and ζ=φ·γ, η=(γ/φ)·π, θ=1/φ the canonical arithmetic constants.

The consequence is exact: since every word acts through νf·ΔNFR, and ΔNFR=0 at
primes, NO valid grammatical program can move a prime's form. A prime is
structurally INERT — it is the fixed point of the entire grammar's action on
arithmetic nodes. Primality is grammatical inertness.

Doctrine compliance
-------------------
The arithmetic ΔNFR is the canonical per-node primality field
(ArithmeticTNFRFormalism), NOT the graph-diffusion Laplacian, so the bridge is
read at the NODAL-EQUATION level (nodal flow EPI += dt·νf·ΔNFR), exactly as
example 102 established. Nothing is imposed; the dual-lever factorization and the
U2 coherence target are measured against the canonical formalism.

Three measured results
----------------------
M1 ONE EQUILIBRIUM, THREE READINGS. n is prime ⟺ ΔNFR(n) = 0 (the §4 theorem)
   ⟺ the local coherence C(n) = 1/(1+|ΔNFR|) equals 1 (maximal). The primes are
   exactly the maximal-coherence, zero-pressure nodes (verified, 0 mismatches).

M2 THE CAPACITY LEVER — PRIMES ARE THE GRAMMATICAL KERNEL. Under the nodal flow
   EPI += dt·νf·ΔNFR, every prime is FROZEN for every νf (12/12 at νf ∈
   {0.5,1,2}); a composite drifts, and its drift FACTORS exactly as (νf gain) ×
   (arithmetic pressure) — doubling νf doubles the drift exactly (27/27). The
   capacity lever scales the RATE but can never move a prime: the primes are the
   kernel of the whole νf-lever sub-grammar.

M3 THE U2 PRESSURE AXIS — THE GRAMMAR'S CONVERGENCE TARGET IS PRIMALITY. U2
   (convergence/boundedness) drives ΔNFR → 0; in coherence terms C → 1. The
   maximal-coherence target C=1 is EXACTLY primality, and C decreases
   monotonically with Ω (mean C: prime 1.000, Ω=2 0.239, Ω=3 0.130, Ω=4 0.089,
   Ω=5 0.085) — factorization complexity is structural coherence debt. A prime
   needs the EMPTY word (the identity of the star-free syntactic monoid, ex 145):
   it is already at the grammar's convergence target.

Honest scope
------------
Primality ⟺ ΔNFR=0 is the existing §4 theorem; the NEW content is the GRAMMAR-
LENS reading of it — primes as the dual-lever kernel (νf-lever-invariant set),
the U2 convergence target ΔNFR→0 identified with primality, and the empty word /
monoid identity as the program a prime needs. The arithmetic ΔNFR is a per-node
function, so the canonical graph operators (which recompute ΔNFR from neighbours)
are deliberately NOT used; the bridge lives at the nodal-equation level. This
restates the primality theorem through the grammar dynamics; it is not new number
theory and closes no open problem. It does deliver the user's thesis concretely:
the grammar's dynamics is a lens that unifies the number-theory module with the
operator grammar.

References
----------
- theory/TNFR_NUMBER_THEORY.md §4 (primality as ΔNFR=0, the canonical constants)
- src/tnfr/mathematics/number_theory.py (ArithmeticTNFRFormalism)
- examples/07_number_theory/102_nodal_flow_primes_equilibria.py (primes = equilibria)
- examples/08_emergent_geometry/130_operators_break_substrate_charges.py (dual-lever)
- examples/08_emergent_geometry/145_syntactic_monoid_starfree.py (the monoid identity)
- AGENTS.md "Operator-Tetrad Synergies" (dual-lever), "Unified Grammar U2"
"""

import os
import sys
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import sympy as sp

from tnfr.mathematics.number_theory import (
    ArithmeticTNFRFormalism,
    ArithmeticStructuralTerms,
    ArithmeticTNFRParameters,
)

PARAMS = ArithmeticTNFRParameters()
F = ArithmeticTNFRFormalism


def arithmetic_terms(n):
    """Canonical structural terms (Omega with multiplicity, tau, sigma)."""
    factorisation = sp.factorint(n)
    big_omega = int(sum(factorisation.values()))   # prime factors w/ multiplicity
    tau = int(sp.divisor_count(n))
    sigma = int(sp.divisor_sigma(n))
    return ArithmeticStructuralTerms(tau=tau, sigma=sigma, omega=big_omega)


def delta_nfr(n):
    return F.delta_nfr_value(n, arithmetic_terms(n), PARAMS)


def experiment_1_three_readings(limit=60):
    print("=" * 72)
    print("M1: one equilibrium, three readings -- prime <=> dNFR=0 <=> C=1")
    print("=" * 72)
    primes = [n for n in range(2, limit + 1) if sp.isprime(n)]
    zero_pressure = []
    max_coherence = []
    for n in range(2, limit + 1):
        d = delta_nfr(n)
        c = F.local_coherence(d)
        if abs(d) <= 1e-12:
            zero_pressure.append(n)
        if abs(c - 1.0) <= 1e-12:
            max_coherence.append(n)
    print(f"  primes in [2,{limit}]:           {len(primes)}")
    print(f"  zero-pressure nodes (dNFR=0):   {len(zero_pressure)}  "
          f"== primes: {zero_pressure == primes}")
    print(f"  maximal-coherence nodes (C=1):  {len(max_coherence)}  "
          f"== primes: {max_coherence == primes}")
    print("  sample (n, dNFR, C):")
    for n in (2, 3, 4, 6, 7, 12, 13, 30):
        d = delta_nfr(n)
        c = F.local_coherence(d)
        kind = "prime" if sp.isprime(n) else "composite"
        print(f"    n={n:3d}  dNFR={d:+8.4f}  C={c:.4f}  ({kind})")
    print("  -> primes are exactly the zero-pressure, maximal-coherence nodes.")


def experiment_2_capacity_lever(limit=40):
    print()
    print("=" * 72)
    print("M2: the capacity lever (nu_f) -- primes are the grammatical kernel")
    print("=" * 72)
    dt, steps = 0.1, 50
    primes = [n for n in range(2, limit + 1) if sp.isprime(n)]
    composites = [n for n in range(2, limit + 1) if not sp.isprime(n)]
    for nu_f in (0.5, 1.0, 2.0):
        frozen = 0
        factored = 0
        for n in range(2, limit + 1):
            d = delta_nfr(n)
            epi = 1.0
            for _ in range(steps):
                epi += dt * nu_f * d
            drift = epi - 1.0
            if sp.isprime(n):
                if abs(drift) <= 1e-12:
                    frozen += 1
            else:
                predicted = steps * dt * nu_f * d   # (nu_f gain) x pressure
                if abs(drift - predicted) <= 1e-9:
                    factored += 1
        print(f"  nu_f={nu_f}:  primes frozen {frozen}/{len(primes)};  "
              f"composite drift = nu_f x pressure exactly {factored}/{len(composites)}")
    # capacity is a pure scalar gain: doubling nu_f doubles the drift
    base = {n: steps * dt * 1.0 * delta_nfr(n) for n in range(2, limit + 1)}
    gain_ok = sum(
        1 for n in composites
        if base[n] != 0 and abs((steps * dt * 2.0 * delta_nfr(n)) / base[n] - 2.0) <= 1e-9
    )
    kernel = all(abs(base[n]) <= 1e-12 for n in primes)
    print(f"  doubling nu_f doubles the composite drift exactly: "
          f"{gain_ok}/{len(composites)}")
    print(f"  every prime is in the kernel (zero drift for all nu_f): {kernel}")
    print("  -> the capacity lever scales the RATE; it can never move a prime.")


def experiment_3_pressure_axis(limit=60):
    print()
    print("=" * 72)
    print("M3: the U2 pressure axis -- the grammar's convergence target IS primality")
    print("=" * 72)
    by_omega = {}
    for n in range(2, limit + 1):
        t = arithmetic_terms(n)
        c = F.local_coherence(delta_nfr(n))
        by_omega.setdefault(t.omega, []).append(c)
    print("  U2 drives dNFR -> 0, i.e. coherence C = 1/(1+|dNFR|) -> 1.")
    print("  mean coherence C by Omega (factorization complexity = coherence debt):")
    prev = None
    monotone = True
    for om in sorted(by_omega):
        mean_c = statistics.mean(by_omega[om])
        label = "prime (Omega=1)" if om == 1 else f"Omega={om}"
        print(f"    {label:16s}  mean C = {mean_c:.4f}   (count {len(by_omega[om])})")
        if prev is not None and mean_c > prev + 1e-9:
            monotone = False
        prev = mean_c
    print(f"  C decreases monotonically with Omega: {monotone}")
    print("  -> the U2 target dNFR->0 (maximal coherence C=1) IS primality;")
    print("     a prime needs the EMPTY word (the identity of the star-free")
    print("     syntactic monoid, ex 145) -- it is already at the convergence")
    print("     target. Primality = grammatical inertness.")


def main():
    print()
    print("#" * 72)
    print("# Example 146 - Primality as Grammatical Inertness")
    print("#" * 72)
    print()
    experiment_1_three_readings()
    experiment_2_capacity_lever()
    experiment_3_pressure_axis()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The grammar acts on form through the single nodal rule dEPI/dt =")
    print("  nu_f * dNFR. On arithmetic nodes dNFR is the primality field, so:")
    print("  primes are the kernel of the capacity (nu_f) lever (frozen under")
    print("  every program), and the U2 pressure target dNFR->0 = maximal")
    print("  coherence C=1 = primality. A prime needs the empty word: it is")
    print("  grammatically inert. The grammar dynamics is the lens that unifies")
    print("  the number-theory module with the operator grammar. Restates the")
    print("  primality theorem through the grammar; no new number theory, no")
    print("  open problem closed.")
    print()


if __name__ == "__main__":
    main()
