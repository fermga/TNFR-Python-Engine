#!/usr/bin/env python3
"""
Example 146 — Arithmetic Zero-Pressure Inertness under a Fixed-Pressure Flow
============================================================================

This example connects the arithmetic pressure criterion to the nodal equation
under one explicit restriction: each number keeps its arithmetic pressure
fixed while EPI is advanced by a scalar Euler step. It does not run canonical
operator implementations or validate a grammar word.

The single bridge is the nodal equation itself
----------------------------------------------
The restricted arithmetic flow used here is

    ∂EPI/∂t = νf · ΔNFR

On an arithmetic node the pressure is the primality diagnostic from
TNFR_NUMBER_THEORY.md §4:

    ΔNFR(n) = (Ω−1) + (τ−2) + (σ/n − (1+1/n)),   n prime ⟺ ΔNFR(n)=0

with Ω = number of prime factors with multiplicity, τ = divisor count, and σ =
divisor sum. The implementation uses canonical unit weights; any positive
weights preserve the same zero set.

The consequence for this fixed-pressure flow is exact: ΔNFR=0 freezes EPI for
every positive νf. This says nothing about an operator that first changes
phase, capacity, topology, or pressure, so it is not a fixed-point theorem for
the entire grammar action.

Doctrine compliance
-------------------
The arithmetic ΔNFR is a per-number diagnostic from
``ArithmeticTNFRFormalism``, not the graph-diffusion Laplacian. The experiment
therefore evaluates only ``EPI += dt*nu_f*DeltaNFR_arith``. Grammar U2 is a
constraint on operator histories and bounded nodal integrals; it is not an
update rule that drives this static arithmetic function toward zero.

Three measured results
----------------------
M1 ONE EQUILIBRIUM, THREE READINGS. n is prime ⟺ ΔNFR(n) = 0 (the §4 theorem)
   ⟺ the local coherence C(n) = 1/(1+|ΔNFR|) equals 1 (maximal). The primes are
   exactly the maximal-coherence, zero-pressure nodes (verified, 0 mismatches).

M2 CAPACITY SCALING. Under the restricted nodal flow
   EPI += dt·νf·ΔNFR, every prime is FROZEN for every νf (12/12 at νf ∈
   {0.5,1,2}); a composite drifts, and its drift FACTORS exactly as (νf gain) ×
   (arithmetic pressure) — doubling νf doubles the drift exactly (27/27). The
   capacity factor scales the RATE but cannot move a zero-pressure node while
   pressure is held fixed.

M3 FINITE COHERENCE PROFILE AND U2 NON-EQUIVALENCE. On ``2 <= n <= 60``, mean
   arithmetic coherence decreases across the observed Ω groups. This is a
   finite grouped statistic, not pointwise monotonicity in factorization
   complexity. With pressure held positive, every composite EPI drifts linearly
   and its infinite-time nodal integral diverges. The fixture is therefore not
   a U2-convergent program and primality is not "the grammar's target."

Honest scope
------------
Primality iff ΔNFR=0 is the existing §4 arithmetic theorem. This example adds a
finite scalar-flow check and an exact capacity-scaling identity. It deliberately
does not apply graph operators; consequently it cannot establish grammatical
inertness, convergence, attraction, or a cross-domain operator equivalence. It
is not new number theory and closes no open problem.

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
import statistics
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
    """Canonical structural terms (Omega with multiplicity, tau, sigma)."""
    factorisation = sp.factorint(n)
    big_omega = int(sum(factorisation.values()))  # prime factors w/ multiplicity
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
    print(
        f"  zero-pressure nodes (dNFR=0):   {len(zero_pressure)}  "
        f"== primes: {zero_pressure == primes}"
    )
    print(
        f"  maximal-coherence nodes (C=1):  {len(max_coherence)}  "
        f"== primes: {max_coherence == primes}"
    )
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
    print("M2: capacity scaling under fixed arithmetic pressure")
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
                predicted = steps * dt * nu_f * d  # (nu_f gain) x pressure
                if abs(drift - predicted) <= 1e-9:
                    factored += 1
        print(
            f"  nu_f={nu_f}:  primes frozen {frozen}/{len(primes)};  "
            f"composite drift = nu_f x pressure exactly {factored}/{len(composites)}"
        )
    # capacity is a pure scalar gain: doubling nu_f doubles the drift
    base = {n: steps * dt * 1.0 * delta_nfr(n) for n in range(2, limit + 1)}
    gain_ok = sum(
        1
        for n in composites
        if base[n] != 0
        and abs((steps * dt * 2.0 * delta_nfr(n)) / base[n] - 2.0) <= 1e-9
    )
    kernel = all(abs(base[n]) <= 1e-12 for n in primes)
    print(
        f"  doubling nu_f doubles the composite drift exactly: "
        f"{gain_ok}/{len(composites)}"
    )
    print(f"  every prime has zero drift for all sampled nu_f: {kernel}")
    print("  -> nu_f scales the rate and cannot move a zero-pressure node while")
    print("     the arithmetic pressure is held fixed.")


def experiment_3_pressure_axis(limit=60):
    print()
    print("=" * 72)
    print("M3: finite coherence profile and why this is not a U2 trajectory")
    print("=" * 72)
    by_omega = {}
    for n in range(2, limit + 1):
        t = arithmetic_terms(n)
        c = F.local_coherence(delta_nfr(n))
        by_omega.setdefault(t.omega, []).append(c)
    print("  mean arithmetic coherence C by Omega on the finite sample:")
    prev = None
    monotone = True
    for om in sorted(by_omega):
        mean_c = statistics.mean(by_omega[om])
        label = "prime (Omega=1)" if om == 1 else f"Omega={om}"
        print(f"    {label:16s}  mean C = {mean_c:.4f}   (count {len(by_omega[om])})")
        if prev is not None and mean_c > prev + 1e-9:
            monotone = False
        prev = mean_c
    print(f"  grouped means decrease across observed Omega values: {monotone}")
    print("  -> this is a finite grouped statistic, not pointwise monotonicity.")
    print("     For every composite the fixed positive pressure makes EPI drift")
    print("     linearly, so its infinite-time nodal integral does not converge.")
    print("     U2 and attraction toward primality are not established here.")


def main():
    print()
    print("#" * 72)
    print("# Example 146 - Arithmetic Zero-Pressure Inertness")
    print("#" * 72)
    print()
    experiment_1_three_readings()
    experiment_2_capacity_lever()
    experiment_3_pressure_axis()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  For the fixed arithmetic pressure used here, prime <=> dNFR=0")
    print("  <=> C=1 and EPI remains fixed for every sampled positive nu_f.")
    print("  Composite drift scales exactly with nu_f. Since composite pressure")
    print("  is held positive, that drift is unbounded in infinite time; this is")
    print("  not a U2-convergent operator history and does not prove that primes")
    print("  are fixed under every grammar word. No open problem is closed.")
    print()


if __name__ == "__main__":
    main()
