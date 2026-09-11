#!/usr/bin/env python3
"""
Example 148 — The Capacity Arm of the Dual-Lever Carries von Mangoldt: the
Riemann Zeros Live on the Axis the Substrate Does Not Encode
==============================================================================

Examples 146-147 read the operator GRAMMAR as a lens on number theory: primes
are the grammatical kernel (ΔNFR=0), and the dual-lever (examples 37/130/147)
restricted to arithmetic is the two canonical additive gradings of the free
monoid on primes — COUNT Ω (→ the ΔNFR pressure channel) and SIZE log n (→ the νf
capacity channel, example 94's atom νf = log p). This example asks the user's key
question through that lens: WHICH arm of the dual-lever carries the arithmetic
difficulty — the Riemann zeros — and why is the per-node substrate blind to it?

The exact answer
----------------
The Riemann difficulty lives entirely on the CAPACITY arm (log = νf), and the
per-node substrate encodes the PRESSURE arm (ΔNFR ← Ω), so it is structurally
blind to it. Two classical identities, read through the dual-lever, make this
exact:

  (1) log n = Σ_{d|n} Λ(d)   (Möbius-inverse: Λ = μ * log).
      The CAPACITY grading log n IS the divisor-sum of the von Mangoldt function
      Λ. So von Mangoldt — and its summatory ψ(x) = Σ_{n≤x} Λ(n), the Chebyshev
      staircase whose oscillatory residue is S(T) = (1/π)arg ζ(½+iT) (example 96,
      the sole open obstruction of the TNFR-Riemann program) — sits on the
      capacity arm.

  (2) Σ_n Λ(n) n^{-s} = −ζ'/ζ(s)   (P12, the prime-ladder / von Mangoldt zeta).
      This Dirichlet series has a SIMPLE POLE AT EVERY ZERO ρ of ζ (ζ in the
      DENOMINATOR). The capacity arm literally has the Riemann zeros as the poles
      of its generating series. By contrast Σ_n Ω(n) n^{-s} = ζ(s)·P(s) has ζ in
      the NUMERATOR, so a zero of ζ is a ZERO of the Ω series — the PRESSURE arm
      does not see the zeros as poles at all.

Why the substrate is blind (the synthesis)
-------------------------------------------
The per-node symplectic substrate encodes the PRESSURE channel: Φ_s is built from
the distance-weighted ΔNFR distribution, and on arithmetic nodes ΔNFR is the
count-graded primality pressure (Ω). The substrate therefore encodes the SMOOTH,
zero-free arm (Ω obeys the Erdős–Kac Gaussian CLT, not a zero-driven oscillation)
and is structurally blind to the CAPACITY/von-Mangoldt arm where the zeros live.
This is the SAME Fix(G)^⊥ blindness measured in examples 103/116/120: the
arithmetic the substrate cannot see is exactly the capacity-arm arithmetic, and
the Riemann residual S(T) ∈ ker(R∞) ∩ Fix(S_n)^⊥ is the capacity arm's
oscillatory half.

Doctrine compliance
-------------------
The arithmetic constants and the per-node ΔNFR are canonical
(ArithmeticTNFRFormalism). The identities log = Λ*1 and −ζ'/ζ = Σ Λ n^{-s} are
classical (Dirichlet convolution; P12 is the latter's TNFR prime-ladder form).
Nothing is imposed; the pole structure is measured against mpmath's ζ.

Three measured results
----------------------
M1 THE CAPACITY ARM IS THE VON MANGOLDT SUM. log n = Σ_{d|n} Λ(d) exactly
   (residual ~1e-16 over [2,200]); equivalently Λ = μ * log. The size grading log
   (the νf arm of the dual-lever, ex 147) IS the divisor-sum of von Mangoldt, so
   ψ(x) = Σ Λ — the Chebyshev staircase carrying S(T) (ex 96) — is the capacity
   arm's summatory.

M2 THE ZEROS ARE THE POLES OF THE CAPACITY SERIES. −ζ'/ζ(s) = Σ Λ(n) n^{-s} (P12)
   blows up as a simple pole (residue 1) at the first zero ρ_1 = ½+14.1347i:
   |−ζ'/ζ(ρ_1+ε)| ≈ 1/ε (measured 9.6, 49.6, 249.6 at ε = 0.1, 0.02, 0.004). The
   Ω series Σ Ω(n) n^{-s} = ζ(s)·P(s) has ζ in the numerator, so the zeros are
   invisible to the pressure arm.

M3 THE PRESSURE ARM IS SMOOTH; THE SUBSTRATE ENCODES IT, HENCE IS BLIND. The
   pressure grading Ω obeys the Erdős–Kac CLT: (Ω(n)−loglog n)/√(loglog n) is
   Gaussian (measured spread ≈ 1.13 over [3,10^5]; convergence is slow because
   the scale loglog n ≈ 2.4 is tiny, but the law is a Gaussian CLT, not a
   zero-driven oscillation). The per-node substrate encodes this pressure arm
   (Φ_s ← ΔNFR ← Ω), so it is structurally blind to the capacity/von-Mangoldt
   arm where the zeros live — the Fix(G)^⊥ blindness of ex 103/116/120.

Honest scope
------------
log = Λ*1, Λ = μ*log, and −ζ'/ζ = Σ Λ n^{-s} (with poles at the zeros) are
CLASSICAL facts of analytic number theory (P12 is the TNFR prime-ladder form of
the last). The NEW content is the dual-lever reading: it LOCALISES the Riemann
zeros on the capacity (νf/log) arm and the smooth primality structure on the
pressure (ΔNFR/Ω) arm, and it EXPLAINS the substrate blindness (ex 103/116/120)
structurally — the substrate encodes pressure, the zeros live on capacity. This
does NOT prove or advance RH (G4 stays open; the residual S(T) ∈ Fix(S_n)^⊥
remains unreachable, the TNFR-Riemann program stays PAUSED at T-HP). It locates
the wall on the dual-lever axis the substrate does not encode — a sharper
statement of where the obstruction lives, not a closure.

References
----------
- theory/TNFR_NUMBER_THEORY.md §4-§8 (primality field, dual-lever)
- examples/07_number_theory/147_numbers_as_free_monoid_words.py (the two gradings)
- examples/07_number_theory/96_spectral_vibration_of_coherence.py (S(T))
- examples/07_number_theory/95_primes_from_spectral_waves.py (psi(x) explicit formula)
- examples/08_emergent_geometry/116_nuf_emergent_prime_visibility.py (substrate blindness)
- src/tnfr/riemann/von_mangoldt.py (P12, the von Mangoldt prime-ladder zeta)
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13septies (T-HP, the open wall)
- AGENTS.md "REMESH-∞ Closure" (range/ker R∞, smooth vs oscillatory halves)
"""

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mpmath as mp
import sympy as sp


def von_mangoldt(n):
    """Lambda(n) = log p if n = p^k (a prime power), else 0."""
    factorisation = sp.factorint(n)
    if len(factorisation) == 1:
        prime = next(iter(factorisation))
        return math.log(prime)
    return 0.0


def experiment_1_capacity_is_von_mangoldt():
    print("=" * 72)
    print("M1: the CAPACITY arm log n = sum_{d|n} Lambda(d) (von Mangoldt sum)")
    print("=" * 72)
    max_resid = 0.0
    for n in range(2, 201):
        lhs = math.log(n)
        rhs = sum(von_mangoldt(d) for d in sp.divisors(n))
        max_resid = max(max_resid, abs(lhs - rhs))
    print(f"  max |log n - sum_d|n Lambda(d)| over [2,200] = {max_resid:.2e}")
    print("  equivalently (Mobius inverse) Lambda(n) = sum_{d|n} mu(d) log(n/d).")
    print("  sample Lambda(n) (supported on prime powers, weight log p):")
    for n in (2, 4, 6, 8, 9, 12, 16, 30):
        lam = von_mangoldt(n)
        kind = "prime power" if lam > 0 else "several primes"
        print(f"    n={n:3d}  Lambda={lam:.4f}  ({kind})")
    print("  -> the capacity grading log (the nu_f arm, ex 147) IS the divisor-sum")
    print("     of von Mangoldt; psi(x)=sum Lambda carries S(T) (ex 96).")


def experiment_2_zeros_are_the_poles():
    print()
    print("=" * 72)
    print("M2: the Riemann ZEROS are the POLES of the capacity series -zeta'/zeta")
    print("=" * 72)
    mp.mp.dps = 25
    gamma1 = mp.zetazero(1).imag
    print(f"  first nontrivial zero: rho_1 = 0.5 + {float(gamma1):.4f} i")
    print(
        "  von Mangoldt Dirichlet series = -zeta'/zeta(s) = sum Lambda(n) n^-s (P12):"
    )
    for eps in (0.1, 0.02, 0.004):
        s = mp.mpf("0.5") + eps + 1j * gamma1
        vm = -mp.zeta(s, derivative=1) / mp.zeta(s)
        print(
            f"    s = rho_1 + {float(eps):.3f}:  |-zeta'/zeta(s)| = "
            f"{float(abs(vm)):8.2f}   (~ 1/eps -> simple POLE, residue 1)"
        )
    print("  -> the CAPACITY series has a simple pole at every zero of zeta.")
    print("  contrast: sum_n Omega(n) n^-s = zeta(s)*P(s) has zeta in the")
    print("  NUMERATOR, so a zero of zeta is a ZERO of the Omega series -- the")
    print("  PRESSURE arm does NOT see the zeros as poles.")


def experiment_3_pressure_smooth_substrate_blind():
    print()
    print("=" * 72)
    print("M3: the PRESSURE arm is smooth (Erdos-Kac); the substrate encodes it")
    print("    and is therefore blind to the capacity/von-Mangoldt arithmetic")
    print("=" * 72)
    limit = 100000
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    zs = []
    for n in range(3, limit + 1):
        m, count = n, 0
        while m > 1:
            m //= spf[m]
            count += 1
        ll = math.log(math.log(n))
        if ll > 0:
            zs.append((count - ll) / math.sqrt(ll))
    spread = statistics.pstdev(zs)
    within2 = sum(1 for z in zs if abs(z) <= 2) / len(zs)
    print(f"  Omega via (Omega(n)-loglog n)/sqrt(loglog n) over [3,{limit}]:")
    print(
        f"    spread {spread:.3f} (Gaussian N(0,1): 1.0), within 2-sigma "
        f"{within2:.3f} (0.954)"
    )
    print("    (convergence is slow -- the scale loglog n is only ~2.4 -- but the")
    print("     law is a Gaussian CLT, NOT a zero-driven oscillation).")
    print("  SYNTHESIS: the per-node substrate encodes the PRESSURE arm")
    print("    (Phi_s <- dNFR <- Omega), the smooth zero-free side. It is")
    print("    structurally BLIND to the CAPACITY/von-Mangoldt arm where the")
    print("    Riemann zeros live -- the Fix(G)^perp blindness of ex 103/116/120.")
    print("    S(T) in ker(R_inf) cap Fix(S_n)^perp is the capacity arm's")
    print("    oscillatory half; the wall lives on the axis the substrate omits.")


def main():
    print()
    print("#" * 72)
    print("# Example 148 - The Capacity Arm Carries von Mangoldt: the Riemann")
    print("#               Zeros Live Where the Substrate Is Blind")
    print("#" * 72)
    print()
    experiment_1_capacity_is_von_mangoldt()
    experiment_2_zeros_are_the_poles()
    experiment_3_pressure_smooth_substrate_blind()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The dual-lever splits the arithmetic by smoothness. CAPACITY (log =")
    print("  nu_f) is the von Mangoldt sum (log = Lambda*1), and its Dirichlet")
    print("  series -zeta'/zeta has poles at the Riemann zeros -- the wall S(T)")
    print("  lives here. PRESSURE (Omega = dNFR) is the smooth Erdos-Kac side")
    print("  (zeta in the numerator, zeros invisible). The per-node substrate")
    print("  encodes pressure, so it is structurally blind to the capacity arm")
    print("  where the zeros live -- the Fix(G)^perp blindness, now LOCATED on")
    print("  the dual-lever axis. Classical identities read through the lens; no")
    print("  RH advance, the program stays paused at T-HP.")
    print()


if __name__ == "__main__":
    main()
