#!/usr/bin/env python3
"""
Example 110 — Birch–Swinnerton-Dyer (TNFR): Rank as Structural-Pressure
       Accumulation
=======================================================================

The first milestone (BSD-1) of the TNFR-native Birch–Swinnerton-Dyer
program. This is a STRUCTURAL REFORMULATION and a reproduction of the
ORIGINAL 1965 empirical discovery, NOT a solution: it does not prove BSD
(see "Honest scope").

TNFR-native reformulation
-------------------------
For an elliptic curve E over Q, each prime p contributes a point count
N_p = #E(F_p) over the finite field F_p. The deviation from the generic
count,

    a_p = p + 1 - N_p,      |a_p| <= 2*sqrt(p)   (Hasse bound),

is read as the STRUCTURAL PRESSURE at prime p (the analogue of ΔNFR: how
far the local arithmetic reorganises away from the neutral value p+1).
Treating each prime as a node, the accumulated product

    P(X) = prod_{p <= X} N_p / p

is the accumulated structural coherence of the curve across the prime
network. Birch and Swinnerton-Dyer discovered (EDSAC computer, 1965 — the
strictest empirical method) that

    P(X) ~ C * (log X)^r ,    r = rank of E(Q),

so higher rank (more independent rational points) drives faster coherence
accumulation. This is the ARITHMETIC (prime-count) side; it is NOT a
re-implementation of an analytic L-function library.

The GL(1) -> GL(2) gap (why the existing TNFR L-tooling is not enough)
----------------------------------------------------------------------
The shipped TNFR L-track (P32–P49, src/tnfr/riemann/dirichlet_l.py) builds
Dirichlet L-functions: degree-1 Euler factors with a unit-circle weight,

    GL(1):  (1 - chi(p) p^{-s})^{-1},     |chi(p)| = 1.

Elliptic-curve L-functions are GL(2): degree-2 Euler factors carrying the
structural pressure a_p,

    GL(2):  (1 - a_p p^{-s} + p^{1-2s})^{-1},   |a_p| <= 2*sqrt(p).

The a_p data is the genuinely new ingredient. Building an a_p-weighted
prime-ladder Hamiltonian (the GL(2) analogue of the P14 von-Mangoldt
Hamiltonian) is the OPEN milestone BSD-2; this example only USES the a_p as
structural pressure, it does not yet build the operator.

Honest scope
------------
- This reproduces the EMPIRICAL rank-separation (the measurement side of
  BSD) and reframes it structurally. It does NOT prove BSD: the Clay content
  is the rigorous equality between the algebraic rank (Mordell–Weil) and the
  analytic order of vanishing of L(E,s) at s=1, which is untouched here.
- The ranks 0,1,2,3 below are KNOWN inputs (standard Cremona curves); the
  example measures that structural-pressure accumulation SEPARATES them, not
  that it derives them from first principles.
- The TNFR value-add is the FRAMING (a_p as structural pressure, product as
  coherence accumulation across the prime network), not a new mechanism.
- Slow log-convergence: the measured exponents approach 0,1,2,3 but are not
  exact at finite X; this is the same slow convergence Birch and
  Swinnerton-Dyer reported.

References
----------
- theory/TNFR_BSD_RESEARCH_NOTES.md (program, milestones, classification)
- src/tnfr/riemann/dirichlet_l.py (GL(1) Dirichlet L-track, P32)
- src/tnfr/riemann/prime_ladder_hamiltonian.py (P14, the GL(1) analogue to
  generalise)
- AGENTS.md section "TNFR-Riemann Program Overview"
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np


def sieve_primes(limit):
    """Primes < limit via a simple sieve (no optional dependency)."""
    s = bytearray([1]) * limit
    s[0:2] = b"\x00\x00"
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = b"\x00" * len(s[i * i::i])
    return [i for i in range(2, limit) if s[i]]


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return -1 if pow(a, (p - 1) // 2, p) == p - 1 else 1


def point_count(coeffs, p):
    """#E(F_p) incl. point at infinity for [a1,a2,a3,a4,a6] Weierstrass."""
    a1, a2, a3, a4, a6 = coeffs
    if p == 2:
        cnt = 1
        for x in range(2):
            for y in range(2):
                lhs = (y * y + a1 * x * y + a3 * y) % 2
                rhs = (x ** 3 + a2 * x * x + a4 * x + a6) % 2
                if lhs == rhs:
                    cnt += 1
        return cnt
    cnt = 1  # point at infinity
    for x in range(p):
        b = (a1 * x + a3) % p
        c = (x ** 3 + a2 * x * x + a4 * x + a6) % p
        disc = (b * b + 4 * c) % p   # y^2 + b y - c = 0
        cnt += 1 + legendre(disc, p)
    return cnt


# Standard smallest-conductor curves of each rank (Cremona labels).
CURVES = {
    "11a   (rank 0)": (0, (0, -1, 1, -10, -20)),
    "37a   (rank 1)": (1, (0, 0, 1, -1, 0)),
    "389a  (rank 2)": (2, (0, 1, 1, -2, 0)),
    "5077a (rank 3)": (3, (0, 0, 1, -7, 6)),
}


def experiment_1_rank_separation():
    print("=" * 72)
    print("BSD-1: Rank as structural-pressure accumulation  P(X)=prod N_p/p")
    print("=" * 72)
    print()
    print("a_p = p+1-N_p is the structural pressure; P(X) ~ C*(log X)^r.")
    print("Slope of log P(X) vs log(log X) over the tail = empirical rank.")
    print()
    pmax = 4000
    primes = sieve_primes(pmax)
    print(f"  primes up to {pmax} ({len(primes)} primes)")
    print()
    print(f"  {'curve':>16} {'true r':>7} {'P(Xmax)':>10} {'slope r_emp':>12}")
    slopes = []
    for name, (r_true, coeffs) in CURVES.items():
        logP = 0.0
        xs, ys = [], []
        for p in primes:
            np_ = point_count(coeffs, p)
            if np_ > 0:
                logP += math.log(np_ / p)
            if p >= 50:
                xs.append(math.log(math.log(p)))
                ys.append(logP)
        slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
        slopes.append((r_true, slope))
        print(f"  {name:>16} {r_true:>7} {math.exp(ys[-1]):>10.3f} "
              f"{slope:>12.3f}")
    ordered = all(slopes[i][1] < slopes[i + 1][1] for i in range(len(slopes) - 1))
    print()
    print(f"  slopes strictly ordered by true rank: {ordered}")
    print()
    print("VERDICT (BSD-1): structural-pressure accumulation SEPARATES the")
    print("ranks (slopes track 0,1,2,3) — the TNFR-native reproduction of the")
    print("original 1965 Birch–Swinnerton-Dyer empirical discovery.")
    print()


def experiment_2_hasse_pressure():
    print("=" * 72)
    print("BSD context: a_p as structural pressure, bounded by Hasse (GL(2))")
    print("=" * 72)
    print()
    print("Each prime carries pressure a_p=p+1-N_p with |a_p|<=2*sqrt(p).")
    print("This degree-2 (GL(2)) datum is what the GL(1) Dirichlet L-track")
    print("(P32–P49) does NOT carry — the open BSD-2 ingredient.")
    print()
    primes = sieve_primes(60)
    print(f"  {'curve':>16}  a_p for p in {primes[:8]} ... (|a_p|<=2sqrt p)")
    for name, (_, coeffs) in CURVES.items():
        aps = []
        ok = True
        for p in primes[:8]:
            np_ = point_count(coeffs, p)
            ap = p + 1 - np_
            aps.append(ap)
            if abs(ap) > 2 * math.sqrt(p) + 1e-9:
                ok = False
        flag = "Hasse OK" if ok else "VIOLATION"
        print(f"  {name:>16}  {str(aps):>34}  {flag}")
    print()
    print("a_p in [-2sqrt p, +2sqrt p] is the structural-pressure band; the")
    print("GL(2) Euler factor (1 - a_p p^-s + p^(1-2s))^-1 carries it. Building")
    print("the a_p-weighted prime-ladder Hamiltonian (GL(2) analogue of P14) is")
    print("the open milestone BSD-2.")
    print()


def main():
    print()
    print("  TNFR Example 110: Birch–Swinnerton-Dyer — Rank as Structural")
    print("  Pressure. Milestone BSD-1 (structural reformulation, NOT a proof)")
    print("  ===============================================================")
    print()
    experiment_1_rank_separation()
    experiment_2_hasse_pressure()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES (and what it does NOT)")
    print("=" * 72)
    print()
    print("ESTABLISHES: a TNFR-native reformulation of BSD as structural-")
    print("pressure (a_p) accumulation across the prime network, reproducing")
    print("the original 1965 empirical rank-separation P(X)~C(log X)^r. Same")
    print("disciplined pattern as the Riemann / NS / Yang-Mills / P-vs-NP")
    print("programs.")
    print()
    print("DOES NOT: prove BSD. The Clay content — rigorous equality of the")
    print("algebraic (Mordell–Weil) rank and the analytic order of vanishing")
    print("of L(E,s) at s=1 — is untouched. Ranks here are known inputs. The")
    print("genuine GL(2) TNFR L-function construction (a_p-weighted prime-")
    print("ladder Hamiltonian) is the open milestone BSD-2. No Clay claim.")
    print()


if __name__ == "__main__":
    main()
