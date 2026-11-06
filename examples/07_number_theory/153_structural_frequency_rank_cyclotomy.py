#!/usr/bin/env python3
"""
Example 153 — The Structural-Frequency Rank of Arithmetic Diffusion Networks:
Two-Arm Primality and the Cyclotomy Law
==============================================================================

Examples 117-123 read the quadratic-residue (QR) residue graph spectrally;
example 119 found that the QR Cayley digraph has exactly 3 distinct eigenvalues
iff the modulus is an odd prime. Examples 146-149 built the dual-lever (pressure
ΔNFR vs capacity νf) and the free monoid on primes (example 147). This example
UNIFIES those two threads through the canonical structural-diffusion operator and
EXTENDS them to a cyclotomy family.

The structural object
---------------------
On an arithmetic Cayley network — vertex set ℤ/mℤ, directed edges a→a+c for c in
a connection set C(m) ⊆ ℤ/mℤ — the canonical structural-diffusion operator is
L_rw = I − D⁻¹W (src/tnfr/physics/structural_diffusion.py). This is exactly the
ΔNFR EPI channel: g = −L_rw·field is the neighbour-mean − self gradient. Its
distinct eigenvalues are the network's STRUCTURAL FREQUENCIES (the relaxation
rates of ∂EPI/∂t = −νf·L_rw·EPI). Let s_C(m) be their count — the structural
rank.

Three connection sets are used:
  * QR:       C = {x² mod m} \\ {0}     (quadratic residues, example 119)
  * k-th pow: C = {xᵏ mod m} \\ {0}     (the cyclotomic generalization)
  * unitary:  C = {u : gcd(u,m)=1}      (Ramanujan-sum network)

The dual-lever (examples 37/130, 147)
-------------------------------------
The per-node arithmetic ΔNFR (TNFR_NUMBER_THEORY.md §4) is the PRESSURE arm:
ΔNFR(n) = ζ(Ω−1) + η(τ−2) + θ(σ/n−(1+1/n)), with n prime ⟺ ΔNFR=0. The structural
rank s_C(m) is a GLOBAL SPECTRAL quantity — a capacity-arm-like reading. This
example measures both on the same numbers.

Doctrine compliance
-------------------
The diffusion operator and its spectrum come from the canonical
structural_diffusion_operator; the per-node ΔNFR from the canonical
ArithmeticTNFRFormalism. This example CONSUMES the canonical residue-network API
in tnfr.mathematics.number_theory (quadratic_residue_set / power_residue_set /
unitary_residue_set, arithmetic_cayley_digraph, residue_network_rank,
power_residue_rank, quadratic_residue_annotated_rank) and the canonical
structural_frequency_rank in tnfr.physics.structural_diffusion. Nothing is
imposed beyond the arithmetic connection set; the eigenvalues are read off and
measured.

Three measured results
----------------------
M1 TWO-ARM PRIMALITY. Primality is a simultaneous fixed point of BOTH dual-lever
   arms: the per-node pressure ΔNFR(n)=0 (the §4 theorem) AND the global spectral
   rank s_QR(m)=3 (example 119) agree with primality with 0 disagreements. They
   are not merely equal at primes: both GROW with factorization complexity —
   corr(ΔNFR, log A)=0.96, corr(log A, ω)=0.97 over odd m in [3,599]. This
   bridges the number-theory pressure field (§4) and the emergent-geometry QR
   spectrum (example 119), the two arms of the dual-lever.

M2 THE CYCLOTOMY LAW. The structural rank of the k-th power residue network on a
   prime p is s_k(p) = gcd(k,p−1)+1 (PROVED for all k via Gauss periods, theory
   §9.11; verified here for k=2..6, and to k=1..40, p<64 in the test suite). The
   maximal rank k+1 is reached exactly when p ≡ 1 (mod k), i.e. exactly when p
   splits completely in the cyclotomic field ℚ(ζ_k) (0 mismatches). The QR result
   is the k=2 special case (gcd(2,p−1)=2 ⟹ uniform rank 3). So the structural rank
   READS p's cyclotomic splitting.

M3 FREE-MONOID EXPONENTIAL GRADING. On squarefree m the structural rank is
   (per-prime rank)^ω: the QR network gives 3^ω (= the multiplicative A(m)), the
   unitary network gives 2^ω. The base is the connection set's per-prime
   association-scheme rank (QR=3, unitary=2). This is the EXPONENTIAL reading of
   the free-monoid word length ω (example 147), whose pressure-arm counterpart is
   the LINEAR ζ(ω−1). The scalar rank collides on mixed composites (the
   conductor-annotated count is the genuinely multiplicative one).

Honest scope
------------
primality ⟺ ΔNFR=0 (§4) and the QR 3-eigenvalue signature (example 119) pre-exist;
the cyclotomy law s_k(p)=gcd(k,p−1)+1 is now PROVED for all k (theory/
TNFR_NUMBER_THEORY.md §9.11) and is, underneath, classical Gauss-period /
cyclotomic-number theory (the eigenvalues of the k-th power Cayley graph are
Gauss periods of degree gcd(k,p−1), taking that many distinct values plus the
trivial one). The NEW content is the unified TNFR structural-diffusion framing:
the two-arm bridge (§4 ↔ example 119), the ω-grading (↔ example 147), and the
cyclotomy family with the complete-splitting reading. It restates classical
objects through the canonical diffusion operator; it is not new number theory and
closes no open problem (RH stays paused at T-HP).

References
----------
- theory/TNFR_NUMBER_THEORY.md §4-§9 (primality field, residue-spectrum arc)
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator)
- src/tnfr/mathematics/number_theory.py (ArithmeticTNFRFormalism)
- examples/08_emergent_geometry/119_phase_sector_directed_residue.py (QR spectrum)
- examples/07_number_theory/147_numbers_as_free_monoid_words.py (free monoid)
- examples/08_emergent_geometry/123_symmetry_sector_decomposition.py (Fix sectors)
- AGENTS.md "Operator-Tetrad Synergies" (dual-lever capacity νf vs pressure ΔNFR)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import sympy as sp

from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRParameters,
    power_residue_rank,
    quadratic_residue_annotated_rank,
    residue_network_rank,
)

PARAMS = ArithmeticTNFRParameters()
F = ArithmeticTNFRFormalism


def arithmetic_terms(n):
    """Canonical structural terms (Omega with multiplicity, tau, sigma)."""
    factorisation = sp.factorint(n)
    return ArithmeticStructuralTerms(
        tau=int(sp.divisor_count(n)),
        sigma=int(sp.divisor_sigma(n)),
        omega=int(sum(factorisation.values())),
    )


def delta_nfr(n):
    return F.delta_nfr_value(n, arithmetic_terms(n), PARAMS)


def is_odd_prime(n):
    return n >= 3 and bool(sp.isprime(n))


def experiment_1_two_arm_primality():
    print("=" * 72)
    print("M1: two-arm primality -- pressure DeltaNFR(n)=0  <=>  spectral s_QR(m)=3")
    print("=" * 72)
    disagree = 0
    rank_matches = 0
    checked = 0
    for m in range(3, 50, 2):
        d = delta_nfr(m)
        s = residue_network_rank(m, "quadratic")
        prime = is_odd_prime(m)
        pressure_prime = abs(d) < 1e-9
        spectral_prime = s == 3
        if not (pressure_prime == spectral_prime == prime):
            disagree += 1
        if s == quadratic_residue_annotated_rank(m):
            rank_matches += 1
        checked += 1
    print(
        f"  agreement DeltaNFR=0 <=> s_QR=3 <=> prime: {checked - disagree}"
        f"/{checked} (disagreements {disagree})"
    )
    print(f"  s_QR(m) == multiplicative A(m): {rank_matches}/{checked}")
    # correlations over a wider range using the closed-form rank A(m)
    odd = np.array([m for m in range(3, 600, 2)])
    dnfr = np.array([delta_nfr(int(m)) for m in odd])
    arank = np.array([quadratic_residue_annotated_rank(int(m)) for m in odd])
    omega = np.array([arithmetic_terms(int(m)).omega for m in odd])
    loga = np.log(arank)
    print("  both arms are correlated complexity measures over odd m in [3,599]:")
    print(f"    corr(DeltaNFR, log A) = {np.corrcoef(dnfr, loga)[0, 1]:+.4f}")
    print(f"    corr(log A,   omega)  = {np.corrcoef(loga, omega)[0, 1]:+.4f}")
    print("  -> primality = simultaneous minimum of pressure (DeltaNFR=0) and")
    print("     spectral rank (s_QR=3); both grade factorization complexity.")


def experiment_2_cyclotomy_law():
    print()
    print("=" * 72)
    print("M2: the cyclotomy law  s_k(p) = gcd(k, p-1) + 1")
    print("=" * 72)
    primes = [p for p in range(5, 48) if sp.isprime(p)]
    fails = 0
    for k in range(2, 7):
        for p in primes:
            s = residue_network_rank(p, "power", k)
            if s != power_residue_rank(p, k):
                fails += 1
    total = len(primes) * 5
    print(
        f"  s_k(p) = gcd(k,p-1)+1 for k=2..6, p in {primes[0]}..{primes[-1]}:"
        f" {total - fails}/{total} exact"
    )
    print("  complete-splitting reading: s_k(p) = k+1  <=>  p = 1 (mod k)")
    split_mismatch = 0
    split_total = 0
    for k in [3, 4, 5, 6]:
        for p in primes:
            s = residue_network_rank(p, "power", k)
            if (s == k + 1) != (p % k == 1):
                split_mismatch += 1
            split_total += 1
    print(
        f"    s_k(p)=k+1 <=> p splits completely in Q(zeta_k): "
        f"{split_total - split_mismatch}/{split_total} exact"
    )
    print("  sample (k=3 cubic, k=4 quartic):")
    for p in [7, 11, 13, 17, 19, 23]:
        s3 = residue_network_rank(p, "power", 3)
        s4 = residue_network_rank(p, "power", 4)
        print(
            f"    p={p:>2}: s_3={s3} (p mod 3={p % 3}), " f"s_4={s4} (p mod 4={p % 4})"
        )
    print("  QR is the k=2 case: gcd(2,p-1)=2 -> uniform rank 3 for every odd p.")


def experiment_3_free_monoid_grading():
    print()
    print("=" * 72)
    print("M3: free-monoid exponential grading -- rank = (per-prime rank)^omega")
    print("=" * 72)
    squarefree = [3, 15, 105]  # 3, 3*5, 3*5*7  (robust scalar spectral count)
    print("  squarefree m: QR rank vs 3^omega, unitary rank vs 2^omega")
    print(f"  {'m':>5} {'omega':>5} {'s_QR':>5} {'3^w':>5} {'s_unit':>7} {'2^w':>5}")
    for m in squarefree:
        w = arithmetic_terms(m).omega
        sqr = residue_network_rank(m, "quadratic")
        sun = residue_network_rank(m, "unitary")
        print(
            f"  {m:>5} {w:>5} {sqr:>5} {3 ** w:>5} {sun:>7} {2 ** w:>5}"
            f"   ({'OK' if sqr == 3 ** w and sun == 2 ** w else 'NO'})"
        )
    # the EXACT multiplicative rank extends the pattern to all m (the scalar
    # spectral count is precision-limited for large dense graphs):
    big = 1155  # 3*5*7*11, omega = 4
    print(
        f"  exact A({big}) = {quadratic_residue_annotated_rank(big)} = 3^4 "
        f"(multiplicative rank, robust for every m)"
    )
    print("  the base = connection-set per-prime association-scheme rank (QR=3,")
    print("  unitary=2). Pressure-arm counterpart is the LINEAR zeta*(omega-1):")
    for m in squarefree:
        w = arithmetic_terms(m).omega
        fact_press = PARAMS.zeta * (w - 1)
        print(
            f"    m={m:>5}: spectral 3^omega={3 ** w:>4}  |  "
            f"pressure zeta*(omega-1)={fact_press:.4f}"
        )
    print("  same free-monoid word length omega (example 147), two gradings:")
    print("  spectral EXPONENTIAL (3^omega), pressure LINEAR (zeta*(omega-1)).")


def main():
    print("Example 153 - Structural-Frequency Rank of Arithmetic Diffusion")
    print("Networks: Two-Arm Primality and the Cyclotomy Law")
    print()
    experiment_1_two_arm_primality()
    experiment_2_cyclotomy_law()
    experiment_3_free_monoid_grading()
    print()
    print("Summary: the canonical structural-diffusion rank of arithmetic Cayley")
    print("networks bottoms out at primes on both dual-lever arms (M1), reads")
    print("cyclotomic splitting via s_k(p)=gcd(k,p-1)+1 (M2), and grades the")
    print("free-monoid word length exponentially (M3). Unification of three")
    print("modules; classical objects in TNFR framing; no open problem advanced.")


if __name__ == "__main__":
    main()
