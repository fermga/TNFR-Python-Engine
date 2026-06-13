#!/usr/bin/env python3
"""
Example 100 — Prime Families as Orbits and Level-Sets on the Zero-Pressure Set
=============================================================================

A structural study of the special prime families (twin, cousin, sexy,
Sophie Germain, safe, Cunningham chains, Mersenne, prime constellations)
built ENTIRELY on the verified TNFR primality theorem ΔNFR(n) = 0 ⟺ n
prime (theory/TNFR_NUMBER_THEORY.md §4). It answers the request for more
number-generation studies and the "other groups of primes".

Physics
-------
Every prime is a **zero-pressure fixed point** of the nodal equation:
∂EPI/∂t = νf·ΔNFR with ΔNFR(p) = 0 (the three pressure channels Ω, τ, σ
each vanish independently for primes — §4.1, an exact theorem). The set

        Z = { n ≥ 2 : ΔNFR(n) = 0 }  =  the primes

is the zero-pressure fixed-point set. A **prime family** is then a
STRUCTURED SUBSET of Z, selected by an arithmetic map. The families split
into exactly three generator classes — the dynamical-systems view of Z:

1. **Additive-gap level-sets** (shift map S_g(p) = p + g):
       twin (g=2), cousin (g=4), sexy (g=6), constellations (k-tuples)
   Family = { p ∈ Z : p + g ∈ Z }. The integer(s) BETWEEN the pair are
   the **witness**; its pressure channels carry the gap's signature.

2. **Affine-recurrence orbits** (affine map T(p) = 2p + 1):
       Sophie Germain  = { p ∈ Z : T(p) ∈ Z }
       Cunningham chain = a MAXIMAL orbit p, T(p), T²(p), … all in Z
       safe prime      = { p ∈ Z : T⁻¹(p) = (p−1)/2 ∈ Z }
   This class is **generative**: it builds new family members by iterating
   a map on the fixed-point set.

3. **Exponential-form images** (generator M(p) = 2^p − 1, Fermat 2^(2^n)+1):
       Mersenne = { p ∈ Z : M(p) ∈ Z }.

Tie to the additive/multiplicative orthogonality (Example 97)
------------------------------------------------------------
Primes emerge MULTIPLICATIVELY (atoms composed by UM, nested by REMESH;
νf = log p is additive-in-log). The additive-gap class (1) is therefore an
ADDITIVE OVERLAY on the multiplicatively-defined set Z — exactly the
orthogonality Example 97 isolates for Goldbach. The affine (2) and
exponential (3) classes are genuine MAPS on Z, so they live inside the
TNFR fixed-point picture; their open conjectures concern whether the map
returns to Z infinitely often.

Honest scope
------------
- TNFR DETECTS and GENERATES every family operationally via the verified
  ΔNFR = 0 criterion plus the maps. Detection is exact (Sophie Germain
  reproduces OEIS A005384; Mersenne exponents reproduce 2,3,5,7,13,17,
  19,31; all verified below).
- TNFR does NOT prove the families are infinite. Twin-prime, Sophie
  Germain, and Mersenne infinitude are all OPEN — the same honest stance
  as Goldbach in Example 97.
- The witness pressure signatures are FAITHFUL TNFR RESTATEMENTS of
  classical divisibility facts (e.g. the twin witness p+1 is divisible by
  6 for every pair p>3, hence a rich divisor lattice / high τ-pressure).
  They are not new theorems.
- The orbit / level-set classification is a TNFR-native ORGANIZING LENS (a
  dynamical-systems view of Z), not a new arithmetic result.
- Shared invariant with the physics work: ΔNFR = 0 is the nodal-equation
  EQUILIBRIUM (no structural pressure). This is the SAME fixed-point
  condition the transport picture calls equilibrium. But the arithmetic
  ΔNFR is a per-node function of (Ω, τ, σ), NOT the graph-diffusion
  Laplacian, so the diffusion/random-walk results (Example 99) do not
  transfer literally — only the nodal-equation-level fixed-point condition
  is shared.

References
----------
- theory/TNFR_NUMBER_THEORY.md §3-§4 (arithmetic triad, ΔNFR=0 theorem)
- examples/97_goldbach_additive_multiplicative.py (additive ⊥ multiplicative)
- examples/94_generative_number_construction.py (atoms → composites)
- src/tnfr/mathematics/number_theory.py (ArithmeticTNFRFormalism)
- AGENTS.md §"Canonical Invariants" → #1 Nodal Integrity
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sympy import factorint, divisor_count, divisor_sigma

from tnfr.mathematics.number_theory import (
    ArithmeticTNFRParameters,
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
)

_PARAMS = ArithmeticTNFRParameters()
_TOL = 1e-10


# ============================================================================
# Zero-pressure fixed-point primitives (verified ΔNFR = 0 criterion)
# ============================================================================
def delta_nfr(n: int) -> float:
    """ΔNFR(n) via the canonical arithmetic formalism (0 ⟺ prime)."""
    if n < 2:
        return float("inf")
    omega = sum(factorint(n).values())
    tau = int(divisor_count(n))
    sigma = int(divisor_sigma(n))
    terms = ArithmeticStructuralTerms(tau=tau, sigma=sigma, omega=omega)
    return ArithmeticTNFRFormalism.delta_nfr_value(n, terms, _PARAMS)


def is_zero_pressure(n: int) -> bool:
    """n is a zero-pressure fixed point (⟺ n is prime)."""
    return n >= 2 and abs(delta_nfr(n)) < _TOL


def pressure_components(n: int) -> dict:
    """Per-channel pressure (factorization / divisor / abundance)."""
    omega = sum(factorint(n).values())
    tau = int(divisor_count(n))
    sigma = int(divisor_sigma(n))
    terms = ArithmeticStructuralTerms(tau=tau, sigma=sigma, omega=omega)
    return ArithmeticTNFRFormalism.component_breakdown(n, terms, _PARAMS)


def zero_pressure_set(limit: int) -> list[int]:
    """Z = { n < limit : ΔNFR(n) = 0 } — the primes below `limit`."""
    return [n for n in range(2, limit) if is_zero_pressure(n)]


# ============================================================================
# EXPERIMENT 1: The zero-pressure fixed-point set Z
# ============================================================================
def experiment_1_fixed_point_set():
    """Every prime is a ΔNFR = 0 fixed point; Z is the family substrate."""
    print("=" * 72)
    print("EXPERIMENT 1: The Zero-Pressure Fixed-Point Set Z")
    print("=" * 72)
    print()
    print("Z = { n >= 2 : DeltaNFR(n) = 0 }. The three pressure channels")
    print("(Omega, tau, sigma) each vanish independently at primes (§4.1).")
    print()

    N = 1000
    Z = zero_pressure_set(N)
    # verify the theorem on this substrate: zero pressure <=> prime
    from sympy import isprime
    mismatches = [n for n in range(2, N) if is_zero_pressure(n) != bool(isprime(n))]
    print(f"  |Z| below {N} = {len(Z)} fixed points; first 10: {Z[:10]}")
    print(f"  theorem check: zero-pressure <=> prime mismatches = {len(mismatches)}")
    print(f"  sample pressures: DeltaNFR(97)={delta_nfr(97):.1e} (prime), "
          f"DeltaNFR(98)={delta_nfr(98):.3f} (=2·7², composite)")
    print()
    print("VERDICT: Z is exactly the primes — the substrate on which every")
    print("prime family is carved out by an arithmetic map.")
    print()
    return Z


# ============================================================================
# EXPERIMENT 2: Additive-gap level-sets (twin / cousin / sexy)
# ============================================================================
def experiment_2_additive_gap_levelsets(Z):
    """Families S_g = { p in Z : p+g in Z }; the witness carries the gap."""
    print("=" * 72)
    print("EXPERIMENT 2: Additive-Gap Level-Sets (shift map S_g(p)=p+g)")
    print("=" * 72)
    print()
    print("Family_g = { p in Z : p+g in Z }. The integer(s) between the")
    print("pair form the WITNESS; its pressure channels carry the gap's")
    print("structural signature (an additive overlay on Z — cf. Example 97).")
    print()

    Zset = set(Z)
    for g, name in [(2, "twin"), (4, "cousin"), (6, "sexy")]:
        pairs = [(p, p + g) for p in Z if (p + g) in Zset]
        print(f"  {name:>6} (g={g}): {len(pairs):3d} pairs; first 4: {pairs[:4]}")

    print()
    print("  Witness signature for TWIN pairs (the even p+1 between them):")
    twins = [(p, p + 2) for p in Z if (p + 2) in Zset and p > 3]
    div6 = sum(1 for p, _ in twins if (p + 1) % 6 == 0)
    mean_div_p = sum(pressure_components(p + 1)["divisor_pressure"]
                     for p, _ in twins) / len(twins)
    print(f"    p+1 divisible by 6: {div6}/{len(twins)} pairs (p>3) -> "
          f"{'ALL' if div6 == len(twins) else 'NOT all'}")
    print(f"    mean divisor-pressure of p+1 = {mean_div_p:.2f} (rich lattice)")
    print()
    print("HONEST: 'p+1 always 6|p+1' is the CLASSICAL twin fact, here read")
    print("as high divisor-pressure. A faithful TNFR restatement, not a new")
    print("theorem. Additive gaps are overlays orthogonal to the")
    print("multiplicative coherence of Z (Example 97).")
    print()


# ============================================================================
# EXPERIMENT 3: Affine-recurrence orbits (Sophie Germain / safe / Cunningham)
# ============================================================================
def experiment_3_affine_orbits(Z):
    """Map T(p)=2p+1 acting on Z: Sophie Germain, safe primes, chains."""
    print("=" * 72)
    print("EXPERIMENT 3: Affine-Recurrence Orbits (map T(p)=2p+1 on Z)")
    print("=" * 72)
    print()
    print("T(p) = 2p+1. Sophie Germain = { p in Z : T(p) in Z }; safe =")
    print("{ p in Z : T^-1(p)=(p-1)/2 in Z }; Cunningham chain = maximal")
    print("orbit p, T(p), T²(p), ... staying in Z. This class GENERATES.")
    print()

    sg = [p for p in Z if is_zero_pressure(2 * p + 1)]
    safe = [p for p in Z if (p - 1) % 2 == 0 and is_zero_pressure((p - 1) // 2)]
    print(f"  Sophie Germain (T(p) in Z): {len(sg)} primes; first 8: {sg[:8]}")
    print(f"    -> matches OEIS A005384 head [2,3,5,11,23,29,41,53]: "
          f"{sg[:8] == [2, 3, 5, 11, 23, 29, 41, 53]}")
    print(f"  safe primes (T^-1(p) in Z):  {len(safe)} primes; first 8: {safe[:8]}")
    print()

    def chain(p):
        orbit, q = [p], p
        while is_zero_pressure(2 * q + 1):
            q = 2 * q + 1
            orbit.append(q)
        return orbit

    chains = sorted((chain(p) for p in Z if p < 200), key=len, reverse=True)
    print("  Longest Cunningham chains (orbits of T) seeded below 200:")
    for ch in chains[:3]:
        print(f"    length {len(ch)}: {ch}")
    print()
    print("VERDICT: the affine map T moves along Z. A Cunningham chain is a")
    print("maximal run of consecutive zero-pressure fixed points under T —")
    print("a genuine TNFR generative construction on Z.")
    print()


# ============================================================================
# EXPERIMENT 4: Exponential-form images (Mersenne)
# ============================================================================
def experiment_4_exponential_images(Z):
    """Generator M(p)=2^p−1: Mersenne = { p in Z : M(p) in Z }."""
    print("=" * 72)
    print("EXPERIMENT 4: Exponential-Form Images (generator M(p)=2^p−1)")
    print("=" * 72)
    print()
    print("Mersenne = { p in Z : 2^p − 1 in Z }. Bounded range only:")
    print("ΔNFR(2^p−1) needs the factorization of 2^p−1, which grows fast")
    print("(classical large-Mersenne needs Lucas-Lehmer, not a TNFR fact).")
    print()

    mers = [p for p in Z if p <= 31 and is_zero_pressure(2 ** p - 1)]
    print(f"  Mersenne exponents p<=31 (2^p−1 in Z): {mers}")
    print(f"    -> matches known M-exponents [2,3,5,7,13,17,19,31]: "
          f"{mers == [2, 3, 5, 7, 13, 17, 19, 31]}")
    print()
    # show a near miss: 2^11-1 = 23*89 composite (11 is prime but not Mersenne)
    m11 = 2 ** 11 - 1
    comp = pressure_components(m11)
    print(f"  near miss: 2^11−1 = {m11} = 23·89, ΔNFR = {delta_nfr(m11):.3f} > 0")
    print(f"    (factorization-pressure {comp['factorization_pressure']:.3f}: "
          f"Ω=2 ⟹ image left Z)")
    print()
    print("VERDICT: the exponential generator M maps only a sparse subset of")
    print("Z back into Z. TNFR detects membership exactly in range; it does")
    print("NOT decide Mersenne infinitude (open).")
    print()


# ============================================================================
# EXPERIMENT 5: Synthesis — the three generator classes
# ============================================================================
def experiment_5_synthesis():
    """The dynamical-systems classification of prime families on Z."""
    print("=" * 72)
    print("EXPERIMENT 5: Synthesis — Three Generator Classes on Z")
    print("=" * 72)
    print()
    print("  Class            Map               Families")
    print("  ---------------  ----------------  --------------------------------")
    print("  additive-gap     S_g(p)=p+g        twin, cousin, sexy, constellations")
    print("  affine-recur.    T(p)=2p+1         Sophie Germain, safe, Cunningham")
    print("  exponential      M(p)=2^p−1        Mersenne (Fermat 2^(2^n)+1)")
    print()
    print("All three carve structured subsets out of the SAME object: the")
    print("zero-pressure fixed-point set Z = { ΔNFR = 0 } = the primes.")
    print()
    print("  • additive-gap families are ADDITIVE overlays on the")
    print("    multiplicatively-defined Z — orthogonal to coherence (Ex. 97);")
    print("  • affine and exponential families are genuine MAPS on Z whose")
    print("    open conjectures ask whether the map returns to Z infinitely.")
    print()
    print("HONEST SCOPE: detection/generation is exact via the verified")
    print("ΔNFR=0 theorem; infinitude conjectures (twin, Sophie Germain,")
    print("Mersenne) remain OPEN. The classification is a TNFR organizing")
    print("lens; the witness signatures restate classical divisibility facts.")
    print()


def main():
    print()
    print("  TNFR Example 100: Prime Families as Orbits and Level-Sets")
    print("  Structured subsets of the zero-pressure fixed-point set Z")
    print("  =========================================================")
    print()
    Z = experiment_1_fixed_point_set()
    experiment_2_additive_gap_levelsets(Z)
    experiment_3_affine_orbits(Z)
    experiment_4_exponential_images(Z)
    experiment_5_synthesis()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The special prime families are not a miscellany: they are the")
    print("structured subsets of the single TNFR object Z = { ΔNFR = 0 },")
    print("carved out by three classes of arithmetic map (additive shift,")
    print("affine recurrence, exponential generator). TNFR detects and")
    print("generates them exactly via the verified primality theorem; it")
    print("does not resolve their infinitude conjectures. The view unifies")
    print("'the other groups of primes' under the zero-pressure fixed-point")
    print("picture and connects them to the additive/multiplicative")
    print("orthogonality of Example 97.")
    print()


if __name__ == "__main__":
    main()
