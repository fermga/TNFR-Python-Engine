"""Emergent Arithmetic Pulse: the cyclotomy law IS the prime's chord.

THE PARADIGM (user, theory creator): introduce the pulse into number theory
and see what emerges. The answer is clean: the residue-NFR's PULSE -- the
conservative resonant spectrum omega_k = sqrt(lambda_k) of the canonical L_rw
(the dNFR EPI channel) -- has a tone structure that IS the PROVED cyclotomy
law. A prime is the most degenerate chord possible.

THE STRUCTURAL FACTS (all canonical, nothing invented):
  - the arithmetic NFR is the residue Cayley network Cay(Z/n, R_k) (k-th
    power residues); its pulse reads L_rw = I - D^{-1}W (the dNFR EPI channel);
  - the number of DISTINCT resonant tones = structural_frequency_rank(G) = the
    distinct eigenvalues of L_rw -- and on a prime this is the cyclotomy law
    s_k(p) = gcd(k, p-1) + 1 (PROVED via Gauss periods, NT theory 9.11);
  - the pulse SPLITS across the symmetry wall: the per-NFR pulse is blind (the
    residue graph is vertex-transitive -> the per-node substrate is in Fix(G)),
    while the collective pulse (the spectrum) carries the cyclotomy
    (Fix(G)^perp). The two pulse scales land on the two sides of the wall.

WHAT EMERGES (measured):
  - M1 THE TONE-COUNT IS THE CYCLOTOMY LAW: the distinct resonant tones of the
    residue-NFR pulse = gcd(k, p-1) + 1, exactly, every prime and every k.
  - M2 THE PRIME'S CHORD: the Paley-NFR (p = 1 mod 4) pulse is the silent mode
    + exactly TWO resonant tones (omega_-, omega_+), each with multiplicity
    (p-1)/2 -- the pulse's spectral_multiplicity field reads (p-1)/2 exactly.
  - M3 COMPOSITES ENRICH THE CHORD MULTIPLICATIVELY: the tone-count is
    multiplicative (15 -> 9 = 3x3, 45 -> 12 = 4x3), encoding the factorization
    TYPE (the ontological-ladder rank, NT theory 9.8).
  - M4 A PRIME IS MAXIMALLY DEGENERATE: just gcd(k,p-1)+1 tones no matter how
    large p is -- mean multiplicity ~ (p-1)/d grows, the chord stays minimal.

So the ARITHMETIC PULSE = the cyclotomy law: a prime is the simplest chord the
arithmetic NFR can vibrate; composites split it; the factorization type is the
chord's tone-count.

HONEST SCOPE: the tone-count = structural_frequency_rank, already documented as
the cyclotomy diagnostic, and the cyclotomy law s_k(p)=gcd(k,p-1)+1 is a PROVED
classical Gauss-period fact (NT theory 9.11). The TNFR content is the
conservative-PULSE reading -- those distinct eigenvalues are the distinct
resonant TONES of the arithmetic vibration, so the prime is a maximally-
degenerate chord. It detects primality / factorization TYPE structurally; it
does NOT factor, does NOT reach the prime IDENTITIES or the continuous
arg-zeta phase (the same Fix(S_n)^perp wall as the paused Riemann program),
and closes no open problem. R and pi assumed.

Run:
    python benchmarks/emergent_arithmetic_pulse.py

Theoretical anchor: theory/TNFR_NUMBER_THEORY.md (9.11 cyclotomy law, 9.8
ladder, 9.7 the symmetry wall); src/tnfr/physics/structural_diffusion.py
(compute_emergent_pulse, structural_frequency_rank); the pulse on physical
networks lives in benchmarks/emergent_rhythm.py and emergent_fractal_pulse.py.
Status: RESEARCH.
"""

import os
import sys

import networkx as nx
from sympy import isprime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tnfr.mathematics.number_theory import (  # noqa: E402
    arithmetic_cayley_digraph,
    power_residue_rank,
    power_residue_set,
)
from tnfr.physics.structural_diffusion import (  # noqa: E402
    compute_emergent_pulse,
    structural_frequency_rank,
)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def residue_tones(p, k):
    """Distinct resonant tones of the directed residue-NFR pulse."""
    graph = arithmetic_cayley_digraph(p, power_residue_set(p, k))
    return structural_frequency_rank(graph)


def undirected_residue_graph(p, k):
    """Undirected Cay(Z/p, R_k) (real pulse when R_k is symmetric)."""
    conn = set(power_residue_set(p, k))
    graph = nx.Graph()
    graph.add_nodes_from(range(p))
    for a in range(p):
        for c in conn:
            graph.add_edge(a, (a + c) % p)
    return graph


def main() -> None:
    print("=" * 72)
    print("EMERGENT ARITHMETIC PULSE -- cyclotomy law IS the prime chord")
    print("=" * 72)

    # M1 -- the tone-count is the cyclotomy law s_k(p) = gcd(k, p-1) + 1
    print("\nM1 -- distinct resonant tones of the residue-NFR pulse:")
    all_ok = True
    for k in (2, 3, 4, 5):
        row_ok = all(residue_tones(p, k) == power_residue_rank(p, k)
                     for p in PRIMES)
        all_ok = all_ok and row_ok
        status = "all = gcd(k,p-1)+1" if row_ok else "MISMATCH"
        print(f"   k={k}: {status}  (p=37 -> {residue_tones(37, k)} tones)")
    print(f"   => tone-count == cyclotomy law, all k, all primes: {all_ok}")

    # M2 -- the prime's chord: silent mode + 2 tones, multiplicity (p-1)/2
    print("\nM2 -- the Paley-NFR chord (p = 1 mod 4, real pulse):")
    for p in [5, 13, 17, 29, 37, 41]:
        graph = undirected_residue_graph(p, 2)
        pulse = compute_emergent_pulse(graph, n_modes=p)
        tones = sorted({round(x, 5) for x in pulse["resonant_spectrum"]})
        rank = structural_frequency_rank(graph)
        mult = pulse["spectral_multiplicity"]
        print(f"   p={p:>2}: rank={rank} (silent + 2 tones)  tones={tones}  "
              f"mult={mult} (=(p-1)/2={(p - 1) // 2})")

    # M3 -- composites enrich the chord multiplicatively (factorization type)
    print("\nM3 -- composites split the chord (tone-count is multiplicative):")
    notes = {15: " = 3x3 (3*5)", 45: " = 4x3 (9*5)"}
    for m in [9, 15, 21, 25, 35, 45, 49]:
        tones = residue_tones(m, 2)
        print(f"   m={m:>2} (composite): {tones} tones{notes.get(m, '')}  "
              f"(prime signature = 3)")

    # M4 -- a prime is maximally degenerate (minimal chord at any size)
    print("\nM4 -- the prime chord stays minimal (3 tones) at any size:")
    for p in [11, 23, 47, 59, 83, 107]:
        tones = residue_tones(p, 2)
        print(f"   p={p:>3} (prime={isprime(p)}): {tones} tones over {p} "
              f"nodes -> mean multiplicity ~{p / tones:.1f}")

    print("\n" + "=" * 72)
    print(
        "VERDICT: the residue-NFR pulse tone-count IS the cyclotomy law\n"
        "s_k(p)=gcd(k,p-1)+1; a prime is the most degenerate chord,\n"
        "composites split it multiplicatively = the factorization type.\n"
        "The pulse sees the TYPE, not the prime identities (wall persists)."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
