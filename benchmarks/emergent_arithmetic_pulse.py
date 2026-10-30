"""Finite power-residue spectra and an auxiliary graph-wave comparison.

The graphs Cay(Z/n, R_k) are built from supplied modular power residues.
The directed normalized-Laplacian distinct-eigenvalue count is compared with
``power_residue_rank`` on the listed prime inputs. The prime cyclotomy formula
s_k(p)=gcd(k,p-1)+1 belongs to the number-theory owner; finite numerical matches
do not prove it or select this arithmetic encoding from the nodal equation.

For the separate undirected Paley examples, ``compute_emergent_pulse`` assigns
wave frequencies sqrt(lambda_k) to a supplied second-order graph-wave model.
A directed complex eigenvalue is not automatically a real oscillation tone.
No phase trajectory or physical vibration is measured here. The remaining
loops display selected composite ranks and quadratic-residue prime ranks;
they do not establish a universal multiplicativity or a maximal-degeneracy
criterion among graphs, identify factors, or classify physical particles.

The selected numerical cutoff and eigenvalue rounding remain those of the
shared spectral owners. Read ``theory/TNFR_NUMBER_THEORY.md`` for conditional
arithmetic results and ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` for the limit
of symmetry/spectrum interpretations. No representation identifying analytic
arg-zeta with a finite symmetry complement is supplied.

Run ``python benchmarks/emergent_arithmetic_pulse.py`` for this finite scan.
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
    """Numerical distinct-eigenvalue count on a supplied residue digraph."""
    graph = arithmetic_cayley_digraph(p, power_residue_set(p, k))
    return structural_frequency_rank(graph)


def undirected_residue_graph(p, k):
    """Undirected support of Cay(Z/p, R_k); nonsymmetric R_k is symmetrized."""
    conn = set(power_residue_set(p, k))
    graph = nx.Graph()
    graph.add_nodes_from(range(p))
    for a in range(p):
        for c in conn:
            graph.add_edge(a, (a + c) % p)
    return graph


def main() -> None:
    print("=" * 72)
    print("ARITHMETIC SPECTRAL CONTROLS -- supplied power-residue graphs")
    print("=" * 72)

    # M1 -- compare selected directed spectral ranks with the prime formula.
    print("\nM1 -- directed distinct-eigenvalue counts:")
    all_ok = True
    for k in (2, 3, 4, 5):
        row_ok = all(residue_tones(p, k) == power_residue_rank(p, k) for p in PRIMES)
        all_ok = all_ok and row_ok
        status = "all = gcd(k,p-1)+1" if row_ok else "MISMATCH"
        print(f"   k={k}: {status}  (p=37 -> rank {residue_tones(37, k)})")
    print(f"   => prime formula matches on the listed p and k: {all_ok}")

    # M2 -- separate undirected graph-wave readout on the listed Paley graphs.
    print("\nM2 -- undirected Paley auxiliary wave frequencies (p = 1 mod 4):")
    for p in [5, 13, 17, 29, 37, 41]:
        graph = undirected_residue_graph(p, 2)
        pulse = compute_emergent_pulse(graph, n_modes=p)
        tones = sorted({round(x, 5) for x in pulse["resonant_spectrum"]})
        rank = structural_frequency_rank(graph)
        mult = pulse["spectral_multiplicity"]
        print(
            f"   p={p:>2}: rank={rank} (zero + 2 distinct positive values)  frequencies={tones}  "
            f"mult={mult} (=(p-1)/2={(p - 1) // 2})"
        )

    # M3 -- selected composites only; no universal rank multiplicativity test.
    print("\nM3 -- selected composite ranks (factorizations supplied in notes):")
    notes = {15: " = 3x3 (3*5)", 45: " = 4x3 (9*5)"}
    for m in [9, 15, 21, 25, 35, 45, 49]:
        tones = residue_tones(m, 2)
        print(
            f"   m={m:>2} (composite): rank={tones}{notes.get(m, '')}  "
            f"(comparison: odd-prime quadratic-residue rank = 3)"
        )

    # M4 -- fixed k=2 on explicitly supplied odd-prime sizes.
    print("\nM4 -- quadratic-residue rank on the listed odd primes:")
    for p in [11, 23, 47, 59, 83, 107]:
        tones = residue_tones(p, 2)
        print(
            f"   p={p:>3} (prime={isprime(p)}): {tones} distinct eigenvalues over {p} "
            f"nodes -> mean multiplicity ~{p / tones:.1f}"
        )

    print("\n" + "=" * 72)
    print(
        f"SUMMARY: selected prime-formula checks passed={all_ok}.\n"
        "Other rows display finite spectra and auxiliary undirected wave readouts.\n"
        "The scan supplies arithmetic graphs; it establishes no physical vibration,\n"
        "general primality test, universal rank multiplicativity or RH bridge."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
