#!/usr/bin/env python3
"""
Example 117 — Emergent Geometry on the Residue Graph (Paley Factorization, Honest)
=================================================================================

Bridges the number-theory arc (examples 100-102, 116) and the
emergent-geometry arc (98-114) by asking the factorization-lab question with
the canonical EMERGENT geometry only: does the structural-diffusion operator
(the literal content of the canonical dNFR) "see" prime/factor structure on
the quadratic-residue graph?

Canonical constraint (doctrine)
-------------------------------
Everything here uses the EMERGENT geometry: the structural-diffusion operator
L_rw = I - D^-1 W is exactly the canonical dNFR EPI channel
(structural_diffusion.py, dNFR = neighbour_mean - self = -L_rw * EPI), and the
symplectic substrate (Phi_s, K_phi, J_dnfr) is populated by the nodal dynamics.
No classical lambda_2 telemetry; the ONLY arithmetic input is x^2 mod n (the
residue-graph topology).

Three measured results (all reproducible below)
-----------------------------------------------
Q1 PRIMALITY (Reading B, non-circular spectral emergence). The emergent
   diffusion spectrum reproduces the Paley/strongly-regular rigidity: primes
   n = 1 mod 4 give a 3-distinct-eigenvalue spectrum (a strongly regular graph
   signature); composites drift to many distinct eigenvalues. This is the
   genuine primes-OUT reading (g(n)=0 of benchmarks/paley_bridge.py), here in
   the emergent operator. Caveat: prime powers (49 = 7^2) also give 3 distinct
   values, so rigidity detects "prime-power-like", not strictly prime.

Q2 FACTORIZATION (factor-OUT, partial). For a semiprime n = p*q the factor p
   appears as an EXACT Fourier mode of the emergent spectrum: the coset-mod-p
   localization eta^2 of a low eigenvector reaches 1.0 for most n = 1 mod 4
   semiprimes, collapsing under a node-label shuffle. The factor is read off
   the eigenvector without ever computing n % k or a gcd. It is PARTIAL: when
   the factor mode sits at high frequency (some n = 3 mod 4) the low-mode scan
   misses it (eta^2 ~ baseline).

Q3 HONEST DOCTRINE (the decisive check). (a) The residue graph is REGULAR /
   circulant, so the emergent random-walk operator L_rw = L_combinatorial / d
   shares the classical Laplacian EIGENVECTORS exactly: the coset signal is the
   residue-graph (CRT) structure re-expressed, NOT something the emergent
   framing adds. (b) The genuinely-emergent symplectic substrate fields
   (Phi_s, K_phi, J_dnfr), populated by the nodal dynamics, are BLIND to the
   cosets (eta^2 ~ 0) - exactly like examples 103/116: the substrate re-expresses
   what lives in the spectrum, it does not independently discover the factor.

Honest scope
------------
This characterizes how the emergent geometry relates to spectral factorization.
The factor signal is the residue-graph spectrum (a classical Paley Gauss-sum
fact) re-expressed in the emergent operator; the emergent per-node substrate is
blind to it. Genuine non-circular emergence (Reading B) EXISTS but is PARTIAL
(misses n = 2 and many n = 3 mod 4) and lives in the real/self-adjoint spectral
sector - the same e-pi / Fix(G)^perp wall as the paused TNFR-Riemann program.
It does NOT factor arbitrary n, does NOT close any open problem.

References
----------
- src/tnfr/physics/structural_diffusion.py (emergent operator, structural_eigenmodes)
- src/tnfr/physics/symplectic_substrate.py (substrate fields)
- factorization-lab/ (the spectral Paley factorizer this characterizes)
- benchmarks/paley_bridge.py, benchmarks/primes_as_consequence.py (Reading B)
- examples/08_emergent_geometry/103_emergent_substrate_meets_riemann.py
- examples/07_number_theory/116_nuf_emergent_prime_visibility.py
- AGENTS.md "Transport Content of the Nodal Equation" (L_rw = emergent dNFR)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import structural_eigenmodes
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def quadratic_residues(n: int) -> set[int]:
    """Nonzero quadratic residues mod n - the ONLY arithmetic input."""
    return {(x * x) % n for x in range(1, n)} - {0}


def residue_graph(n: int) -> nx.Graph:
    """Undirected residue graph: edge (i,j) iff (i-j) mod n is a QR.

    For prime n = 1 mod 4 this is the Paley graph. Regular/circulant by
    construction, so the emergent random-walk operator and the classical
    Laplacian share eigenvectors.
    """
    R = quadratic_residues(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = (i - j) % n
            if d in R or (n - d) in R:
                G.add_edge(i, j)
    return G


def coset_eta2(vec: np.ndarray, n: int, p: int) -> float:
    """Variance fraction of an eigenvector explained by the coset label i mod p.

    eta^2 = between-coset variance / total variance. ~1 => the mode is a pure
    function of (i mod p) (the factor signature); ~1/p is the random baseline.
    """
    labels = np.array([i % p for i in range(n)])
    v = np.asarray(vec, float)
    grand = v.mean()
    total = float(np.sum((v - grand) ** 2))
    if total < 1e-15:
        return 0.0
    between = sum(
        (labels == c).sum() * (v[labels == c].mean() - grand) ** 2
        for c in range(p)
        if (labels == c).sum()
    )
    return float(between / total)


def best_coset_eta2(eigvecs: np.ndarray, n: int, p: int, k: int = 8) -> float:
    """Max coset localization over the k lowest non-trivial emergent modes."""
    return max(
        (
            coset_eta2(eigvecs[:, j], n, p)
            for j in range(1, min(k + 1, eigvecs.shape[1]))
        ),
        default=0.0,
    )


def _seed_and_evolve(
    G: nx.Graph, steps: int = 12, dt: float = 0.05, seed: int = 1
) -> None:
    """Populate the emergent substrate by running the nodal equation."""
    rng = np.random.default_rng(seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.3, 0.3)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)
    for _ in range(steps):
        default_compute_delta_nfr(G)
        for nd in G.nodes():
            e = float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0))
            v = float(get_attr(G.nodes[nd], ALIAS_VF, 0.0))
            d = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            set_attr(G.nodes[nd], ALIAS_EPI, e + dt * v * d)
    default_compute_delta_nfr(G)


def experiment_1_primality_rigidity():
    """Q1: emergent diffusion spectrum reproduces Paley/SRG rigidity."""
    print("=" * 74)
    print("EXPERIMENT 1: Primality Rigidity in the Emergent Diffusion Spectrum")
    print("=" * 74)
    print()
    print("The emergent operator L_rw = I - D^-1 W (canonical dNFR EPI channel).")
    print("Strongly-regular Paley primes (n = 1 mod 4) -> 3 distinct eigenvalues.")
    print()
    print(f"  {'n':>4} {'class':>9} {'n_distinct':>11} {'rigid?':>7}")
    cases = [
        (13, "prime"),
        (29, "prime"),
        (37, "prime"),
        (53, "prime"),
        (21, "3*7"),
        (33, "3*11"),
        (65, "5*13"),
        (25, "5^2"),
        (49, "7^2"),
    ]
    for n, cls in cases:
        ev, _ = structural_eigenmodes(residue_graph(n))
        distinct = len(np.unique(np.round(ev, 6)))
        rigid = "YES" if distinct == 3 else "no"
        print(f"  {n:>4} {cls:>9} {distinct:>11} {rigid:>7}")
    print()
    print("-> primes n=1 mod4 are rigid (3 distinct); composites drift. Reading B")
    print("   (primes-OUT) in the emergent spectrum. Caveat: 49=7^2 is also rigid")
    print("   (prime-power-like), so rigidity != strict primality.")
    print()


def experiment_2_factor_cosets():
    """Q2: the factor p is an exact Fourier mode of the emergent spectrum."""
    print("=" * 74)
    print("EXPERIMENT 2: Factor Recovery as Coset Localization (factor-OUT)")
    print("=" * 74)
    print()
    print("For n=p*q, does a low emergent eigenvector localize on cosets mod p?")
    print("eta^2 ~ 1 => the factor is read off the eigenvector (no n%k, no gcd).")
    print()
    print(
        f"  {'n=p*q':>9} {'p':>3} {'eta2(mod p)':>12} {'baseline':>9} "
        f"{'shuffled':>9} {'verdict':>8}"
    )
    rng = np.random.default_rng(0)
    cases = [
        (21, 3, 7),
        (33, 3, 11),
        (65, 5, 13),
        (85, 5, 17),
        (77, 7, 11),
        (57, 3, 19),
        (51, 3, 17),
        (91, 7, 13),
    ]
    for n, p, q in cases:
        _, vecs = structural_eigenmodes(residue_graph(n))
        eta = best_coset_eta2(vecs, n, p)
        base = 1.0 / p
        eta_shuf = best_coset_eta2(vecs[rng.permutation(n)], n, p)
        # the shuffle control is the real test: strong localization that the
        # label permutation destroys.
        verdict = "SIGNAL" if (eta > 0.5 and eta > 4 * eta_shuf) else "miss"
        print(
            f"  {f'{n}={p}*{q}':>9} {p:>3} {eta:>12.4f} {base:>9.4f} "
            f"{eta_shuf:>9.4f} {verdict:>8}"
        )
    print()
    print("-> the factor p appears as an EXACT coset mode (eta^2=1) for most")
    print("   n=1 mod4 semiprimes, collapsing under shuffle. PARTIAL: when the")
    print("   factor mode is high-frequency (51, 91) the low-mode scan misses it.")
    print()


def experiment_3_doctrine_check():
    """Q3: regular graph => emergent=classical eigenvectors; substrate blind."""
    print("=" * 74)
    print("EXPERIMENT 3: Honest Doctrine Check (regularity + substrate blindness)")
    print("=" * 74)
    print()
    print("(a) Residue graph regularity (spread 0 => L_rw = L_classical / d, so")
    print("    the emergent operator SHARES the classical eigenvectors):")
    for n in (21, 65, 85):
        degs = [d for _, d in residue_graph(n).degree()]
        tag = "REGULAR" if max(degs) == min(degs) else "irregular"
        print(
            f"    n={n:>3}: degree {min(degs)}..{max(degs)} "
            f"(spread {max(degs) - min(degs)}) -> {tag}"
        )
    print()
    print("    -> the coset signal is the CRT structure of the residue graph")
    print("       re-expressed; the emergent framing does not add it.")
    print()
    print("(b) Symplectic substrate (dynamics-populated) coset localization:")
    print(
        f"    {'n=p*q':>9} {'p':>3} {'eta2(diff)':>11} {'eta2(Phi_s)':>12} "
        f"{'eta2(K_phi)':>12} {'eta2(J_dnfr)':>13}"
    )
    for n, p, q in [(21, 3, 7), (65, 5, 13), (85, 5, 17)]:
        G = residue_graph(n)
        _, vecs = structural_eigenmodes(G)
        eta_diff = best_coset_eta2(vecs, n, p)
        _seed_and_evolve(G)
        pt = extract_phase_space_point(G)
        idx = {nd: i for i, nd in enumerate(pt.nodes)}
        phis = np.array([pt.phi_s[idx[i]] for i in range(n)])
        kphi = np.array([pt.k_phi[idx[i]] for i in range(n)])
        jd = np.array([pt.j_dnfr[idx[i]] for i in range(n)])
        print(
            f"    {f'{n}={p}*{q}':>9} {p:>3} {eta_diff:>11.4f} "
            f"{coset_eta2(phis, n, p):>12.4f} {coset_eta2(kphi, n, p):>12.4f} "
            f"{coset_eta2(jd, n, p):>13.4f}"
        )
    print()
    print("    -> diffusion eigenvectors carry the factor (eta2~1); the emergent")
    print("       per-node substrate fields are BLIND (eta2~0). The substrate")
    print("       re-expresses the spectrum, it does not discover the factor.")
    print()


def main():
    print()
    print("  TNFR Example 117: Emergent Geometry on the Residue Graph")
    print("  Paley factorization, honestly: spectrum carries it, substrate blind")
    print("  ==================================================================")
    print()
    experiment_1_primality_rigidity()
    experiment_2_factor_cosets()
    experiment_3_doctrine_check()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print()
    print("Using the EMERGENT geometry for everything (diffusion operator +")
    print("symplectic substrate), the factor signal lives in the residue-graph")
    print("SPECTRUM (a classical Paley Gauss-sum fact), which the emergent")
    print("operator re-expresses exactly because the graph is regular. The")
    print("genuinely-emergent per-node substrate is BLIND to the factor. This")
    print("unifies factorization-lab with the emergent-geometry arc: Reading B")
    print("(non-circular primes-OUT) is real but PARTIAL and spectral; the")
    print("substrate adds no factoring power. Same e-pi / Fix(G)^perp wall as")
    print("the paused Riemann program. No open problem is closed.")
    print()


if __name__ == "__main__":
    main()
