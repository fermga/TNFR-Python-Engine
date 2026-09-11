#!/usr/bin/env python3
"""
Example 117 — Finite Spectral Probes on Quadratic-Residue Graphs
===============================================================

The random-walk Laplacian is the operator of the isolated EPI diffusion
channel.  This example applies it to graphs whose edges were constructed from
``x*x mod n`` and records three finite observations.

1. Four sampled Paley primes have three distinct eigenvalues, while the
   sampled composites vary; ``49`` is a direct counterexample to treating this
   predicate as a primality test.
2. Given a known candidate factor ``p``, ``eta^2`` measures whether selected
   eigenvectors are constant on the supplied ``i mod p`` classes.  Supplying
   ``p`` and computing these classes makes this a factor-conditioned
   localization audit, not factor recovery.  Degenerate eigenspaces can also
   change individual eigenvectors, so the table is tied to the returned basis.
3. Extracted auxiliary substrate fields are compared on three seeded evolved
   snapshots.  They do not reproduce the perfect localization of selected
   diffusion eigenvectors in this table.  This finite observation is not a
   general blindness theorem.

Regularity explains why the random-walk and combinatorial Laplacians share
eigenspaces on these graphs.  The auxiliary symplectic substrate remains a
declared ambient read-out initialized from graph fields; running the EPI update
before extraction does not derive its harmonic flow from the nodal equation.
No factoring algorithm, primality criterion, or open-problem result follows.

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

    eta^2 = between-coset variance / total variance. ~1 means that this vector
    is nearly a function of the already supplied label ``i mod p``.  For
    exchangeable noise, the finite-sample expectation is ``(p-1)/(n-1)``.
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
    """Run a finite explicit-Euler pure-EPI update before field extraction."""
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
    """Q1: finite eigenvalue-count table for residue graphs."""
    print("=" * 74)
    print("EXPERIMENT 1: Eigenvalue Counts in a Finite Residue-Graph Table")
    print("=" * 74)
    print()
    print("L_rw = I - D^-1 W is the isolated canonical dNFR EPI channel.")
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
    print("-> all four sampled Paley primes have 3 values, but 49=7^2 does too.")
    print("   This finite table therefore rejects the predicate as strict primality;")
    print("   it does not characterize all primes, prime powers, or composites.")
    print()


def experiment_2_factor_cosets():
    """Q2: test localization after a candidate factor is supplied."""
    print("=" * 74)
    print("EXPERIMENT 2: Factor-Conditioned Coset Localization")
    print("=" * 74)
    print()
    print("For n=p*q, supply p and test low diffusion modes on classes i mod p.")
    print("Because p labels the classes, this validates a candidate; it does not")
    print("recover an unknown factor from the eigenvector.")
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
        base = (p - 1.0) / (n - 1.0)
        eta_shuf = best_coset_eta2(vecs[rng.permutation(n)], n, p)
        # the shuffle control is the real test: strong localization that the
        # label permutation destroys.
        verdict = "SIGNAL" if (eta > 0.5 and eta > 4 * eta_shuf) else "miss"
        print(
            f"  {f'{n}={p}*{q}':>9} {p:>3} {eta:>12.4f} {base:>9.4f} "
            f"{eta_shuf:>9.4f} {verdict:>8}"
        )
    print()
    print("-> six supplied candidates reach eta^2=1 in the returned low-mode")
    print("   basis and two do not. This is a finite, basis-sensitive validation")
    print("   table; the shuffled column is a control, not factor recovery.")
    print()


def experiment_3_doctrine_check():
    """Q3: regularity identity plus finite auxiliary-field comparisons."""
    print("=" * 74)
    print("EXPERIMENT 3: Regularity and Auxiliary-Field Localization")
    print("=" * 74)
    print()
    print("(a) Residue graph regularity (spread 0 => L_rw = L_classical / d, so")
    print("    the random-walk operator SHARES classical Laplacian eigenspaces):")
    for n in (21, 65, 85):
        degs = [d for _, d in residue_graph(n).degree()]
        tag = "REGULAR" if max(degs) == min(degs) else "irregular"
        print(
            f"    n={n:>3}: degree {min(degs)}..{max(degs)} "
            f"(spread {max(degs) - min(degs)}) -> {tag}"
        )
    print()
    print("    -> any coset signal comes from the constructed residue graph;")
    print("       changing vocabulary does not add spectral information.")
    print()
    print("(b) Auxiliary fields after one seeded finite EPI trajectory:")
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
    print("    -> the selected diffusion modes can localize perfectly for the")
    print("       supplied p; these auxiliary field snapshots do not. Values up")
    print("       to about 0.31 rule out calling every field 'near zero'.")
    print()


def main():
    print()
    print("  TNFR Example 117: Finite Probes on Quadratic-Residue Graphs")
    print("  Eigenvalue counts, factor-conditioned localization, field snapshots")
    print("  ==================================================================")
    print()
    experiment_1_primality_rigidity()
    experiment_2_factor_cosets()
    experiment_3_doctrine_check()
    print("=" * 74)
    print("SCOPED FINDINGS")
    print("=" * 74)
    print()
    print("The sampled Paley primes have a three-value spectrum, but composite")
    print("49 shows that this is not a primality criterion. Coset localization")
    print("is strong in six rows only after the candidate factor p is supplied,")
    print("so the calculation is not a factorizer. Three auxiliary-field")
    print("snapshots fail to reproduce that perfect localization. These finite")
    print("results establish no universal blindness or arithmetic theorem.")
    print()


if __name__ == "__main__":
    main()
