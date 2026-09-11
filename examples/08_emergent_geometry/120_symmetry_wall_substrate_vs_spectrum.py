#!/usr/bin/env python3
"""Example 120 — finite residue-digraph and auxiliary-readout comparison.

The residue digraph is constructed from the quadratic residues modulo ``n``;
its arithmetic content is therefore an input. The example verifies that cyclic
translation preserves this constructed edge set on selected values of ``n``.
It then compares two read-outs:

* the number of numerically distinct eigenvalues of the directed diffusion
  matrix, rounded to a configured precision;
* the standard deviation of ``Phi_s`` obtained by initializing the auxiliary
  phase-space model from graph fields after a fixed explicit update protocol.

Random fixed-outdegree digraphs provide a finite control family. Prime/composite
labels come independently from SymPy. Agreement between the spectral statistic
and those labels over the sampled range is empirical evidence for this graph
family, not a general primality proof or arithmetic emergence: the residue graph
already contains ``x^2 mod n``. Differences between the two read-outs do not
place all arithmetic in an orthogonal representation sector.

``extract_phase_space_point`` reads an auxiliary symplectic model initialized
from the graph state. This script does not show that the engine generated that
model, characterize every possible node observable, define an ``S(T)`` map, or
identify a Riemann obstruction.

Status: RESEARCH example; finite comparison with constructed inputs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np
from sympy import isprime

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def _qr(n):
    return {(x * x) % n for x in range(1, n)} - {0}


def residue_digraph(n):
    """Directed residue Cayley graph: edge i->j iff (j-i) mod n is a QR."""
    R = _qr(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and ((j - i) % n) in R:
                G.add_edge(i, j)
    return G


def random_fixed_outdegree_digraph(n, outdeg, seed):
    """Construct a random digraph with the same out-degree at every node."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        choices = [j for j in range(n) if j != i]
        succ = rng.choice(choices, size=outdeg, replace=False)
        for j in succ:
            G.add_edge(i, int(j))
    return G


# Historical API alias; this construction is not necessarily a tournament.
random_regular_tournament = random_fixed_outdegree_digraph


def _seed_random(G, seed):
    """Assign the same seeded random-state protocol to each graph."""
    rng = np.random.default_rng(seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _evolve(G, steps=16, dt=0.05):
    """Apply a fixed explicit-Euler nodal update for finite comparison."""
    for _ in range(steps):
        default_compute_delta_nfr(G)
        for nd in G.nodes():
            epi = float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0))
            vf = float(get_attr(G.nodes[nd], ALIAS_VF, 0.0))
            dnfr = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            set_attr(G.nodes[nd], ALIAS_EPI, epi + dt * vf * dnfr)


def _n_distinct(G, decimals=4):
    """Count rounded eigenvalues of the directed diffusion matrix."""
    _, L = structural_diffusion_operator(G)
    ev = np.linalg.eigvals(L)
    return len(np.unique(np.round(ev, decimals)))


def _auxiliary_phi_std(G):
    """Phi_s dispersion in the auxiliary phase-space read-out."""
    p = extract_phase_space_point(G)
    return float(np.std(p.phi_s))


def experiment_1_vertex_transitivity():
    """R1: i -> i+1 mod n is an automorphism of the residue digraph."""
    print("=" * 74)
    print("EXPERIMENT 1: Vertex-Transitivity (the translation automorphism)")
    print("=" * 74)
    print("The residue digraph is a Cayley digraph of Z_n. The translation")
    print("i -> i+1 (mod n) preserves the difference j-i, hence the QR edge")
    print("set: it is ALWAYS an automorphism. All nodes are equivalent.")
    print()
    print(f"  {'n':>4} {'mod4':>5} {'prime':>6}  translation is automorphism?")
    all_ok = True
    for n in [7, 11, 13, 17, 19, 23, 25, 29]:
        G = residue_digraph(n)
        E = set(G.edges())
        E_shift = {((i + 1) % n, (j + 1) % n) for (i, j) in E}
        ok = E == E_shift
        all_ok = all_ok and ok
        print(f"  {n:>4} {n % 4:>5} {str(isprime(n)):>6}  {ok}")
    print()
    print(f"  -> automorphism for every tested n: {all_ok}")
    print("     The encoded residue differences are invariant under this symmetry;")
    print("     any translation-invariant node read-out is constant on its orbit.")


def experiment_2_finite_readout_comparison():
    """Compare two finite read-outs on residue and random control graphs."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Residue Graphs vs Random Fixed-Outdegree Controls")
    print("=" * 74)
    print("Residue digraph (constructed QR structure) vs random digraphs")
    print("of the same out-degree. All receive the same seed/update protocol.")
    print("The table reports a spectral count and auxiliary Phi_s dispersion.")
    print()
    print(
        f"  {'n':>4} | {'Paley_dist':>10} {'rand_dist':>10} | "
        f"{'Paley_phiStd':>12} {'rand_phiStd':>12}"
    )
    for n in [11, 19, 23, 31, 43, 47]:
        Gp = residue_digraph(n)
        outdeg = Gp.out_degree(0)
        p_dist = _n_distinct(Gp)
        _seed_random(Gp, 0)
        _evolve(Gp)
        p_phi = _auxiliary_phi_std(Gp)
        r_dists, r_phis = [], []
        for s in range(5):
            Gr = random_fixed_outdegree_digraph(n, outdeg, seed=200 + s)
            r_dists.append(_n_distinct(Gr))
            _seed_random(Gr, 0)
            _evolve(Gr)
            r_phis.append(_auxiliary_phi_std(Gr))
        print(
            f"  {n:>4} | {p_dist:>10} {np.mean(r_dists):>10.1f} | "
            f"{p_phi:>12.4f} {np.mean(r_phis):>12.4f}"
        )
    print()
    print("  -> In this sample the residue and random constructions give different")
    print("     spectral counts. Their Phi_s summaries are reported side by side.")
    print("     No equality of distributions or universal blindness is tested.")


def experiment_3_sampled_label_comparison():
    """Compare finite spectral and Phi_s summaries with external labels."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Sampled Spectral Rule and Auxiliary Phi_s Summary")
    print("=" * 74)
    print("Across odd n=7,...,41, compare the rule '3 rounded eigenvalues'")
    print("with independent SymPy labels and report the auxiliary Phi_s std.")
    print()
    print(
        f"  {'n':>4} {'prime':>6} {'spec_dist':>10} {'spec_prime?':>12} "
        f"{'phiStd':>8}"
    )
    n_correct = 0
    n_total = 0
    for n in range(7, 42):
        if n % 2 == 0:
            continue
        G = residue_digraph(n)
        dist = _n_distinct(G)
        spec_prime = dist == 3
        _seed_random(G, 0)
        _evolve(G)
        phi = _auxiliary_phi_std(G)
        ok = spec_prime == isprime(n)
        n_correct += int(ok)
        n_total += 1
        flag = "OK" if ok else "XX"
        print(
            f"  {n:>4} {str(isprime(n)):>6} {dist:>10} "
            f"{str(spec_prime):>12} {phi:>8.4f}  {flag}"
        )
    print()
    print(f"  -> finite spectral-label agreement: {n_correct}/{n_total}.")
    print("     The listed Phi_s values are descriptive outputs of this one")
    print("     seeded protocol; no classifier or monotonic law about them is proved.")


# Historical callable names remain aliases for compatibility only.
_substrate_phi_std = _auxiliary_phi_std
experiment_2_double_dissociation = experiment_2_finite_readout_comparison
experiment_3_substrate_tracks_size = experiment_3_sampled_label_comparison


def main():
    print()
    print("  TNFR Example 120: Finite Residue-Graph Read-out Comparison")
    print("  ==========================================================")
    print()
    experiment_1_vertex_transitivity()
    experiment_2_finite_readout_comparison()
    experiment_3_sampled_label_comparison()
    print()
    print("=" * 74)
    print("SCOPED RESULT")
    print("=" * 74)
    print("Translation preserves every sampled residue-digraph edge set. The")
    print("rounded spectral count agrees with the supplied arithmetic labels on")
    print("the reported finite range, while Phi_s supplies a different auxiliary")
    print("summary after the fixed update protocol. The graph already encodes")
    print("quadratic residues, and no universal representation-sector location,")
    print("engine-generated symplectic model, S(T) map, or RH claim follows.")


if __name__ == "__main__":
    main()
