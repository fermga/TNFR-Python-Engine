#!/usr/bin/env python3
"""
Example 109 — finite MAX-CUT search and exact verification.

This auxiliary classical baseline applies a selected antialignment heuristic:

    dtheta_i/dt = sum_{j ~ i} sin(theta_i - theta_j)

The continuous vector field is the negative gradient of
V = sum_{(i,j)} cos(theta_i - theta_j). The implemented finite Euler step
has no monotone-energy certificate for its selected step size. On binary
phases {0, pi}, V = |E| - 2 * cut, so minimizing the binary objective is
equivalent to MAX-CUT. The heuristic rounds its final continuous phases.

Small supplied regular graphs admit an exhaustive reference optimum. Seeded
restarts report the finite fraction reaching that optimum and an empirical
trend; neither decreasing hit rate nor convergence to a local optimum is
assumed. Exhaustive verification of the optimum costs more than evaluating a
single supplied cut, which requires one pass over the edges.

This is not the canonical circular-neighbor pressure channel, a TNFR operator
schedule, or a derivation from the nodal equation. The held pure-EPI gradient
bridge does not make arbitrary phase dynamics canonical. Finite heuristic
performance establishes no asymptotic complexity result or answer to P vs NP.

References
----------
- theory/TNFR_P_VS_NP_RESEARCH_NOTES.md (parked comparison and scope)
- theory/NODAL_PARAMETER_FOUNDATIONS.md (constitutive dependencies)
- theory/TNFR_VARIATIONAL_PRINCIPLE.md (restricted gradient bridge)
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np


def max_cut_bruteforce(G):
    """Exact MAX-CUT by enumeration (feasible for small n)."""
    nodes = list(G.nodes())
    edges = list(G.edges())
    best = -1
    for bits in itertools.product((0, 1), repeat=len(nodes)):
        a = dict(zip(nodes, bits))
        cut = sum(1 for u, v in edges if a[u] != a[v])
        if cut > best:
            best = cut
    return best


def tnfr_phase_relaxation(G, seed, steps=400, dt=0.1):
    """Legacy-named auxiliary antialignment Euler heuristic.

    Its continuous field descends V; a selected Euler step need not do so.
    Returns the cut value of the rounded {0, pi} assignment.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    th = rng.uniform(0, 2 * np.pi, size=len(nodes))
    adj = [[idx[j] for j in G.neighbors(v)] for v in nodes]
    for _ in range(steps):
        f = np.empty_like(th)
        for i in range(len(nodes)):
            f[i] = np.sum(np.sin(th[i] - th[adj[i]])) if adj[i] else 0.0
        th = th + dt * f
    assign = (np.cos(th) < 0).astype(int)
    return sum(1 for u, v in G.edges() if assign[idx[u]] != assign[idx[v]])


def experiment_trapping():
    print("=" * 72)
    print("Finite MAX-CUT: heuristic search and exact verification")
    print("=" * 72)
    print()
    print("Verification (evaluate a cut)   = O(|E|), polynomial -- cheap.")
    print("Synthesis (relaxation -> global) = measured hit rate below.")
    print()
    print(
        f"  {'n':>3} {'|E|':>4} {'global':>6} {'hit_rate':>9} "
        f"{'restarts~1/hr':>13} {'best/all':>9}"
    )

    R = 200
    rows = []
    for n in (8, 10, 12, 14, 16, 18):
        hit_rates = []
        reached_global = True
        edges_last = 0
        for inst in range(3):
            G = nx.random_regular_graph(3, n, seed=100 + inst)
            edges_last = G.number_of_edges()
            gc = max_cut_bruteforce(G)
            best = 0
            hits = 0
            for s in range(R):
                c = tnfr_phase_relaxation(G, seed=1000 * inst + s)
                best = max(best, c)
                if c >= gc:
                    hits += 1
            hit_rates.append(hits / R)
            reached_global = reached_global and (best >= gc)
        hr = float(np.mean(hit_rates))
        restarts = (1.0 / hr) if hr > 0 else float("inf")
        rows.append((n, hr))
        print(
            f"  {n:>3} {edges_last:>4} {'yes' if reached_global else 'NO':>6} "
            f"{hr:>9.3f} {restarts:>13.2f} {'reached' if reached_global else 'MISS':>9}"
        )

    ns = np.array([r[0] for r in rows], float)
    hrs = np.array([r[1] for r in rows], float)
    slope = float(np.polyfit(ns, hrs, 1)[0])
    print()
    print(f"  hit-rate trend slope d(hit_rate)/dn = {slope:+.4f} per node")
    print(f"  monotone decreasing: {bool(np.all(np.diff(hrs) <= 1e-9))}")
    print()
    print("The finite hit rates and empirical slope are reported above.")
    print("Neither a decreasing trend nor local trapping is assumed.")
    print("Evaluating one supplied cut requires one pass over the edges;")
    print("the exhaustive reference optimum is a separate computation.")
    print("A successful restart records one reached optimum, without a")
    print("guarantee for another instance, restart or asymptotic size.")
    print()


def main():
    print()
    print("  TNFR Example 109: Auxiliary finite MAX-CUT baseline")
    print("  Supplied classical heuristic and exhaustive small-graph reference")
    print("  ===============================================================")
    print()
    experiment_trapping()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES (and what it does NOT)")
    print("=" * 72)
    print()
    print("REPORTS: finite outcomes of one supplied classical heuristic")
    print("against enumerated optima. The implemented phase update is")
    print("not the canonical TNFR pressure channel or operator schedule.")
    print("Rounded endpoints do not prove continuous local optimality,")
    print("and measured hit rates do not establish an asymptotic law.")
    print()
    print("DOES NOT: prove P != NP or derive an optimization law from TNFR.")
    print("No canonical operator catalog is executed in this baseline.")
    print("Its finite comparison does not reopen the parked P vs NP branch;")
    print("the research plan owns any subsequent admission decision.")
    print()


if __name__ == "__main__":
    main()
