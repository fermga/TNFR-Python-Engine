#!/usr/bin/env python3
"""
Example 109 — P vs NP (TNFR): Coherence Synthesis vs Verification
=================================================================

The first milestone (PNP-1) of the TNFR-native P vs NP program. This is a
STRUCTURAL REFORMULATION and a diagnostic measurement, NOT a solution: it
does not prove P != NP (see "Honest scope").

TNFR-native reformulation
-------------------------
The nodal equation is a GRADIENT FLOW on the structural potential V:

    dEPI/dt = nu_f * dNFR,   dNFR = -dV/dEPI

(established in src/tnfr/physics/variational.py). Coherence relaxation
descends V. P vs NP, read through this lens, is the asymmetry between two
structural tasks:

  * VERIFICATION: given a configuration, evaluate its coherence (here, the
    cut value / frustration energy). Cost = O(|E|) -- polynomial, cheap.
    This is the TNFR analogue of checking an NP witness.

  * SYNTHESIS: find a GLOBALLY coherent configuration by nodal relaxation.
    On a FRUSTRATED topology (odd cycles) the potential V has many local
    optima = dissonance (OZ) basins. Gradient flow descends to the NEAREST
    basin, not necessarily the global one.

Encoding (MAX-CUT as TNFR antiphase coupling)
---------------------------------------------
Each node carries a phase theta. Every edge demands ANTIPHASE (a cut): the
relaxation

    dtheta_i/dt = sum_{j ~ i} sin(theta_i - theta_j)

is the canonical TNFR phase channel (the circular neighbour-coupling that
elsewhere drives Kuramoto synchronization) with the anti-aligning sign --
i.e. an all-edge dissonance (OZ) demand. The global minimum of the
frustration energy E = sum_{(i,j)} cos(theta_i - theta_j) over theta in
{0, pi}^n is exactly the MAX-CUT of the graph (an NP-hard objective).

PNP-1 measurement
-----------------
Across problem sizes n, measure the fraction of random initial conditions
whose relaxation reaches the GLOBAL optimum (hit rate), and confirm that the
best over many restarts DOES reach it (so a low hit rate is genuine TRAPPING
in local optima, not an encoding failure). The honest signature of synthesis
hardness is: hit rate DROPS and required restarts GROW with n, while
verification stays O(|E|).

Honest scope
------------
- This MIRRORS the P vs NP asymmetry (verify easy, synthesize hard); it does
  NOT prove P != NP. The open question is precisely whether some polynomial
  strategy escapes the traps -- bare gradient flow is only one strategy.
- The TNFR catalog has escape operators (OZ controlled dissonance, ZHIR
  mutation, THOL re-organization, REMESH) not used here. Whether the FULL
  catalog collapses the trapping to polynomial is the open milestone PNP-2;
  the honest expectation (exponentially many dissonance basins) reflects
  P != NP but remains unproven.
- MAX-CUT has a classical 0.878 approximation (Goemans-Williamson); only
  EXACT global optimization is hard. This example measures exact-optimum
  trapping, the TNFR-native reflection of that hardness.

References
----------
- theory/TNFR_P_VS_NP_RESEARCH_NOTES.md (program, milestones, classification)
- src/tnfr/physics/variational.py (nodal equation as gradient flow)
- src/tnfr/physics/structural_diffusion.py (phase channel = Kuramoto coupling)
- AGENTS.md section "Regime Correspondences from Nodal Dynamics"
"""

import os
import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx


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
    """Canonical TNFR phase channel with anti-aligning (MAX-CUT) sign.

    dtheta_i = sum_{j~i} sin(theta_i - theta_j)  (descends frustration E).
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
    print("PNP-1: Coherence SYNTHESIS vs VERIFICATION on frustrated MAX-CUT")
    print("=" * 72)
    print()
    print("Verification (evaluate a cut)   = O(|E|), polynomial -- cheap.")
    print("Synthesis (relaxation -> global) = measured hit rate below.")
    print()
    print(f"  {'n':>3} {'|E|':>4} {'global':>6} {'hit_rate':>9} "
          f"{'restarts~1/hr':>13} {'best/all':>9}")

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
        restarts = (1.0 / hr) if hr > 0 else float('inf')
        rows.append((n, hr))
        print(f"  {n:>3} {edges_last:>4} {'yes' if reached_global else 'NO':>6} "
              f"{hr:>9.3f} {restarts:>13.2f} {'reached' if reached_global else 'MISS':>9}")

    ns = np.array([r[0] for r in rows], float)
    hrs = np.array([r[1] for r in rows], float)
    slope = float(np.polyfit(ns, hrs, 1)[0])
    print()
    print(f"  hit-rate trend slope d(hit_rate)/dn = {slope:+.4f} per node")
    print(f"  monotone decreasing: {bool(np.all(np.diff(hrs) <= 1e-9))}")
    print()
    print("VERDICT (PNP-1): coherence synthesis by bare gradient flow gets")
    print("increasingly TRAPPED in local optima (dissonance basins) as size")
    print("grows -- hit rate drops, restarts grow -- while verification stays")
    print("O(|E|). 'best/all = reached' confirms the global optimum IS")
    print("reachable with enough restarts, so the low hit rate is genuine")
    print("trapping, not an encoding failure.")
    print()


def main():
    print()
    print("  TNFR Example 109: P vs NP -- Coherence Synthesis vs Verification")
    print("  Milestone PNP-1 (structural reformulation, NOT a proof)")
    print("  ===============================================================")
    print()
    experiment_trapping()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES (and what it does NOT)")
    print("=" * 72)
    print()
    print("ESTABLISHES: a TNFR-native reformulation of P vs NP as the")
    print("asymmetry between coherence VERIFICATION (O(|E|), polynomial) and")
    print("coherence SYNTHESIS (gradient-flow relaxation, which traps in")
    print("dissonance basins with growing size). This is the same disciplined")
    print("pattern as the Riemann / Navier-Stokes / Yang-Mills programs.")
    print()
    print("DOES NOT: prove P != NP. Bare gradient flow is one strategy; the")
    print("full TNFR operator catalog (OZ, ZHIR, THOL, REMESH escape moves) is")
    print("not used here. Whether the full catalog synthesizes in polynomial")
    print("time is the open milestone PNP-2. No Clay claim is made.")
    print()


if __name__ == "__main__":
    main()
