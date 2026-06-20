#!/usr/bin/env python3
"""
Example 129 — The Spectral Gap Is the Base->Fiber Coupling Clock (Line C):
Relaxation, Cheeger Bottleneck, Instability Threshold, and the Co-Emergent Tree
==============================================================================

The two-layer optic (example 126) showed the spectral gap lambda_2 is a BASE
quantity (it depends only on the topology) but it is the CLOCK of the base->fiber
coupling: it sets the rate at which the nodal equation's linear field relaxes,
which is what drives the fiber substrate. This example measures the four faces of
that clock, and closes the loop with the co-emergent footing of example 128.

The five faces of lambda_2 (all canonical)
------------------------------------------
  1. RELAXATION RATE. The slowest linear-field decay rate is nu_f*lambda_2: the
     non-uniform modes decay as exp(-nu_f*lambda_k*t), and the spectral gap is
     the slowest of them -- the clock that times the base->fiber coupling.
  2. CHEEGER BOTTLENECK. lambda_2 is bounded by the conductance of the network's
     weakest cut (Cheeger): h^2/2 <= lambda_2 <= 2h. The gap is the bottleneck;
     the Fiedler partition IS that weakest cut.
  3. INSTABILITY THRESHOLD. The dispersion relation sigma_k = r - nu_f*lambda_k
     gives the structural instability threshold r_c = nu_f*lambda_2: below it
     only the uniform mode grows (homogenization); above it the Fiedler pattern
     emerges (structural pattern formation). This is the spectral form of U2.
  4. THE CO-EMERGENT CLOCK. The co-emergent fixed point of the nodal dynamics
     (example 128) is a spanning TREE; trees are the weakest-connected spanning
     structures, so they have the SMALLEST lambda_2 -- the slowest, most fragile
     base->fiber clock. The dynamics settles on the weakest self-consistent base.
  5. THE LYAPUNOV CLOCK. The same nu_f*lambda_2 is the relaxation rate of the
     structural Lyapunov energy's gradient sectors, via the proven diffusion
     H-theorem (theorem 8.6). The conservation/Lyapunov module exposes it as
     diffusion_gap = lambda_2(L_sym), NOT the combinatorial lambda_2(D-A).

Doctrine compliance
-------------------
All four faces are canonical: the relaxation rate and Cheeger gap come from
`structural_eigenmodes` / `verify_structural_diffusion`; the threshold from
`instability_threshold` (r_c = nu_f*lambda_2) and `dispersion_relation`; the
Fiedler cut from `fiedler_partition`; the co-emergent tree from the canonical
REMESH helper `_mst_edges_from_epi`. Nothing is imposed.

Five measured results
---------------------
M1 lambda_2 IS THE CLOCK. The canonical verify confirms nu_f*lambda_2 is the
   slowest relaxation rate on a path, a cycle and a complete graph. Small gap
   (path 0.04) = slow clock; large gap (complete 1.14) = fast clock.

M2 CHEEGER BOTTLENECK. lambda_2 sits between h^2/2 and 2h for the conductance h
   of the Fiedler cut on every test graph (path, barbell, complete, cycle): the
   spectral gap is set by the network's weakest cut. (Honest: h is the Fiedler-
   cut conductance, a proxy for the exact Cheeger constant, which is NP-hard.)

M3 INSTABILITY THRESHOLD r_c = nu_f*lambda_2. On a barbell (two cliques + a
   bottleneck) the dispersion relation has 0-1 unstable modes for r < r_c and
   turns the Fiedler mode unstable for r > r_c: the structural pattern (the
   weakest cut) is the first to grow. This is the spectral form of grammar U2.

M4 THE CO-EMERGENT TREE HAS THE SMALLEST GAP. The co-emergent fixed point (a
   spanning tree from example 128) has the smallest lambda_2 (0.029) of all the
   test graphs (cycle 0.099, random 0.40, complete 1.08): the dynamics settles
   on the weakest-connected self-consistent structure -- the slowest base->fiber
   clock.

M5 lambda_2 IS ALSO THE LYAPUNOV CLOCK. analyze_spectral_gap's diffusion_gap
   (the conservation/Lyapunov relaxation rate) equals the structural-diffusion
   lambda_2 on every test graph, while the combinatorial lambda_2(D-A) differs:
   the four faces above and the conservation Lyapunov share ONE clock
   (theorem 8.6).

Honest scope
------------
The four faces of lambda_2 are standard spectral graph theory (relaxation,
Cheeger, linear stability) re-expressed in the canonical operator; the
contribution is the clean statement that the spectral gap is the base->fiber
COUPLING CLOCK in the two-layer optic, and that the co-emergent attractor (a
tree) is its slowest setting. The Cheeger bound uses the Fiedler-cut conductance
as a bottleneck proxy (the exact Cheeger constant is NP-hard). It is not new
mathematics and closes no open problem.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_eigenmodes,
  instability_threshold, dispersion_relation, fiedler_partition,
  verify_structural_diffusion)
- src/tnfr/operators/remesh.py (_mst_edges_from_epi: the co-emergent tree)
- examples/08_emergent_geometry/126_two_layers_base_fiber.py (the two-layer optic)
- examples/08_emergent_geometry/128_base_substrate_coemergence.py (the tree attractor)
- examples/08_emergent_geometry/112_structure_predicts_coherence_flow.py (nu_f*lambda_2)
- src/tnfr/physics/lyapunov.py (analyze_spectral_gap: diffusion_gap = lambda_2(L_sym))
- theory/STRUCTURAL_CONSERVATION_THEOREM.md section 8.6 (the relaxation-rate identity)
- AGENTS.md "Transport Content of the Nodal Equation" (the dispersion relation, U2)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.alias import set_attr, get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.operators.remesh import _mst_edges_from_epi, _get_networkx_modules
from tnfr.physics.structural_diffusion import (
    dispersion_relation,
    fiedler_partition,
    instability_threshold,
    structural_eigenmodes,
    verify_structural_diffusion,
)
from tnfr.physics.lyapunov import analyze_spectral_gap

NXMOD, _ = _get_networkx_modules()


def _seed(G, rng):
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _fiedler_conductance(G, A):
    """h(A) = cut(A,B) / min(vol A, vol B) -- the Fiedler-cut conductance."""
    A = set(A)
    cut = sum(1 for u, v in G.edges() if (u in A) != (v in A))
    deg = dict(G.degree())
    volA = sum(deg[u] for u in A)
    volB = 2 * G.number_of_edges() - volA
    denom = min(volA, volB)
    return cut / denom if denom > 0 else 0.0


def _coemergent_tree(n, seed):
    """The co-emergent fixed point (a spanning tree) from example 128."""
    G = nx.cycle_graph(n)
    _seed(G, np.random.default_rng(seed))
    G.graph["DNFR_WEIGHTS"] = {"epi": 1.0, "phase": 0, "vf": 0, "topo": 0}
    for _ in range(3):
        default_compute_delta_nfr(G)
        for nd in G.nodes():
            e = get_attr(G.nodes[nd], ALIAS_EPI, 0.0)
            d = get_attr(G.nodes[nd], ALIAS_DNFR, 0.0)
            set_attr(G.nodes[nd], ALIAS_EPI, e + 0.1 * d)
    epi = {nd: get_attr(G.nodes[nd], ALIAS_EPI, 0.0) for nd in G.nodes()}
    edges = _mst_edges_from_epi(NXMOD, list(G.nodes()), epi)
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(edges)
    for nd in T.nodes():
        set_attr(T.nodes[nd], ALIAS_VF, 1.0)
    return T


def experiment_1_clock():
    """M1: nu_f*lambda_2 is the slowest relaxation rate (the clock)."""
    print("=" * 74)
    print("EXPERIMENT 1: lambda_2 Is the Clock (Slowest Relaxation = nu_f*lambda_2)")
    print("=" * 74)
    print("The non-uniform modes decay as exp(-nu_f*lambda_k*t); the spectral gap")
    print("is the slowest of them -- the clock that times the base->fiber coupling.")
    print()
    print(f"  {'graph':14s} {'lambda_2':>9} {'nu_f*lambda_2':>14} {'clock':>10}")
    for name, G in [("path P12", nx.path_graph(12)),
                    ("cycle C12", nx.cycle_graph(12)),
                    ("complete K8", nx.complete_graph(8))]:
        _seed(G, np.random.default_rng(0))
        cert = verify_structural_diffusion(G)
        speed = ("slow" if cert.slowest_relaxation_rate < 0.1
                 else "fast" if cert.slowest_relaxation_rate > 0.5
                 else "medium")
        print(f"  {name:14s} {cert.spectral_gap:>9.4f} "
              f"{cert.slowest_relaxation_rate:>14.4f} {speed:>10}")
    print()
    print("  -> nu_f*lambda_2 is the slowest relaxation rate: small gap (path)")
    print("     = slow clock, large gap (complete) = fast clock.")


def experiment_2_cheeger():
    """M2: lambda_2 is set by the conductance bottleneck (Cheeger)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Cheeger -- lambda_2 Is Set by the Bottleneck Conductance")
    print("=" * 74)
    print("Cheeger: h^2/2 <= lambda_2 <= 2h. The Fiedler partition is the")
    print("weakest cut; its conductance h bounds the spectral gap.")
    print()
    print(f"  {'graph':20s} {'lambda_2':>9} {'h(Fiedler)':>11} "
          f"{'h^2/2':>8} {'2h':>7} {'in bounds?':>11}")
    cases = [
        ("path P12", nx.path_graph(12)),
        ("barbell (2 K5)", nx.barbell_graph(5, 0)),
        ("complete K10", nx.complete_graph(10)),
        ("cycle C12", nx.cycle_graph(12)),
    ]
    for name, G in cases:
        for nd in G.nodes():
            set_attr(G.nodes[nd], ALIAS_VF, 1.0)
        ev, _ = structural_eigenmodes(G)
        lam2 = ev[1]
        A, _ = fiedler_partition(G)
        h = _fiedler_conductance(G, A)
        in_bounds = (h * h / 2 - 1e-9) <= lam2 <= (2 * h + 1e-9)
        print(f"  {name:20s} {lam2:>9.4f} {h:>11.4f} {h*h/2:>8.4f} "
              f"{2*h:>7.4f} {str(in_bounds):>11}")
    print()
    print("  -> lambda_2 sits between h^2/2 and 2h: the spectral gap = the")
    print("     network's weakest cut. (Honest: h is the Fiedler-cut conductance,")
    print("     a proxy for the exact Cheeger constant, which is NP-hard.)")


def experiment_3_threshold():
    """M3: instability threshold r_c = nu_f*lambda_2 (dispersion, U2)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Instability Threshold r_c = nu_f*lambda_2 (Dispersion, U2)")
    print("=" * 74)
    print("sigma_k = r - nu_f*lambda_k. Below r_c only the uniform mode grows;")
    print("above r_c the Fiedler pattern (the weakest cut) emerges.")
    print()
    G = nx.barbell_graph(5, 0)
    _seed(G, np.random.default_rng(0))
    r_c = instability_threshold(G)
    print(f"  barbell (2 K5): r_c = nu_f*lambda_2 = {r_c:.4f}")
    print(f"  {'reaction r':>12} {'unstable modes':>15} {'regime':>22}")
    for label, r in [("0", 0.0), ("0.5 r_c", 0.5 * r_c),
                     ("0.99 r_c", 0.99 * r_c), ("1.5 r_c", 1.5 * r_c)]:
        sigma = dispersion_relation(G, reaction_rate=r)
        n_unstable = int(np.sum(sigma > 1e-9))
        regime = ("uniform only" if n_unstable <= 1 else "Fiedler pattern grows")
        print(f"  {label:>12} {n_unstable:>15} {regime:>22}")
    print()
    print("  -> r < r_c: homogenization (uniform mode); r > r_c: the Fiedler")
    print("     structural pattern grows. The spectral form of grammar U2.")


def experiment_4_coemergent_tree():
    """M4: the co-emergent tree (ex 128) has the smallest gap."""
    print()
    print("=" * 74)
    print("EXPERIMENT 4: The Co-Emergent Tree (ex 128) Has the Smallest Gap")
    print("=" * 74)
    print("The co-emergent fixed point is a spanning tree; trees are the")
    print("weakest-connected spanning structures -> smallest lambda_2 -> the")
    print("slowest, most fragile base->fiber clock.")
    print()
    N = 14
    print(f"  {'graph':18s} {'lambda_2':>9} {'is_tree':>8}")
    graphs = [
        ("co-emergent TREE", _coemergent_tree(N, 0)),
        ("cycle C14", nx.cycle_graph(N)),
        ("random G(14,0.3)", nx.gnp_random_graph(N, 0.3, seed=9)),
        ("complete K14", nx.complete_graph(N)),
    ]
    for name, G in graphs:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        for nd in G.nodes():
            set_attr(G.nodes[nd], ALIAS_VF, 1.0)
        ev, _ = structural_eigenmodes(G)
        print(f"  {name:18s} {ev[1]:>9.4f} {str(nx.is_tree(G)):>8}")
    print()
    print("  -> the co-emergent tree has the SMALLEST gap: the dynamics settles")
    print("     on the weakest-connected self-consistent base = the slowest clock.")


def experiment_5_lyapunov_clock():
    """M5: the same lambda_2 is the conservation/Lyapunov relaxation clock."""
    print()
    print("=" * 74)
    print("EXPERIMENT 5: lambda_2 Is Also the Conservation/Lyapunov Clock (8.6)")
    print("=" * 74)
    print("The structural Lyapunov energy's gradient sectors relax by the proven")
    print("diffusion H-theorem at rate nu_f*lambda_2 -- the SAME clock. The")
    print("conservation/Lyapunov module (analyze_spectral_gap) exposes it as")
    print("diffusion_gap = lambda_2(L_sym), NOT the combinatorial lambda_2(D-A).")
    print()
    print(f"  {'graph':18s} {'diffusion l2':>13} {'Lyapunov gap':>13} "
          f"{'combinatorial':>14} {'match?':>7}")
    cases = [
        ("path P12", nx.path_graph(12)),
        ("barbell (2 K5)", nx.barbell_graph(5, 0)),
        ("cycle C12", nx.cycle_graph(12)),
        ("complete K8", nx.complete_graph(8)),
    ]
    for name, G in cases:
        for nd in G.nodes():
            set_attr(G.nodes[nd], ALIAS_VF, 1.0)
        ev, _ = structural_eigenmodes(G)
        lam2 = float(ev[1])
        sg = analyze_spectral_gap(G)
        match = abs(lam2 - sg.diffusion_gap) < 1e-9
        print(f"  {name:18s} {lam2:>13.4f} {sg.diffusion_gap:>13.4f} "
              f"{sg.spectral_gap:>14.4f} {str(match):>7}")
    print()
    print("  -> the diffusion lambda_2 and the Lyapunov diffusion_gap are the")
    print("     SAME (match=True); the combinatorial gap differs. The four faces")
    print("     above and the conservation Lyapunov share ONE clock (8.6).")


def main():
    print()
    print("  TNFR Example 129: The Spectral Gap Is the Base->Fiber Coupling Clock")
    print("  Relaxation, Cheeger, Instability Threshold, the Co-Emergent Tree")
    print("  ===================================================================")
    print()
    experiment_1_clock()
    experiment_2_cheeger()
    experiment_3_threshold()
    experiment_4_coemergent_tree()
    experiment_5_lyapunov_clock()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("The spectral gap lambda_2 is the base->fiber COUPLING CLOCK of the")
    print("two-layer optic (example 126), with five canonical faces: (1) the")
    print("slowest relaxation rate nu_f*lambda_2 (the clock); (2) the Cheeger")
    print("bottleneck (lambda_2 set by the network's weakest cut, the Fiedler")
    print("partition); (3) the instability threshold r_c = nu_f*lambda_2 (the")
    print("dispersion relation, the spectral form of grammar U2); and (4) the")
    print("co-emergent tree (example 128) has the SMALLEST gap -- the slowest,")
    print("most fragile clock, on which the nodal dynamics self-consistently")
    print("settles; and (5) the conservation/Lyapunov energy relaxes on this")
    print("SAME clock (theorem 8.6, diffusion_gap = lambda_2(L_sym)).")
    print("HONEST SCOPE: the five faces are standard spectral graph")
    print("theory (relaxation, Cheeger, linear stability) re-expressed in the")
    print("canonical operator; the Cheeger bound uses the Fiedler-cut conductance")
    print("as a bottleneck proxy (the exact constant is NP-hard); the")
    print("contribution is the clean base->fiber clock statement, not new")
    print("mathematics, closes no open problem.")


if __name__ == "__main__":
    main()
