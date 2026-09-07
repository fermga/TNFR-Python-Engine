#!/usr/bin/env python3
"""
Example 129 — Spectral-Gap Clock and Scoped Graph Comparisons
============================================================

For fixed connected symmetric graphs with one common positive structural
frequency, the pure EPI channel is exactly graph diffusion. Its non-uniform
modes decay as ``exp(-nu_f * lambda_k * t)``, so
``nu_f * lambda_2(L_sym)`` is the exact slowest non-uniform amplitude-decay
rate. This restricted identity is the clock measured in experiment 1.

The remaining experiments are comparisons around that identity:

1. Cheeger's inequality uses the *optimal* conductance ``h_*``. A sign cut of
   a Fiedler vector is a spectral candidate and need not attain ``h_*``.
2. ``r_c = nu_f * lambda_2`` is the first non-uniform threshold of the stated
   linear reaction-diffusion model. The uniform mode grows for every ``r > 0``;
   this model result is not a theorem about grammar U2.
3. The EPI-distance MST returned by ``_mst_edges_from_epi`` minimizes its edge
   cost. Its measured normalized gap is compared with three graphs; no
   universal minimum over spanning trees or connected graphs is inferred.
4. ``analyze_spectral_gap.diffusion_gap`` and ``structural_eigenmodes`` share
   the same normalized-Laplacian convention. Their numerical agreement is an
   API consistency check. It does not make the five-term structural-energy
   diagnostic decay at this rate.

For homogeneous pure EPI diffusion, the corresponding centered quadratic
energy decays with the bound ``V(t) <= exp(-2*nu_f*lambda_2*t) V(0)``. That
proved restricted energy is distinct from the tetrad-plus-current structural
energy and from any auxiliary base/fiber module.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_eigenmodes,
  instability_threshold, dispersion_relation, fiedler_partition,
  verify_structural_diffusion)
- src/tnfr/operators/remesh.py (_mst_edges_from_epi: private MST helper)
- examples/08_emergent_geometry/126_two_layers_base_fiber.py (dependency labels)
- examples/08_emergent_geometry/128_base_substrate_coemergence.py (finite MST survey)
- examples/08_emergent_geometry/112_structure_predicts_coherence_flow.py (nu_f*lambda_2)
- src/tnfr/physics/lyapunov.py (analyze_spectral_gap: diffusion_gap = lambda_2(L_sym))
- theory/TNFR_DIFFUSION_STABILITY_THEOREM.md (restricted diffusion theorem)
- AGENTS.md "Transport content (structural diffusion)"
"""

import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.operators.remesh import _get_networkx_modules, _mst_edges_from_epi
from tnfr.physics.lyapunov import analyze_spectral_gap
from tnfr.physics.structural_diffusion import (
    dispersion_relation,
    fiedler_partition,
    instability_threshold,
    structural_eigenmodes,
    verify_structural_diffusion,
)

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


def _exact_conductance(G):
    """Return the exact conductance of a small undirected graph.

    Exhaustive enumeration is practical only for the small demonstration
    graphs below. Restricting cardinality to at most half the nodes avoids
    evaluating both sides of every cut.
    """
    nodes = list(G.nodes())
    best = float("inf")
    for size in range(1, len(nodes) // 2 + 1):
        for subset in combinations(nodes, size):
            best = min(best, _fiedler_conductance(G, subset))
    return best


def _coemergent_tree(n, seed):
    """Build the EPI-distance MST sample used by example 128."""
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
    """M1: verify the exact homogeneous pure-EPI relaxation clock."""
    print("=" * 74)
    print("EXPERIMENT 1: Homogeneous Pure-EPI Diffusion Clock")
    print("=" * 74)
    print("On each fixed connected symmetric graph below, nu_f=1 for every node.")
    print("Non-uniform EPI amplitudes decay as exp(-nu_f*lambda_k*t), making")
    print("nu_f*lambda_2 the exact slowest non-uniform amplitude-decay rate.")
    print()
    print(f"  {'graph':14s} {'lambda_2':>9} {'nu_f*lambda_2':>14} {'clock':>10}")
    for name, G in [
        ("path P12", nx.path_graph(12)),
        ("cycle C12", nx.cycle_graph(12)),
        ("complete K8", nx.complete_graph(8)),
    ]:
        _seed(G, np.random.default_rng(0))
        cert = verify_structural_diffusion(G)
        speed = (
            "slow"
            if cert.slowest_relaxation_rate < 0.1
            else "fast" if cert.slowest_relaxation_rate > 0.5 else "medium"
        )
        print(
            f"  {name:14s} {cert.spectral_gap:>9.4f} "
            f"{cert.slowest_relaxation_rate:>14.4f} {speed:>10}"
        )
    print()
    print("  -> Within this homogeneous pure-EPI model, the path relaxes more")
    print("     slowly than the cycle and complete graph in the sampled comparison.")


def experiment_2_cheeger():
    """M2: compare the exact Cheeger constant with the Fiedler sign cut."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Exact Conductance vs Fiedler Sign-Cut Candidate")
    print("=" * 74)
    print("For the optimal conductance h_*, Cheeger gives")
    print("h_*^2/2 <= lambda_2 <= 2h_*. The Fiedler sign cut supplies a")
    print("candidate h_F >= h_*; it is not assumed to be optimal.")
    print()
    print(
        f"  {'graph':20s} {'lambda_2':>9} {'h_* exact':>10} "
        f"{'h_F':>9} {'Fiedler=opt?':>13} {'Cheeger?':>9}"
    )
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
        h_fiedler = _fiedler_conductance(G, A)
        h_exact = _exact_conductance(G)
        fiedler_optimal = abs(h_fiedler - h_exact) <= 1e-12
        cheeger_ok = (
            h_exact * h_exact / 2 - 1e-9
            <= lam2
            <= 2 * h_exact + 1e-9
        )
        print(
            f"  {name:20s} {lam2:>9.4f} {h_exact:>10.4f} "
            f"{h_fiedler:>9.4f} {str(fiedler_optimal):>13} "
            f"{str(cheeger_ok):>9}"
        )
    print()
    print("  -> The exact h_* verifies Cheeger's inequality on these small graphs.")
    print("     Fiedler=opt? is a finite comparison, not a general optimality claim;")
    print("     exhaustive h_* computation does not scale to large networks.")


def experiment_3_threshold():
    """M3: inspect the first non-uniform linear reaction threshold."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Linear Reaction-Diffusion Threshold")
    print("=" * 74)
    print("For sigma_k = r - nu_f*lambda_k, r_c is the first non-uniform")
    print("threshold. The uniform mode already grows for every r > 0.")
    print()
    G = nx.barbell_graph(5, 0)
    _seed(G, np.random.default_rng(0))
    r_c = instability_threshold(G)
    print(f"  barbell (2 K5): r_c = nu_f*lambda_2 = {r_c:.4f}")
    print(f"  {'reaction r':>12} {'unstable modes':>15} {'regime':>22}")
    for label, r in [
        ("0", 0.0),
        ("0.5 r_c", 0.5 * r_c),
        ("0.99 r_c", 0.99 * r_c),
        ("1.5 r_c", 1.5 * r_c),
    ]:
        sigma = dispersion_relation(G, reaction_rate=r)
        n_unstable = int(np.sum(sigma > 1e-9))
        if n_unstable == 0:
            regime = "no growing mode"
        elif n_unstable == 1:
            regime = "uniform mode only"
        else:
            regime = "non-uniform mode grows"
        print(f"  {label:>12} {n_unstable:>15} {regime:>22}")
    print()
    print("  -> Crossing r_c activates the first non-uniform mode of this linear")
    print("     model. This does not establish a grammar-U2 stability theorem or")
    print("     the nonlinear engine trajectory beyond the stated dispersion law.")


def experiment_4_coemergent_tree():
    """M4: place one EPI-distance MST in a finite gap comparison."""
    print()
    print("=" * 74)
    print("EXPERIMENT 4: EPI-Distance MST Sample in a Finite Graph Comparison")
    print("=" * 74)
    print("_mst_edges_from_epi minimizes total absolute EPI edge distance on its")
    print("complete weighted construction. That objective does not minimize the")
    print("normalized spectral gap, so the gap is measured rather than inferred.")
    print()
    N = 14
    print(f"  {'graph':18s} {'lambda_2':>9} {'is_tree':>8}")
    graphs = [
        ("EPI-distance MST", _coemergent_tree(N, 0)),
        ("cycle C14", nx.cycle_graph(N)),
        ("random G(14,0.3)", nx.gnp_random_graph(N, 0.3, seed=9)),
        ("complete K14", nx.complete_graph(N)),
    ]
    measured = []
    for name, G in graphs:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        for nd in G.nodes():
            set_attr(G.nodes[nd], ALIAS_VF, 1.0)
        ev, _ = structural_eigenmodes(G)
        gap = float(ev[1])
        measured.append((name, gap))
        print(f"  {name:18s} {gap:>9.4f} {str(nx.is_tree(G)):>8}")
    smallest = min(measured, key=lambda item: item[1])
    print()
    print(f"  -> Smallest gap in this four-graph table: {smallest[0]} "
          f"({smallest[1]:.4f}).")
    print("     This finite ranking is not a universal extremal property of trees,")
    print("     REMESH outputs, or connected graphs.")


def experiment_5_gap_api_consistency():
    """M5: check the shared normalized-Laplacian API convention."""
    print()
    print("=" * 74)
    print("EXPERIMENT 5: Normalized-Gap API Consistency")
    print("=" * 74)
    print("structural_eigenmodes and analyze_spectral_gap.diffusion_gap both use")
    print("lambda_2(L_sym). The latter also reports the distinct combinatorial")
    print("gap lambda_2(D-A). Equality of the normalized values is expected by")
    print("construction and is checked here for API consistency.")
    print()
    print(
        f"  {'graph':18s} {'diffusion l2':>13} {'API diffusion':>13} "
        f"{'combinatorial':>14} {'match?':>7}"
    )
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
        print(
            f"  {name:18s} {lam2:>13.4f} {sg.diffusion_gap:>13.4f} "
            f"{sg.spectral_gap:>14.4f} {str(match):>7}"
        )
    print()
    print("  -> match=True confirms the shared normalized-Laplacian convention.")
    print("     The exact homogeneous pure-EPI amplitude rate is nu_f*lambda_2;")
    print("     its centered quadratic energy bound has exponent 2*nu_f*lambda_2.")
    print("     No decay law for the five-term structural-energy diagnostic is")
    print("     inferred from this API equality.")


def main():
    print()
    print("  TNFR Example 129: Spectral-Gap Clock and Scoped Comparisons")
    print("  Diffusion, Conductance, Linear Threshold, MST, API Convention")
    print("  =================================================================")
    print()
    experiment_1_clock()
    experiment_2_cheeger()
    experiment_3_threshold()
    experiment_4_coemergent_tree()
    experiment_5_gap_api_consistency()
    print()
    print("=" * 74)
    print("SCOPED FINDINGS")
    print("=" * 74)
    print("1. nu_f*lambda_2 is the exact slowest non-uniform amplitude clock for")
    print("   fixed connected symmetric homogeneous pure-EPI diffusion.")
    print("2. Cheeger's inequality concerns exact optimal conductance; a Fiedler")
    print("   sign cut is a candidate whose optimality must be checked separately.")
    print("3. r_c is a first non-uniform threshold of the displayed linear model,")
    print("   independently of grammar-U2 policy claims.")
    print("4. The EPI-distance MST has the smallest gap only in the finite table")
    print("   printed here; MST cost minimality is not spectral-gap minimality.")
    print("5. The two APIs agree because they share L_sym. The comparison proves")
    print("   no relaxation rate for tetrad/current energy or auxiliary modules.")


if __name__ == "__main__":
    main()
