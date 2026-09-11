#!/usr/bin/env python3
"""
Example 128 — Finite Alternating EPI/MST Topology Survey
=======================================================

This example alternates two explicitly chosen steps:

1. three Euler steps of the isolated EPI diffusion channel on the current graph;
2. replacement of all edges by the private ``_mst_edges_from_epi`` helper.

The helper is one implementation component used by an MST remeshing mode.  Calling
it directly does not execute a canonical REMESH operator word, test grammar, or
implement the fixed-delay temporal echo in the REMESH contract.  The alternating
map is therefore a numerical surrogate selected by this example.

For ``N=14``, one EPI seed, four initial graphs, ``dt=0.1`` and twenty
iterations, the returned topology becomes constant.  The EPI values continue
to evolve, so this is topology stationarity rather than a fixed point of the
joint graph/state dynamics.  The terminal graph also equals the helper output
for the terminal EPI snapshot.  Pairwise endpoint Jaccards describe this finite
fixture; they do not establish convergence, attractor basins, uniqueness, or
behavior of kNN/community modes or the ``tau_g -> infinity`` limit.

References
----------
- src/tnfr/operators/remesh.py (_mst_edges_from_epi: topology from the EPI field)
- src/tnfr/dynamics/dnfr.py (the canonical dNFR = -L_rw*EPI)
- examples/08_emergent_geometry/127_base_is_emergent_not_imposed.py (one-shot regen)
- examples/08_emergent_geometry/126_two_layers_base_fiber.py (dependency labels)
- AGENTS.md "Transport content (structural diffusion)"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.operators.remesh import _get_networkx_modules, _mst_edges_from_epi

NXMOD, _ = _get_networkx_modules()


def _seed_state(G, rng):
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.5, 0.5)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)


def _evolve_epi(G, steps, dt=0.1):
    """Nodal equation EPI += dt*vf*dNFR on the CURRENT topology (mid-transient)."""
    G.graph["DNFR_WEIGHTS"] = {"epi": 1.0, "phase": 0, "vf": 0, "topo": 0}
    for _ in range(steps):
        default_compute_delta_nfr(G)
        for nd in G.nodes():
            epi = float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0))
            vf = float(get_attr(G.nodes[nd], ALIAS_VF, 0.0))
            dnfr = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            set_attr(G.nodes[nd], ALIAS_EPI, epi + dt * vf * dnfr)


def _topology_from_epi(nodes, epi):
    """Call the private MST helper used by one remeshing implementation mode."""
    return _mst_edges_from_epi(NXMOD, list(nodes), epi)


def _norm_edges(edges):
    """Orientation-independent edge set."""
    return frozenset(tuple(sorted(e)) for e in edges)


def _jaccard(e1, e2):
    e1, e2 = _norm_edges(e1), _norm_edges(e2)
    if not e1 and not e2:
        return 1.0
    return len(e1 & e2) / len(e1 | e2)


def _coemergence_loop(init_topo, seed, n_iter=20, evolve_steps=3):
    """Run the selected finite Euler/MST alternating map."""
    rng = np.random.default_rng(seed)
    G = init_topo.copy()
    _seed_state(G, rng)
    nodes = list(G.nodes())
    traj = []
    for _ in range(n_iter):
        _evolve_epi(G, evolve_steps)
        epi = {nd: get_attr(G.nodes[nd], ALIAS_EPI, 0.0) for nd in nodes}
        new_edges = _topology_from_epi(nodes, epi)
        traj.append(_norm_edges(new_edges))
        H = nx.Graph()
        H.add_nodes_from(nodes)
        H.add_edges_from(new_edges)
        for nd in nodes:
            set_attr(H.nodes[nd], ALIAS_EPI, epi[nd])
            H.nodes[nd]["theta"] = G.nodes[nd]["theta"]
            set_attr(H.nodes[nd], ALIAS_VF, 1.0)
        G = H
    return traj, G


N = 14


def experiment_1_fixed_point():
    """M1: report topology stationarity for one finite fixture."""
    print("=" * 74)
    print("EXPERIMENT 1: Topology Stationarity in One Finite Run")
    print("=" * 74)
    print("Close the loop topology -> evolve EPI -> MST(EPI) -> ... and measure")
    print("whether its edge sequence stabilizes and matches MST(EPI) at the end.")
    print()
    traj, Gfix = _coemergence_loop(nx.cycle_graph(N), seed=0)
    js = [_jaccard(traj[k], traj[k + 1]) for k in range(len(traj) - 1)]
    print("  Jaccard(T_k, T_{k+1}) along the loop (cycle init, seed 0):")
    print("   " + " ".join(f"{j:.2f}" for j in js[:18]))
    print(f"  terminal topology unchanged = {js[-1] > 0.999} "
          f"(Jaccard {js[-1]:.3f})")
    nodes = list(Gfix.nodes())
    epi = {nd: get_attr(Gfix.nodes[nd], ALIAS_EPI, 0.0) for nd in nodes}
    sc = _jaccard(Gfix.edges(), _topology_from_epi(nodes, epi))
    print(f"  terminal T vs MST(EPI_terminal) Jaccard = {sc:.3f}")
    # convergence speed
    stable_at = next(
        (
            k
            for k in range(len(traj) - 1)
            if all(traj[m] == traj[k] for m in range(k, len(traj)))
        ),
        len(traj),
    )
    print(f"  constant edge-set suffix starts at iteration {stable_at}.")
    print("  EPI still evolves, so this is not a joint-state fixed point.")


def experiment_2_washout():
    """M2: compare initial and terminal edges in four finite runs."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Initial/Terminal Edge Overlap")
    print("=" * 74)
    print("The MST helper always returns a spanning tree here. Compare its")
    print("terminal edge set with each imposed initial topology.")
    print()
    print(
        f"  {'initial topology':18s} {'init edges':>10} {'terminal':>12} "
        f"{'Jaccard':>9}"
    )
    for name, G0 in [
        ("cycle C14", nx.cycle_graph(N)),
        ("path P14", nx.path_graph(N)),
        ("complete K14", nx.complete_graph(N)),
        ("random G(14,0.3)", nx.gnp_random_graph(N, 0.3, seed=9)),
    ]:
        G0 = _ensure_connected(G0)
        traj, Gfix = _coemergence_loop(G0, seed=0)
        survive = _jaccard(G0.edges(), Gfix.edges())
        is_tree = nx.is_tree(Gfix)
        print(
            f"  {name:18s} {G0.number_of_edges():>10} "
            f"{Gfix.number_of_edges():>12} {survive:>9.2f}"
            f"  (tree={is_tree})"
        )
    print()
    print("  -> terminal graphs are trees and initial/terminal edge Jaccards are")
    print("     0.10-0.24 in these four runs. Jaccard is not a survival fraction.")


def experiment_3_basins():
    """M3: compare terminal topology suffixes across four initial graphs."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Terminal Topologies Depend on the Initial Graph")
    print("=" * 74)
    print("Different initial topologies (same seed) -> do they reach the same")
    print("constant edge-set suffix, or different ones in this fixture?")
    print()
    fixed = {}
    for name, G0 in [
        ("cycle C14", nx.cycle_graph(N)),
        ("path P14", nx.path_graph(N)),
        ("complete K14", nx.complete_graph(N)),
        ("random G(14,0.3)", nx.gnp_random_graph(N, 0.3, seed=9)),
    ]:
        _, Gfix = _coemergence_loop(_ensure_connected(G0), seed=0)
        fixed[name] = _norm_edges(Gfix.edges())
    names = list(fixed)
    print("  pairwise Jaccard between terminal edge sets:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = len(fixed[names[i]] & fixed[names[j]]) / len(
                fixed[names[i]] | fixed[names[j]]
            )
            print(f"    {names[i]:18s} vs {names[j]:18s}: {ov:.3f}")
    print()
    print("  -> all values are < 1: the four terminal edge sets differ.")
    print("     This finite dependence does not identify dynamical basins or attractors.")


def _ensure_connected(G):
    G = G.copy()
    if not nx.is_connected(G):
        comp = list(nx.connected_components(G))
        for a, b in zip(comp, comp[1:]):
            G.add_edge(next(iter(a)), next(iter(b)))
    return G


def main():
    print()
    print("  TNFR Example 128: Finite Alternating EPI/MST Survey")
    print("  Topology stationarity under one selected numerical surrogate")
    print("  =======================================================")
    print()
    experiment_1_fixed_point()
    experiment_2_washout()
    experiment_3_basins()
    print()
    print("=" * 74)
    print("SCOPED FINDINGS")
    print("=" * 74)
    print("For the stated seed, graph size, Euler step and horizon, the selected")
    print("EPI/MST alternating map reaches a constant topology suffix. EPI keeps")
    print("changing, and the four terminal edge sets differ. The calculation")
    print("uses a private MST helper, not a complete REMESH operator trajectory.")
    print("It proves no universal convergence, co-emergence, basin, other-mode,")
    print("or tau_g-infinity result.")


if __name__ == "__main__":
    main()
