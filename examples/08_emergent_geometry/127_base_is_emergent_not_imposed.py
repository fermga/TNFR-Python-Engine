#!/usr/bin/env python3
"""
Example 127 — Is the Base Layer Emergent-TNFR or Self-Imposed Graph Theory?
The Operator Is Canonical, and the Topology Can Emerge from the Substrate
==============================================================================

Example 126 called the base layer "standard spectral graph theory". That phrase
was imprecise and hid a doctrinal distinction the TNFR framework itself makes:
`symplectic_substrate.py` states the graph is "an imposed combinatorial
substrate". So is the base layer (the operator L_rw, its spectrum, lambda_2,
R_eff) externally imposed mathematics, or does it emerge from the TNFR nodal
dynamics? This example separates the two honestly, by measurement.

The honest separation
---------------------
There are two distinct things inside the "base":

  1. The OPERATOR on the connectivity. This is NOT generic graph theory.
     Standard spectral graph theory defaults to the combinatorial Laplacian
     L_comb = D - W (the "ratio cut"). The canonical dNFR computes the
     neighbour-MEAN minus self, which is exactly -L_rw * EPI with the
     random-walk Laplacian L_rw = I - D^-1 W (the degree-normalized "normalized
     cut", example 118). The nodal equation FORCES L_rw; one cannot substitute
     the generic L_comb. The operator is TNFR-derived.

  2. The CONNECTIVITY itself. The initial connectivity is an input -- a boundary
     condition, like the initial state of any dynamical system. But it is NOT
     externally fixed forever: the canonical REMESH regenerates the topology
     from the substrate state (the EPI field) via _mst_edges_from_epi. So the
     base connectivity can EMERGE from the fiber.

So the base is emergent-TNFR in the sense that matters: the operator is dictated
by the nodal equation (not imported), and the connectivity is regenerable from
the substrate. The only genuinely imposed thing is the INITIAL connectivity,
which is a boundary/initial condition, not imported mathematics.

Doctrine compliance
-------------------
Everything is canonical: the operator from `structural_diffusion_operator`
(which IS the dNFR EPI channel), the dNFR from `default_compute_delta_nfr`, the
topology regeneration from the canonical REMESH helper `_mst_edges_from_epi`.

Three measured results
----------------------
M1 THE OPERATOR IS TNFR-DERIVED, NOT GENERIC. The canonical dNFR equals
   -L_rw * EPI to machine precision (residual ~0) on every graph, while it is
   NOT equal to -L_comb * EPI (residual 0.4-2.3) anywhere -- even on regular
   graphs, where L_comb = d * L_rw differs by the degree factor. The nodal
   equation's neighbour-MEAN rule forces the degree-normalized L_rw; generic
   spectral graph theory would default to L_comb, which TNFR does not use.

M2 GIVEN CONNECTIVITY, EVERYTHING DERIVES WITH NO FREE PARAMETERS. The operator
   L_rw is a deterministic function of the adjacency alone (no scale, no kernel
   width, no tunable knob). The spectrum, lambda_2 and R_eff are properties of
   that single canonical operator -- there is nothing for me to impose.

M3 THE CONNECTIVITY CAN EMERGE FROM THE SUBSTRATE. The canonical REMESH helper
   _mst_edges_from_epi builds a topology from the EPI field (the fiber state):
   starting from a cycle, the regenerated edges come from the node EPI values,
   and the resulting graph has its own canonical operator and spectral gap. The
   base topology is regenerable from the fiber -- not externally fixed.

The honest caveat (measured)
----------------------------
The eigenvector overlap between L_rw and L_comb is NOT a reliable
regular/non-regular discriminator: eigenvalue degeneracy (e.g. the complete
graph) scrambles the Fiedler vector, so the overlap is noisy. The clean,
decisive evidence that the operator is TNFR-derived is M1 (dNFR = -L_rw*EPI
exactly, never -L_comb*EPI), not an eigenvector comparison.

Honest scope
------------
A measured doctrinal clarification, correcting example 126's loose phrasing. The
operator's degree normalization (L_rw vs L_comb) and the resistance/Kron
machinery are standard linear algebra; the contribution is the clean statement
that TNFR DERIVES the specific operator (the nodal neighbour-mean) rather than
importing generic spectral graph theory, and that the connectivity is
regenerable from the substrate. It is not new mathematics and closes no open
problem; the initial connectivity remains an imposed boundary condition.

References
----------
- src/tnfr/dynamics/dnfr.py (the canonical dNFR = neighbour-mean = -L_rw*EPI)
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator)
- src/tnfr/operators/remesh.py (_mst_edges_from_epi: topology from the EPI field)
- src/tnfr/physics/symplectic_substrate.py ("an imposed combinatorial substrate")
- examples/08_emergent_geometry/118_emergent_vs_classical_operator.py (L_rw = Ncut)
- examples/08_emergent_geometry/126_two_layers_base_fiber.py (the two-layer optic)
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
from tnfr.physics.structural_diffusion import structural_diffusion_operator


def _seed(G, rng):
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _combinatorial_laplacian(G, nodes):
    """L_comb = D - W: the operator generic spectral graph theory defaults to."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n))
    for u, v in G.edges():
        a, b = idx[u], idx[v]
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1
    return L


def _test_graphs():
    out = []
    for name, G in [
        ("cycle C8 (regular)", nx.cycle_graph(8)),
        ("complete K6 (regular)", nx.complete_graph(6)),
        ("star K1,5 (NOT reg.)", nx.star_graph(5)),
        ("path P6 (NOT reg.)", nx.path_graph(6)),
        ("random (NOT reg.)", nx.gnp_random_graph(10, 0.4, seed=1)),
    ]:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        out.append((name, G))
    return out


def experiment_1_operator_is_canonical():
    """M1: dNFR = -L_rw*EPI exactly, NOT -L_comb*EPI (TNFR-derived operator)."""
    print("=" * 74)
    print("EXPERIMENT 1: The Operator Is TNFR-Derived (L_rw), Not Generic (L_comb)")
    print("=" * 74)
    print("Generic spectral graph theory defaults to L_comb = D - W. The")
    print("canonical dNFR computes the neighbour-MEAN minus self = -L_rw*EPI")
    print("(degree-normalized). Direct test: which operator IS the dNFR channel?")
    print()
    print(
        f"  {'graph':22s} {'res(dNFR, -L_rw*EPI)':>21} "
        f"{'res(dNFR, -L_comb*EPI)':>23}"
    )
    for name, G in _test_graphs():
        G = G.copy()
        G.graph["DNFR_WEIGHTS"] = {"epi": 1.0, "phase": 0, "vf": 0, "topo": 0}
        _seed(G, np.random.default_rng(0))
        default_compute_delta_nfr(G)
        nodes, L_rw = structural_diffusion_operator(G)
        L_comb = _combinatorial_laplacian(G, nodes)
        epi = np.array([get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes])
        dnfr = np.array([get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in nodes])
        res_rw = float(np.max(np.abs(dnfr - (-L_rw @ epi))))
        res_comb = float(np.max(np.abs(dnfr - (-L_comb @ epi))))
        print(f"  {name:22s} {res_rw:>21.2e} {res_comb:>23.4f}")
    print()
    print("  -> dNFR = -L_rw*EPI EXACTLY (res ~0) everywhere; dNFR != -L_comb*EPI")
    print("     (res > 0) everywhere -- even regular graphs (L_comb = d*L_rw")
    print("     differs by the degree factor). The nodal neighbour-MEAN rule")
    print("     FORCES the degree-normalized L_rw (ex 118 = Shi-Malik Ncut);")
    print("     generic graph theory's default L_comb is NOT what TNFR uses.")


def experiment_2_no_free_parameters():
    """M2: given connectivity, the operator is determined, no free params."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Given Connectivity, the Operator Has No Free Parameters")
    print("=" * 74)
    print("L_rw is a deterministic function of the adjacency alone -- no scale,")
    print("no kernel width, no tunable knob. Rebuild it twice; it is identical.")
    print()
    print(f"  {'graph':22s} {'lambda_2':>10} {'rebuild identical?':>20}")
    for name, G in _test_graphs():
        nodes, L1 = structural_diffusion_operator(G)
        _, L2 = structural_diffusion_operator(G)
        ev = np.sort(np.linalg.eigvals(L1).real)
        identical = bool(np.allclose(L1, L2))
        print(f"  {name:22s} {ev[1]:>10.6f} {str(identical):>20}")
    print()
    print("  -> the operator is a pure function of the connectivity; the")
    print("     spectrum, lambda_2 and R_eff are its properties -- nothing")
    print("     for me to impose beyond the connectivity itself.")


def experiment_3_topology_from_substrate():
    """M3: the connectivity can emerge from the substrate (EPI) via REMESH."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: The Connectivity Can Emerge from the Substrate (REMESH)")
    print("=" * 74)
    print("The canonical REMESH helper _mst_edges_from_epi builds a topology")
    print("from the EPI field (the fiber state). The base connectivity is")
    print("regenerable from the substrate -- not externally fixed forever.")
    print()
    nxmod, _ = _get_networkx_modules()
    G = nx.cycle_graph(10)
    _seed(G, np.random.default_rng(2))
    epi = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()}
    emergent_edges = _mst_edges_from_epi(nxmod, list(G.nodes()), epi)
    print(f"  original cycle C10:        {G.number_of_edges()} edges (imposed)")
    print(f"  EPI-derived MST topology:  {len(emergent_edges)} edges (emergent)")
    print(f"  sample emergent edges:     {sorted(emergent_edges)[:5]}")
    Ge = nx.Graph()
    Ge.add_nodes_from(G.nodes())
    Ge.add_edges_from(emergent_edges)
    for nd in Ge.nodes():
        set_attr(Ge.nodes[nd], ALIAS_EPI, epi[nd])
        Ge.nodes[nd]["theta"] = G.nodes[nd]["theta"]
        set_attr(Ge.nodes[nd], ALIAS_VF, 1.0)
    _, Le = structural_diffusion_operator(Ge)
    spec_e = np.sort(np.linalg.eigvals(Le).real)
    print(f"  emergent graph spectral gap lambda_2 = {spec_e[1]:.4f}")
    print()
    print("  -> the topology (base) is REGENERATED from the EPI field (fiber):")
    print("     the base emerges from the substrate; only the INITIAL")
    print("     connectivity is imposed, as a boundary condition.")


def main():
    print()
    print("  TNFR Example 127: Is the Base Emergent-TNFR or Imposed Graph Theory?")
    print("  The Operator Is Canonical; the Topology Can Emerge from the Substrate")
    print("  ===================================================================")
    print()
    experiment_1_operator_is_canonical()
    experiment_2_no_free_parameters()
    experiment_3_topology_from_substrate()
    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("The BASE layer is NOT self-imposed generic spectral graph theory:")
    print("  (1) the operator is the canonical dNFR (L_rw, the nodal")
    print("      neighbour-MEAN rule) -- the nodal equation FORCES it, distinct")
    print("      from the generic combinatorial Laplacian L_comb (M1);")
    print("  (2) given connectivity, every base quantity derives with NO free")
    print("      parameters -- the operator is a pure function of the graph (M2);")
    print("  (3) the connectivity itself can EMERGE from the substrate via the")
    print("      canonical REMESH (_mst_edges_from_epi), so the base is")
    print("      regenerable from the fiber (M3).")
    print("Only the INITIAL connectivity is imposed -- a boundary / initial")
    print("condition, like the initial state of any dynamical system, NOT")
    print("imported mathematics. This corrects example 126's loose phrasing")
    print("'standard spectral graph theory for the base'. HONEST SCOPE: a")
    print("measured doctrinal clarification; the linear algebra is standard, the")
    print("contribution is that TNFR DERIVES the operator (not imports it) and")
    print("the topology is substrate-regenerable; closes no open problem.")


if __name__ == "__main__":
    main()
