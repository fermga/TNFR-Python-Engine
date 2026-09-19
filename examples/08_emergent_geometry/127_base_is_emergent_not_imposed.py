#!/usr/bin/env python3
"""
Example 127 — Pressure Realization and State-Dependent Topology
==============================================================

The nodal product x_dot = nu_f * DeltaNFR requires a pressure realization.
For the engine's declared neighbor-mean EPI channel on supported nodes,
DeltaNFR_epi = -L_rw*x, where L_rw = I - D^-1 W. This exact identity follows
from that channel definition, not uniquely from the nodal product. Both L_rw
and the combinatorial Laplacian B = D-W are standard graph operators.

The example checks three distinct constructions:

M1 PRESSURE-OPERATOR AGREEMENT. The supplied unweighted graph fixtures compare
   the production EPI channel with -L_rw*x and -B*x. On a d-regular graph,
   B=d*L_rw. The actions can coincide, for example on a unit P2 (d=1), or
   on a constant EPI field where both vanish. A nonzero residual in these
   selected fixtures is not a universal inequality between the two actions.

M2 REPRODUCIBLE OPERATOR. Fixed effective conductance W and the declared
   normalization determine L_rw and its spectrum. Connectivity alone does
   not determine weighted conductance. Nor does L_rw determine resistance:
   W and c*W have the same L_rw for c>0, while B and effective resistance
   scale by c and 1/c respectively. On a unit P2, changing the conductance
   from 1 to 2 leaves L_rw unchanged and changes resistance from 1 to 1/2.
   The repeated construction below checks L_rw only; it does not compute
   resistance or infer a physical metric scale.

M3 STATE-DEPENDENT RECONSTRUCTION. The REMESH helper _mst_edges_from_epi
   constructs a minimum spanning tree using absolute EPI differences. This
   uses internal state, but the distance convention, MST selection and when
   to invoke it are supplied algorithmic choices. The initial graph is not
   the only premise. The construction does not derive an autonomous topology
   law, phase law or capacity law from the nodal equation.

Scope
-----
These are finite implementation checks and a selected graph reconstruction.
They use the shared pressure, diffusion and REMESH owners without establishing
that the selected reconstruction is uniquely implied by TNFR. Spectral
eigenvector overlap is also insufficient to distinguish regularity because
degenerate eigenspaces admit different bases. No dynamical convergence or
physical-emergence claim follows from these snapshot comparisons.

References
----------
- src/tnfr/dynamics/dnfr.py (the canonical dNFR = neighbour-mean = -L_rw*EPI)
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator)
- src/tnfr/operators/remesh.py (_mst_edges_from_epi: topology from the EPI field)
- theory/NODAL_PARAMETER_FOUNDATIONS.md (supplied channels and closure scope)
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
    """L_comb = D-W on the simple unweighted graph fixtures used here."""
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
    """M1: compare the declared EPI pressure channel with two graph operators."""
    print("=" * 74)
    print("EXPERIMENT 1: The Declared EPI Pressure Channel and Its Graph Operator")
    print("=" * 74)
    print("L_comb = D-W and L_rw = I-D^-1 W use different normalizations.")
    print("The declared EPI channel computes neighbor-MEAN minus self = -L_rw*EPI.")
    print("This checks that channel realization on the selected graph fixtures.")
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
    print("  -> The residual checks the neighbor-mean EPI channel identity.")
    print("     On d-regular graphs L_comb=d*L_rw; their actions can coincide")
    print("     at d=1 or on constant EPI. These fixtures do not prove otherwise.")
    print("     The pressure definition selects L_rw; the nodal product alone")
    print("     does not uniquely select a pressure law.")


def experiment_2_no_free_parameters():
    """M2: rebuild the normalized operator from the same conductance."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Rebuilding the Operator from Fixed Conductance")
    print("=" * 74)
    print("Fixed conductance and normalization determine L_rw.")
    print("Rebuild it twice on these unweighted fixtures; compare the results.")
    print()
    print(f"  {'graph':22s} {'lambda_2':>10} {'rebuild identical?':>20}")
    for name, G in _test_graphs():
        nodes, L1 = structural_diffusion_operator(G)
        _, L2 = structural_diffusion_operator(G)
        ev = np.sort(np.linalg.eigvals(L1).real)
        identical = bool(np.allclose(L1, L2))
        print(f"  {name:22s} {ev[1]:>10.6f} {str(identical):>20}")
    print()
    print("  -> L_rw and its spectrum are reproducible for the same conductance.")
    print("     W -> c*W leaves L_rw unchanged but scales resistance by 1/c.")
    print("     Thus L_rw alone does not fix the resistance scale.")


def experiment_3_topology_from_substrate():
    """M3: construct topology from EPI through the selected MST policy."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: State-Dependent Topology under the REMESH MST Policy")
    print("=" * 74)
    print("The REMESH helper _mst_edges_from_epi constructs a topology")
    print("from absolute EPI distances under a supplied MST rule.")
    print("The state supplies the distances; the reconstruction policy is chosen.")
    print()
    nxmod, _ = _get_networkx_modules()
    G = nx.cycle_graph(10)
    _seed(G, np.random.default_rng(2))
    epi = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()}
    emergent_edges = _mst_edges_from_epi(nxmod, list(G.nodes()), epi)
    print(f"  original cycle C10:        {G.number_of_edges()} edges (initial graph)")
    print(f"  EPI-distance MST topology: {len(emergent_edges)} edges (selected rule)")
    print(f"  sample reconstructed edges: {sorted(emergent_edges)[:5]}")
    Ge = nx.Graph()
    Ge.add_nodes_from(G.nodes())
    Ge.add_edges_from(emergent_edges)
    for nd in Ge.nodes():
        set_attr(Ge.nodes[nd], ALIAS_EPI, epi[nd])
        Ge.nodes[nd]["theta"] = G.nodes[nd]["theta"]
        set_attr(Ge.nodes[nd], ALIAS_VF, 1.0)
    _, Le = structural_diffusion_operator(Ge)
    spec_e = np.sort(np.linalg.eigvals(Le).real)
    print(f"  reconstructed graph spectral gap lambda_2 = {spec_e[1]:.4f}")
    print()
    print("  -> Topology is reconstructed from the EPI field using a chosen rule.")
    print("     Initial connectivity, distance convention, rule and invocation")
    print("     remain explicit premises; this is not an autonomous nodal law.")


def main():
    print()
    print("  TNFR Example 127: Pressure Realization and State-Dependent Topology")
    print("  Shared Operators, Conductance Geometry and a Selected REMESH Rule")
    print("  ===================================================================")
    print()
    experiment_1_operator_is_canonical()
    experiment_2_no_free_parameters()
    experiment_3_topology_from_substrate()
    print()
    print("=" * 74)
    print("SCOPE OF THE THREE CHECKS")
    print("=" * 74)
    print("  (1) The declared neighbor-mean EPI pressure agrees with -L_rw*EPI.")
    print("      The bare nodal product does not uniquely select that channel.")
    print("  (2) Fixed conductance determines L_rw and its spectrum. Resistance")
    print("      also retains the conductance scale discarded by L_rw.")
    print("  (3) The MST helper constructs a graph from EPI distances under")
    print("      a supplied rule. Its selection and timing are not derived here.")
    print("These finite checks reuse the engine owners and standard graph algebra.")
    print("They establish no autonomous phase, capacity or topology closure.")


if __name__ == "__main__":
    main()
