#!/usr/bin/env python3
"""
Example 126 — Fixed-Graph Data and Auxiliary Field Read-outs
============================================================

This example uses a limited bookkeeping split.  For a fixed unweighted graph,
the random-walk Laplacian, its spectrum and effective resistance depend on the
graph.  The extracted tuple ``(K_phi, J_phi, Phi_s, J_dNFR)`` also depends on
the node snapshot.  Calling these groups "base" and "fiber" can organize the
printed quantities, but it is not a canonical bundle construction.

The exact bridge tested here is narrower: in the isolated EPI channel,
``dNFR_epi = -L_rw*EPI``.  With a common positive capacity on a fixed connected
symmetric graph, ``nu_f*lambda_2`` is the slowest nonuniform amplitude-decay
rate.  This identity evolves EPI; it does not generate the separately declared
harmonic flow declared by ``symplectic_substrate.py`` or prove that extracted
field tuples are invariant fiber coordinates.

Engine operators act on graph/node state.  A changed auxiliary read-out after
an operator call is a response of the extractor, not proof that the operator
acts as a canonical map on an ambient substrate.  Likewise, sorting prior
examples into labels is commentary rather than a mathematical result.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator,
  effective_resistance, verify_structural_diffusion)
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point,
  substrate_hamiltonian, polarization_vector)
- examples/08_emergent_geometry/123_symmetry_sector_decomposition.py (Fix split)
- examples/08_emergent_geometry/125_node_is_the_emergent_substrate.py (per-node auxiliary read-outs)
- examples/08_emergent_geometry/112_structure_predicts_coherence_flow.py (nu_f*lambda_2)
- AGENTS.md "Transport Content of the Nodal Equation", "Emergent Symplectic Substrate"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import (
    effective_resistance,
    structural_diffusion_operator,
    verify_structural_diffusion,
)
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    polarization_vector,
    substrate_hamiltonian,
)


def _seed(G, rng):
    """Seed a graph snapshot for fixed-topology and field comparisons."""
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def experiment_1_base_state_independent():
    """M1: the base layer (topology) is state-independent."""
    print("=" * 74)
    print("EXPERIMENT 1: Fixed-Unweighted-Graph Quantities")
    print("=" * 74)
    print("The operator L_rw and everything derived (lambda_2, R_eff, spectrum)")
    print("is a function of the graph alone. Vary the node states; the base")
    print("does not move in this fixture.")
    print()
    G = nx.cycle_graph(10)
    print(
        f"  {'seed':>5} {'lambda_2':>10} {'spec[2]':>10} {'R_eff(0,5)':>12} "
        f"{'trace(L)':>10}"
    )
    for s in range(4):
        _seed(G, np.random.default_rng(s))
        _, L = structural_diffusion_operator(G)
        ev = np.sort(np.linalg.eigvals(L).real)
        _, R = effective_resistance(G)
        print(
            f"  {s:>5} {ev[1]:>10.6f} {ev[2]:>10.6f} {R[0, 5]:>12.6f} "
            f"{np.trace(L):>10.4f}"
        )
    print()
    print("  -> identical because topology and edge weights are fixed.")


def experiment_2_fiber_state_dependent():
    """M2: the fiber layer (substrate) is state-dependent."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: State-Dependent Auxiliary Read-outs")
    print("=" * 74)
    print("The extracted ambient coordinates are functions of graph state. The")
    print("same state variation moves H_sub, polarization, and Stokes charge.")
    print()
    G = nx.cycle_graph(10)
    print(f"  {'seed':>5} {'H_sub':>10} {'|polarization|':>15} {'P_3':>10}")
    for s in range(4):
        _seed(G, np.random.default_rng(s))
        p = extract_phase_space_point(G)
        pol = polarization_vector(p)
        print(
            f"  {s:>5} {substrate_hamiltonian(p):>10.4f} "
            f"{np.sqrt(pol['magnitude_sq']):>15.4f} {pol['p_3']:>10.4f}"
        )
    print()
    print("  -> these derived auxiliary read-outs move with the seeded state.")


def experiment_3_coupling_exact():
    """M3: the coupling is exact (dNFR_epi = -L_rw*EPI, canonical verify)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Exact Isolated EPI-Channel Identity")
    print("=" * 74)
    print("The verifier checks dNFR_epi = -L_rw*EPI. For the homogeneous,")
    print("fixed-graph pure-EPI model, nu_f*lambda_2 is its slowest nonuniform")
    print("amplitude rate. No auxiliary harmonic flow is tested here.")
    print()
    print(
        f"  {'graph':18s} {'dNFR=-L_rw*EPI':>15} {'residual':>10} "
        f"{'lambda_2':>9} {'nu_f*lambda_2':>13}"
    )
    cases = [
        ("path P12", nx.path_graph(12)),
        ("cycle C10", nx.cycle_graph(10)),
        ("random G(14,0.4)", nx.gnp_random_graph(14, 0.4, seed=2)),
    ]
    for name, G in cases:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        _seed(G, np.random.default_rng(0))
        cert = verify_structural_diffusion(G)
        print(
            f"  {name:18s} {str(cert.dnfr_is_graph_laplacian):>15} "
            f"{cert.max_laplacian_residual:>10.1e} {cert.spectral_gap:>9.4f} "
            f"{cert.slowest_relaxation_rate:>13.4f}"
        )
    print()
    print("  -> residual ~0 verifies the isolated EPI-channel identity.")
    print("     The reported rate has the restricted homogeneous diffusion scope;")
    print("     no rate for the auxiliary substrate follows.")


def experiment_4_reorganization_map():
    """M4: present the optional bookkeeping labels with explicit scope."""
    print()
    print("=" * 74)
    print("EXPERIMENT 4: Optional Bookkeeping Map")
    print("=" * 74)
    print("The following labels summarize dependencies; they prove no bundle.")
    print()
    print("  BASE (topology / spectrum -- state-independent):")
    print("    99 structural diffusion   118 Shi-Malik normalized cut")
    print("    119/122 arithmetic spectrum   123 Fix(G)/Fix(G)^perp split")
    print("    124 effective resistance / Kron reduction")
    print("    -> these entries primarily inspect fixed-graph operators.")
    print()
    print("  AUXILIARY READ-OUTS (state and graph dependent):")
    print("    98 symplectic substrate   106 per-node polarization")
    print("    114 harmonic-flow charges   125 per-node read-outs")
    print("    -> operator before/after tables are extractor response audits.")
    print()
    print("  RESTRICTED BRIDGE (isolated pure-EPI diffusion):")
    print("    112 structure predicts the flow (nu_f*lambda_2)")
    print("    -> nu_f*lambda_2 clocks nonuniform homogeneous EPI amplitudes.")


def main():
    print()
    print("  TNFR Example 126: Fixed-Graph Data and Auxiliary Read-outs")
    print("  Bookkeeping labels plus the restricted pure-EPI bridge")
    print("  =================================================================")
    print()
    experiment_1_base_state_independent()
    experiment_2_fiber_state_dependent()
    experiment_3_coupling_exact()
    experiment_4_reorganization_map()
    print()
    print("=" * 74)
    print("SCOPED FINDINGS")
    print("=" * 74)
    print("Fixed unweighted topology keeps its graph-only quantities constant,")
    print("whereas extracted auxiliary fields change with the seeded snapshot.")
    print("The exact relation dNFR_epi=-L_rw*EPI belongs to isolated pure-EPI")
    print("diffusion. It does not derive the auxiliary harmonic substrate or")
    print("turn these bookkeeping labels into a canonical base/fiber theory.")


if __name__ == "__main__":
    main()
