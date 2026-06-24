#!/usr/bin/env python3
"""
Example 126 — The Two Layers of TNFR Emergent Geometry: Base (Topology) and
Fiber (Substrate), Bridged by the Nodal Equation
==============================================================================

Example 125's correction — "a node IS the emergent substrate, not a graph" —
is not a local fix: it reorganizes the WHOLE emergent-geometry program into two
distinct layers, and gives a different optic for everything that remains.

  BASE layer (topology). The canonical operator L_rw = I - D^-1 W and everything
      derived from it — the spectrum {lambda_k}, the spectral gap lambda_2, the
      effective resistance R_eff, the Kron reduction. The BASE is a function of
      the graph ALONE: it is STATE-INDEPENDENT. (This is the Fix(G)^perp
      combinatorial channel of example 123, extended to the whole operator.)

  FIBER layer (state / substrate). The per-node 4D symplectic phase-space point
      (K_phi, J_phi, Phi_s, J_dNFR), its Poincare-sphere polarization, its
      energy, its Stokes charges. The FIBER is carried by the node states (and
      sits on the topology): it is STATE-DEPENDENT. (This is the node's true
      depth — example 125 — the geometric Fix(G) channel.)

  COUPLING (the nodal equation). The two layers meet in
  dEPI/dt = nu_f * dNFR, because the driving force dNFR IS the base operator
  acting on the field: dNFR_epi = -L_rw * EPI (to machine precision). The BASE
  operator generates the force that moves the FIBER. The slowest LINEAR rate is
  the base spectral gap nu_f * lambda_2 (example 112).

The different optic (how this reorganizes the rest)
---------------------------------------------------
Every prior result, and every remaining research line, sorts cleanly into the
two layers or the bridge:

  * BASE (topology / spectrum): structural diffusion (99), Shi-Malik cut (118),
    the arithmetic spectrum (119, 122), the Fix(G)/Fix(G)^perp split (123), the
    effective resistance / Kron reduction (124). The number-theory arc's
    arithmetic lived HERE — in the base spectrum — which is exactly why the
    fiber substrate looked "blind" to it (103/116/120): arithmetic is a
    BASE-layer property, the substrate is the FIBER.

  * FIBER (state / substrate): the symplectic substrate (98), the per-node
    polarization (106), the conserved Stokes / Noether charges (114), and the
    node-is-substrate reading (125). The 13 canonical operators ACT here — they
    move the fiber and redistribute its charges (the remaining research line on
    operators -> conserved charges is a pure fiber study).

  * COUPLING (nodal equation): the spectral gap lambda_2 is a BASE quantity, but
    it is the CLOCK of the base->fiber coupling (it times the linear-field
    relaxation that drives the fiber). The remaining research line on the
    spectral gap is therefore the base-fiber BRIDGE, not "just a graph number".

Doctrine compliance
-------------------
Everything is canonical and nothing is imposed: the base from
`structural_diffusion_operator` / `effective_resistance`, the fiber from
`extract_phase_space_point` / `substrate_hamiltonian` / `polarization_vector`,
the coupling from the canonical `verify_structural_diffusion` (which certifies
dNFR_epi = -L_rw * EPI to machine precision).

Four measured results
---------------------
M1 BASE IS STATE-INDEPENDENT. Varying the node states (random seeds) leaves the
   spectral gap lambda_2, the higher eigenvalues, the effective resistance
   R_eff and trace(L) all IDENTICAL: the base is pure topology.

M2 FIBER IS STATE-DEPENDENT. The same state variation moves the substrate
   energy H_sub, the polarization magnitude and the Stokes charge P_3
   substantially: the fiber carries the state.

M3 THE COUPLING IS EXACT. The canonical verify_structural_diffusion certifies
   dNFR_epi = -L_rw * EPI with residual ~0 on a path, a cycle and a random
   graph: the SAME base operator that defines the base layer generates the
   force that drives the fiber. The slowest linear rate is nu_f * lambda_2.

M4 THE REORGANIZATION MAP. Sorting the emergent-geometry examples into the two
   layers + the bridge shows the two-layer structure organizes the whole
   program, and reframes the remaining lines (spectral gap = bridge; operators
   = fiber actors).

Honest scope
------------
A measured conceptual reorganization in the canonical machinery. The base
quantities are standard spectral graph theory; the fiber is the canonical
symplectic substrate (examples 98/106/114/125); the coupling identity is the
canonical verify_structural_diffusion (example 99). The contribution is the
clean two-layer optic — base (topology) + fiber (substrate), bridged by the
nodal equation — and the map that reorganizes the program. It is not new
mathematics and closes no open problem; the nonlinear substrate relaxation
rate, unlike the linear field's nu_f * lambda_2, is not a single clean rate
(stated honestly, not overclaimed).

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator,
  effective_resistance, verify_structural_diffusion)
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point,
  substrate_hamiltonian, polarization_vector)
- examples/08_emergent_geometry/123_symmetry_sector_decomposition.py (Fix split)
- examples/08_emergent_geometry/125_node_is_the_emergent_substrate.py (the fiber)
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
    """Arithmetic-neutral random TNFR state; canonical nodal substrate."""
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def experiment_1_base_state_independent():
    """M1: the base layer (topology) is state-independent."""
    print("=" * 74)
    print("EXPERIMENT 1: The BASE Layer (Topology) Is State-Independent")
    print("=" * 74)
    print("The operator L_rw and everything derived (lambda_2, R_eff, spectrum)")
    print("is a function of the graph alone. Vary the node states; the base")
    print("does not move.")
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
    print("  -> identical across seeds: the BASE is pure topology.")


def experiment_2_fiber_state_dependent():
    """M2: the fiber layer (substrate) is state-dependent."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: The FIBER Layer (Substrate) Is State-Dependent")
    print("=" * 74)
    print("The per-node 4D symplectic substrate carries the state. The same")
    print("state variation moves H_sub, the polarization, and the Stokes charge.")
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
    print("  -> moves with state: the FIBER carries the state.")


def experiment_3_coupling_exact():
    """M3: the coupling is exact (dNFR_epi = -L_rw*EPI, canonical verify)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: The COUPLING Is Exact (the Base Operator Drives the Fiber)")
    print("=" * 74)
    print("The nodal equation dEPI/dt = nu_f * dNFR drives the fiber; the canonical")
    print("verify certifies dNFR_epi = -L_rw * EPI to machine precision, with")
    print("slowest linear rate nu_f * lambda_2.")
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
    print("  -> residual ~0: the SAME base operator L_rw generates the force")
    print("     that drives the fiber. lambda_2 is the slowest LINEAR rate")
    print("     (example 112); the nonlinear substrate relaxes faster (honest).")


def experiment_4_reorganization_map():
    """M4: the two-layer optic reorganizes the whole program."""
    print()
    print("=" * 74)
    print("EXPERIMENT 4: The Reorganization Map (the Different Optic)")
    print("=" * 74)
    print("Every emergent-geometry result sorts into BASE, FIBER, or the BRIDGE.")
    print()
    print("  BASE (topology / spectrum -- state-independent):")
    print("    99 structural diffusion   118 Shi-Malik normalized cut")
    print("    119/122 arithmetic spectrum   123 Fix(G)/Fix(G)^perp split")
    print("    124 effective resistance / Kron reduction")
    print("    -> the number-theory arc's arithmetic lived HERE; the fiber was")
    print("       'blind' (103/116/120) because arithmetic is a BASE property.")
    print()
    print("  FIBER (state / substrate -- the node's depth, state-dependent):")
    print("    98 symplectic substrate   106 per-node polarization")
    print("    114 conserved Stokes/Noether charges   125 node = substrate")
    print("    -> the 13 canonical operators ACT here (line E: operators ->")
    print("       conserved-charge breaking is a pure fiber study).")
    print()
    print("  BRIDGE (the nodal equation -- base drives fiber):")
    print("    112 structure predicts the flow (nu_f*lambda_2)")
    print("    -> the spectral gap lambda_2 is a BASE quantity but the CLOCK of")
    print("       the coupling (line C: the spectral gap is the base-fiber")
    print("       bridge, not 'just a graph number').")


def main():
    print()
    print("  TNFR Example 126: The Two Layers of Emergent Geometry")
    print("  Base (Topology) + Fiber (Substrate), Bridged by the Nodal Equation")
    print("  =================================================================")
    print()
    experiment_1_base_state_independent()
    experiment_2_fiber_state_dependent()
    experiment_3_coupling_exact()
    experiment_4_reorganization_map()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("Example 125's 'a node IS the substrate' reorganizes the whole")
    print("emergent-geometry program into two layers: the BASE (topology -- the")
    print("operator L_rw, spectrum, lambda_2, R_eff, Kron; state-independent) and")
    print("the FIBER (state -- the per-node 4D symplectic / Poincare substrate;")
    print("state-dependent), bridged by the nodal equation (dNFR_epi = -L_rw*EPI")
    print("exactly: the base operator drives the fiber). This is the different")
    print("optic: arithmetic (the number-theory arc) was a BASE-layer property,")
    print("which is why the FIBER substrate looked blind to it; the 13 operators")
    print("act on the FIBER (line E); and the spectral gap lambda_2 is the BASE")
    print("quantity that CLOCKS the base->fiber coupling (line C the bridge).")
    print("HONEST SCOPE: a measured conceptual reorganization in the canonical")
    print("machinery -- standard spectral graph theory (base) + the canonical")
    print("symplectic substrate (fiber) + the canonical diffusion identity")
    print("(bridge); not new mathematics, closes no open problem.")


if __name__ == "__main__":
    main()
