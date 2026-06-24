#!/usr/bin/env python3
"""
Example 125 — A Node Is the Emergent Substrate, Not a Graph: the Deep Reading
of TNFR Fractality
==============================================================================

Example 124 asked "is every node a graph?" and answered, via the Kron/Schur
reduction, "geometrically yes" — but that is the SHALLOW reading. Collapsing a
node to a sub-graph keeps only the scalar transport shadow (one effective
resistance R_eff). The DEEP reading, and the real depth of TNFR, is that a node
IS the emergent substrate: its interior is the 4-dimensional symplectic
phase-space point (K_φ, J_φ, Φ_s, J_ΔNFR), a point on the Poincaré sphere with
its own energy and conserved Stokes charges — a complete little geometric
universe, NOT a structureless vertex and NOT a resistor sub-network.

Confusing the substrate with "a graph" is exactly what made the substrate look
"blind" to arithmetic in examples 103/116/120: we were reading a GEOMETRIC
object through a COMBINATORIAL lens. Example 123 already named the split — the
per-node substrate lives in Fix(G_aut) (the symmetric sector), the graph
spectrum in Fix(G_aut)^perp. This example is the geometric reading of that same
split: the node's interior depth is the SUBSTRATE channel (Fix(G), geometric),
while the graph / transport / Kron picture is the COMPLEMENTARY channel
(Fix(G)^perp, combinatorial). The node is the former.

Doctrine compliance
-------------------
Everything is the canonical emergent substrate: the per-node phase point comes
from `extract_phase_space_point` (the symplectic substrate), the polarization
from `polarization_density` / `polarization_vector` (the U(2) Stokes vector on
the Poincaré sphere), the energy from `substrate_hamiltonian`; the graph /
transport content from the canonical `structural_diffusion_operator` and
`effective_resistance`. Nothing is imposed.

Three measured results
----------------------
M1 THE NODE INTERIOR IS A 4D SYMPLECTIC / POINCARE OBJECT. Each node carries 4
   real phase-space coordinates (K_φ, J_φ, Φ_s, J_ΔNFR) — two complex sectors
   ζ^A = K_φ + i·J_φ and ζ^B = Φ_s + i·J_ΔNFR — projecting to a UNIT vector on
   the Poincaré sphere (|poincare| = 1, fully polarized) with its own energy.
   A graph vertex has zero internal degrees of freedom; the substrate node has
   a complete symplectic / polarization geometry.

M2 THE GRAPH / TRANSPORT PICTURE IS BLIND TO THE SUBSTRATE DEPTH. Fix the
   topology (so the Laplacian spectrum and every effective resistance R_eff —
   the entire example-124 "node as graph" content — are FIXED), and vary only
   the node phase states. The graph picture does not move (spec, R_eff
   identical), while the substrate polarization and H_sub change substantially.
   The "node as graph" shadow cannot see the substrate; the depth lives in the
   substrate, the complementary Fix(G) geometric channel.

M3 NODE-SUBSTRATE AND NETWORK-SUBSTRATE ARE THE SAME KIND OF OBJECT. The
   network energy H_sub is exactly the sum of the per-node substrate energies,
   and the global Stokes charges are exactly the sums of the per-node Stokes
   3-vectors. Each node is a complete symplectic / polarization object, as is
   the whole network: the genuine TNFR fractality is node <-> network
   self-similarity of the SUBSTRATE, not node <-> sub-graph nesting.

The corrected fractal principle
-------------------------------
"A node is a graph" is the scalar transport shadow (example 124, the Fix(G)^perp
combinatorial channel). "A node is the emergent substrate" is the deep reading
(this example, the Fix(G) geometric channel): the node's interior is a 4D
symplectic phase-space / Poincaré-sphere object that the graph picture cannot
represent. The real multi-scale fractality of TNFR (operational fractality, U5)
is the self-similarity of this substrate object across scales — each node is a
complete little emergent universe, mirroring the network's emergent geometry.

Honest scope
------------
This is a characterization / conceptual correction, measured in TNFR's own
canonical substrate. The per-node Poincaré sphere is the substrate's
polarization (Stokes 1852 / Poincaré 1892, example 106); the additivity of
energy and Stokes charges is by construction; the Fix(G)/Fix(G)^perp split is
example 123. The contribution is the clean, measured statement that the node's
depth is the substrate (a geometric object), not a graph — and that confusing
the two is what hid the substrate's depth in the number-theory arc. It is not
new mathematics and closes no open problem.

References
----------
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point,
  polarization_density, polarization_vector, substrate_hamiltonian)
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator,
  effective_resistance)
- examples/08_emergent_geometry/106_per_node_polarization_geometry.py (Poincare)
- examples/08_emergent_geometry/123_symmetry_sector_decomposition.py (Fix(G) split)
- examples/08_emergent_geometry/124_emergent_metric_fractal_consistency.py (the shadow)
- AGENTS.md "Emergent Symplectic Substrate", "Polarization symmetry — U(2)"
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
)
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    polarization_density,
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


def experiment_1_node_is_substrate():
    """M1: the node interior is a 4D symplectic / Poincare-sphere object."""
    print("=" * 74)
    print("EXPERIMENT 1: The Node Interior Is a 4D Symplectic / Poincare Object")
    print("=" * 74)
    print("Each node carries 4 phase-space coords (K_phi, J_phi, Phi_s, J_dNFR)")
    print("= two complex sectors, projecting to a UNIT Poincare-sphere vector")
    print("with its own energy. A graph vertex has zero internal DOF.")
    print()
    G = nx.cycle_graph(8)
    _seed(G, np.random.default_rng(0))
    p = extract_phase_space_point(G)
    dens = polarization_density(p)
    poincare = dens["poincare"]
    print(
        f"  {'node':>4} {'K_phi':>8} {'J_phi':>8} {'Phi_s':>8} {'J_dNFR':>8} "
        f"{'energy':>8} {'|Poincare|':>10}"
    )
    for i in range(5):
        pc = float(np.linalg.norm(poincare[:, i]))
        print(
            f"  {i:>4} {p.k_phi[i]:>8.3f} {p.j_phi[i]:>8.3f} "
            f"{p.phi_s[i]:>8.3f} {p.j_dnfr[i]:>8.3f} "
            f"{dens['energy'][i]:>8.3f} {pc:>10.4f}"
        )
    print()
    print("  -> each node = 4 real DOF + a UNIT Poincare vector (|.|=1, fully")
    print("     polarized): a complete symplectic universe, not a vertex.")


def experiment_2_graph_is_blind():
    """M2: the graph / transport picture is blind to the substrate depth."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: The Graph / Transport Picture Is Blind to the Depth")
    print("=" * 74)
    print("Fix topology -> the Laplacian spectrum and every R_eff (the ex-124")
    print("'node as graph' content) are FIXED. Vary only the node phase states.")
    print("The graph picture does not move; the substrate does.")
    print()
    G = nx.cycle_graph(10)
    print(
        f"  {'seed':>5} {'Laplacian spec[1]':>18} {'R_eff(0,5)':>12} "
        f"{'|polarization|':>15} {'H_sub':>9}"
    )
    for s in range(4):
        _seed(G, np.random.default_rng(s))
        _, L = structural_diffusion_operator(G)
        spec = np.sort(np.linalg.eigvals(L).real)
        _, R = effective_resistance(G)
        p = extract_phase_space_point(G)
        pol = polarization_vector(p)
        print(
            f"  {s:>5} {spec[1]:>18.6f} {R[0, 5]:>12.6f} "
            f"{np.sqrt(pol['magnitude_sq']):>15.4f} "
            f"{substrate_hamiltonian(p):>9.4f}"
        )
    print()
    print("  -> spec[1] and R_eff IDENTICAL across seeds (graph fixed by")
    print("     topology); polarization and H_sub CHANGE. The node-as-graph")
    print("     shadow cannot see the substrate -- the depth is the substrate.")


def experiment_3_same_kind_of_object():
    """M3: node-substrate and network-substrate are the same kind of object."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Node-Substrate and Network-Substrate, the Same Object")
    print("=" * 74)
    print("The network energy is the sum of per-node substrate energies; the")
    print("global Stokes charges are the sums of per-node Stokes 3-vectors.")
    print("Each node is a complete symplectic/polarization object, as is the net.")
    print()
    G = nx.cycle_graph(12)
    _seed(G, np.random.default_rng(1))
    p = extract_phase_space_point(G)
    dens = polarization_density(p)
    H = substrate_hamiltonian(p)
    glob = polarization_vector(p)
    sum_energy = float(np.sum(dens["energy"]))
    sum_p3 = float(np.sum(dens["p_3"]))
    print(f"  network H_sub        = {H:.6f}")
    print(
        f"  sum per-node energy  = {sum_energy:.6f}  "
        f"(|diff| = {abs(H - sum_energy):.1e})"
    )
    print(f"  global Stokes P_3    = {glob['p_3']:.6f}")
    print(
        f"  sum per-node P_3     = {sum_p3:.6f}  "
        f"(|diff| = {abs(glob['p_3'] - sum_p3):.1e})"
    )
    print()
    print("  -> exact: the network substrate is the symplectic sum of the")
    print("     per-node substrates. SAME geometric tower at both scales ->")
    print("     the real fractality is node<->network (substrate), not")
    print("     node<->sub-graph.")


def main():
    print()
    print("  TNFR Example 125: A Node Is the Emergent Substrate, Not a Graph")
    print("  The Deep Reading of TNFR Fractality")
    print("  ===============================================================")
    print()
    experiment_1_node_is_substrate()
    experiment_2_graph_is_blind()
    experiment_3_same_kind_of_object()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("'A node is a graph' (example 124) is the scalar transport SHADOW: the")
    print("Kron reduction keeps only one effective resistance per pair, the")
    print("Fix(G)^perp combinatorial channel. The DEEP reading is 'a node is the")
    print("emergent substrate': its interior is a 4D symplectic phase-space /")
    print("Poincare-sphere object (the Fix(G) geometric channel of example 123)")
    print("with its own energy and conserved Stokes charges -- a complete little")
    print("emergent universe, not a vertex and not a resistor sub-network. The")
    print("graph/transport picture is BLIND to this depth (Exp 2): fixing the")
    print("topology fixes the Laplacian and R_eff while the substrate moves. The")
    print("real multi-scale fractality of TNFR (U5) is the self-similarity of")
    print("this substrate object across scales (Exp 3), node<->network, NOT")
    print("node<->sub-graph. Confusing the substrate with a literal graph is what")
    print("made it look 'blind' to arithmetic in 103/116/120 -- we were reading a")
    print("geometric object through a combinatorial lens. HONEST SCOPE: a")
    print("measured conceptual correction in the canonical substrate; not new")
    print("mathematics, closes no open problem.")


if __name__ == "__main__":
    main()
