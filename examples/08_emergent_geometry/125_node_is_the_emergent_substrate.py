#!/usr/bin/env python3
"""
Example 125 — Per-Node Read-outs of the Auxiliary Substrate
===========================================================

``extract_phase_space_point`` assigns four derived numbers to each graph node:
``(K_phi, J_phi, Phi_s, J_dNFR)``.  The auxiliary substrate treats those values
as ambient canonical coordinates and packages each nonzero complex doublet as
a Stokes/Poincare vector.  The graph fields depend on neighbors and on the full
snapshot, so this representation does not prove that a node has four
independent internal degrees of freedom or that it literally *is* a substrate.

The experiments verify three narrower facts.  Nonzero per-node doublets obey a
unit-vector normalization identity.  With unweighted topology fixed, topology-
only Laplacian and resistance values remain fixed while state-derived read-outs
change.  Finally, the implemented global energy and Stokes components equal
sums of their per-node densities by definition.

These algebraic aggregation identities do not establish U5 nesting,
node/network self-similarity, a Fix(G) sector assignment, or a dynamically
invariant symplectic manifold.  They characterize the auxiliary read-out only.

References
----------
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point,
  polarization_density, polarization_vector, substrate_hamiltonian)
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator,
  effective_resistance)
- examples/08_emergent_geometry/106_per_node_polarization_geometry.py (Poincare)
- examples/08_emergent_geometry/123_symmetry_sector_decomposition.py (symmetry-sector diagnostics)
- examples/08_emergent_geometry/124_emergent_metric_fractal_consistency.py (Kron diagnostics)
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
    """Seed one random graph snapshot for auxiliary-field extraction."""
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def experiment_1_node_is_substrate():
    """M1: verify per-node auxiliary doublet normalization."""
    print("=" * 74)
    print("EXPERIMENT 1: Per-Node Auxiliary Doublet Normalization")
    print("=" * 74)
    print("Package four extracted fields as two complex sectors and normalize")
    print("the associated nonzero Stokes vectors on the Poincare sphere.")
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
    print("  -> the displayed nonzero doublets have unit Poincare vectors.")
    print("     This normalization does not establish independent node DOF.")


def experiment_2_graph_is_blind():
    """M2: separate fixed-topology quantities from state-derived read-outs."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Fixed-Topology and State-Dependent Quantities")
    print("=" * 74)
    print("Fix topology -> the Laplacian spectrum and every R_eff (the ex-124")
    print("'node as graph' content) are FIXED. Vary only the node phase states.")
    print("The former stay fixed by construction; extracted fields can change.")
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
    print("  -> spec[1] and R_eff are identical because the unweighted graph is")
    print("     fixed; polarization and H_sub are functions of the seeded state.")


def experiment_3_same_kind_of_object():
    """M3: verify the implemented additive aggregation identities."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Additivity of Auxiliary Energy and Stokes Densities")
    print("=" * 74)
    print("The network energy is the sum of per-node substrate energies; the")
    print("global Stokes components are sums of per-node Stokes 3-vectors.")
    print("This is an algebraic property of the implemented definitions.")
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
    print("  -> both displayed equalities hold to numerical precision.")
    print("     Additivity alone does not prove U5 nesting or scale self-similarity.")


def main():
    print()
    print("  TNFR Example 125: Per-Node Auxiliary Substrate Read-outs")
    print("  Normalization, dependency split, and additive definitions")
    print("  ===============================================================")
    print()
    experiment_1_node_is_substrate()
    experiment_2_graph_is_blind()
    experiment_3_same_kind_of_object()
    print()
    print("=" * 74)
    print("SCOPED FINDINGS")
    print("=" * 74)
    print("The auxiliary model provides one derived doublet per node. Fixed")
    print("unweighted topology keeps its Laplacian and resistance values fixed")
    print("while state-derived read-outs vary. Global energy and Stokes values")
    print("are sums of per-node densities by definition. These checks do not")
    print("turn the derived tuples into independent node interiors and do not")
    print("prove U5 fractality, a Fix(G) split, or substrate emergence.")


if __name__ == "__main__":
    main()
