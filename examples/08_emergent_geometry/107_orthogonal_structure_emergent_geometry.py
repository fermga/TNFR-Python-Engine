#!/usr/bin/env python3
"""
Example 107 — Graph Hodge Orthogonality and a Winding-Ring Family
================================================================

The first experiment verifies one standard incidence-matrix fact: an EPI
edge gradient is orthogonal to a chosen cycle flow.  The cycle flow is created
for the calculation; it is not extracted from the auxiliary symplectic
substrate.  Consequently this does not decompose the engine's nodal dynamics
into dissipative and Hamiltonian parts, and ``B.T @ cycle = 0`` is a graph
divergence identity rather than Liouville's phase-volume theorem.

The second experiment examines a specific analytic family of winding rings.
For these uniformly spaced phases, the implemented phase-gradient magnitude
tracks winding while the extracted curvature and phase current vanish.  This
shows that this family lies in the kernel of those two read-outs.  It does not
prove that tetrad summaries are independent coordinates or that winding and
polarization decouple on arbitrary graphs and phase fields.

Both results are finite checks of classical graph/circular-field identities.
They neither establish completeness of the tetrad nor connect the auxiliary
harmonic substrate flow to an engine trajectory.

References
----------
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py (symplectic tower, Liouville)
- examples/08_emergent_geometry/99_structural_diffusion.py (diffusion current = grad EPI)
- examples/08_emergent_geometry/106_per_node_polarization_geometry.py (polarization vector)
- src/tnfr/physics/structural_diffusion.py (structural_current)
- src/tnfr/physics/emergent_particles.py (winding_ring, winding_number)
- src/tnfr/physics/canonical.py (compute_phase_gradient/curvature)
- AGENTS.md §"Emergent Symplectic Substrate", §"Transport Content"
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.physics.canonical import compute_phase_curvature, compute_phase_gradient
from tnfr.physics.emergent_particles import winding_number, winding_ring
from tnfr.physics.extended import compute_phase_current


# ============================================================================
# EXPERIMENT 1: incidence-matrix gradient/cycle orthogonality
# ============================================================================
def experiment_1_hodge():
    """Compare one constructed incidence gradient with one cycle vector."""
    print("=" * 72)
    print("EXPERIMENT 1: Gradient/Cycle Orthogonality on One Graph")
    print("=" * 72)
    print()
    print("On a graph an edge flow splits (discrete Helmholtz–Hodge) into")
    print("orthogonal subspaces: GRADIENT/cut (irrotational) ⊕ CYCLE")
    print("(solenoidal). The diffusion current J = grad(EPI) is pure")
    print("gradient; a circulation is pure cycle; they are orthogonal.")
    print()

    import random

    rng = random.Random(3)
    G = nx.watts_strogatz_graph(30, 4, 0.35, seed=3)
    for nd in G.nodes():
        G.nodes[nd]["EPI"] = rng.uniform(-0.5, 0.5)
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(G.edges())
    m, n = len(edges), len(nodes)
    epi = np.array([G.nodes[nd]["EPI"] for nd in nodes])

    # oriented incidence B (m x n): edge (u,v) -> -1@u, +1@v
    B = np.zeros((m, n))
    for e, (u, v) in enumerate(edges):
        B[e, idx[u]] = -1.0
        B[e, idx[v]] = +1.0

    print(f"  graph: {n} nodes, {m} edges, {m - n + 1} independent cycles")
    print()

    # the diffusion current is a pure gradient
    j_diff = B @ epi
    bpinv = np.linalg.pinv(B)
    grad_part = B @ (bpinv @ j_diff)
    cycle_part = j_diff - grad_part
    print("  TRANSPORT tower:  J_diff = grad(EPI) = B·EPI")
    print(
        f"    ||J_diff|| = {np.linalg.norm(j_diff):.4f}, "
        f"||cycle part|| = {np.linalg.norm(cycle_part):.1e}"
    )
    print("    → J_diff is IRROTATIONAL (curl-free): the gradient/cut")
    print("      subspace. Circulation around every cycle = 0 (telescoping).")

    # ``cycle_basis`` returns each simple cycle in cyclic node order, which is
    # required by the explicit circulation construction below.
    cycles = nx.cycle_basis(G)
    max_circ = 0.0
    for cyc in cycles:
        k = len(cyc)
        circ = sum(
            epi[idx[cyc[(i + 1) % k]]] - epi[idx[cyc[i]]]
            for i in range(k)
            if G.has_edge(cyc[i], cyc[(i + 1) % k])
        )
        max_circ = max(max_circ, abs(circ))
    print(f"    max circulation of J_diff over cycles = {max_circ:.1e}")
    print()

    # a circulation is divergence-free
    edge_idx = {frozenset(e): i for i, e in enumerate(edges)}
    rot = np.zeros(m)
    cyc = cycles[0]
    for i in range(len(cyc)):
        a, b = cyc[i], cyc[(i + 1) % len(cyc)]
        key = frozenset((a, b))
        if key in edge_idx:
            e = edge_idx[key]
            rot[e] = 1.0 if edges[e] == (a, b) else -1.0
    div = B.T @ rot
    print("  CYCLE subspace:  a constructed circulation")
    print(f"    ||divergence|| = {np.linalg.norm(div):.1e}  → SOLENOIDAL")
    print("      (a graph-incidence identity, distinct from Liouville's theorem).")
    print()

    ortho = float(j_diff @ rot)
    print(f"  HODGE ORTHOGONALITY:  ⟨J_diff, circulation⟩ = {ortho:.1e}")
    print()
    print("VERDICT: this EPI gradient is orthogonal to the constructed cycle")
    print("flow, as required by incidence-matrix Hodge theory. No engine")
    print("Hamiltonian component or substrate flow is identified here.")
    print()


# ============================================================================
# EXPERIMENT 2 (C): winding–polarization decoupling
# ============================================================================
def experiment_2_winding_decoupling():
    """The topological charge and the polarization live in different channels."""
    print("=" * 72)
    print("EXPERIMENT 2: Uniform Winding-Ring Read-outs")
    print("=" * 72)
    print()
    print("A uniform winding φ_i = 2π·W·i/n has a CONSTANT gradient")
    print("|∇φ| = 2π·W/n (linear in W) and therefore zero curvature/current")
    print("(K_φ = J_φ = 0). Where does the topological charge live?")
    print()

    print(
        f"  {'W':>3} {'n':>4} {'measW':>6} {'mean|∇φ|':>9} "
        f"{'2πW/n':>7} {'|K_φ|':>7} {'|J_φ|':>7}"
    )
    rows = []
    for w in (1, 2, 3, 4, 5):
        nn = 60 + w  # vary n per W → fresh topology → no tetrad-cache collision
        G = winding_ring(nn, w)
        wm, _ = winding_number(G)
        grad = compute_phase_gradient(G)
        kphi = compute_phase_curvature(G)
        jphi = compute_phase_current(G)
        mg = float(np.mean(list(grad.values())))
        mk = float(np.mean(np.abs(list(kphi.values()))))
        mj = float(np.mean(np.abs(list(jphi.values()))))
        rows.append((w, mg))
        print(
            f"  {w:>3} {nn:>4} {wm:>6} {mg:>9.4f} {2 * math.pi * w / nn:>7.4f} "
            f"{mk:>7.4f} {mj:>7.4f}"
        )
    ws = np.array([r[0] for r in rows], float)
    gs = np.array([r[1] for r in rows], float)
    r = float(np.corrcoef(ws, gs)[0, 1])
    print()
    print(f"  r(W, mean|∇φ|) = {r:.4f}  → winding is visible in this read-out.")
    print("  K_φ and J_φ (the polarization sector ζ^A = K_φ + i·J_φ) vanish.")
    print()
    print("VERDICT: in this uniform winding-ring family, winding is visible in")
    print("|∇φ| while K_φ and J_φ vanish. This kernel example does not prove")
    print("global independence or decoupling of the diagnostic fields.")
    print()


def main():
    print()
    print("  TNFR Example 107: Graph Hodge Orthogonality and Winding Read-outs")
    print("  Incidence-matrix identity + a finite uniform-ring family")
    print("  ===================================================================")
    print()
    experiment_1_hodge()
    experiment_2_winding_decoupling()
    print("=" * 72)
    print("SCOPED FINDINGS")
    print("=" * 72)
    print()
    print("The sampled EPI gradient and constructed cycle circulation are")
    print("orthogonal to numerical precision. Uniform winding rings place their")
    print("winding signal in |grad phi| while the implemented K_phi and J_phi")
    print("read-outs vanish. These facts have the finite and algebraic scopes")
    print("stated above; they do not establish a nodal-flow decomposition,")
    print("tetrad completeness, or a general substrate decoupling theorem.")
    print()


if __name__ == "__main__":
    main()
