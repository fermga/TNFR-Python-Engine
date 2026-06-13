#!/usr/bin/env python3
"""
Example 107 — The Orthogonal Structure of the Emergent Geometry
==============================================================

Closes the emergent-geometry arc by measuring how its pieces fit together
ORTHOGONALLY. Two exact, cache-free decompositions:

  (A) Helmholtz–Hodge decomposition of the nodal flow: the two emergent
      towers — the dissipative TRANSPORT tower (diffusion, Example 99) and
      the conservative SYMPLECTIC tower (Hamiltonian, Example 98) — are the
      two ORTHOGONAL Hodge components of an edge flow on the graph. The
      diffusion current is the gradient (irrotational) part; a circulation
      is the cycle (solenoidal) part; they are orthogonal.

  (C) Winding–polarization decoupling: the integer topological charge (the
      winding number, Example "emergent particles") and the continuous
      polarization vector (Example 106) live in DIFFERENT tetrad channels —
      the winding in the phase gradient |∇φ|, the polarization in the
      curvature/current K_φ, J_φ — and are structurally decoupled.

Both are anchored to empirically-demonstrated classical structure
(Helmholtz 1858 / Hodge theory; optical vortices and their gradient phase
circulation) and are TNFR-native (the tetrad's own components).

Physics
-------
(A) On a graph, an edge flow decomposes (discrete Helmholtz–Hodge theorem)
into orthogonal subspaces: the GRADIENT/cut space (irrotational, curl-free)
and the CYCLE space (solenoidal, divergence-free). The structural diffusion
current J_ij = EPI_i − EPI_j = grad(EPI) (Example 99) is a PURE gradient, so
it lives entirely in the gradient subspace (circulation around every cycle
= 0, telescoping). A circulation (cycle flow) is divergence-free (the
discrete Liouville statement). The two are orthogonal. So the nodal flow's
dissipative part (transport) and conservative part (symplectic rotation)
are the two orthogonal Helmholtz–Hodge components — one object unifying the
session's two towers.

(C) A uniform winding φ_i = 2π·W·i/n has a CONSTANT phase gradient
|∇φ| = 2π·W/n (linear in the winding W) and therefore ZERO phase curvature
and current (K_φ = J_φ = 0: a constant gradient has no second-order
structure). So the topological charge lives entirely in the gradient
channel and leaves the polarization sector ζ^A = K_φ + i·J_φ empty. The
tetrad's gradient (1st order) and curvature (2nd order) channels are
genuinely independent (cf. the minimal-degrees-of-freedom argument), so a
pure vortex and the polarization vector are decoupled.

Honest scope
------------
- Both decompositions are EXACT (machine precision) and cache-free (A is
  pure graph linear algebra; C uses analytic winding rings).
- (A) restates and UNIFIES known facts (diffusion = gradient current,
  Example 99; symplectic flow divergence-free = Liouville, Example 98) as
  the two Hodge components — it is an organizing identity, not a new
  theorem.
- (C) is an honest DECOUPLING (a clean negative on the naive "spin–orbit
  coupling" intuition): a uniform vortex carries its charge in |∇φ| and
  leaves the polarization vector at the pole. This is faithful to the
  tetrad's channel independence, not a new physical coupling.
- A third probe (does the nodal flow TRANSPORT the polarization texture?)
  was measured and found near-trivial/inconclusive — the geometric
  polarization sector |ζ^A| collapses under phase synchronization (because
  K_φ → 0 when phases align, which is almost definitional), with no clean
  single-pole rotation because both polarization sectors co-vary through
  the multi-channel ΔNFR. It is NOT canonized here.

References
----------
- examples/98_emergent_symplectic_substrate.py (symplectic tower, Liouville)
- examples/99_structural_diffusion.py (diffusion current = grad EPI)
- examples/106_per_node_polarization_geometry.py (polarization vector)
- src/tnfr/physics/structural_diffusion.py (structural_current)
- src/tnfr/physics/emergent_particles.py (winding_ring, winding_number)
- src/tnfr/physics/canonical.py (compute_phase_gradient/curvature)
- AGENTS.md §"Emergent Symplectic Substrate", §"Transport Content"
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.physics.emergent_particles import winding_ring, winding_number
from tnfr.physics.canonical import (
    compute_phase_gradient,
    compute_phase_curvature,
)
from tnfr.physics.extended import compute_phase_current


# ============================================================================
# EXPERIMENT 1 (A): Helmholtz–Hodge decomposition of the nodal flow
# ============================================================================
def experiment_1_hodge():
    """The two towers are the two orthogonal Hodge components of the flow."""
    print("=" * 72)
    print("EXPERIMENT 1: Helmholtz–Hodge Decomposition of the Nodal Flow")
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
    print(f"    ||J_diff|| = {np.linalg.norm(j_diff):.4f}, "
          f"||cycle part|| = {np.linalg.norm(cycle_part):.1e}")
    print("    → J_diff is IRROTATIONAL (curl-free): the gradient/cut")
    print("      subspace. Circulation around every cycle = 0 (telescoping).")

    cycles = nx.minimum_cycle_basis(G)
    max_circ = 0.0
    for cyc in cycles:
        k = len(cyc)
        circ = sum(epi[idx[cyc[(i + 1) % k]]] - epi[idx[cyc[i]]]
                   for i in range(k) if G.has_edge(cyc[i], cyc[(i + 1) % k]))
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
    print("  SYMPLECTIC tower:  a circulation (cycle flow)")
    print(f"    ||divergence|| = {np.linalg.norm(div):.1e}  → SOLENOIDAL")
    print("      (divergence-free = the discrete Liouville statement).")
    print()

    ortho = float(j_diff @ rot)
    print(f"  HODGE ORTHOGONALITY:  ⟨J_diff, circulation⟩ = {ortho:.1e}")
    print()
    print("VERDICT: the dissipative TRANSPORT tower (gradient/irrotational)")
    print("and the conservative SYMPLECTIC tower (cycle/solenoidal) are the")
    print("two ORTHOGONAL Helmholtz–Hodge components of the nodal flow — one")
    print("object unifying the session's two towers (Helmholtz 1858 / Hodge).")
    print()


# ============================================================================
# EXPERIMENT 2 (C): winding–polarization decoupling
# ============================================================================
def experiment_2_winding_decoupling():
    """The topological charge and the polarization live in different channels."""
    print("=" * 72)
    print("EXPERIMENT 2: Winding–Polarization Decoupling")
    print("=" * 72)
    print()
    print("A uniform winding φ_i = 2π·W·i/n has a CONSTANT gradient")
    print("|∇φ| = 2π·W/n (linear in W) and therefore zero curvature/current")
    print("(K_φ = J_φ = 0). Where does the topological charge live?")
    print()

    print(f"  {'W':>3} {'n':>4} {'measW':>6} {'mean|∇φ|':>9} "
          f"{'2πW/n':>7} {'|K_φ|':>7} {'|J_φ|':>7}")
    rows = []
    for w in (1, 2, 3, 4, 5):
        nn = 60 + w   # vary n per W → fresh topology → no tetrad-cache collision
        G = winding_ring(nn, w)
        wm, _ = winding_number(G)
        grad = compute_phase_gradient(G)
        kphi = compute_phase_curvature(G)
        jphi = compute_phase_current(G)
        mg = float(np.mean(list(grad.values())))
        mk = float(np.mean(np.abs(list(kphi.values()))))
        mj = float(np.mean(np.abs(list(jphi.values()))))
        rows.append((w, mg))
        print(f"  {w:>3} {nn:>4} {wm:>6} {mg:>9.4f} {2 * math.pi * w / nn:>7.4f} "
              f"{mk:>7.4f} {mj:>7.4f}")
    ws = np.array([r[0] for r in rows], float)
    gs = np.array([r[1] for r in rows], float)
    r = float(np.corrcoef(ws, gs)[0, 1])
    print()
    print(f"  r(W, mean|∇φ|) = {r:.4f}  → the winding lives in the GRADIENT.")
    print("  K_φ and J_φ (the polarization sector ζ^A = K_φ + i·J_φ) vanish.")
    print()
    print("VERDICT: the topological winding number lives in the phase")
    print("gradient |∇φ| (1st-order channel); the polarization vector lives")
    print("in K_φ, J_φ (2nd-order channel). A pure vortex carries its charge")
    print("in the gradient and leaves the polarization at the pole — the two")
    print("are STRUCTURALLY DECOUPLED (different, independent tetrad channels).")
    print("The naive optical 'spin–orbit coupling' is NOT automatic in TNFR.")
    print()


def main():
    print()
    print("  TNFR Example 107: The Orthogonal Structure of the Emergent Geometry")
    print("  Helmholtz–Hodge of the flow + winding–polarization decoupling")
    print("  ===================================================================")
    print()
    experiment_1_hodge()
    experiment_2_winding_decoupling()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The emergent geometry has a clean ORTHOGONAL structure, measured")
    print("exactly. (1) The nodal flow's two towers — dissipative transport")
    print("(diffusion) and conservative symplectic (Hamiltonian rotation) —")
    print("are the two orthogonal Helmholtz–Hodge components of an edge flow:")
    print("the diffusion current is the gradient (irrotational) part, a")
    print("circulation is the cycle (solenoidal) part, orthogonal to machine")
    print("precision. (2) The topological winding number and the polarization")
    print("vector occupy DIFFERENT tetrad channels (gradient |∇φ| vs")
    print("curvature/current K_φ, J_φ) and are structurally decoupled. Both")
    print("are exact, cache-free, TNFR-native, and anchored to classical")
    print("structure (Helmholtz–Hodge; optical vortices). This closes the")
    print("emergent-geometry arc: the flat tower is complete and its pieces")
    print("fit together orthogonally — characterization, not new physics.")
    print()


if __name__ == "__main__":
    main()
