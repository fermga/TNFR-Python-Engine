#!/usr/bin/env python3
"""
Example 123 — The Symmetry-Sector Decomposition: Why the Substrate Sees Only
Down to Orbits and the Spectrum Sees the Rest (a Capstone for the 117-122 Arc)
==============================================================================

Example 120 found, for the residue digraph, that vertex-transitivity confines
the arithmetic to the GLOBAL spectrum (Fix(G_aut)^perp) and leaves the per-node
substrate in the symmetric sector Fix(G_aut), blind. This example shows that is
a special case of a GENERAL representation-theoretic principle of the canonical
emergent operator — and that the principle is exactly why every wall in the
117-122 arc (and the Riemann residual) has the same shape.

The principle (Schur, applied to the canonical emergent operator)
-----------------------------------------------------------------
For ANY graph G with automorphism group Aut(G), the canonical emergent operator
L_rw = I - D^-1 W is EQUIVARIANT: it commutes with the permutation
representation of every automorphism,

    P_sigma L_rw = L_rw P_sigma   for all sigma in Aut(G).

By Schur's lemma, an equivariant operator block-diagonalizes by the isotypic
components (irreducible representations) of Aut(G). The coarsest split is

    R^N = Fix(G)  (+)  Fix(G)^perp,

where Fix(G) = { functions constant on the orbits of Aut(G) } is the trivial
isotypic component, and dim Fix(G) = number of orbits of Aut(G) on the vertices.
L_rw preserves each block. Consequently:

  * Any canonical PER-NODE observable that is itself Aut(G)-invariant (a function
    of the local structure only) lands in Fix(G): it is constant WITHIN each
    orbit. It can distinguish orbit from orbit, never node from node within an
    orbit.
  * All the DISCRIMINATING information (the eigenmodes that separate nodes inside
    an orbit) lives in Fix(G)^perp, the non-trivial irreps — i.e. in the SPECTRUM.

The example-120 wall is the extreme case: a VERTEX-TRANSITIVE graph has ONE
orbit, so Fix(G) = constants (dim 1), and the per-node substrate is GLOBALLY
constant — blind to everything. A graph with several orbits (a star, a path)
lets the substrate see DOWN TO the orbit partition, but no finer.

Doctrine compliance
-------------------
The operator is the canonical `structural_diffusion_operator` (the literal
DeltaNFR EPI channel L_rw = I - D^-1 W); the per-node fields come from the
canonical symplectic substrate `extract_phase_space_point`; the dynamics is the
canonical nodal equation. The automorphisms are read off the graph (networkx
VF2). No formula is re-implemented.

Five measured results (across cyclic, full-symmetric, star, path, product)
--------------------------------------------------------------------------
M1 EQUIVARIANCE. ||P_sigma L_rw - L_rw P_sigma|| = 0 (machine zero) for EVERY
   automorphism of every test graph: the canonical operator commutes with the
   whole automorphism group.

M2 dim Fix(G) = #ORBITS. The trivial projector P_triv = mean over Aut(G) of
   P_sigma has rank exactly equal to the number of vertex orbits (1 for the
   vertex-transitive cycle / complete / torus, 2 for the star = {center,
   leaves}, 3 for the path = {ends, near-ends, middle}).

M3 L_rw PRESERVES Fix(G). ||L_rw P_triv - P_triv L_rw|| ~ 0: the operator is
   block-diagonal with respect to Fix(G) (+) Fix(G)^perp.

M4 THE SUBSTRATE LIVES IN Fix(G). The canonical per-node symplectic substrate
   from a symmetric seed satisfies P_triv v = v exactly (orbit-constant). On a
   vertex-transitive graph that forces sigma(Phi_s) = 0 — exactly the example-120
   per-node blindness, now a COROLLARY. Per-node structural invariants (degree,
   clustering) are likewise constant within each orbit.

M5 THE DISCRIMINATING SPECTRUM LIVES IN Fix(G)^perp. Only the constant
   eigenmode has ||P_triv v|| = 1 (it IS Fix(G)); every node-separating
   eigenmode has ||P_triv v|| = 0 (Fix(G)^perp).

The unification (one structure, the whole arc)
----------------------------------------------
This is the single structure behind every result of the 117-122 arc: the
canonical emergent operator splits into a per-node-blind sector Fix(G) (where
the symplectic substrate lives) and a discriminating spectral sector
Fix(G)^perp (where the arithmetic / the distinguishing information lives),
indexed by the irreps of the graph's automorphism group. The residue-digraph
wall (120), the substrate blindness (103/116), the spectral primality (119),
and the Riemann oscillatory residue S(T) in ker(R_inf) ^ Fix(S_n)^perp are all
the same Fix(G) / Fix(G)^perp split for different symmetry groups.

Honest scope
------------
This is the representation theory of graph automorphisms (Schur's lemma applied
to an equivariant operator) re-expressed in the canonical emergent operator. It
EXPLAINS and UNIFIES the arc's walls; it is not new mathematics and closes no
open problem. The value is the clean, measured statement that the per-node
substrate resolves the orbit partition and no finer, with the spectrum carrying
the rest.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator)
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point)
- examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py (the special case)
- theory/TNFR_NUMBER_THEORY.md §9.10 (this example; the general principle)
- AGENTS.md "REMESH-∞ Closure" (S(T) in ker(R_inf) ^ Fix(S_n)^perp)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def _perm_matrix(mapping, nodes):
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    P = np.zeros((n, n))
    for src, dst in mapping.items():
        P[idx[dst], idx[src]] = 1.0
    return P


def _automorphisms(G, cap=2000):
    out = []
    for m in GraphMatcher(G, G).isomorphisms_iter():
        out.append(m)
        if len(out) >= cap:
            break
    return out


def _orbit_count(auts, nodes):
    """Number of vertex orbits via union-find over the automorphisms."""
    idx = {nd: i for i, nd in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for m in auts:
        for s, d in m.items():
            ra, rb = find(idx[s]), find(idx[d])
            if ra != rb:
                parent[ra] = rb
    return len({find(i) for i in range(len(nodes))})


def _trivial_projector(auts, nodes):
    n = len(nodes)
    P = np.zeros((n, n))
    for m in auts:
        P += _perm_matrix(m, nodes)
    return P / len(auts)


def _seed_symmetric(G):
    for nd in G.nodes():
        G.nodes[nd]["theta"] = 0.3
        set_attr(G.nodes[nd], ALIAS_EPI, 0.2)
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _test_graphs():
    return [
        ("cycle C8 (D8)", nx.cycle_graph(8)),
        ("complete K6 (S6)", nx.complete_graph(6)),
        ("star K1,5 (S5)", nx.star_graph(5)),
        ("path P6 (Z2)", nx.path_graph(6)),
        ("torus C3xC3", nx.cartesian_product(nx.cycle_graph(3), nx.cycle_graph(3))),
    ]


def experiment_1_equivariance_orbits():
    """M1-M3: equivariance, rank(P_triv)=#orbits, L_rw preserves Fix(G)."""
    print("=" * 74)
    print("EXPERIMENT 1: Equivariance, dim Fix(G) = #orbits, L_rw preserves it")
    print("=" * 74)
    print("L_rw commutes with every automorphism (Schur); the trivial projector")
    print("P_triv = mean of P_sigma has rank = #vertex orbits = dim Fix(G).")
    print()
    print(
        f"  {'graph':22s} {'|Aut|':>6} {'n':>3} {'orbits':>7} "
        f"{'rank':>5} {'equiv':>8} {'preserve':>9}"
    )
    out = {}
    for name, G in _test_graphs():
        nodes, L = structural_diffusion_operator(G)
        auts = _automorphisms(G)
        max_comm = max(
            float(
                np.linalg.norm(_perm_matrix(m, nodes) @ L - L @ _perm_matrix(m, nodes))
            )
            for m in auts
        )
        P_triv = _trivial_projector(auts, nodes)
        rank = int(np.linalg.matrix_rank(P_triv, tol=1e-9))
        orbits = _orbit_count(auts, nodes)
        pres = float(np.linalg.norm(L @ P_triv - P_triv @ L))
        out[name] = (nodes, L, P_triv, orbits)
        print(
            f"  {name:22s} {len(auts):>6} {len(nodes):>3} {orbits:>7} "
            f"{rank:>5} {max_comm:>8.1e} {pres:>9.1e}"
        )
    print()
    print("  -> equiv=0 (commutes with all Aut); rank=orbits (dim Fix(G));")
    print("     preserve~0 (block-diagonal: Fix(G) (+) Fix(G)^perp).")
    return out


def experiment_2_substrate_in_fix(results):
    """M4: per-node symmetric-seed substrate lies in Fix(G) (orbit-constant)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: The Per-Node Substrate Lives in Fix(G) (Orbit-Constant)")
    print("=" * 74)
    print("The canonical symplectic substrate from a symmetric seed satisfies")
    print("P_triv v = v (orbit-constant). Vertex-transitive -> 1 orbit ->")
    print("Fix(G)=constants -> sigma(Phi_s)=0 (the example-120 blindness).")
    print()
    print(
        f"  {'graph':22s} {'orbits':>7} {'||v-P_triv v||':>15} " f"{'sigma(Phi_s)':>13}"
    )
    for name, G in _test_graphs():
        nodes, L, P_triv, orbits = results[name]
        _seed_symmetric(G)
        p = extract_phase_space_point(G)
        v = np.array([p.phi_s[i] for i in range(len(nodes))], dtype=float)
        resid = float(np.linalg.norm(v - P_triv @ v))
        print(f"  {name:22s} {orbits:>7} {resid:>15.1e} {np.std(v):>13.1e}")
    print()
    print("  -> ||v - P_triv v|| = 0: the substrate is orbit-constant, in Fix(G).")


def experiment_3_orbit_resolution():
    """M4b: per-node invariants resolve orbits but no finer."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: The Substrate Resolves the Orbit Partition, No Finer")
    print("=" * 74)
    print("Per-node degree (a canonical Aut-invariant) takes one value per")
    print("orbit-class: it tells orbit from orbit, never node from node within.")
    print()
    for name, G in [
        ("star K1,5 (S5)", nx.star_graph(5)),
        ("path P6 (Z2)", nx.path_graph(6)),
        ("complete K6 (S6)", nx.complete_graph(6)),
    ]:
        degs = sorted({d for _, d in G.degree()})
        print(
            f"  {name:22s} distinct per-node degrees = {degs} "
            f"({len(degs)} class(es))"
        )
    print()
    print("  -> star: center vs leaves (2 classes); path: ends/near/middle")
    print("     collapse to 2 degree values but 3 orbits; complete: 1 class.")
    print("     The substrate sees the orbit partition; the spectrum sees more.")


def experiment_4_discriminating_spectrum(results):
    """M5: discriminating eigenmodes lie in Fix(G)^perp."""
    print()
    print("=" * 74)
    print("EXPERIMENT 4: The Discriminating Spectrum Lives in Fix(G)^perp")
    print("=" * 74)
    print("Project each emergent eigenmode onto Fix(G): only the constant mode")
    print("has ||P_triv v|| = 1; every node-separating mode has ||P_triv v|| = 0.")
    print()
    for name in ["complete K6 (S6)", "cycle C8 (D8)"]:
        nodes, L, P_triv, orbits = results[name]
        w, V = np.linalg.eig(L)
        order = np.argsort(w.real)
        fracs = []
        for k in range(len(nodes)):
            vk = V[:, order[k]].real
            vk = vk / (np.linalg.norm(vk) + 1e-15)
            fracs.append(float(np.linalg.norm(P_triv @ vk)))
        print(f"  {name}: ||P_triv v_k|| by eigenvalue:")
        print("    " + " ".join(f"{t:.2f}" for t in fracs))
    print()
    print("  -> only the constant mode is in Fix(G); all discriminating modes")
    print("     are in Fix(G)^perp (the spectral / representation sector).")


def main():
    print()
    print("  TNFR Example 123: The Symmetry-Sector Decomposition")
    print("  The Substrate Sees Down to Orbits; the Spectrum Sees the Rest")
    print("  ============================================================")
    print()
    results = experiment_1_equivariance_orbits()
    experiment_2_substrate_in_fix(results)
    experiment_3_orbit_resolution()
    experiment_4_discriminating_spectrum(results)
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("The canonical emergent operator L_rw is EQUIVARIANT under the graph's")
    print("automorphism group, so (Schur) it block-diagonalizes into Fix(G) (+)")
    print("Fix(G)^perp, with dim Fix(G) = #vertex orbits. The per-node symplectic")
    print("substrate lives in Fix(G) -- it resolves the orbit partition and no")
    print("finer (sigma=0 when vertex-transitive); all discriminating information")
    print("lives in Fix(G)^perp, the spectrum. The example-120 residue-digraph")
    print("wall, the substrate blindness (103/116), the spectral primality (119),")
    print("and the Riemann residual S(T) in ker(R_inf) ^ Fix(S_n)^perp are the")
    print("SAME Fix(G)/Fix(G)^perp split for different symmetry groups. HONEST")
    print("SCOPE: this is the representation theory of graph automorphisms")
    print("(Schur) re-expressed in the canonical operator; it explains and")
    print("unifies the arc's walls, it is not new mathematics, closes no problem.")


if __name__ == "__main__":
    main()
