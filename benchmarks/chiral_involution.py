"""
benchmarks/chiral_involution.py

Camino 6 -- is the additive inverse of Z the SAME thing as the antiparticle?

emergent_rationals.py (Camino 4) read the additive inverse -n as a consequence
of bipartite coupling symmetry: spec(A) = -spec(A). The emergent-particles layer
(tnfr.physics.emergent_particles) reads sign(W) of the winding number as
structural chirality -- matter (W>0) vs. antimatter (W<0). This harness asks
whether those are two readings of ONE structural operation: the chiral
(sublattice) Z_2 involution of a bipartite TNFR graph.

THE CLAIM (one involution, two representations):
  A bipartite graph 2-colours into sublattices X, Y. Define the chirality
  operator  Gamma = diag(+1 on X, -1 on Y). Then:
    * NUMBER reading  -- Gamma ANTICOMMUTES with the coupling matrix A:
        Gamma A Gamma = -A      (equivalently {Gamma, A} = 0).
      Hence if A v = lambda v then A (Gamma v) = -lambda (Gamma v): every mode n
      has a partner -n, so spec(A) = -spec(A). The additive inverse -n (N -> Z)
      is forced by the coupling, not injected.
    * PARTICLE reading -- the same conjugation acts on the phase field as the
      charge conjugation  C : phi -> -phi, which sends the integer winding
      W = (1/2pi) closed-loop circulation  to  -W. sign(W) flips:
        matter (W>0)  <->  antimatter (W<0).
  Both Gamma (on A) and C (on phi) are involutions (g^2 = id): one abstract Z_2,
  realised on the coupling operator and on the phase field. And both inverses
  share one neutral element / vacuum:
        n + (-n) = 0          (additive identity of Z)
        W + (-W) = 0          (a matter-antimatter pair has zero net topological
                               charge -> annihilates to the scalar |W|=0 vacuum).

CONTRAST WITH CAMINO 5 (the two different Z_2 of a bipartite graph):
  equivariance_wall.py used graph AUTOMORPHISMS -- permutation matrices P that
  COMMUTE with A (P A P^T = +A); the commuting symmetry builds the equivariance
  WALL (a symmetric state cannot enter Fix(G)^perp). The chiral Gamma here is a
  diagonal sign matrix, NOT a permutation: it ANTICOMMUTES (Gamma A Gamma = -A)
  and is not an element of Aut(G). The commuting Z_2 makes the wall; the
  anticommuting Z_2 makes the additive inverse / the antiparticle. Same graph,
  two distinct Z_2 actions.

ENGINE (known theorems -- the independent ground truth, all pre-TNFR):
  - A graph is bipartite iff it is 2-colourable iff there is a diagonal sign
    matrix Gamma with Gamma A Gamma = -A (chiral / sublattice symmetry of
    bipartite tight-binding / SSH Hamiltonians). Then spec(A) = -spec(A).
  - The topological winding number W in Z is ODD under phase conjugation:
    C : phi -> -phi  =>  W -> -W (definitional: circulation reverses sign).
  - Gamma^2 = I and C^2 = id: each generates a Z_2.

TNFR reading (AGENTS.md): L = D - A is the discrete dNFR / phase-curvature
operator; A is the coupling matrix. The additive inverse from bipartite coupling
is emergent_rationals.py piece (1); sign(W) = chirality / matter-vs-antimatter is
the emergent_particles classification. This harness shows they coincide: one
chiral Z_2.

HONEST SCOPE:
  This is a structural ANALOGY made exact at the level of Z_2 group actions on a
  finite graph: the additive inverse and the charge conjugate are the same
  involution in two representations. It does NOT claim TNFR derives the CPT
  theorem, the physical existence of antimatter, or the Standard Model -- those
  carry analytic and field-theoretic content far beyond a sublattice sign flip.
  It also does not make arithmetic primality emergent (that is Camino 1/2). R
  (continuum) and the constants phi, gamma, pi, e remain assumed substrate; this
  is Z, not R. Nothing here touches G4 = RH, Navier-Stokes, or Yang-Mills.

Run:
    python benchmarks/chiral_involution.py

Status: RESEARCH (chiral-involution falsifier; Camino 6 of the unification map).
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from composition_arithmetic import (  # noqa: E402
    adj_spectrum,
    automorphism_matrices,
)
from emergent_rationals import integer_spectrum, is_pm_symmetric  # noqa: E402

from tnfr.physics.emergent_particles import (  # noqa: E402
    classify_particle,
    winding_number,
    winding_ring,
)

TOL = 1e-9
_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# The chiral (sublattice) involution Gamma of a bipartite graph
# --------------------------------------------------------------------------- #
def adjacency(G, nodes):
    """Coupling/adjacency matrix A in the fixed node order `nodes`."""
    return nx.to_numpy_array(G, nodelist=nodes)


def chirality_operator(G, nodes):
    """Gamma = diag(+1 on sublattice X, -1 on sublattice Y) from the bipartite
    2-colouring. Raises nx.NetworkXError if G is not bipartite (then no chiral
    involution exists -- the additive inverse / antiparticle has no clean form)."""
    color = nx.algorithms.bipartite.color(G)
    signs = np.array([1.0 if color[v] == 0 else -1.0 for v in nodes])
    return np.diag(signs)


def commutator_norm(M, P):
    """Frobenius norm ||M P - P M||."""
    return float(np.linalg.norm(M @ P - P @ M))


def anticommutator_norm(M, P):
    """Frobenius norm ||M P + P M||."""
    return float(np.linalg.norm(M @ P + P @ M))


def conjugate_phase_node(phi):
    """Charge conjugation C : phi -> -phi, re-wrapped to [0, 2pi)."""
    y = (-phi) % _TWO_PI
    return float(y)


# --------------------------------------------------------------------------- #
# (1) chiral involution -> additive inverse (Z)
# --------------------------------------------------------------------------- #
def test_chiral_gives_additive_inverse():
    print("=" * 78)
    print("(1) chiral involution Gamma: Gamma A Gamma = -A  =>  spec(A) = -spec(A)")
    print("    the additive inverse -n (N -> Z) is forced by bipartite coupling")
    print("=" * 78)
    bipartite = [
        ("C6", nx.cycle_graph(6)),
        ("K_{3,3}", nx.complete_bipartite_graph(3, 3)),
        ("Q3 (hypercube)", nx.hypercube_graph(3)),
        ("P4", nx.path_graph(4)),
    ]
    non_bipartite = [("C5", nx.cycle_graph(5)), ("K4", nx.complete_graph(4))]

    all_ok = True
    for name, G in bipartite:
        nodes = list(G.nodes())
        A = adjacency(G, nodes)
        Gamma = chirality_operator(G, nodes)
        invol = float(np.linalg.norm(Gamma @ Gamma - np.eye(len(nodes))))
        anti = anticommutator_norm(Gamma, A)                 # {Gamma, A} = 0
        chiral = float(np.linalg.norm(Gamma @ A @ Gamma + A))  # Gamma A Gamma = -A
        spec = adj_spectrum(G)
        sym = is_pm_symmetric(spec)
        ok = invol < TOL and anti < TOL and chiral < TOL and sym
        all_ok &= ok
        print(f"  {name:<16} Gamma^2=I:{invol:.1e}  {{Gamma,A}}=0:{anti:.1e}  "
              f"GammaAGamma+A:{chiral:.1e}  spec+/-sym:{sym}  -> "
              f"{'OK' if ok else 'FAIL'}")

    # non-bipartite: no 2-colouring => no chiral Gamma => spectrum not +/- symmetric
    none_chiral = True
    for name, G in non_bipartite:
        nodes = list(G.nodes())
        try:
            chirality_operator(G, nodes)
            has_gamma = True
        except nx.NetworkXError:
            has_gamma = False
        sym = is_pm_symmetric(adj_spectrum(G))
        none_chiral &= (not has_gamma) and (not sym)
        print(f"  {name:<16} bipartite/chiral Gamma exists? {has_gamma}  "
              f"spec +/- symmetric? {sym}   (non-bipartite contrast)")

    # integral bipartite -> SIGNED INTEGERS Z
    q3 = integer_spectrum(adj_spectrum(nx.hypercube_graph(3)))
    signed = sorted(set(int(v) for v in q3))
    z_ok = (-min(signed) == max(signed))
    print(f"  Q3 gives SIGNED INTEGERS: {signed}  -> N extends to Z   "
          f"({'OK' if z_ok else 'FAIL'})")

    ok = all_ok and none_chiral and z_ok
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the chiral Z_2 builds -n; "
          "without bipartiteness there is no such involution")
    print()
    return ok


# --------------------------------------------------------------------------- #
# (2) the same involution -> antiparticle (sign(W) flip)
# --------------------------------------------------------------------------- #
def test_same_involution_gives_antiparticle():
    print("=" * 78)
    print("(2) the same conjugation C: phi -> -phi flips the winding W -> -W")
    print("    sign(W) = chirality: matter (W>0) <-> antimatter (W<0)")
    print("=" * 78)
    n = 12
    all_ok = True
    for k in (1, 2, 3):
        matter = winding_ring(n, k)
        w_m, _ = winding_number(matter)

        # charge conjugate: negate every phase (the chiral Z_2 on the phase field)
        antimatter = matter.copy()
        for v in antimatter.nodes():
            phi = antimatter.nodes[v]["phase"]
            cphi = conjugate_phase_node(phi)
            antimatter.nodes[v]["phase"] = cphi
            antimatter.nodes[v]["theta"] = cphi
        w_a, _ = winding_number(antimatter)

        cm = classify_particle(matter)
        ca = classify_particle(antimatter)

        # winding_ring(n, -k) is the SAME field (phi -> -phi); cross-check it
        direct = winding_ring(n, -k)
        w_d, _ = winding_number(direct)
        same_field = all(
            abs(conjugate_phase_node(matter.nodes[v]["phase"])
                - direct.nodes[v]["phase"]) < 1e-9
            for v in matter.nodes()
        )

        # C^2 = id: conjugating twice recovers matter
        back = antimatter.copy()
        for v in back.nodes():
            cphi = conjugate_phase_node(back.nodes[v]["phase"])
            back.nodes[v]["phase"] = cphi
            back.nodes[v]["theta"] = cphi
        w_back, _ = winding_number(back)

        ok = (w_m == k and w_a == -k and w_d == -k and same_field
              and w_back == k and cm.chirality == 1 and ca.chirality == -1
              and abs(cm.winding) == abs(ca.winding))
        all_ok &= ok
        print(f"  k={k}: W(matter)={w_m:+d} (chirality {cm.chirality:+d}), "
              f"C: W(antimatter)={w_a:+d} (chirality {ca.chirality:+d}), "
              f"same |W|={abs(cm.winding) == abs(ca.winding)}")
        print(f"        winding_ring(n,-k)=W={w_d:+d} is the conjugate field? "
              f"{same_field};  C^2: W back to {w_back:+d}  -> "
              f"{'OK' if ok else 'FAIL'}")
    print(f"  VERDICT: {'PASS' if all_ok else 'FAIL'} -- charge conjugation is "
          "the chiral Z_2 acting on the phase field")
    print()
    return all_ok


# --------------------------------------------------------------------------- #
# (3) shared vacuum: n + (-n) = 0  <->  W + (-W) = 0  (annihilation)
# --------------------------------------------------------------------------- #
def test_shared_vacuum_annihilation():
    print("=" * 78)
    print("(3) shared neutral element: n + (-n) = 0  <->  W + (-W) = 0")
    print("    additive identity of Z  <->  matter-antimatter pair annihilates")
    print("=" * 78)
    # number side: emergent eigenvalue n and its chiral mirror -n sum to 0
    q3 = integer_spectrum(adj_spectrum(nx.hypercube_graph(3)))
    pos = sorted(set(int(v) for v in q3 if v > 0))
    number_ok = True
    for n in pos:
        mirror_present = (-n) in set(int(v) for v in q3)
        sums_to_zero = (n + (-n)) == 0
        number_ok &= mirror_present and sums_to_zero
        print(f"  number: mode n={n:+d} has chiral partner -n={-n:+d} present? "
              f"{mirror_present};  n+(-n)={n + (-n)}  (additive identity)")

    # particle side: matter W=+k and antimatter W=-k sum to net 0 = scalar vacuum
    nring = 12
    particle_ok = True
    for k in (1, 2, 3):
        w_m, _ = winding_number(winding_ring(nring, k))
        w_a, _ = winding_number(winding_ring(nring, -k))
        net = w_m + w_a
        # net winding 0 == the scalar / boson-like |W|=0 vacuum class
        vac = classify_particle(winding_ring(nring, 0))
        annihilates = (net == 0) and (vac.winding == 0)
        particle_ok &= annihilates
        print(f"  particle: W(matter)={w_m:+d} + W(antimatter)={w_a:+d} = "
              f"net {net}  -> scalar |W|=0 vacuum? {vac.winding == 0}  "
              f"({'annihilates' if annihilates else 'FAIL'})")

    ok = number_ok and particle_ok
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the additive zero of Z and "
          "the |W|=0 vacuum are the same neutral element of the chiral Z_2")
    print()
    return ok


# --------------------------------------------------------------------------- #
# (4) it is ONE Z_2 -- and it ANTICOMMUTES (the contrast with Camino 5)
# --------------------------------------------------------------------------- #
def test_one_z2_anticommuting_contrast():
    print("=" * 78)
    print("(4) ONE chiral Z_2 vs. Camino 5: automorphisms COMMUTE, Gamma ANTICOMMUTES")
    print("=" * 78)
    G = nx.cycle_graph(6)                      # bipartite, Aut = D_6
    nodes = list(G.nodes())
    A = adjacency(G, nodes)
    eye = np.eye(len(nodes))

    # Camino-5 Z_2: a graph automorphism (permutation) commutes with A
    mats = automorphism_matrices(G, nodes)
    P = next(M for M in mats if np.linalg.norm(M - eye) > 1e-9)   # any non-identity
    p_is_perm = bool(np.allclose(P.sum(axis=0), 1) and np.allclose(P.sum(axis=1), 1)
                     and np.allclose(P, P.astype(bool)))
    p_comm = commutator_norm(A, P)             # [A, P] = 0  (P A P^T = +A)
    p_invol = float(np.linalg.norm(P @ P - eye))

    # Camino-6 Z_2: the chiral diagonal Gamma anticommutes with A
    Gamma = chirality_operator(G, nodes)
    g_is_perm = bool(np.allclose(Gamma.sum(axis=0), 1)
                     and np.allclose(np.abs(Gamma).sum(axis=1), 1)
                     and np.all(Gamma >= 0))
    g_comm = commutator_norm(A, Gamma)         # [A, Gamma] != 0
    g_anti = anticommutator_norm(A, Gamma)     # {A, Gamma} = 0
    g_invol = float(np.linalg.norm(Gamma @ Gamma - eye))

    print(f"  automorphism P (Camino 5): permutation? {p_is_perm}  "
          f"P^2=I:{p_invol:.1e}  [A,P]={p_comm:.2e} (COMMUTES, P A P^T=+A)")
    print(f"  chiral Gamma  (Camino 6): permutation? {g_is_perm}  "
          f"Gamma^2=I:{g_invol:.1e}  [A,Gamma]={g_comm:.2e} (NOT 0)  "
          f"{{A,Gamma}}={g_anti:.2e} (ANTICOMMUTES, Gamma A Gamma=-A)")
    print("  => both are involutions (Z_2), but the COMMUTING one builds the")
    print("     equivariance wall (Camino 5) while the ANTICOMMUTING one builds")
    print("     the additive inverse / the antiparticle (Camino 6). Two distinct")
    print("     Z_2 actions on the same bipartite graph.")

    ok = (p_is_perm and p_invol < TOL and p_comm < TOL
          and (not g_is_perm) and g_invol < TOL and g_comm > 1e-3 and g_anti < TOL)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- one abstract Z_2, two reps; "
          "anticommuting (not the Camino-5 commuting wall)")
    print()
    return ok


def main():
    print(__doc__)
    results = [
        ("(1) chiral involution -> additive inverse (Z)",
         test_chiral_gives_additive_inverse()),
        ("(2) same involution -> antiparticle (sign(W) flip)",
         test_same_involution_gives_antiparticle()),
        ("(3) shared vacuum: n+(-n)=0 <-> W+(-W)=0",
         test_shared_vacuum_annihilation()),
        ("(4) one Z_2, anticommuting (contrast with Camino 5)",
         test_one_z2_anticommuting_contrast()),
    ]
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {name:<52}: {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")
    print()
    print("  Reading: the additive inverse -n of Z and the antiparticle are ONE")
    print("  structural operation -- the chiral (sublattice) Z_2 of a bipartite")
    print("  TNFR graph. Gamma = diag(+/-1) anticommutes with the coupling A")
    print("  (Gamma A Gamma = -A), forcing spec(A) = -spec(A): that is -n. The")
    print("  same conjugation on the phase field, C: phi -> -phi, flips the")
    print("  winding W -> -W: that is matter <-> antimatter. Their shared neutral")
    print("  element is the same: n+(-n)=0 in Z is the |W|=0 scalar vacuum where a")
    print("  matter-antimatter pair annihilates. CONTRAST: this Z_2 ANTICOMMUTES")
    print("  with A, unlike the COMMUTING automorphism Z_2 that builds the Camino-5")
    print("  equivariance wall -- two different Z_2 on the same graph. HONEST SCOPE:")
    print("  exact finite Z_2 group actions on a graph; a precise structural")
    print("  analogy, NOT a derivation of CPT, antimatter, or the Standard Model.")
    print("  R and phi,gamma,pi,e remain assumed substrate; this is Z, not R;")
    print("  nothing here touches G4 = RH, Navier-Stokes, or Yang-Mills.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
