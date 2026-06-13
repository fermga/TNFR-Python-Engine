#!/usr/bin/env python3
"""
Example 111 — Hodge (TNFR): Discrete Hodge Theory and the Honest Gap
===================================================================

The first milestone (HC-1) of the TNFR-native Hodge program. Unlike the
sibling programs (Riemann / NS / Yang-Mills / P-vs-NP / BSD), the honest
verdict here is a STRONG NEGATIVE: TNFR's discrete Hodge theory captures the
*topological* content exactly but is structurally BLIND to the two features
that constitute the Hodge conjecture. This is NOT a solution and NOT even an
open attack surface on the conjecture itself (see "Honest scope").

TNFR-native object
------------------
Example 107 established the k=1 Helmholtz-Hodge decomposition of the phase
field on a graph (gradient + cycle). Extending the field to a 2-complex
(triangles) gives the FULL discrete Hodge decomposition. The tetrad supplies
the natural cochain tower:

    phase value        -> 0-cochain (vertices)
    phase gradient |grad phi| -> 1-cochain (edges)
    phase curvature K_phi     -> 2-cochain (triangles; holonomy / discrete curl)

The combinatorial Hodge Laplacians are

    L0 = d1 d1^T,   L1 = d1^T d1 + d2 d2^T,   L2 = d2^T d2,

where d1 (edges->vertices) and d2 (triangles->edges) are the simplicial
boundary maps. Eckmann's theorem (1944): the harmonic k-cochains are
isomorphic to the homology H_k, so

    dim ker L_k = b_k   (the k-th Betti number).

What this measures (HC-1)
-------------------------
  (1) the chain complex d1 d2 = 0 (the tetrad cochain tower is a complex);
  (2) harmonic dimensions = Betti numbers EXACTLY (Eckmann), on a torus and
      on a sphere -- the harmonic count tracks topology across spaces;
  (3) harmonic 1-forms are closed and co-closed to machine precision.

The HONEST gap (why this is NOT the Hodge conjecture)
-----------------------------------------------------
The Hodge conjecture states: on a non-singular complex projective variety,
every Hodge class (a rational cohomology class of type (p,p)) is a rational
combination of classes of ALGEBRAIC cycles (subvarieties cut out by
polynomials). Two features make it hard, and TNFR's discrete Hodge has
NEITHER:

  A. COMPLEX (p,p) BIGRADING. The conjecture lives in the Hodge
     decomposition H^k = ⊕_{p+q=k} H^{p,q} of a Kähler manifold, which
     requires a complex structure. The real combinatorial Laplacian L_k has
     no (p,q) bigrading -- there is only one real harmonic space per degree.

  B. ALGEBRAICITY. "Algebraic cycle" = cut out by polynomial equations,
     strictly stronger than "integer topological cycle." In the combinatorial
     setting EVERY harmonic class is already an integer simplicial cycle
     (Eckmann), so the discrete analogue of the conjecture is TRIVIALLY true
     -- precisely because the discrete setting cannot even express the
     algebraicity distinction that is the whole difficulty.

So TNFR's discrete Hodge is structurally BLIND to the actual Hodge
conjecture: it captures the topological half (harmonic = homology) exactly
and says nothing about the (p,p) bigrading or algebraicity. This is the
honest analogue of the Riemann result that the emergent substrate is "blind"
to the arithmetic content -- here the discrete cochain tower is blind to the
complex-algebraic content.

Honest scope
------------
- HC-1 reproduces classical combinatorial Hodge theory (Eckmann 1944) on the
  TNFR cochain tower. It does NOT prove or even attack the Hodge conjecture.
- The obstruction is classified Branch B3-leaning: there is no TNFR closure
  of the *actual* conjecture in the discrete setting, because the discrete
  setting cannot pose the (p,p)-bigrading / algebraicity question at all.
  (Contrast: P-vs-NP and BSD are Branch B -- open attack surfaces with a
  concrete next milestone; Hodge has no such discrete next milestone toward
  the conjecture.)
- The value is a PRECISE localisation of why Hodge is hard: it isolates
  exactly what the discrete/structural setting can and cannot see.

References
----------
- examples/107_orthogonal_structure_emergent_geometry.py (k=1 Helmholtz-Hodge)
- theory/TNFR_HODGE_RESEARCH_NOTES.md (program, milestones, classification)
- AGENTS.md section "Transport Content of the Nodal Equation"
"""

import os
import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def boundaries(verts, edges, tris):
    """Simplicial boundary maps d1 (E->V) and d2 (T->E) with orientation."""
    eidx = {e: k for k, e in enumerate(edges)}
    d1 = np.zeros((len(verts), len(edges)))
    for k, (a, b) in enumerate(edges):
        d1[a, k] = -1.0   # d[a,b] = [b] - [a]
        d1[b, k] = +1.0
    d2 = np.zeros((len(edges), len(tris)))
    for k, (a, b, c) in enumerate(tris):
        # d[a,b,c] = [b,c] - [a,c] + [a,b]
        d2[eidx[(b, c)], k] += 1.0
        d2[eidx[(a, c)], k] -= 1.0
        d2[eidx[(a, b)], k] += 1.0
    return d1, d2


def complex_from_triangles(triangles):
    verts, edges = set(), set()
    for t in triangles:
        verts.update(t)
        for e in itertools.combinations(sorted(t), 2):
            edges.add(e)
    return sorted(verts), sorted(edges), [tuple(sorted(t)) for t in triangles]


def torus_triangles(n):
    def vid(i, j):
        return (i % n) * n + (j % n)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b = vid(i, j), vid(i + 1, j)
            c, d = vid(i, j + 1), vid(i + 1, j + 1)
            tris += [(a, b, c), (b, d, c)]
    return tris


def dim_ker(M, tol=1e-9):
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    return int(np.sum(np.abs(ev) < tol))


def hodge_dims(verts, edges, tris):
    d1, d2 = boundaries(verts, edges, tris)
    L0 = d1 @ d1.T
    L1 = d1.T @ d1 + d2 @ d2.T
    L2 = d2.T @ d2
    return d1, d2, (dim_ker(L0), dim_ker(L1), dim_ker(L2))


def experiment_1_eckmann_torus():
    print("=" * 72)
    print("HC-1: Discrete Hodge on the TNFR cochain tower (triangulated torus)")
    print("=" * 72)
    print()
    V, E, T = complex_from_triangles(torus_triangles(5))
    d1, d2, dims = hodge_dims(V, E, T)
    print(f"  torus complex: |V|={len(V)} |E|={len(E)} |T|={len(T)}  "
          f"Euler={len(V) - len(E) + len(T)}")
    print(f"  (1) chain complex ||d1 d2|| = {np.abs(d1 @ d2).max():.2e}  (=0)")
    print(f"  (2) harmonic dims (ker L0, L1, L2) = {dims}  (Betti torus 1,2,1)")
    # harmonic 1-forms closed + co-closed
    L1 = d1.T @ d1 + d2 @ d2.T
    ev, U = np.linalg.eigh((L1 + L1.T) / 2)
    H = U[:, np.abs(ev) < 1e-9]
    print(f"  (3) harmonic 1-form subspace dim = {H.shape[1]} "
          f"(2 torus loops); closed |d1 h|={np.abs(d1 @ H).max():.1e}, "
          f"co-closed |d2^T h|={np.abs(d2.T @ H).max():.1e}")
    print()
    print("  Eckmann (1944): harmonic = homology, EXACT. The 2 harmonic")
    print("  1-forms are the 2 independent loops a TNFR phase field can wind")
    print("  around (the topological holes; cf. example 107).")
    print()


def experiment_2_topology_tracking():
    print("=" * 72)
    print("HC-1 contrast: harmonic count tracks topology EXACTLY")
    print("=" * 72)
    print()
    octa = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (0, 2, 5), (2, 1, 5), (1, 3, 5), (3, 0, 5)]
    Vs, Es, Ts = complex_from_triangles(octa)
    _, _, ds = hodge_dims(Vs, Es, Ts)
    Vt, Et, Tt = complex_from_triangles(torus_triangles(5))
    _, _, dt = hodge_dims(Vt, Et, Tt)
    print(f"  {'space':>22} {'|V|':>4} {'|E|':>4} {'|T|':>4} "
          f"{'harmonic dims':>14} {'Betti':>9}")
    print(f"  {'octahedron (sphere)':>22} {len(Vs):>4} {len(Es):>4} "
          f"{len(Ts):>4} {str(ds):>14} {'(1,0,1)':>9}")
    print(f"  {'torus (5x5)':>22} {len(Vt):>4} {len(Et):>4} "
          f"{len(Tt):>4} {str(dt):>14} {'(1,2,1)':>9}")
    print()
    print("  The sphere has NO 1-loops (harmonic_1 = 0); the torus has 2.")
    print("  The harmonic count is a faithful TOPOLOGICAL invariant -- and")
    print("  topology is exactly the half of Hodge that is NOT the hard part.")
    print()


def main():
    print()
    print("  TNFR Example 111: Hodge -- Discrete Hodge Theory and the Honest Gap")
    print("  Milestone HC-1 (structural reformulation; strong-negative verdict)")
    print("  =================================================================")
    print()
    experiment_1_eckmann_torus()
    experiment_2_topology_tracking()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES (and the honest strong negative)")
    print("=" * 72)
    print()
    print("ESTABLISHES: the TNFR cochain tower (phase value / gradient |grad")
    print("phi| / curvature K_phi over vertices / edges / triangles) carries a")
    print("complete discrete Hodge decomposition; harmonic = homology exactly")
    print("(Eckmann), tracking topology across spaces.")
    print()
    print("HONEST STRONG NEGATIVE: this is NOT the Hodge conjecture and not")
    print("even an open attack surface on it. The conjecture needs a complex")
    print("(p,p) bigrading (Kähler) and ALGEBRAIC cycles (polynomial-cut-out);")
    print("the real combinatorial setting has neither, and its harmonic")
    print("classes are TRIVIALLY integer cycles (Eckmann). TNFR's discrete")
    print("Hodge is structurally BLIND to the algebraic-complex content that")
    print("IS the difficulty -- the honest analogue of the Riemann substrate")
    print("being blind to arithmetic content. Obstruction Branch B3-leaning.")
    print("No Clay claim.")
    print()


if __name__ == "__main__":
    main()
