#!/usr/bin/env python3
"""
Example 111 — Auxiliary Finite Simplicial Hodge Baseline
=========================================================

This HC-1 baseline constructs finite simplicial incidence matrices and
reproduces standard combinatorial Hodge results. It does not derive a cochain
complex from the canonical TNFR tetrad.

Scope correction
----------------
``compute_phase_gradient`` returns a nonnegative mean per node and loses edge
orientation and sign. ``compute_phase_curvature`` also returns a wrapped scalar
per node, not a value on an oriented face. Neither can be identified with the
1- and 2-cochains used below. Example 107 checks an oriented EPI edge gradient
against a selected circulation; it does not establish this missing bridge.

The combinatorial Hodge Laplacians are

    L0 = d1 d1^T,   L1 = d1^T d1 + d2 d2^T,   L2 = d2^T d2,

where d1 (edges->vertices) and d2 (triangles->edges) are the simplicial
boundary maps. Eckmann's theorem (1944): the harmonic k-cochains are
isomorphic to the homology H_k, so

    dim ker L_k = b_k   (the k-th Betti number).

What this measures (HC-1)
-------------------------
  (1) the independently built chain complex satisfies d1 d2 = 0;
  (2) numerically detected harmonic dimensions match the expected Betti
      numbers on one triangulated torus and sphere;
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
     strictly stronger than a topological cycle. Finite simplicial Hodge
     theory gives harmonic representatives of real cohomology; it neither
     makes every representative integral nor supplies algebraic cycles.

The auxiliary finite complex says nothing about the (p,p) bigrading or
algebraicity. A TNFR-specific result would first require oriented edge and
face observables and a proof connecting them to canonical telemetry.

Honest scope
------------
- HC-1 reproduces a classical combinatorial Hodge baseline (Eckmann 1944).
- It does NOT prove or attack the Hodge conjecture and does not establish that
  the tetrad is a cochain tower.
- The current baseline cannot pose the (p,p)-bigrading or algebraicity question.

References
----------
- examples/08_emergent_geometry/107_orthogonal_structure_emergent_geometry.py (k=1 Helmholtz-Hodge)
- theory/TNFR_HODGE_RESEARCH_NOTES.md (program, milestones, classification)
- AGENTS.md sections "Transport content" and "The structural tetrad"
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np


def boundaries(verts, edges, tris):
    """Simplicial boundary maps d1 (E->V) and d2 (T->E) with orientation."""
    eidx = {e: k for k, e in enumerate(edges)}
    d1 = np.zeros((len(verts), len(edges)))
    for k, (a, b) in enumerate(edges):
        d1[a, k] = -1.0  # d[a,b] = [b] - [a]
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
    print("HC-1: Auxiliary simplicial Hodge baseline (triangulated torus)")
    print("=" * 72)
    print()
    V, E, T = complex_from_triangles(torus_triangles(5))
    d1, d2, dims = hodge_dims(V, E, T)
    print(
        f"  torus complex: |V|={len(V)} |E|={len(E)} |T|={len(T)}  "
        f"Euler={len(V) - len(E) + len(T)}"
    )
    print(f"  (1) chain complex ||d1 d2|| = {np.abs(d1 @ d2).max():.2e}  (=0)")
    print(f"  (2) harmonic dims (ker L0, L1, L2) = {dims}  (Betti torus 1,2,1)")
    # harmonic 1-forms closed + co-closed
    L1 = d1.T @ d1 + d2 @ d2.T
    ev, U = np.linalg.eigh((L1 + L1.T) / 2)
    H = U[:, np.abs(ev) < 1e-9]
    print(
        f"  (3) harmonic 1-form subspace dim = {H.shape[1]} "
        f"(2 torus loops); closed |d1 h|={np.abs(d1 @ H).max():.1e}, "
        f"co-closed |d2^T h|={np.abs(d2.T @ H).max():.1e}"
    )
    print()
    print("  Eckmann (1944) gives the exact finite-complex theorem; the values")
    print("  above are its numerical realization at tolerance 1e-9.")
    print()


def experiment_2_topology_tracking():
    print("=" * 72)
    print("HC-1 contrast: Betti dimensions distinguish the two complexes")
    print("=" * 72)
    print()
    octa = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (0, 2, 5),
        (2, 1, 5),
        (1, 3, 5),
        (3, 0, 5),
    ]
    Vs, Es, Ts = complex_from_triangles(octa)
    _, _, ds = hodge_dims(Vs, Es, Ts)
    Vt, Et, Tt = complex_from_triangles(torus_triangles(5))
    _, _, dt = hodge_dims(Vt, Et, Tt)
    print(
        f"  {'space':>22} {'|V|':>4} {'|E|':>4} {'|T|':>4} "
        f"{'harmonic dims':>14} {'Betti':>9}"
    )
    print(
        f"  {'octahedron (sphere)':>22} {len(Vs):>4} {len(Es):>4} "
        f"{len(Ts):>4} {str(ds):>14} {'(1,0,1)':>9}"
    )
    print(
        f"  {'torus (5x5)':>22} {len(Vt):>4} {len(Et):>4} "
        f"{len(Tt):>4} {str(dt):>14} {'(1,2,1)':>9}"
    )
    print()
    print("  The sphere has NO 1-loops (harmonic_1 = 0); the torus has 2.")
    print("  Betti numbers are topological invariants, but not complete ones.")
    print()


def main():
    print()
    print("  TNFR Example 111: Auxiliary Finite Simplicial Hodge Baseline")
    print("  Milestone HC-1 (external comparison with an explicit scope gap)")
    print("  =================================================================")
    print()
    experiment_1_eckmann_torus()
    experiment_2_topology_tracking()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("ESTABLISHES: the independently constructed finite simplicial")
    print("complexes reproduce the expected combinatorial Hodge dimensions and")
    print("chain identity. This is a standard mathematical baseline.")
    print()
    print("SCOPE BOUNDARY: this is NOT the Hodge conjecture. The conjecture")
    print("needs a complex")
    print("(p,p) bigrading (Kähler) and ALGEBRAIC cycles (polynomial-cut-out);")
    print("the real combinatorial setting has neither. Canonical |grad phi|")
    print("and K_phi are node summaries, so they do not instantiate the edge")
    print("and face cochains used here. The TNFR bridge remains OPEN.")
    print("No Clay claim and no tetrad-completeness claim.")
    print()


if __name__ == "__main__":
    main()
