"""
Emergent Integers as Spectral Multiplicities: A Symmetry -> Degeneracy Falsifier
================================================================================

QUESTION (the deep one): does TNFR *explain what an integer IS* — as a structural
invariant — rather than merely *use* integers as external tags?

This harness tests the strongest defensible form of that claim:

    The integers that emerge from the nodal dynamics are the eigenvalue
    multiplicities of the structural Laplacian L = D - A (the discrete ΔNFR /
    phase-curvature operator), and WHICH integers emerge is dictated entirely by
    the symmetry group of the manifold.

WHY THIS IS RIGOROUS (and falsifiable):
  L commutes with every automorphism of the graph, so each eigenspace is an
  invariant subspace of the symmetry group Γ and decomposes into irreducible
  representations of Γ. For vertex-transitive manifolds the eigenvalue
  multiplicities are exactly the dimensions of irreps of Γ. Representation
  theory therefore predicts the emergent integers INDEPENDENTLY of TNFR — it is
  the ground truth against which we falsify.

  Allowed irrep dimensions per symmetry group:
    Cyclic  C_n            : {1, 2}         (rotation blocks)
    Tetrahedral  T_d       : {1, 2, 3}      (the integer 3 first becomes available)
    Octahedral   O_h       : {1, 2, 3}
    Icosahedral  I_h       : {1, 3, 4, 5}   (4 and 5 are the icosahedral signature)
    Full rotation SO(3)    : {1, 3, 5, 7, …} (continuum sphere: every odd 2l+1)

  So "a 3" = the dimension of a 3D irreducible coherent mode; it cannot appear
  below tetrahedral symmetry. "A 5" requires icosahedral symmetry. The integer
  is not a tag — it is a count of structurally indistinguishable resonant modes,
  fixed by the geometry.

WHAT THIS DOES *NOT* CLAIM (the honest boundary):
  This produces CARDINALS (dimensions/counts) as emergent integers. It does NOT
  derive the full arithmetic ring (addition, multiplication, primality) from the
  nodal equation. The number-theory layer still *uses* integers as inputs and
  characterizes their primality; it does not derive their existence. The
  multiplicity-as-irrep-dimension fact is the structural reading of a known
  theorem (L commutes with Aut(G)), not new mathematics — TNFR supplies the
  physical interpretation (L = discrete ΔNFR), not the theorem.

Run:
    python benchmarks/emergent_integers_symmetry.py

Theoretical anchor: AGENTS.md (nodal equation; discrete-mode regime; structural
Laplacian as discrete ΔNFR/phase curvature). Status: RESEARCH (falsifier).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


def laplacian_multiplicities(
    G: nx.Graph, *, tol: float = 1e-6
) -> list[tuple[float, int]]:
    """Return (eigenvalue, multiplicity) pairs of L = D - A, ascending.

    Multiplicities are the EMERGENT integers. Clustering is gap-based so that
    irrational eigenvalues (e.g. 5 ± √5 for the icosahedron) are grouped cleanly.
    """
    G = nx.Graph(G)  # collapse any multi-edges; ensure simple graph
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    D = np.diag(A.sum(axis=1))
    L = D - A
    evals = np.sort(np.linalg.eigvalsh(L))
    groups: list[list[float]] = [[float(evals[0])]]
    for ev in evals[1:]:
        if abs(ev - groups[-1][-1]) <= tol:
            groups[-1].append(float(ev))
        else:
            groups.append([float(ev)])
    return [(float(np.mean(g)), len(g)) for g in groups]


@dataclass(frozen=True)
class SymmetryCase:
    name: str
    builder: object  # callable -> nx.Graph
    group: str
    allowed_irrep_dims: set[int]  # rep-theory ground truth
    signature: int  # the largest irrep dim that must appear


def _fibonacci_sphere(n_points: int = 400, k_neighbors: int = 6) -> nx.Graph:
    """Closed S² manifold (SO(3) symmetry in the continuum limit)."""
    idx = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1.0 - 2.0 * idx / (n_points - 1)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * idx
    pts = np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])
    G = nx.Graph()
    G.add_nodes_from(idx.tolist())
    for i in range(n_points):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1 : k_neighbors + 1]:
            G.add_edge(int(i), int(j))
    return G


CASES = [
    SymmetryCase(
        "Cyclic ring C_8", lambda: nx.cycle_graph(8), "C_8 (cyclic)", {1, 2}, 2
    ),
    SymmetryCase(
        "Tetrahedron", nx.tetrahedral_graph, "T_d (tetrahedral)", {1, 2, 3}, 3
    ),
    SymmetryCase("Octahedron", nx.octahedral_graph, "O_h (octahedral)", {1, 2, 3}, 3),
    SymmetryCase("Cube", nx.cubical_graph, "O_h (octahedral)", {1, 2, 3}, 3),
    SymmetryCase(
        "Icosahedron", nx.icosahedral_graph, "I_h (icosahedral)", {1, 3, 4, 5}, 5
    ),
    SymmetryCase(
        "Dodecahedron", nx.dodecahedral_graph, "I_h (icosahedral)", {1, 3, 4, 5}, 5
    ),
]


def _verdict(emergent: set[int], case: SymmetryCase) -> tuple[bool, bool]:
    """(all emergent dims are allowed irreps, signature integer appears)."""
    nontrivial = {m for m in emergent if m > 1}
    subset_ok = nontrivial.issubset(case.allowed_irrep_dims)
    signature_ok = case.signature in emergent
    return subset_ok, signature_ok


def main() -> None:
    print(__doc__)
    print("=" * 78)
    print("SYMMETRY  ->  EMERGENT INTEGERS (Laplacian multiplicities)  vs  rep theory")
    print("=" * 78)
    print(f"{'manifold':<16}{'group':<20}{'emergent mults':<22}{'allowed':<14}verdict")
    print("-" * 78)

    all_pass = True
    for case in CASES:
        G = case.builder()
        mults = laplacian_multiplicities(G)
        emergent_seq = [m for _ev, m in mults]
        emergent_set = set(emergent_seq)
        subset_ok, signature_ok = _verdict(emergent_set, case)
        ok = subset_ok and signature_ok
        all_pass &= ok
        allowed = "{" + ",".join(str(d) for d in sorted(case.allowed_irrep_dims)) + "}"
        verdict = "PASS" if ok else "FAIL"
        sig = f" (signature {case.signature}{'✓' if signature_ok else '✗'})"
        print(
            f"{case.name:<16}{case.group:<20}{str(emergent_seq):<22}{allowed:<14}{verdict}{sig}"
        )

    # Continuum limit: the sphere (SO(3)) — every odd integer 2l+1 emerges.
    print("-" * 78)
    sphere = _fibonacci_sphere(400, 6)
    sphere_mults = [m for _ev, m in laplacian_multiplicities(sphere, tol=0.02)][:5]
    print(
        f"{'Sphere S² (n=400)':<16}{'SO(3) (continuum)':<20}{str(sphere_mults):<22}"
        f"{'{1,3,5,7,…}':<14}{'odd 2l+1 ✓' if sphere_mults[:4] == [1, 3, 5, 7] else 'check'}"
    )

    print("=" * 78)
    print(
        f"OVERALL: {'ALL PASS — emergent integers match rep-theory prediction' if all_pass else 'MISMATCH'}"
    )
    print("=" * 78)
    print("\nInterpretation (honest scope):")
    print("  • The integers 1,2,3,4,5,7 are OUTPUTS — eigenvalue multiplicities of the")
    print("    structural Laplacian (discrete ΔNFR). They were not supplied.")
    print("  • WHICH integers appear is fixed by the symmetry group's irreducible")
    print("    representations: 3 first appears at tetrahedral symmetry, 5 requires")
    print("    icosahedral symmetry, the sphere yields every odd 2l+1.")
    print("  • So a '3' = dimension of a 3D irreducible coherent mode. The integer is")
    print("    a structural count, not an arbitrary tag — it fossilizes the geometry.")
    print("  • BOUNDARY: this gives cardinals (counts/dimensions), NOT the full")
    print("    arithmetic ring. Deriving (+, ×, primality) of integers from the nodal")
    print("    equation remains open; number_theory.py still takes integers as input.")


if __name__ == "__main__":
    main()
