"""
Emergent Integers as Spectral Multiplicities: A Symmetry -> Degeneracy Falsifier
================================================================================

QUESTION (the deep one): does TNFR *explain what an integer IS* — as a structural
invariant — rather than merely *use* integers as external tags?

This harness compares spectral counts on explicitly supplied graph controls:

    The integers that emerge from the nodal dynamics are the eigenvalue
    multiplicities of L_rw = I - D^-1 W, the isolated scalar EPI pressure
    operator. On these regular unit-conductance graphs D - A = d*L_rw,
    so these two operators share eigenspaces. This does not identify nonlinear
    phase curvature with L_rw or derive the chosen graph from nodal evolution.

WHY THIS IS RIGOROUS (and falsifiable):
  L commutes with every automorphism of the graph, so each eigenspace is an
  invariant subspace of the symmetry group Γ and decomposes into irreducible
  representations of Γ. An eigenvalue multiplicity can be a SUM of irrep
  contributions, including repeated copies; vertex transitivity does not make
  every eigenspace irreducible. Nor do all equivariant operators share their
  complete eigenspaces: different sectors can coincide at one eigenvalue.
  The existing inverse_spectrum_to_symmetry.py supplies a concrete control:
  the truncated-cube graph has multiplicity five with octahedral symmetry,
  splitting as 2+3. Its combinatorial eigenvalues 3 and 5 each have exact
  nullity five; the corresponding normalized eigenvalues are 1 and 5/3.

  Reference irrep dimensions used for these selected cases (not a general
  upper bound or exhaustive prediction for eigenspace multiplicities):
    Cyclic  C_n            : {1, 2}         (rotation blocks)
    Tetrahedral  T_d       : {1, 2, 3}      (the integer 3 first becomes available)
    Octahedral   O_h       : {1, 2, 3}
    Icosahedral  I_h       : {1, 3, 4, 5}   (4 and 5 are the icosahedral signature)
    Full rotation SO(3)    : {1, 3, 5, 7, …} (continuum sphere: every odd 2l+1)

  A measured multiplicity counts spectral modes. A count of three or five
  alone neither proves irreducibility nor identifies tetrahedral or icosahedral
  symmetry. Protected sectors must be distinguished from coincident sectors.

WHAT THIS DOES *NOT* CLAIM (the honest boundary):
  This produces CARDINALS (dimensions/counts) as emergent integers. It does NOT
  derive the full arithmetic ring (addition, multiplication, primality) from the
  nodal equation. The number-theory layer still *uses* integers as inputs and
  characterizes their primality; it does not derive their existence. The
  invariant-eigenspace statement is standard representation theory, not a
  derivation of arithmetic, graph selection or Euclidean space. The capacity,
  forcing and operator maps must separately preserve a graph symmetry before
  it constrains a complete TNFR dynamics. Reported eigenvalue clusters are
  numerical observations; the finite sphere fixture is not exact SO(3).

Run:
    python benchmarks/emergent_integers_symmetry.py

Theoretical anchor: AGENTS.md (nodal equation; isolated scalar EPI diffusion).
Status: RESEARCH (supplied symmetry controls).
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
    allowed_irrep_dims: set[int]  # reference catalog, not a multiplicity bound
    signature: int  # signature checked in this supplied fixture


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
    """Check this fixture against its catalog; no general multiplicity theorem."""
    nontrivial = {m for m in emergent if m > 1}
    subset_ok = nontrivial.issubset(case.allowed_irrep_dims)
    signature_ok = case.signature in emergent
    return subset_ok, signature_ok


def main() -> None:
    print(__doc__)
    print("=" * 78)
    print(
        "SUPPLIED SYMMETRY CONTROLS: Laplacian multiplicities vs listed irrep dimensions"
    )
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
        f"OVERALL: {'ALL PASS — selected fixture counts match the listed values' if all_pass else 'MISMATCH'}"
    )
    print("=" * 78)
    print("\nInterpretation (honest scope):")
    print("  • The integers 1,2,3,4,5,7 are OUTPUTS — eigenvalue multiplicities of")
    print("    EPI operator L_rw, equal to D - A's counts on these regular graphs.")
    print("    The graph geometries are inputs; the spectral counts are read-outs.")
    print("  • Eigenspaces are symmetry-invariant but can combine irrep sectors.")
    print("    A count of five alone does not identify icosahedral symmetry:")
    print("    the truncated cube has an accidental five = two + three.")
    print("  • These counts establish neither graph selection nor an ambient")
    print("    Euclidean dimension. The finite sphere is a numerical comparison.")
    print("  • BOUNDARY: this gives cardinals (counts/dimensions), NOT the full")
    print("    arithmetic ring. Deriving (+, ×, primality) of integers from the nodal")
    print("    equation remains open; number_theory.py still takes integers as input.")


if __name__ == "__main__":
    main()
