"""
Emergent Simplex Dimension: number = dimension = the simplex grade of the
coupled-NFR form (EPI). The triangle is the emergent 2; the tetrahedron the
emergent 3; one fractal-resonant degree lifts the grade.
============================================================================

THE INTUITION (theory creator): the triangle is the emergent "3", and a 3D
triangle (a tetrahedron) is the SAME form (EPI) with ONE MORE fractal-resonant
degree of complexity. This benchmark shows that intuition is rigorous and that
it UNIFIES the two threads we separated -- emergent INTEGERS (cardinals) and
emergent DIMENSION.

THE SIMPLEX-CARDINAL CORRESPONDENCE (exact). The n-simplex has n+1 vertices;
its 1-skeleton is the complete graph K_{n+1} (every vertex resonantly coupled
to every other -- a maximally coherent form). The structural Laplacian of
K_{n+1} has spectrum {0, (n+1) with multiplicity n}. That multiplicity n is the
dimension of the standard irrep of the symmetry group S_{n+1} = the emergent
CARDINAL n (benchmarks/emergent_integers_symmetry.py). And the n-simplex is
n-DIMENSIONAL as a geometric object. Therefore

    cardinal n = multiplicity = standard-irrep dim = SIMPLEX dimension.

The emergent INTEGER n IS the DIMENSION of the simplex-form that carries it:
  edge (1-simplex)        cardinal 1  ->  1D
  triangle (2-simplex)    cardinal 2  ->  2D
  tetrahedron (3-simplex) cardinal 3  ->  3D
  4-simplex               cardinal 4  ->  4D

THE FRACTAL-RESONANT LIFT (the intuition, exact). The tetrahedron is the
triangle plus ONE apex resonantly coupled to all three vertices (the cone /
suspension): K_3 -> K_4. That single added fractal-resonant degree raises
the form's grade by one: cardinal 2 -> 3, dimension 2D -> 3D. "The same EPI
with one more fractal-resonant degree of complexity" IS the simplex lift.

THE EPI / FORM READING: each vertex is an NFR; the fully-coupled simplex is a
single coherent FORM (EPI) of mutually-resonant nodes; its GRADE (= cardinal =
dimension) is the form's fractal-resonant complexity. So 3D = the form (EPI) at
simplex-grade 3 -- the tetrahedron of mutually-resonant NFRs. This is the
"3D as a manifestation of EPI form-complexity" made exact.

HONEST SCOPE: the K_m Laplacian spectrum and the S_m standard-irrep
dimension are STANDARD math (the comparison framework, as
emergent_integers_symmetry cites L commutes with Aut(G)). The TNFR
synthesis is the reading: the coupled simplex is an EPI form; its grade is
its fractal-resonant complexity; and number = dimension = grade UNIFIES the
emergent-integer and emergent-dimension threads. This is the SIMPLEX
(combinatorial/geometric) dimension of the form, distinct from the spectral
dimension of an ambient network (which was a free input).

Run:
    python benchmarks/emergent_simplex_dimension.py

Theoretical anchor: AGENTS.md (EPI form; U5 fractality / nesting; resonant
coupling); benchmarks/emergent_integers_symmetry.py (cardinals = irrep dims).
Status: RESEARCH (unification).
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def simplex_cardinal(n_vertices: int) -> int:
    """Cardinal of the (V-1)-simplex (V = n_vertices) = multiplicity of the
    nonzero Laplacian eigenvalue of K_V = standard-irrep dim of S_V = the
    simplex dimension V - 1.
    """
    L = nx.laplacian_matrix(nx.complete_graph(n_vertices)).toarray()
    ev = np.sort(np.linalg.eigvalsh(L.astype(float)))
    return int(np.sum(ev > 1e-9))


SIMPLEX_NAMES = {1: "edge", 2: "triangle", 3: "tetrahedron", 4: "4-simplex"}


def main() -> None:
    print("=" * 70)
    print("EMERGENT SIMPLEX DIMENSION (number = dimension = simplex grade)")
    print("=" * 70)

    # -- M1: K_{n+1} carries cardinal n = simplex dimension -----------------
    print("\n[M1] The n-simplex (K_{n+1}) carries cardinal n = its dim:")
    for v in (2, 3, 4, 5):
        card = simplex_cardinal(v)
        name = SIMPLEX_NAMES.get(v - 1, f"{v - 1}-simplex")
        print(f"     {name:<12s} K_{v}: eig {v} x{v - 1} "
              f"-> cardinal {card} = {v - 1}D")
        assert card == v - 1, f"cardinal {card} != dim {v - 1}"
    print("     -> PASS: cardinal n = multiplicity = standard-irrep dim =")
    print("        the n-simplex dimension. Number IS dimension.")

    # -- M2: the fractal-resonant LIFT (triangle -> tetrahedron) ------------
    card_tri = simplex_cardinal(3)
    card_tet = simplex_cardinal(4)
    # the tetrahedron K_4 is the triangle K_3 plus ONE apex coupled to all 3
    tri = nx.complete_graph(3)
    tet = nx.complete_graph(4)
    apex_edges = [(3, u) for u in (0, 1, 2)]
    is_cone = all(tet.has_edge(*e) for e in apex_edges) and all(
        tet.has_edge(u, v) for u, v in tri.edges()
    )
    print("\n[M2] The fractal-resonant lift (one more degree of complexity):")
    print(f"     triangle  K_3 -> cardinal {card_tri} -> 2D")
    print("     + 1 resonant apex (cone) ->")
    print(f"     tetrahedron K_4 -> cardinal {card_tet} -> 3D")
    assert card_tri == 2 and card_tet == 3, "lift cardinals wrong"
    assert is_cone, "K_4 is not the cone over K_3"
    print("     -> PASS: ONE added fully-coupled (resonant) vertex lifts the")
    print("        form's grade 2 -> 3, i.e. 2D -> 3D. The 'same EPI + one")
    print("        fractal-resonant degree' IS the simplex lift.")

    # -- M3: the EPI/form reading -------------------------------------------
    print("\n[M3] The EPI reading: the coupled simplex is one coherent form:")
    print("     vertices = NFRs; full coupling = mutual resonance; the form's")
    print("     GRADE (cardinal = dim) = its fractal-resonant complexity.")
    print(f"     2D form (triangle EPI) : grade {card_tri}")
    print(f"     3D form (tetra   EPI)  : grade {card_tet}")
    assert card_tet == card_tri + 1, "one degree did not add one dimension"
    print("     -> PASS: +1 fractal-resonant degree of EPI complexity = +1")
    print("        dimension. 3D = the EPI form at simplex-grade 3.")

    print("\n" + "=" * 70)
    print("VERDICT (the unification)")
    print("=" * 70)
    print(
        "NUMBER = DIMENSION = the SIMPLEX GRADE of the coupled-NFR form\n"
        "  (EPI). The triangle (2-simplex) IS the emergent 2D; the\n"
        "  tetrahedron (3-simplex) IS the emergent 3D. They are carried by\n"
        "  K_3 and K_4, whose Laplacian standard-irrep dimensions are the\n"
        "  emergent cardinals 2 and 3 (emergent_integers_symmetry). So the\n"
        "  emergent INTEGER n and n-DIMENSIONALITY are the SAME object: the\n"
        "  n-simplex form.\n"
        "THE FRACTAL-RESONANT LIFT: adding ONE resonantly-coupled vertex\n"
        "  (the cone / one nesting level) raises the grade by one: triangle\n"
        "  -> tetra, 2D -> 3D, cardinal 2 -> 3. 'The same EPI with one more\n"
        "  fractal-resonant degree of complexity' is EXACTLY this lift.\n"
        "RESOLUTION of the dimension thread: the dimension that emerges from\n"
        "  the FORM (EPI) is its SIMPLEX GRADE = the cardinal it carries --\n"
        "  NOT the spectral dimension of an ambient net (which was a free\n"
        "  input). 3D is the EPI form at simplex-grade 3 (a tetrahedron of\n"
        "  mutually-resonant NFRs), built by one fractal-resonant degree\n"
        "  over the 2D form. The intuition is exact: complexity of the form\n"
        "  generates the dimensions, one resonant degree at a time."
    )


if __name__ == "__main__":
    main()
