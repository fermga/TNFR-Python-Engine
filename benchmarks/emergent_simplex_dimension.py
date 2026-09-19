"""A specified simplex's grade and complete-graph spectral multiplicity.

For K_(n+1), the combinatorial Laplacian has nonzero eigenvalue n+1 with
multiplicity n, matching its standard representation dimension. A chosen
filled n-simplex also has dimension n. The graph alone is its one-dimensional
skeleton; filling the clique is an additional simplicial construction.
Adding one apex to K_m produces K_(m+1), a cone operation, not a suspension
or automatically a THOL/U5 nesting event. No phase compatibility, autonomous
formation or physical dimension selection follows from the graph identity.
The equality of these counts holds in the specified family; it does not equate
all numbers, form coordinates and dimensions throughout TNFR.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
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
    # The nonzero-eigenvalue multiplicity is an irrep dimension of S_V, shared by
    # every Aut(K_V)-equivariant operator, so it is operator-invariant. Computed
    # on the canonical EMERGENT operator L_sym (self-adjoint twin of the ΔNFR
    # random-walk L_rw); K_V is vertex-transitive so the count equals D - A's.
    from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian

    _, L = symmetric_normalized_laplacian(nx.complete_graph(n_vertices))
    ev = np.sort(np.linalg.eigvalsh(L))
    return int(np.sum(ev > 1e-9))


SIMPLEX_NAMES = {1: "edge", 2: "triangle", 3: "tetrahedron", 4: "4-simplex"}


def main() -> None:
    print("=" * 70)
    print("COMPLETE-GRAPH MULTIPLICITY AND CHOSEN SIMPLEX GRADE")
    print("=" * 70)

    # -- M1: K_{n+1} carries cardinal n = simplex dimension -----------------
    print("\n[M1] The n-simplex (K_{n+1}) carries cardinal n = its dim:")
    for v in (2, 3, 4, 5):
        card = simplex_cardinal(v)
        name = SIMPLEX_NAMES.get(v - 1, f"{v - 1}-simplex")
        print(
            f"     {name:<12s} K_{v}: eig {v} x{v - 1} "
            f"-> cardinal {card} = {v - 1}D"
        )
        assert card == v - 1, f"cardinal {card} != dim {v - 1}"
    print("     -> PASS: cardinal n = multiplicity = standard-irrep dim =")
    print(
        "        the n-simplex dimension. The counts agree in this chosen simplex family."
    )

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
    print("\n[M2] The chosen cone construction (one additional apex):")
    print(f"     triangle  K_3 -> cardinal {card_tri} -> 2D")
    print("     + 1 supplied apex (cone) ->")
    print(f"     tetrahedron K_4 -> cardinal {card_tet} -> 3D")
    assert card_tri == 2 and card_tet == 3, "lift cardinals wrong"
    assert is_cone, "K_4 is not the cone over K_3"
    print("     -> PASS: ONE added fully connected vertex raises the chosen simplex")
    print("        grade from 2 to 3; this is a combinatorial construction,")
    print("        not a derived state-dependent operator or physical dimension.")

    # -- M3: the EPI/form reading -------------------------------------------
    print("\n[M3] Geometric interpretation of the selected filled clique:")
    print(
        "     vertices = NFRs; all edges present; phase resonance not checked; the form's"
    )
    print("     grade counts vertices minus one in this filled-simplex family.")
    print(f"     2D form (triangle EPI) : grade {card_tri}")
    print(f"     3D form (tetra   EPI)  : grade {card_tet}")
    assert card_tet == card_tri + 1, "one degree did not add one dimension"
    print("     -> PASS: +1 apex raises the specified simplex grade by one:")
    print("        dimension of the chosen filled simplex, not physical space.")

    print("\n" + "=" * 70)
    print("SCOPE OF THE COUNTING IDENTITY")
    print("=" * 70)
    print(
        "SCOPED IDENTITY: K_(n+1) has nonzero combinatorial-Laplacian multiplicity n.\nThe standard representation has dimension n, as does a chosen filled n-simplex.\nThe graph skeleton itself is one-dimensional; clique filling is an added construction.\nAdding an apex selects the next complete graph, without executing THOL or deriving phase laws.\nThis family does not establish physical spatial dimension or autonomous form generation."
    )


if __name__ == "__main__":
    main()
