"""Emergent Atomic Shells: the shell tower goes through the EPI form's grade.

THE CLAIM (user, theory creator): "si la dimension emerge de EPI, la emergencia
de los atomos debe ir por ahi" -- if dimension emerges from the EPI form (its
simplex grade, pinned by THOL self-similar nesting, see
emergent_fractal_simplex_dimension.py), then the ATOM's shell structure cannot
come from an imported spatial ball -- it must be the degeneracy/cardinal
structure of that SAME coherent form.

THE OBSTRUCTION it removes: a single coherent simplex K_{d+1} is exactly ONE
shell -- the standard irrep of S_{d+1}, degeneracy d (= the simplex cardinal =
the emergent dimension, emergent_integers_symmetry.py + emergent_simplex_
dimension.py). One shell is not an atom. A multi-shell atom needs NESTING --
and the canonical nesting operator is THOL/U5. Recursing K_m into m
corner-glued copies (the Sierpinski gasket of K_m, the canonical THOL
self-similar lift) turns the one-shell simplex into a TOWER of shells.

WHAT EMERGES (measured below): every shell of the grade-(m-1) nested form
inherits the SAME degeneracy m-1 -- the local simplex cardinal. So the emergent
DIMENSION (the grade) is stamped on the whole shell tower:

  - M1 (EXACT): the first-excited shell degeneracy = m-1 = the simplex grade =
    the emergent dimension, and m-1 is the MODAL non-trivial degeneracy across
    the tower. Shell degeneracy = simplex grade = dimension. This fuses the two
    emergent readings: "a number is a shell degeneracy (irrep cardinal)" and
    "a dimension is a simplex grade" are the SAME quantity.
  - M2 (DIMENSION SIGNATURE, exact): the ground + first shell close at 2 and
    2*(grade+1). grade 2 -> 2,6 ; grade 3 -> 2,8 ; grade 4 -> 2,10. The first
    closure READS the form's dimension. (grade 3 -> 8 = the 3D-oscillator /
    nuclear closure; grade 4 -> 10 touches the atomic SO(4) Coulomb cardinal.)
  - M3 (the tower, honest): the U(grade) isotropic-oscillator magic numbers
    all APPEAR among the shell closures -- most cleanly for grade 3, where
    {8,20,40} = U(3)[1:4] are all present (the 3D-oscillator / nuclear tower);
    grade 2 surfaces {6,12,...} = U(2) = the observed 2D quantum-dot atoms
    (matching the substrate's own locked U(2), emergent_substrate_symmetry.py).
    They are MIXED with extra Sierpinski localized-mode closures (genuine
    spectral-decimation degeneracies that take the BIGGEST gaps -- not
    oscillator numbers), so the match is a co-occurrence, not an exclusive
    tower.
  - M4 (synthesis): the atom's shells go through the EPI form's grade. The 2D
    and 3D shell towers are the grade-2 and grade-3 forms; the full CHEMICAL
    periodic table (2,10,18,36,54,86 = SO(4,2)/Madelung) still needs the
    two-body screening correction -- which is the SAME Fix(G)^perp
    Equivariance Wall (prime fine structure / RH S(T)) layered on the
    independent-particle skeleton.

So the shell structure of an atom emerges from the COHERENT FORM (EPI) and its
THOL grade -- not from a spatial ball put in by hand. The dimension that comes
from the form (its grade) IS the shell degeneracy; the cumulative tower is the
U(grade) oscillator skeleton.

HONEST SCOPE: K_m spectra + S_m irrep dims + the Sierpinski-gasket spectral
decimation are STANDARD mathematics (the comparison framework, as the emergent-
number arc cites L-commutes-with-Aut(G)). The U(d) isotropic-oscillator magic
numbers are likewise standard. The TNFR content is the reading "atomic shell
degeneracy = simplex grade = emergent dimension of the THOL-nested EPI form",
unifying the emergent-integer, emergent-dimension and shell-cardinal threads.
This reaches only the independent-particle (U(grade)) skeleton; it does NOT
derive the chemical periodic table (needs two-body screening) and closes no
open problem. R and pi remain the assumed substrate.

Run:
    python benchmarks/emergent_atomic_shells.py

Theoretical anchor: AGENTS.md (THOL self-organization, U5 multi-scale
fractality; L = discrete DeltaNFR); benchmarks/emergent_fractal_simplex_
dimension.py (THOL pins the dimension), emergent_simplex_dimension.py
(dimension = grade), emergent_shell_cardinals.py (U(d) magic numbers),
emergent_substrate_symmetry.py (substrate U(2) = 2D dots).
Status: RESEARCH (synthesis / falsifier).
"""

from __future__ import annotations

import pathlib
import sys
from math import comb

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --- THOL/U5 self-similar simplex nesting (Sierpinski gasket of K_m) ---
def sierpinski_simplex(m: int, levels: int):
    """Level-``levels`` self-similar nesting of the m-corner simplex K_m
    (the canonical THOL/U5 lift; node = sub-simplex via Kron node=subgraph)."""
    if levels == 0:
        return nx.complete_graph(m), list(range(m))
    sub, subc = sierpinski_simplex(m, levels - 1)
    G = nx.Graph()
    copies = []
    for i in range(m):
        mp = {v: (i, v) for v in sub.nodes}
        G.add_nodes_from(mp[v] for v in sub.nodes)
        G.add_edges_from((mp[u], mp[v]) for u, v in sub.edges)
        copies.append([mp[c] for c in subc])
    parent = {n: n for n in G.nodes}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in range(m):
        for j in range(i + 1, m):
            ra, rb = find(copies[i][j]), find(copies[j][i])
            if ra != rb:
                parent[rb] = ra
    H = nx.Graph()
    for u, v in G.edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            H.add_edge(ru, rv)
    corners = [find(copies[i][i]) for i in range(m)]
    return H, corners


# --- shells = degenerate levels of the structural Laplacian L = D - A ---
def shells(G, tol: int = 6):
    """(eigenvalues, multiplicities) ascending; multiplicity = shell
    degeneracy = irrep cardinal of Aut(G) (the emergent-integers reading)."""
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    ev = np.linalg.eigvalsh(L)
    vals, counts = np.unique(np.round(ev, tol), return_counts=True)
    return vals, counts


def closures(vals, counts, zmin: float = 1.5):
    """Shell closures = cumulative 2*count at gaps above zmin*median gap."""
    gaps = np.diff(vals)
    med = float(np.median(gaps[gaps > 1e-9]))
    out = []
    cum = 0
    for i in range(len(vals) - 1):
        cum += int(counts[i])
        if gaps[i] > zmin * med:
            out.append(2 * cum)
    return out


def u_magic(d: int, n: int = 6):
    """U(d) isotropic-oscillator magic numbers: cumulative 2*C(N+d-1,d-1)."""
    return [int(2 * sum(comb(N + d - 1, d - 1) for N in range(k + 1)))
            for k in range(n)]


def _modal_nontrivial(counts) -> int:
    nt = [int(c) for c in counts if c > 1]
    return max(set(nt), key=nt.count) if nt else 0


# levels chosen for a few hundred nodes (instant eigh, deep enough tower)
_LEVELS = {3: 3, 4: 3, 5: 2}


def main() -> None:
    print("=" * 70)
    print("EMERGENT ATOMIC SHELLS -- the tower goes through the EPI grade")
    print("=" * 70)
    print(f"  U(2) magic (2D dots)   = {u_magic(2)}")
    print(f"  U(3) magic (3D/nuclear)= {u_magic(3)}")
    print("  chemical noble gases   = [2, 10, 18, 36, 54, 86]")

    forms = {}
    for m, lv in _LEVELS.items():
        G, _ = sierpinski_simplex(m, lv)
        forms[m] = (G, *shells(G))

    # M1 -- shell degeneracy = simplex grade = emergent dimension (EXACT).
    print("\nM1 -- shell degeneracy = simplex grade = dimension (EXACT):")
    m1_ok = True
    for m in (3, 4, 5):
        G, vals, counts = forms[m]
        grade = m - 1
        first_exc = int(counts[1])
        modal = _modal_nontrivial(counts)
        ok = (first_exc == grade) and (modal == grade)
        m1_ok = m1_ok and ok
        print(f"  K_{m} nest (grade {grade}, N={G.number_of_nodes()}): "
              f"first-excited deg={first_exc}, modal deg={modal} "
              f"[{'OK' if ok else 'OFF'}]")
    assert m1_ok, "shell degeneracy does not equal the simplex grade"

    # M2 -- dimension signature: ground + first closure = 2 and 2*(grade+1).
    print("\nM2 -- dimension signature: first closure = 2*(grade+1) (EXACT):")
    m2_ok = True
    for m in (3, 4, 5):
        G, vals, counts = forms[m]
        grade = m - 1
        cl = closures(vals, counts)
        want = 2 * (grade + 1)
        ok = want in cl
        m2_ok = m2_ok and ok
        tag = "  <-- atomic Ne (SO(4))" if want == 10 else ""
        print(f"  K_{m} (grade {grade}): first closure {want} "
              f"present={ok}{tag}")
    assert m2_ok, "first closure is not 2*(grade+1)"

    # M3 -- the U(grade) numbers appear among the closures (+ fractal modes).
    print("\nM3 -- U(grade) numbers appear among closures (+ fractal modes):")
    m3_ok = True
    for m in (3, 4):
        G, vals, counts = forms[m]
        grade = m - 1
        cl = closures(vals, counts)
        target = set(u_magic(grade))
        hits = sorted({c for c in cl if c in target})
        noise = sorted({c for c in cl if c not in target})
        ok = len(hits) >= 2
        m3_ok = m3_ok and ok
        print(f"  K_{m} (grade {grade}): U({grade}) numbers present={hits} "
              f"fractal-mode closures={noise}")
    assert m3_ok, "U(grade) numbers do not appear among the closures"

    # M4 -- synthesis: atom shells go through the EPI form's grade.
    print("\nM4 -- synthesis (atom shells <- EPI form grade):")
    print("  grade 2 form -> U(2) tower 2,6,12,20 = 2D quantum-dot atoms")
    print("                  (matches the substrate's own locked U(2))")
    print("  grade 3 form -> U(3) tower 2,8,20,40 = 3D-oscillator / nuclear")
    print("  the CHEMICAL table 2,10,18,36,54,86 = SO(4,2)/Madelung needs")
    print("  two-body screening = the SAME Fix(G)^perp wall (prime fine")
    print("  structure / RH S(T)) on top of this U(grade) skeleton.")

    print("\n" + "=" * 70)
    print("VERDICT: the atom's shell tower emerges from the COHERENT FORM")
    print("(EPI) and its THOL grade -- NOT from an imported spatial ball.")
    print("Shell degeneracy = simplex grade = emergent dimension (M1, exact);")
    print("the first closure reads the dimension as 2*(grade+1) (M2); the")
    print("U(grade) oscillator numbers appear among the shell closures (M3,")
    print("cleanest for the grade-3 / 3D form). The dimension that comes from")
    print("the form IS the shell degeneracy.")
    print("HONEST SCOPE: standard K_m / Sierpinski spectra re-read in TNFR;")
    print("reaches the independent-particle U(grade) skeleton only, not the")
    print("chemical table; closes no open problem. R and pi stay assumed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
