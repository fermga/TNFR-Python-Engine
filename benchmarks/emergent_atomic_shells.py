"""Finite spectral-shell comparisons on directly constructed simplex gaskets.

The constructor glues selected K_m copies, without THOL, U5 admission or nodal
evolution. Eigenvalues of D-A are grouped after rounding to six decimals.
M1 checks first-excited and modal nontrivial multiplicity, not every shell.
M2 tests membership of 2*(grade+1) among configured closure labels, not its
position as the first closure. M3 requires at least two oscillator-label hits.
The gap cutoff and factor-two occupancy convention are selected inputs.

These finite comparisons derive neither atoms, spin, a chemical table,
physical dimension, canonical THOL growth nor a Riemann correspondence.
Simplex grade, spectral multiplicity, fractal dimension and U(2) sector count
are distinct objects. The historical Ne display tag is a numerical analogy.
Scope: theory/EMERGENT_ONTOLOGY.md sections 3.2 and 9.1.
Run: python benchmarks/emergent_atomic_shells.py
Status: RESEARCH (finite auxiliary spectral comparison).
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


# --- Directly constructed simplex gasket; no THOL execution ---
def sierpinski_simplex(m: int, levels: int):
    """Construct the selected corner-glued simplex gasket recursively.
    This graph construction supplies no canonical operator trace."""
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


# --- shells = degenerate levels of the combinatorial L = D - A (used for its
# additive Cartesian-product spectrum; the separate nodal transport operator is
# L_rw = I - D^-1 W) ---
def shells(G, tol: int = 6):
    """Group numerical eigenvalues after rounding to tol decimal places.
    A grouped eigenspace need not be one irreducible representation."""
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
    return [
        int(2 * sum(comb(N + d - 1, d - 1) for N in range(k + 1))) for k in range(n)
    ]


def _modal_nontrivial(counts) -> int:
    nt = [int(c) for c in counts if c > 1]
    return max(set(nt), key=nt.count) if nt else 0


# levels chosen for a few hundred nodes (instant eigh, deep enough tower)
_LEVELS = {3: 3, 4: 3, 5: 2}


def main() -> None:
    print("=" * 70)
    print("SIMPLEX GASKET SHELLS -- finite configured spectral comparison")
    print("=" * 70)
    print(f"  U(2) comparison labels   = {u_magic(2)}")
    print(f"  U(3) comparison labels= {u_magic(3)}")
    print("  historical atom labels = [2, 10, 18, 36, 54, 86]")

    forms = {}
    for m, lv in _LEVELS.items():
        G, _ = sierpinski_simplex(m, lv)
        forms[m] = (G, *shells(G))

    # M1 -- first and modal grouped multiplicities of the selected family.
    print("\nM1 -- first and modal grouped multiplicities versus selected grade:")
    m1_ok = True
    for m in (3, 4, 5):
        G, vals, counts = forms[m]
        grade = m - 1
        first_exc = int(counts[1])
        modal = _modal_nontrivial(counts)
        ok = (first_exc == grade) and (modal == grade)
        m1_ok = m1_ok and ok
        print(
            f"  K_{m} nest (grade {grade}, N={G.number_of_nodes()}): "
            f"first-excited deg={first_exc}, modal deg={modal} "
            f"[{'OK' if ok else 'OFF'}]"
        )
    assert m1_ok, "shell degeneracy does not equal the simplex grade"

    # M2 -- membership of 2*(grade+1) in the configured closure labels.
    print("\nM2 -- presence of 2*(grade+1) among configured closure labels:")
    m2_ok = True
    for m in (3, 4, 5):
        G, vals, counts = forms[m]
        grade = m - 1
        cl = closures(vals, counts)
        want = 2 * (grade + 1)
        ok = want in cl
        m2_ok = m2_ok and ok
        tag = "  <-- atomic Ne (SO(4))" if want == 10 else ""
        print(
            f"  K_{m} (grade {grade}): comparison closure {want} " f"present={ok}{tag}"
        )
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
        print(
            f"  K_{m} (grade {grade}): U({grade}) numbers present={hits} "
            f"fractal-mode closures={noise}"
        )
    assert m3_ok, "U(grade) numbers do not appear among the closures"

    # M4 -- scope of the finite numerical shell-label comparison.
    print("\nM4 -- interpretation boundary:")
    print("  grade 2 comparison: selected U(2) labels 2,6,12,20")
    print("                  (not an identity with auxiliary substrate sector count)")
    print("  grade 3 comparison: selected U(3) labels 2,8,20,40")
    print("  No chemical table, atomic state map or occupancy law is derived.")
    print("  Screening and arithmetic questions have distinct mathematical")
    print("  inputs; this calculation establishes no shared obstruction.")

    print("\n" + "=" * 70)
    print("RESULT: finite numerical spectra of configured simplex gaskets.")
    print("Graphs are constructed directly, without THOL or nodal evolution.")
    print("M1 checks first and modal multiplicities, not every shell.")
    print("M2 checks one closure-label membership under a selected cutoff.")
    print("M3 checks at least two oscillator-label coincidences; it does")
    print("not establish an exclusive oscillator tower or physical atoms.")
    print("Graph grade and physical or spectral dimension remain distinct.")
    print("SCOPE: graph spectra with selected rounding and gap policies;")
    print("the factor-two occupancy convention is an input, not derived spin.")
    print("No chemical, particle or open mathematical problem is resolved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
