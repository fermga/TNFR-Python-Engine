"""Prescribed simplex-gasket construction and finite spectral-dimension estimates.

The recursive constructor selects m corner-glued copies of K_m; it does not
execute THOL or derive its branching/scale ratio. For a corresponding continuum
similarity construction with ratio 1/2, log(m)/log(2) is the comparison dimension.
The spectral-dimension reference 2 log(m)/log(m+2) concerns that infinite family;
finite heat-kernel fits do not prove its limiting theorem.
Kron reduction preserves specified boundary response, not arbitrary form or a
uniquely selected self-similar geometry. K4 has simplex grade three; its associated
similarity dimension two and the auxiliary U(2) sector count two are different
quantities. Their equality does not identify spatial dimension or physical form.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import math
import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --- THOL/U5 self-similar simplex nesting (Sierpinski gasket of K_m) ---
def sierpinski_simplex(m: int, levels: int):
    """Construct m corner-glued copies recursively, returning graph and corners.

    Branching and gluing are supplied; this is not a THOL operator invocation."""
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


def _node_count_recurrence(m: int, levels: int) -> int:
    """N(m,k) = m*N(m,k-1) - C(m,2); N(m,0) = m."""
    n = m
    for _ in range(levels):
        n = m * n - m * (m - 1) // 2
    return n


# --- canonical structural-Laplacian spectrum + spectral dimension ---
def l_sym_eigvals(G) -> np.ndarray:
    """Eigenvalues of the canonical symmetric structural Laplacian
    L_sym = I - D^-1/2 W D^-1/2 (same spectrum as the dNFR EPI-channel
    operator L_rw)."""
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    d = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(d)
    L = np.eye(len(nodes)) - (dinv[:, None] * A * dinv[None, :])
    return np.clip(np.linalg.eigvalsh(L), 0.0, None)


def spectral_dimension(eigvals: np.ndarray) -> float:
    """d_s from the heat-kernel return probability p(t)=Z(t)/n ~
    t^(-d_s/2), central-plateau median log-slope (the ex.134 estimator)."""
    nz = eigvals[eigvals > 1e-9]
    ts = np.logspace(math.log10(1.0 / nz.max()), math.log10(1.0 / nz.min()), 60)
    n = len(eigvals)
    p = np.array([float(np.exp(-eigvals * t).sum()) / n for t in ts])
    slope = np.gradient(np.log(p), np.log(ts))
    k = len(slope)
    return -2.0 * float(np.median(slope[k // 4 : k - k // 4]))


def _canonical_anchor_ok(G) -> bool | None:
    """Anchor: hand-built spectrum equals the canonical engine spectrum."""
    try:
        from tnfr.physics.structural_diffusion import structural_eigenmodes
    except Exception:
        return None
    try:
        out = structural_eigenmodes(G)
        ev = np.asarray(out[0] if isinstance(out, tuple) else out, float)
        mine = np.sort(l_sym_eigvals(G))
        ev = np.sort(ev[: len(mine)])
        return bool(np.allclose(ev, mine, atol=1e-8))
    except Exception:
        return None


def main() -> None:
    print("=" * 70)
    print("PRESCRIBED SIMPLEX-GASKET DIMENSION COMPARISON")
    print("=" * 70)

    # M1 -- the self-similar nesting is well-formed (node=subgraph).
    print("\nM1 -- Explicit self-similar graph construction:")
    m1_ok = True
    for m in (3, 4, 5):
        for lv in range(0, 4):
            G, _ = sierpinski_simplex(m, lv)
            exp = _node_count_recurrence(m, lv)
            m1_ok = m1_ok and (G.number_of_nodes() == exp)
    G3, _ = sierpinski_simplex(3, 3)
    anchor = _canonical_anchor_ok(G3)
    print(f"  node-count recurrence N(m,k)=m*N(m,k-1)-C(m,2): {m1_ok}")
    anchor_label = "UNAVAILABLE" if anchor is None else str(anchor)
    print(f"  L_sym spectrum == canonical structural_eigenmodes: {anchor_label}")
    assert m1_ok, "self-similar construction node counts are wrong"

    # M2 -- DEFINITE similarity dimension, SET BY THE LOCAL GRADE.
    print("\nM2 -- exact similarity dim d = log(m)/log(2) (grade m-1):")
    for m in (3, 4, 5):
        d_sim = math.log(m) / math.log(2.0)
        tag = "  <-- tetrahedron (grade 3 = 3D form) = EXACTLY 2" if m == 4 else ""
        print(f"  K_{m} (grade {m - 1}): d = {d_sim:.4f}{tag}")
    print(
        f"  grade rises -> dim rises: "
        f"{math.log(3)/math.log(2):.3f} < "
        f"{math.log(4)/math.log(2):.3f} < "
        f"{math.log(5)/math.log(2):.3f}"
    )
    assert abs(math.log(4) / math.log(2) - 2.0) < 1e-12

    # M3 -- the FREE spectral d_s becomes DEFINITE (self-similar).
    print("\nM3 -- spectral d_s: FREE graph -> DEFINITE self-similar:")
    m3_ok = True
    for m in (3, 4, 5):
        theory = 2.0 * math.log(m) / math.log(m + 2)
        G, _ = sierpinski_simplex(m, 4)
        d_s = spectral_dimension(l_sym_eigvals(G))
        ok = abs(d_s - theory) < 0.1
        m3_ok = m3_ok and ok
        status = "OK" if ok else "OFF"
        print(
            f"  K_{m} level-4 (N={G.number_of_nodes()}): "
            f"d_s={d_s:.4f} -> self-similar={theory:.4f}  [{status}]"
        )
    # contrast: a random tree has a NON-self-similar (free) d_s
    tree = nx.random_labeled_tree(514, seed=1)
    d_tree = spectral_dimension(l_sym_eigvals(tree))
    print(
        f"  contrast: random tree (N=514) d_s={d_tree:.4f} -- "
        f"a free input, not a self-similar invariant"
    )
    assert m3_ok, "self-similar spectral dimension did not converge"

    # M4 -- reconciliation: tetrahedron fractal dim == U(2) fiber == 2.
    print("\nM4 -- reconciliation (form-grade <-> dim <-> U(2) fiber):")
    d_tet = math.log(4) / math.log(2.0)
    print("  tetrahedron K_4: supplied simplex grade 3")
    print(f"  self-similar (THOL/U5) fractal dim = log4/log2 = {d_tet:.4f}")
    print("  auxiliary complex sector count (U(2) model)     = 2")
    print("  => the grade-3 form, fractally nested, is 2-dimensional --")
    print("     meeting the 2D fiber. (Honest: a numerical convergence on")
    print("     2; the Sierpinski-tetrahedron Hausdorff dim is 2, not a")
    print("     derived identity with the U(2) sector count.)")
    assert abs(d_tet - 2.0) < 1e-12

    print("\n" + "=" * 70)
    print("RESULT SCOPE: the self-similar construction is explicitly selected.")
    print(
        "The branching m and comparison scale ratio 1/2 set its similarity dimension."
    )
    print(
        "The finite graph heat trace supplies a separate estimated spectral dimension."
    )
    print("Neither estimate selects a topology or supplies an autonomous birth law.")
    print("K4 has simplex grade three and comparison similarity dimension two.")
    print("The auxiliary U(2) sector count is a distinct algebraic quantity.")
    print("Equality of those numeric values is not a physical identity.")
    print("This script executes no canonical THOL or U5 certificate.")
    print("The dimension-selection problem remains open.")
    print("=" * 70)


if __name__ == "__main__":
    main()
