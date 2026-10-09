"""Finite graph-product identities and representation-dimension controls.

For explicitly supplied graphs, the combinatorial Laplacian of the Cartesian
product has pairwise-sum eigenvalues, while tensor-product adjacency has
pairwise-product eigenvalues. These familiar identities concern different
matrices; random-walk normalization does not generally preserve Cartesian
additivity. Constructing a product is an input, not an executed TNFR Coupling
law or a derivation of arithmetic operations from nodal dynamics.

The representation comparison retains a useful negative control: the
four-dimensional standard representation of S5 on K5 is irreducible although
four is composite. A product of two two-dimensional representations also has
dimension four. Arithmetic primality and representation irreducibility are
therefore different properties. Shared spectral/character helpers are imported
by other finite controls; this module does not open a research campaign.
See theory/TNFR_NUMBER_THEORY.md and
theory/TNFR_STRUCTURAL_OBSERVABILITY.md#6-limits-beyond-linear-symmetry.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher


# --------------------------------------------------------------------------- #
# Spectra. L = D - A is the COMBINATORIAL graph Laplacian, whose product spectra
# has additive Cartesian-product spectra; tensor-product adjacency has
# multiplicative spectra. The declared pure-EPI model uses L_rw instead.
# --------------------------------------------------------------------------- #
def lap_spectrum(G, nodes=None):
    """Sorted eigenvalues of the combinatorial Laplacian L = D - A."""
    A = nx.to_numpy_array(G, nodelist=nodes if nodes else list(G.nodes()))
    L = np.diag(A.sum(axis=1)) - A
    return np.sort(np.linalg.eigvalsh(L))


def adj_spectrum(G, nodes=None):
    """Sorted eigenvalues of the adjacency matrix A."""
    A = nx.to_numpy_array(G, nodelist=nodes if nodes else list(G.nodes()))
    return np.sort(np.linalg.eigvalsh(A))


def multiset_close(a, b, tol=1e-8):
    """True if two float multisets coincide (as sorted sequences) within tol."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol))


def outer_sum(x, y):
    """All pairwise sums {x_i + y_j}."""
    return np.array([xi + yj for xi in x for yj in y])


def outer_prod(x, y):
    """All pairwise products {x_i * y_j}."""
    return np.array([xi * yj for xi in x for yj in y])


# --------------------------------------------------------------------------- #
# Emergence of + and ×
# --------------------------------------------------------------------------- #
def cartesian_addition_emerges(G, H, tol=1e-8):
    """Laplacian spectrum of G □ H equals the outer SUM of the factor spectra."""
    specG = lap_spectrum(G)
    specH = lap_spectrum(H)
    spec_prod = lap_spectrum(nx.cartesian_product(G, H))
    return multiset_close(spec_prod, outer_sum(specG, specH), tol), specG, specH


def tensor_multiplication_emerges(G, H, tol=1e-8):
    """Adjacency spectrum of G × H equals the outer PRODUCT of the factor spectra."""
    specG = adj_spectrum(G)
    specH = adj_spectrum(H)
    spec_prod = adj_spectrum(nx.tensor_product(G, H))
    return multiset_close(spec_prod, outer_prod(specG, specH), tol), specG, specH


# --------------------------------------------------------------------------- #
# Character irreducibility (reused engine: <χ,χ> over Aut(G))
# --------------------------------------------------------------------------- #
def automorphism_matrices(G, nodes, limit=20000):
    """Permutation matrices of Aut(G), in the fixed node order `nodes`."""
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    mats = []
    for k, mapping in enumerate(GraphMatcher(G, G).isomorphisms_iter()):
        if k >= limit:
            break
        M = np.zeros((n, n))
        for src, dst in mapping.items():
            M[index[dst], index[src]] = 1.0
        mats.append(M)
    return mats


def eigenspaces(G, nodes, tol=1e-6):
    """Return [(eigenvalue, multiplicity, projector)] of the Laplacian."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    vals, vecs = np.linalg.eigh(L)
    groups = []
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and abs(vals[j] - vals[i]) < tol:
            j += 1
        U = vecs[:, i:j]
        groups.append((float(np.mean(vals[i:j])), j - i, U @ U.T))
        i = j
    return groups


def character_norm(P, mats, order):
    """<χ,χ> = (1/|Aut|) Σ_g trace(P·M_g)^2 ; ≈1 irreducible, k>1 reducible."""
    return sum(float(np.trace(P @ M)) ** 2 for M in mats) / order


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_addition():
    print("=" * 78)
    print("Pairwise spectral addition for a supplied Cartesian product")
    print("=" * 78)
    cases = [
        ("C4", nx.cycle_graph(4), "C5", nx.cycle_graph(5)),
        ("K3", nx.complete_graph(3), "P3", nx.path_graph(3)),
        ("K3", nx.complete_graph(3), "K3", nx.complete_graph(3)),
    ]
    all_ok = True
    for nG, G, nH, H in cases:
        ok, sG, sH = cartesian_addition_emerges(G, H)
        all_ok &= ok
        print(f"  {nG} [] {nH}: spec(L) == {{lambda_i + mu_j}} ? {ok}")
        print(f"      spec({nG}) = {np.round(sG, 3)}    spec({nH}) = {np.round(sH, 3)}")
    print(
        f"  VERDICT: {'PASS' if all_ok else 'FAIL'} "
        "-- the declared graph product satisfies the spectral sum identity"
    )
    return all_ok


def test_multiplication():
    print()
    print("=" * 78)
    print("Pairwise spectral multiplication for supplied tensor-product adjacency")
    print("=" * 78)
    cases = [
        ("K3", nx.complete_graph(3), "K3", nx.complete_graph(3)),
        ("K3", nx.complete_graph(3), "C5", nx.cycle_graph(5)),
        ("K4", nx.complete_graph(4), "K3", nx.complete_graph(3)),
    ]
    all_ok = True
    for nG, G, nH, H in cases:
        ok, sG, sH = tensor_multiplication_emerges(G, H)
        all_ok &= ok
        print(f"  {nG} x {nH}: spec(A) == {{alpha_i * beta_j}} ? {ok}")
        print(f"      spec({nG}) = {np.round(sG, 3)}    spec({nH}) = {np.round(sH, 3)}")
    print(
        f"  VERDICT: {'PASS' if all_ok else 'FAIL'} "
        "-- the declared tensor product satisfies the spectral product identity"
    )
    return all_ok


def test_cardinals_multiply():
    print()
    print("=" * 78)
    print("CARDINALS multiply: two 2-fold modes compose into a 4-fold mode")
    print("=" * 78)
    # K3 has Laplacian spectrum {0, 3, 3}: the 3 is a 2D irrep of S3.
    G = nx.complete_graph(3)
    print(
        f"  K3 Laplacian spectrum    = {np.round(lap_spectrum(G), 3)}  "
        "(degeneracy 2 at lambda=3)"
    )
    prod = nx.cartesian_product(G, G)
    groups = eigenspaces(prod, list(prod.nodes()))
    print("  K3 [] K3 Laplacian levels:")
    for val, mult, _ in groups:
        tag = ""
        if abs(val - 6.0) < 1e-6:
            tag = (
                "  <- 6 = 3+3 : multiplicity 2*2 = 4 (PRODUCT of the two 2-fold modes)"
            )
        elif abs(val - 3.0) < 1e-6:
            tag = "  <- 3 = 0+3 & 3+0 : accidental sum, NOT a product"
        print(f"      lambda = {val:5.2f}   multiplicity = {mult}{tag}")
    has_4 = any(abs(v - 6.0) < 1e-6 and m == 4 for v, m, _ in groups)
    print(
        f"  VERDICT: {'PASS' if has_4 else 'FAIL'} "
        "-- the supplied Cartesian product contains the expected tensor eigenspace"
    )
    return has_4


def test_irreducibility_is_not_primality():
    print()
    print("=" * 78)
    print("Representation irreducibility and arithmetic primality differ")
    print("=" * 78)
    # K5: Aut = S5, Laplacian {0, 5,5,5,5}. The 4-fold mode is the standard irrep
    # of S5, which is IRREDUCIBLE — yet 4 = 2 x 2 arithmetically.
    K5 = nx.complete_graph(5)
    nodes5 = list(K5.nodes())
    mats5 = automorphism_matrices(K5, nodes5)
    order5 = len(mats5)
    print(f"  K5: |Aut| = {order5} (expected 5! = 120)")
    four_irreducible = False
    for val, mult, P in eigenspaces(K5, nodes5):
        chi = character_norm(P, mats5, order5)
        tag = ""
        if mult == 4:
            four_irreducible = abs(chi - 1.0) < 0.4
            tag = "  <- dim 4 is COMPOSITE (2*2) yet IRREDUCIBLE (atomic mode)"
        print(f"      lambda = {val:5.2f}  mult = {mult}  <chi,chi> = {chi:4.1f}{tag}")
    print()
    print(
        "  Meanwhile (test above) K3 [] K3 produced a 4-fold mode at lambda = 6 = 3+3"
    )
    print("  whose multiplicity is exactly 2*2 -- a 4 of COMPOSITIONAL origin.")
    print(
        "  So the cardinal 4 is irreducible in K5 (symmetric group S5) and compositional"
    )
    print("  in K3 [] K3 (product group): whether it 'factorises' depends on the")
    print("  SYSTEM's symmetry, not on the integer. Arithmetic unique factorisation")
    print("  is a strictly stronger structure than representational composition.")
    print(
        f"  VERDICT: {'PASS' if four_irreducible else 'FAIL'} "
        "-- 'prime <=> irreducible' is correctly REFUTED"
    )
    return four_irreducible


def main():
    print(__doc__)
    r1 = test_addition()
    r2 = test_multiplication()
    r3 = test_cardinals_multiply()
    r4 = test_irreducibility_is_not_primality()
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Cartesian-product sum identity         : {'PASS' if r1 else 'FAIL'}")
    print(f"  Tensor-product spectral identity            : {'PASS' if r2 else 'FAIL'}")
    print(f"  cardinals multiply (2 x 2 = 4)          : {'PASS' if r3 else 'FAIL'}")
    print(f"  irreducibility != primality (frontier)  : {'PASS' if r4 else 'FAIL'}")
    overall = all([r1, r2, r3, r4])
    print(f"\n  OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    print()
    print("  These are finite checks of graph-product spectral identities.")
    print("  The graph constructions are inputs; no Coupling event is executed.")
    print("  The representation control separates irreducibility from primality.")


if __name__ == "__main__":
    main()
