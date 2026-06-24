"""
benchmarks/composition_arithmetic.py

Composition arithmetic — does the OPERATION (+ and ×) emerge from coupling
coherent systems, instead of being injected by hand?

This is the forward edge of the emergent-number programme. Two earlier harnesses
established:
  - emergent_integers_symmetry.py  : geometry -> integers OUT (cardinals = irrep dims)
  - inverse_spectrum_to_symmetry.py: partial spectrum -> group -> predict a hidden cardinal
Both PRODUCE cardinals but never the arithmetic operation itself. The primality
module (primality-test/tnfr_primality) goes the other way: it CONSUMES divisibility
(trial division n % i) to re-read primality as the equilibrium condition ΔNFR = 0.

The open frontier is exactly: can the additive/multiplicative COMPOSITION of
integers itself arise from composing systems, with no arithmetic put in by hand?

ENGINE (known theorems — the independent ground truth):
  - Cartesian product  G □ H : Laplacian eigenvalues = {λ_i + μ_j}   -> ADDITION
  - Tensor   product   G × H : adjacency eigenvalues = {α_i · β_j}   -> MULTIPLICATION
  - Aut(G) × Aut(H) acts on the product; product irreps are tensor products,
    dim(ρ ⊗ σ) = dim ρ · dim σ                                       -> CARDINALS multiply

TNFR reading (AGENTS.md): L = D − A is the discrete ΔNFR / phase-curvature operator.
Coupling two coherent systems is a physical act, and the spectrum of the composite
realises + and × with no arithmetic supplied externally. This connects to the
B0★-α canonical graph-product programme (Q1 = G □ G, Q2 = G × G) in AGENTS.md.

HONEST SCOPE:
  This PRODUCES the additive and multiplicative composition of spectra, and the
  multiplication of degeneracy cardinals. It does NOT make arithmetic primality
  equal to representational irreducibility. We exhibit a COMPOSITE integer (4) that
  is the dimension of an IRREDUCIBLE coherent mode (K5 / S5): "physically
  indivisible" is NOT "arithmetically prime". And the SAME cardinal 4 is of
  compositional origin (2 + 2 -> multiplicity 2·2) in a product system (K3 □ K3)
  yet atomic in an indivisible one (K5). Whether a cardinal "factorises" is a
  property of the SYSTEM's symmetry, not of the integer — in contrast with the
  unique-factorisation theorem. The frontier is mapped, not erased.

Run:
    python benchmarks/composition_arithmetic.py

Status: RESEARCH (composition-arithmetic falsifier).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher


# --------------------------------------------------------------------------- #
# Spectra (L = D - A is the discrete ΔNFR operator; A is the coupling matrix)
# --------------------------------------------------------------------------- #
def lap_spectrum(G, nodes=None):
    """Sorted eigenvalues of the Laplacian L = D - A."""
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
    print("ADDITION emerges from the Cartesian product (Laplacian spectrum)")
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
        "-- '+' is read off the composite, not supplied"
    )
    return all_ok


def test_multiplication():
    print()
    print("=" * 78)
    print("MULTIPLICATION emerges from the tensor product (adjacency spectrum)")
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
        "-- 'x' is read off the composite, not supplied"
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
        "-- 2 x 2 = 4 realised by coupling, no 'x' put in"
    )
    return has_4


def test_irreducibility_is_not_primality():
    print()
    print("=" * 78)
    print("HONEST FRONTIER: irreducibility (physics) is NOT primality (arithmetic)")
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
    print("  So the cardinal 4 is atomic in K5 (simple group S5) and compositional")
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
    print(f"  '+' emerges (Cartesian product)         : {'PASS' if r1 else 'FAIL'}")
    print(f"  'x' emerges (tensor product)            : {'PASS' if r2 else 'FAIL'}")
    print(f"  cardinals multiply (2 x 2 = 4)          : {'PASS' if r3 else 'FAIL'}")
    print(f"  irreducibility != primality (frontier)  : {'PASS' if r4 else 'FAIL'}")
    overall = all([r1, r2, r3, r4])
    print(f"\n  OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    print()
    print("  Reading: the additive and multiplicative COMPOSITION of integers")
    print("  emerges from coupling coherent systems (no arithmetic injected) -- '+'")
    print("  from the Cartesian product, 'x' from the tensor product, and cardinals")
    print("  multiply. But representational irreducibility does NOT reproduce")
    print("  arithmetic primality: the same cardinal factorises or not depending on")
    print("  the system's symmetry. Coupling PRODUCES (+, x) on spectra and cardinals;")
    print("  the unique factorisation of integers remains a separate, stronger fact.")


if __name__ == "__main__":
    main()
