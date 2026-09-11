"""
benchmarks/bridge_primes_riemann.py

The bridge: does the prime structure of Z link to the TNFR-Riemann program?

This harness connects two threads that, so far, ran in parallel:
  (A) composition_arithmetic.py -- the additive/multiplicative composition of
      integers emerges from coupling coherent systems (graph products), and a
      cardinal "factorises" or not depending on the SYSTEM's symmetry group.
  (B) the TNFR-Riemann program -- the discrete prime-ladder Hamiltonian P14
      reproduces -zeta'/zeta exactly, yet G4 = RH stays open; the residual
      obstruction is the oscillatory term S(T) = (1/pi)*arg zeta(1/2 + iT),
      which the Euler-Orthogonality Lemma (research notes section 13vicies-
      novies.11) pins inside Fix(S_n)^perp.

CLAIM UNDER TEST: the two threads meet at ONE structural object -- the
prime-relabelling symmetry S_n -- through ONE shared machinery -- graph products.

The canonical prime-ladder graph (build_prime_ladder_graph) is literally n
disjoint identical copies of a path P_K, one ladder per prime, with NO edges
between distinct primes (Euler-product orthogonality enforced at graph level).
Hence:
  * permuting the primes is a graph automorphism: S_n is a subgroup of Aut(G);
  * the graph (Laplacian / adjacency) cannot tell primes apart -- its spectral
    degeneracies are cardinals (= n) carrying the permutation rep of S_n;
  * the individual primes enter ONLY through the diagonal label
    nu_f = k*log(p) (the von Mangoldt weight), which breaks S_n by hand.

So the SAME S_n that decides "factorises or not" in (A) is the obstruction that,
in (B), traps every catalog construction in Fix(S_n) and leaves the
RH-equivalent oscillatory residue S(T) in Fix(S_n)^perp unreachable.

HONEST SCOPE:
  This MAPS the link; it does NOT close G4 = RH. It shows (i) the prime-ladder
  graph is S_n-symmetric, (ii) its spectral degeneracies are reducible cardinals
  under S_n (<chi,chi> = 2 = trivial + standard, NOT irreducible), (iii) the prime
  content lives only in the consumed diagonal label k*log(p), (iv) graph products
  multiply the cardinals (the operation emerges, research notes Q1/Q2 of B0*-alpha)
  yet preserve S_n x S_n equivariance -- so the product route cannot encode the
  fine prime distribution either. The prime structure of Z is INPUT on the
  fine-grained side of BOTH threads; it is not derived from pure dynamics.

Run:
    python benchmarks/bridge_primes_riemann.py

Status: RESEARCH (A<->B bridge falsifier; shared obstruction = S_n).
"""

from __future__ import annotations

import itertools
import os
import sys

import networkx as nx
import numpy as np

# Reuse the composition-arithmetic engine (same benchmarks/ directory). When run
# as `python benchmarks/bridge_primes_riemann.py`, this file's directory is on
# sys.path; the explicit insert keeps the import robust under other launchers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composition_arithmetic import (  # noqa: E402
    character_norm,
    eigenspaces,
    lap_spectrum,
)

from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def laplacian(G, nodes):
    """Laplacian L = D - A in the fixed node order `nodes`."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    return np.diag(A.sum(axis=1)) - A


def nu_f_diagonal(G, nodes):
    """Diagonal von Mangoldt label diag(nu_f) = diag(k*log p) in order `nodes`."""
    return np.diag([float(G.nodes[node]["nu_f"]) for node in nodes])


def commutator_norm(X, M):
    """Frobenius norm of [X, M] = X M - M X."""
    return float(np.linalg.norm(X @ M - M @ X))


def prime_permutation_matrices(nodes, primes):
    """Permutation matrices of S_n acting by prime relabelling (p_i,k)->(p_sigma(i),k)."""
    index = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    mats = []
    for perm in itertools.permutations(range(len(primes))):
        relabel = {primes[i]: primes[perm[i]] for i in range(len(primes))}
        M = np.zeros((n_nodes, n_nodes))
        for node in nodes:
            p, k = node
            dst = (relabel[p], k)
            M[index[dst], index[node]] = 1.0
        mats.append(M)
    return mats


def product_prime_permutation_matrices(prod_nodes, primes):
    """Matrices of S_n x S_n on a Cartesian product, relabelling each factor's prime."""
    index = {node: i for i, node in enumerate(prod_nodes)}
    n_nodes = len(prod_nodes)
    perms = list(itertools.permutations(range(len(primes))))
    mats = []
    for pa in perms:
        rel_a = {primes[i]: primes[pa[i]] for i in range(len(primes))}
        for pb in perms:
            rel_b = {primes[i]: primes[pb[i]] for i in range(len(primes))}
            M = np.zeros((n_nodes, n_nodes))
            for node in prod_nodes:
                (p1, k1), (p2, k2) = node
                dst = ((rel_a[p1], k1), (rel_b[p2], k2))
                M[index[dst], index[node]] = 1.0
            mats.append(M)
    return mats


# --------------------------------------------------------------------------- #
# Test 1 -- the prime-ladder graph is S_n-symmetric (primes are interchangeable)
# --------------------------------------------------------------------------- #
def test_graph_is_sn_symmetric():
    print("=" * 78)
    print("(1) The prime-ladder graph is S_n-symmetric: primes are interchangeable")
    print("=" * 78)
    primes = [2, 3, 5, 7]
    K = 4
    G = build_prime_ladder_graph(len(primes), max_power=K, primes=primes)
    nodes = list(G.nodes())
    L = laplacian(G, nodes)
    mats = prime_permutation_matrices(nodes, primes)
    max_comm = max(commutator_norm(L, M) for M in mats)

    spec_G = lap_spectrum(G)
    spec_PK = lap_spectrum(nx.path_graph(K))
    replicated = np.sort(np.concatenate([spec_PK] * len(primes)))
    spectra_match = bool(np.allclose(spec_G, replicated, atol=1e-8))

    ok = max_comm < 1e-9 and spectra_match
    print(f"  primes = {primes}, ladder length K = {K}  ->  {len(nodes)} nodes")
    print(
        f"  max ||[L, P_sigma]|| over S_{len(primes)} = {max_comm:.2e}  (0 => L is S_n-invariant)"
    )
    print(f"  spec(L_G) == spec(L_P{K}) replicated {len(primes)}x ? {spectra_match}")
    print(f"      spec(L_P{K})  = {np.round(spec_PK, 3)}")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the GRAPH cannot tell primes apart"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# Test 2 -- spectral degeneracies are cardinals equal to the number of primes
# --------------------------------------------------------------------------- #
def test_degeneracies_are_cardinals():
    print("=" * 78)
    print("(2) Spectral degeneracies are cardinals = number of primes (n)")
    print("=" * 78)
    primes = [2, 3, 5, 7]
    K = 4
    G = build_prime_ladder_graph(len(primes), max_power=K, primes=primes)
    nodes = list(G.nodes())
    spaces = eigenspaces(G, nodes)
    n = len(primes)
    all_n = True
    for val, mult, _ in spaces:
        flag = "<- = n_primes" if mult == n else "<- UNEXPECTED"
        all_n &= mult == n
        print(f"      lambda = {val:6.3f}   multiplicity = {mult}  {flag}")
    print(f"  every Laplacian level has multiplicity exactly n = {n}")
    print(
        f"  VERDICT: {'PASS' if all_n else 'FAIL'} -- the cardinal n is read off, "
        "not supplied"
    )
    print()
    return all_n


# --------------------------------------------------------------------------- #
# Test 3 -- the degeneracy-n carries the REDUCIBLE permutation rep of S_n
# --------------------------------------------------------------------------- #
def test_prime_degeneracy_is_reducible():
    print("=" * 78)
    print("(3) The prime degeneracy is REDUCIBLE under S_n (trivial + standard)")
    print("=" * 78)
    primes = [2, 3, 5, 7]
    K = 4
    G = build_prime_ladder_graph(len(primes), max_power=K, primes=primes)
    nodes = list(G.nodes())
    mats = prime_permutation_matrices(nodes, primes)
    order = len(mats)  # |S_n| = n!
    spaces = eigenspaces(G, nodes)
    # Permutation rep of S_n on n points = trivial (+) standard => <chi,chi> = 2.
    all_reducible = True
    for val, mult, proj in spaces:
        chi = character_norm(proj, mats, order)
        reducible = abs(chi - 2.0) < 1e-6
        all_reducible &= reducible
        tag = "REDUCIBLE (1 + (n-1))" if reducible else "?"
        print(f"      lambda = {val:6.3f}  mult = {mult}  <chi,chi> = {chi:.3f}  {tag}")
    print("  Contrast: in composition_arithmetic.py the K5 / S5 dim-4 mode gives")
    print("  <chi,chi> = 1 (IRREDUCIBLE). Here every prime level gives 2: the part")
    print("  that would distinguish individual primes (the standard irrep) is present")
    print("  but COUPLED to the trivial mode -- nothing in the GRAPH separates them.")
    print(
        f"  VERDICT: {'PASS' if all_reducible else 'FAIL'} -- primes are not "
        "individuated by the dynamics"
    )
    print()
    return all_reducible


# --------------------------------------------------------------------------- #
# Test 4 -- the prime content lives ONLY in the diagonal von Mangoldt label
# --------------------------------------------------------------------------- #
def test_prime_content_is_diagonal_input():
    print("=" * 78)
    print("(4) The Riemann content lives ONLY in the diagonal label nu_f = k*log p")
    print("=" * 78)
    primes = [2, 3, 5, 7]
    K = 4
    G = build_prime_ladder_graph(len(primes), max_power=K, primes=primes)
    nodes = list(G.nodes())
    L = laplacian(G, nodes)
    D = nu_f_diagonal(G, nodes)
    mats = prime_permutation_matrices(nodes, primes)
    max_comm_L = max(commutator_norm(L, M) for M in mats)
    max_comm_D = max(commutator_norm(D, M) for M in mats)
    ok = max_comm_L < 1e-9 and max_comm_D > 1e-3
    print(
        f"  max ||[L, P_sigma]||         = {max_comm_L:.2e}   (graph commutes with S_n)"
    )
    print(
        f"  max ||[diag(nu_f), P_sigma]|| = {max_comm_D:.2e}   (label BREAKS S_n by hand)"
    )
    print("  The values {k*log p} -- the entire Euler-product / von Mangoldt content")
    print("  that P14 feeds into -zeta'/zeta -- sit in the diagonal label, CONSUMED as")
    print("  input. The S_n-invariant graph dynamics carries none of it. This is")
    print("  exactly the Euler-Orthogonality Lemma (13vicies-novies.11).")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- prime structure is consumed, "
        "not generated"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# Test 5 -- products multiply cardinals yet preserve S_n x S_n equivariance
# --------------------------------------------------------------------------- #
def test_product_multiplies_but_preserves_symmetry():
    print("=" * 78)
    print("(5) Graph products multiply cardinals BUT preserve S_n x S_n equivariance")
    print("=" * 78)
    primes = [2, 3]
    K = 3
    G = build_prime_ladder_graph(len(primes), max_power=K, primes=primes)
    n = len(primes)

    # Cardinal at lambda = 0 of G == number of components == n (sum the
    # multiplicity of every lambda~0 level, not the count of levels).
    mult_G = sum(m for v, m, _ in eigenspaces(G, list(G.nodes())) if abs(v) < 1e-9)

    Q1 = nx.cartesian_product(G, G)
    prod_nodes = list(Q1.nodes())
    mult_Q1 = sum(m for v, m, _ in eigenspaces(Q1, prod_nodes) if abs(v) < 1e-9)
    cardinals_multiply = mult_Q1 == mult_G * mult_G  # n^2

    L_Q1 = laplacian(Q1, prod_nodes)
    prod_mats = product_prime_permutation_matrices(prod_nodes, primes)
    max_comm = max(commutator_norm(L_Q1, M) for M in prod_mats)
    equivariance_preserved = max_comm < 1e-9

    ok = cardinals_multiply and equivariance_preserved
    print(f"  lambda=0 multiplicity in G        = {mult_G}   (= n)")
    print(f"  lambda=0 multiplicity in G [] G   = {mult_Q1}   (= n^2 = {n}x{n})")
    print(f"  cardinals multiply (n x n)?         {cardinals_multiply}")
    print(f"  max ||[L_(G[]G), P_sigma (x) P_tau]|| over S_n x S_n = {max_comm:.2e}")
    print(f"  S_n x S_n equivariance preserved?   {equivariance_preserved}")
    print("  The product PRODUCES x on cardinals (operation emerges, like Q1/Q2 of")
    print("  B0*-alpha) yet commutes with prime relabelling on BOTH factors -- the")
    print("  Canonical Product Equivariance Lemma. So the product route still cannot")
    print("  break S_n, hence cannot reach the fine prime distribution / S(T).")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- operation emerges, obstruction "
        "persists"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
def main():
    print(__doc__)
    r1 = test_graph_is_sn_symmetric()
    r2 = test_degeneracies_are_cardinals()
    r3 = test_prime_degeneracy_is_reducible()
    r4 = test_prime_content_is_diagonal_input()
    r5 = test_product_multiplies_but_preserves_symmetry()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"  (1) prime-ladder graph is S_n-symmetric        : {'PASS' if r1 else 'FAIL'}"
    )
    print(
        f"  (2) degeneracies are cardinals (= n_primes)    : {'PASS' if r2 else 'FAIL'}"
    )
    print(
        f"  (3) prime degeneracy reducible under S_n       : {'PASS' if r3 else 'FAIL'}"
    )
    print(
        f"  (4) prime content is diagonal von Mangoldt input: {'PASS' if r4 else 'FAIL'}"
    )
    print(
        f"  (5) products multiply yet keep S_n x S_n        : {'PASS' if r5 else 'FAIL'}"
    )
    overall = all([r1, r2, r3, r4, r5])
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    print()
    print("  Reading: YES, the prime structure of Z links to the TNFR-Riemann")
    print("  program -- through the prime-relabelling symmetry S_n and the shared")
    print("  graph-product machinery. The SAME S_n that, in composition_arithmetic.py,")
    print("  decides whether a cardinal 'factorises' is the obstruction that, in the")
    print("  Riemann program (Euler-Orthogonality Lemma), traps every catalog")
    print("  construction in Fix(S_n). The individual primes enter only as the")
    print("  consumed diagonal label k*log p (von Mangoldt) on the fine-grained side")
    print("  of BOTH threads. The link is real and structural; it does NOT close")
    print("  G4 = RH: the RH-equivalent residue S(T) lives in Fix(S_n)^perp, exactly")
    print("  where S_n-invariant dynamics cannot reach.")


if __name__ == "__main__":
    main()
