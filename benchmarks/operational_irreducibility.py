"""
benchmarks/operational_irreducibility.py

Camino 2 — operational irreducibility: when does a degeneracy FACTORISE?

This is the mirror image of composition_arithmetic.py (Camino 1). There we found a
COMPOSITE integer (4 = 2x2) realised as the dimension of an IRREDUCIBLE coherent
mode (the standard rep of S5 on K5): "arithmetically composite" did NOT imply
"operationally divisible". Here we close the logical square from the other side:
a PRIME integer (5) realised as an OPERATIONALLY COMPOSITE degeneracy that splits
3 + 2 under a symmetry-preserving perturbation.

OPERATIONAL IRREDUCIBILITY (the definition):
  Fix a coherent system with symmetry group G (its operator commutes with rep(G)).
  A degenerate level is OPERATIONALLY IRREDUCIBLE iff its eigenspace carries a
  single irreducible representation of G. Equivalently (Schur's lemma): NO
  G-equivariant perturbation can lift the degeneracy -- the level is rigid.
  A level is OPERATIONALLY COMPOSITE iff its eigenspace is reducible: it then
  splits into smaller irreducible blocks under a generic symmetry-preserving
  perturbation. The measurement is exactly Camino 1's <chi,chi>:
      <chi,chi> = 1  -> irreducible  (operationally prime, rigid)
      <chi,chi> = k  -> reducible    (operationally composite, splittable)

ENGINE (known theorems — the independent ground truth):
  - L = D - A commutes with every automorphism; eigenspaces carry reps of Aut(G).
  - Schur's lemma: a G-equivariant operator acts as a scalar on each irrep, so it
    cannot split an irreducible eigenspace; it can only SHIFT it.
  - Wigner-von Neumann non-crossing rule: two levels of the SAME irrep repel
    (avoided crossing); two levels of DIFFERENT irreps may CROSS along a
    symmetry-preserving 1-parameter family -> an accidental, reducible degeneracy
    that splits immediately off the crossing point.

CONSTRUCTION:
  * Operationally COMPOSITE 5: the octahedron = Johnson graph J(4,2) on the 6
    two-subsets of {0,1,2,3}. Under S4 the 6-space decomposes 1 (+) 2 (+) 3.
    The S4-invariant family M(t) = A_octa + t*A_match sends the 3-dim (standard)
    level to -t and the 2-dim (E) level to t-2; they CROSS at t=1, giving a
    5-fold degeneracy that is reducible (<chi,chi>=2) and splits back into an
    irreducible 3 and an irreducible 2 for t != 1.
  * Operationally PRIME 5: K6 with Aut = S6. The Laplacian level lambda=6 has
    multiplicity 5 = the standard irrep of S6 (irreducible, <chi,chi>=1). The full
    S6-commutant is <I, J>, which acts as a scalar on the 5-space: NO
    symmetry-preserving perturbation can split it.

TNFR reading (AGENTS.md): L = D - A is the discrete ΔNFR / phase-curvature
operator. "Factorising a degeneracy" is a structural act of the coupled system,
and whether the cardinal 5 factors is decided by the system's symmetry, not by the
integer. This is the same S_n machinery as bridge_primes_riemann.py: test (4)
shows the n=5 prime ladder makes 5 operationally composite (1 + 4) under prime
relabelling S5, with the prime content k*log p staying diagonal input.

HONEST SCOPE:
  This exhibits operational (representation-theoretic) irreducibility and shows it
  is LOGICALLY INDEPENDENT of arithmetic primality (both directions falsified). It
  does NOT redefine arithmetic primality, and it does NOT touch G4 = RH.

Run:
    python benchmarks/operational_irreducibility.py

Status: RESEARCH (operational-irreducibility falsifier, Camino 2).
"""
from __future__ import annotations

import itertools
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composition_arithmetic import character_norm  # noqa: E402

from tnfr.riemann.prime_ladder_hamiltonian import (  # noqa: E402
    build_prime_ladder_graph,
)

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Matrix-level helpers (eig_levels = matrix version of composition.eigenspaces)
# --------------------------------------------------------------------------- #
def eig_levels(M, tol=1e-6):
    """Return [(eigenvalue, multiplicity, projector)] of a symmetric matrix M."""
    vals, vecs = np.linalg.eigh(M)
    levels = []
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and abs(vals[j] - vals[i]) < tol:
            j += 1
        U = vecs[:, i:j]
        levels.append((float(np.mean(vals[i:j])), j - i, U @ U.T))
        i = j
    return levels


def commutator_norm(X, M):
    """Frobenius norm of the commutator [X, M] = XM - MX."""
    return float(np.linalg.norm(X @ M - M @ X))


def max_commutator(M, mats):
    """max over a group of generators of ||[M, P]|| (0 == M is G-equivariant)."""
    return max(commutator_norm(M, P) for P in mats)


def sym_points_matrices(n):
    """The n! permutation matrices of S_n acting on n points (Aut of K_n)."""
    mats = []
    for perm in itertools.permutations(range(n)):
        P = np.zeros((n, n))
        for j in range(n):
            P[perm[j], j] = 1.0
        mats.append(P)
    return mats


def laplacian(G, nodes):
    """L = D - A in a fixed node order."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    return np.diag(A.sum(axis=1)) - A


# --------------------------------------------------------------------------- #
# Octahedron / Johnson scheme J(4,2): the operationally-composite-5 system
# --------------------------------------------------------------------------- #
def octahedron_scheme():
    """Octahedron = J(4,2) on the 6 two-subsets of {0,1,2,3}.

    Returns (nodes, A_octa, A_match, s4_mats):
      A_octa  : share exactly one point (4-regular octahedron),
      A_match : share zero points (perfect matching, 1-regular),
      s4_mats : the 24 permutation matrices of S4 relabelling {0,1,2,3}.
    Both A_octa and A_match commute with every S4 matrix (S4-invariant scheme).
    """
    nodes = list(itertools.combinations(range(4), 2))  # 6 two-subsets
    idx = {s: i for i, s in enumerate(nodes)}
    n = len(nodes)
    A_octa = np.zeros((n, n))
    A_match = np.zeros((n, n))
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            inter = len(set(a) & set(b))
            if inter == 1:
                A_octa[idx[a], idx[b]] = 1.0
            elif inter == 0:
                A_match[idx[a], idx[b]] = 1.0
    s4 = []
    for perm in itertools.permutations(range(4)):
        P = np.zeros((n, n))
        for s in nodes:
            img = tuple(sorted((perm[s[0]], perm[s[1]])))
            P[idx[img], idx[s]] = 1.0
        s4.append(P)
    return nodes, A_octa, A_match, s4


# --------------------------------------------------------------------------- #
# Prime-ladder relabelling (shared with bridge_primes_riemann.py)
# --------------------------------------------------------------------------- #
def prime_relabel_matrices(nodes, primes):
    """S_n permutation matrices relabelling the primes of the prime ladder."""
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    mats = []
    for perm in itertools.permutations(primes):
        relabel = dict(zip(primes, perm))
        P = np.zeros((n, n))
        for (p, k) in nodes:
            P[idx[(relabel[p], k)], idx[(p, k)]] = 1.0
        mats.append(P)
    return mats


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_composite5_splits_under_S4():
    print("=" * 78)
    print("(1) Operationally COMPOSITE 5: octahedron / S4, accidental 5 -> 3 + 2")
    print("=" * 78)
    _, A_octa, A_match, s4 = octahedron_scheme()
    order = len(s4)  # 24

    # One-parameter S4-invariant family M(t) = A_octa + t * A_match.
    #   standard irrep (3d): eigenvalue  -t
    #   E irrep        (2d): eigenvalue  t - 2
    # Different irreps -> the crossing at t = 1 is ALLOWED (Wigner-von Neumann).
    def M(t):
        return A_octa + t * A_match

    sym_ok = max(max_commutator(M(t), s4) for t in (0.8, 1.0, 1.2)) < TOL
    print(f"  M(t) commutes with all of S4 (t=0.8,1.0,1.2)?  "
          f"max||[M,P]|| < {TOL:.0e}: {sym_ok}")

    for t in (0.8, 1.0, 1.2):
        levels = eig_levels(M(t))
        desc = "   ".join(f"{v:+.2f}(x{m})" for v, m, _ in levels)
        print(f"    t={t:>4}:  {desc}")

    # at the crossing t = 1: the 5-fold level is REDUCIBLE (standard + E)
    levels1 = eig_levels(M(1.0))
    deg5 = [(v, m, P) for v, m, P in levels1 if m == 5]
    ok_deg = len(deg5) == 1
    chi5 = character_norm(deg5[0][2], s4, order) if ok_deg else float("nan")
    print(f"  at t=1: single 5-fold level? {ok_deg};   "
          f"<chi,chi> = {chi5:.3f}  (REDUCIBLE = standard(3) + E(2))")

    # just off the crossing: an irreducible 3 and an irreducible 2
    off = eig_levels(M(0.8))
    pieces = sorted(m for _, m, _ in off if m in (2, 3))
    chis = {m: character_norm(P, s4, order) for _, m, P in off if m in (2, 3)}
    ok_split = pieces == [2, 3] and all(abs(chis[m] - 1.0) < 1e-6 for m in (2, 3))
    print(f"  off-crossing (t=0.8): pieces = {pieces}, "
          f"<chi,chi>: dim3={chis.get(3, float('nan')):.3f}, "
          f"dim2={chis.get(2, float('nan')):.3f}  (both irreducible)")

    ok = sym_ok and ok_deg and abs(chi5 - 2.0) < 1e-6 and ok_split
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- 5 (arithmetic PRIME) is "
          "operationally COMPOSITE: a symmetry-preserving")
    print("           perturbation splits it 3 + 2 (two different irreps crossing)")
    return ok


def test_prime5_does_not_split_under_S6():
    print()
    print("=" * 78)
    print("(2) Operationally PRIME 5: K6 / S6, irreducible 5 cannot be split")
    print("=" * 78)
    G = nx.complete_graph(6)
    nodes = list(G.nodes())
    L = laplacian(G, nodes)  # eigenvalues 0 (x1), 6 (x5)
    s6 = sym_points_matrices(6)  # 720 matrices
    order = len(s6)
    eye = np.eye(6)
    ones = np.ones((6, 6))

    deg5 = [(v, m, P) for v, m, P in eig_levels(L) if m == 5][0]
    chi = character_norm(deg5[2], s6, order)
    print(f"  L(K6): 5-fold level at lambda={deg5[0]:.1f};   "
          f"<chi,chi> = {chi:.3f}  (IRREDUCIBLE standard rep of S6)")

    # symmetry-preserving perturbation = element of the commutant <I, J>
    P_sym = 0.7 * eye + 0.3 * ones  # any a*I + b*J commutes with S6
    sym_ok = max_commutator(P_sym, s6) < TOL
    lv_sym = eig_levels(L + P_sym)
    still5 = any(m == 5 for _, m, _ in lv_sym)
    print(f"  + symmetry-preserving (a*I + b*J): commutes? {sym_ok};   "
          f"still one 5-fold level? {still5}  -> CANNOT split")

    # generic (symmetry-BREAKING) perturbation does split it: proves the rigidity
    # comes from the symmetry, not from a numerical accident
    rng = np.random.default_rng(5)
    R = rng.standard_normal((6, 6))
    R = 0.05 * (R + R.T)
    breaks = max_commutator(R, s6) > 1e-3
    lv_bad = eig_levels(L + R)
    max_mult = max(m for _, m, _ in lv_bad)
    print(f"  + generic symmetric noise: breaks S6? {breaks};   "
          f"largest multiplicity now = {max_mult}  -> splits (symmetry gone)")

    ok = abs(chi - 1.0) < 1e-6 and sym_ok and still5 and breaks and max_mult < 5
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- 5 (arithmetic PRIME) is "
          "operationally PRIME here: no symmetry-")
    print("           preserving perturbation can factor an irreducible mode (Schur)")
    return ok


def test_independence_of_primality_and_irreducibility():
    print()
    print("=" * 78)
    print("(3) Arithmetic primality and operational irreducibility are INDEPENDENT")
    print("=" * 78)
    # Camino 1 fact, recomputed live: K5 / S5 has an IRREDUCIBLE dim-4 mode (4=2x2).
    G = nx.complete_graph(5)
    nodes = list(G.nodes())
    L = laplacian(G, nodes)  # 0 (x1), 5 (x4)
    s5 = sym_points_matrices(5)
    deg4 = [(v, m, P) for v, m, P in eig_levels(L) if m == 4][0]
    chi4 = character_norm(deg4[2], s5, len(s5))
    four_irreducible = abs(chi4 - 1.0) < 1e-6
    print(f"  4 = 2 x 2 (composite) but K5/S5 dim-4 mode <chi,chi>={chi4:.3f}  "
          f"-> IRREDUCIBLE: {four_irreducible}")
    print("  5 is prime          but octahedron/S4 dim-5 mode splits 3+2 -> COMPOSITE")

    print()
    print("   integer | arithmetic | operational (the system decides)")
    print("   --------+------------+----------------------------------")
    print("      4    | composite  | PRIME / irreducible   (K5,  S5)")
    print("      5    | prime      | COMPOSITE  3 + 2      (octahedron, S4)")
    ok = four_irreducible
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the two notions are logically "
          "independent; factorisation")
    print("           is a property of the SYMMETRY, not of the integer")
    return ok


def test_prime_ladder_5_is_composite():
    print()
    print("=" * 78)
    print("(4) Bridge link: the n=5 prime ladder makes 5 operationally COMPOSITE (1+4)")
    print("=" * 78)
    G = build_prime_ladder_graph(5, max_power=4)
    nodes = list(G.nodes())
    primes = sorted({p for (p, _k) in nodes})
    L = laplacian(G, nodes)
    s5 = prime_relabel_matrices(nodes, primes)
    order = len(s5)

    five = [(v, m, P) for v, m, P in eig_levels(L) if m == 5]
    chis = [character_norm(P, s5, order) for _, _, P in five]
    all_reducible = bool(five) and all(abs(c - 2.0) < 1e-6 for c in chis)
    print(f"  primes used = {primes}")
    print(f"  every Laplacian level has multiplicity 5;  "
          f"<chi,chi> over S5 = {np.round(chis, 3)}")
    print("  <chi,chi> = 2 = trivial(1) + standard(4)  ->  5 = 1 + 4 "
          "(operationally composite)")
    print("  same S_n machinery as bridge_primes_riemann.py: the graph never")
    print("  individuates the primes; the content k*log p stays diagonal input.")
    ok = all_reducible
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the prime ladder factors "
          "5 as 1 + 4 under S5")
    return ok


def main():
    print(__doc__)
    results = [
        ("(1) composite 5 splits 3+2 under S4", test_composite5_splits_under_S4()),
        ("(2) prime 5 rigid under S6", test_prime5_does_not_split_under_S6()),
        ("(3) primality independent of irreducibility",
         test_independence_of_primality_and_irreducibility()),
        ("(4) prime ladder: 5 = 1+4 under S5", test_prime_ladder_5_is_composite()),
    ]
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {name:<46}: {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")
    print()
    print("  Reading: operational irreducibility is well-defined and it is NOT")
    print("  arithmetic primality. A degeneracy FACTORISES under a symmetry-")
    print("  preserving perturbation exactly when its eigenspace is reducible")
    print("  (<chi,chi> > 1); it is rigid exactly when irreducible (<chi,chi> = 1,")
    print("  Schur). The prime integer 5 is operationally COMPOSITE on the")
    print("  octahedron/S4 (splits 3+2) and on the n=5 prime ladder/S5 (splits")
    print("  1+4), while the composite integer 4 is operationally PRIME on K5/S5.")
    print("  So 'factorises or not' is decided by the SYSTEM's symmetry, not by")
    print("  the number. This completes the mirror of Camino 1 and stays bounded:")
    print("  it does not redefine arithmetic primality and does not touch G4 = RH.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
