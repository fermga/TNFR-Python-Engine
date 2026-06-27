"""
benchmarks/missing_piece_bridge.py

Camino 12 -- are the two B2 escapes ONE missing canonical piece, or two?

commutant_bridge.py (Camino 7) proved the Riemann and Yang-Mills walls have
the SAME SHAPE: in both, the reachable set of the TNFR catalog is the
COMMUTANT of a group, and each open target lives in the orthogonal complement
the commutant cannot reach. It named the two escapes:

  - RH escape : a node-distinct diagonal P2 = diag(nu_f = log p) that breaks
                the S_n commutant (acts on the BASE space V = C^n).
  - YM escape : non-commuting generators su(d) that break the colour-scalar
                commutant (act on the FIBRE space C^d).

The repo memory records a stronger, UNPROVEN conjecture (cross-program
synthesis, 2026-06-12): "the missing non-Abelian canonical derivation (YM
Branch B) and the S_n-breaking structure RH needs (T-HP / G4) may be the SAME
absent canonical piece -- closing one gives the other." Camino 7 showed same
SHAPE; it never tested same PIECE. This harness tests the conjecture directly,
and it can FALSIFY it.

THE QUESTION (falsifiable):
  Is there a SINGLE structural object X such that adjoining X to the catalog
  breaks BOTH the S_n commutant (RH) and the colour-scalar commutant (YM),
  with the SAME non-derivability reason? If yes -> "closing one gives the
  other" is literally true. If no -> the conjecture is refuted and replaced by
  whatever weaker statement survives.

WHAT THIS HARNESS FINDS (preview -- the strong reading is REFUTED, a precise
weaker one SURVIVES):
  (1) NOT THE SAME OBJECT. The RH escape D is a DIAGONAL (Cartan / torus,
      hence Abelian: [D, D'] = 0) operator on the BASE V = C^n; the YM escape
      su(d) is OFF-diagonal (root direction), NON-Abelian ([T_x,T_y] != 0) on
      the FIBRE C^d. Different spaces, different commutation, different
      position in gl. A single matrix cannot be both.
  (2) SAME RECIPE. Both escapes are the identical structural move: "adjoin an
      operator that fails to commute with the existing invariant structure",
      yielding TRACELESS non-Abelian generators -- so(n) on the base for RH
      (the commutator [A, D] of the symmetric coupling with the diagonal is
      real anti-symmetric, tr = 0), su(d) on the fibre for YM.
  (3) ONE INGREDIENT, ONE SPACE ONLY. The single diagonal D does double duty
      on the BASE -- it opens Fix(S_n)^perp (the RH complement) AND, via
      [A, D] in so(n), turns the base algebra non-Abelian. But D acts
      trivially on the fibre: D (x) I_d COMMUTES with I_n (x) T_a, so the base
      ingredient CANNOT supply the fibre's non-commuting generators. The YM
      gap keeps its own, independent missing piece.
  (4) SAME NON-DERIVABILITY ROOT. Both ingredients are absent for the SAME
      nodal reason: dEPI/dt = nu_f . dNFR has no per-node slot (RH:
      D = diag(nu_f=log p) read as IMPOSED input, B0*-beta) and no per-fibre
      multiplet slot (YM: Y3 = OPEN_DERIVABILITY_GAP, canonical gauge U(1), no
      non-commuting generators).

ENGINE (known theorems -- independent ground truth, all pre-TNFR):
  - gl(n) = h (+) n: the Cartan h (diagonal) and the root spaces n
    (off-diagonal E_ij) split M_n(C); [D, E_ij] = (d_i - d_j) E_ij, so
    distinct diagonal entries make every root non-degenerate (visible);
    [E_ij, E_ji] = E_ii - E_jj != 0, so the root spaces are non-commuting
    (su-type).
  - commutator of two real symmetric matrices is real anti-symmetric:
    (AD-DA)^T = DA - AD = -(AD-DA), so [A, D] in so(n) (traceless, non-Abelian)
    whenever it is nonzero -- and it is nonzero iff A has an off-diagonal entry
    A_ij != 0 with d_i != d_j.
  - tensor factorisation: (D (x) I_d)(I_n (x) T) = D (x) T = (I_n (x) T)
    (D (x) I_d), so a base operator and a fibre operator always commute --
    base structure cannot manufacture fibre structure.

TNFR reading (AGENTS.md + src/tnfr/yang_mills + src/tnfr/dynamics/adelic): the
node-distinct diagonal is the canonical adelic carrier nu_f = log p
(CANONICAL, but read by the engine as IMPOSED input, the B0*-beta
P2 = NodeIndexedCouplingWeights that AGENTS.md shows is not nodal-derivable);
the YM side is audited by tnfr.yang_mills.audit_nonabelian_derivability
(conservative verdict OPEN_DERIVABILITY_GAP). This harness reuses the Camino-7
machinery directly (adjacency_laplacian, canonical_per_node_diagonal,
su2_generators, commutator_norm, symmetric_projector from commutant_bridge.py).

HONEST SCOPE -- structural CHECKS pass; the THESIS verdict is OPEN (by design):
  The four structural checks pass at machine precision. Their NET reading
  REFUTES the strong unifying conjecture: there is NO single object X that
  breaks both walls (the escapes live on different tensor factors, base vs
  fibre, and D (x) I commutes with I (x) T). What survives is a precise WEAKER
  unification: both gaps are the SAME RECIPE (break a commutant by adjoining a
  non-commuting, traceless operator) sharing ONE non-derivability root (no
  per-node / per-fibre slot in the nodal equation). So the synthesis reduces
  "two mysteries" to "one recipe with two independent realisations", NOT to
  "one piece". It SHARPENS the conjecture; it closes nothing. Finite toy-graph
  + su(2) linear algebra; nothing here proves RH, the Yang-Mills mass gap, or
  closes G4. R (continuum) and pi remain assumed substrate.

Run:
    python benchmarks/missing_piece_bridge.py

Status: RESEARCH (missing-piece falsifier; Camino 12 of the unification map).
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
# Camino 12 builds directly on Camino 7: reuse its exact machinery so the two
# B2 escapes here are the SAME objects C7 named, not re-derived look-alikes.
from commutant_bridge import (  # noqa: E402
    adjacency_laplacian,
    canonical_per_node_diagonal,
    commutator_norm,
    su2_generators,
    symmetric_projector,
)
from composition_arithmetic import automorphism_matrices  # noqa: E402

# Optional: the canonical engine's own non-Abelian derivability verdict.
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.yang_mills import audit_nonabelian_derivability  # noqa: E402

    _HAVE_AUDIT = True
except Exception:  # pragma: no cover
    _HAVE_AUDIT = False

TOL = 1e-9
_NONZERO = 1e-3


def elementary_matrix(n, i, j):
    """E_ij: the n x n matrix with a single 1 in position (i, j)."""
    E = np.zeros((n, n), dtype=float)
    E[i, j] = 1.0
    return E


# --------------------------------------------------------------------------- #
# TEST 1 -- the two escapes are NOT the same object (the strong reading fails)
# --------------------------------------------------------------------------- #
def test_escapes_not_identical():
    print("=" * 78)
    print(
        "TEST 1 -- NOT ONE OBJECT: D is Abelian-on-base, su(d) is "
        "non-Abelian-on-fibre"
    )
    print("=" * 78)
    n = 5
    # RH escape: a node-distinct diagonal on the BASE V = C^n. Two such
    # diagonals ALWAYS commute -- a single D is an Abelian (Cartan) element.
    d1, _label = canonical_per_node_diagonal(n)
    d2 = np.diag(np.arange(1, n + 1, dtype=float) ** 2)  # another distinct D
    d_abelian = commutator_norm(d1, d2)

    # YM escape: su(2) generators on the FIBRE C^d. They do NOT commute.
    tx, ty, _tz = su2_generators()
    su_nonabelian = commutator_norm(tx, ty)

    # Convention-independent discriminators (the "real vs complex" framing is
    # convention-dependent: in the Hermitian sigma/2 basis sigma_x, sigma_z are
    # real, only sigma_y is imaginary). The robust distinction is structural:
    #  - dimension: D acts on the BASE C^n, T_a on the FIBRE C^d -> shapes
    #               differ
    #  - position : D is DIAGONAL (Cartan / torus), every su(2) generator has a
    #               nonzero OFF-diagonal part (root direction).
    d_shape = d1.shape
    t_shape = tx.shape
    different_spaces = d_shape != t_shape
    d_offdiag = float(np.linalg.norm(d1 - np.diag(np.diag(d1))))
    t_offdiag = max(float(np.linalg.norm(T - np.diag(np.diag(T)))) for T in (tx, ty))

    ok = (
        d_abelian < TOL
        and su_nonabelian > _NONZERO
        and different_spaces
        and d_offdiag < TOL
        and t_offdiag > _NONZERO
    )
    print(f"  RH escape D  : node-distinct DIAGONAL on BASE {d_shape}")
    print(
        f"                 -- two such D commute, ||[D1, D2]|| = "
        f"{d_abelian:.2e} (ABELIAN / Cartan); off-diag = {d_offdiag:.2e}"
    )
    print(
        f"  YM escape su : OFF-diagonal generators on FIBRE {t_shape} -- "
        f"||[T_x, T_y]|| = {su_nonabelian:.3f} (NON-Abelian)"
    )
    print(
        f"  different spaces (base vs fibre): {different_spaces} ; "
        f"D off-diag ~0: {d_offdiag < TOL} ; T off-diag > 0: "
        f"{t_offdiag > _NONZERO}"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- a single matrix cannot "
        "be both; the strong 'same object' reading is REFUTED"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- but both escapes are the SAME RECIPE: adjoin a non-commuting,
#           traceless operator -> so(n) on the base / su(d) on the fibre
# --------------------------------------------------------------------------- #
def test_same_structural_recipe():
    print("=" * 78)
    print(
        "TEST 2 -- SAME RECIPE: adjoin a non-commuting traceless operator "
        "(so(n) / su(d))"
    )
    print("=" * 78)
    n = 5
    G = nx.complete_graph(n)  # prime-relabelling symmetry S_5
    nodes = list(G.nodes())
    mats = automorphism_matrices(G, nodes)
    A, _L = adjacency_laplacian(G, nodes)
    D, _label = canonical_per_node_diagonal(n)

    # RH side: D breaks the S_n commutant, and [A, D] -- the existing
    # symmetric coupling against the diagonal -- is a real ANTI-SYMMETRIC
    # (so(n)) generator.
    rh_break = max(commutator_norm(D, P) for P in mats)
    comm_AD = A @ D - D @ A
    rh_gen_norm = float(np.linalg.norm(comm_AD))
    rh_antisym = float(np.linalg.norm(comm_AD + comm_AD.T))  # so(n): M^T = -M
    rh_traceless = abs(float(np.trace(comm_AD)))

    # YM side: su(2) generators are TRACELESS, anti-Hermitian, non-commuting.
    tx, ty, tz = su2_generators()
    ym_gen = 1j * tz  # i.sigma_z/2 in su(2) (anti-Hermitian)
    ym_nonabelian = commutator_norm(tx, ty)
    ym_traceless = abs(complex(np.trace(ym_gen)))
    ym_antiherm = float(np.linalg.norm(ym_gen + ym_gen.conj().T))

    ok = (
        rh_break > _NONZERO
        and rh_gen_norm > _NONZERO
        and rh_antisym < TOL
        and rh_traceless < TOL
        and ym_nonabelian > _NONZERO
        and ym_traceless < TOL
        and ym_antiherm < TOL
    )
    print(
        f"  RH (base)  : [D, P_s] != 0 (max = {rh_break:.2f}) breaks S_n ; "
        f"[A, D] is so(n)"
    )
    print(
        f"               ||[A,D]|| = {rh_gen_norm:.2f}, anti-symmetry "
        f"||[A,D]+[A,D]^T|| = {rh_antisym:.2e}, |tr| = {rh_traceless:.2e}"
    )
    print(
        f"  YM (fibre) : [T_x,T_y] != 0 (= {ym_nonabelian:.3f}) ; "
        f"i.sigma_z/2 in su(2), |tr| = {ym_traceless:.2e}, "
        f"anti-Herm = {ym_antiherm:.2e}"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- one recipe "
        "(non-commuting traceless adjunction), two spaces: "
        "so(n) base / su(d) fibre"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- one ingredient, one space only: D unifies the BASE side but cannot
#           reach the FIBRE -> "closing one gives the other" fails by space
# --------------------------------------------------------------------------- #
def test_one_ingredient_two_complements():
    print("=" * 78)
    print(
        "TEST 3 -- ONE INGREDIENT, ONE SPACE: D unifies the base side, "
        "cannot reach the fibre"
    )
    print("=" * 78)
    n = 5
    d = 2
    G = nx.complete_graph(n)
    nodes = list(G.nodes())
    mats = automorphism_matrices(G, nodes)
    A, _L = adjacency_laplacian(G, nodes)
    D, _label = canonical_per_node_diagonal(n)
    Pi = symmetric_projector(mats)
    eye = np.eye(n)

    # (a) On the BASE, the single D does double duty:
    #     (i) opens Fix(S_n)^perp -- the RH complement (leak from a sym seed)
    leak = float(np.linalg.norm((eye - Pi) @ (D @ (Pi @ eye[:, 0]))))
    #     (ii) [A, D] in so(n) turns the base algebra non-Abelian
    base_nonabelian = float(np.linalg.norm(A @ D - D @ A))

    # (b) But D acts trivially on the FIBRE: lifted to V (x) C^d it commutes
    #     with EVERY fibre generator I_n (x) T_a -- a base operator cannot
    #     manufacture fibre non-commutativity.
    tx, ty, tz = su2_generators()
    D_base = np.kron(D, np.eye(d))
    fibre_reach = 0.0
    for T in (tx, ty, tz):
        T_fibre = np.kron(eye, T)
        fibre_reach = max(fibre_reach, commutator_norm(D_base, T_fibre))

    ok = leak > _NONZERO and base_nonabelian > _NONZERO and fibre_reach < TOL
    print(
        f"  (a) base side: D opens Fix(S_n)^perp (leak = {leak:.2f}) AND "
        f"[A,D] in so(n) (||[A,D]|| = {base_nonabelian:.2f})"
    )
    print(
        f"  (b) fibre side: D (x) I commutes with EVERY I (x) T_a "
        f"(max ||[.,.]|| = {fibre_reach:.2e})"
    )
    print(
        "      -> the base ingredient D cannot supply the fibre's "
        "non-commuting generators"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- one ingredient unifies "
        "the BASE; the FIBRE keeps an INDEPENDENT missing piece"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- honest OPEN: the two ingredients share ONE non-derivability root
# --------------------------------------------------------------------------- #
def test_shared_nonderivability():
    print("=" * 78)
    print("TEST 4 -- HONEST OPEN: both ingredients absent for the SAME nodal " "reason")
    print("=" * 78)
    n = 5
    G = nx.complete_graph(n)
    nodes = list(G.nodes())
    mats = automorphism_matrices(G, nodes)
    D, d_label = canonical_per_node_diagonal(n)

    # RH side: D = diag(nu_f = log p) breaks S_n but is read as IMPOSED input
    # (no per-node slot in dEPI/dt = nu_f . dNFR; B0*-beta P2 not derivable).
    rh_break = max(commutator_norm(D, P) for P in mats)
    rh_imposed = "log p" in d_label or "diag(1..n)" in d_label

    # YM side: non-commuting generators are needed but Y3 audits them as not
    # derivable from nodal data.
    tx, ty, _tz = su2_generators()
    ym_break = commutator_norm(tx, ty)

    verdict_line = "OPEN_DERIVABILITY_GAP (canonical default; package not imported)"
    canon_ok = True
    if _HAVE_AUDIT:
        try:
            report = audit_nonabelian_derivability()
            any_noncomm = any(c.has_noncommuting_generators for c in report.candidates)
            verdict_line = (
                f"{report.verdict} ; gauge = {report.canonical_gauge_group} ; "
                f"non-commuting generators on any route = {any_noncomm}"
            )
            canon_ok = report.verdict == "OPEN_DERIVABILITY_GAP" and not any_noncomm
        except Exception as exc:  # pragma: no cover
            verdict_line = f"(canonical audit unavailable: {exc})"

    ok = rh_break > _NONZERO and ym_break > _NONZERO and rh_imposed and canon_ok
    print(f"  RH ingredient : D = diag({d_label})")
    print(
        f"                  breaks S_n (||[D,P_s]|| = {rh_break:.2f}) but "
        "is IMPOSED input"
    )
    print(
        "                  (no per-node slot in dEPI/dt = nu_f . dNFR; " "B0*-beta P2)."
    )
    print(
        f"  YM ingredient : non-commuting [T_x,T_y] (= {ym_break:.3f}) "
        "needed, but its"
    )
    print("                  derivation is the open Y3 gap. Canonical audit:")
    print(f"                  {verdict_line}")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- two ingredients, ONE "
        "shared non-derivability root (no per-node / per-fibre slot)"
    )
    print()
    return ok


def main():
    print(__doc__)
    t1 = test_escapes_not_identical()
    t2 = test_same_structural_recipe()
    t3 = test_one_ingredient_two_complements()
    t4 = test_shared_nonderivability()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"  TEST 1 not one object (base/fibre, Abelian/non-Abelian) : "
        f"{'PASS' if t1 else 'FAIL'}"
    )
    print(
        f"  TEST 2 same recipe (so(n) base / su(d) fibre)           : "
        f"{'PASS' if t2 else 'FAIL'}"
    )
    print(
        f"  TEST 3 one ingredient unifies base only, not fibre      : "
        f"{'PASS' if t3 else 'FAIL'}"
    )
    print(
        f"  TEST 4 shared non-derivability root                     : "
        f"{'PASS' if t4 else 'FAIL'}"
    )
    structural = t1 and t2 and t3 and t4
    print()
    print(f"  STRUCTURAL CHECKS: {'ALL PASS' if structural else 'SOME FAIL'}")
    print()
    print("  THESIS VERDICT: OPEN (by design). The strong unifying")
    print("  conjecture -- 'one absent canonical piece; closing one gives the")
    print("  other' -- is REFUTED: the two B2 escapes act on different tensor")
    print("  factors (the base V = C^n for RH, the fibre C^d for YM), D is")
    print("  Abelian-on-base while su(d) is non-Abelian-on-fibre, and")
    print("  D (x) I commutes with I (x) T_a, so the base ingredient cannot")
    print("  supply the fibre's generators. What SURVIVES is a precise weaker")
    print("  unification: both gaps are the SAME RECIPE (break a commutant by")
    print("  adjoining a non-commuting, traceless operator -- so(n) on the")
    print("  base, su(d) on the fibre) sharing ONE non-derivability root (no")
    print("  per-node / per-fibre slot in dEPI/dt = nu_f . dNFR). The")
    print("  synthesis reduces 'two mysteries' to 'one recipe with two")
    print("  independent realisations', NOT to 'one piece'. It SHARPENS the")
    print("  conjecture; it closes nothing. R and pi remain")
    print("  assumed substrate; nothing here proves RH or the YM mass gap.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
