"""
benchmarks/equivariance_wall.py

Camino 5 -- is the "wall" that blocks every deep TNFR programme GENERIC?

The number-theory <-> Riemann link, the Navier-Stokes cascade-development gap,
and the Yang-Mills U(1) -> non-Abelian gap all stop at the same place. This
harness asks whether that is a coincidence or a single group-theoretic fact:
every operator built on L = D - A is equivariant under the canonical graph's
symmetry group G, so the open residue -- which lives in Fix(G)^perp -- is
UNREACHABLE by the catalog, whatever the programme.

ENGINE (known theorems -- the independent ground truth, all pre-TNFR):
  - Automorphism definition: P_s A P_s^T = A for every s in Aut(Gamma), hence
    [A, P_s] = 0, [L, P_s] = 0 and [f(A, L), P_s] = 0 for any matrix function f.
  - Schur's lemma: a G-equivariant operator is block-diagonal across the isotypic
    decomposition; eigenspaces of L are G-invariant and their degeneracy = sum of
    irrep dimensions (character norm <chi,chi> = sum of squared multiplicities, a
    positive integer).
  - Reynolds / group-averaging projector: Pi = (1/|G|) sum_s P_s is the orthogonal
    projector onto Fix(G) = V^G. Every equivariant M commutes with Pi, so M maps
    Fix(G) -> Fix(G) and Fix(G)^perp -> Fix(G)^perp. A symmetric input can NEVER
    acquire a Fix(G)^perp component under any equivariant M.

TNFR reading (AGENTS.md): the 13-operator catalog is built on L = D - A, so every
catalog operator is a function of A and L and is therefore G-equivariant -- this is
the Canonical Catalog Equivariance Theorem (CCET) on G_P14. R_infinity (REMESH-inf,
N15) is exactly the orthogonal projector onto the symmetric subspace: range(R_inf) =
Fix(G) (smooth half, reachable) and ker(R_inf) = Fix(G)^perp (oscillatory residue,
open). The ONLY operator that breaks the wall is a node-indexed diagonal -- exactly
the P2 = NodeIndexedCouplingWeights candidate that AGENTS.md (B0*-beta,
S13sexagesima-sexta) shows is NOT derivable from dEPI/dt = nu_f . dNFR (the nodal
equation has no per-node-weight slot).

THE SAME WALL IN THREE PROGRAMMES (one mechanism, three groups):
  G = S_n on K_n : Riemann. Fix(S_n)^perp = standard rep = "individuate one prime";
                   the residue S(T) = (1/pi) arg zeta(1/2 + iT) lives here. G4 = RH OPEN.
  G = D_n on C_n : chemistry / degeneracy. The (2l+1)-style Fourier doublets; lifting
                   the degeneracy (aufbau ordering) is the Fix(D_n)^perp residue.
  G = Z_2 on P_n : the simplest mirror reflection; clean even/odd split. Bridges to the
                   chiral / antiparticle Z_2 of Camino 6.

HONEST SCOPE:
  Everything here is finite, exact, toy-graph representation theory (Schur + the
  Reynolds projector). It PROVES that equivariant operators cannot reach Fix(G)^perp
  -- the common algebraic SHAPE of the three walls -- and that the only escape is a
  non-derivable per-node diagonal. It does NOT prove RH, Navier-Stokes regularity, or
  the Yang-Mills mass gap: the real programmes add analytic content (continuum limits,
  cascade development, non-Abelian existence) on top of this shape. The achievement is
  UNIFICATION OF THE OBSTRUCTION, not its removal. R (continuum) and the constants
  phi, gamma, pi, e remain assumed substrate. Nothing here closes G4 = RH.

Run:
    python benchmarks/equivariance_wall.py

Status: RESEARCH (equivariance-wall falsifier; Camino 5 of the unification map).
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from composition_arithmetic import (  # noqa: E402
    automorphism_matrices,
    character_norm,
    eigenspaces,
)

# Optional: the canonical adelic carrier (nu_f = log p). The per-node diagonal that
# breaks S_n is exactly this carrier's prime log-frequencies, which the engine reads
# as IMPOSED input -- the same adelic discipline as phase_wall.py / paley_bridge.py /
# boundary_vibration.py (derive the carrier, never derive the wall-breaking input).
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402
    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

TOL = 1e-9
CHAR_TOL = 0.4


def _sieve(n):
    """Primes up to n (Sieve of Eratosthenes) -- fallback if adelic is absent."""
    flag = [True] * (n + 1)
    out = []
    for p in range(2, n + 1):
        if flag[p]:
            out.append(p)
            for k in range(p * p, n + 1, p):
                flag[k] = False
    return out


def canonical_per_node_diagonal(n):
    """The per-node diagonal that breaks S_n is the canonical adelic carrier's prime
    log-frequencies nu_f = log p (distinct per node), which dEPI/dt = nu_f . dNFR
    reads as IMPOSED input -- the P2 = NodeIndexedCouplingWeights candidate that
    AGENTS.md (B0*-beta, S13sexagesima-sexta) shows is NOT nodal-derivable. Falls
    back to a prime sieve, then to diag(1..n); the wall-breaking conclusion is the
    same (any distinct-per-node diagonal leaves the equivariant commutant)."""
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=max(15, 4 * n))
        nu = np.asarray(eng.nu_f, dtype=float)[:n]
        if nu.size == n:
            return np.diag(nu), "nu_f = log p (canonical adelic carrier, IMPOSED)"
    primes = np.array(_sieve(max(15, 4 * n)), dtype=float)[:n]
    if primes.size == n:
        return np.diag(np.log(primes)), "nu_f = log p (sieve fallback, IMPOSED)"
    return np.diag(np.arange(1, n + 1, dtype=float)), "diag(1..n) (abstract per-node)"


# --------------------------------------------------------------------------- #
# Operators (L = D - A is the discrete dNFR / phase-curvature operator)
# --------------------------------------------------------------------------- #
def adjacency_laplacian(G, nodes):
    """Return (A, L) with L = D - A the discrete dNFR / phase-curvature operator."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    return A, L


def _matrix_function(S, f):
    """Apply scalar f to a symmetric matrix S via its spectral decomposition."""
    w, V = np.linalg.eigh(S)
    return (V * f(w)) @ V.T


def catalog_operators(A, L):
    """A representative slice of the TNFR catalog: every entry is a function of A
    and L only (polynomials + functional calculus), so each MUST be equivariant.

    exp(-L/2) is the heat / diffusion semigroup -- the REMESH-inf smooth-half
    kernel; (L + I)^-1 is the structural resolvent. Including them shows the wall
    holds for the full functional calculus, not just low-degree polynomials.
    """
    return {
        "A": A,
        "L = D - A": L,
        "L^2": L @ L,
        "A.L + L.A": A @ L + L @ A,
        "exp(-L/2)": _matrix_function(L, lambda x: np.exp(-0.5 * x)),
        "(L + I)^-1": _matrix_function(L, lambda x: 1.0 / (x + 1.0)),
    }


def commutator_norm(M, P):
    """Frobenius norm ||M P - P M||."""
    return float(np.linalg.norm(M @ P - P @ M))


def symmetric_projector(mats):
    """Reynolds projector Pi = (1/|G|) sum_g P_g onto Fix(G) = V^G."""
    n = mats[0].shape[0]
    Pi = np.zeros((n, n))
    for M in mats:
        Pi += M
    return Pi / len(mats)


# --------------------------------------------------------------------------- #
# The equivariance wall, checked for one (graph, group)
# --------------------------------------------------------------------------- #
def run_wall(name, G, group_label, programme):
    print("=" * 78)
    print(f"{name}: G = {group_label}")
    print(f"   programme: {programme}")
    print("=" * 78)
    nodes = list(G.nodes())
    n = len(nodes)
    mats = automorphism_matrices(G, nodes)
    order = len(mats)
    A, L = adjacency_laplacian(G, nodes)
    ops = catalog_operators(A, L)
    eye = np.eye(n)

    # (E) every catalog operator commutes with every group element
    e_worst = max(commutator_norm(M, P) for M in ops.values() for P in mats)
    e_ok = e_worst < TOL
    print(f"  (E) equivariance  : |Aut| = {order:4d} ; "
          f"max ||[M, P_g]|| over catalog = {e_worst:.2e}  "
          f"-> {'OK' if e_ok else 'FAIL'}")

    # (I) Laplacian eigenspaces are G-invariant and realise irreps
    groups = eigenspaces(G, nodes)
    i_worst = 0.0
    chars = []
    for val, mult, P in groups:
        i_worst = max(i_worst, max(commutator_norm(P, M) for M in mats))
        chars.append((val, mult, character_norm(P, mats, order)))
    chi_int = all(abs(c - round(c)) < CHAR_TOL and round(c) >= 1
                  for _, _, c in chars)
    i_ok = i_worst < TOL and chi_int
    print(f"  (I) isotypic      : max ||[P_eig, P_g]|| = {i_worst:.2e} ; "
          f"eigenspaces realise irreps (<chi,chi> integer)? {chi_int}  "
          f"-> {'OK' if i_ok else 'FAIL'}")
    for val, mult, chi in chars:
        print(f"        lambda = {val:6.3f}   degeneracy = {mult}   "
              f"<chi,chi> = {chi:4.1f}")

    # (W) the wall: Pi onto Fix(G); the catalog cannot reach Fix(G)^perp
    Pi = symmetric_projector(mats)
    fix_dim = int(round(np.trace(Pi)))
    perp_dim = n - fix_dim
    w_comm = max(commutator_norm(M, Pi) for M in ops.values())        # [M, Pi] = 0
    v = Pi @ eye[:, 0]                                                # symmetric seed
    leak = max(float(np.linalg.norm((eye - Pi) @ (M @ v)))
               for M in ops.values())
    w_break = (eye - Pi) @ eye[:, 0]                                  # residue target
    w_norm = float(np.linalg.norm(w_break))
    overlap = max(abs(float(w_break @ (M @ v))) for M in ops.values())
    w_ok = (w_comm < TOL and leak < TOL and overlap < TOL
            and perp_dim >= 1 and w_norm > 1e-6)
    print(f"  (W) the wall      : dim Fix(G) = {fix_dim}, "
          f"dim Fix(G)^perp = {perp_dim} ; max ||[M, Pi]|| = {w_comm:.2e}")
    print(f"        symmetric seed v in Fix(G): max leak ||(I-Pi) M v|| = "
          f"{leak:.2e}  (catalog never enters Fix(G)^perp)")
    print(f"        residue w in Fix(G)^perp (||w|| = {w_norm:.3f}): "
          f"max |<w, M v>| = {overlap:.2e}  -> {'OK' if w_ok else 'FAIL'}")

    ok = e_ok and i_ok and w_ok
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the catalog is "
          "G-equivariant; the Fix(G)^perp residue is unreachable")
    print()
    return ok


# --------------------------------------------------------------------------- #
# Negative control: the wall is REAL, not vacuous
# --------------------------------------------------------------------------- #
def test_negative_control():
    print("=" * 78)
    print("NEGATIVE CONTROL: the wall is REAL, not vacuous -- symmetry IS breakable,")
    print("but only by a NON-canonical per-node diagonal "
          "(P2 = NodeIndexedCouplingWeights)")
    print("=" * 78)
    G = nx.complete_graph(5)
    nodes = list(G.nodes())
    n = len(nodes)
    mats = automorphism_matrices(G, nodes)
    Pi = symmetric_projector(mats)
    eye = np.eye(n)

    # N = the per-node diagonal that breaks S_n. The CANONICAL such diagonal is the
    # adelic carrier's prime log-frequencies nu_f = log p (distinct per node), which
    # the nodal equation reads as IMPOSED input -- NOT a function of A, L.
    N, n_label = canonical_per_node_diagonal(n)
    comm = max(commutator_norm(N, P) for P in mats)
    v = Pi @ eye[:, 0]                                # symmetric seed (constant)
    leak = float(np.linalg.norm((eye - Pi) @ (N @ v)))

    breaks = comm > 1e-3 and leak > 1e-3
    print(f"  N = diag({n_label})")
    print(f"      diag = {np.round(np.diag(N), 4).tolist()}")
    print(f"  max ||[N, P_g]|| = {comm:.3e} (NOT equivariant) ; symmetric seed v in")
    print(f"  Fix(G):  ||(I-Pi) N v|| = {leak:.3e}  (N DOES reach Fix(G)^perp)")
    print("  => symmetry-breaking is possible, but the operator that does it is the")
    print("     per-node prime-frequency diagonal nu_f = log p. The adelic engine")
    print("     reads nu_f as INPUT; AGENTS.md (B0*-beta, S13sexagesima-sexta) shows")
    print("     this P2 = NodeIndexedCouplingWeights is NOT derivable from")
    print("     dEPI/dt = nu_f . dNFR (no per-node slot). So the wall is a property")
    print("     of the CATALOG: equivariant operators CANNOT, the imposed nu_f CAN.")
    print(f"  VERDICT: {'PASS' if breaks else 'FAIL'} -- control breaks the wall "
          "as expected")
    print()
    return breaks


def main():
    print(__doc__)
    cases = [
        ("K5", nx.complete_graph(5), "S_5 (prime relabelling)",
         "Riemann G4=RH : residue S(T)=(1/pi)arg zeta(1/2+iT) in Fix(S_n)^perp [OPEN]"),
        ("C6", nx.cycle_graph(6), "D_6 (dihedral: rotations + reflections)",
         "Chemistry : (2l+1)-style degeneracy ; aufbau lifts it (Fix(D_n)^perp)"),
        ("P5", nx.path_graph(5), "Z_2 (mirror reflection)",
         "simplest reflection ; even/odd split ; bridges to chiral Z_2 (Camino 6)"),
    ]
    wall_results = [(name, run_wall(name, G, gl, pr)) for name, G, gl, pr in cases]
    control = test_negative_control()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in wall_results:
        print(f"  equivariance wall on {name:<3}        : {'PASS' if ok else 'FAIL'}")
    print(f"  negative control (P2 diagonal)   : {'PASS' if control else 'FAIL'}")
    overall = all(ok for _, ok in wall_results) and control
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")
    print()
    print("  Reading: three different symmetry groups (S_n, D_n, Z_2), ONE mechanism.")
    print("  Every operator built on L = D - A commutes with the graph's symmetry")
    print("  group, so it block-diagonalises across Fix(G) + Fix(G)^perp and can")
    print("  NEVER move a symmetric state into the symmetry-breaking complement. The")
    print("  open residue of each programme lives in that complement: S(T) for")
    print("  Riemann (S_n), the degeneracy-lifting for chemistry (D_n), the chiral")
    print("  sign for reflection (Z_2). The only escape is a per-node diagonal, which")
    print("  is not nodal-equation-derivable (P2). HONEST SCOPE: this is exact toy-")
    print("  graph representation theory (Schur + Reynolds projector); it UNIFIES the")
    print("  obstruction (one algebraic shape behind all three walls) but does NOT")
    print("  remove it. It does not prove RH, Navier-Stokes, or Yang-Mills; R and the")
    print("  constants phi,gamma,pi,e stay assumed substrate; nothing here closes G4=RH.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
