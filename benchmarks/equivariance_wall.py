"""Finite graph-automorphism and fixed-subspace controls.

The supplied unweighted graphs are K5, C6 and P5. Six explicitly constructed
linear matrices (A, L, polynomials, a heat kernel and a resolvent) are compared
with their graph automorphisms and Reynolds projector. In exact arithmetic,
commutation with every group element implies commutation with the projector;
a linear equivariant map therefore preserves both Fix(G) and its orthogonal
complement. Numerical residuals below TOL test these finite instances.

A Laplacian eigenspace carries a representation that can be reducible. An
integer character norm is consistent with such a representation; it does not
by itself establish irreducibility. A supplied nonconstant node diagonal is
one contrasting matrix that can move a symmetric seed out of Fix(G).

These matrices are not the engine's 13 operator implementations. Named node
selection, state, capacity, history and support changes need separate actions
and hypotheses. Nonlinear equivariance preserves the fixed set but need not
preserve its orthogonal complement. No representation of analytic S(T),
Riemann zeros, particles or PDE solutions is constructed here; this does not
unify obstructions to RH, Yang-Mills or Navier-Stokes.

The helper names remain for existing callers. See
``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` section 6 for the symmetry boundary.
Run ``python benchmarks/equivariance_wall.py`` for the finite matrix controls.
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
from composition_arithmetic import (  # noqa: E402
    automorphism_matrices,
    character_norm,
    eigenspaces,
)

# Optional arithmetic carrier: distinct supplied log-prime values provide one
# noncommuting diagonal. They are not a derived universal capacity law.
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
    """Return a supplied nonconstant diagonal for the complete-graph control.

    The legacy name is retained. Distinct log-prime or integer entries give
    one non-equivariant matrix under S_n, not the unique way to break symmetry.
    The nodal model admits local capacities; this choice is not derived here.
    """
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=max(15, 4 * n))
        nu = np.asarray(eng.nu_f, dtype=float)[:n]
        if nu.size == n:
            return np.diag(nu), "nu_f = log p (supplied adelic carrier)"
    primes = np.array(_sieve(max(15, 4 * n)), dtype=float)[:n]
    if primes.size == n:
        return np.diag(np.log(primes)), "nu_f = log p (sieve fallback, IMPOSED)"
    return np.diag(np.arange(1, n + 1, dtype=float)), "diag(1..n) (abstract per-node)"


# --------------------------------------------------------------------------- #
# Supplied matrix controls. L = D - A agrees with a scalar multiple of L_rw
# only on regular graphs (K5/C6 here, not P5). Automorphism equivariance can
# hold for either Laplacian without their eigenspaces being identical.
# --------------------------------------------------------------------------- #
def adjacency_laplacian(G, nodes):
    """Return adjacency and combinatorial Laplacian on the supplied node order."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    return A, L


def _matrix_function(S, f):
    """Apply scalar f to a symmetric matrix S via its spectral decomposition."""
    w, V = np.linalg.eigh(S)
    return (V * f(w)) @ V.T


def catalog_operators(A, L):
    """Six supplied linear matrix functions; legacy helper name retained.

    Automorphisms commuting with A and L also commute with these matrices in
    exact arithmetic. The heat kernel exp(-L/2) is not an engine REMESH call;
    this helper neither enumerates nor executes the 13 named TNFR operators.
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

    # (E) compare the six matrices with each enumerated automorphism.
    e_worst = max(commutator_norm(M, P) for M in ops.values() for P in mats)
    e_ok = e_worst < TOL
    print(
        f"  (E) equivariance  : |Aut| = {order:4d} ; "
        f"max ||[M, P_g]|| over six matrices = {e_worst:.2e}  "
        f"-> {'OK' if e_ok else 'FAIL'}"
    )

    # (I) invariant eigenspaces can carry reducible representations.
    groups = eigenspaces(G, nodes)
    i_worst = 0.0
    chars = []
    for val, mult, P in groups:
        i_worst = max(i_worst, max(commutator_norm(P, M) for M in mats))
        chars.append((val, mult, character_norm(P, mats, order)))
    chi_int = all(abs(c - round(c)) < CHAR_TOL and round(c) >= 1 for _, _, c in chars)
    i_ok = i_worst < TOL and chi_int
    print(
        f"  (I) eigenspaces   : max ||[P_eig, P_g]|| = {i_worst:.2e} ; "
        f"character norms near positive integers? {chi_int}  "
        f"-> {'OK' if i_ok else 'FAIL'}"
    )
    for val, mult, chi in chars:
        print(
            f"        lambda = {val:6.3f}   degeneracy = {mult}   "
            f"<chi,chi> = {chi:4.1f}"
        )

    # (W) a fixed-set seed under the six commuting linear maps.
    Pi = symmetric_projector(mats)
    fix_dim = int(round(np.trace(Pi)))
    perp_dim = n - fix_dim
    w_comm = max(commutator_norm(M, Pi) for M in ops.values())  # [M, Pi] = 0
    v = Pi @ eye[:, 0]  # symmetric seed
    leak = max(float(np.linalg.norm((eye - Pi) @ (M @ v))) for M in ops.values())
    w_break = (eye - Pi) @ eye[:, 0]  # residue target
    w_norm = float(np.linalg.norm(w_break))
    overlap = max(abs(float(w_break @ (M @ v))) for M in ops.values())
    w_ok = (
        w_comm < TOL
        and leak < TOL
        and overlap < TOL
        and perp_dim >= 1
        and w_norm > 1e-6
    )
    print(
        f"  (W) the wall      : dim Fix(G) = {fix_dim}, "
        f"dim Fix(G)^perp = {perp_dim} ; max ||[M, Pi]|| = {w_comm:.2e}"
    )
    print(
        f"        symmetric seed v in Fix(G): max leak ||(I-Pi) M v|| = "
        f"{leak:.2e}  (selected matrix/seed controls)"
    )
    print(
        f"        residue w in Fix(G)^perp (||w|| = {w_norm:.3f}): "
        f"max |<w, M v>| = {overlap:.2e}  -> {'OK' if w_ok else 'FAIL'}"
    )

    ok = e_ok and i_ok and w_ok
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- finite commutator, character "
        "and symmetric-seed residual checks"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# Negative control: one supplied matrix outside the commutant.
# --------------------------------------------------------------------------- #
def test_negative_control():
    print("=" * 78)
    print("NEGATIVE CONTROL: one supplied nonconstant node diagonal")
    print("=" * 78)
    G = nx.complete_graph(5)
    nodes = list(G.nodes())
    n = len(nodes)
    mats = automorphism_matrices(G, nodes)
    Pi = symmetric_projector(mats)
    eye = np.eye(n)

    # N is a supplied input, not a function of this graph's A and L alone.
    N, n_label = canonical_per_node_diagonal(n)
    comm = max(commutator_norm(N, P) for P in mats)
    v = Pi @ eye[:, 0]  # symmetric seed (constant)
    leak = float(np.linalg.norm((eye - Pi) @ (N @ v)))

    breaks = comm > 1e-3 and leak > 1e-3
    print(f"  N = diag({n_label})")
    print(f"      diag = {np.round(np.diag(N), 4).tolist()}")
    print(f"  max ||[N, P_g]|| = {comm:.3e} (NOT equivariant) ; symmetric seed v in")
    print(f"  Fix(G):  ||(I-Pi) N v|| = {leak:.3e}  (N DOES reach Fix(G)^perp)")
    print("  => this diagonal is a counterexample to unrestricted equivariance.")
    print("     Its assigned node values break the chosen graph symmetry;")
    print("     they are neither forbidden local capacities nor a uniquely")
    print("     derived law. Other non-equivariant maps are not excluded.")
    print(
        f"  VERDICT: {'PASS' if breaks else 'FAIL'} -- control breaks the wall "
        "as expected"
    )
    print()
    return breaks


def main():
    print(__doc__)
    cases = [
        (
            "K5",
            nx.complete_graph(5),
            "S_5 (vertex permutations)",
            "one transitive orbit; fixed vectors are constant",
        ),
        (
            "C6",
            nx.cycle_graph(6),
            "D_6 (dihedral: rotations + reflections)",
            "cycle eigenspace multiplicities under rotations and reflections",
        ),
        (
            "P5",
            nx.path_graph(5),
            "Z_2 (mirror reflection)",
            "even/odd reflection split on a supplied path",
        ),
    ]
    wall_results = [(name, run_wall(name, G, gl, pr)) for name, G, gl, pr in cases]
    control = test_negative_control()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in wall_results:
        print(f"  equivariance wall on {name:<3}        : {'PASS' if ok else 'FAIL'}")
    print(f"  negative control (node diagonal): {'PASS' if control else 'FAIL'}")
    overall = all(ok for _, ok in wall_results) and control
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")
    print()
    print("  Reading: the six supplied linear matrices preserve the fixed set")
    print("  within the selected numerical tolerances on K5, C6 and P5.")
    print("  One nonconstant diagonal gives a contrasting non-equivariant map.")
    print("  These controls do not establish catalog-wide equivariance, a")
    print("  representation of analytic S(T), or a common obstruction to RH/PDEs.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
