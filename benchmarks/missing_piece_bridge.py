"""Finite comparison of independent base-space and fibre-space matrices.

The historical Camino-12 benchmark juxtaposes two externally supplied objects:
a node-distinct diagonal ``D`` on ``C^n`` and Pauli generators on ``C^2``. It
checks that diagonal matrices commute with one another, selected Pauli matrices
do not, ``[A,D]`` is real antisymmetric and traceless, and operators on separate
tensor factors commute.

Those statements are elementary finite linear algebra. They show that the two
objects act on different spaces and cannot be identified by this construction.
They do not model an RH residual or a Yang--Mills mass-gap target, prove that
either object is required by those problems, or establish a shared obstruction
or a shared non-derivability cause. The ``log(p)`` diagonal, SU(2) generators,
group actions and tensor-product model are all selected inputs. The optional
repository audit is reported independently.

The imported function ``canonical_per_node_diagonal`` retains its historical
API name; its result is treated here only as a supplied diagonal.

Run:
    python benchmarks/missing_piece_bridge.py

Status: RESEARCH benchmark; finite algebra and negative identification result.
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
# Reuse the finite matrix helpers while retaining the historical imports.
from commutant_bridge import (  # noqa: E402
    adjacency_laplacian,
    canonical_per_node_diagonal,
    commutator_norm,
    selected_per_node_diagonal,
    su2_generators,
    symmetric_projector,
)
from composition_arithmetic import automorphism_matrices  # noqa: E402

assert canonical_per_node_diagonal is selected_per_node_diagonal

# Optional repository audit, reported as an independent model output.
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
# TEST 1 -- the selected base and fibre objects are not the same matrix
# --------------------------------------------------------------------------- #
def test_base_fibre_inputs_differ():
    print("=" * 78)
    print(
        "TEST 1 -- DISTINCT INPUTS: D is diagonal-on-base, su(2) is "
        "non-Abelian-on-fibre"
    )
    print("=" * 78)
    n = 5
    # A node-distinct diagonal on the base. Two diagonal matrices commute.
    d1, _label = selected_per_node_diagonal(n)
    d2 = np.diag(np.arange(1, n + 1, dtype=float) ** 2)  # another distinct D
    d_abelian = commutator_norm(d1, d2)

    # Independently selected su(2) generators on the fibre do not commute.
    tx, ty, _tz = su2_generators()
    su_nonabelian = commutator_norm(tx, ty)

    # Convention-independent discriminators (the "real vs complex" framing is
    # convention-dependent: in the Hermitian sigma/2 basis sigma_x, sigma_z are
    # real, only sigma_y is imaginary). The robust distinction is structural:
    #  - dimension: D acts on the BASE C^n, T_a on the FIBRE C^d -> shapes
    #               differ
    #  - position : D is diagonal, while the selected T_x and T_y generators
    #               have nonzero off-diagonal parts.
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
    print(f"  base input D : node-distinct DIAGONAL on shape {d_shape}")
    print(
        f"                 -- two such D commute, ||[D1, D2]|| = "
        f"{d_abelian:.2e} (ABELIAN / Cartan); off-diag = {d_offdiag:.2e}"
    )
    print(
        f"  fibre input  : OFF-diagonal generators on shape {t_shape} -- "
        f"||[T_x, T_y]|| = {su_nonabelian:.3f} (NON-Abelian)"
    )
    print(
        f"  different spaces (base vs fibre): {different_spaces} ; "
        f"D off-diag ~0: {d_offdiag < TOL} ; T off-diag > 0: "
        f"{t_offdiag > _NONZERO}"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the supplied matrices have "
        "different dimensions and commutation properties"
    )
    print("  They therefore cannot be identified by this finite construction.")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- compare antisymmetric/traceless commutators on separate spaces
# --------------------------------------------------------------------------- #
def test_independent_commutator_properties():
    print("=" * 78)
    print(
        "TEST 2 -- PARALLEL FINITE CALCULATIONS: so(n) base / su(2) fibre"
    )
    print("=" * 78)
    n = 5
    G = nx.complete_graph(n)
    nodes = list(G.nodes())
    mats = automorphism_matrices(G, nodes)
    A, _L = adjacency_laplacian(G, nodes)
    D, _label = selected_per_node_diagonal(n)

    # Base calculation: [A,D] is real antisymmetric and traceless.
    base_break = max(commutator_norm(D, P) for P in mats)
    comm_AD = A @ D - D @ A
    base_gen_norm = float(np.linalg.norm(comm_AD))
    base_antisym = float(np.linalg.norm(comm_AD + comm_AD.T))
    base_traceless = abs(float(np.trace(comm_AD)))

    # Fibre calculation: selected su(2) generators have analogous properties.
    tx, ty, tz = su2_generators()
    fibre_gen = 1j * tz  # i.sigma_z/2 in su(2) (anti-Hermitian)
    fibre_nonabelian = commutator_norm(tx, ty)
    fibre_traceless = abs(complex(np.trace(fibre_gen)))
    fibre_antiherm = float(np.linalg.norm(fibre_gen + fibre_gen.conj().T))

    ok = (
        base_break > _NONZERO
        and base_gen_norm > _NONZERO
        and base_antisym < TOL
        and base_traceless < TOL
        and fibre_nonabelian > _NONZERO
        and fibre_traceless < TOL
        and fibre_antiherm < TOL
    )
    print(
        f"  base model : [D, P_s] != 0 (max = {base_break:.2f}) ; "
        f"[A, D] is so(n)"
    )
    print(
        f"               ||[A,D]|| = {base_gen_norm:.2f}, anti-symmetry "
        f"||[A,D]+[A,D]^T|| = {base_antisym:.2e}, |tr| = {base_traceless:.2e}"
    )
    print(
        f"  fibre model: [T_x,T_y] != 0 (= {fibre_nonabelian:.3f}) ; "
        f"i.sigma_z/2 in su(2), |tr| = {fibre_traceless:.2e}, "
        f"anti-Herm = {fibre_antiherm:.2e}"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- both independent "
        "commutator calculations satisfy their stated matrix identities"
    )
    print("  Similar algebraic properties do not identify their physical roles.")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- tensor-factor separation
# --------------------------------------------------------------------------- #
def test_tensor_factor_separation():
    print("=" * 78)
    print(
        "TEST 3 -- TENSOR FACTORS: a base operator commutes with fibre operators"
    )
    print("=" * 78)
    n = 5
    d = 2
    G = nx.complete_graph(n)
    nodes = list(G.nodes())
    mats = automorphism_matrices(G, nodes)
    A, _L = adjacency_laplacian(G, nodes)
    D, _label = selected_per_node_diagonal(n)
    Pi = symmetric_projector(mats)
    eye = np.eye(n)

    # (a) On the base, D moves the selected invariant seed into its orthogonal
    #     complement and has a nonzero commutator with A.
    leak = float(np.linalg.norm((eye - Pi) @ (D @ (Pi @ eye[:, 0]))))
    base_nonabelian = float(np.linalg.norm(A @ D - D @ A))

    # (b) Its lift commutes with the separately supplied fibre generators.
    tx, ty, tz = su2_generators()
    D_base = np.kron(D, np.eye(d))
    fibre_commutator = 0.0
    for T in (tx, ty, tz):
        T_fibre = np.kron(eye, T)
        fibre_commutator = max(fibre_commutator, commutator_norm(D_base, T_fibre))

    ok = leak > _NONZERO and base_nonabelian > _NONZERO and fibre_commutator < TOL
    print(
        f"  (a) base side: orthogonal-component norm = {leak:.2f}; "
        f"[A,D] in so(n) (||[A,D]|| = {base_nonabelian:.2f})"
    )
    print(
        f"  (b) fibre side: D (x) I commutes with each selected I (x) T_a "
        f"(max ||[.,.]|| = {fibre_commutator:.2e})"
    )
    print(
        "      -> this tensor-product construction keeps base and fibre "
        "operations separate"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the selected base lift "
        "commutes with every tested fibre generator"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- report input provenance and the independent repository audit
# --------------------------------------------------------------------------- #
def test_input_and_audit_provenance():
    print("=" * 78)
    print("TEST 4 -- PROVENANCE: supplied diagonal and independent audit output")
    print("=" * 78)
    n = 5
    G = nx.complete_graph(n)
    nodes = list(G.nodes())
    mats = automorphism_matrices(G, nodes)
    D, d_label = selected_per_node_diagonal(n)

    # The diagonal is supplied by the imported helper.
    base_break = max(commutator_norm(D, P) for P in mats)
    input_recognized = "log p" in d_label or "diag(1..n)" in d_label

    # The SU(2) generators are separate supplied matrices.
    tx, ty, _tz = su2_generators()
    fibre_break = commutator_norm(tx, ty)

    verdict_line = "audit unavailable; finite algebra only"
    canon_ok = True
    if _HAVE_AUDIT:
        try:
            report = audit_nonabelian_derivability()
            any_noncomm = any(c.has_noncommuting_generators for c in report.candidates)
            verdict_line = (
                f"{report.verdict} ; "
                f"non-commuting generators on any route = {any_noncomm}"
            )
            canon_ok = report.verdict == "OPEN_DERIVABILITY_GAP" and not any_noncomm
        except Exception as exc:  # pragma: no cover
            verdict_line = f"(repository audit unavailable: {exc})"

    ok = (
        base_break > _NONZERO
        and fibre_break > _NONZERO
        and input_recognized
        and canon_ok
    )
    print(f"  base input   : D = diag({d_label})")
    print(
        f"                  breaks S_n (||[D,P_s]|| = {base_break:.2f}) but "
        "is IMPOSED input"
    )
    print(
        "                  (the values are supplied before the matrix checks)."
    )
    print(
        f"  fibre input  : non-commuting [T_x,T_y] (= {fibre_break:.3f})"
    )
    print("  independent repository audit:")
    print(f"                  {verdict_line}")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- input provenance and the "
        "reported audit condition match this finite protocol"
    )
    print("  The protocol does not prove a shared derivation or obstruction.")
    print()
    return ok


# Historical callable names remain aliases for compatibility only.
test_escapes_not_identical = test_base_fibre_inputs_differ
test_same_structural_recipe = test_independent_commutator_properties
test_one_ingredient_two_complements = test_tensor_factor_separation
test_shared_nonderivability = test_input_and_audit_provenance


def main():
    print(__doc__)
    t1 = test_base_fibre_inputs_differ()
    t2 = test_independent_commutator_properties()
    t3 = test_tensor_factor_separation()
    t4 = test_input_and_audit_provenance()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"  TEST 1 distinct base/fibre inputs                       : "
        f"{'PASS' if t1 else 'FAIL'}"
    )
    print(
        f"  TEST 2 independent so(n)/su(2) calculations             : "
        f"{'PASS' if t2 else 'FAIL'}"
    )
    print(
        f"  TEST 3 tensor-factor separation                         : "
        f"{'PASS' if t3 else 'FAIL'}"
    )
    print(
        f"  TEST 4 input and audit provenance                       : "
        f"{'PASS' if t4 else 'FAIL'}"
    )
    structural = t1 and t2 and t3 and t4
    print()
    print(f"  STRUCTURAL CHECKS: {'ALL PASS' if structural else 'SOME FAIL'}")
    print()
    print("  SCOPE VERDICT: the diagonal, graph, Pauli generators and tensor")
    print("  factors are externally selected. The calculations distinguish the")
    print("  base and fibre models; they do not establish a shared missing piece,")
    print("  recipe, derivation failure, or RH/Yang--Mills obstruction.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
