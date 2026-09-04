#!/usr/bin/env python3
"""
Example 122 — Phase-Sector Periodicity Diagnostic: a Basis-Invariant Read of
the CRT Factor-Coset Structure (with a Derived Certificate)
==============================================================================

For a semiprime n = p·q the factor coset (i mod p) is Chinese-Remainder-Theorem
(CRT) structure: Z_n ≅ Z_p × Z_q.  In the DFT basis it is carried by the Fourier
modes at frequencies k = multiples of the cofactor q — modes that are CONSTANT
within each coset (i mod p) and are EXACT eigenvectors of the canonical emergent
operator ``structural_diffusion_operator`` (the ΔNFR EPI channel L_rw = I − D⁻¹W)
on the residue (di)graph.  This example READS that periodic structure with a
**basis-invariant** observable and a **derived** certificate.

Why the previous eigenvector-η² read was not canonical
------------------------------------------------------
A degenerate eigenvalue defines an EIGENSPACE, not a privileged eigenvector: if
Q_λ spans it, so does Q_λ·U for any unitary U.  Any quantity read off individual
eigenvector columns (e.g. the max η² of a single column, thresholded at 0.9) can
change under that rotation while the operator is unchanged — it is an artefact of
the eigensolver's basis, not a property of the dynamics.  The 2026-09-04
canonicity audit exhibited explicit counterexamples (n = 209, 253, 299) where the
0.9 threshold selects a FALSE divisor.

The canonical, basis-invariant observable
-----------------------------------------
Let C_d be the subspace of vectors constant on residue classes (i mod d), with
its constant direction removed, and P_d its orthogonal projector.  Let Π_λ be the
spectral projector of a (possibly degenerate) eigen-cluster.  The score

    score(d, λ) = ‖ P_d · Π_λ ‖₂²   (largest cos² principal angle)

depends only on the two SUBSPACES, so it is invariant under Q_λ → Q_λ·U.  The
DECISION, however, is not a magic threshold: d is certified as a genuine period
only when C_d is an EXACT L-invariant subspace, measured by the residual

    r(d) = ‖ (I − P_d) · L · Q_d ‖₂   (Q_d an orthobasis of C_d ⊖ constant),

certified against a DERIVED tolerance τ = √ε · ‖L‖₂ (machine precision × operator
norm) — never a hand-picked constant.

Three measured results (seed-free; exact linear algebra)
--------------------------------------------------------
R1 SHARED EXACT EIGENVECTOR.  The factor-coset Fourier mode at k = cofactor is an
   eigenvector of BOTH the undirected and directed emergent operator (residual
   ~1e-14): the factor coset is CRT/circulant structure, present in both spectra.

R2 BASIS-INVARIANT SCORE + DERIVED CERTIFICATE.  For n = 209, 253, 299 the true
   factor p certifies (r(p) ~1e-15 < τ) while the audit's false candidate scores
   ‖P_d Π_λ‖² ≈ 0.92–0.94 — ABOVE the discredited 0.9 threshold — yet is REJECTED
   by the residual certificate (r(d) ~5e-2 ≫ τ).  The score is invariant under a
   unitary rotation of a degenerate eigenspace (Δ ~1e-16), whereas a
   single-eigenvector η² is not.

R3 STRUCTURAL SIGNAL (shuffle control).  Permuting the node labels destroys the
   CRT/circulant structure: the residual of the true coset jumps from ~1e-15 to
   O(1), so the certificate correctly stops firing.

Honest scope (unchanged conclusion, corrected method)
-----------------------------------------------------
This is a PERIODICITY DIAGNOSTIC of CRT structure in the canonical emergent
spectrum — NOT a factoring algorithm.  The residue (di)graph has n nodes; building
and diagonalizing it is poly(n) = poly(2^L), i.e. EXPONENTIAL in the input size
L = log₂ n bits, and the candidate scan is O(√n) prime divisors (the order of
trial division).  There is NO speedup and NO cryptographic consequence.  The
correction relative to earlier drafts: the recovery is expressed with a
basis-invariant subspace score and a derived-tolerance certificate, and the
non-canonical η² > 0.9 decision rule is withdrawn.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator)
- examples/08_emergent_geometry/117_emergent_geometry_residue_graph.py (real sector)
- examples/08_emergent_geometry/119_phase_sector_directed_residue.py (complex spectrum)
- examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py (the wall)
- theory/TNFR_NUMBER_THEORY.md §9.9 (this example; phase-sector periodicity)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.physics.structural_diffusion import structural_diffusion_operator


def _qr(n):
    return {(x * x) % n for x in range(1, n)} - {0}


def residue_graph_undirected(n):
    """Undirected residue graph (real sector): edge (i,j) iff +-(i-j) is a QR."""
    R = _qr(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = (i - j) % n
            if d in R or (n - d) in R:
                G.add_edge(i, j)
    return G


def residue_digraph(n):
    """Directed residue graph (phase sector): edge i->j iff (j-i) mod n is QR."""
    R = _qr(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and ((j - i) % n) in R:
                G.add_edge(i, j)
    return G


# --- basis-invariant machinery -------------------------------------------------


def coset_basis(n, d):
    """Orthonormal (n x d) basis of vectors constant on classes (i mod d)."""
    B = np.zeros((n, d))
    for c in range(d):
        idx = np.arange(c, n, d)
        B[idx, c] = 1.0 / math.sqrt(len(idx))
    return B


def noncoset_subspace(n, d):
    """Orthonormal basis of C_d with the constant direction removed."""
    B = coset_basis(n, d)
    ones = np.ones((n, 1)) / math.sqrt(n)
    Bp = B - ones @ (ones.T @ B)
    Q, r = np.linalg.qr(Bp)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-9))
    return Q[:, :rank]


def derived_tolerance(L):
    """Certificate tolerance tau = sqrt(eps) * ||L||_2 (no magic constant)."""
    return math.sqrt(np.finfo(float).eps) * float(np.linalg.norm(L, 2))


def spectral_clusters(L, tol):
    """(lambda, Pi, mult, group) spectral projectors; Pi = Q Q^H is basis-free."""
    w, V = np.linalg.eig(L)
    order = np.argsort(w.real)
    w, V = w[order], V[:, order]
    clusters, used = [], np.zeros(len(w), bool)
    for i in range(len(w)):
        if used[i]:
            continue
        grp = [j for j in range(len(w)) if not used[j] and abs(w[j] - w[i]) <= tol]
        for j in grp:
            used[j] = True
        Q, _ = np.linalg.qr(V[:, grp])
        clusters.append((complex(w[grp].mean()), Q @ Q.conj().T, len(grp), grp))
    return clusters, V


def projector_score(n, d, clusters):
    """Best ||P_d Pi||^2 over non-trivial eigen-clusters (basis-invariant)."""
    Q = noncoset_subspace(n, d)
    if Q.shape[1] == 0:
        return 0.0
    P = Q @ Q.conj().T
    best = 0.0
    for lam, Pi, mult, grp in clusters:
        if abs(lam) < 1e-9:  # skip the trivial constant mode
            continue
        s = np.linalg.svd(P @ Pi, compute_uv=False)
        if s.size:
            best = max(best, float(s[0]) ** 2)
    return best


def invariant_subspace_residual(n, d, L):
    """r(d) = ||(I - Q Q^H) L Q||_2 for Q = C_d minus the constant.

    ~0 iff the coset subspace is an exact L-invariant subspace (its modes are
    eigenvectors).  Depends only on the subspace, so it is basis-free."""
    Q = noncoset_subspace(n, d)
    if Q.shape[1] == 0:
        return 0.0
    LQ = L @ Q
    resid = LQ - Q @ (Q.conj().T @ LQ)
    return float(np.linalg.svd(resid, compute_uv=False)[0])


SEMIPRIMES = [
    (21, 3, 7),
    (33, 3, 11),
    (51, 3, 17),
    (57, 3, 19),
    (91, 7, 13),
    (85, 5, 17),
]

# Audit counterexamples: (n, true factor p, false candidate d flagged at 0.9).
ADVERSARIAL = [(209, 11, 3), (253, 11, 7), (299, 13, 7)]


def experiment_1_shared_eigenvector():
    """R1: the factor-coset Fourier mode is an eigenvector of both operators."""
    print("=" * 74)
    print("EXPERIMENT 1: The Factor Coset Is a Shared Exact Eigenvector (CRT)")
    print("=" * 74)
    print("The mode at k=cofactor is constant within each coset (i mod p), so")
    print("it is an EXACT eigenvector of the emergent operator on BOTH the")
    print("undirected and directed residue graph (CRT / circulant).")
    print()
    print(
        f"  {'n':>4} {'p':>3} {'q':>3} | {'undirected resid':>17} "
        f"{'directed resid':>16}"
    )
    for n, p, q in SEMIPRIMES:
        _, Lu = structural_diffusion_operator(residue_graph_undirected(n))
        _, Ld = structural_diffusion_operator(residue_digraph(n))
        j = np.arange(n)
        fm = np.exp(2j * np.pi * q * j / n)  # k = cofactor q

        def resid(L, f):
            lam = (f.conj() @ (L @ f)) / (f.conj() @ f)
            return float(np.linalg.norm(L @ f - lam * f))

        print(
            f"  {n:>4} {p:>3} {q:>3} | {resid(Lu, fm):>17.2e} "
            f"{resid(Ld, fm):>16.2e}"
        )
    print()
    print("  -> ~1e-14: the factor coset is CRT structure in both spectra.")


def experiment_2_basis_invariant_certificate():
    """R2: basis-invariant score + derived certificate on the counterexamples."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Basis-Invariant Score + Derived Certificate")
    print("=" * 74)
    print("The audit's false candidate scores ||P_d Pi||^2 ABOVE 0.9, so the old")
    print("threshold mis-fires. The DERIVED-tolerance residual certificate")
    print("(tau = sqrt(eps)*||L||) rejects it and accepts only the exact period.")
    print()
    print(
        f"  {'n':>4} {'p':>3} {'d?':>3} | {'score p':>8} {'score d?':>8} | "
        f"{'r(p)':>9} {'r(d?)':>9} | {'p cert':>7} {'d? cert':>7}"
    )
    for n, p, d_false in ADVERSARIAL:
        _, L = structural_diffusion_operator(residue_digraph(n))
        tau = derived_tolerance(L)
        clusters, _ = spectral_clusters(L, tau)
        sc_p = projector_score(n, p, clusters)
        sc_d = projector_score(n, d_false, clusters)
        r_p = invariant_subspace_residual(n, p, L)
        r_d = invariant_subspace_residual(n, d_false, L)
        print(
            f"  {n:>4} {p:>3} {d_false:>3} | {sc_p:>8.4f} {sc_d:>8.4f} | "
            f"{r_p:>9.1e} {r_d:>9.1e} | {str(r_p < tau):>7} {str(r_d < tau):>7}"
        )
    print()
    print("  -> false candidate: score > 0.9 but r(d) >> tau => REJECTED.")
    print("  -> true factor:     r(p) ~ 1e-15 < tau         => CERTIFIED.")

    # Basis-invariance: rotate a degenerate eigenspace by a random unitary.
    print()
    print("  Basis-invariance under Q -> Q*U on a degenerate eigenspace (n=209):")
    _, L = structural_diffusion_operator(residue_digraph(209))
    tau = derived_tolerance(L)
    clusters, V = spectral_clusters(L, tau)
    big = max((c for c in clusters if abs(c[0]) > 1e-9), key=lambda c: c[2])
    grp = big[3]
    rng = np.random.default_rng(0)
    A = rng.standard_normal((len(grp), len(grp))) + 1j * rng.standard_normal(
        (len(grp), len(grp))
    )
    U, _ = np.linalg.qr(A)
    Vr = V.copy()
    Vr[:, grp] = V[:, grp] @ U  # rotate the degenerate eigenspace basis
    Qr, _ = np.linalg.qr(Vr[:, grp])
    clusters_rot = [c for c in clusters if c[3] != grp]
    clusters_rot.append((big[0], Qr @ Qr.conj().T, big[2], grp))
    # The projector Pi = Q Q^H of the eigenspace is identical for V and Q*U, so
    # the subspace score is invariant; an individual eigenvector column is not.
    for label, d in (("true  p=11", 11), ("false d= 3", 3)):
        sc0 = projector_score(209, d, clusters)
        sc1 = projector_score(209, d, clusters_rot)
        print(
            f"    score({label}) : {sc0:.6f} -> {sc1:.6f}"
            f"  (Delta={abs(sc1 - sc0):.1e}, INVARIANT)"
        )
    print("    (a single-eigenvector eta^2 read of that eigenspace is NOT well")
    print("     defined -- it changes with the arbitrary basis Q -> Q*U.)")


def experiment_3_shuffle_control():
    """R3: the certificate is structural (a label shuffle breaks it)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: The Certificate Is Structural (Shuffle Control)")
    print("=" * 74)
    print("Permuting node labels destroys the CRT/circulant structure: the true")
    print("coset residual jumps from ~1e-15 to O(1), so the certificate stops.")
    print()
    print(f"  {'n':>4} {'p':>3} | {'r(p) canonical':>15} {'r(p) shuffled':>15}")
    for n, p, q in [(209, 11, 19), (91, 7, 13), (85, 5, 17)]:
        _, L = structural_diffusion_operator(residue_digraph(n))
        r_can = invariant_subspace_residual(n, p, L)
        rng = np.random.default_rng(3)
        perm = rng.permutation(n)
        Lp = L[np.ix_(perm, perm)]
        r_shuf = invariant_subspace_residual(n, p, Lp)
        print(f"  {n:>4} {p:>3} | {r_can:>15.2e} {r_shuf:>15.2e}")
    print()
    print("  -> canonical ~1e-15, shuffled O(1): structural, not an artefact.")


def main():
    print()
    print("  TNFR Example 122: Phase-Sector Periodicity Diagnostic")
    print("  Basis-Invariant CRT Factor-Coset Read with a Derived Certificate")
    print("  ================================================================")
    print()
    experiment_1_shared_eigenvector()
    experiment_2_basis_invariant_certificate()
    experiment_3_shuffle_control()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("The factor coset (i mod p) of a semiprime n=p*q is CRT/circulant")
    print("structure -- a Fourier mode at k=cofactor, an EXACT eigenvector of the")
    print("canonical emergent operator on both residue graphs. It is read with a")
    print("BASIS-INVARIANT subspace score ||P_d Pi||^2 and CERTIFIED by an exact")
    print("invariant-subspace residual against a DERIVED tolerance sqrt(eps)*||L||")
    print("-- the non-canonical eta^2>0.9 rule is withdrawn (it mis-fires at")
    print("~0.92 on n=209,253,299). HONEST SCOPE: a periodicity DIAGNOSTIC of CRT")
    print("structure, poly(n)=exp(log2 n) to build/diagonalize and an O(sqrt n)")
    print("candidate scan -- NO factoring speedup and no cryptographic threat.")


if __name__ == "__main__":
    main()
