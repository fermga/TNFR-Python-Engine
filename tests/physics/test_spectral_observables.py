r"""Canonicity tests for the basis-invariant spectral observables (C4).

Covers :mod:`tnfr.physics.spectral_projectors` and
:mod:`tnfr.physics.spectral_certificates`: the projector/cluster primitives,
their basis- and relabel-invariance, and the derived-tolerance invariant-
subspace certificate that replaces the withdrawn single-eigenvector ``η² > 0.9``
rule (2026-09-04 audit) on the audit counterexamples ``n = 209, 253, 299``.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.spectral_certificates import (
    certify_invariant_subspace,
    invariant_subspace_residual,
    projector_score,
)
from tnfr.physics.spectral_projectors import (
    NonNormalOperatorError,
    commutator_norm,
    derived_tolerance,
    is_normal,
    orthonormal_basis,
    spectral_clusters,
    subspace_projector,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator

# (n, true factor p, false candidate d flagged by the naive 0.9 threshold)
ADVERSARIAL = [(209, 11, 3), (253, 11, 7), (299, 13, 7)]


def _qr(n):
    return {(x * x) % n for x in range(1, n)} - {0}


def _residue_digraph(n):
    R = _qr(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and ((j - i) % n) in R:
                G.add_edge(i, j)
    return G


def _noncoset_subspace(n, d):
    """Orthonormal basis of the (i mod d) coset space with the constant removed."""
    B = np.zeros((n, d))
    for c in range(d):
        idx = np.arange(c, n, d)
        B[idx, c] = 1.0 / math.sqrt(len(idx))
    ones = np.ones((n, 1)) / math.sqrt(n)
    Bp = B - ones @ (ones.T @ B)
    Q, r = np.linalg.qr(Bp)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-9))
    return Q[:, :rank]


# --- primitives ---------------------------------------------------------------


def test_derived_tolerance_scales_with_operator_norm():
    L = np.diag([1.0, 2.0, 3.0])
    assert derived_tolerance(10.0 * L) == pytest.approx(10.0 * derived_tolerance(L))


def test_is_normal_detects_symmetric_and_rejects_triangular():
    sym = np.array([[2.0, 1.0], [1.0, 2.0]])
    triangular = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert is_normal(sym)
    assert not is_normal(triangular)
    assert commutator_norm(triangular) > 0.0


def test_subspace_projector_is_idempotent_hermitian_and_basis_free():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((7, 3))
    P = subspace_projector(basis)
    assert np.allclose(P @ P, P, atol=1e-10)
    assert np.allclose(P, P.conj().T, atol=1e-10)
    # a different basis of the SAME column space yields the same projector
    mixed = basis @ rng.standard_normal((3, 3))
    assert np.allclose(subspace_projector(mixed), P, atol=1e-10)


def test_rank_deficient_basis_does_not_create_spurious_projector_directions():
    basis = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    q = orthonormal_basis(basis)
    assert q.shape == (2, 1)
    assert np.trace(subspace_projector(basis)) == pytest.approx(1.0)
    assert np.allclose(subspace_projector(basis), np.diag([1.0, 0.0]))


def test_zero_basis_produces_zero_projector():
    basis = np.zeros((3, 2))
    assert np.allclose(subspace_projector(basis), np.zeros((3, 3)))


def test_spectral_clusters_projectors_partition_identity():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    H = A + A.conj().T  # Hermitian => normal, real spectrum
    clusters = spectral_clusters(H)
    total = sum(c.projector for c in clusters)
    assert np.allclose(total, np.eye(6), atol=1e-8)
    for c in clusters:
        assert np.allclose(c.projector @ c.projector, c.projector, atol=1e-8)
        assert np.allclose(c.projector, c.projector.conj().T, atol=1e-8)


def test_spectral_clusters_rejects_non_normal():
    triangular = np.array([[1.0, 5.0], [0.0, 2.0]])
    with pytest.raises(NonNormalOperatorError):
        spectral_clusters(triangular)


def test_spectral_clusters_exact_degenerate_projector():
    rng = np.random.default_rng(2)
    G = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    U, _ = np.linalg.qr(G)  # random unitary
    lam = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    L = U @ np.diag(lam) @ U.conj().T  # normal, with a 2-fold eigenspace at λ=1
    analytic = U[:, 1:3] @ U[:, 1:3].conj().T
    clusters = spectral_clusters(L)
    match = min(clusters, key=lambda c: abs(c.eigenvalue - 1.0))
    assert match.multiplicity == 2
    assert np.linalg.norm(match.projector - analytic) < 1e-10


def test_unitary_basis_rotation_leaves_projector_invariant():
    rng = np.random.default_rng(3)
    G = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    U, _ = np.linalg.qr(G)
    grp = U[:, 1:4]  # a 3-dim subspace
    R = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    rotation, _ = np.linalg.qr(R)
    assert np.allclose(
        subspace_projector(grp), subspace_projector(grp @ rotation), atol=1e-10
    )


def test_ambiguous_cluster_identity_returns_unresolved_decision():
    matrix = np.diag([0.0, 1.0, 1.0 + 3e-8])
    basis = np.array([[0.0], [1.0], [0.0]])
    certificate = certify_invariant_subspace(matrix, basis, tol=2e-8)
    assert certificate.is_invariant_subspace
    assert certificate.cluster_identity_resolved is False
    assert certificate.decision is None


# --- certificate on the residue digraph (via the module) ----------------------


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_certificate_certifies_true_factor_and_rejects_false(n, p, d_false):
    _, L = structural_diffusion_operator(_residue_digraph(n))
    true_cert = certify_invariant_subspace(L, _noncoset_subspace(n, p))
    false_cert = certify_invariant_subspace(L, _noncoset_subspace(n, d_false))
    assert true_cert.is_invariant_subspace
    assert true_cert.residual < true_cert.tolerance
    assert not false_cert.is_invariant_subspace
    assert false_cert.residual > false_cert.tolerance


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_projector_score_false_candidate_exceeds_naive_threshold(n, p, d_false):
    _, L = structural_diffusion_operator(_residue_digraph(n))
    clusters = spectral_clusters(L)
    score_true = projector_score(
        _noncoset_subspace(n, p), clusters, min_abs_eigenvalue=1e-9
    )
    score_false = projector_score(
        _noncoset_subspace(n, d_false), clusters, min_abs_eigenvalue=1e-9
    )
    assert score_true > 1.0 - 1e-9  # true period is an exact eigen-subspace
    assert score_false > 0.9  # why the withdrawn 0.9 threshold mis-fires


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_relabel_invariance_of_residual_and_score(n, p, d_false):
    _, L = structural_diffusion_operator(_residue_digraph(n))
    Q = _noncoset_subspace(n, p)
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    Lp = L[np.ix_(perm, perm)]
    Qp = Q[perm, :]  # permute the subspace WITH the operator
    r0 = invariant_subspace_residual(Q, L)
    r1 = invariant_subspace_residual(Qp, Lp)
    assert abs(r0 - r1) < 1e-10
    c0 = spectral_clusters(L)
    c1 = spectral_clusters(Lp)
    s0 = projector_score(Q, c0, min_abs_eigenvalue=1e-9)
    s1 = projector_score(Qp, c1, min_abs_eigenvalue=1e-9)
    assert abs(s0 - s1) < 1e-8


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_shuffle_breaks_certificate_non_cayley_control(n, p, d_false):
    """Permuting only the operator (not the subspace) destroys the CRT
    structure: the true-factor coset is no longer an eigen-subspace."""
    _, L = structural_diffusion_operator(_residue_digraph(n))
    Q = _noncoset_subspace(n, p)
    assert invariant_subspace_residual(Q, L) < derived_tolerance(L)
    rng = np.random.default_rng(11)
    perm = rng.permutation(n)
    L_shuffled = L[np.ix_(perm, perm)]
    assert invariant_subspace_residual(Q, L_shuffled) > derived_tolerance(L_shuffled)
