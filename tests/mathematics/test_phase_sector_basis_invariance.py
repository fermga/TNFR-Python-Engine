"""Canonicity regression: phase-sector observables are basis-invariant.

Locks in the 2026-09-04 audit fix for Example 122 (invariants #4, #6): the
factor-coset read uses an eigenspace-PROJECTOR score (invariant under a unitary
rotation Q → Q·U of a degenerate eigenspace) with a DERIVED-tolerance
invariant-subspace residual certificate — not a single-eigenvector η² thresholded
at the non-canonical 0.9.  Covers the audit counterexamples n = 209, 253, 299.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.structural_diffusion import structural_diffusion_operator

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
    B = np.zeros((n, d))
    for c in range(d):
        B[np.arange(c, n, d), c] = 1.0 / math.sqrt(len(np.arange(c, n, d)))
    ones = np.ones((n, 1)) / math.sqrt(n)
    Bp = B - ones @ (ones.T @ B)
    Q, r = np.linalg.qr(Bp)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-9))
    return Q[:, :rank]


def _tau(L):
    return math.sqrt(np.finfo(float).eps) * float(np.linalg.norm(L, 2))


def _clusters(L, tol):
    w, V = np.linalg.eig(L)
    order = np.argsort(w.real)
    w, V = w[order], V[:, order]
    out, used = [], np.zeros(len(w), bool)
    for i in range(len(w)):
        if used[i]:
            continue
        grp = [j for j in range(len(w)) if not used[j] and abs(w[j] - w[i]) <= tol]
        for j in grp:
            used[j] = True
        Q, _ = np.linalg.qr(V[:, grp])
        out.append((complex(w[grp].mean()), Q @ Q.conj().T, grp))
    return out, V


def _score(n, d, clusters):
    Q = _noncoset_subspace(n, d)
    if Q.shape[1] == 0:
        return 0.0
    P = Q @ Q.conj().T
    best = 0.0
    for lam, Pi, grp in clusters:
        if abs(lam) < 1e-9:
            continue
        s = np.linalg.svd(P @ Pi, compute_uv=False)
        if s.size:
            best = max(best, float(s[0]) ** 2)
    return best


def _residual(n, d, L):
    Q = _noncoset_subspace(n, d)
    if Q.shape[1] == 0:
        return 0.0
    LQ = L @ Q
    resid = LQ - Q @ (Q.conj().T @ LQ)
    return float(np.linalg.svd(resid, compute_uv=False)[0])


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_residual_certificate_separates_true_from_false(n, p, d_false):
    """True period certifies (r < τ); false candidate is rejected (r ≫ τ)."""
    _, L = structural_diffusion_operator(_residue_digraph(n))
    tau = _tau(L)
    r_true = _residual(n, p, L)
    r_false = _residual(n, d_false, L)
    assert r_true < tau  # exact invariant subspace
    assert r_false > tau  # not an eigen-subspace, despite a high score


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_false_candidate_score_exceeds_naive_threshold(n, p, d_false):
    """The false candidate scores > 0.9 — why the old threshold mis-fires."""
    _, L = structural_diffusion_operator(_residue_digraph(n))
    clusters, _ = _clusters(L, _tau(L))
    assert _score(n, d_false, clusters) > 0.9
    assert _score(n, p, clusters) > 1.0 - 1e-9  # true period is exact


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_eigenspace_score_unitary_invariant(n, p, d_false):
    """score(d, λ) is invariant under Q_λ → Q_λ·U on a degenerate eigenspace."""
    _, L = structural_diffusion_operator(_residue_digraph(n))
    tau = _tau(L)
    clusters, V = _clusters(L, tau)
    big = max((c for c in clusters if abs(c[0]) > 1e-9), key=lambda c: len(c[2]))
    grp = big[2]
    if len(grp) < 2:
        pytest.skip("no degenerate eigenspace to rotate")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((len(grp), len(grp))) + 1j * rng.standard_normal(
        (len(grp), len(grp))
    )
    U, _ = np.linalg.qr(A)
    Vr = V.copy()
    Vr[:, grp] = V[:, grp] @ U
    Qr, _ = np.linalg.qr(Vr[:, grp])
    rot = [c for c in clusters if c[2] != grp]
    rot.append((big[0], Qr @ Qr.conj().T, grp))
    for d in (p, d_false):
        assert abs(_score(n, d, clusters) - _score(n, d, rot)) < 1e-10


@pytest.mark.parametrize("n,p,d_false", ADVERSARIAL)
def test_shuffle_breaks_certificate(n, p, d_false):
    """A label permutation destroys the CRT structure and the certificate."""
    _, L = structural_diffusion_operator(_residue_digraph(n))
    tau = _tau(L)
    assert _residual(n, p, L) < tau
    rng = np.random.default_rng(3)
    perm = rng.permutation(n)
    Lp = L[np.ix_(perm, perm)]
    assert _residual(n, p, Lp) > tau
