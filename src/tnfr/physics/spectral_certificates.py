r"""Basis-invariant spectral certificates: subspace scores and residuals.

Built on the projectors of :mod:`tnfr.physics.spectral_projectors`, this module
answers a single question — *does a candidate subspace ``S`` (a periodicity /
coset / symmetry subspace) live in the spectrum of the canonical operator
``L``?* — with two basis-invariant quantities:

* **projector score** ``score(S) = max_λ ‖P_S Π_λ‖₂²`` — the largest squared
  principal cosine between ``S`` and any eigenspace.  It depends only on the two
  subspaces, hence is invariant under a unitary rotation of any eigenspace.
* **invariant-subspace residual** ``r(S) = ‖(I − P_S) L Q_S‖₂`` — zero iff ``S``
  is an exact ``L``-invariant subspace (its columns span eigenvectors).  The
  decision uses the *derived* tolerance ``τ = √ε·‖L‖₂`` (no hand-picked cut).

A high score is necessary but **not** sufficient; the residual certificate is
what separates a genuine eigen-subspace from a merely well-aligned one.  This is
the canonical replacement for the withdrawn single-eigenvector ``η² > 0.9`` rule
(2026-09-04 audit), which was basis-dependent and produced false positives on
the audit counterexamples ``n = 209, 253, 299``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_projectors import (
    EigenspaceCluster,
    derived_tolerance,
    spectral_clusters,
    subspace_projector,
)

__all__ = [
    "SubspaceCertificate",
    "projector_score",
    "invariant_subspace_residual",
    "certify_invariant_subspace",
]


def _orthonormal(basis) -> np.ndarray:
    b = np.asarray(basis, dtype=complex)
    if b.ndim == 1:
        b = b[:, None]
    if b.shape[1] == 0:
        return b
    q, _ = np.linalg.qr(b)
    return q


def projector_score(
    subspace_basis,
    clusters,
    *,
    min_abs_eigenvalue: float | None = None,
) -> float:
    r"""Basis-invariant score ``max_λ ‖P_S Π_λ‖₂²`` over eigen-clusters.

    Parameters
    ----------
    subspace_basis:
        Columns spanning the candidate subspace ``S`` (orthonormalised here).
    clusters:
        Eigen-clusters from :func:`spectral_clusters`.
    min_abs_eigenvalue:
        If set, clusters with ``|λ| ≤ min_abs_eigenvalue`` are skipped — e.g. to
        drop the trivial constant mode ``λ = 0`` of a connected diffusion
        operator.  Default ``None`` keeps every cluster.
    """
    p = subspace_projector(subspace_basis)
    if not p.any():
        return 0.0
    best = 0.0
    for cluster in clusters:
        if (
            min_abs_eigenvalue is not None
            and abs(cluster.eigenvalue) <= min_abs_eigenvalue
        ):
            continue
        s = np.linalg.svd(p @ cluster.projector, compute_uv=False)
        if s.size:
            best = max(best, float(s[0]) ** 2)
    return best


def invariant_subspace_residual(subspace_basis, matrix) -> float:
    r"""Residual ``‖(I − Q Qᴴ) L Q‖₂`` for an orthonormal basis ``Q`` of ``S``.

    Zero (within numerical error) iff ``S`` is an exact ``L``-invariant
    subspace.  Depends only on the subspace, so it is basis-free.
    """
    q = _orthonormal(subspace_basis)
    if q.shape[1] == 0:
        return 0.0
    a = np.asarray(matrix, dtype=complex)
    lq = a @ q
    resid = lq - q @ (q.conj().T @ lq)
    s = np.linalg.svd(resid, compute_uv=False)
    return float(s[0]) if s.size else 0.0


@dataclass(frozen=True)
class SubspaceCertificate:
    r"""Certificate that a candidate subspace is (not) an ``L``-invariant one.

    ``is_invariant_subspace`` is ``residual < tolerance`` with the derived
    tolerance ``τ = √ε·‖L‖₂``; ``score`` is the basis-invariant projector score
    (reported for context — it is necessary but not sufficient on its own).
    """

    residual: float
    tolerance: float
    score: float
    is_invariant_subspace: bool


def certify_invariant_subspace(
    matrix,
    subspace_basis,
    *,
    tol: float | None = None,
    clusters: list[EigenspaceCluster] | None = None,
    min_abs_eigenvalue: float | None = None,
) -> SubspaceCertificate:
    r"""Certify whether ``S`` is an exact ``L``-invariant subspace.

    Combines the basis-invariant projector score with the derived-tolerance
    residual test ``r(S) < τ = √ε·‖L‖₂``.  When ``clusters`` is omitted it is
    computed with :func:`spectral_clusters` (which rejects non-normal ``L``).
    """
    a = np.asarray(matrix, dtype=complex)
    if tol is None:
        tol = derived_tolerance(a)
    if clusters is None:
        clusters = spectral_clusters(a, tol=tol)
    residual = invariant_subspace_residual(subspace_basis, a)
    score = projector_score(
        subspace_basis, clusters, min_abs_eigenvalue=min_abs_eigenvalue
    )
    return SubspaceCertificate(
        residual=residual,
        tolerance=tol,
        score=score,
        is_invariant_subspace=residual < tol,
    )
