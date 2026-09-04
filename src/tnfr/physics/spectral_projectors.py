r"""Basis-invariant spectral projectors for the canonical emergent operator.

A degenerate eigenvalue of the canonical structural-diffusion operator
``L_rw = I − D⁻¹W`` defines an eigen*space*, not a privileged eigenvector: if
the columns of ``Q_λ`` span it, so do the columns of ``Q_λ · U`` for any unitary
``U``.  Any observable read off individual eigenvector columns (for example the
maximum coset ``η²`` of a single column, thresholded at a hand-picked value) can
therefore change under that rotation while the operator is unchanged — it is an
artefact of the eigensolver's arbitrary basis.

This module represents degenerate spectral information as **projectors**
``Π_λ = Q_λ Q_λ^H``.  A projector depends only on the subspace, so it is
invariant under ``Q_λ → Q_λ · U``.  Every downstream spectral observable in TNFR
is built from these projectors (see :mod:`tnfr.physics.spectral_certificates`),
which makes basis independence structural rather than incidental.

Scope
=====
The projector construction is exact for **normal** operators
(``L Lᴴ = Lᴴ L``), whose eigenvectors can be chosen orthonormal — this covers
every circulant / vertex-transitive Cayley operator in the TNFR number-theory
programme.  For **non-normal** directed operators the right eigenvectors are not
orthogonal and ``Q Qᴴ`` is not the spectral projector; such operators require a
Schur/Riesz construction (the R9 directed non-normal programme) and are rejected
here rather than silently mishandled.  This module never calls ``eigh`` on a
non-symmetric operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EigenspaceCluster",
    "NonNormalOperatorError",
    "operator_norm",
    "derived_tolerance",
    "commutator_norm",
    "is_normal",
    "subspace_projector",
    "spectral_clusters",
]


class NonNormalOperatorError(ValueError):
    """Raised when a non-normal operator reaches a normal-only spectral path.

    Non-normal directed operators need a Schur/Riesz projector construction (the
    R9 directed non-normal programme); ``Q Qᴴ`` from non-orthogonal right
    eigenvectors is not the spectral projector, so it is refused here.
    """


def operator_norm(matrix) -> float:
    """Spectral norm ``‖L‖₂`` (the largest singular value)."""
    return float(np.linalg.norm(np.asarray(matrix), 2))


def derived_tolerance(matrix) -> float:
    r"""Backward-error-scaled tolerance ``τ = √ε · ‖L‖₂``.

    Not a magic constant: ``√ε`` is the square root of machine epsilon and
    ``‖L‖₂`` sets the problem scale, so ``τ`` tracks the numerical error of an
    eigendecomposition of ``L`` rather than a hand-picked threshold.
    """
    return math.sqrt(np.finfo(float).eps) * operator_norm(matrix)


def commutator_norm(matrix) -> float:
    r"""``‖L Lᴴ − Lᴴ L‖₂``; zero iff ``L`` is normal."""
    a = np.asarray(matrix, dtype=complex)
    return float(np.linalg.norm(a @ a.conj().T - a.conj().T @ a, 2))


def is_normal(matrix, *, tol: float | None = None) -> bool:
    r"""Whether ``L`` is normal (``L Lᴴ = Lᴴ L``) within ``tol``."""
    a = np.asarray(matrix, dtype=complex)
    if tol is None:
        tol = derived_tolerance(a)
    return commutator_norm(a) <= tol


def subspace_projector(basis) -> np.ndarray:
    r"""Orthogonal projector ``Q Qᴴ`` onto ``span(basis)``.

    The columns of ``basis`` are orthonormalised first, so the result depends
    only on the spanned subspace, not on the particular basis given (hence it is
    invariant under any change of basis of that subspace).
    """
    b = np.asarray(basis, dtype=complex)
    if b.ndim == 1:
        b = b[:, None]
    if b.shape[1] == 0:
        return np.zeros((b.shape[0], b.shape[0]), dtype=complex)
    q, _ = np.linalg.qr(b)
    return q @ q.conj().T


@dataclass(frozen=True)
class EigenspaceCluster:
    r"""A basis-free spectral projector for one (possibly degenerate) eigenvalue.

    Attributes
    ----------
    eigenvalue:
        Cluster representative (the mean of the grouped eigenvalues).
    projector:
        ``Π_λ = Q Qᴴ`` onto the eigenspace; invariant under ``Q → Q·U``.
    multiplicity:
        Number of eigenvalues grouped into the cluster.
    indices:
        Indices into the real-part-sorted spectrum that form the cluster.
    """

    eigenvalue: complex
    projector: np.ndarray
    multiplicity: int
    indices: tuple[int, ...]


def spectral_clusters(
    matrix,
    *,
    tol: float | None = None,
    assume_normal: bool | None = None,
) -> list[EigenspaceCluster]:
    r"""Group near-degenerate eigenvalues into basis-free projectors ``Π_λ``.

    Eigenvalues within ``tol`` (default :func:`derived_tolerance`) are grouped;
    each group's eigenvectors are orthonormalised (QR) into ``Q`` and the
    projector ``Π = Q Qᴴ`` is stored.  Because ``Π`` depends only on the
    eigenspace, the clusters are invariant under any per-eigenspace unitary
    rotation of the eigenvectors returned by the eigensolver.

    Parameters
    ----------
    matrix:
        The operator ``L`` (e.g. the structural-diffusion operator).
    tol:
        Degeneracy-grouping tolerance; defaults to ``√ε·‖L‖₂``.
    assume_normal:
        Override normality detection.  When the operator is non-normal a
        :class:`NonNormalOperatorError` is raised (use the R9 directed path).
    """
    a = np.asarray(matrix, dtype=complex)
    if tol is None:
        tol = derived_tolerance(a)
    normal = is_normal(a, tol=tol) if assume_normal is None else assume_normal
    if not normal:
        raise NonNormalOperatorError(
            "spectral_clusters requires a normal operator (L Lᴴ = Lᴴ L); the "
            "given operator is non-normal, so Q Qᴴ is not its spectral "
            "projector. Use the directed non-normal (Schur/Riesz) path."
        )
    w, v = np.linalg.eig(a)
    order = np.argsort(w.real)
    w, v = w[order], v[:, order]
    clusters: list[EigenspaceCluster] = []
    used = np.zeros(len(w), dtype=bool)
    for i in range(len(w)):
        if used[i]:
            continue
        group = [
            j for j in range(len(w)) if not used[j] and abs(w[j] - w[i]) <= tol
        ]
        for j in group:
            used[j] = True
        q, _ = np.linalg.qr(v[:, group])
        clusters.append(
            EigenspaceCluster(
                eigenvalue=complex(w[group].mean()),
                projector=q @ q.conj().T,
                multiplicity=len(group),
                indices=tuple(group),
            )
        )
    return clusters
