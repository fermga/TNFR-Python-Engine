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
Schur/Riesz construction and are rejected here rather than silently mishandled.
This module never calls ``eigh`` on a
non-symmetric operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # scipy is an OPTIONAL dependency; the Schur certificate is gated on it
    from scipy.linalg import schur as _scipy_schur

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without scipy
    _HAS_SCIPY = False

__all__ = [
    "EigenspaceCluster",
    "NonNormalOperatorError",
    "operator_norm",
    "derived_tolerance",
    "commutator_norm",
    "is_normal",
    "subspace_projector",
    "orthonormal_basis",
    "spectral_clusters",
    "matrix_exponential",
    "spectral_abscissa",
    "transient_gain",
    "pseudospectral_bound",
    "schur_residual",
    "scipy_available",
]


class NonNormalOperatorError(ValueError):
    """Raised when a non-normal operator reaches a normal-only spectral path.

    Non-normal directed operators need a Schur/Riesz projector construction;
    ``Q Qᴴ`` from non-orthogonal right
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


def orthonormal_basis(basis, *, tol: float | None = None) -> np.ndarray:
    r"""Return a rank-aware orthonormal basis for the supplied columns."""
    b = np.asarray(basis, dtype=complex)
    if b.ndim == 1:
        b = b[:, None]
    if b.ndim != 2:
        raise ValueError("basis must be a one- or two-dimensional array")
    if b.shape[1] == 0:
        return np.zeros((b.shape[0], 0), dtype=complex)
    if not np.all(np.isfinite(b)):
        raise ValueError("basis must contain finite values")
    u, singular_values, _ = np.linalg.svd(b, full_matrices=False)
    if tol is None:
        tol = max(b.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > max(float(tol), 0.0)))
    return u[:, :rank]


def subspace_projector(basis) -> np.ndarray:
    r"""Orthogonal projector ``Q Qᴴ`` onto ``span(basis)``.

    The columns of ``basis`` are orthonormalised first, so the result depends
    only on the spanned subspace, not on the particular basis given (hence it is
    invariant under any change of basis of that subspace).
    """
    b = np.asarray(basis, dtype=complex)
    if b.ndim == 1:
        b = b[:, None]
    q = orthonormal_basis(b)
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
        :class:`NonNormalOperatorError` is raised (use a directed non-normal path).
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


# ════════════════════════════════════════════════════════════════════════════
# Directed non-normal dynamics certificates
# For a non-normal generator the eigendecomposition is not an orthonormal basis,
# so stability (spectral abscissa) and transient behaviour must be measured
# separately: a stable spectrum does NOT preclude transient amplification.
# ════════════════════════════════════════════════════════════════════════════
def scipy_available() -> bool:
    r"""Whether SciPy is importable (the Schur certificate needs it)."""
    return _HAS_SCIPY


def matrix_exponential(matrix, *, terms: int = 18) -> np.ndarray:
    r"""``e^{A}`` via scaling-and-squaring with a Taylor inner series (numpy-only).

    Self-contained so the non-normal certificates need no SciPy: ``A`` is scaled
    by ``2^{-s}`` until its ∞-norm is ``≤ 1``, a truncated Taylor series is
    evaluated, and the result is squared ``s`` times.
    """
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    norm = float(np.linalg.norm(a, np.inf))
    s = max(0, int(np.ceil(np.log2(norm + 1.0))))
    b = a / (2 ** s)
    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, terms):
        term = term @ b / k
        result = result + term
    for _ in range(s):
        result = result @ result
    return result


def spectral_abscissa(matrix) -> float:
    r"""``α(A) = max_i Re λ_i(A)`` — the asymptotic growth rate.

    For a diffusion generator ``−L`` asymptotic stability is ``α(−L) ≤ 0``.  Uses
    ``eigvals`` (never ``eigh``), so it is valid for non-symmetric operators.
    """
    return float(np.max(np.linalg.eigvals(np.asarray(matrix)).real))


def transient_gain(
    matrix, *, t_max: float = 10.0, samples: int = 200
) -> float:
    r"""``max_{t∈[0, t_max]} ‖e^{tA}‖₂`` — the peak transient amplification.

    For a **normal** stable ``A`` the semigroup is a contraction (gain ``≤ 1``);
    a **non-normal** stable ``A`` can have gain ``> 1`` — transient growth that a
    stable spectrum alone does not reveal.
    """
    a = np.asarray(matrix, dtype=float)
    gain = 0.0
    for t in np.linspace(0.0, t_max, samples):
        gain = max(gain, float(np.linalg.norm(matrix_exponential(a * t), 2)))
    return gain


def pseudospectral_bound(matrix, *, grid: int = 40) -> float:
    r"""Kreiss lower bound ``sup_{Re z>0} Re(z)·‖(zI − A)⁻¹‖₂`` on the transient
    gain (a pseudospectral certificate).

    The Kreiss matrix theorem gives ``K(A) ≤ sup_t ‖e^{tA}‖``, so a value ``> 1``
    proves transient amplification from the resolvent alone, without exponentials.
    """
    a = np.asarray(matrix, dtype=complex)
    n = a.shape[0]
    eig = np.linalg.eigvals(a)
    span = max(1.0, float(np.max(np.abs(eig))))
    best = 0.0
    for re in np.linspace(span / grid, span, grid):
        for im in np.linspace(-span, span, grid):
            z = complex(re, im)
            resolvent = np.linalg.inv(z * np.eye(n) - a)
            best = max(best, re * float(np.linalg.norm(resolvent, 2)))
    return best


def schur_residual(matrix) -> float:
    r"""``‖A − Q T Qᴴ‖₂`` for the Schur decomposition (the correct unitary
    factorisation for non-normal operators).

    Requires SciPy (``scipy.linalg.schur``); raises when it is unavailable, so
    the numpy-only core never silently mishandles a non-normal operator.
    """
    if not _HAS_SCIPY:
        raise NonNormalOperatorError(
            "schur_residual requires SciPy (scipy.linalg.schur); install SciPy "
            "or use the numpy-only certificates (transient_gain, "
            "pseudospectral_bound)."
        )
    a = np.asarray(matrix, dtype=float)
    t, q = _scipy_schur(a)
    return float(np.linalg.norm(a - q @ t @ q.conj().T, 2))
