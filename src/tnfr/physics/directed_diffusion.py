r"""Directed non-normal structural dynamics (R9).

The canonical structural-diffusion operator ``L_rw = I − D_out⁻¹ W`` is symmetric
(hence **normal**) only for undirected or vertex-transitive graphs.  A general
**directed** graph makes ``L_rw`` non-normal, and then the naive spectral reading
breaks down:

* ``eigh`` is invalid (the operator is not symmetric);
* a stable spectrum (``spectral_abscissa(−L) ≤ 0``) does **not** exclude
  **transient amplification** of ``‖e^{−tL}‖``;
* right eigenvectors are not orthogonal, so ``Q Qᴴ`` is not the spectral
  projector — the correct object is the Schur / Riesz form.

Directed **circulant** Cayley graphs (the residue digraphs of R2) are a benign
special case: they are normal, so their transient gain is ``1`` and the R2/C4
spectral machinery applies unchanged.  This module separates the two regimes and
certifies the non-normal one with numpy-only measures (transient gain,
pseudospectral / Kreiss bound), gating the Schur residual on SciPy.

**U2 scope (honest).**  The reading ``r_c = ν_f λ₂`` describes **asymptotic**
relaxation only.  For non-normal operators a positive transient gain means
``C(t)`` can dip before it relaxes; the U2 integral convergence still holds
asymptotically, but no generalized U2 bound is claimed until the transient
contribution is derived (``NT-P09`` OPEN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_projectors import (
    commutator_norm,
    derived_tolerance,
    is_normal,
    pseudospectral_bound,
    schur_residual,
    scipy_available,
    spectral_abscissa,
    transient_gain,
)

__all__ = [
    "directed_rw_laplacian",
    "directed_cayley_adjacency",
    "is_directed_circulant",
    "DirectedDynamicsCertificate",
    "certify_directed_dynamics",
]


def directed_rw_laplacian(adjacency) -> np.ndarray:
    r"""Directed random-walk Laplacian ``L_rw = I − D_out⁻¹ W`` (OUTGOING
    convention, C1).

    ``W[i][j] = 1`` means an edge ``i → j``; the row is normalised by the
    out-degree ``D_out[i]``.  A sink (out-degree 0) contributes a zero row (it
    neither relaxes nor drives), which keeps ``L_rw`` well defined.
    """
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    out_degree = w.sum(axis=1)
    inv = np.divide(
        1.0, out_degree, out=np.zeros_like(out_degree), where=out_degree > 0
    )
    transition = (w.T * inv).T  # D_out^{-1} W
    laplacian = np.eye(n) - transition
    # zero-out sink rows so isolated sinks do not spuriously self-relax
    laplacian[out_degree == 0] = 0.0
    return laplacian


def directed_cayley_adjacency(n: int, connection) -> np.ndarray:
    r"""Directed circulant Cayley adjacency on ``ℤ/nℤ`` (edge ``i → i + s``).

    A circulant is **normal**, so its directed diffusion has no transient growth
    — the benign special case (R2 residue digraphs).
    """
    conn = {int(s) % n for s in connection}
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for s in conn:
            w[i][(i + s) % n] = 1.0
    return w


def is_directed_circulant(adjacency, *, tol: float | None = None) -> bool:
    r"""Whether the adjacency is circulant (each row a cyclic shift of row 0)."""
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    if tol is None:
        tol = derived_tolerance(w)
    first = w[0]
    for i in range(1, n):
        if not np.allclose(w[i], np.roll(first, i), atol=max(tol, 1e-12)):
            return False
    return True


@dataclass(frozen=True)
class DirectedDynamicsCertificate:
    """Spectral / transient certificate for a directed diffusion generator ``−L``.

    Attributes
    ----------
    normal:
        Whether ``L`` is normal (``L Lᴴ = Lᴴ L``).
    commutator:
        ``‖L Lᴴ − Lᴴ L‖₂`` (zero iff normal).
    abscissa:
        Spectral abscissa ``α(−L) = max Re λ(−L)``; ``≤ 0`` ⇒ asymptotically
        stable.
    asymptotically_stable:
        ``abscissa ≤ tol``.
    transient_gain:
        Peak ``‖e^{−tL}‖₂``; ``> 1`` ⇒ transient amplification.
    has_transient_amplification:
        ``transient_gain > 1 + tol``.
    pseudospectral_bound:
        Kreiss lower bound on the transient gain from the resolvent.
    schur_residual:
        ``‖L − Q T Qᴴ‖₂`` if SciPy is available, else ``None`` (gated).
    """

    normal: bool
    commutator: float
    abscissa: float
    asymptotically_stable: bool
    transient_gain: float
    has_transient_amplification: bool
    pseudospectral_bound: float
    schur_residual: float | None


def certify_directed_dynamics(
    adjacency, *, t_max: float = 10.0, samples: int = 200
) -> DirectedDynamicsCertificate:
    r"""Certify the directed diffusion generator ``−L_rw`` of a digraph.

    Measures normality, asymptotic stability (spectral abscissa), and transient
    behaviour (gain + pseudospectral bound) — the transient measures are what a
    stable spectrum alone cannot provide for a non-normal operator.
    """
    laplacian = directed_rw_laplacian(adjacency)
    generator = -laplacian  # the diffusion flow ∂x/∂t = −L x
    tol = derived_tolerance(laplacian)
    commutator = commutator_norm(laplacian)
    normal = is_normal(laplacian, tol=tol)
    abscissa = spectral_abscissa(generator)
    gain = transient_gain(generator, t_max=t_max, samples=samples)
    kreiss = pseudospectral_bound(generator)
    schur = schur_residual(laplacian) if scipy_available() else None
    return DirectedDynamicsCertificate(
        normal=normal,
        commutator=commutator,
        abscissa=abscissa,
        asymptotically_stable=abscissa <= tol,
        transient_gain=gain,
        has_transient_amplification=gain > 1.0 + tol,
        pseudospectral_bound=kreiss,
        schur_residual=schur,
    )
