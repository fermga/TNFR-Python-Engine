r"""Pointed-pulse amplitudes on normal circulants (R2, N12).

R2 fixed the *order* of the pointed pulse (Hankel rank = #distinct eigenvalues =
``gcd(k, p−1) + 1``).  This module fixes its **amplitudes**.  The k-th power
residue operator ``L_rw`` on ``ℤ/pℤ`` is a **circulant**, hence normal and
diagonalized by the Fourier basis ``f_j[x] = ω^{jx}/√p`` (``ω = e^{2πi/p}``).  The
pointed seed ``e₀`` has **uniform** Fourier weight, ``|⟨e₀, f_j⟩|² = 1/p`` for
every ``j``, so the pulse

    ``h(t) = e₀^* e^{−tL} e₀ = Σ_j (1/p) e^{−t λ_j} = Σ_λ (m_λ/p) e^{−t λ}``

carries, on each distinct eigenvalue ``λ`` of multiplicity ``m_λ``, the exact
**amplitude** ``a_λ = m_λ / p``.

**Theorem (NT-P02b).**  The pointed-circulant pulse amplitudes are the normalized
spectral multiplicities: ``a_λ = m_λ / n`` (here ``n = p``).  Two independent
checks confirm it: the orthogonal spectral projector gives ``e₀^* P_λ e₀ = m_λ/p``
(basis-independent), and ``Σ_λ (m_λ/p) λ^m`` reconstructs the **exact rational**
moments ``μ_m = e₀ᵀ L^m e₀`` for every ``m``.  Complexity: exponential in the
input size ``log₂ p`` (a ``p``-node network), like the R2 rank — not a fast test.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .arithmetic_pulse import cyclotomic_rank, power_residue_laplacian
from .krylov import moment_sequence
from .number_theory import power_residue_set

__all__ = [
    "circulant_eigenvalues",
    "fourier_basis",
    "pulse_amplitudes",
    "projection_amplitudes",
    "moment_reconstruction_residual",
    "amplitudes_basis_invariance_residual",
    "PulseAmplitudeCertificate",
    "certify_pulse_amplitudes",
]


def circulant_eigenvalues(p: int, k: int) -> np.ndarray:
    r"""The ``p`` Fourier eigenvalues ``λ_j = 1 − (1/d) Σ_{r∈R_k} ω^{jr}``."""
    residues = sorted(power_residue_set(p, k))
    d = len(residues)
    if d == 0:
        raise ValueError("empty power-residue connection set")
    omega = np.exp(2j * np.pi / p)
    return np.array([
        1.0 - sum(omega ** ((j * r) % p) for r in residues) / d
        for j in range(p)
    ])


def fourier_basis(p: int) -> np.ndarray:
    r"""The unitary Fourier basis ``F[x, j] = ω^{jx} / √p`` (eigenvectors of any
    circulant on ``ℤ/pℤ``)."""
    j = np.arange(p)
    return np.exp(2j * np.pi * np.outer(j, j) / p) / np.sqrt(p)


def _groups(eigs: np.ndarray, tol: float) -> list[list[int]]:
    r"""Index groups of (numerically) equal eigenvalues."""
    reps: list[complex] = []
    groups: list[list[int]] = []
    for idx, lam in enumerate(eigs):
        for g, rep in enumerate(reps):
            if abs(rep - lam) <= tol:
                groups[g].append(idx)
                break
        else:
            reps.append(complex(lam))
            groups.append([idx])
    return groups


def pulse_amplitudes(p: int, k: int, *, tol: float = 1e-9
                     ) -> list[tuple[complex, int, Fraction]]:
    r"""``(eigenvalue, multiplicity, amplitude = m_λ/p)`` per distinct eigenvalue.

    The amplitude is exact (an integer multiplicity over ``p``); the eigenvalue
    is the complex Fourier value.
    """
    eigs = circulant_eigenvalues(p, k)
    out = []
    for grp in _groups(eigs, tol):
        rep = complex(np.mean(eigs[grp]))
        out.append((rep, len(grp), Fraction(len(grp), p)))
    return out


def projection_amplitudes(p: int, k: int, *, tol: float = 1e-9
                          ) -> list[tuple[complex, float]]:
    r"""``a_λ = e₀^* P_λ e₀`` from the orthogonal spectral projector.

    Independent of the multiplicity count: ``P_λ = Σ_{j∈λ} f_j f_j^*`` and
    ``e₀^* P_λ e₀ = Σ_{j∈λ} |f_j[0]|²``.
    """
    eigs = circulant_eigenvalues(p, k)
    fbasis = fourier_basis(p)
    e0 = np.zeros(p)
    e0[0] = 1.0
    out = []
    for grp in _groups(eigs, tol):
        v = fbasis[:, grp]
        proj = v @ v.conj().T
        amp = float((e0 @ proj @ e0).real)
        out.append((complex(np.mean(eigs[grp])), amp))
    return out


def moment_reconstruction_residual(p: int, k: int, *,
                                   count: int | None = None) -> float:
    r"""``max_m |Σ_λ a_λ λ^m − μ_m|`` for the exact rational moments
    ``μ_m = e₀ᵀ L^m e₀`` — the dynamical confirmation that ``a_λ = m_λ/p``."""
    if count is None:
        count = 2 * cyclotomic_rank(p, k) + 4
    laplacian = power_residue_laplacian(p, k)
    e0 = [Fraction(0)] * p
    e0[0] = Fraction(1)
    moments = moment_sequence(laplacian, e0, count)
    amps = pulse_amplitudes(p, k)
    resid = 0.0
    for m, mu in enumerate(moments):
        recon = sum(complex(a) * (lam ** m) for lam, _, a in amps)
        resid = max(resid, abs(recon - complex(float(mu))))
    return resid


def amplitudes_basis_invariance_residual(p: int, k: int, *,
                                         tol: float = 1e-9,
                                         seed: int = 0) -> float:
    r"""``max_λ |e₀^*P_λe₀ − e₀^*P_λ'e₀|`` for a unitarily-rotated eigenbasis.

    Rotating the eigenvectors inside each degenerate eigenspace leaves the
    orthogonal projector — and hence the amplitude — unchanged.
    """
    eigs = circulant_eigenvalues(p, k)
    fbasis = fourier_basis(p)
    e0 = np.zeros(p)
    e0[0] = 1.0
    rng = np.random.default_rng(seed)
    resid = 0.0
    for grp in _groups(eigs, tol):
        v = fbasis[:, grp]
        a_ref = float((e0 @ (v @ v.conj().T) @ e0).real)
        m = len(grp)
        gmat = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        unitary, _ = np.linalg.qr(gmat)  # random unitary mixing of the eigenspace
        vr = v @ unitary
        a_rot = float((e0 @ (vr @ vr.conj().T) @ e0).real)
        resid = max(resid, abs(a_ref - a_rot))
    return resid


@dataclass(frozen=True)
class PulseAmplitudeCertificate:
    """The pointed-pulse amplitude spectrum on a normal circulant (R2, N12)."""

    p: int
    k: int
    n_distinct: int              # #amplitudes = #distinct eigenvalues
    rank_matches_cyclotomy: bool  # == gcd(k, p-1) + 1
    amplitudes_sum_to_one: bool
    amplitude_equals_multiplicity: bool  # a_λ == m_λ/p (projection check)
    moment_reconstruction_residual: float
    basis_invariance_residual: float
    tolerance: float
    claim_status: str


def certify_pulse_amplitudes(p: int, k: int, *, tol: float = 1e-9
                             ) -> PulseAmplitudeCertificate:
    r"""Bundle the amplitude theorem checks for the ``(p, k)`` pointed pulse."""
    amps = pulse_amplitudes(p, k, tol=tol)
    proj = projection_amplitudes(p, k, tol=tol)
    total = sum(a for _, _, a in amps)
    equals_mult = all(
        abs(pa - float(a)) < max(tol, 1e-6)
        for (_, _, a), (_, pa) in zip(amps, proj)
    )
    return PulseAmplitudeCertificate(
        p=p,
        k=k,
        n_distinct=len(amps),
        rank_matches_cyclotomy=(len(amps) == cyclotomic_rank(p, k)),
        amplitudes_sum_to_one=(total == Fraction(1)),
        amplitude_equals_multiplicity=equals_mult,
        moment_reconstruction_residual=moment_reconstruction_residual(p, k),
        basis_invariance_residual=amplitudes_basis_invariance_residual(p, k,
                                                                       tol=tol),
        tolerance=tol,
        claim_status=(
            "amplitudes = normalized spectral multiplicities m_λ/n DERIVED "
            "(uniform Fourier weight of e₀ on a circulant) + MEASURED (exact "
            "moment reconstruction, basis-invariant); NT-P02b"
        ),
    )
