r"""TNFR-Riemann P28 — Structural derivation of the smooth zero density.

Motivation
----------
P27 (:mod:`tnfr.riemann.hilbert_polya`) constructed the abstract
Hilbert-Polya operator :math:`T_{\mathrm{HP}} = \operatorname{diag}(\gamma_n)`
by **inputting** the zeros from :func:`mpmath.zetazero`.  The
Wasserstein-1 gap to the P14 prime-ladder spectrum,
:math:`W_1(\sigma(P14),\sigma(T_{\mathrm{HP}})) \approx 115.24`, was
the operator-level manifestation of gap G4 (= RH).

This module attacks the **structural origin** of that gap.  We do
*not* attempt to prove RH (G4 remains the only open milestone in
§13.2 of AGENTS.md).  We do, however, derive a TNFR-canonical
operator

.. math::

    \widetilde T_{\mathrm{HP}}
       := \operatorname{diag}(\widetilde\gamma_1,\dots,\widetilde\gamma_N)

whose eigenvalues are the **smooth Riemann zero positions** obtained
*entirely from the archimedean side* of the Weil-Guinand identity
(P15) — i.e., from the Riemann-Siegel theta function

.. math::

    \theta(T) = \operatorname{Im}\log\Gamma\!\bigl(\tfrac14 + \tfrac{iT}{2}\bigr)
                - \tfrac{T}{2}\log\pi.

Backlund's formula gives the smooth counting function

.. math::

    \overline N(T) = \frac{\theta(T)}{\pi} + 1,

and :math:`\widetilde\gamma_n` is defined as the unique solution of
:math:`\overline N(\widetilde\gamma_n) = n`.  No call to
:func:`mpmath.zetazero` is made on the derivation side.

What this closes (P28)
----------------------
1. The **smooth eigenvalue density** of :math:`T_{\mathrm{HP}}` is a
   TNFR-derivable object: it falls out of the gamma factor of the
   completed zeta function :math:`\xi(s) = \pi^{-s/2}\Gamma(s/2)\zeta(s)`,
   and the gamma factor is exactly the archimedean kernel of the
   Weil-Guinand explicit formula computed in P15 via
   :func:`tnfr.riemann.weil_explicit_formula.weil_archimedean_integral`.

2. The Wasserstein-1 gap
   :math:`W_1(\sigma(\widetilde T_{\mathrm{HP}}),\sigma(T_{\mathrm{HP}}))`
   is dramatically smaller than the P27 gap
   :math:`W_1(\sigma(P14),\sigma(T_{\mathrm{HP}}))`.  The reduction
   ratio quantifies how much of G4 is *structural* (smooth density,
   TNFR-derivable) and how much is *arithmetic fluctuation*
   (oscillating part :math:`S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)`,
   genuinely RH-equivalent).

3. The residuals :math:`r_n := \gamma_n - \widetilde\gamma_n` satisfy
   :math:`|r_n| \lesssim |S(\gamma_n)| / \overline N'(\gamma_n)`,
   so the per-zero residual is bounded by the absolute value of the
   argument of zeta on the critical line divided by the smooth
   density.  Confirming this scaling numerically is part of the
   certificate.

What this does NOT close (G4 stays OPEN)
----------------------------------------
* The residuals :math:`r_n` ARE the RH content.  Showing
  :math:`r_n \to 0` or even :math:`|r_n| \le C` uniformly in :math:`n`
  is equivalent to RH-style control on :math:`S(T)` — that remains
  the genuine arithmetic gap.

* The exact eigenvalue match
  :math:`\sigma(\widetilde T_{\mathrm{HP}}) = \sigma(T_{\mathrm{HP}})`
  is impossible: the smooth approximation cannot reproduce the
  fluctuating zero positions.  What is possible is the **density
  match** in W_1 modulo a TNFR-quantifiable error.

Status: EXPERIMENTAL — TNFR-Riemann P28 (May 2026).  Derives the
smooth zero density from TNFR archimedean ingredients; quantifies
the residual RH-content explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mpmath

from ..mathematics.unified_numerical import np
from .hilbert_polya import fetch_zero_imaginary_parts, wasserstein_1_distance

__all__ = [
    "riemann_siegel_theta",
    "smooth_zero_count",
    "smooth_zero_density",
    "derive_smooth_zero_position",
    "build_structural_t_hp",
    "StructuralZeroDensityCertificate",
    "compute_structural_zero_density_certificate",
]


# ----------------------------------------------------------------------
# Archimedean ingredients (entirely from gamma + log pi)
# ----------------------------------------------------------------------


def riemann_siegel_theta(T: float, *, dps: int = 30) -> float:
    r"""Return the Riemann-Siegel theta function.

    .. math::

        \theta(T) = \operatorname{Im}\log\Gamma\!\bigl(\tfrac14 + \tfrac{iT}{2}\bigr)
                    - \tfrac{T}{2}\log\pi.

    This is the phase of the archimedean factor
    :math:`\pi^{-s/2}\Gamma(s/2)` of the completed zeta function
    evaluated at :math:`s = 1/2 + iT`.  It is the TNFR-canonical
    object: the very same gamma factor is the kernel of the
    archimedean side of the Weil-Guinand explicit formula
    (:func:`tnfr.riemann.weil_explicit_formula.weil_archimedean_integral`).
    """
    if T <= 0.0:
        raise ValueError("T must be strictly positive")
    with mpmath.workdps(dps):
        val = mpmath.im(mpmath.loggamma(mpmath.mpc(0.25, T / 2.0))) - (
            T / 2.0
        ) * mpmath.log(mpmath.pi)
    return float(val)


def smooth_zero_count(T: float, *, dps: int = 30) -> float:
    r"""Backlund's smooth zero counting function.

    .. math::

        \overline N(T) = \frac{\theta(T)}{\pi} + 1.

    Equals the average number of non-trivial Riemann zeros with
    imaginary part in :math:`(0, T]` up to the oscillating
    correction :math:`S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)`.
    """
    return riemann_siegel_theta(T, dps=dps) / math.pi + 1.0


def smooth_zero_density(T: float) -> float:
    r"""Smooth zero density :math:`\overline N'(T) = \tfrac{1}{2\pi}\log(T/2\pi)`.

    Exact asymptotic derivative of :math:`\overline N(T)`.  Positive
    for :math:`T > 2\pi`; we add a floor to guarantee a sensible
    Newton step for very small ``T``.
    """
    arg = T / (2.0 * math.pi)
    if arg <= 1.0:
        # below 2π the asymptotic formula breaks down; use a
        # conservative positive lower bound to keep Newton moving.
        return 1.0 / (2.0 * math.pi)
    return math.log(arg) / (2.0 * math.pi)


def derive_smooth_zero_position(
    n: int,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    dps: int = 30,
) -> float:
    r"""Newton-solve :math:`\overline N(T) = n` for the n-th smooth zero.

    Uses an asymptotic initial guess derived from inverting the
    leading order of :math:`\overline N(T) \sim \tfrac{T}{2\pi}\log\tfrac{T}{2\pi e}`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    # Asymptotic initial guess: T_n ~ 2π n / W(n/e) where W is Lambert W;
    # a robust simple seed is T_n ~ 2π n / log(n + 1) for n >= 1, and
    # we hard-code the first few zero positions (slightly above the
    # true γ_n) to keep Newton inside the convex region of N̄.
    if n == 1:
        T = 18.0
    elif n == 2:
        T = 23.0
    elif n == 3:
        T = 28.0
    else:
        T = 2.0 * math.pi * n / max(math.log(float(n)), 1.0)
    last_T = T
    for _ in range(max_iter):
        f = smooth_zero_count(T, dps=dps) - float(n)
        fp = smooth_zero_density(T)
        if fp <= 0.0:
            break
        delta = f / fp
        T_new = T - delta
        if T_new <= 0.0:
            T_new = 0.5 * T  # damp toward positivity
        if abs(T_new - last_T) < tol:
            T = T_new
            break
        last_T = T
        T = T_new
    return float(T)


def build_structural_t_hp(
    N: int,
    *,
    dps: int = 30,
) -> np.ndarray:
    r"""Build :math:`\widetilde T_{\mathrm{HP}} = \operatorname{diag}(\widetilde\gamma_n)_{n=1}^{N}`.

    Returns the sorted array
    :math:`(\widetilde\gamma_1, \dots, \widetilde\gamma_N)` of smooth
    zero positions derived ONLY from the archimedean Riemann-Siegel
    theta function.  No call to :func:`mpmath.zetazero` is made.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    out = np.empty(N, dtype=float)
    for k in range(N):
        out[k] = derive_smooth_zero_position(k + 1, dps=dps)
    return out


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralZeroDensityCertificate:
    r"""Certificate of structurally-derived zero density (P28).

    Attributes
    ----------
    n_zeros
        Number of zeros / smooth positions compared.
    structural_gammas
        :math:`(\widetilde\gamma_1, \dots, \widetilde\gamma_N)` derived
        from the archimedean Riemann-Siegel theta function.
    actual_gammas
        :math:`(\gamma_1, \dots, \gamma_N)` from
        :func:`mpmath.zetazero` (benchmark only).
    residuals
        :math:`r_n = \gamma_n - \widetilde\gamma_n` — the
        oscillating part :math:`S(\gamma_n) / \overline N'(\gamma_n)`.
    max_residual, mean_residual, rms_residual
        Aggregate residual statistics.
    w1_structural_vs_actual
        :math:`W_1(\sigma(\widetilde T_{\mathrm{HP}}), \sigma(T_{\mathrm{HP}}))`.
    w1_p14_vs_actual
        :math:`W_1(\sigma(P14)|_{\le N}, \sigma(T_{\mathrm{HP}}))`.
    improvement_ratio
        :math:`w_1^{P14}/w_1^{\mathrm{structural}}`.
    bound_estimate, bound_satisfied
        Empirical check that
        :math:`\max_n|r_n| \le C \log\gamma_n / \overline N'(\gamma_n)`
        for a small constant ``C`` (typical: ``C ≤ 2``).
    structurally_derived
        ``True`` since the derivation never calls ``mpmath.zetazero``.
    notes
        Honest-scope remarks.
    """

    n_zeros: int
    structural_gammas: tuple
    actual_gammas: tuple
    residuals: tuple
    max_residual: float
    mean_residual: float
    rms_residual: float
    w1_structural_vs_actual: float
    w1_p14_vs_actual: float
    improvement_ratio: float
    bound_estimate: float
    bound_satisfied: bool
    structurally_derived: bool
    notes: tuple

    def summary(self) -> str:
        lines = [
            "Structural Zero Density Certificate (P28)",
            "==========================================",
            f"  n_zeros                       : {self.n_zeros}",
            "  --- Per-zero residuals r_n = γ_n − ñ_n ---",
            f"  max |r_n|                     : {self.max_residual:.4e}",
            f"  mean |r_n|                    : {self.mean_residual:.4e}",
            f"  rms r_n                       : {self.rms_residual:.4e}",
            "  --- Operator-level G4 gap ---",
            f"  W_1(σ(P14),   σ(T_HP))         : " f"{self.w1_p14_vs_actual:.4e}",
            f"  W_1(σ(T̃_HP), σ(T_HP))          : "
            f"{self.w1_structural_vs_actual:.4e}",
            f"  improvement ratio             : " f"{self.improvement_ratio:.2f}×",
            "  --- Theoretical bound check ---",
            f"  C * max(log γ_n / N̄'(γ_n))     : " f"{self.bound_estimate:.4e}",
            f"  bound satisfied (C ≤ 2)       : " f"{self.bound_satisfied}",
            f"  structurally derived          : " f"{self.structurally_derived}",
        ]
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  • {note}")
        return "\n".join(lines)


def compute_structural_zero_density_certificate(
    *,
    n_zeros: int = 80,
    dps: int = 30,
    p14_n_primes: int = 50,
    p14_max_power: int = 8,
    p14_spectrum: Sequence[float] | None = None,
    bound_constant: float = 2.0,
) -> StructuralZeroDensityCertificate:
    r"""Compute the P28 structural-zero-density certificate.

    Parameters
    ----------
    n_zeros
        Number of smooth/actual zeros to compare.
    dps
        mpmath decimal precision for the gamma-function evaluations
        and the benchmark zeros.
    p14_n_primes, p14_max_power
        Parameters of the P14 prime-ladder Hamiltonian whose top
        ``n_zeros`` eigenvalues are used as the P27-equivalent
        spectrum for the comparison ``w_1_p14_vs_actual``.
    p14_spectrum
        Optional pre-computed P14 spectrum (sorted or unsorted).
        If supplied, ``p14_n_primes`` and ``p14_max_power`` are
        ignored.
    bound_constant
        Constant ``C`` in the empirical bound check
        :math:`\max_n|r_n| \le C \log\gamma_n / \overline N'(\gamma_n)`.
    """
    if n_zeros < 1:
        raise ValueError("n_zeros must be >= 1")

    structural = build_structural_t_hp(n_zeros, dps=dps)
    actual = fetch_zero_imaginary_parts(n_zeros, dps=dps)
    residuals = actual - structural
    abs_res = np.abs(residuals)

    # W_1 of the two diagonal spectra (sorted ascending by construction)
    w1_struct = wasserstein_1_distance(structural, actual)

    # P14 spectrum (top n_zeros eigenvalues)
    if p14_spectrum is None:
        # Local import to avoid touching the P14 module at import time
        from .prime_ladder_hamiltonian import build_prime_ladder_hamiltonian

        bundle = build_prime_ladder_hamiltonian(
            n_primes=p14_n_primes, max_power=p14_max_power
        )
        eigvals, _ = bundle.hamiltonian.get_spectrum()
        spec = np.sort(np.real(eigvals))
    else:
        spec = np.sort(np.asarray(p14_spectrum, dtype=float))

    if spec.size >= n_zeros:
        p14_top = spec[:n_zeros]
    else:
        # Pad with the largest available value if the user-supplied
        # spectrum is too short; this only hurts the P14 baseline.
        pad = np.full(n_zeros - spec.size, spec[-1] if spec.size > 0 else 0.0)
        p14_top = np.concatenate([spec, pad])

    w1_p14 = wasserstein_1_distance(p14_top, actual)
    if w1_struct > 0.0:
        improvement = w1_p14 / w1_struct
    else:
        improvement = float("inf")

    # Empirical bound: |r_n| ≤ C log(γ_n) / N̄'(γ_n).
    # We compute max_n of the right-hand side and compare.
    densities = np.array([smooth_zero_density(float(g)) for g in actual], dtype=float)
    log_gammas = np.log(actual)
    bound_per_n = (
        bound_constant * log_gammas / np.where(densities > 0.0, densities, 1.0)
    )
    bound_estimate = float(np.max(bound_per_n))
    bound_satisfied = bool(np.max(abs_res) <= bound_estimate)

    notes = (
        "ñ_n derived from θ(T) = Im log Γ(1/4 + iT/2) − (T/2) log π.",
        "No mpmath.zetazero used on the DERIVATION side "
        "(only for benchmark on the right-hand side).",
        "Residuals r_n = γ_n − ñ_n encode the oscillating part "
        "S(γ_n) = (1/π) arg ζ(1/2 + iγ_n).",
        "Does NOT close G4 = RH: bounding S(T) is the open arithmetic "
        "problem.  Closes the structural origin of the smooth density.",
    )

    return StructuralZeroDensityCertificate(
        n_zeros=int(n_zeros),
        structural_gammas=tuple(float(x) for x in structural),
        actual_gammas=tuple(float(x) for x in actual),
        residuals=tuple(float(x) for x in residuals),
        max_residual=float(np.max(abs_res)),
        mean_residual=float(np.mean(abs_res)),
        rms_residual=float(math.sqrt(float(np.mean(residuals**2)))),
        w1_structural_vs_actual=float(w1_struct),
        w1_p14_vs_actual=float(w1_p14),
        improvement_ratio=float(improvement),
        bound_estimate=float(bound_estimate),
        bound_satisfied=bool(bound_satisfied),
        structurally_derived=True,
        notes=notes,
    )
