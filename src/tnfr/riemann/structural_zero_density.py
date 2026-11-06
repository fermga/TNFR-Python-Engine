r"""Classical smooth-counting targets and finite zero comparisons (P28).

The source of the targets is the classical Riemann-Siegel theta function
 theta(T)=Im log Gamma(1/4+i*T/2)-(T/2)*log(pi),
with smooth count Nbar(T)=theta(T)/pi+1 under the implemented branch convention.
The code numerically inverts this function to prescribe a finite target list.
It does not derive theta, its gamma factor or target selection from the nodal
identity. A diagonal matrix containing those targets has them as eigenvalues
by construction.

Known zero ordinates are used separately for finite error comparisons. A smaller
Wasserstein discrepancy than the prime-ladder baseline is a comparison of
chosen targets, not a percentage of RH proved. The analytic argument term S(T)
is not itself an RH-equivalent proposition without a precise quantified
statement and proof. Local density-based residual estimates need their own
branch, derivative and domain assumptions; a finite fit is not a universal
bound. No nodal Hilbert-Polya bridge or all-zero location theorem follows.

Current interpretation and historical records are maintained in
 theory/TNFR_RIEMANN_RESEARCH_NOTES.md."""

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
    r"""Evaluate the classical Riemann-Siegel theta function at the requested precision.

    This reads Im(loggamma(1/4+i*T/2))-(T/2)*log(pi) with the library's continued
    log-gamma convention. The special function is an arithmetic input, not a
    nodal-law derivation."""
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
    r"""Return a floored leading-order approximation log(T/(2*pi))/(2*pi).

    This is not the exact derivative of theta(T)/pi+1. For T<=2*pi the
    implementation returns 1/(2*pi), a numerical Newton-step policy; that floor
    is not a derived physical constant or analytic lower-bound theorem."""
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
    r"""Return N supplied smooth targets obtained by numerically inverting theta counting.

    No zero oracle is called to construct this array. That independence does
    not derive its classical theta function or a Hilbert-Polya mechanism from
    the nodal equation."""
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
    r"""Finite comparison of classical smooth targets with supplied zero ordinates.

    structural_gammas are numerically inverted theta-counting targets;
    actual_gammas use mpmath.zetazero. residuals are their differences, not an
    exact equality with S(gamma)/Nbar_prime(gamma). The W1 and aggregate fields
    report the finite comparison. bound_satisfied checks the maximum residual
    against the maximum chosen envelope value, not a pointwise bound at every
    index or an analytic all-height theorem.

    structurally_derived is a legacy flag meaning the smooth target construction
    does not call the zero oracle; it does not derive the targets from nodal
    physics. The returned notes clarify this compatibility interpretation."""

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
        "Smooth targets invert the supplied classical theta counting function "
        "theta(T)/pi+1; they are not derived from a nodal evolution.",
        "No zero oracle is used for the smooth target construction; known "
        "ordinates are used separately for this finite comparison.",
        "Residuals are differences between known ordinates and smooth targets. "
        "They are not exactly S(gamma)/Nbar_prime(gamma) or an RH criterion.",
        "The legacy structurally_derived flag records construction provenance. "
        "The envelope check compares two finite maxima, not pointwise or "
        "all-height bounds. No nodal bridge or RH result is established.",
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
