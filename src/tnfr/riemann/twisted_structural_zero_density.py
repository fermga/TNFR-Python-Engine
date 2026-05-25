r"""TNFR-Riemann P46: chi-twisted structural zero density.

L-track analogue of P28 (:mod:`structural_zero_density`).  Derives the
smooth zero positions ``tilde gamma_n^{(chi)}`` of ``L(s, chi)`` for a
primitive real Dirichlet character ``chi`` purely from the archimedean
side of the chi-twisted Weil-Guinand explicit formula (P36) -- i.e.,
from the phase of the gamma factor of the completed L-function

    Lambda(s, chi) = (q/pi)^((s+a)/2) Gamma((s+a)/2) L(s, chi)

with ``a = 0`` if ``chi`` is even (``chi(-1) = +1``) and ``a = 1`` if
``chi`` is odd (``chi(-1) = -1``).

The L-track theta-like phase function is

    theta_chi(T) = Im log Gamma((1/2 + a)/2 + i T / 2)
                   + (T / 2) log(q / pi)

(reduces to the Riemann-Siegel ``theta(T)`` of P28 when ``q = 1``,
``a = 0``: ``log(q/pi) = -log pi``).  Backlund's smooth counting
function for ``L(s, chi)`` is then

    bar N_chi(T) = theta_chi(T) / pi + 1

with smooth derivative

    bar N_chi'(T) approx (1 / (2 pi)) log(q T / (2 pi)),

and the n-th smooth zero ``tilde gamma_n^{(chi)}`` is the unique
solution of ``bar N_chi(tilde gamma_n^{(chi)}) = n``.  No call to
``find_dirichlet_l_zeros`` is made on the **derivation** side; the
true zeros (from P36 Hardy-Z bisection) are used only as benchmark
for the residuals

    r_n^{(chi)} = gamma_n^{(chi)} - tilde gamma_n^{(chi)},

which encode the oscillating part ``S_chi(T) = (1/pi) arg L(1/2 + iT, chi)``.

What this closes (P46, operationally)
-------------------------------------
1. The smooth eigenvalue density of the chi-twisted Hilbert-Polya
   slot ``T_HP^{(chi)}`` (built in P45 by *inputting* Hardy-Z zeros)
   is a TNFR-derivable object: it falls out of the gamma factor of
   ``Lambda(s, chi)``, which is exactly the archimedean kernel of the
   chi-twisted Weil-Guinand formula computed in P36
   (``twisted_weil_archimedean_integral``).

2. The Wasserstein-1 gap
   ``W_1(spec(tilde T_HP^{(chi)}), spec(T_HP^{(chi)}))`` is
   dramatically smaller than the P45 baseline
   ``W_1(spec(P34|p not dividing q), spec(T_HP^{(chi)}))``: the
   reduction ratio quantifies how much of the L-track structural gap
   is *smooth-density* (TNFR-derivable) and how much is *arithmetic
   fluctuation* ``S_chi(T)``.

3. The residuals ``r_n^{(chi)}`` are bounded empirically by
   ``C log(gamma_n^{(chi)}) / bar N_chi'(gamma_n^{(chi)})`` for a
   small constant ``C`` (typical: ``C <= 2``).

What this does NOT close (GRH for L(s, chi) stays open)
-------------------------------------------------------
* The residuals ``r_n^{(chi)}`` ARE the GRH content for ``L(s, chi)``.
  Showing ``|r_n^{(chi)}| -> 0`` or ``|r_n^{(chi)}| <= C`` uniformly
  in ``n`` is equivalent to bounding ``S_chi(T)``.  That is the open
  arithmetic problem.

* Exact eigenvalue match
  ``spec(tilde T_HP^{(chi)}) = spec(T_HP^{(chi)})`` is impossible:
  the smooth approximation cannot reproduce the fluctuating
  ``gamma_n^{(chi)}``.

* P46 is the L-track structural mirror of P28; it does **not**
  contribute to closing gap G4 = RH or any GRH conjecture.

Status: EXPERIMENTAL -- TNFR-Riemann P46 (May 2026).  L-track
analogue of P28; derives the smooth chi-twisted zero density from
TNFR archimedean ingredients; quantifies the residual GRH-content
explicitly for the three primitive real characters ``chi_3``,
``chi_4``, ``chi_5``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mpmath

from ..mathematics.unified_numerical import np
from .dirichlet_l import DirichletCharacter
from .hilbert_polya import wasserstein_1_distance
from .twisted_hilbert_polya import fetch_chi_zero_imaginary_parts
from .twisted_weil_explicit_formula import character_parity


__all__ = [
    "twisted_theta",
    "twisted_smooth_zero_count",
    "twisted_smooth_zero_density",
    "derive_twisted_smooth_zero_position",
    "build_twisted_structural_t_hp",
    "TwistedStructuralZeroDensityCertificate",
    "compute_twisted_structural_zero_density_certificate",
]


# ----------------------------------------------------------------------
# Archimedean ingredients (entirely from gamma + log(q/pi))
# ----------------------------------------------------------------------


def twisted_theta(
    T: float,
    chi: DirichletCharacter,
    *,
    dps: int = 30,
) -> float:
    r"""Return the chi-twisted theta-like phase function.

    .. math::

        \theta_\chi(T) = \operatorname{Im}\log\Gamma\!\bigl(
            \tfrac{1/2 + a}{2} + \tfrac{iT}{2}\bigr)
            + \tfrac{T}{2}\log(q/\pi)

    where ``a = 0`` (resp. ``1``) if ``chi`` is even (resp. odd) and
    ``q`` is the modulus of ``chi``.

    This is the phase of the archimedean factor
    ``(q/pi)^((s+a)/2) Gamma((s+a)/2)`` of the completed L-function
    evaluated at ``s = 1/2 + iT``.  Identical to the kernel used in
    :func:`twisted_weil_archimedean_integral` (P36).
    """
    if T <= 0.0:
        raise ValueError("T must be strictly positive")
    a = character_parity(chi)
    q = int(chi.modulus)
    half_a = mpmath.mpf("0.5") * (mpmath.mpf("0.5") + mpmath.mpf(a))
    with mpmath.workdps(dps):
        val = mpmath.im(
            mpmath.loggamma(mpmath.mpc(half_a, T / 2.0))
        ) + (T / 2.0) * mpmath.log(mpmath.mpf(q) / mpmath.pi)
    return float(val)


def twisted_smooth_zero_count(
    T: float,
    chi: DirichletCharacter,
    *,
    dps: int = 30,
) -> float:
    r"""Backlund-style smooth zero counting for ``L(s, chi)``.

    .. math::

        \bar N_\chi(T) = \theta_\chi(T) / \pi + 1.

    Equals the average number of non-trivial zeros of ``L(s, chi)``
    with imaginary part in ``(0, T]`` up to the oscillating
    correction ``S_chi(T) = (1/pi) arg L(1/2 + iT, chi)``.
    """
    return twisted_theta(T, chi, dps=dps) / math.pi + 1.0


def twisted_smooth_zero_density(
    T: float,
    chi: DirichletCharacter,
) -> float:
    r"""Smooth chi-twisted zero density
    ``bar N_chi'(T) approx (1/(2 pi)) log(q T / (2 pi))``.

    Positive for ``T > 2 pi / q``; we add a conservative floor for
    very small ``T`` to keep the Newton iteration well-conditioned.
    """
    q = float(chi.modulus)
    arg = q * T / (2.0 * math.pi)
    if arg <= 1.0:
        return 1.0 / (2.0 * math.pi)
    return math.log(arg) / (2.0 * math.pi)


def derive_twisted_smooth_zero_position(
    n: int,
    chi: DirichletCharacter,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    dps: int = 30,
) -> float:
    r"""Newton-solve ``bar N_chi(T) = n`` for the n-th smooth
    chi-twisted zero ``tilde gamma_n^{(chi)}``.

    Uses an asymptotic initial guess
    ``T_n approx 2 pi n / log(q n)`` (for ``n >= 4``) and a small
    hard-coded seed table for ``n in {1, 2, 3}`` chosen slightly
    above the true ``gamma_1^{(chi)}`` to keep Newton inside the
    convex region of ``bar N_chi``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    q = float(chi.modulus)
    if n == 1:
        T = max(12.0 / max(math.sqrt(q), 1.0), 6.0)
    elif n == 2:
        T = max(18.0 / max(math.sqrt(q), 1.0), 9.0)
    elif n == 3:
        T = max(24.0 / max(math.sqrt(q), 1.0), 12.0)
    else:
        denom = max(math.log(q * float(n)), 1.0)
        T = 2.0 * math.pi * n / denom
    last_T = T
    for _ in range(max_iter):
        f = twisted_smooth_zero_count(T, chi, dps=dps) - float(n)
        fp = twisted_smooth_zero_density(T, chi)
        if fp <= 0.0:
            break
        delta = f / fp
        T_new = T - delta
        if T_new <= 0.0:
            T_new = 0.5 * T
        if abs(T_new - last_T) < tol:
            T = T_new
            break
        last_T = T
        T = T_new
    return float(T)


def build_twisted_structural_t_hp(
    N: int,
    chi: DirichletCharacter,
    *,
    dps: int = 30,
) -> np.ndarray:
    r"""Build ``tilde T_HP^{(chi)} = diag(tilde gamma_1^{(chi)}, ...,
    tilde gamma_N^{(chi)})``.

    Returns the sorted array of smooth chi-twisted zero positions
    derived ONLY from the archimedean theta-like function
    ``theta_chi``.  No call to ``find_dirichlet_l_zeros`` is made.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    out = np.empty(N, dtype=float)
    for k in range(N):
        out[k] = derive_twisted_smooth_zero_position(k + 1, chi, dps=dps)
    return out


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedStructuralZeroDensityCertificate:
    r"""Certificate of structurally-derived chi-twisted zero density (P46).

    Attributes
    ----------
    character_name, character_modulus, character_parity
        Identification of the primitive real Dirichlet character.
    n_zeros
        Number of zeros / smooth positions compared.
    structural_gammas
        ``(tilde gamma_1^{(chi)}, ..., tilde gamma_N^{(chi)})`` derived
        from ``theta_chi``.
    actual_gammas
        ``(gamma_1^{(chi)}, ..., gamma_N^{(chi)})`` from Hardy-Z
        bisection of ``L(s, chi)`` (benchmark only).
    residuals
        ``r_n^{(chi)} = gamma_n^{(chi)} - tilde gamma_n^{(chi)}`` --
        the oscillating part ``S_chi(gamma_n^{(chi)}) /
        bar N_chi'(gamma_n^{(chi)})``.
    max_residual, mean_residual, rms_residual
        Aggregate residual statistics.
    w1_structural_vs_actual
        ``W_1(spec(tilde T_HP^{(chi)}), spec(T_HP^{(chi)}))``.
    w1_p34_vs_actual
        ``W_1(spec(P34 | p not dividing q)|_{<= N},
        spec(T_HP^{(chi)}))`` -- the P45 baseline.
    improvement_ratio
        ``w1_p34_vs_actual / w1_structural_vs_actual``.
    bound_estimate, bound_satisfied
        Empirical check
        ``max_n |r_n^{(chi)}| <= C log(gamma_n^{(chi)}) /
        bar N_chi'(gamma_n^{(chi)})``.
    structurally_derived
        ``True`` since the derivation side never invokes Hardy-Z
        bisection.
    notes
        Honest-scope remarks.
    """

    character_name: str
    character_modulus: int
    character_parity: int
    n_zeros: int
    structural_gammas: tuple
    actual_gammas: tuple
    residuals: tuple
    max_residual: float
    mean_residual: float
    rms_residual: float
    w1_structural_vs_actual: float
    w1_p34_vs_actual: float
    improvement_ratio: float
    bound_estimate: float
    bound_satisfied: bool
    structurally_derived: bool
    notes: tuple

    def summary(self) -> str:
        lines = [
            "chi-twisted Structural Zero Density Certificate (P46)",
            "=" * 56,
            f"  character                     : {self.character_name}",
            f"  modulus q                     : {self.character_modulus}",
            f"  parity a                      : {self.character_parity}",
            f"  n_zeros                       : {self.n_zeros}",
            "  --- Per-zero residuals r_n = gamma_n - tilde gamma_n ---",
            f"  max |r_n|                     : {self.max_residual:.4e}",
            f"  mean |r_n|                    : {self.mean_residual:.4e}",
            f"  rms r_n                       : {self.rms_residual:.4e}",
            "  --- Operator-level L-track gap ---",
            f"  W_1(spec(P34|p!|q), T_HP^chi) : "
            f"{self.w1_p34_vs_actual:.4e}",
            f"  W_1(spec(t.T_HP^chi), T_HP^chi): "
            f"{self.w1_structural_vs_actual:.4e}",
            f"  improvement ratio             : "
            f"{self.improvement_ratio:.2f}x",
            "  --- Theoretical bound check ---",
            f"  C * max(log gamma_n / N'_chi) : "
            f"{self.bound_estimate:.4e}",
            f"  bound satisfied (C <= 2)      : "
            f"{self.bound_satisfied}",
            f"  structurally derived          : "
            f"{self.structurally_derived}",
        ]
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def compute_twisted_structural_zero_density_certificate(
    chi: DirichletCharacter,
    *,
    n_zeros: int = 30,
    dps: int = 30,
    p34_n_primes: int = 30,
    p34_max_power: int = 6,
    p34_spectrum: Sequence[float] | None = None,
    bound_constant: float = 2.0,
    hardy_z_initial_t_max: float = 60.0,
    hardy_z_initial_step: float = 0.25,
    hardy_z_max_doublings: int = 6,
) -> TwistedStructuralZeroDensityCertificate:
    r"""Compute the P46 chi-twisted structural-zero-density certificate.

    Parameters
    ----------
    chi
        Primitive real Dirichlet character (``chi_3``, ``chi_4`` or
        ``chi_5``).
    n_zeros
        Number of smooth/actual zeros to compare.
    dps
        Mpmath decimal precision for the gamma-function evaluations
        and Hardy-Z bisection.
    p34_n_primes, p34_max_power
        Parameters of the P34 chi-twisted prime-ladder Hamiltonian
        whose top ``n_zeros`` eigenvalues (after removing primes
        dividing the modulus) are used as the P45-equivalent
        spectrum for ``w1_p34_vs_actual``.
    p34_spectrum
        Optional pre-computed active P34 spectrum.  If supplied,
        ``p34_n_primes`` and ``p34_max_power`` are ignored.
    bound_constant
        Constant ``C`` in the empirical bound check.
    hardy_z_initial_t_max, hardy_z_initial_step, hardy_z_max_doublings
        Passed through to :func:`fetch_chi_zero_imaginary_parts`.
    """
    if n_zeros < 1:
        raise ValueError("n_zeros must be >= 1")

    structural = build_twisted_structural_t_hp(n_zeros, chi, dps=dps)
    actual = fetch_chi_zero_imaginary_parts(
        chi,
        n_zeros,
        initial_t_max=hardy_z_initial_t_max,
        initial_step=hardy_z_initial_step,
        dps=dps,
        max_doublings=hardy_z_max_doublings,
    )
    residuals = actual - structural
    abs_res = np.abs(residuals)

    w1_struct = wasserstein_1_distance(structural, actual)

    if p34_spectrum is None:
        from .twisted_prime_ladder_hamiltonian import (
            build_twisted_prime_ladder_hamiltonian,
        )

        bundle = build_twisted_prime_ladder_hamiltonian(
            chi,
            n_primes=p34_n_primes,
            max_power=p34_max_power,
        )
        # Active sector: prime-ladder spectrum mu_{p,k} = k log p
        # over primes coprime to q (primes dividing q already excluded
        # at construction since chi(p) = 0 kills their weights).
        spec = np.sort(
            np.asarray(bundle.spectrum.eigenvalues, dtype=float)
        )
    else:
        spec = np.sort(np.asarray(p34_spectrum, dtype=float))

    if spec.size >= n_zeros:
        p34_top = spec[:n_zeros]
    else:
        pad = np.full(
            n_zeros - spec.size, spec[-1] if spec.size > 0 else 0.0
        )
        p34_top = np.concatenate([spec, pad])

    w1_p34 = wasserstein_1_distance(p34_top, actual)
    if w1_struct > 0.0:
        improvement = w1_p34 / w1_struct
    else:
        improvement = float("inf")

    densities = np.array(
        [twisted_smooth_zero_density(float(g), chi) for g in actual],
        dtype=float,
    )
    log_gammas = np.log(actual)
    bound_per_n = bound_constant * log_gammas / np.where(
        densities > 0.0, densities, 1.0
    )
    bound_estimate = float(np.max(bound_per_n))
    bound_satisfied = bool(np.max(abs_res) <= bound_estimate)

    notes = (
        "tilde gamma_n derived from theta_chi(T) = "
        "Im log Gamma((1/2+a)/2 + iT/2) + (T/2) log(q/pi).",
        "No find_dirichlet_l_zeros call on the DERIVATION side "
        "(only for benchmark).",
        "Residuals r_n encode S_chi(gamma_n) = "
        "(1/pi) arg L(1/2 + i gamma_n, chi).",
        "Does NOT close GRH for L(s, chi) or G4 = RH: bounding "
        "S_chi(T) is the open arithmetic problem.  Closes the "
        "structural origin of the smooth chi-twisted density.",
    )

    return TwistedStructuralZeroDensityCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        character_parity=int(character_parity(chi)),
        n_zeros=int(n_zeros),
        structural_gammas=tuple(float(x) for x in structural),
        actual_gammas=tuple(float(x) for x in actual),
        residuals=tuple(float(x) for x in residuals),
        max_residual=float(np.max(abs_res)),
        mean_residual=float(np.mean(abs_res)),
        rms_residual=float(math.sqrt(float(np.mean(residuals ** 2)))),
        w1_structural_vs_actual=float(w1_struct),
        w1_p34_vs_actual=float(w1_p34),
        improvement_ratio=float(improvement),
        bound_estimate=float(bound_estimate),
        bound_satisfied=bool(bound_satisfied),
        structurally_derived=True,
        notes=notes,
    )
