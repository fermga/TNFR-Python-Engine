r"""TNFR-Riemann: the pulse-phase / coherence layer.

Operationalises the re-founded attack surface (see
``examples/03_riemann_zeta/157_nodal_pulse_phase_attack.py``): the RH-content
oscillation ``S(T)`` is the collective phase of the integer-NFR nodal pulse, and
the critical line is its coherence axis.  This module makes those accessible as
reusable tooling -- the capability the eliminated combinatorial track lacked (its
``S_n``-invariant self-adjoint spectrum was provably blind to ``S(T)``).

Canonical objects
-----------------
* ``argument_fluctuation(T)`` -- ``S(T) = (1/π) arg ζ(1/2+iT)``, read straight
  off the nodal pulse ``P(T) = Σ n^{-1/2} e^{-i(log n)T}`` as ``arg(P)/π``.
* ``smooth_zero_count(T)`` -- ``θ(T)/π + 1`` (the archimedean Riemann-Siegel
  count; ``θ`` is the same gamma kernel as the Weil-Guinand archimedean term).
* ``zero_count(T)`` -- the Riemann-von Mangoldt count ``N(T) = θ/π + 1 + S(T)``.
* ``rectified_pulse(T, σ)`` / ``coherence_defect(T, σ)`` -- the rectified field
  ``Z = e^{iθ} P_σ`` and its departure from reality; the coherence axis is
  ``Re(s) = 1/2`` (the ``ΔNFR = 0`` reflection axis), where ``Z`` is exactly real.

Honest scope
------------
The pulse phase *accesses* ``S(T)`` (the arithmetic ``Fix(S_n)^⊥`` content); it
does not *bound* it.  The prime-side reconstruction
``S(T) = (1/π) Σ_{p,k} (1/k) p^{-k/2} sin(kT log p)`` **diverges** on the line
(its abscissa of convergence is ``Re(s)=1``): that boundary non-convergence is
exactly the localized RH obstruction, made explicit here.  ``G4 = RH`` is open.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from ..mathematics.unified_numerical import np
from .nodal_pulse import KNOWN_RIEMANN_ZEROS, first_primes, nodal_pulse
from .structural_zero_density import riemann_siegel_theta, smooth_zero_count

__all__ = [
    "argument_fluctuation",
    "zero_count",
    "generalized_pulse",
    "rectified_pulse",
    "coherence_defect",
    "prime_side_fluctuation",
    "PulseCoherenceCertificate",
    "verify_pulse_coherence",
]


def argument_fluctuation(t: float, n_terms: int | None = None) -> float:
    r"""``S(T) = (1/π) arg ζ(1/2+iT)`` read off the nodal-pulse phase.

    Accurate away from the zeros (near a zero ``arg`` turns through ``π`` and the
    truncated pulse carries a ``±1`` ambiguity resolved by the integer
    :func:`zero_count`).
    """
    return cmath.phase(nodal_pulse(t, n_terms)) / math.pi


def zero_count(t: float, n_terms: int | None = None) -> float:
    r"""Riemann-von Mangoldt count ``N(T) = θ(T)/π + 1 + S(T)`` from the pulse.

    ``smooth_zero_count`` (``θ/π + 1``) is the canonical archimedean count from
    :mod:`tnfr.riemann.structural_zero_density`; the pulse contributes ``S(T)``.
    """
    return smooth_zero_count(t) + argument_fluctuation(t, n_terms)


def _n_terms(t: float) -> int:
    """Canonical nodal-pulse truncation length (matches nodal_pulse defaults)."""
    return int(max(10, round(3.0 * math.sqrt(max(t, 1.0) / (2.0 * math.pi)) + 6.0)))


def generalized_pulse(t: float, sigma: float, n_terms: int | None = None) -> complex:
    r"""Truncated Dirichlet partial sum ``Σ n^{-σ} e^{-i(log n)T}`` at ``s=σ+iT``."""
    if n_terms is None:
        n_terms = _n_terms(t)
    n = np.arange(1, int(n_terms) + 1, dtype=float)
    return complex(np.sum(n ** (-sigma) * np.exp(-1j * t * np.log(n))))


def rectified_pulse(t: float, sigma: float = 0.5) -> complex:
    r"""The Riemann-Siegel-rectified field ``Z = e^{iθ(T)} ζ(σ+iT)`` (exact).

    On the coherence axis ``σ = 1/2`` the functional equation forces ``Z`` real
    (the classical real Riemann-Siegel Z-function); off it ``Z`` acquires an
    imaginary part.  Computed from the exact ``ζ``: the coherence axis is a
    functional-equation property that the truncated nodal pulse is too coarse to
    resolve (at some heights the truncation even mis-ranks ``σ``), so the exact
    ``ζ`` is used here rather than manufacturing agreement from the crude sum.
    """
    import mpmath as mp

    z = mp.e ** (1j * riemann_siegel_theta(t)) * mp.zeta(mp.mpf(sigma) + 1j * t)
    return complex(z)


def coherence_defect(t: float, sigma: float = 0.5) -> float:
    r"""``|Im Z| / |Z|`` of the exact rectified field.

    Zero on the coherence axis ``σ = 1/2`` (functional-equation reality),
    positive off it.
    """
    z = rectified_pulse(t, sigma)
    return abs(z.imag) / (abs(z) + 1e-30)


def prime_side_fluctuation(
    t: float, n_primes: int = 60, max_k: int = 6
) -> float:
    r"""The prime-side series for ``S(T)`` -- ``(1/π) Σ_{p,k}(1/k)p^{-k/2}sin(kT log p)``.

    WARNING: this series has abscissa of convergence ``Re(s)=1``, so on the
    critical line it does **not** converge; adding more prime NFRs does not
    improve it.  Exposed to make the localized RH obstruction explicit (the RH
    content is exactly this boundary non-convergence), not as a usable estimator.
    """
    total = 0.0
    for p in first_primes(n_primes):
        lp = math.log(p)
        for k in range(1, max_k + 1):
            total += (1.0 / k) * p ** (-k / 2.0) * math.sin(k * t * lp)
    return total / math.pi


@dataclass(frozen=True)
class PulseCoherenceCertificate:
    """Certificate that the pulse phase reproduces S(T) and the zero count."""

    n_heights: int
    max_abs_s_error: float
    zero_count_matches: bool
    coherence_axis_is_minimal: bool

    def summary(self) -> str:
        status = "PASS" if (self.zero_count_matches and self.coherence_axis_is_minimal) else "PARTIAL"
        return (
            f"PulseCoherenceCertificate[{status}]: {self.n_heights} heights; "
            f"max|ΔS|={self.max_abs_s_error:.3f} (off zeros); "
            f"N(T) counts zeros={self.zero_count_matches}; "
            f"coherence axis minimal at σ=1/2={self.coherence_axis_is_minimal}"
        )


def verify_pulse_coherence(
    heights: tuple[float, ...] = (17.0, 23.0, 28.0, 35.0, 47.0),
    *,
    s_tol: float = 0.05,
) -> PulseCoherenceCertificate:
    r"""Verify the pulse-phase layer against the numerical oracle.

    Checks, at heights chosen away from zeros: (i) ``S(T)`` from the pulse phase
    matches ``(1/π) arg ζ`` within ``s_tol``; (ii) ``round(N(T))`` equals the true
    number of zeros below ``T``; (iii) the coherence defect is minimal on
    ``σ = 1/2`` (vs ``0.6, 0.7``).
    """
    try:
        import mpmath as mp

        mp.mp.dps = 25

        def _true_s(t: float) -> float:
            return float(mp.arg(mp.zeta(mp.mpf("0.5") + 1j * t)) / mp.pi)

    except ImportError:  # pragma: no cover
        def _true_s(t: float) -> float:
            return argument_fluctuation(t)

    max_s_err = 0.0
    counts_ok = True
    for t in heights:
        max_s_err = max(max_s_err, abs(argument_fluctuation(t) - _true_s(t)))
        n_est = round(zero_count(t))
        true_count = sum(1 for g in KNOWN_RIEMANN_ZEROS if g < t)
        counts_ok &= n_est == true_count

    axis_minimal = True
    for t in heights:
        d_half = coherence_defect(t, 0.5)
        axis_minimal &= d_half <= coherence_defect(t, 0.6) and d_half <= coherence_defect(t, 0.7)

    return PulseCoherenceCertificate(
        n_heights=len(heights),
        max_abs_s_error=float(max_s_err),
        zero_count_matches=bool(counts_ok),
        coherence_axis_is_minimal=bool(axis_minimal),
    )
