r"""Finite arithmetic phase diagnostics and numerical zeta comparisons.

``argument_fluctuation(T)`` returns the principal phase of the finite sum
``P_N(T)=sum(n**(-1/2)*exp(-i*log(n)*T), n=1..N)``, divided by pi.
``zero_count(T)`` adds that diagnostic to the shared smooth Riemann-Siegel
count. These retained API names do not identify the partial sum with analytic
zeta, specify the continuous argument branch defining the classical S(T), or
certify an integer zero count. Finite oracle comparisons remain useful under
their declared truncation, heights and tolerances.

``rectified_pulse`` instead evaluates ``exp(i*theta(T))*zeta(sigma+i*T)``
numerically with mpmath and converts it to a Python complex value. The exact
functional equation makes this expression real on sigma=1/2; numerical values
retain approximation error. ``coherence_defect`` measures its regularized
relative imaginary part, not nodal pressure or physical coherence.

``prime_side_fluctuation`` is a separately truncated prime-power sine sum.
The absolutely convergent Euler-product expansion at Re(s)>1 supplies no
automatic error bound for this unregularized critical-line truncation.
Neither construction derives frequencies from the nodal law, identifies a
DeltaNFR=0 axis, establishes an autonomous TNFR pulse, or proves RH.
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
    r"""Principal ``arg(P_N(T))/pi`` of the declared finite arithmetic sum.

    The principal branch can jump at its cut. No analytic-continuation error
    or continuous zeta-argument branch is supplied, and :func:`zero_count`
    does not resolve that missing branch information.
    """
    return cmath.phase(nodal_pulse(t, n_terms)) / math.pi


def zero_count(t: float, n_terms: int | None = None) -> float:
    r"""Return the smooth count plus the finite-sum principal-phase diagnostic.

    ``smooth_zero_count`` supplies ``theta(T)/pi+1``. The added phase comes
    from :func:`argument_fluctuation`; this finite real-valued estimate is
    not an exact Riemann-von Mangoldt count. Rounding it is a comparison
    procedure, with no general zero-count or remainder certificate.
    """
    return smooth_zero_count(t) + argument_fluctuation(t, n_terms)


def _n_terms(t: float) -> int:
    """Finite truncation policy matching the nodal_pulse default length."""
    return int(max(10, round(3.0 * math.sqrt(max(t, 1.0) / (2.0 * math.pi)) + 6.0)))


def generalized_pulse(t: float, sigma: float, n_terms: int | None = None) -> complex:
    r"""Truncated Dirichlet partial sum ``Σ n^{-σ} e^{-i(log n)T}`` at ``s=σ+iT``."""
    if n_terms is None:
        n_terms = _n_terms(t)
    n = np.arange(1, int(n_terms) + 1, dtype=float)
    return complex(np.sum(n ** (-sigma) * np.exp(-1j * t * np.log(n))))


def rectified_pulse(t: float, sigma: float = 0.5) -> complex:
    r"""Numerically evaluate ``exp(i*theta(T))*zeta(sigma+i*T)``.

    This uses mpmath's zeta evaluation and returns a Python complex value,
    independently of :func:`generalized_pulse`. For real T, the exact
    functional equation gives the real Riemann-Siegel Z-function at sigma=1/2.
    The numerical result has finite precision; reality at a point neither
    identifies nodal pressure nor characterizes the critical line uniquely.
    """
    import mpmath as mp

    z = mp.e ** (1j * riemann_siegel_theta(t)) * mp.zeta(mp.mpf(sigma) + 1j * t)
    return complex(z)


def coherence_defect(t: float, sigma: float = 0.5) -> float:
    r"""Return ``abs(Im Z)/(abs(Z)+1e-30)`` for the numerical rectified value.

    The denominator includes the implemented numerical regularizer. Exact
    critical-line reality motivates this diagnostic, but finite residuals
    remain and zero can also occur away from that line. This is not a
    TNFR pressure or a certified zero-location test.
    """
    z = rectified_pulse(t, sigma)
    return abs(z.imag) / (abs(z) + 1e-30)


def prime_side_fluctuation(t: float, n_primes: int = 60, max_k: int = 6) -> float:
    r"""Evaluate the finite ``sum(p**(-k/2)*sin(k*T*log(p))/k)/pi``.

    Prime and power cutoffs are supplied numerical choices. The Euler-product
    argument from Re(s)>1 does not give this critical-line truncation a
    certified limit, zeta-argument branch or error bound. Increasing a cutoff
    is not itself a convergence test or a reformulation of RH.
    """
    total = 0.0
    for p in first_primes(n_primes):
        lp = math.log(p)
        for k in range(1, max_k + 1):
            total += (1.0 / k) * p ** (-k / 2.0) * math.sin(k * t * lp)
    return total / math.pi


@dataclass(frozen=True)
class PulseCoherenceCertificate:
    """Finite pulse comparison with a declared error tolerance and oracle status."""

    n_heights: int
    max_abs_s_error: float | None
    zero_count_matches: bool
    coherence_axis_is_minimal: bool
    s_tolerance: float = 0.05
    independent_oracle_available: bool = True

    @property
    def s_tolerance_satisfied(self) -> bool:
        """Whether a nonempty independent comparison meets the declared cut."""
        return (
            self.independent_oracle_available
            and self.n_heights > 0
            and self.max_abs_s_error is not None
            and math.isfinite(self.max_abs_s_error)
            and math.isfinite(self.s_tolerance)
            and self.s_tolerance >= 0.0
            and 0.0 <= self.max_abs_s_error <= self.s_tolerance
        )

    def summary(self) -> str:
        status = (
            "PASS"
            if (
                self.s_tolerance_satisfied
                and self.zero_count_matches
                and self.coherence_axis_is_minimal
            )
            else "PARTIAL"
        )
        error = (
            "unavailable"
            if self.max_abs_s_error is None
            else f"{self.max_abs_s_error:.3f}"
        )
        return (
            f"PulseCoherenceCertificate[{status}]: {self.n_heights} heights; "
            f"max|ΔS|={error} (off zeros), tolerance={self.s_tolerance:g}, "
            f"independent oracle={self.independent_oracle_available}; "
            f"N(T) counts zeros={self.zero_count_matches}; "
            f"coherence axis minimal at σ=1/2={self.coherence_axis_is_minimal}"
        )


def verify_pulse_coherence(
    heights: tuple[float, ...] = (17.0, 23.0, 28.0, 35.0, 47.0),
    *,
    s_tol: float = 0.05,
) -> PulseCoherenceCertificate:
    r"""Compare finite arithmetic diagnostics with a numerical zeta oracle.

    At heights chosen away from zeros, report (i) the maximum discrepancy of
    the pulse phase from ``(1/π) arg ζ``; (ii) whether ``round(N(T))`` matches
    the retained reference zero count; (iii) whether the coherence defect is
    minimal on ``σ = 1/2`` compared with ``0.6, 0.7``. ``s_tol`` must be finite,
    nonnegative and nonboolean. A PASS additionally requires a nonempty
    independent comparison with maximum error at most ``s_tol``. If mpmath is
    unavailable, the report is explicitly partial with no measured error;
    comparing the pulse with itself cannot supply oracle evidence.
    Arithmetic uses a temporary 25-digit context, restored even on failure.
    Both argument values use their principal branches; a finite PASS does not
    identify the classical continuous S(T), certify untested zero counts, or
    establish a nodal pressure law or a physical pulse.
    """
    if isinstance(s_tol, (bool, np.bool_)):
        raise ValueError("s_tol must be finite, nonnegative and nonboolean")
    try:
        tolerance = float(s_tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("s_tol must be finite, nonnegative and nonboolean") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("s_tol must be finite, nonnegative and nonboolean")

    try:
        import mpmath as mp
    except ImportError:
        return PulseCoherenceCertificate(
            n_heights=len(heights),
            max_abs_s_error=None,
            zero_count_matches=False,
            coherence_axis_is_minimal=False,
            s_tolerance=tolerance,
            independent_oracle_available=False,
        )

    with mp.workdps(25):
        max_s_err = 0.0
        counts_ok = True
        for t in heights:
            true_s = float(mp.arg(mp.zeta(mp.mpf("0.5") + 1j * t)) / mp.pi)
            error = abs(argument_fluctuation(t) - true_s)
            max_s_err = max(max_s_err, error) if math.isfinite(error) else math.inf
            n_est = round(zero_count(t))
            true_count = sum(1 for g in KNOWN_RIEMANN_ZEROS if g < t)
            counts_ok &= n_est == true_count

        axis_minimal = True
        for t in heights:
            d_half = coherence_defect(t, 0.5)
            axis_minimal &= d_half <= coherence_defect(
                t, 0.6
            ) and d_half <= coherence_defect(t, 0.7)

        return PulseCoherenceCertificate(
            n_heights=len(heights),
            max_abs_s_error=float(max_s_err),
            zero_count_matches=bool(counts_ok),
            coherence_axis_is_minimal=bool(axis_minimal),
            s_tolerance=tolerance,
            independent_oracle_available=True,
        )
