r"""P13: TNFR analytic continuation of the prime-ladder von Mangoldt zeta.

The prime-ladder Dirichlet trace built in :mod:`tnfr.riemann.von_mangoldt`,

.. math::

    Z_{\mathrm{vM}}(s) \;=\; \sum_{p,k} \log(p)\, e^{-s\,k\log p}
                       \;=\; \sum_{n\ge 1} \Lambda(n)\, n^{-s}
                       \;=\; -\frac{\zeta'(s)}{\zeta(s)},
                       \qquad \mathrm{Re}(s) > 1,

converges only on the right half-plane :math:`\mathrm{Re}(s) > 1`.  Its
classical analytic continuation is the meromorphic function
:math:`-\zeta'(s)/\zeta(s)` on :math:`\mathbb{C}\setminus\{1\}\cup
\{\rho\}\cup\{-2k\}`, whose pole structure is, via the Hadamard product:

* a simple pole at :math:`s=1` with residue :math:`+1`
  (Chebyshev / Mertens dominant term),
* a simple pole at each non-trivial zero
  :math:`\rho = 1/2 + i\,t_n` of :math:`\zeta` with residue equal to
  :math:`-m_\rho` (multiplicity of :math:`\rho`),
* a simple pole at each trivial zero :math:`s=-2k`, :math:`k\ge 1`,
  with residue :math:`-1`.

TNFR interpretation
-------------------
In the TNFR prime-ladder reading of :mod:`tnfr.riemann.von_mangoldt`,
each prime :math:`p` contributes a REMESH echo ladder
:math:`\mu_{p,k} = k\log p`.  Continuing :math:`Z_{\mathrm{vM}}` to
:math:`\mathrm{Re}(s) \le 1` exposes a discrete set of
**resonance poles** which carry the entire arithmetic content:

* The :math:`s=1` pole encodes the linear envelope
  :math:`\psi(x) \sim x` (prime number theorem leading term).
* Each pole at :math:`\rho = 1/2 + it_n` acts as a coherent
  **resonant frequency** of the prime-ladder REMESH spectrum:
  the explicit formula

  .. math::

      \psi_0(x) \;=\; x \;-\; \sum_\rho \frac{x^\rho}{\rho}
                          \;-\; \log(2\pi)
                          \;-\; \tfrac{1}{2}\log\!\bigl(1 - x^{-2}\bigr)

  decomposes :math:`\psi(x) = \sum_{n\le x}\Lambda(n)` into a smooth
  Chebyshev background :math:`x` plus oscillatory contributions
  :math:`x^\rho/\rho` whose frequencies :math:`t_n` and amplitudes
  :math:`|\rho|^{-1}` are entirely determined by the resonance poles.

Honesty disclaimer
------------------
This module does **not** prove the Riemann Hypothesis.  It does not
construct a new continuation either: the continuation
:math:`-\zeta'(s)/\zeta(s)` is the unique meromorphic extension and is
implemented here via :mod:`mpmath`.  What is new is the **operational
TNFR reading**: every analytic feature of :math:`-\zeta'/\zeta` is
labelled by a structural mechanism of the prime-ladder REMESH spectrum
(Chebyshev envelope, resonance frequencies, trivial-zero curvature).

The module exposes four tools:

#. :func:`von_mangoldt_zeta_continued` — high-precision evaluation of
   the continuation for arbitrary :math:`s\in\mathbb{C}`.
#. :func:`verify_continuation_agreement` — numerical certificate that
   the prime-ladder series agrees with the continuation on a chosen
   subset of :math:`\mathrm{Re}(s) > 1`.
#. :func:`scan_critical_line_for_poles` — detects the resonance
   frequencies :math:`t_n` along :math:`\mathrm{Re}(s) = 1/2` and
   matches them against the known Riemann zeros.
#. :func:`reconstruct_psi_via_explicit_formula` — rebuilds
   :math:`\psi(x)` from a truncated sum over resonance poles to
   quantify how each new zero refines the prime-ladder envelope.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P13 program.

References
----------
- :mod:`tnfr.riemann.von_mangoldt` -- prime-ladder Dirichlet trace (P12).
- :mod:`tnfr.riemann.spectral_zeta` -- Mellin bridge for the graph
  spectral zeta :math:`\zeta_H` (P5).
- :mod:`tnfr.riemann.complex_extension` -- non-Hermitian
  :math:`H(s)` and ``KNOWN_RIEMANN_ZEROS`` (P4).
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp

from ..mathematics.unified_numerical import np
from .complex_extension import KNOWN_RIEMANN_ZEROS
from .von_mangoldt import (
    PrimeLadderSpectrum,
    mangoldt_lambda,
    tnfr_log_zeta_derivative,
)

__all__ = [
    # Continuation evaluator
    "von_mangoldt_zeta_continued",
    # Agreement certificate
    "ContinuationAgreement",
    "verify_continuation_agreement",
    # Pole detection on the critical line
    "CriticalLinePoleScan",
    "scan_critical_line_for_poles",
    # Explicit formula reconstruction
    "ExplicitFormulaResult",
    "reconstruct_psi_via_explicit_formula",
    # Convenience: pre-tabulated nontrivial zeros from mpmath
    "fetch_riemann_zeros",
]


# ============================================================================
# Section 1 -- High-precision continuation evaluator
# ============================================================================

def von_mangoldt_zeta_continued(
    s: complex | float,
    *,
    dps: int = 30,
) -> complex:
    r"""Evaluate :math:`-\zeta'(s)/\zeta(s)` for arbitrary :math:`s\in\mathbb{C}`.

    The classical analytic continuation of the prime-ladder Dirichlet
    trace.  Implemented via :mod:`mpmath` at ``dps`` decimal digits of
    precision and converted back to a Python ``complex``.

    Parameters
    ----------
    s : complex or float
        Spectral parameter.  May be anywhere in :math:`\mathbb{C}`
        except at the poles :math:`s=1`, :math:`s=\rho`, :math:`s=-2k`.
    dps : int, default 30
        Working decimal precision for the internal mpmath call.

    Returns
    -------
    complex
        :math:`-\zeta'(s)/\zeta(s)` evaluated at :math:`s`.

    Raises
    ------
    ValueError
        If the evaluation point is exactly at a pole detected by mpmath
        (returns ``inf`` and we surface that as an exception).
    """
    with mp.workdps(dps):
        s_mp = mp.mpc(s)
        zeta_val = mp.zeta(s_mp)
        if zeta_val == 0:
            raise ValueError(
                f"s={s} hits a zeta zero (pole of -zeta'/zeta)"
            )
        zeta_deriv = mp.zeta(s_mp, derivative=1)
        result = -zeta_deriv / zeta_val
    return complex(result)


# ============================================================================
# Section 2 -- Agreement certificate on the convergent half-plane
# ============================================================================

@dataclass(frozen=True)
class ContinuationAgreement:
    r"""Numerical agreement between prime-ladder sum and continuation.

    On :math:`\mathrm{Re}(s) > 1` the prime-ladder Dirichlet trace
    :math:`Z_{TNFR}(s)` converges (geometrically in
    :math:`\log p`) to :math:`-\zeta'(s)/\zeta(s)`.  This dataclass
    records the per-point discrepancy and a global quality tag.
    """

    s_values: np.ndarray
    """Sampled spectral parameters (complex)."""

    z_prime_ladder: np.ndarray
    """Prime-ladder Dirichlet trace values."""

    z_continued: np.ndarray
    """Analytic continuation values via mpmath."""

    abs_diff: np.ndarray
    """Pointwise absolute difference."""

    rel_diff: np.ndarray
    """Pointwise relative difference (|continued|>0 assumed)."""

    max_abs_diff: float
    max_rel_diff: float

    agreement_quality: str
    """One of {'excellent', 'good', 'poor'}."""


def verify_continuation_agreement(
    spectrum: PrimeLadderSpectrum,
    s_values: Sequence[complex],
    *,
    dps: int = 30,
    excellent_threshold: float = 1e-3,
    good_threshold: float = 1e-1,
) -> ContinuationAgreement:
    r"""Verify :math:`Z_{TNFR}(s) \approx -\zeta'(s)/\zeta(s)` on :math:`\mathrm{Re}(s)>1`.

    For every sampled :math:`s` with :math:`\mathrm{Re}(s) > 1` the
    prime-ladder partial sum must approach the mpmath-evaluated
    continuation as :math:`|\mathcal{P}|, K \to \infty`.  Convergence is
    geometric in :math:`p^{-(K+1)\mathrm{Re}(s)}` per prime, so values
    of :math:`\mathrm{Re}(s)` close to :math:`1` require larger
    ``n_primes`` and ``max_power`` to reach a given tolerance.

    Parameters
    ----------
    spectrum : PrimeLadderSpectrum
        Prime-ladder built by
        :func:`tnfr.riemann.von_mangoldt.build_prime_ladder_spectrum`.
    s_values : sequence of complex
        Points to sample.  Real points with :math:`s > 1` are perfectly
        valid (passed through ``complex(...)`` internally).
    dps : int, default 30
        Mpmath working precision for the reference values.
    excellent_threshold, good_threshold : float
        Maximum allowed relative discrepancy for the ``'excellent'``
        and ``'good'`` quality tags, respectively.
    """
    s_arr = np.array([complex(s) for s in s_values])
    z_pl = np.empty(s_arr.shape, dtype=complex)
    z_co = np.empty(s_arr.shape, dtype=complex)

    for idx, s in enumerate(s_arr):
        if s.real <= 1.0:
            raise ValueError(
                "verify_continuation_agreement requires Re(s) > 1; "
                f"got s={s} with Re(s)={s.real}"
            )
        z_pl[idx] = tnfr_log_zeta_derivative(spectrum, complex(s))
        z_co[idx] = von_mangoldt_zeta_continued(complex(s), dps=dps)

    abs_diff = np.abs(z_pl - z_co)
    denom = np.maximum(np.abs(z_co), 1e-300)
    rel_diff = abs_diff / denom
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())

    if max_rel <= excellent_threshold:
        quality = "excellent"
    elif max_rel <= good_threshold:
        quality = "good"
    else:
        quality = "poor"

    return ContinuationAgreement(
        s_values=s_arr,
        z_prime_ladder=z_pl,
        z_continued=z_co,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        agreement_quality=quality,
    )


# ============================================================================
# Section 3 -- Resonance poles along the critical line
# ============================================================================

@dataclass(frozen=True)
class CriticalLinePoleScan:
    r"""Scan of :math:`|{-\zeta'(s)/\zeta(s)}|` along :math:`s=1/2+it`.

    The continuation has a simple pole at every non-trivial
    zero :math:`\rho_n = 1/2 + i t_n` of :math:`\zeta`.  Sampling the
    magnitude :math:`|-\zeta'/\zeta|` along the critical line therefore
    produces sharp peaks at :math:`t = t_n`; the locations of those
    peaks are the TNFR-detected resonance frequencies of the
    prime-ladder spectrum.
    """

    t_values: np.ndarray
    """Imaginary parts at which we sampled :math:`s = 1/2 + i t`."""

    magnitudes: np.ndarray
    r"""Sampled :math:`|-\zeta'(s)/\zeta(s)|`."""

    detected_peaks: np.ndarray
    """Detected peak locations (``t`` values) in ascending order."""

    matched_zeros: tuple[tuple[float, float, float], ...]
    """``(t_detected, t_known, |t_detected-t_known|)`` for every
    detected peak that lies within ``match_tolerance`` of a known
    Riemann zero (defaults to the first 20 zeros from
    :data:`KNOWN_RIEMANN_ZEROS`)."""

    detection_quality: str
    """Summary tag: ``'all_matched'``, ``'partial_match'`` or
    ``'no_match'`` against the supplied reference list."""


def scan_critical_line_for_poles(
    t_min: float = 10.0,
    t_max: float = 80.0,
    n_samples: int = 4001,
    *,
    dps: int = 25,
    peak_prominence: float = 5.0,
    match_tolerance: float = 0.05,
    reference_zeros: Sequence[float] | None = None,
) -> CriticalLinePoleScan:
    r"""Detect resonance poles of :math:`-\zeta'/\zeta` on :math:`\mathrm{Re}(s)=1/2`.

    The implementation samples ``n_samples`` evenly spaced
    :math:`t \in [t_{\min}, t_{\max}]`, evaluates
    :math:`m(t) = |{-\zeta'(1/2+it)/\zeta(1/2+it)}|`
    via :func:`von_mangoldt_zeta_continued`, and selects local maxima
    above ``peak_prominence``.  Each detected peak is matched against
    ``reference_zeros`` (defaults to :data:`KNOWN_RIEMANN_ZEROS`).

    Parameters
    ----------
    t_min, t_max : float
        Inclusive range of imaginary parts to sample.  Must satisfy
        ``t_min < t_max``.
    n_samples : int
        Number of evenly spaced samples.  Resolution
        ``(t_max - t_min)/(n_samples - 1)`` must be much smaller than
        the typical spacing between Riemann zeros in the range
        (~ :math:`2\pi/\log(t/2\pi)`).
    dps : int, default 25
        Mpmath working precision.  20-30 is sufficient for the first
        :math:`\sim 100` zeros.
    peak_prominence : float, default 5.0
        Minimum height above the local floor to qualify as a peak.
        Magnitudes near zeros routinely exceed :math:`10^3`, so the
        default is comfortably above background.
    match_tolerance : float, default 0.05
        Maximum :math:`|t_{\mathrm{detected}} - t_{\mathrm{known}}|`
        to declare a match.
    reference_zeros : sequence of float, optional
        Known zero imaginary parts.  Defaults to the first 20 zeros
        from :data:`KNOWN_RIEMANN_ZEROS`.
    """
    if t_min >= t_max:
        raise ValueError("t_min must be < t_max")
    if n_samples < 11:
        raise ValueError("n_samples must be >= 11")

    if reference_zeros is None:
        reference_zeros = KNOWN_RIEMANN_ZEROS

    t_grid = np.linspace(t_min, t_max, n_samples)
    magnitudes = np.empty(n_samples)
    for i, t in enumerate(t_grid):
        try:
            value = von_mangoldt_zeta_continued(complex(0.5, float(t)), dps=dps)
            magnitudes[i] = abs(value)
        except ValueError:
            # We landed exactly on a zero -- treat as infinite peak.
            magnitudes[i] = np.inf

    # Local-maximum detection: a point i is a peak if it strictly
    # exceeds both neighbours and the prominence above the floor of
    # its surrounding window exceeds peak_prominence.
    peaks: list[float] = []
    window = max(3, n_samples // 200)
    for i in range(1, n_samples - 1):
        m_i = magnitudes[i]
        if not np.isfinite(m_i) and not np.isnan(m_i):
            # Infinite peak (exact zero hit) is unambiguously a peak.
            peaks.append(float(t_grid[i]))
            continue
        if not (m_i > magnitudes[i - 1] and m_i > magnitudes[i + 1]):
            continue
        lo = max(0, i - window)
        hi = min(n_samples, i + window + 1)
        local_floor = float(np.median(magnitudes[lo:hi]))
        if m_i - local_floor >= peak_prominence:
            peaks.append(float(t_grid[i]))

    matches: list[tuple[float, float, float]] = []
    for t_det in peaks:
        nearest_known = min(reference_zeros, key=lambda z: abs(z - t_det))
        delta = abs(t_det - nearest_known)
        if delta <= match_tolerance:
            matches.append((t_det, float(nearest_known), float(delta)))

    eligible_known = [z for z in reference_zeros if t_min <= z <= t_max]
    if matches and len(matches) >= len(eligible_known):
        quality = "all_matched"
    elif matches:
        quality = "partial_match"
    else:
        quality = "no_match"

    return CriticalLinePoleScan(
        t_values=t_grid,
        magnitudes=magnitudes,
        detected_peaks=np.asarray(peaks, dtype=float),
        matched_zeros=tuple(matches),
        detection_quality=quality,
    )


# ============================================================================
# Section 4 -- Explicit formula reconstruction of psi(x)
# ============================================================================

@dataclass(frozen=True)
class ExplicitFormulaResult:
    r"""Reconstruction of :math:`\psi(x)` via the truncated explicit formula.

    The von Mangoldt explicit formula reads (Riemann / von Mangoldt 1859):

    .. math::

        \psi_0(x) \;=\; x \;-\; \sum_{\rho} \frac{x^\rho}{\rho}
                            \;-\; \log(2\pi)
                            \;-\; \tfrac{1}{2}
                                  \log\!\bigl(1 - x^{-2}\bigr),

    where :math:`\psi_0(x) = \psi(x) - \tfrac{1}{2}\Lambda(x)` at prime
    powers and :math:`\psi(x)` elsewhere, and the sum ranges over
    every non-trivial zero (counted with multiplicity, with the
    :math:`\rho`/:math:`\bar\rho` pairing implicit).  Truncating the
    sum to ``n_zeros`` pairs gives an approximation that quantifies
    how much arithmetic information each resonance pole carries.
    """

    x_values: np.ndarray
    r"""Real arguments at which :math:`\psi(x)` was reconstructed."""

    psi_classical: np.ndarray
    r"""Direct evaluation :math:`\psi(x) = \sum_{n\le x}\Lambda(n)`."""

    psi_explicit: np.ndarray
    """Truncated explicit formula estimate."""

    abs_error: np.ndarray
    r"""Pointwise :math:`|\psi_{\mathrm{classical}} - \psi_{\mathrm{explicit}}|`."""

    rel_error: np.ndarray
    r"""Relative error (normalised by :math:`\psi_{\mathrm{classical}}`)."""

    n_zeros_used: int


def _psi_classical(x: float) -> float:
    r"""Direct evaluation of :math:`\psi(x) = \sum_{n\le x}\Lambda(n)`."""
    floor_x = int(np.floor(x))
    if floor_x < 2:
        return 0.0
    total = 0.0
    for n in range(2, floor_x + 1):
        total += mangoldt_lambda(n)
    return float(total)


def fetch_riemann_zeros(n_zeros: int, *, dps: int = 30) -> np.ndarray:
    r"""Return the first ``n_zeros`` non-trivial zeros via :func:`mpmath.zetazero`.

    Returns a complex array of shape ``(n_zeros,)`` with
    :math:`\rho_n = 1/2 + i t_n`.  Only the upper half-plane zeros are
    returned; the explicit formula sums each :math:`\rho` together with
    its conjugate :math:`\bar\rho` automatically.

    Parameters
    ----------
    n_zeros : int
        Number of zeros to fetch (in ascending :math:`t_n`).  Values of
        :math:`n_{\mathrm{zeros}}` up to a few hundred are very fast.
    dps : int, default 30
        Mpmath working precision when computing each zero.
    """
    if n_zeros < 1:
        raise ValueError("n_zeros must be >= 1")
    rho = np.empty(n_zeros, dtype=complex)
    with mp.workdps(dps):
        for n in range(1, n_zeros + 1):
            rho[n - 1] = complex(mp.zetazero(n))
    return rho


def reconstruct_psi_via_explicit_formula(
    x_values: Sequence[float],
    *,
    n_zeros: int = 50,
    include_trivial: bool = True,
    zeros: Sequence[complex] | None = None,
    dps: int = 30,
) -> ExplicitFormulaResult:
    r"""Reconstruct :math:`\psi(x)` from a truncated sum over resonance poles.

    Implementation evaluates

    .. math::

        \widetilde\psi(x; N) \;=\; x \;-\;
            \sum_{n=1}^{N}\!\bigl(\tfrac{x^{\rho_n}}{\rho_n}
                                  + \tfrac{x^{\bar\rho_n}}{\bar\rho_n}\bigr)
                       \;-\; \log(2\pi)
                       \;-\; \tfrac{1}{2}\,\log\!\bigl(1 - x^{-2}\bigr)

    and compares against the direct sum
    :math:`\psi(x) = \sum_{n\le x}\Lambda(n)`.  The error
    :math:`|\psi - \widetilde\psi|` decays as :math:`N^{-1/2}` (slowly!)
    because the explicit formula converges only conditionally, but
    every added zero visibly damps a specific oscillatory mode at
    angular frequency :math:`t_n / \log x`.

    Parameters
    ----------
    x_values : sequence of float
        Points :math:`x > 1` at which to reconstruct :math:`\psi(x)`.
    n_zeros : int, default 50
        Number of conjugate pairs of non-trivial zeros to include.
        Ignored if ``zeros`` is supplied.
    include_trivial : bool, default True
        Include the correction
        :math:`-\tfrac{1}{2}\log(1 - x^{-2})` from the trivial zeros
        at :math:`s = -2k`.
    zeros : sequence of complex, optional
        Pre-fetched non-trivial zeros (upper half-plane only).  If
        omitted, the function calls :func:`fetch_riemann_zeros`
        internally.
    dps : int, default 30
        Mpmath precision when fetching zeros internally.
    """
    x_arr = np.asarray(x_values, dtype=float)
    if (x_arr <= 1.0).any():
        raise ValueError("explicit formula requires x > 1")

    if zeros is None:
        rho = fetch_riemann_zeros(n_zeros, dps=dps)
    else:
        rho = np.asarray(zeros, dtype=complex)
        n_zeros = int(rho.size)

    psi_classical = np.array([_psi_classical(x) for x in x_arr])

    log_2pi = float(np.log(2 * np.pi))
    psi_explicit = np.empty_like(x_arr)
    for i, x in enumerate(x_arr):
        log_x = float(np.log(x))
        # Oscillatory sum: pair each zero with its conjugate.  Using
        # 2 * Re(x^rho / rho) avoids carrying complex conjugates and
        # keeps the running sum real.
        osc = 0.0
        for r in rho:
            term = (x ** r) / r  # complex
            osc += 2.0 * float(term.real)
        smooth = x - log_2pi
        if include_trivial:
            smooth -= 0.5 * float(np.log(1.0 - x ** (-2.0)))
        psi_explicit[i] = smooth - osc

    abs_err = np.abs(psi_classical - psi_explicit)
    denom = np.where(psi_classical > 0, psi_classical, 1.0)
    rel_err = abs_err / denom

    return ExplicitFormulaResult(
        x_values=x_arr,
        psi_classical=psi_classical,
        psi_explicit=psi_explicit,
        abs_error=abs_err,
        rel_error=rel_err,
        n_zeros_used=n_zeros,
    )
