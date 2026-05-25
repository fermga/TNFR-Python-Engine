r"""P33: TNFR analytic continuation of the χ-twisted prime-ladder L-series.

The χ-twisted prime-ladder Dirichlet trace built in
:mod:`tnfr.riemann.dirichlet_l`,

.. math::

    Z_{TNFR}(s, \chi)
       \;=\; \sum_{p,\,k\ge 1} \chi(p)^k\, \log(p)\, e^{-s\,k\log p}
       \;=\; \sum_{n\ge 1} \chi(n)\, \Lambda(n)\, n^{-s}
       \;=\; -\frac{L'(s, \chi)}{L(s, \chi)},
       \qquad \mathrm{Re}(s) > 1,

converges only on the right half-plane :math:`\mathrm{Re}(s) > 1`.  Its
classical analytic continuation is the meromorphic function
:math:`-L'(s,\chi)/L(s,\chi)`.  For a non-principal primitive character
:math:`\chi` mod :math:`q`, the function :math:`L(s,\chi)` is **entire**
(no pole at :math:`s = 1`), so the poles of :math:`-L'/L` are exactly
the **zeros of** :math:`L(s,\chi)`:

* a discrete set of **non-trivial zeros** :math:`\rho_n^{(\chi)} =
  1/2 + i t_n^{(\chi)}` on the critical line
  :math:`\mathrm{Re}(s) = 1/2` (conjectured by the generalised Riemann
  hypothesis, GRH, and verified extensively for small :math:`q`),
* **trivial zeros** at :math:`s = -2k` (resp. :math:`s = -2k - 1`)
  according to the parity of :math:`\chi`.

For the principal character :math:`\chi_0` mod :math:`q`,
:math:`L(s,\chi_0) = \zeta(s) \prod_{p\mid q}(1 - p^{-s})`, so
:math:`-L'/L` has the same :math:`s = 1` pole as :math:`-\zeta'/\zeta`
plus additional poles at :math:`s = (2\pi i n)/\log p` for every
:math:`p \mid q`, none of which appear in this module's critical-line
scans.

TNFR interpretation
-------------------
In the χ-twisted prime-ladder reading of :mod:`tnfr.riemann.dirichlet_l`,
each prime :math:`p \nmid q` contributes a REMESH echo ladder
:math:`\mu_{p,k} = k\log p` with χ-twisted weight
:math:`w_{p,k}^{(\chi)} = \chi(p)^k \log p`.  Primes :math:`p \mid q`
satisfy :math:`\chi(p) = 0` and decouple from the spectrum entirely.

Continuing :math:`Z_{TNFR}(\cdot,\chi)` to :math:`\mathrm{Re}(s) \le 1`
exposes a discrete set of **resonance poles** carrying the entire
arithmetic content of the L-function:

* Each pole at :math:`\rho = 1/2 + i t_n^{(\chi)}` acts as a coherent
  **resonant frequency** of the χ-twisted prime-ladder REMESH spectrum.
* The explicit formula for the χ-twisted summatory function
  :math:`\psi(x, \chi) = \sum_{n \le x} \chi(n)\Lambda(n)` decomposes
  as the (non-principal) sum :math:`-\sum_{\rho} x^{\rho}/\rho` plus
  smaller polar contributions from the trivial zeros.

Honesty disclaimer
------------------
This module does **not** prove the generalised Riemann hypothesis.  It
does not construct a new continuation either: the continuation
:math:`-L'(s,\chi)/L(s,\chi)` is the unique meromorphic extension and
is implemented here via :mod:`mpmath.dirichlet` (which evaluates
:math:`L(s,\chi)` and its first derivative).  What is new is the
**operational TNFR reading**: every analytic feature of
:math:`-L'/L` is labelled by a structural mechanism of the χ-twisted
prime-ladder REMESH spectrum.

The module exposes four tools:

#. :func:`dirichlet_l_continued` — high-precision evaluation of
   :math:`L(s,\chi)` for arbitrary :math:`s \in \mathbb{C}`.
#. :func:`dirichlet_log_l_derivative_continued` — high-precision
   evaluation of :math:`-L'(s,\chi)/L(s,\chi)`.
#. :func:`verify_twisted_continuation_agreement` — numerical
   certificate that the χ-twisted prime-ladder series agrees with the
   continuation on a chosen subset of :math:`\mathrm{Re}(s) > 1`.
#. :func:`scan_critical_line_for_l_poles` — detects the resonance
   frequencies :math:`t_n^{(\chi)}` along :math:`\mathrm{Re}(s) = 1/2`.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P33
program (analytic continuation extension of P32).  Does NOT close
gap G4 nor the generalised Riemann hypothesis.

References
----------
- :mod:`tnfr.riemann.dirichlet_l` -- χ-twisted prime-ladder spectrum
  (P32).
- :mod:`tnfr.riemann.analytic_continuation` -- ζ analogue (P13).
- :mod:`tnfr.riemann.von_mangoldt` -- prime-ladder Dirichlet trace
  for ζ (P12).
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13duodecies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp

from ..mathematics.unified_numerical import np
from .dirichlet_l import DirichletCharacter, TwistedPrimeLadderSpectrum, tnfr_log_l_derivative


__all__ = [
    # Continuation evaluators
    "dirichlet_l_continued",
    "dirichlet_log_l_derivative_continued",
    # Agreement certificate
    "TwistedContinuationAgreement",
    "verify_twisted_continuation_agreement",
    # Pole detection on the critical line
    "DirichletCriticalLinePoleScan",
    "scan_critical_line_for_l_poles",
]


# ============================================================================
# Section 1 -- High-precision continuation evaluators
# ============================================================================


def _chi_to_mpmath_list(chi: DirichletCharacter) -> list:
    """Convert a DirichletCharacter to the list form expected by mp.dirichlet.

    mpmath.dirichlet expects ``[chi(0), chi(1), ..., chi(q-1)]``.
    Returns mpmath-compatible real or complex values.
    """
    out: list = []
    for v in chi.values:
        if abs(v.imag) < 1e-15:
            out.append(mp.mpf(v.real))
        else:
            out.append(mp.mpc(v.real, v.imag))
    return out


def dirichlet_l_continued(
    chi: DirichletCharacter,
    s: complex | float,
    *,
    dps: int = 30,
) -> complex:
    r"""Evaluate :math:`L(s, \chi)` for arbitrary :math:`s \in \mathbb{C}`.

    Implementation: :func:`mpmath.dirichlet` at ``dps`` decimal digits
    of precision.  For a non-principal primitive character, :math:`L(s,
    \chi)` is entire and this evaluator returns finite values
    everywhere.  For the principal character :math:`\chi_0` mod
    :math:`q`, the pole at :math:`s = 1` is reported as ``inf``.

    Parameters
    ----------
    chi : DirichletCharacter
        Character (any of the canonical constructors in
        :mod:`tnfr.riemann.dirichlet_l`, or a user-supplied
        ``DirichletCharacter``).
    s : complex or float
        Spectral parameter.
    dps : int, default 30
        Working decimal precision for the mpmath call.

    Returns
    -------
    complex
        :math:`L(s, \chi)` evaluated at :math:`s`.
    """
    chi_list = _chi_to_mpmath_list(chi)
    with mp.workdps(dps):
        s_mp = mp.mpc(s)
        result = mp.dirichlet(s_mp, chi_list)
    return complex(result)


def dirichlet_log_l_derivative_continued(
    chi: DirichletCharacter,
    s: complex | float,
    *,
    dps: int = 30,
) -> complex:
    r"""Evaluate :math:`-L'(s,\chi)/L(s,\chi)` for arbitrary :math:`s`.

    The classical analytic continuation of the χ-twisted prime-ladder
    Dirichlet trace.  Implemented via two :func:`mpmath.dirichlet`
    calls (one for :math:`L`, one for :math:`L'`) at ``dps`` decimal
    digits of precision.

    Parameters
    ----------
    chi : DirichletCharacter
        Character.
    s : complex or float
        Spectral parameter.  Must not coincide with a zero of
        :math:`L(s, \chi)` (those points are precisely the poles of
        the logarithmic derivative).
    dps : int, default 30
        Working decimal precision.

    Returns
    -------
    complex
        :math:`-L'(s,\chi)/L(s,\chi)` evaluated at :math:`s`.

    Raises
    ------
    ValueError
        If :math:`L(s,\chi) = 0` at the evaluation point (pole of
        :math:`-L'/L`).
    """
    chi_list = _chi_to_mpmath_list(chi)
    with mp.workdps(dps):
        s_mp = mp.mpc(s)
        l_val = mp.dirichlet(s_mp, chi_list, 0)
        if abs(l_val) < mp.mpf(10) ** (-dps + 5):
            raise ValueError(
                f"s={s} hits an L(s, chi) zero (pole of -L'/L)"
            )
        l_deriv = mp.dirichlet(s_mp, chi_list, 1)
        result = -l_deriv / l_val
    return complex(result)


# ============================================================================
# Section 2 -- Agreement certificate on the convergent half-plane
# ============================================================================


@dataclass(frozen=True)
class TwistedContinuationAgreement:
    r"""Numerical agreement between χ-twisted prime ladder and continuation.

    On :math:`\mathrm{Re}(s) > 1` the χ-twisted prime-ladder Dirichlet
    trace :math:`Z_{TNFR}(s, \chi)` converges (geometrically in
    :math:`\log p`) to :math:`-L'(s, \chi)/L(s, \chi)`.  This dataclass
    records the per-point discrepancy and a global quality tag.
    """

    chi_name: str
    """Identifier of the character used."""

    s_values: np.ndarray
    """Sampled spectral parameters (complex)."""

    z_prime_ladder: np.ndarray
    """χ-twisted prime-ladder Dirichlet trace values."""

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


def verify_twisted_continuation_agreement(
    spectrum: TwistedPrimeLadderSpectrum,
    chi: DirichletCharacter,
    s_values: Sequence[complex],
    *,
    dps: int = 30,
    excellent_threshold: float = 1e-3,
    good_threshold: float = 1e-1,
) -> TwistedContinuationAgreement:
    r"""Verify :math:`Z_{TNFR}(s, \chi) \approx -L'(s,\chi)/L(s,\chi)`.

    For every sampled :math:`s` with :math:`\mathrm{Re}(s) > 1` the
    χ-twisted prime-ladder partial sum must approach the
    mpmath-evaluated continuation as :math:`|\mathcal{P}|, K \to
    \infty`.  Convergence is geometric in :math:`p^{-(K+1)\mathrm{Re}(s)}`
    per prime, so values of :math:`\mathrm{Re}(s)` close to :math:`1`
    require larger ``n_primes`` and ``max_power`` to reach a given
    tolerance.

    Parameters
    ----------
    spectrum : TwistedPrimeLadderSpectrum
        Spectrum built by
        :func:`tnfr.riemann.dirichlet_l.build_twisted_prime_ladder_spectrum`
        using the same character ``chi`` supplied below.  The function
        does not re-check character agreement; the caller is
        responsible for consistency.
    chi : DirichletCharacter
        Character used for the continuation reference.
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
                "verify_twisted_continuation_agreement requires Re(s) > 1; "
                f"got s={s} with Re(s)={s.real}"
            )
        z_pl[idx] = tnfr_log_l_derivative(spectrum, complex(s))
        z_co[idx] = dirichlet_log_l_derivative_continued(
            chi, complex(s), dps=dps
        )

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

    return TwistedContinuationAgreement(
        chi_name=chi.name or f"chi_mod_{chi.modulus}",
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
class DirichletCriticalLinePoleScan:
    r"""Scan of :math:`|{-L'(s,\chi)/L(s,\chi)}|` along :math:`s = 1/2 + it`.

    The continuation has a simple pole at every non-trivial zero
    :math:`\rho_n^{(\chi)} = 1/2 + i t_n^{(\chi)}` of :math:`L(s,
    \chi)`.  Sampling the magnitude :math:`|-L'/L|` along the critical
    line therefore produces sharp peaks at :math:`t = t_n^{(\chi)}`;
    the locations of those peaks are the TNFR-detected resonance
    frequencies of the χ-twisted prime-ladder spectrum.
    """

    chi_name: str
    """Identifier of the character used."""

    t_values: np.ndarray
    """Imaginary parts at which we sampled :math:`s = 1/2 + i t`."""

    magnitudes: np.ndarray
    r"""Sampled :math:`|-L'(s,\chi)/L(s,\chi)|`."""

    detected_peaks: np.ndarray
    """Detected peak locations (``t`` values) in ascending order."""

    n_peaks: int
    """Number of peaks detected."""

    median_spacing: float
    """Median spacing between consecutive detected peaks (or ``nan`` if
    fewer than two peaks were found)."""


def scan_critical_line_for_l_poles(
    chi: DirichletCharacter,
    t_min: float = 5.0,
    t_max: float = 60.0,
    n_samples: int = 4001,
    *,
    dps: int = 25,
    peak_prominence: float = 5.0,
) -> DirichletCriticalLinePoleScan:
    r"""Detect resonance poles of :math:`-L'/L` on :math:`\mathrm{Re}(s) = 1/2`.

    The implementation samples ``n_samples`` evenly spaced
    :math:`t \in [t_{\min}, t_{\max}]`, evaluates

    .. math::

        m(t) \;=\; \bigl| -L'(1/2 + it, \chi) /
                          L(1/2 + it, \chi) \bigr|

    via :func:`dirichlet_log_l_derivative_continued`, and selects
    local maxima above ``peak_prominence`` over the surrounding
    floor.  Each detected peak corresponds to a non-trivial zero of
    :math:`L(s, \chi)` on the critical line (a TNFR resonance pole of
    the χ-twisted prime-ladder REMESH spectrum).

    Unlike the ζ analogue in :func:`tnfr.riemann.analytic_continuation
    .scan_critical_line_for_poles`, this function does **not** match
    against a pre-tabulated reference list.  For small modulus
    characters, the first few zeros of :math:`L(s, \chi)` are tabulated
    in the LMFDB; comparing the detected peaks against those tables is
    a manual verification step left to the caller.

    Parameters
    ----------
    chi : DirichletCharacter
        Character.  For the principal character, the continuation has
        additional poles at :math:`s = 1` and at
        :math:`s = (2\pi i n)/\log p` for :math:`p \mid q`; users should
        restrict the scan range or interpret extra peaks accordingly.
    t_min, t_max : float
        Inclusive range of imaginary parts to sample.  Must satisfy
        ``t_min < t_max``.
    n_samples : int
        Number of evenly spaced samples.  Resolution
        ``(t_max - t_min)/(n_samples - 1)`` must be much smaller than
        the typical spacing between zeros in the range.
    dps : int, default 25
        Mpmath working precision.
    peak_prominence : float, default 5.0
        Minimum height above the local floor to qualify as a peak.

    Returns
    -------
    DirichletCriticalLinePoleScan
    """
    if t_min >= t_max:
        raise ValueError("t_min must be < t_max")
    if n_samples < 11:
        raise ValueError("n_samples must be >= 11")

    t_grid = np.linspace(t_min, t_max, n_samples)
    magnitudes = np.empty(n_samples)
    for i, t in enumerate(t_grid):
        try:
            value = dirichlet_log_l_derivative_continued(
                chi, complex(0.5, float(t)), dps=dps
            )
            magnitudes[i] = abs(value)
        except ValueError:
            # We landed exactly on an L zero -- treat as infinite peak.
            magnitudes[i] = np.inf

    # Local-maximum detection: a point i is a peak if it strictly
    # exceeds both neighbours and the prominence above the floor of
    # its surrounding window exceeds peak_prominence.
    peaks: list[float] = []
    window = max(3, n_samples // 200)
    for i in range(1, n_samples - 1):
        m_i = magnitudes[i]
        if not np.isfinite(m_i) and not np.isnan(m_i):
            peaks.append(float(t_grid[i]))
            continue
        if not (m_i > magnitudes[i - 1] and m_i > magnitudes[i + 1]):
            continue
        lo = max(0, i - window)
        hi = min(n_samples, i + window + 1)
        local_floor = float(np.median(magnitudes[lo:hi]))
        if m_i - local_floor >= peak_prominence:
            peaks.append(float(t_grid[i]))

    peaks_arr = np.asarray(peaks, dtype=float)
    if peaks_arr.size >= 2:
        spacings = np.diff(peaks_arr)
        median_spacing = float(np.median(spacings))
    else:
        median_spacing = float("nan")

    return DirichletCriticalLinePoleScan(
        chi_name=chi.name or f"chi_mod_{chi.modulus}",
        t_values=t_grid,
        magnitudes=magnitudes,
        detected_peaks=peaks_arr,
        n_peaks=int(peaks_arr.size),
        median_spacing=median_spacing,
    )
