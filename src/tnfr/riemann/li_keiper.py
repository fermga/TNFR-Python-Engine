r"""P16: finite Li-Keiper zero-sum comparisons with explicit truncation.

Li's criterion (Xian-Jin Li, 1997)
----------------------------------
Define, for every integer :math:`n \ge 1`,

.. math::

    \lambda_n \;=\; \sum_{\rho} \Bigl[ 1 - \bigl(1 - \tfrac{1}{\rho}\bigr)^n
                             \Bigr]
            \;=\; \frac{1}{(n-1)!}\,
                  \frac{d^{\,n}}{ds^{\,n}}
                  \Bigl[\, s^{\,n-1}\,\log \xi(s)\,\Bigr]_{s=1},

where the full sum ranges over all non-trivial zeros :math:`\rho` of
:math:`\zeta(s)` with multiplicity and the prescribed symmetric limiting
convention. The classical criterion gives the
equivalence

.. math::

    \text{RH}\;\Longleftrightarrow\;\lambda_n > 0
    \quad\text{for every } n \ge 1.

Li's criterion is therefore strictly equivalent to the Riemann
Hypothesis, restated as a positivity condition on a real sequence.

Implemented comparison and its boundary
--------------------------------------
This module evaluates only a finite conjugate-paired sum. It supplies no
certified omitted-zero tail or numerical error enclosure, so its sign flags
are not certified signs of the complete coefficients, even at one index.
The classical zero list comes from :func:`mpmath.zetazero`; the optional
P13 branch evaluates a classical meromorphic function along the critical
line and places every detected ordinate at real part 1/2.

For any supplied :math:`\rho=1/2+it`, the factor
:math:`b=1-1/\rho` has modulus one. Its exact paired contribution is
:math:`2[1-\cos(n\arg b)]\geq0`, whether or not :math:`t` is a zero.
Nonnegative sums of these terms therefore do not independently validate
zero location, pole detection or a nodal resonance mechanism. A negative
numerical sum requires examination of inputs and arithmetic; it is not a
refutation of RH. The public certificate/field names are retained for
compatibility and describe finite observations, not analytic certificates.

Current scope is centralized in ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``.

Public API
----------
``li_coefficients_from_zeros``      Compute finite conjugate-paired sums
                                     from supplied upper-half-plane data.
``LiKeiperCertificate``             Frozen result with positivity flags,
                                     classical/TNFR comparison and summary.
``verify_li_keiper_criterion``      Finite comparison: fetch
                                     classical zeros, optionally compare
                                     against P13 detected resonance peaks,
                                     return certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mpmath

from ..mathematics.unified_numerical import np
from .analytic_continuation import fetch_riemann_zeros, scan_critical_line_for_poles

# ---------------------------------------------------------------------------
# Core: Li coefficients from a list of upper-half-plane zeros
# ---------------------------------------------------------------------------


def li_coefficients_from_zeros(
    zeros_upper: Sequence[complex],
    n_max: int,
    *,
    dps: int = 50,
) -> np.ndarray:
    r"""Compute truncated Li-type sums from a finite supplied zero list.

    Implementation uses the explicit form

    .. math::

        \lambda_n^{(K)} \;=\; \sum_{k=1}^{K} 2\,\Re\!\Bigl[
                       1 - \bigl(1 - \tfrac{1}{\rho_k}\bigr)^n \Bigr],
        \qquad \rho_k = \tfrac{1}{2} + i\, t_k,

    paired with :math:`\bar\rho_k`.  Computation is performed at
    arbitrary precision via :mod:`mpmath` to absorb cancellation
    between :math:`1` and :math:`(1-1/\rho)^n` as :math:`n` grows.
    Inputs first pass through binary64 real and imaginary parts; higher
    working precision does not restore digits discarded by that conversion.
    No omitted-zero or rounding bound is returned. The function does not
    establish that supplied coordinates are zeros.

    Parameters
    ----------
    zeros_upper : sequence of complex
        Upper half-plane non-trivial zeros, e.g.
        :math:`\rho_k = 1/2 + i\, t_k` with :math:`t_k > 0`.  Order
        does not matter (sum is symmetric).
    n_max : int
        Highest Li-Keiper index to compute (1-indexed).
    dps : int, default 50
        :mod:`mpmath` working precision (decimal places).

    Returns
    -------
    np.ndarray
        Shape ``(n_max,)`` real array with
        ``arr[n-1] = float(lambda_n_truncated)``.
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    if len(zeros_upper) == 0:
        raise ValueError("zeros_upper must contain at least one zero")

    out = np.zeros(n_max, dtype=float)
    with mpmath.workdps(dps):
        # Convert each zero to mpmath complex once and reuse.
        mp_zeros = [mpmath.mpc(float(z.real), float(z.imag)) for z in zeros_upper]
        # Pre-compute base = 1 - 1/rho for each zero.
        bases = [mpmath.mpc(1) - mpmath.mpc(1) / r for r in mp_zeros]

        for n_idx in range(1, n_max + 1):
            total = mpmath.mpf(0)
            for b in bases:
                # 2 * Re[1 - b^n] handles the rho/conj(rho) pairing
                total += 2 * (mpmath.mpf(1) - (b**n_idx).real)
            out[n_idx - 1] = float(total)
    return out


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiKeiperCertificate:
    r"""Compatibility-named record of finite zero-sum sign comparisons.

    Attributes
    ----------
    n_max
        Highest Li index computed.
    n_zeros_classical
        Number of zeros from :func:`mpmath.zetazero` used in the
        classical evaluation.
    lambda_classical
        Array of shape ``(n_max,)`` with truncated known-zero sums.
    lambda_tnfr
        Array of shape ``(n_max,)`` with truncated sums from
        line-restricted P13 peak coordinates (``None`` if not
        requested).
    positivity_classical
        ``True`` iff every materialized known-zero sum is positive.
    positivity_tnfr
        Same for the peak-coordinate sums (``None`` if not requested).
    max_abs_difference
        :math:`\max_n |\lambda_n^{\mathrm{classical}} -
        \lambda_n^{\mathrm{TNFR}}|` (``None`` if not requested).
    notes
        Extra contextual information (peak detection quality, etc.).
    """

    n_max: int
    n_zeros_classical: int
    lambda_classical: np.ndarray
    lambda_tnfr: np.ndarray | None
    positivity_classical: bool
    positivity_tnfr: bool | None
    max_abs_difference: float | None
    notes: dict[str, Any]

    def summary(self) -> str:
        r"""Return a multi-line human-readable summary."""
        lam = self.lambda_classical
        lines = [
            "Li-Keiper criterion certificate (TNFR-Riemann P16)",
            "-" * 60,
            f"  n_max               = {self.n_max}",
            f"  n_zeros (classical) = {self.n_zeros_classical}",
            f"  lambda_1            = {lam[0]:+.6e}",
            f"  lambda_{self.n_max}".ljust(22) + f"= {lam[-1]:+.6e}",
            f"  min_n lambda_n      = {float(lam.min()):+.6e}",
            f"  positivity (cls.)   = {self.positivity_classical}",
        ]
        if self.lambda_tnfr is not None:
            lines.extend(
                [
                    f"  positivity (TNFR)   = {self.positivity_tnfr}",
                    f"  max |Δλ|            = " f"{self.max_abs_difference:.3e}",
                ]
            )
        if self.notes:
            lines.append("  notes:")
            for k, v in self.notes.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end verification
# ---------------------------------------------------------------------------


def verify_li_keiper_criterion(
    *,
    n_max: int = 50,
    n_zeros: int = 200,
    dps: int = 50,
    compare_tnfr: bool = False,
    tnfr_t_min: float = 10.0,
    tnfr_t_max: float = 80.0,
    tnfr_n_samples: int = 4001,
) -> LiKeiperCertificate:
    r"""Compare signs of truncated zero sums up to index ``n_max``.

    Steps
    -----
    1. Fetch ``n_zeros`` non-trivial zeros via :func:`mpmath.zetazero`
       (classical reference).
    2. Compute finite sums for :math:`n = 1, \ldots, n_{\max}`
       via :func:`li_coefficients_from_zeros`.
    3. Check positivity of every computed sum.
    4. Optionally repeat with zeros detected by the P13 critical-line
       scan (:func:`scan_critical_line_for_poles`) and report the
       maximum absolute difference between the two truncated arrays.

    Returns
    -------
    LiKeiperCertificate

    Notes
    -----
    No certified tail bound is computed for the selected cutoff or indices.
    Both branches use real part 1/2, which makes every exact paired term
    nonnegative independently of zero membership. Positivity flags therefore
    do not verify the full Li criterion or validate the scan independently.
    Differences also reflect the distinct zero windows and truncations.
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    if n_zeros < 1:
        raise ValueError("n_zeros must be >= 1")

    # --- Step 1+2: classical Li coefficients --------------------------------
    classical_zeros = fetch_riemann_zeros(n_zeros, dps=dps)
    lambda_classical = li_coefficients_from_zeros(
        classical_zeros,
        n_max,
        dps=dps,
    )

    # --- Step 3: positivity check ------------------------------------------
    pos_classical = bool(np.all(lambda_classical > 0))

    # --- Step 4: optional TNFR-derived comparison --------------------------
    lambda_tnfr: np.ndarray | None = None
    pos_tnfr: bool | None = None
    max_abs_diff: float | None = None
    notes: dict[str, Any] = {}

    if compare_tnfr:
        scan = scan_critical_line_for_poles(
            t_min=tnfr_t_min,
            t_max=tnfr_t_max,
            n_samples=tnfr_n_samples,
            dps=min(dps, 25),
        )
        if scan.detected_peaks.size == 0:
            notes["tnfr_scan"] = "no peaks detected -- skipping TNFR side"
        else:
            tnfr_zeros = np.array(
                [complex(0.5, float(t)) for t in scan.detected_peaks],
                dtype=complex,
            )
            lambda_tnfr = li_coefficients_from_zeros(
                tnfr_zeros,
                n_max,
                dps=dps,
            )
            pos_tnfr = bool(np.all(lambda_tnfr > 0))
            max_abs_diff = float(np.max(np.abs(lambda_classical - lambda_tnfr)))
            notes["tnfr_n_peaks"] = int(scan.detected_peaks.size)
            notes["tnfr_detection_quality"] = scan.detection_quality
            notes["tnfr_t_window"] = (tnfr_t_min, tnfr_t_max)

    return LiKeiperCertificate(
        n_max=n_max,
        n_zeros_classical=n_zeros,
        lambda_classical=lambda_classical,
        lambda_tnfr=lambda_tnfr,
        positivity_classical=pos_classical,
        positivity_tnfr=pos_tnfr,
        max_abs_difference=max_abs_diff,
        notes=notes,
    )


__all__ = [
    "li_coefficients_from_zeros",
    "LiKeiperCertificate",
    "verify_li_keiper_criterion",
]
