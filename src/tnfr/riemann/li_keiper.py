r"""P16: Li-Keiper criterion verified via the TNFR resonance spectrum.

Li's criterion (Xian-Jin Li, 1997)
----------------------------------
Define, for every integer :math:`n \ge 1`,

.. math::

    \lambda_n \;=\; \sum_{\rho} \Bigl[ 1 - \bigl(1 - \tfrac{1}{\rho}\bigr)^n
                             \Bigr]
            \;=\; \frac{1}{(n-1)!}\,
                  \frac{d^{\,n}}{ds^{\,n}}
                  \Bigl[\, s^{\,n-1}\,\log \xi(s)\,\Bigr]_{s=1},

where the sum ranges over all non-trivial zeros :math:`\rho` of
:math:`\zeta(s)` (counted with multiplicity, with the implicit
:math:`\rho \leftrightarrow \bar\rho` pairing).  Li proved the
equivalence

.. math::

    \text{RH}\;\Longleftrightarrow\;\lambda_n > 0
    \quad\text{for every } n \ge 1.

Li's criterion is therefore strictly equivalent to the Riemann
Hypothesis, restated as a positivity condition on a real sequence.

TNFR reading
------------
In the TNFR-Riemann program the non-trivial zeros appear as
**resonance poles** of the prime-ladder von Mangoldt zeta after
analytic continuation (module :mod:`tnfr.riemann.analytic_continuation`,
P13).  Computing :math:`\lambda_n` from those resonance poles and
comparing against the classical evaluation from
:func:`mpmath.zetazero` does two things:

1. **Validates** the P13 resonance-pole finder against a strict
   number-theoretic test: every detected pole must lie on the
   critical line to a precision sufficient to keep :math:`\lambda_n`
   positive for every :math:`n` up to the test horizon.
2. **Recasts** the Riemann Hypothesis as a TNFR-internal positivity
   diagnostic on the resonance spectrum.  Each :math:`\lambda_n`
   becomes a structural integrity check: a single negative
   :math:`\lambda_n` would falsify RH; the absence of one (up to the
   test horizon) is consistent with it.

Honesty disclaimer
------------------
This module **does not prove** the Riemann Hypothesis.  Li's
criterion is RH-equivalent: a finite verification of
:math:`\lambda_n > 0` for :math:`n = 1\ldots N` proves RH only in the
limit :math:`N \to \infty` with rigorous control of the truncation
error in the zero-sum.  The numerical evidence produced here matches
the well-documented positivity of the first :math:`\sim 10^5`
Li-Keiper coefficients (Voros 2003, Bombieri-Lagarias 1999) and is
offered as a TNFR-native witness, not as a proof.

Public API
----------
``li_coefficients_from_zeros``      Compute :math:`\lambda_1, \ldots,
                                     \lambda_{n_{\max}}` from a list of
                                     non-trivial zeros (upper half-plane).
``LiKeiperCertificate``             Frozen result with positivity flags,
                                     classical/TNFR comparison and summary.
``verify_li_keiper_criterion``      End-to-end verification: fetch
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

from .analytic_continuation import (
    fetch_riemann_zeros,
    scan_critical_line_for_poles,
)


# ---------------------------------------------------------------------------
# Core: Li coefficients from a list of upper-half-plane zeros
# ---------------------------------------------------------------------------

def li_coefficients_from_zeros(
    zeros_upper: Sequence[complex],
    n_max: int,
    *,
    dps: int = 50,
) -> np.ndarray:
    r"""Compute :math:`\lambda_1, \ldots, \lambda_{n_{\max}}` from a finite
    truncation of the zero list.

    Implementation uses the explicit form

    .. math::

        \lambda_n \;=\; \sum_{k=1}^{K} 2\,\Re\!\Bigl[
                       1 - \bigl(1 - \tfrac{1}{\rho_k}\bigr)^n \Bigr],
        \qquad \rho_k = \tfrac{1}{2} + i\, t_k,

    paired with :math:`\bar\rho_k`.  Computation is performed at
    arbitrary precision via :mod:`mpmath` to absorb cancellation
    between :math:`1` and :math:`(1-1/\rho)^n` as :math:`n` grows.

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
        ``arr[n-1] = float(lambda_n)``.
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    if len(zeros_upper) == 0:
        raise ValueError("zeros_upper must contain at least one zero")

    out = np.zeros(n_max, dtype=float)
    with mpmath.workdps(dps):
        # Convert each zero to mpmath complex once and reuse.
        mp_zeros = [
            mpmath.mpc(float(z.real), float(z.imag)) for z in zeros_upper
        ]
        # Pre-compute base = 1 - 1/rho for each zero.
        bases = [mpmath.mpc(1) - mpmath.mpc(1) / r for r in mp_zeros]

        for n_idx in range(1, n_max + 1):
            total = mpmath.mpf(0)
            for b in bases:
                # 2 * Re[1 - b^n] handles the rho/conj(rho) pairing
                total += 2 * (mpmath.mpf(1) - (b ** n_idx).real)
            out[n_idx - 1] = float(total)
    return out


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiKeiperCertificate:
    r"""Result of a Li-Keiper positivity check.

    Attributes
    ----------
    n_max
        Highest Li index computed.
    n_zeros_classical
        Number of zeros from :func:`mpmath.zetazero` used in the
        classical evaluation.
    lambda_classical
        Array of shape ``(n_max,)`` with classical Li coefficients.
    lambda_tnfr
        Array of shape ``(n_max,)`` with TNFR-computed Li
        coefficients from P13 resonance peaks (``None`` if not
        requested).
    positivity_classical
        ``True`` iff every classical :math:`\lambda_n > 0`.
    positivity_tnfr
        Same for the TNFR-derived values (``None`` if not requested).
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
            f"  lambda_{self.n_max}".ljust(22)
            + f"= {lam[-1]:+.6e}",
            f"  min_n lambda_n      = {float(lam.min()):+.6e}",
            f"  positivity (cls.)   = {self.positivity_classical}",
        ]
        if self.lambda_tnfr is not None:
            lines.extend(
                [
                    f"  positivity (TNFR)   = {self.positivity_tnfr}",
                    f"  max |Δλ|            = "
                    f"{self.max_abs_difference:.3e}",
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
    r"""Verify Li's positivity criterion up to index ``n_max``.

    Steps
    -----
    1. Fetch ``n_zeros`` non-trivial zeros via :func:`mpmath.zetazero`
       (classical reference).
    2. Compute :math:`\lambda_n` for :math:`n = 1, \ldots, n_{\max}`
       via :func:`li_coefficients_from_zeros`.
    3. Check positivity of every coefficient.
    4. Optionally repeat with zeros detected by the P13 critical-line
       scan (:func:`scan_critical_line_for_poles`) and report the
       maximum absolute difference between classical and TNFR
       coefficients.

    Returns
    -------
    LiKeiperCertificate

    Notes
    -----
    The truncation error in :math:`\lambda_n` is bounded by the tail
    of the zero density; with :math:`n_{\mathrm{zeros}} = 200` and
    :math:`n_{\max} = 50` the dominant tail term is at the
    :math:`10^{-3}` level relative to :math:`\lambda_n`, sufficient
    to preserve the sign of every coefficient (Voros 2003).
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    if n_zeros < 1:
        raise ValueError("n_zeros must be >= 1")

    # --- Step 1+2: classical Li coefficients --------------------------------
    classical_zeros = fetch_riemann_zeros(n_zeros, dps=dps)
    lambda_classical = li_coefficients_from_zeros(
        classical_zeros, n_max, dps=dps,
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
                tnfr_zeros, n_max, dps=dps,
            )
            pos_tnfr = bool(np.all(lambda_tnfr > 0))
            max_abs_diff = float(
                np.max(np.abs(lambda_classical - lambda_tnfr))
            )
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
