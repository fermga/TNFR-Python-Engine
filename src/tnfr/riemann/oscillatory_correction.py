r"""P31 — Prime-Ladder Oscillatory Correction of Smooth Zero Targets.

Branch B1 of §13octies: structural reconstruction of the oscillatory
remainder :math:`S(T) = \pi^{-1}\arg\zeta(\tfrac12+iT)` from canonical
TNFR ingredients only.

Motivation
----------
P28 (§13sexies) and P30 (§13nonies) close the **smooth half** of
Conjecture T-HP (§13septies) at the density and operator levels
respectively.  The residual Wasserstein-1 gap
:math:`W_1(\{\widetilde\gamma_i\}, \{\gamma_n\})` observed in P30
(:math:`\approx 1.4`-:math:`1.7` for :math:`N \in \{20, 40\}`) is
exactly the **oscillatory remainder** :math:`S(T)`.

§13nonies.4 tried three multiplicative perturbations of the smooth
targets built from constants :math:`(\varphi, \gamma, \pi, e)` only.
All three yielded :math:`\approx 0\%` improvement.  The failure is
structural: :math:`S(T)` is a **prime-indexed multi-frequency** sum,
not a single canonical frequency.

This module pursues branch B1 with a **different canonical
ingredient**: the prime-ladder spectrum itself.  Concretely, the
classical Riemann-von Mangoldt formula on the critical line gives

.. math::

    \pi\,S(T) \;=\; -\sum_p\sum_{k\geq 1}
        \frac{\sin(kT\log p)}{k\,p^{k/2}}
        \;+\; \mathcal O(1/T).

Re-expressing the prime-power sum in terms of the canonical TNFR
prime-ladder spectrum :math:`\Sigma_{N,K} = \{(\mu_{p,k}, w_{p,k})\}`
with :math:`\mu_{p,k} = k\log p` and :math:`w_{p,k} = \log p` yields

.. math::

    \pi\,S_{\mathrm{TNFR}}(T;\,N,K) \;=\;
        -\sum_{(\mu,w)\in\Sigma_{N,K}}
        \frac{w}{\mu}\cdot\frac{\sin(T\mu)}{e^{\mu/2}}.

All three factors (:math:`\mu`, :math:`w`, :math:`e^{\mu/2}`) come from
the canonical P12 / P14 spectral data; :math:`\pi` is canonical (tetrad
:math:`\pi \leftrightarrow K_\varphi`).  No new operator and no
non-canonical input is used.

The position-level correction follows from
:math:`N(\gamma_n) = \overline N(\gamma_n) + S(\gamma_n) = n`:

.. math::

    \gamma_n \;\approx\; \widetilde\gamma_n
        - \frac{S_{\mathrm{TNFR}}(\widetilde\gamma_n)}
               {\overline N'(\widetilde\gamma_n)}.

What this module closes / does not close
----------------------------------------
**Closes**: the operator-level smooth half of T-HP already closed by
P30 is now complemented by an explicit canonical operator-level
candidate for the oscillatory half built from prime-ladder data.

**Does NOT close**:

* Canonicity from the nodal equation (sub-problem (2) of T-HP):
  the prime-ladder spectrum is canonical, but expressing
  :math:`S_{\mathrm{TNFR}}` as a **derivation** of the canonical
  nodal evolution (rather than as an ingredient plugged into the
  Riemann-von Mangoldt template) is still open.
* Positivity coincidence with the Weil quadratic form
  (sub-problem (3) of T-HP).
* Gap **G4 = the Riemann Hypothesis** itself.

Numerical positivity in the certificate constitutes **branch B1
evidence**: that the canonical 13-operator catalog plus the
prime-ladder spectrum suffices to reproduce :math:`S(T)`.  Numerical
negativity (no improvement) would corroborate **branch B2**: a
genuinely new canonical operator is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hilbert_polya import fetch_zero_imaginary_parts, wasserstein_1_distance
from .structural_zero_density import build_structural_t_hp, smooth_zero_density
from .von_mangoldt import PrimeLadderSpectrum, build_prime_ladder_spectrum

__all__ = [
    "prime_ladder_oscillatory_sum",
    "apply_oscillatory_correction",
    "OscillatoryCorrectionCertificate",
    "compute_oscillatory_correction_certificate",
]


# ----------------------------------------------------------------------
# Core canonical reconstruction of S(T)
# ----------------------------------------------------------------------


def prime_ladder_oscillatory_sum(
    T: float | np.ndarray,
    spectrum: PrimeLadderSpectrum,
) -> float | np.ndarray:
    r"""Evaluate :math:`S_{\mathrm{TNFR}}(T;\,N,K)` from canonical data.

    Computes

    .. math::

        S_{\mathrm{TNFR}}(T) = -\frac{1}{\pi}
            \sum_{(\mu,w)\in\Sigma_{N,K}}
            \frac{w}{\mu}\cdot\frac{\sin(T\mu)}{e^{\mu/2}}

    using ONLY the prime-ladder spectrum (P12 / P14 canonical) and
    the canonical constant :math:`\pi`.

    Parameters
    ----------
    T : float or np.ndarray
        Evaluation height(s) on the critical line.
    spectrum : PrimeLadderSpectrum
        Canonical TNFR prime-ladder spectrum.

    Returns
    -------
    float or np.ndarray
        :math:`S_{\mathrm{TNFR}}(T)`, same shape as ``T``.
    """
    mu = np.asarray(spectrum.eigenvalues, dtype=float)
    w = np.asarray(spectrum.weights, dtype=float)
    if mu.size == 0:
        raise ValueError("empty prime-ladder spectrum")
    if np.any(mu <= 0.0):
        raise ValueError("prime-ladder eigenvalues must be positive")
    # Pre-compute amplitude coefficients a_j = (w_j / mu_j) * exp(-mu_j/2).
    amp = (w / mu) * np.exp(-0.5 * mu)
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        s = float(np.sum(amp * np.sin(float(T_arr) * mu)))
        return -s / math.pi
    # Vectorised over T: shape (len(T), len(mu)).
    sines = np.sin(np.outer(T_arr, mu))
    s_vec = sines @ amp
    return -s_vec / math.pi


# ----------------------------------------------------------------------
# Position-level correction
# ----------------------------------------------------------------------


def apply_oscillatory_correction(
    smooth_targets: np.ndarray,
    spectrum: PrimeLadderSpectrum,
    *,
    damping: float = 1.0,
) -> np.ndarray:
    r"""Correct :math:`\{\widetilde\gamma_i\}` using the TNFR S(T).

    Applies the first-order Newton step

    .. math::

        \gamma_i^{\mathrm{corr}} = \widetilde\gamma_i
            - d\cdot\frac{S_{\mathrm{TNFR}}(\widetilde\gamma_i)}
                         {\overline N'(\widetilde\gamma_i)},

    where ``d`` is the optional damping factor (default ``1.0``).

    The smooth density :math:`\overline N'` is the canonical
    P28 density (archimedean Riemann-Siegel theta derivative);
    :math:`S_{\mathrm{TNFR}}` is the canonical prime-ladder
    reconstruction.

    Parameters
    ----------
    smooth_targets : np.ndarray
        Canonical P28 smooth targets :math:`\widetilde\gamma_i`.
    spectrum : PrimeLadderSpectrum
        Canonical prime-ladder spectrum used to evaluate
        :math:`S_{\mathrm{TNFR}}`.
    damping : float, default 1.0
        Multiplicative damping ``d``.  ``d = 0`` reproduces the smooth
        targets unchanged; ``d = 1`` is the unmoderated Newton step.

    Returns
    -------
    np.ndarray
        Corrected zero-position candidates, sorted ascending.
    """
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    targets = np.asarray(smooth_targets, dtype=float)
    if targets.ndim != 1:
        raise ValueError("smooth_targets must be 1-D")
    s_vals = np.asarray(prime_ladder_oscillatory_sum(targets, spectrum), dtype=float)
    densities = np.array([smooth_zero_density(float(t)) for t in targets], dtype=float)
    if np.any(densities <= 0.0):
        raise RuntimeError("smooth density vanished at a target; refusing to divide")
    delta = -damping * s_vals / densities
    corrected = targets + delta
    if np.any(corrected <= 0.0):
        raise RuntimeError(
            "oscillatory correction drove a target non-positive; " "reduce damping"
        )
    return np.sort(corrected)


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OscillatoryCorrectionCertificate:
    r"""Certificate for the P31 prime-ladder oscillatory correction.

    Attributes
    ----------
    n_targets
        Number of smooth targets / true zeros compared.
    n_primes
        Number of primes used in the prime-ladder spectrum.
    max_power
        Maximum REMESH echo index :math:`K`.
    best_damping
        Damping :math:`d` (in the swept range) minimising
        :math:`W_1(\{\gamma_i^{\mathrm{corr}}\}, \{\gamma_n\})`.
    w1_smooth_vs_true
        :math:`W_1(\{\widetilde\gamma_i\}, \{\gamma_n\})` — baseline
        (P30 smooth-half residual).
    w1_corrected_vs_true
        :math:`W_1(\{\gamma_i^{\mathrm{corr}}\}, \{\gamma_n\})` at
        ``best_damping``.
    improvement_over_smooth
        :math:`(W_1^{\mathrm{smooth}} - W_1^{\mathrm{corr}})
                / W_1^{\mathrm{smooth}}`.  Positive ⇒ the canonical
        prime-ladder reconstruction of :math:`S(T)` reduces the gap.
    max_abs_s_at_targets
        :math:`\max_i |S_{\mathrm{TNFR}}(\widetilde\gamma_i)|`,
        sanity check on the reconstruction magnitude.
    damping_sweep
        List of ``(damping, W_1)`` pairs across the sweep.
    notes
        Honest-scope reminder.  Positive improvement is **branch B1
        evidence**, not a closure of G4 = RH.
    """

    n_targets: int
    n_primes: int
    max_power: int
    best_damping: float
    w1_smooth_vs_true: float
    w1_corrected_vs_true: float
    improvement_over_smooth: float
    max_abs_s_at_targets: float
    damping_sweep: tuple[tuple[float, float], ...]
    notes: str

    def summary(self) -> str:
        lines = [
            "P31 — Prime-Ladder Oscillatory Correction Certificate",
            f"  n_targets               : {self.n_targets}",
            f"  n_primes                : {self.n_primes}",
            f"  max_power (K)           : {self.max_power}",
            f"  best damping            : {self.best_damping:.4f}",
            f"  W_1 smooth vs true      : {self.w1_smooth_vs_true:.4e}",
            f"  W_1 corrected vs true   : {self.w1_corrected_vs_true:.4e}",
            "  improvement over smooth : "
            f"{100.0 * self.improvement_over_smooth:+.2f} %",
            "  max |S_TNFR(t_i)|       : " f"{self.max_abs_s_at_targets:.4e}",
            f"  notes                   : {self.notes}",
        ]
        return "\n".join(lines)


def compute_oscillatory_correction_certificate(
    n_targets: int,
    *,
    n_primes: int = 200,
    max_power: int = 8,
    damping_grid: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    ),
    dps: int = 30,
) -> OscillatoryCorrectionCertificate:
    r"""Run the full P31 reconstruction and emit a certificate.

    Parameters
    ----------
    n_targets : int
        Number of smooth zeros / true zeros compared.
    n_primes : int, default 200
        Primes used in the canonical prime-ladder spectrum.  Larger
        values improve the resolution of :math:`S_{\mathrm{TNFR}}`
        but are bounded above by the canonical truncation tolerance.
    max_power : int, default 8
        REMESH echo cap :math:`K`.
    damping_grid : tuple of float, default (0, .25, .5, .75, 1, 1.25, 1.5)
        Damping factors swept; best is retained.
    dps : int, default 30
        mpmath precision for the true reference zeros and for the
        smooth-target Newton solver.

    Returns
    -------
    OscillatoryCorrectionCertificate
    """
    if n_targets < 1:
        raise ValueError("n_targets must be >= 1")
    spectrum = build_prime_ladder_spectrum(n_primes, max_power=max_power)
    smooth_targets = build_structural_t_hp(n_targets, dps=dps)
    true_gammas = fetch_zero_imaginary_parts(n_targets, dps=dps)

    s_at_targets = np.asarray(
        prime_ladder_oscillatory_sum(smooth_targets, spectrum),
        dtype=float,
    )
    max_abs_s = float(np.max(np.abs(s_at_targets)))

    w1_smooth = wasserstein_1_distance(smooth_targets, true_gammas)

    sweep: list[tuple[float, float]] = []
    best_d = 0.0
    best_w1 = w1_smooth
    for d in damping_grid:
        try:
            corrected = apply_oscillatory_correction(
                smooth_targets, spectrum, damping=float(d)
            )
        except RuntimeError:
            # Correction drove a target non-positive; skip.
            sweep.append((float(d), float("inf")))
            continue
        w1_d = wasserstein_1_distance(corrected, true_gammas)
        sweep.append((float(d), w1_d))
        if w1_d < best_w1:
            best_w1 = w1_d
            best_d = float(d)

    improvement = (w1_smooth - best_w1) / w1_smooth if w1_smooth > 0 else 0.0

    notes = (
        "Honest scope: positive improvement is branch B1 evidence — "
        "the canonical prime-ladder spectrum suffices to reduce the "
        "S(T) residual.  This does NOT close G4 = RH, does NOT prove "
        "canonicity from the nodal equation (sub-problem (2)), and "
        "does NOT establish positivity coincidence with the Weil "
        "quadratic form (sub-problem (3)).  Negative improvement "
        "corroborates branch B2 (a new canonical operator required)."
    )

    return OscillatoryCorrectionCertificate(
        n_targets=int(n_targets),
        n_primes=int(spectrum.n_primes),
        max_power=int(max_power),
        best_damping=float(best_d),
        w1_smooth_vs_true=float(w1_smooth),
        w1_corrected_vs_true=float(best_w1),
        improvement_over_smooth=float(improvement),
        max_abs_s_at_targets=max_abs_s,
        damping_sweep=tuple(sweep),
        notes=notes,
    )
