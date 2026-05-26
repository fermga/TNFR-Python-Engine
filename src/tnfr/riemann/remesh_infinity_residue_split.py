r"""P50 — R\_infinity residue split of the P31 oscillatory correction.

Diagnostic milestone that lifts the N15 REMESH-\ :math:`\infty` closure
(`theory/REMESH_INFINITY_DERIVATION.md`, master commits a1f298fd /
badac156 / 48b0574a) to the canonical TNFR-Riemann program.

Background
----------
The N15 derivation proved that the asymptotic REMESH limit

.. math::

    \mathcal{R}_\infty
        := \lim_{\tau_g \to \infty}
            \mathcal{R}_{\tau_l, \tau_g, \alpha}

is a bounded self-adjoint orthogonal projection on :math:`H^2(D)`,
equal to the projector onto :math:`\ker(I - \mathcal{R})`.  Its
fixed-mode subspace is spanned by Fourier components at the resonant
angular frequencies

.. math::

    \omega_k = \frac{2\pi k}{\operatorname{lcm}(\tau_l, \tau_g)},
        \qquad k \in \mathbb{Z},

with uniform spectral density
:math:`\rho = \operatorname{lcm}(\tau_l, \tau_g) / \pi`.

The §13septies / §13nonies analysis (`theory/TNFR_RIEMANN_RESEARCH_NOTES.md`)
identifies the residual obstruction of T-HP with the **oscillatory
half** :math:`S(T) = \pi^{-1}\arg\zeta(\tfrac12 + iT)`, structurally
matched with :math:`\ker(\mathcal{R}_\infty)`.  The **smooth half** of
the admissible rescaling operator :math:`\mathcal{F}` is closed at
the density level by P28 and lifted to the operator level by P30
(`structural_zero_density.py`, `admissible_rescaling.py`).

What P50 measures
-----------------
P50 takes the canonical TNFR prime-ladder reconstruction
:math:`S_{\mathrm{TNFR}}(T)` of :math:`S(T)` from P31
(`oscillatory_correction.py`), evaluates it on a uniform :math:`T`
grid aligned with the REMESH-\ :math:`\infty` resonant lattice, and
splits it via the Fourier-mode projector

.. math::

    \mathcal{R}_\infty[f](T)
        = \sum_{k\,:\,\omega_k\in\operatorname{lattice}}
            \hat f_k\, e^{i\omega_k T},

into a *range part* :math:`\mathcal{R}_\infty\,S_{\mathrm{TNFR}}` and
a *kernel part* :math:`(I - \mathcal{R}_\infty)\,S_{\mathrm{TNFR}}`.

A priori structural prediction
------------------------------
The prime-ladder spectrum (P12 / P14 canonical) has Fourier content
exclusively at the transcendental frequencies :math:`\{k \log p\}`.
These are linearly independent over :math:`\mathbb{Q}` (Baker's
theorem on linear independence of logarithms of algebraic numbers)
and in particular none coincides with a rational multiple of
:math:`\pi/\operatorname{lcm}(\tau_l, \tau_g)`.  Therefore the
N15-resonant lattice and the prime-ladder Fourier support are
disjoint, and the prediction is

.. math::

    \|\mathcal{R}_\infty\,S_{\mathrm{TNFR}}\|
        / \|S_{\mathrm{TNFR}}\| \;\to\; 0

as the diagnostic window length :math:`L \to \infty`.

Pre-registered verdicts
-----------------------
* ``RESIDUE_IN_KER_ONLY``
    Range fraction below the canonical threshold (default 5%).
    Confirms the §13septies / §13nonies structural identification:
    the P31 oscillatory correction lives in
    :math:`\ker(\mathcal{R}_\infty)`, structurally matching the
    location predicted for the T-HP residual obstruction.
* ``RESIDUE_IN_RANGE_ONLY``
    Kernel fraction below the threshold.  Would refute the P31
    construction as an oscillatory attack — the correction would be
    fully absorbed by the smooth half already closed by P30.
* ``RESIDUE_MIXED``
    Both fractions above the threshold.  Indicates either a gauge
    leak in P30 (smooth half not cleanly separated) or a numerical
    boundary artefact (window too short for the asymptotic limit).

Honest scope (mandatory)
------------------------
* P50 is a **structural-compatibility diagnostic only**.  It does NOT
  advance G4 = RH.  It does NOT close T-HP.  It does NOT promote any
  new canonical operator beyond the 13-operator catalog.
* Positive verdict (``RESIDUE_IN_KER_ONLY``) is **branch B2
  evidence** at the function-space level: it corroborates that any
  closure of T-HP through the oscillatory half requires structure
  that lives in :math:`\ker(\mathcal{R}_\infty)`, where the
  prime-ladder content already sits.  The decision between branches
  B1 / B2 / B3 of §13septies / §13octies remains open.
* This module imports ONLY canonical TNFR ingredients
  (`prime_ladder_oscillatory_sum` from P31, REMESH-canonical
  :math:`(\tau_l, \tau_g) = (4, 8)`, :math:`\alpha = 0.5`).  No
  external zeros, no mpmath, no fitting.

References
----------
* N15 master:  `theory/REMESH_INFINITY_DERIVATION.md` §§1-23.
* T-HP gap:    `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` §13septies,
               §13nonies.
* P31:         `oscillatory_correction.py`
               (`prime_ladder_oscillatory_sum`).
* P30:         `admissible_rescaling.py` (smooth half of T-HP).
* AGENTS.md:   "REMESH-\u221E Closure: Catalog Completeness Theorem".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .oscillatory_correction import prime_ladder_oscillatory_sum
from .von_mangoldt import (
    PrimeLadderSpectrum,
    build_prime_ladder_spectrum,
)

__all__ = [
    "build_resonant_bin_mask",
    "split_residue_by_remesh_infinity",
    "ResidueSplitCertificate",
    "compute_residue_split_certificate",
]


# ----------------------------------------------------------------------
# R_infinity projector as a DFT-bin mask
# ----------------------------------------------------------------------


def build_resonant_bin_mask(
    n_samples: int,
    *,
    tau_l: int = 4,
    tau_g: int = 8,
) -> np.ndarray:
    r"""Boolean mask over DFT bins selecting the N15-resonant lattice.

    On a uniform sample grid of length ``n_samples`` with unit spacing
    in :math:`T`-units, DFT bin :math:`k` corresponds to angular
    frequency :math:`\omega_k = 2\pi k / n_{\text{samples}}` (with
    bins :math:`k > n_{\text{samples}}/2` aliasing to the negative
    half).

    The N15-resonant subspace of :math:`\mathcal{R}_\infty` consists
    of Fourier modes at :math:`\omega = 2\pi m / L` for integer
    :math:`m`, where :math:`L = \operatorname{lcm}(\tau_l, \tau_g)`.
    For these to coincide with DFT bins we require
    ``n_samples`` to be a positive integer multiple of :math:`L`, and
    the resonant bins are :math:`k \in \{0, M, 2M, \dots\}` with
    :math:`M = n_{\text{samples}} / L`.

    Parameters
    ----------
    n_samples : int
        Length of the DFT.  Must be a positive multiple of
        ``lcm(tau_l, tau_g)``.
    tau_l, tau_g : int
        Canonical REMESH delays (default :math:`(\tau_l, \tau_g) =
        (4, 8)`, the documented TNFR canonical pair).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_samples,)``; ``True`` entries
        mark resonant DFT bins (including the negative-frequency
        aliases at :math:`k = n_{\text{samples}} - jM`).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if tau_l <= 0 or tau_g <= 0:
        raise ValueError("tau_l and tau_g must be positive")
    period = math.lcm(int(tau_l), int(tau_g))
    if n_samples % period != 0:
        raise ValueError(
            f"n_samples ({n_samples}) must be a multiple of "
            f"lcm(tau_l, tau_g) = {period}"
        )
    step = n_samples // period
    mask = np.zeros(n_samples, dtype=bool)
    mask[::step] = True
    return mask


def split_residue_by_remesh_infinity(
    signal: np.ndarray,
    *,
    tau_l: int = 4,
    tau_g: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Split a signal into range / kernel of :math:`\mathcal{R}_\infty`.

    Parameters
    ----------
    signal : np.ndarray
        Real 1-D signal sampled on the canonical unit-spacing grid in
        :math:`T`-units.  Length must be a multiple of
        :math:`\operatorname{lcm}(\tau_l, \tau_g)`.
    tau_l, tau_g : int
        Canonical REMESH delays.

    Returns
    -------
    range_part : np.ndarray
        :math:`\mathcal{R}_\infty[\text{signal}]`, real-valued, same
        shape as ``signal``.
    kernel_part : np.ndarray
        :math:`(I - \mathcal{R}_\infty)[\text{signal}]`, real-valued,
        same shape as ``signal``.

    Notes
    -----
    By construction ``range_part + kernel_part == signal`` exactly
    (up to FFT round-off).
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1-D")
    mask = build_resonant_bin_mask(
        sig.size, tau_l=tau_l, tau_g=tau_g
    )
    spectrum = np.fft.fft(sig)
    range_spectrum = np.where(mask, spectrum, 0.0 + 0.0j)
    kernel_spectrum = spectrum - range_spectrum
    range_part = np.real(np.fft.ifft(range_spectrum))
    kernel_part = np.real(np.fft.ifft(kernel_spectrum))
    return range_part, kernel_part


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResidueSplitCertificate:
    r"""Certificate for the P50 :math:`\mathcal{R}_\infty` residue split.

    Attributes
    ----------
    n_samples
        Length of the diagnostic window in :math:`T`-units.
    tau_l, tau_g
        Canonical REMESH delays.
    lcm_period
        :math:`\operatorname{lcm}(\tau_l, \tau_g)`.
    n_primes, max_power
        P12 / P14 prime-ladder parameters used to build
        :math:`S_{\mathrm{TNFR}}`.
    t_min, t_max
        Diagnostic window endpoints.
    norm_total
        :math:`\|S_{\mathrm{TNFR}}\|_2` on the window (L\ :sup:`2`
        norm, FFT convention).
    norm_in_range
        :math:`\|\mathcal{R}_\infty\,S_{\mathrm{TNFR}}\|_2`.
    norm_in_kernel
        :math:`\|(I - \mathcal{R}_\infty)\,S_{\mathrm{TNFR}}\|_2`.
    ratio_in_range, ratio_in_kernel
        Energy fractions in each subspace.  Sum exactly to 1 by
        Parseval.
    range_control_resonant
        Diagnostic sanity check: range fraction for the canonical
        positive-control signal :math:`\sin(\omega_1 T)` with
        :math:`\omega_1 = 2\pi /\operatorname{lcm}(\tau_l, \tau_g)`,
        which lies entirely in :math:`\operatorname{range}
        (\mathcal{R}_\infty)`.  Should be :math:`1` up to round-off.
    range_control_nonresonant
        Diagnostic sanity check: range fraction for the canonical
        negative-control signal :math:`\sin(\gamma T)` with
        :math:`\gamma` the Euler-Mascheroni constant (transcendental,
        non-resonant), which lies entirely in :math:`\ker
        (\mathcal{R}_\infty)` asymptotically.  Should be near zero.
    threshold
        Decision threshold on the dominant fraction (default 5%).
    verdict
        One of ``RESIDUE_IN_KER_ONLY``, ``RESIDUE_IN_RANGE_ONLY``,
        ``RESIDUE_MIXED``.
    notes
        Honest-scope reminder.
    """

    n_samples: int
    tau_l: int
    tau_g: int
    lcm_period: int
    n_primes: int
    max_power: int
    t_min: float
    t_max: float
    norm_total: float
    norm_in_range: float
    norm_in_kernel: float
    ratio_in_range: float
    ratio_in_kernel: float
    range_control_resonant: float
    range_control_nonresonant: float
    threshold: float
    verdict: str
    notes: str

    def summary(self) -> str:
        lines = [
            "P50 — REMESH-infinity Residue Split Certificate",
            f"  n_samples              : {self.n_samples}",
            f"  (tau_l, tau_g)         : ({self.tau_l}, {self.tau_g})",
            f"  lcm period             : {self.lcm_period}",
            f"  n_primes               : {self.n_primes}",
            f"  max_power (K)          : {self.max_power}",
            f"  T window               : "
            f"[{self.t_min:.3f}, {self.t_max:.3f}]",
            f"  ||S_TNFR||_2           : {self.norm_total:.4e}",
            f"  ||R_inf S_TNFR||_2     : {self.norm_in_range:.4e}",
            f"  ||(I-R_inf) S_TNFR||_2 : {self.norm_in_kernel:.4e}",
            f"  range fraction         : "
            f"{100.0 * self.ratio_in_range:7.4f} %",
            f"  kernel fraction        : "
            f"{100.0 * self.ratio_in_kernel:7.4f} %",
            "  controls (sanity):",
            f"    range[sin(omega_1 T)] : "
            f"{100.0 * self.range_control_resonant:7.4f} %  "
            "(expect ~100)",
            f"    range[sin(gamma T)]   : "
            f"{100.0 * self.range_control_nonresonant:7.4f} %  "
            "(expect ~0)",
            f"  threshold              : "
            f"{100.0 * self.threshold:.2f} %",
            f"  verdict                : {self.verdict}",
            f"  notes                  : {self.notes}",
        ]
        return "\n".join(lines)


def compute_residue_split_certificate(
    *,
    n_primes: int = 200,
    max_power: int = 8,
    tau_l: int = 4,
    tau_g: int = 8,
    n_periods: int = 64,
    t_min: float = 1.0,
    threshold: float = 0.05,
) -> ResidueSplitCertificate:
    r"""Run the full P50 diagnostic and emit a certificate.

    Builds the canonical P12 / P14 prime-ladder spectrum, evaluates
    :math:`S_{\mathrm{TNFR}}(T)` on a uniform :math:`T` grid of
    length :math:`n_{\text{periods}} \cdot \operatorname{lcm}
    (\tau_l, \tau_g)`, splits via the
    :math:`\mathcal{R}_\infty` Fourier-mode projector, and reports
    range / kernel norms plus two canonical control signals.

    Parameters
    ----------
    n_primes : int, default 200
        Primes used in the canonical prime-ladder spectrum (P12).
    max_power : int, default 8
        REMESH echo cap :math:`K` (P12).
    tau_l, tau_g : int, default (4, 8)
        Canonical REMESH delays.  Default is the documented TNFR
        canonical pair.
    n_periods : int, default 64
        Window length in units of :math:`\operatorname{lcm}(\tau_l,
        \tau_g)`.  Larger values sharpen the asymptotic verdict.
    t_min : float, default 1.0
        Diagnostic window start in :math:`T`-units (kept positive to
        avoid the :math:`T = 0` singularity of the smooth density).
    threshold : float, default 0.05
        Decision threshold on the dominant energy fraction.

    Returns
    -------
    ResidueSplitCertificate
    """
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    if not (0.0 < threshold < 0.5):
        raise ValueError("threshold must be in (0, 0.5)")
    period = math.lcm(int(tau_l), int(tau_g))
    n_samples = n_periods * period
    spectrum: PrimeLadderSpectrum = build_prime_ladder_spectrum(
        n_primes, max_power=max_power
    )

    t_grid = t_min + np.arange(n_samples, dtype=float)
    t_max = float(t_grid[-1])
    signal = np.asarray(
        prime_ladder_oscillatory_sum(t_grid, spectrum), dtype=float
    )

    range_part, kernel_part = split_residue_by_remesh_infinity(
        signal, tau_l=tau_l, tau_g=tau_g
    )

    norm_total = float(np.linalg.norm(signal))
    norm_range = float(np.linalg.norm(range_part))
    norm_kernel = float(np.linalg.norm(kernel_part))
    if norm_total <= 0.0:
        raise RuntimeError(
            "S_TNFR vanished on the diagnostic window; refusing to "
            "normalise"
        )
    ratio_range = (norm_range / norm_total) ** 2
    ratio_kernel = (norm_kernel / norm_total) ** 2

    # Canonical sanity controls
    omega_resonant = 2.0 * math.pi / period
    control_resonant = np.sin(omega_resonant * t_grid)
    rng_res, _ = split_residue_by_remesh_infinity(
        control_resonant, tau_l=tau_l, tau_g=tau_g
    )
    n_res = float(np.linalg.norm(control_resonant))
    ctrl_res_frac = (float(np.linalg.norm(rng_res)) / n_res) ** 2

    # Euler-Mascheroni constant: transcendental, non-resonant.
    gamma_em = 0.5772156649015329
    control_nonres = np.sin(gamma_em * t_grid)
    rng_nonres, _ = split_residue_by_remesh_infinity(
        control_nonres, tau_l=tau_l, tau_g=tau_g
    )
    n_nonres = float(np.linalg.norm(control_nonres))
    ctrl_nonres_frac = (float(np.linalg.norm(rng_nonres)) / n_nonres) ** 2

    if ratio_range < threshold and ratio_kernel >= threshold:
        verdict = "RESIDUE_IN_KER_ONLY"
        notes = (
            "Branch B2 evidence at the function-space level: the "
            "P31 oscillatory correction lives in ker(R_inf), "
            "structurally matching the location predicted for the "
            "T-HP residual obstruction. Does NOT advance G4 = RH."
        )
    elif ratio_kernel < threshold and ratio_range >= threshold:
        verdict = "RESIDUE_IN_RANGE_ONLY"
        notes = (
            "Refutes P31 as an oscillatory attack: the correction "
            "would already be absorbed by the smooth half (P30). "
            "Requires audit of P30 / P31 reconstruction. Does NOT "
            "advance G4 = RH."
        )
    else:
        verdict = "RESIDUE_MIXED"
        notes = (
            "Both fractions above threshold: gauge leak in P30 "
            "smooth half or boundary artefact (window too short). "
            "Increase n_periods and re-test. Does NOT advance "
            "G4 = RH."
        )

    return ResidueSplitCertificate(
        n_samples=n_samples,
        tau_l=int(tau_l),
        tau_g=int(tau_g),
        lcm_period=period,
        n_primes=int(n_primes),
        max_power=int(max_power),
        t_min=float(t_min),
        t_max=t_max,
        norm_total=norm_total,
        norm_in_range=norm_range,
        norm_in_kernel=norm_kernel,
        ratio_in_range=ratio_range,
        ratio_in_kernel=ratio_kernel,
        range_control_resonant=ctrl_res_frac,
        range_control_nonresonant=ctrl_nonres_frac,
        threshold=float(threshold),
        verdict=verdict,
        notes=notes,
    )
