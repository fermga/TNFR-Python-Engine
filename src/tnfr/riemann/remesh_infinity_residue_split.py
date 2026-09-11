r"""P50 — fixed-delay Fourier split of the P31 oscillatory correction.

The public names and verdict strings are retained from the historical N15
programme for compatibility. The implementation is a finite cyclic DFT-bin
projection; it is not a literal runtime :math:`\tau_g\to\infty` limit.

Background
----------
The corrected N15 analysis defines, for a finite cyclic history window, the
Cesàro limit of the fixed-coefficient filter

.. math::

    P_d := \lim_{N\to\infty}\frac1N\sum_{j=0}^{N-1}F^j,
    \qquad
    F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g},

where :math:`S` is the unitary cyclic shift. For
:math:`0<\alpha<1`, :math:`F` is a normal contraction and :math:`P_d`
is the orthogonal projector onto :math:`\ker(I-F)`. Its
fixed-mode subspace is spanned by Fourier components at the resonant
angular frequencies

.. math::

    \omega_k = \frac{2\pi k}{\gcd(\tau_l, \tau_g)},
        \qquad k \in \mathbb{Z},

on a compatible discrete window. There is no continuum spectral-density or
runtime-limit conclusion.

The §13septies / §13nonies analysis (`theory/TNFR_RIEMANN_RESEARCH_NOTES.md`)
historically compared the residual obstruction of T-HP with the **oscillatory
half** :math:`S(T) = \pi^{-1}\arg\zeta(\tfrac12 + iT)` and the complement
of this selected periodic subspace. No intertwining theorem identifies those
objects. The **smooth half** of
the admissible rescaling operator :math:`\mathcal{F}` is closed at
the density level by P28 and lifted to the operator level by P30
(`structural_zero_density.py`, `admissible_rescaling.py`).

What P50 measures
-----------------
P50 takes the finite TNFR prime-ladder reconstruction
:math:`S_{\mathrm{TNFR}}(T)` of :math:`S(T)` from P31
(`oscillatory_correction.py`), evaluates it on a uniform :math:`T`
grid, and splits its DFT via the fixed-delay Fourier projector

.. math::

    P_d[f](T)
        = \sum_{k\,:\,k\in\operatorname{fixed\ bins}}
            \hat f_k\, e^{i\omega_k T},

into a *range part* :math:`P_d S_{\mathrm{TNFR}}` and a *complement
part* :math:`(I-P_d)S_{\mathrm{TNFR}}`. The legacy API calls the latter
``kernel_part`` because it is in :math:`\ker P_d`.

A priori structural prediction
------------------------------
The finite prime-ladder signal uses frequencies :math:`k\log p`, which do
not generally align with the selected periodic modes. A finite rectangular
window nevertheless spreads off-grid frequencies across DFT bins, so the two
computed parts are not exact analytic spectral supports. For fixed finite
prime-ladder content, the testable large-window expectation is

.. math::

    \|P_d S_{\mathrm{TNFR}}\|
        / \|S_{\mathrm{TNFR}}\| \;\to\; 0

as the diagnostic window length grows. The certificate reports a finite
sample and does not prove that limit.

Pre-registered verdicts
-----------------------
* ``RESIDUE_IN_KER_ONLY``
    Range fraction below the selected threshold (default 5%). The finite
    sample has little energy in the declared periodic subspace.
* ``RESIDUE_IN_RANGE_ONLY``
    Complement fraction below the threshold. The finite sample lies mostly
    in the declared periodic subspace.
* ``RESIDUE_MIXED``
    Both fractions exceed the threshold. This is a descriptive finite-window
    outcome and can change with the window or signal truncation.

Honest scope (mandatory)
------------------------
* P50 is a finite Fourier diagnostic only. It does not advance G4 = RH,
  close T-HP, identify its smooth/oscillatory split, or certify a runtime
  REMESH limit.
* A positive legacy verdict is evidence only about the selected bins, window,
  threshold, and finite prime-ladder signal.
* Computing this auxiliary projection requires no additional registry entry;
  that fact does not prove the 13-operator catalog complete.

References
----------
* Corrected N15 record: `theory/REMESH_INFINITY_DERIVATION.md` §§1-23.
* P31:         `oscillatory_correction.py`
               (`prime_ladder_oscillatory_sum`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .oscillatory_correction import prime_ladder_oscillatory_sum
from .von_mangoldt import PrimeLadderSpectrum, build_prime_ladder_spectrum

__all__ = [
    "build_resonant_bin_mask",
    "split_residue_by_remesh_infinity",
    "ResidueSplitCertificate",
    "compute_residue_split_certificate",
]


# ----------------------------------------------------------------------
# Fixed-delay cyclic-filter projector as a DFT-bin mask
# ----------------------------------------------------------------------


def build_resonant_bin_mask(
    n_samples: int,
    *,
    tau_l: int = 4,
    tau_g: int = 8,
) -> np.ndarray:
    r"""Return DFT bins fixed by the finite cyclic delay filter.

    On a uniform sample grid of length ``n_samples`` with unit spacing
    in :math:`T`-units, DFT bin :math:`k` corresponds to angular
    frequency :math:`\omega_k = 2\pi k / n_{\text{samples}}` (with
    bins :math:`k > n_{\text{samples}}/2` aliasing to the negative
    half).

    For positive mixing coefficients, a unit-circle mode is fixed precisely
    when both delay phases equal one. Thus its period is
    :math:`d=\gcd(\tau_l,\tau_g)`. The public API retains the historical
    requirement that ``n_samples`` be a multiple of
    :math:`\operatorname{lcm}(\tau_l,\tau_g)`; this is stricter than needed
    but guarantees compatibility. The fixed bins are
    :math:`k\in\{0,M,2M,\ldots,(d-1)M\}` with
    :math:`M=n_{\text{samples}}/d`.

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
        Boolean array of shape ``(n_samples,)`` whose ``True`` entries mark
        the common fixed modes of the two delays.
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
    fixed_period = math.gcd(int(tau_l), int(tau_g))
    step = n_samples // fixed_period
    mask = np.zeros(n_samples, dtype=bool)
    mask[::step] = True
    return mask


def split_residue_by_remesh_infinity(
    signal: np.ndarray,
    *,
    tau_l: int = 4,
    tau_g: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Split a signal into range / kernel of the cyclic projector.

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
        Projection onto the common fixed-delay DFT bins, real-valued and the
        same shape as ``signal``.
    kernel_part : np.ndarray
        Orthogonal complement, equivalently the kernel of this projection.

    Notes
    -----
    By construction ``range_part + kernel_part == signal`` exactly
    (up to FFT round-off).
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1-D")
    mask = build_resonant_bin_mask(sig.size, tau_l=tau_l, tau_g=tau_g)
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
    r"""Certificate for the legacy P50 fixed-delay Fourier split.

    Attributes
    ----------
    n_samples
        Length of the diagnostic window in :math:`T`-units.
    tau_l, tau_g
        Canonical REMESH delays.
    lcm_period
        Historical sampling-alignment period
        :math:`\operatorname{lcm}(\tau_l, \tau_g)`. The fixed-mode period is
        instead available as :attr:`fixed_period_gcd`.
    n_primes, max_power
        P12 / P14 prime-ladder parameters used to build
        :math:`S_{\mathrm{TNFR}}`.
    t_min, t_max
        Diagnostic window endpoints.
    norm_total
        :math:`\|S_{\mathrm{TNFR}}\|_2` on the window (L\ :sup:`2`
        norm, FFT convention).
    norm_in_range
        Norm of the component in the selected fixed-delay subspace.
    norm_in_kernel
        Norm of the orthogonal complement.
    ratio_in_range, ratio_in_kernel
        Energy fractions in each subspace.  Sum exactly to 1 by
        Parseval.
    range_control_resonant
        Diagnostic sanity check: range fraction for a cosine at the first
        common fixed-delay frequency. Should be :math:`1` up to round-off.
    range_control_nonresonant
        Diagnostic sanity check for an exactly unselected DFT bin. Should be
        zero up to round-off whenever the selected subspace is proper.
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

    @property
    def fixed_period_gcd(self) -> int:
        """Period of the modes fixed by both declared delays."""

        return math.gcd(self.tau_l, self.tau_g)

    def summary(self) -> str:
        lines = [
            "P50 — Fixed-Delay Fourier Split Certificate",
            f"  n_samples              : {self.n_samples}",
            f"  (tau_l, tau_g)         : ({self.tau_l}, {self.tau_g})",
            f"  lcm sample alignment   : {self.lcm_period}",
            f"  gcd fixed-mode period  : {self.fixed_period_gcd}",
            f"  n_primes               : {self.n_primes}",
            f"  max_power (K)          : {self.max_power}",
            f"  T window               : " f"[{self.t_min:.3f}, {self.t_max:.3f}]",
            f"  ||S_TNFR||_2           : {self.norm_total:.4e}",
            f"  ||P_d S_TNFR||_2       : {self.norm_in_range:.4e}",
            f"  ||(I-P_d) S_TNFR||_2   : {self.norm_in_kernel:.4e}",
            f"  range fraction         : " f"{100.0 * self.ratio_in_range:7.4f} %",
            f"  kernel fraction        : " f"{100.0 * self.ratio_in_kernel:7.4f} %",
            "  controls (sanity):",
            f"    range[fixed cosine]   : "
            f"{100.0 * self.range_control_resonant:7.4f} %  "
            "(expect ~100)",
            f"    range[other DFT bin]  : "
            f"{100.0 * self.range_control_nonresonant:7.4f} %  "
            "(expect ~0)",
            f"  threshold              : " f"{100.0 * self.threshold:.2f} %",
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
    r"""Run the finite P50 Fourier diagnostic and emit a certificate.

    Builds the canonical P12 / P14 prime-ladder spectrum, evaluates
    :math:`S_{\mathrm{TNFR}}(T)` on a uniform :math:`T` grid of
    length :math:`n_{\text{periods}} \cdot \operatorname{lcm}
    (\tau_l, \tau_g)`, splits it with the common fixed-delay DFT projector,
    and reports range/complement norms plus two numerical controls. The LCM
    sets backward-compatible sample alignment; the fixed modes are determined
    by :math:`\gcd(\tau_l,\tau_g)`.

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
        \tau_g)`. Larger values change spectral leakage; no universal
        convergence rate is implied.
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
    signal = np.asarray(prime_ladder_oscillatory_sum(t_grid, spectrum), dtype=float)

    range_part, kernel_part = split_residue_by_remesh_infinity(
        signal, tau_l=tau_l, tau_g=tau_g
    )

    norm_total = float(np.linalg.norm(signal))
    norm_range = float(np.linalg.norm(range_part))
    norm_kernel = float(np.linalg.norm(kernel_part))
    if norm_total <= 0.0:
        raise RuntimeError(
            "S_TNFR vanished on the diagnostic window; refusing to " "normalise"
        )
    ratio_range = (norm_range / norm_total) ** 2
    ratio_kernel = (norm_kernel / norm_total) ** 2

    # Sanity controls for the declared finite DFT projection.
    fixed_period = math.gcd(int(tau_l), int(tau_g))
    sample_index = np.arange(n_samples, dtype=float)
    omega_resonant = 2.0 * math.pi / fixed_period
    control_resonant = np.cos(omega_resonant * sample_index)
    rng_res, _ = split_residue_by_remesh_infinity(
        control_resonant, tau_l=tau_l, tau_g=tau_g
    )
    n_res = float(np.linalg.norm(control_resonant))
    ctrl_res_frac = (float(np.linalg.norm(rng_res)) / n_res) ** 2

    mask = build_resonant_bin_mask(n_samples, tau_l=tau_l, tau_g=tau_g)
    unselected = np.flatnonzero(~mask)
    if unselected.size:
        k_nonres = int(unselected[0])
        control_nonres = np.cos(
            2.0 * math.pi * k_nonres * sample_index / n_samples
        )
        rng_nonres, _ = split_residue_by_remesh_infinity(
            control_nonres, tau_l=tau_l, tau_g=tau_g
        )
        n_nonres = float(np.linalg.norm(control_nonres))
        ctrl_nonres_frac = (float(np.linalg.norm(rng_nonres)) / n_nonres) ** 2
    else:
        # Degenerate one-dimensional/all-selected sample space: there is no
        # non-range control vector. Keep the compatibility field finite.
        ctrl_nonres_frac = 0.0

    if ratio_range < threshold and ratio_kernel >= threshold:
        verdict = "RESIDUE_IN_KER_ONLY"
        notes = (
            "Finite-window evidence that the P31 signal has little energy "
            "in the selected fixed-delay periodic bins. This does not "
            "identify a runtime REMESH kernel or advance G4 = RH."
        )
    elif ratio_kernel < threshold and ratio_range >= threshold:
        verdict = "RESIDUE_IN_RANGE_ONLY"
        notes = (
            "The finite-window P31 signal lies mostly in the selected "
            "fixed-delay periodic bins. No conclusion about the T-HP "
            "smooth/oscillatory split or G4 = RH follows."
        )
    else:
        verdict = "RESIDUE_MIXED"
        notes = (
            "Both finite-window fractions exceed the selected threshold. "
            "Report the window and truncation before comparing runs; no "
            "conclusion about G4 = RH follows."
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
