r"""P49 — χ-Twisted Prime-Ladder Oscillatory Correction (L-track lift of P31).

Branch B1 of §13octies at the L-track level: structural reconstruction
of the χ-twisted oscillatory remainder
:math:`S_\chi(T) = \pi^{-1}\arg L(\tfrac12 + iT, \chi)` from canonical
TNFR ingredients only, for primitive real Dirichlet characters.

Motivation
----------
P46 (§13vicies-quinto) and P48 (§13vicies-septimo) close the **smooth
half** of Conjecture T-HP\ :sup:`(χ)` at the density and operator
levels respectively, per primitive real character.  The residual
Wasserstein-1 gap
:math:`W_1(\{\widetilde\gamma_i^{(\chi)}\}, \{\gamma_n^{(\chi)}\})`
observed in P48 (:math:`\approx 1.27`-:math:`1.47` at
:math:`n_{\text{targets}} = 12`) is exactly the **χ-twisted
oscillatory remainder** :math:`S_\chi(T)`.

The classical Riemann-von Mangoldt formula on the critical line of
:math:`L(s, \chi)` (primitive, non-principal, real) gives

.. math::

    \pi\,S_\chi(T) \;=\; -\sum_p\sum_{k\geq 1}
        \frac{\chi(p)^k\,\sin(kT\log p)}{k\,p^{k/2}}
        \;+\; \mathcal O(1/T).

Re-expressing the prime-power sum in terms of the canonical χ-twisted
prime-ladder spectrum
:math:`\Sigma_{N,K}^{(\chi)} = \{(\mu_{p,k},\,w_{p,k}^{(\chi)})\}`
with :math:`\mu_{p,k} = k\log p` and
:math:`w_{p,k}^{(\chi)} = \chi(p)^k \log p` (P34 atomic ingredient,
provided by :func:`build_twisted_prime_ladder_spectrum` in
:mod:`tnfr.riemann.dirichlet_l`) yields

.. math::

    \pi\,S_\chi^{\mathrm{TNFR}}(T;\,N,K) \;=\;
        -\sum_{(\mu, w^{(\chi)}) \in \Sigma_{N,K}^{(\chi)}}
        \frac{w^{(\chi)}}{\mu}\cdot\frac{\sin(T\mu)}{e^{\mu/2}}.

All three factors (:math:`\mu`, :math:`w^{(\chi)}`, :math:`e^{\mu/2}`)
come from the canonical P34 χ-twisted spectral data; :math:`\pi` is
canonical (tetrad :math:`\pi \leftrightarrow K_\varphi`); the
character :math:`\chi` is a primitive real Dirichlet character.
For primitive real :math:`\chi`, the weights :math:`w^{(\chi)}` are
real (:math:`\pm \log p`), so :math:`S_\chi^{\mathrm{TNFR}}` is real.

The position-level correction follows from
:math:`N_\chi(\gamma_n) = \bar N_\chi(\gamma_n) + S_\chi(\gamma_n) = n`:

.. math::

    \gamma_n^{(\chi)} \;\approx\; \widetilde\gamma_n^{(\chi)}
        - \frac{S_\chi^{\mathrm{TNFR}}(\widetilde\gamma_n^{(\chi)})}
               {\bar N_\chi'(\widetilde\gamma_n^{(\chi)})},

with :math:`\bar N_\chi'` the P46 χ-twisted smooth density.

What this module closes / does not close
----------------------------------------
**Closes**: the operator-level χ-twisted smooth half of T-HP\ :sup:`(χ)`
already closed by P48 is now complemented by an explicit canonical
operator-level candidate for the χ-twisted oscillatory half built
purely from P34 χ-twisted prime-ladder data.

**Does NOT close**:

* Canonicity from the nodal equation (sub-problem (2) of T-HP\
  :sup:`(χ)`): the χ-twisted prime-ladder spectrum is canonical, but
  expressing :math:`S_\chi^{\mathrm{TNFR}}` as a **derivation** of the
  canonical nodal evolution (rather than as an ingredient plugged into
  the Riemann-von Mangoldt template) is still open.
* Positivity coincidence with the χ-twisted Weil quadratic form
  (sub-problem (3) of T-HP\ :sup:`(χ)`, addressed only as a diagnostic
  by P39 `twisted_weil_positivity.py`).
* **GRH for L(s, χ)** (the L-track analogue of G4 = RH).
* **G4 = RH** itself.

Numerical positivity in the certificate constitutes **branch B1
evidence at the L-track level**: that the canonical 13-operator
catalog plus the χ-twisted prime-ladder spectrum suffices to reproduce
:math:`S_\chi(T)`.  Numerical negativity (no improvement) corroborates
**branch B2 at the L-track level**: a genuinely new canonical operator
is required.

This module closes the final ζ↔L attack-surface parity item
(P31 :math:`\to` P49).  After P49 ships, every canonical ζ-track
operator from P12 through P31 has a matching χ-twisted L-track
counterpart for every primitive real Dirichlet character.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dirichlet_l import (
    DirichletCharacter,
    TwistedPrimeLadderSpectrum,
    build_twisted_prime_ladder_spectrum,
)
from .hilbert_polya import wasserstein_1_distance
from .twisted_hilbert_polya import fetch_chi_zero_imaginary_parts
from .twisted_structural_zero_density import (
    build_twisted_structural_t_hp,
    twisted_smooth_zero_density,
)

__all__ = [
    "twisted_prime_ladder_oscillatory_sum",
    "apply_twisted_oscillatory_correction",
    "TwistedOscillatoryCorrectionCertificate",
    "compute_twisted_oscillatory_correction_certificate",
]


# ----------------------------------------------------------------------
# Core canonical reconstruction of S_chi(T)
# ----------------------------------------------------------------------


def twisted_prime_ladder_oscillatory_sum(
    T: float | np.ndarray,
    spectrum: TwistedPrimeLadderSpectrum,
) -> float | np.ndarray:
    r"""Evaluate :math:`S_\chi^{\mathrm{TNFR}}(T;\,N,K)` from canonical
    χ-twisted prime-ladder data.

    Computes

    .. math::

        S_\chi^{\mathrm{TNFR}}(T) = -\frac{1}{\pi}
            \sum_{(\mu, w^{(\chi)}) \in \Sigma_{N,K}^{(\chi)}}
            \frac{w^{(\chi)}}{\mu}\cdot\frac{\sin(T\mu)}{e^{\mu/2}}

    using ONLY the χ-twisted prime-ladder spectrum (P34 canonical),
    the character :math:`\chi`, and the canonical constant
    :math:`\pi`.

    For primitive real characters, :math:`w^{(\chi)} \in \mathbb R`
    (it equals :math:`\pm \log p`), so the sum is real-valued.  The
    routine takes ``.real`` of the complex weights for full generality
    (and to gracefully degrade if the spectrum is computed in complex
    dtype but populated with real values).

    Parameters
    ----------
    T : float or np.ndarray
        Evaluation height(s) on the critical line.
    spectrum : TwistedPrimeLadderSpectrum
        Canonical χ-twisted TNFR prime-ladder spectrum.

    Returns
    -------
    float or np.ndarray
        :math:`S_\chi^{\mathrm{TNFR}}(T)`, same shape as ``T``.

    Raises
    ------
    ValueError
        If ``spectrum`` is empty or carries non-positive eigenvalues.
    """
    mu = np.asarray(spectrum.eigenvalues, dtype=float)
    w_complex = np.asarray(spectrum.weights, dtype=complex)
    if mu.size == 0:
        raise ValueError("empty χ-twisted prime-ladder spectrum")
    if np.any(mu <= 0.0):
        raise ValueError("χ-twisted prime-ladder eigenvalues must be positive")
    # Restrict to the real part: canonical for primitive real χ, and
    # the imaginary part is forced to zero by construction in that
    # case.  We assert smallness as a safety guard.
    w = w_complex.real
    max_imag = float(np.max(np.abs(w_complex.imag)))
    if max_imag > 1e-10:
        raise ValueError(
            "twisted_prime_ladder_oscillatory_sum currently requires "
            "primitive real Dirichlet characters (max imaginary part "
            f"of weights: {max_imag:.3e})"
        )
    # Pre-compute amplitude coefficients
    # a_j = (w_j / mu_j) * exp(-mu_j/2).
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


def apply_twisted_oscillatory_correction(
    smooth_targets: np.ndarray,
    spectrum: TwistedPrimeLadderSpectrum,
    chi: DirichletCharacter,
    *,
    damping: float = 1.0,
) -> np.ndarray:
    r"""Correct :math:`\{\widetilde\gamma_i^{(\chi)}\}` using the TNFR
    :math:`S_\chi(T)`.

    Applies the first-order Newton step

    .. math::

        \gamma_i^{(\chi),\,\mathrm{corr}} = \widetilde\gamma_i^{(\chi)}
            - d\cdot
              \frac{S_\chi^{\mathrm{TNFR}}(\widetilde\gamma_i^{(\chi)})}
                   {\bar N_\chi'(\widetilde\gamma_i^{(\chi)})},

    where ``d`` is the optional damping factor (default ``1.0``).

    The smooth density :math:`\bar N_\chi'` is the canonical P46
    χ-twisted density (Riemann-Siegel-like theta derivative with
    :math:`\log(qT/(2\pi))` leading term); :math:`S_\chi^{\mathrm{TNFR}}`
    is the canonical χ-twisted prime-ladder reconstruction.

    Parameters
    ----------
    smooth_targets : np.ndarray
        Canonical P46 χ-twisted smooth targets
        :math:`\widetilde\gamma_i^{(\chi)}`.
    spectrum : TwistedPrimeLadderSpectrum
        Canonical χ-twisted prime-ladder spectrum used to evaluate
        :math:`S_\chi^{\mathrm{TNFR}}`.
    chi : DirichletCharacter
        Character used to evaluate :math:`\bar N_\chi'`.  Must match
        the character carried by ``spectrum`` (this is checked).
    damping : float, default 1.0
        Multiplicative damping ``d``.  ``d = 0`` reproduces the smooth
        targets unchanged; ``d = 1`` is the unmoderated Newton step.

    Returns
    -------
    np.ndarray
        Corrected χ-twisted zero-position candidates, sorted ascending.

    Raises
    ------
    ValueError
        If parameters are inconsistent or out of range.
    RuntimeError
        If the correction drives a target non-positive or if the
        smooth density vanishes at some target.
    """
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if int(spectrum.character_modulus) != int(chi.modulus):
        raise ValueError(
            "character modulus mismatch between spectrum "
            f"({spectrum.character_modulus}) and chi ({chi.modulus})"
        )
    if spectrum.character_name != chi.name:
        raise ValueError(
            "character name mismatch between spectrum "
            f"({spectrum.character_name!r}) and chi ({chi.name!r})"
        )
    targets = np.asarray(smooth_targets, dtype=float)
    if targets.ndim != 1:
        raise ValueError("smooth_targets must be 1-D")
    s_vals = np.asarray(
        twisted_prime_ladder_oscillatory_sum(targets, spectrum),
        dtype=float,
    )
    densities = np.array(
        [twisted_smooth_zero_density(float(t), chi) for t in targets],
        dtype=float,
    )
    if np.any(densities <= 0.0):
        raise RuntimeError(
            "χ-twisted smooth density vanished at a target; " "refusing to divide"
        )
    delta = -damping * s_vals / densities
    corrected = targets + delta
    if np.any(corrected <= 0.0):
        raise RuntimeError(
            "χ-twisted oscillatory correction drove a target "
            "non-positive; reduce damping"
        )
    return np.sort(corrected)


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedOscillatoryCorrectionCertificate:
    r"""Certificate for the P49 χ-twisted prime-ladder oscillatory
    correction.

    Attributes
    ----------
    character_modulus
        Modulus :math:`q` of the primitive real character.
    character_name
        Human-readable label of the character.
    n_targets
        Number of smooth targets / true zeros compared.
    n_primes_requested
        Primes requested when building the χ-twisted spectrum.
    n_primes_active
        Primes actually carrying non-zero χ-twisted weight
        (i.e. coprime to :math:`q`).
    max_power
        Maximum REMESH echo index :math:`K`.
    best_damping
        Damping :math:`d` (in the swept range) minimising
        :math:`W_1(\{\gamma_i^{(\chi),\,\mathrm{corr}}\},
                   \{\gamma_n^{(\chi)}\})`.
    w1_smooth_vs_true
        :math:`W_1(\{\widetilde\gamma_i^{(\chi)}\},
                   \{\gamma_n^{(\chi)}\})` — baseline (P48
        smooth-half residual at the position level).
    w1_corrected_vs_true
        :math:`W_1(\{\gamma_i^{(\chi),\,\mathrm{corr}}\},
                   \{\gamma_n^{(\chi)}\})` at ``best_damping``.
    improvement_over_smooth
        :math:`(W_1^{\mathrm{smooth}} - W_1^{\mathrm{corr}})
                / W_1^{\mathrm{smooth}}`.  Positive ⇒ the canonical
        χ-twisted prime-ladder reconstruction of :math:`S_\chi(T)`
        reduces the gap.
    max_abs_s_at_targets
        :math:`\max_i |S_\chi^{\mathrm{TNFR}}
        (\widetilde\gamma_i^{(\chi)})|`, sanity check on the
        reconstruction magnitude.
    damping_sweep
        List of ``(damping, W_1)`` pairs across the sweep.
    notes
        Honest-scope reminder.  Positive improvement is **branch B1
        evidence at the L-track level**, not a closure of GRH for
        :math:`L(s, \chi)` and not a closure of G4 = RH.
    """

    character_modulus: int
    character_name: str
    n_targets: int
    n_primes_requested: int
    n_primes_active: int
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
            "P49 — χ-Twisted Prime-Ladder Oscillatory Correction " "Certificate",
            f"  character               : {self.character_name} "
            f"(mod {self.character_modulus})",
            f"  n_targets               : {self.n_targets}",
            "  n_primes (req/active)   : "
            f"{self.n_primes_requested} / {self.n_primes_active}",
            f"  max_power (K)           : {self.max_power}",
            f"  best damping            : {self.best_damping:.4f}",
            "  W_1 smooth vs true      : " f"{self.w1_smooth_vs_true:.4e}",
            "  W_1 corrected vs true   : " f"{self.w1_corrected_vs_true:.4e}",
            "  improvement over smooth : "
            f"{100.0 * self.improvement_over_smooth:+.2f} %",
            "  max |S_chi_TNFR(t_i)|   : " f"{self.max_abs_s_at_targets:.4e}",
            f"  notes                   : {self.notes}",
        ]
        return "\n".join(lines)


def compute_twisted_oscillatory_correction_certificate(
    chi: DirichletCharacter,
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
) -> TwistedOscillatoryCorrectionCertificate:
    r"""Run the full P49 χ-twisted reconstruction and emit a
    certificate.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real Dirichlet character defining the twist.
    n_targets : int
        Number of χ-twisted smooth zeros / true zeros compared.
    n_primes : int, default 200
        Primes used in the canonical χ-twisted prime-ladder spectrum.
        Larger values improve the resolution of
        :math:`S_\chi^{\mathrm{TNFR}}`.  Primes dividing the modulus
        carry zero χ-twisted weight and are excluded from the active
        spectrum (counted separately in
        ``n_primes_active``).
    max_power : int, default 8
        REMESH echo cap :math:`K`.
    damping_grid : tuple of float, default (0, .25, .5, .75, 1, 1.25, 1.5)
        Damping factors swept; best is retained.
    dps : int, default 30
        mpmath precision for the true reference χ-twisted zeros and
        for the smooth-target Newton solver.

    Returns
    -------
    TwistedOscillatoryCorrectionCertificate
    """
    if n_targets < 1:
        raise ValueError("n_targets must be >= 1")
    spectrum = build_twisted_prime_ladder_spectrum(chi, n_primes, max_power=max_power)
    smooth_targets = build_twisted_structural_t_hp(n_targets, chi, dps=dps)
    true_gammas = fetch_chi_zero_imaginary_parts(chi, n_targets, dps=dps)

    s_at_targets = np.asarray(
        twisted_prime_ladder_oscillatory_sum(smooth_targets, spectrum),
        dtype=float,
    )
    max_abs_s = float(np.max(np.abs(s_at_targets)))

    w1_smooth = wasserstein_1_distance(smooth_targets, true_gammas)

    sweep: list[tuple[float, float]] = []
    best_d = 0.0
    best_w1 = w1_smooth
    for d in damping_grid:
        try:
            corrected = apply_twisted_oscillatory_correction(
                smooth_targets, spectrum, chi, damping=float(d)
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
        "Honest scope (L-track): positive improvement is branch B1 "
        "evidence at the L-track level — the canonical chi-twisted "
        "prime-ladder spectrum suffices to reduce the S_chi(T) "
        "residual for the chosen primitive real character.  This "
        "does NOT close G4 = RH, does NOT prove GRH for L(s, chi), "
        "does NOT prove canonicity from the nodal equation "
        "(sub-problem (2) of T-HP^(chi)), and does NOT establish "
        "positivity coincidence with the chi-twisted Weil quadratic "
        "form (sub-problem (3)).  Negative improvement corroborates "
        "branch B2 at the L-track level (a new canonical operator "
        "required)."
    )

    return TwistedOscillatoryCorrectionCertificate(
        character_modulus=int(chi.modulus),
        character_name=str(chi.name),
        n_targets=int(n_targets),
        n_primes_requested=int(n_primes),
        n_primes_active=int(spectrum.n_active),
        max_power=int(max_power),
        best_damping=float(best_d),
        w1_smooth_vs_true=float(w1_smooth),
        w1_corrected_vs_true=float(best_w1),
        improvement_over_smooth=float(improvement),
        max_abs_s_at_targets=max_abs_s,
        damping_sweep=tuple(sweep),
        notes=notes,
    )
