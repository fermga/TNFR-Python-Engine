"""νf-Type Signature — Diagnostic for the νf-Type Conjecture (§13triginta-prima).

This module implements a purely diagnostic quantity, the **νf-Type
Signature** :math:`\\mathcal{S}_{\\nu_f}`, that quantifies on canonical
TNFR-Riemann data (the P14 prime-ladder spectrum) the irreducible
measure-valued content of :math:`\\nu_f`.

Scope (mandatory honesty)
-------------------------
This module is a *diagnostic only*.  It does **not** construct, promote,
or modify any canonical operator.  It does **not** advance G4 = RH.
It does **not** by itself decide the νf-Type Conjecture (which requires
the foundational analysis of §13triginta-prima.3–.5 about the
Conjugate-Pair-via-Pontryagin principle).

A high signature is a *necessary-condition* check: it says only that
the canonical P14 data carries irreducible measure-valued structure
that a scalar :math:`\\nu_f` cannot represent without loss.  It does
**not** prove that the canonical type of :math:`\\nu_f` is a measure.

A low signature would *falsify* the practical relevance of the
promotion on the P14 data.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13triginta-prima
- ``src/tnfr/riemann/prime_ladder_hamiltonian.py`` (P14)
- ``src/tnfr/riemann/von_mangoldt.py`` (P12, ``PrimeLadderSpectrum``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .von_mangoldt import PrimeLadderSpectrum, build_prime_ladder_spectrum

__all__ = [
    "NufTypeSignatureCertificate",
    "compute_nuf_type_signature",
]


def _shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats of a probability vector.

    Zero-probability entries are skipped (``0 · log 0 := 0``).
    """
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _binned_distribution(
    values: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Normalised binned weight distribution on ``n_bins`` uniform bins."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0 or float(np.sum(w)) <= 0.0:
        return np.zeros(n_bins, dtype=float)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi <= lo:
        # Degenerate: all mass at one point.
        p = np.zeros(n_bins, dtype=float)
        p[0] = 1.0
        return p
    counts, _ = np.histogram(v, bins=n_bins, range=(lo, hi), weights=w)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(n_bins, dtype=float)
    return counts / total


@dataclass(frozen=True)
class NufTypeSignatureCertificate:
    """Result of the νf-Type Signature diagnostic on a P14-style spectrum.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{\\nu_f} = H(\\mu_{\\mathrm{spec}}) / \\log B`
        in :math:`[0, 1]`.  ``0`` means scalar-adequate (single-bin mass);
        ``1`` means maximum non-scalar (uniform) content.
    n_modes : int
        Number of distinct eigenvalues in the spectrum.
    effective_modes : float
        :math:`N_{\\mathrm{eff}} = \\exp(H)` — effective number of
        populated bins.  Scalar-adequate iff :math:`N_{\\mathrm{eff}} \\approx 1`.
    spectral_entropy_nats : float
        Raw binned entropy :math:`H(\\mu_{\\mathrm{spec}})` in nats.
    n_bins : int
        Number of histogram bins used.
    spectral_spread : float
        Coefficient of variation :math:`\\sigma / |\\mu|` of the
        weighted spectrum (dimensionless dispersion).
    verdict : str
        One of ``"SCALAR_ADEQUATE"`` (signature < 0.15),
        ``"MEASURE_VALUED_NECESSARY"`` (signature > 0.5),
        or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (n_primes, max_power, total_weight, etc.).
    """

    signature: float
    n_modes: int
    effective_modes: float
    spectral_entropy_nats: float
    n_bins: int
    spectral_spread: float
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "νf-Type Signature certificate (diagnostic only — §13triginta-prima.6)",
            f"  signature S_νf       : {self.signature:.6f}   (0 = scalar, 1 = uniform)",
            f"  effective modes N_eff: {self.effective_modes:.2f} / {self.n_modes} distinct",
            f"  spectral entropy     : {self.spectral_entropy_nats:.4f} nats"
            f" over {self.n_bins} bins",
            f"  spectral spread CV   : {self.spectral_spread:.4f}",
            f"  verdict              : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
        ]
        return "\n".join(lines)


def compute_nuf_type_signature(
    spectrum: PrimeLadderSpectrum | None = None,
    *,
    n_primes: int = 200,
    max_power: int = 8,
    n_bins: int = 64,
    scalar_threshold: float = 0.15,
    measure_threshold: float = 0.5,
) -> NufTypeSignatureCertificate:
    """Compute the νf-Type Signature on a canonical P14 prime-ladder spectrum.

    Parameters
    ----------
    spectrum : PrimeLadderSpectrum, optional
        Pre-built spectrum.  If ``None``, a fresh one is built from
        ``n_primes`` and ``max_power``.
    n_primes : int, default 200
        Number of primes when ``spectrum`` is ``None``.
    max_power : int, default 8
        REMESH echo cap when ``spectrum`` is ``None``.
    n_bins : int, default 64
        Histogram resolution :math:`B`.  The maximum entropy is
        :math:`\\log B`; the signature is normalised by this maximum
        so that :math:`\\mathcal{S}_{\\nu_f} \\in [0, 1]`.
    scalar_threshold : float, default 0.15
        Below this value the verdict is ``"SCALAR_ADEQUATE"``.
    measure_threshold : float, default 0.5
        Above this value the verdict is ``"MEASURE_VALUED_NECESSARY"``.

    Returns
    -------
    NufTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    The diagnostic uses Shannon entropy of the binned weighted spectral
    measure :math:`\\mu_{\\mathrm{spec}} = \\sum_n w_n \\delta_{\\lambda_n}`.
    The normalisation by :math:`\\log B` makes the signature
    bin-resolution-comparable across different :math:`n_{\\mathrm{primes}}`
    and :math:`\\mathrm{max\\_power}`.

    This is a *purely diagnostic* computation on already-canonical data.
    It does not construct any new operator and does not modify the
    13-operator catalog.
    """
    if spectrum is None:
        spectrum = build_prime_ladder_spectrum(
            n_primes=n_primes,
            max_power=max_power,
        )
    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=float)
    weights = np.asarray(spectrum.weights, dtype=float)
    if eigenvalues.size == 0:
        raise ValueError("PrimeLadderSpectrum is empty")
    if int(n_bins) <= 1:
        raise ValueError("n_bins must be >= 2")

    # Binned probability distribution.
    p = _binned_distribution(eigenvalues, weights, int(n_bins))
    entropy_nats = _shannon_entropy(p)
    max_entropy = math.log(float(n_bins))
    signature = float(entropy_nats / max_entropy) if max_entropy > 0.0 else 0.0
    signature = max(0.0, min(1.0, signature))

    # Effective number of populated bins.
    effective_modes = float(math.exp(entropy_nats))

    # Coefficient of variation (weighted).
    total_w = float(np.sum(weights))
    if total_w > 0.0:
        mean_w = float(np.sum(weights * eigenvalues) / total_w)
        var_w = float(np.sum(weights * (eigenvalues - mean_w) ** 2) / total_w)
        std_w = math.sqrt(max(var_w, 0.0))
        spread = float(std_w / abs(mean_w)) if abs(mean_w) > 0.0 else 0.0
    else:
        mean_w = 0.0
        spread = 0.0

    # Verdict.
    if signature < scalar_threshold:
        verdict = "SCALAR_ADEQUATE"
    elif signature > measure_threshold:
        verdict = "MEASURE_VALUED_NECESSARY"
    else:
        verdict = "INDETERMINATE"

    n_modes = int(np.unique(eigenvalues).size)

    diagnostics: dict[str, Any] = {
        "n_primes": int(spectrum.n_primes),
        "max_power": int(spectrum.max_power),
        "total_weight": total_w,
        "weighted_mean_eigenvalue": mean_w,
        "scalar_threshold": float(scalar_threshold),
        "measure_threshold": float(measure_threshold),
        "scope": (
            "Necessary-condition diagnostic for νf-Type Conjecture "
            "(§13triginta-prima). Does NOT advance G4 = RH."
        ),
    }

    return NufTypeSignatureCertificate(
        signature=signature,
        n_modes=n_modes,
        effective_modes=effective_modes,
        spectral_entropy_nats=entropy_nats,
        n_bins=int(n_bins),
        spectral_spread=spread,
        verdict=verdict,
        diagnostics=diagnostics,
    )
