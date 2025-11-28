"""TNFR Life Module: Autopoiesis, Metabolic Resonance, and Reproductive Recursivity

This module provides computational tools to detect and quantify life-like behavior in TNFR networks
based on the mathematical derivation in docs/LIFE_MATHEMATICAL_DERIVATION.md.

Contracts and Invariants (TNFR):
- No direct EPI mutation; always observe via metrics (Invariant #1)
- Structural units preserved (νf in Hz_str) (Invariant #2)
- ΔNFR semantics preserved as structural pressure (Invariant #3)
- Operator closure: this module only measures, does not alter operator sequences (Invariant #4)
- Phase verification upheld in coupling metrics (U3) (Invariant #5)

Metrics:
- Vitality Index (Vi)
- Autopoietic Coefficient (A)
- Self-Organization Index (S)
- Stability Margin (M)

See also:
- docs/LIFE_EMERGENCE_THEORETICAL_FRAMEWORK.md
- docs/LIFE_MATHEMATICAL_DERIVATION.md
"""
from __future__ import annotations
from dataclasses import dataclass

from typing import Sequence, Optional
import numpy as np

# Public API dataclasses


@dataclass
class LifeTelemetry:
    """Container for life-emergence telemetry time series.

    Attributes
    ----------
    times: Sequence[float]
        Structural time stamps.
    vitality_index: np.ndarray
        Vi(t) in [0, 1].
    autopoietic_coefficient: np.ndarray
        A(t) dimensionless.
    self_org_index: np.ndarray
        S(t) dimensionless.
    stability_margin: np.ndarray
        M(t) in [-0.5, 0.5] (per derivation).
    life_threshold_time: Optional[float]
        First time t where A(t) > 1, else None.
    """
    times: Sequence[float]
    vitality_index: np.ndarray
    autopoietic_coefficient: np.ndarray
    self_org_index: np.ndarray
    stability_margin: np.ndarray
    life_threshold_time: Optional[float]


# Core computations

def _safe_div(a: np.ndarray, b: np.ndarray | float, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)


def compute_self_generation(epi_series: np.ndarray, gamma: float, epi_max: float) -> np.ndarray:
    """Compute G(EPI) per canonical logistic form G = γ‖EPI‖(1 - ‖EPI‖/EPI_max).

    Parameters
    ----------
    epi_series: np.ndarray
        Time series of ‖EPI‖ (non-negative). Shape (T,).
    gamma: float
        Autopoietic strength γ [units: ΔNFR/‖EPI‖].
    epi_max: float
        Carrying capacity EPI_max [units: ‖EPI‖].

    Returns
    -------
    np.ndarray
        G(EPI)(t) with same shape as epi_series.
    """
    epi = np.clip(np.asarray(epi_series, dtype=float), 0.0, np.inf)
    return gamma * epi * (1.0 - _safe_div(epi, epi_max))


def compute_autopoietic_coefficient(
    G_epi: np.ndarray,
    dEPI_dt: np.ndarray,
    dnfr_external: np.ndarray,
) -> np.ndarray:
    """Compute autopoietic coefficient A = <G(EPI)·∂EPI/∂t> / <|ΔNFR_ext|^2> (instantaneous form).

    Instantaneous estimator uses moving ratio per time step; for robust estimates,
    apply smoothing/averaging upstream.
    """
    numerator = G_epi * dEPI_dt
    denominator = np.square(np.abs(dnfr_external))
    return _safe_div(numerator, denominator)


def compute_self_org_index(
    epi_series: np.ndarray,
    epsilon: float,
    gamma: float,
    epi_max: float,
    d_dnfr_external_dt: np.ndarray,
    delta: float = 1e-9,
) -> np.ndarray:
    """Compute S = ε·|∂G/∂‖EPI‖| / (|∂ΔNFR_ext/∂t| + δ).
    """
    epi = np.clip(np.asarray(epi_series, dtype=float), 0.0, np.inf)
    dG_dEPI = gamma * (1.0 - 2.0 * _safe_div(epi, epi_max))
    return _safe_div(epsilon * np.abs(dG_dEPI), np.abs(d_dnfr_external_dt) + delta)


def compute_stability_margin(epi_series: np.ndarray, epi_max: float) -> np.ndarray:
    """Compute M = (‖EPI‖ - EPI_max/2)/EPI_max.
    """
    epi = np.asarray(epi_series, dtype=float)
    return (epi - 0.5 * epi_max) / epi_max


def detect_life_emergence(
    times: Sequence[float],
    epi_series: np.ndarray,
    dEPI_dt: np.ndarray,
    dnfr_external: np.ndarray,
    d_dnfr_external_dt: np.ndarray,
    epsilon: float,
    gamma: float,
    epi_max: float,
) -> LifeTelemetry:
    """Detect life emergence per TNFR derivation.

    Parameters
    ----------
    times: Sequence[float]
        Structural times (monotonic).
    epi_series: np.ndarray
        Series of ‖EPI‖ (≥ 0). Shape (T,).
    dEPI_dt: np.ndarray
        Time derivative of ‖EPI‖. Shape (T,).
    dnfr_external: np.ndarray
        External ΔNFR(t). Shape (T,).
    d_dnfr_external_dt: np.ndarray
        Time derivative of external ΔNFR. Shape (T,).
    epsilon: float
        Self-feedback strength ε ∈ [0, 1].
    gamma: float
        Autopoietic strength γ [ΔNFR/‖EPI‖].
    epi_max: float
        Carrying capacity EPI_max [‖EPI‖].

    Returns
    -------
    LifeTelemetry
        Telemetry including Vi, A, S, M and threshold time.
    """
    times = list(times)
    epi = np.asarray(epi_series, dtype=float)
    dEPI = np.asarray(dEPI_dt, dtype=float)
    dnfr_ext = np.asarray(dnfr_external, dtype=float)
    d_dnfr_ext_dt = np.asarray(d_dnfr_external_dt, dtype=float)

    G_epi = compute_self_generation(epi, gamma=gamma, epi_max=epi_max)
    A = compute_autopoietic_coefficient(G_epi, dEPI, dnfr_ext)
    S = compute_self_org_index(epi, epsilon, gamma, epi_max, d_dnfr_ext_dt)
    M = compute_stability_margin(epi, epi_max)

    # Vitality Index: Vi = (ΔNFR_internal / ΔNFR_total) × C(t)
    # We don't have C(t) here; provide structural ratio (0..1). Users can multiply by C(t).
    dnfr_internal_est = epsilon * G_epi
    Vi = _safe_div(np.abs(dnfr_internal_est), np.abs(dnfr_internal_est) + np.abs(dnfr_ext))

    # Refined threshold detection: interpolate to find exact crossing at A = 1.0
    life_time: Optional[float] = None
    
    # Check if A ever exceeds 1.0
    if (A > 1.0).any():
        # Find crossings from ≤1 to >1
        crossings = np.where((A[:-1] <= 1.0) & (A[1:] > 1.0))[0]
        if len(crossings) > 0:
            # Linear interpolation between first crossing points
            i = crossings[0]
            t0, t1 = times[i], times[i + 1]
            A0, A1 = A[i], A[i + 1]
            # Solve: A0 + (A1 - A0) * α = 1.0 for α
            if A1 != A0:  # Avoid division by zero
                alpha = (1.0 - A0) / (A1 - A0)
                life_time = t0 + alpha * (t1 - t0)
            else:
                life_time = t0  # Fallback if no gradient
        else:
            # All A > 1.0 from start, use first time point
            life_time = times[0]

    return LifeTelemetry(
        times=times,
        vitality_index=Vi,
        autopoietic_coefficient=A,
        self_org_index=S,
        stability_margin=M,
        life_threshold_time=life_time,
    )


__all__ = [
    "LifeTelemetry",
    "compute_self_generation",
    "compute_autopoietic_coefficient",
    "compute_self_org_index",
    "compute_stability_margin",
    "detect_life_emergence",
]
