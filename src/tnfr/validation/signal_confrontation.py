"""Data-agnostic confrontation of a real signal with canonical TNFR magnitudes.

The empirical arm as a first-class engine capability: point :func:`confront_signal`
at **any** real multichannel signal (EEG, grid telemetry, coupled oscillators) and
read the canonical TNFR magnitudes -- the emergent phase-locking graph, the
structural tetrad (``|∇φ|``, ``K_φ``, ``Φ_s``, ``ξ_C``), the pulse
(``ω_k = √λ_k``), the coherence ``C`` (the universal attractor kernel) and the
Kuramoto order ``R`` -- plus the two-face diagnosis (the diffusive/over-damped
certificate).  Bring your own signal; **no bundled data, no ML dependencies**.

This composes the existing engine pipeline
(:mod:`tnfr.validation.multichannel_interface` for the phase-locking graph and
tetrad, :mod:`tnfr.physics.structural_diffusion` for the pulse and the face
certificate, :func:`tnfr.metrics.common.structural_coherence` for the universal
coherence kernel) into a single confrontation entry point.

Honest scope: this is the falsifiable *instrument*, not a competitive model.  On
real EEG the canonical read-outs carry genuine state (the local tetrad
discriminates seizure cross-patient; the single-``νf`` wave ties strong
baselines) but do **not** out-predict standard methods -- see
``docs/EMPIRICAL_CONFRONTATION_EEG.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metrics.common import is_structural_equilibrium, structural_coherence
from ..physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from ..physics.structural_diffusion import (
    compute_emergent_pulse,
    structural_diffusion_operator,
    symmetric_normalized_laplacian,
)
from .multichannel_interface import (
    build_coupling_graph,
    kuramoto_order_parameter,
    phase_amplitude_matrices,
)

__all__ = [
    "SignalConfrontation",
    "confront_signal",
    "estimate_quality_factor",
    "emergent_wave_fraction",
    "NodalPredictionSkill",
    "nodal_prediction_skill",
]


def _mean_abs(values: Any) -> float:
    vals = list(values.values())
    return float(np.mean([abs(v) for v in vals])) if vals else float("nan")


def estimate_quality_factor(signals: Any) -> float:
    """Dimensionless quality factor ``Q`` of a multichannel signal (fs-free).

    ``Q = k_peak / FWHM`` of the dominant spectral peak, measured in FFT-bin
    units so it needs **no** sampling rate.  For the damped substrate wave
    ``q̈ + γq̇ + Lq = 0`` a single mode has ``Q = ω₀/γ`` and ``Q = 1/2`` is
    critical damping.  This is a **secondary spectral read-out** only; the
    two-face verdict uses :func:`emergent_wave_fraction` (read from the emergent
    modal dynamics, robust for diffusive fields where this input-spectrum Q
    floors near 1/2).  Aggregated as the median across channels.
    """
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2:
        raise ValueError("signals must be (n_channels, n_samples)")
    if data.shape[1] < 8:
        return 0.0
    qs: list[float] = []
    for row in data:
        x = row - float(np.mean(row))
        power = np.abs(np.fft.rfft(x)) ** 2
        if power.size < 3:
            continue
        power[0] = 0.0  # drop DC
        k_peak = int(np.argmax(power))
        peak = float(power[k_peak])
        if k_peak < 1 or peak <= 0.0:
            continue
        half = peak / 2.0
        lo = k_peak
        while lo > 0 and power[lo] >= half:
            lo -= 1
        hi = k_peak
        while hi < power.size - 1 and power[hi] >= half:
            hi += 1
        qs.append(k_peak / max(hi - lo, 1))
    return float(np.median(qs)) if qs else 0.0


def _wave_fraction_from_graph(graph: Any, data: "np.ndarray") -> float:
    """Energy-weighted fraction of emergent modes whose dynamics oscillate.

    Projects the signal onto the emergent ``L_sym`` eigenmodes and reads the
    damping from each modal coordinate's *own* dynamics via an AR(2) fit:
    complex characteristic roots (``φ₁² + 4φ₂ < 0``) mean the mode oscillates
    (under-damped / conservative-wave face); real roots mean it relaxes
    (over-damped / diffusive face).  The trivial λ≈0 mode carries no restoring
    force (ω = √λ = 0) and is excluded.  Damping is read from the **emergent
    dynamics**, not from the raw input spectrum.
    """
    nodes, lsym = symmetric_normalized_laplacian(graph)
    try:
        idx = np.asarray([int(nd) for nd in nodes])
        x = np.asarray(data, dtype=float)[idx]
    except (ValueError, IndexError, TypeError):
        x = np.asarray(data, dtype=float)
    w, vecs = np.linalg.eigh(lsym)
    modal = vecs.T @ x  # (n_modes, n_time)
    num = den = 0.0
    for k in range(len(w)):
        if w[k] < 1e-9:  # trivial uniform mode: no graph oscillation
            continue
        a = modal[k] - float(np.mean(modal[k]))
        energy = float(np.var(a))
        if a.size < 8 or energy < 1e-20:
            continue
        y = a[2:]
        xd = np.column_stack([a[1:-1], a[:-2]])
        phi, *_ = np.linalg.lstsq(xd, y, rcond=None)
        den += energy
        if phi[0] * phi[0] + 4.0 * phi[1] < 0.0:  # complex roots: oscillatory
            num += energy
    return num / den if den > 0.0 else 0.0


def emergent_wave_fraction(signals: Any, *, k_neighbours: int = 4) -> float:
    """Emergent two-face reading of a multichannel signal.

    Builds the emergent phase-locking graph and returns the energy-weighted
    fraction of its eigenmodes whose dynamics oscillate (the conservative-wave
    face); ``> 1/2`` ⇒ wave face, ``<= 1/2`` ⇒ diffusive face.  Unlike the
    input-spectrum quality factor, the damping is read from the **emergent
    modal dynamics**, so a diffusive field is certified on the diffusive face.
    Needs adequate data: the per-mode AR(2) fit is under-powered for very short
    windows / few channels (oscillatory fields converge to the wave face only
    for windows of roughly a thousand samples or more).
    """
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError(
            "signals must be (n_channels, n_samples) with >= 3 channels"
        )
    phase, amp = phase_amplitude_matrices(data)
    graph = build_coupling_graph(phase, amp, k_neighbours=k_neighbours)
    return _wave_fraction_from_graph(graph, data)


@dataclass(frozen=True)
class SignalConfrontation:
    """Canonical TNFR read-out of one multichannel signal (the empirical arm)."""

    n_channels: int
    n_samples: int
    kuramoto_R: float
    grad_phi: float
    k_phi: float
    phi_s: float
    xi_c: float
    coherence: float
    at_equilibrium: bool
    pulse_fundamental: float
    dominant_beat: float
    vibration_energy: float
    quality_factor: float
    wave_fraction: float
    diffusive_face_valid: bool

    def summary(self) -> str:
        """Human-readable one-line canonical read-out."""
        eq = "at ΔNFR=0" if self.at_equilibrium else "off equilibrium"
        face = (
            "DIFFUSIVE/over-damped"
            if self.diffusive_face_valid
            else "WAVE/under-damped"
        )
        return (
            f"SignalConfrontation[{self.n_channels}ch × {self.n_samples}]: "
            f"R={self.kuramoto_R:.3f}, C={self.coherence:.3f} ({eq}); "
            f"tetrad |∇φ|={self.grad_phi:.3f} |K_φ|={self.k_phi:.3f} "
            f"Φ_s={self.phi_s:.3f} ξ_C={self.xi_c:.3f}; pulse "
            f"ω₀={self.pulse_fundamental:.3f} beat={self.dominant_beat:.3f}; "
            f"face={face} (wave_frac={self.wave_fraction:.2f}, "
            f"Q={self.quality_factor:.2f})"
        )


def confront_signal(
    signals: Any, *, k_neighbours: int = 4
) -> SignalConfrontation:
    """Confront a real multichannel signal with the canonical TNFR magnitudes.

    Parameters
    ----------
    signals : array-like, shape ``(n_channels, n_samples)``
        Any real multichannel signal (EEG, telemetry, coupled oscillators).
    k_neighbours : int
        Neighbours per channel in the emergent phase-locking coupling graph.

    Returns
    -------
    SignalConfrontation
        The canonical read-outs plus the two-face diagnosis.
    """
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError(
            "signals must be (n_channels, n_samples) with >= 3 channels"
        )
    n_channels, n_samples = data.shape

    phase, amp = phase_amplitude_matrices(data)
    graph = build_coupling_graph(phase, amp, k_neighbours=k_neighbours)

    r_order = kuramoto_order_parameter(phase)
    grad = _mean_abs(compute_phase_gradient(graph))
    kphi = _mean_abs(compute_phase_curvature(graph))
    phis = _mean_abs(compute_structural_potential(graph))

    dnfr_vals = [abs(float(graph.nodes[n].get("dnfr", 0.0))) for n in graph]
    mean_dnfr = float(np.mean(dnfr_vals)) if dnfr_vals else 0.0
    coherence = structural_coherence(mean_dnfr)
    at_eq = is_structural_equilibrium(mean_dnfr, 0.0, eps_dnfr=1e-2)

    pulse = compute_emergent_pulse(graph)
    omega0 = float(pulse["fundamental"])
    # xi_C from the EMERGENT spectral gap (1/sqrt(lambda_2)) -- the robust
    # emergent-geometry coherence length (the autocorrelation estimator is nan
    # on these coupling graphs; Network.nfr() uses the same spectral-gap form).
    xi_c = (1.0 / omega0) if omega0 > 0.0 else float("inf")

    # Emergent two-face diagnosis.  The damping is read from the EMERGENT
    # dynamics, not the raw input spectrum: project the signal onto the L_sym
    # eigenmodes and, per mode, fit an AR(2) -- complex characteristic roots
    # mean the mode oscillates (under-damped / conservative-wave face), real
    # roots mean it relaxes (over-damped / diffusive face).  The face is the
    # energy-weighted majority (> 1/2 wave, <= 1/2 diffusive).  The input
    # spectrum's quality factor Q is kept only as a secondary read-out.
    q_factor = estimate_quality_factor(data)
    try:
        wave_fraction = _wave_fraction_from_graph(graph, data)
    except Exception:  # pragma: no cover - degenerate graph guard
        wave_fraction = 0.0
    face_valid = wave_fraction <= 0.5  # diffusive (over-damped) face

    return SignalConfrontation(
        n_channels=int(n_channels),
        n_samples=int(n_samples),
        kuramoto_R=float(r_order),
        grad_phi=grad,
        k_phi=kphi,
        phi_s=phis,
        xi_c=xi_c,
        coherence=float(coherence),
        at_equilibrium=bool(at_eq),
        pulse_fundamental=float(pulse["fundamental"]),
        dominant_beat=float(pulse["dominant_beat"]),
        vibration_energy=float(pulse["vibration_energy"]),
        quality_factor=float(q_factor),
        wave_fraction=float(wave_fraction),
        diffusive_face_valid=bool(face_valid),
    )


@dataclass(frozen=True)
class NodalPredictionSkill:
    """One-step predictive skill of the nodal equation on a real signal.

    Confronts the **evolution law** (not just the static read-outs): the EPI
    channel of ``dEPI/dt = ν_f·ΔNFR`` is graph diffusion, so the one-step
    predictor is ``x̂(t+1) = x(t) − c·L_rw·x(t)`` with a single diffusion step
    ``c = ν_f·dt``.  Skill is the fraction of the one-step increment variance it
    explains beyond persistence; a per-channel AR-1 is the standard baseline.
    """

    n_channels: int
    n_samples: int
    diffusivity: float  # fitted c = ν_f·dt (the structural diffusion step)
    nodal_skill: float  # 1 − MSE(nodal) / MSE(persistence)
    ar1_skill: float    # 1 − MSE(per-channel AR-1) / MSE(persistence)

    @property
    def beats_persistence(self) -> bool:
        """Whether the nodal diffusion predictor improves on persistence."""
        return self.nodal_skill > 0.0

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        return (
            f"NodalPredictionSkill[{self.n_channels}ch × {self.n_samples}]: "
            f"nodal_skill={self.nodal_skill:+.3f} vs AR-1 "
            f"{self.ar1_skill:+.3f} (c=ν_f·dt={self.diffusivity:+.3f}); "
            f"beats persistence={self.beats_persistence}"
        )


def _nodal_skill_from_graph(
    graph: Any, data: "np.ndarray"
) -> tuple[float, float, float]:
    """One-step (c, nodal_skill, ar1_skill) from an already-built graph."""
    nodes, lrw = structural_diffusion_operator(graph)
    try:
        idx = np.asarray([int(nd) for nd in nodes])
        x = np.asarray(data, dtype=float)[idx]
    except (ValueError, IndexError, TypeError):
        x = np.asarray(data, dtype=float)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    xc = (x - mu) / sd
    r = xc[:, 1:] - xc[:, :-1]           # actual one-step increment
    d = -(np.asarray(lrw, dtype=float) @ xc[:, :-1])  # diffusion direction
    den = float(np.sum(d * d))
    c = float(np.sum(r * d) / den) if den > 0.0 else 0.0
    mse_persist = float(np.mean(r * r))
    if mse_persist <= 0.0:
        return c, 0.0, 0.0
    mse_nodal = float(np.mean((r - c * d) ** 2))
    ar_res = []
    for i in range(xc.shape[0]):
        xi, xn = xc[i, :-1], xc[i, 1:]
        denom = float(np.dot(xi, xi))
        a = float(np.dot(xn, xi) / denom) if denom > 0.0 else 0.0
        ar_res.append(xn - a * xi)
    mse_ar = float(np.mean(np.concatenate(ar_res) ** 2))
    return c, 1.0 - mse_nodal / mse_persist, 1.0 - mse_ar / mse_persist


def nodal_prediction_skill(
    signals: Any, *, k_neighbours: int = 4
) -> NodalPredictionSkill:
    """Confront the nodal equation as a one-step predictor on a real signal.

    The EPI channel of ``dEPI/dt = ν_f·ΔNFR`` is graph diffusion
    ``dEPI/dt = −ν_f·L_rw·EPI``, so its one-step (Euler) predictor on the
    emergent coupling graph is ``x̂(t+1) = x(t) − c·L_rw·x(t)`` with a single
    diffusion step ``c = ν_f·dt`` fitted by least squares.  Returns the fraction
    of one-step increment variance explained beyond persistence, and the
    per-channel AR-1 baseline.  **Diffusion-selective**: positive on diffusive
    dynamics, weak on oscillatory ones (where AR-1 wins) -- the nodal EPI
    channel is a diffusion law, so this is a structural-fidelity confrontation,
    not a claim of superior forecasting.
    """
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError(
            "signals must be (n_channels, n_samples) with >= 3 channels"
        )
    if data.shape[1] < 8:
        raise ValueError("signals must have >= 8 time samples")
    phase, amp = phase_amplitude_matrices(data)
    graph = build_coupling_graph(phase, amp, k_neighbours=k_neighbours)
    c, nodal, ar1 = _nodal_skill_from_graph(graph, data)
    return NodalPredictionSkill(
        n_channels=int(data.shape[0]),
        n_samples=int(data.shape[1]),
        diffusivity=float(c),
        nodal_skill=float(nodal),
        ar1_skill=float(ar1),
    )
