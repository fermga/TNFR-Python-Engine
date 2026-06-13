"""TNFR multi-channel interface analysis for coupled-oscillator networks.

This module is the *phase-coupled network* counterpart of
:mod:`tnfr.validation.temporal_interface`.  Where the temporal module embeds a
**single** scalar series into a delay-coordinate graph, this module treats a set
of **simultaneously-measured channels** as a network of coupled oscillators —
the setting for which the TNFR Structural Field Tetrad is genuinely native.

Why multi-channel is the strong case
-------------------------------------
For a single scalar series the established critical-slowing-down indicators
(rolling variance, lag-1 autocorrelation) are the right tool and typically win;
the temporal module reports that honestly.  In a multi-channel phase-coupled
network the structural fields acquire a genuine *spatial* meaning:

- ``|∇φ|``  — phase desynchronisation between coupled channels,
- ``K_φ``   — curvature of the phase field over the coupling graph,
- ``ξ_C``   — how far phase coherence extends across the network (a length the
  global synchrony order parameter cannot express),
- ``Φ_s``   — structural potential from the (phase-independent) pressure field.

Pipeline
--------
``multi-channel signals -> per-channel Hilbert phase/amplitude -> phase-locking
coupling graph (nodes = channels) -> TNFR spatial tetrad per window`` compared
against the recognised synchronisation baselines: the **Kuramoto order
parameter** ``R`` (the gold-standard global phase-synchrony measure), the mean
phase-locking value, and the spatial phase dispersion.

Scope and honesty
-----------------
- The Kuramoto order parameter ``R = |⟨e^{iφ}⟩|`` is a strong, standard baseline,
  included so the comparison is fair rather than a strawman.
- ``|∇φ|`` is *partially* redundant with ``1 − R`` (both fall as the network
  synchronises).  The genuinely distinct TNFR contributions are ``ξ_C`` (a
  spatial coherence *length*) and ``K_φ`` (phase-field curvature); neither has a
  direct order-parameter analogue.  This partial redundancy is stated up front,
  not hidden.
- The per-node structural pressure ``ΔNFR`` is derived from the analytic
  **amplitude** envelope, which is independent of the phase field, so ``Φ_s`` and
  ``ξ_C`` are not trivially reproductions of ``|∇φ|``.
- All functions are read-only telemetry; the only graph mutation is the
  construction of new coupling graphs and the setting of ``phase``/``theta`` and
  ``dnfr``/``delta_nfr`` node attributes at build time.

References
----------
- ``src/tnfr/validation/temporal_interface.py`` — single-series counterpart
- ``src/tnfr/physics/canonical.py`` — canonical tetrad field functions
- AGENTS.md §"Telemetry & Structural Field Tetrad"
- Kuramoto, "Chemical Oscillations, Waves, and Turbulence" (1984);
  Acebrón et al., Rev. Mod. Phys. 77, 137 (2005) — order parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None  # type: ignore[assignment]

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency guard
    nx = None  # type: ignore[assignment]

from ..physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)

__all__ = [
    "MultichannelConfig",
    "MultichannelWindowSeries",
    "SynchronyDiscrimination",
    "fft_bandpass",
    "analytic_phase_amplitude",
    "phase_amplitude_matrices",
    "mean_field_order",
    "kuramoto_order_parameter",
    "phase_locking_matrix",
    "phase_offsets",
    "amplitude_pressure",
    "build_coupling_graph",
    "multichannel_window_series",
    "evaluate_synchrony_discrimination",
]

_EPS = 1e-12


def _require_numpy() -> None:
    if np is None:  # pragma: no cover - core dependency guard
        raise RuntimeError("numpy is required for multi-channel analysis")


def _require_networkx() -> None:
    if nx is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("networkx is required for multi-channel analysis")


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultichannelConfig:
    """Configuration for the multi-channel coupled-oscillator pipeline.

    Parameters
    ----------
    window:
        Number of samples per analysis window.
    step:
        Stride (in samples) between successive windows.
    k_neighbours:
        Neighbours per channel in the phase-locking coupling graph.
    sampling_rate:
        Sampling rate (Hz) of the signals; only used when ``bandpass`` is set.
    bandpass:
        Optional ``(low, high)`` band (Hz) applied to each channel before the
        Hilbert transform.  ``None`` uses the broadband signal.
    """

    window: int = 512
    step: int = 128
    k_neighbours: int = 4
    sampling_rate: float = 1.0
    bandpass: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.window < 16:
            raise ValueError("window must be >= 16 samples")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.k_neighbours < 1:
            raise ValueError("k_neighbours must be >= 1")
        if self.sampling_rate <= 0.0:
            raise ValueError("sampling_rate must be > 0")
        if self.bandpass is not None:
            low, high = self.bandpass
            if not (0.0 <= low < high):
                raise ValueError("bandpass must satisfy 0 <= low < high")


@dataclass(frozen=True)
class MultichannelWindowSeries:
    """Per-window TNFR spatial tetrad plus the matched synchronisation baselines.

    All arrays are aligned by index; entry ``i`` corresponds to the window whose
    most-recent sample is ``window_end[i]``.  TNFR channels are ``grad_phi``,
    ``k_phi``, ``xi_c`` and ``phi_s``; baselines are ``order_parameter``
    (Kuramoto ``R``), ``mean_plv`` and ``phase_dispersion``.
    """

    window_end: "np.ndarray"
    grad_phi: "np.ndarray"
    k_phi: "np.ndarray"
    xi_c: "np.ndarray"
    phi_s: "np.ndarray"
    order_parameter: "np.ndarray"
    mean_plv: "np.ndarray"
    phase_dispersion: "np.ndarray"

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "window_end": [int(v) for v in self.window_end],
            "grad_phi": [float(v) for v in self.grad_phi],
            "k_phi": [float(v) for v in self.k_phi],
            "xi_c": [float(v) for v in self.xi_c],
            "phi_s": [float(v) for v in self.phi_s],
            "order_parameter": [float(v) for v in self.order_parameter],
            "mean_plv": [float(v) for v in self.mean_plv],
            "phase_dispersion": [float(v) for v in self.phase_dispersion],
        }


@dataclass(frozen=True)
class SynchronyDiscrimination:
    """Honest comparison of TNFR tetrad vs synchronisation baselines.

    ``auc`` maps each indicator to its discriminative ROC-AUC against a binary
    regime label (``max(auc, 1 - auc)`` so the arbitrary regime direction does
    not matter).  An AUC near 0.5 means the indicator does not separate the two
    regimes; near 1.0 means clean separation.
    """

    indicators: tuple[str, ...]
    auc: Mapping[str, float]
    tnfr_indicators: tuple[str, ...]
    baseline_indicators: tuple[str, ...]
    best_tnfr: tuple[str, float]
    best_baseline: tuple[str, float]
    n_windows: int
    n_positive_windows: int
    interpretation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        bt, bv = self.best_tnfr
        bb, bbv = self.best_baseline
        return (
            f"best TNFR: {bt}={bv:.3f} | best baseline: {bb}={bbv:.3f} | "
            f"windows={self.n_windows} (positive={self.n_positive_windows})"
        )


# ---------------------------------------------------------------------------
# Signal -> phase/amplitude
# ---------------------------------------------------------------------------


def fft_bandpass(
    signal: Sequence[float], *, sampling_rate: float, low: float, high: float
) -> "np.ndarray":
    """Zero-phase FFT band-pass (pure numpy; no SciPy dependency).

    A brick-wall mask keeps Fourier components with frequency in ``[low, high]``.
    Suitable for isolating an oscillatory band (e.g. EEG alpha) before phase
    extraction; ringing from the sharp cut-off is acceptable for telemetry.
    """
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < 2:
        return x.copy()
    x = x - float(np.mean(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate))
    spectrum = np.fft.rfft(x)
    mask = (freqs >= low) & (freqs <= high)
    spectrum = spectrum * mask
    return np.fft.irfft(spectrum, n=n)


def _analytic_signal(x: "np.ndarray") -> "np.ndarray":
    """Discrete analytic signal via FFT (identical to ``scipy.signal.hilbert``)."""
    n = x.size
    spectrum = np.fft.fft(x)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def analytic_phase_amplitude(
    signal: Sequence[float],
    *,
    sampling_rate: float = 1.0,
    bandpass: tuple[float, float] | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return measured instantaneous ``(phase, amplitude)`` of one channel.

    The signal mean is removed first; if ``bandpass`` is given the signal is
    band-limited before the Hilbert transform so the analytic phase tracks a
    well-defined oscillatory component.
    """
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    n = x.size
    if n < 2:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    x = x - float(np.mean(x))
    if bandpass is not None:
        x = fft_bandpass(
            x, sampling_rate=sampling_rate, low=bandpass[0], high=bandpass[1]
        )
    analytic = _analytic_signal(x)
    return np.angle(analytic), np.abs(analytic)


def phase_amplitude_matrices(
    signals: "np.ndarray",
    *,
    sampling_rate: float = 1.0,
    bandpass: tuple[float, float] | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Compute phase and amplitude matrices for all channels.

    ``signals`` has shape ``(n_channels, n_samples)``.  Returns ``(phase,
    amplitude)`` of the same shape.
    """
    _require_numpy()
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2:
        raise ValueError("signals must have shape (n_channels, n_samples)")
    n_channels, n_samples = data.shape
    phase = np.empty_like(data)
    amplitude = np.empty_like(data)
    for j in range(n_channels):
        phase[j], amplitude[j] = analytic_phase_amplitude(
            data[j], sampling_rate=sampling_rate, bandpass=bandpass
        )
    return phase, amplitude


# ---------------------------------------------------------------------------
# Synchronisation descriptors
# ---------------------------------------------------------------------------


def mean_field_order(
    phase_matrix: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return the Kuramoto order time series ``(R(t), Ψ(t))``.

    ``R(t) = |⟨e^{iφ_j(t)}⟩_j|`` is the instantaneous global synchrony and
    ``Ψ(t)`` the collective phase.
    """
    _require_numpy()
    phases = np.asarray(phase_matrix, dtype=float)
    z = np.mean(np.exp(1j * phases), axis=0)
    return np.abs(z), np.angle(z)


def kuramoto_order_parameter(phase_matrix: "np.ndarray") -> float:
    """Mean Kuramoto order parameter ``⟨R(t)⟩`` over the window (baseline)."""
    r, _ = mean_field_order(phase_matrix)
    return float(np.mean(r))


def phase_locking_matrix(phase_matrix: "np.ndarray") -> "np.ndarray":
    """Pairwise phase-locking value matrix ``PLV_{jk} = |⟨e^{i(φ_j-φ_k)}⟩_t|``.

    Returns an ``(n_channels, n_channels)`` matrix with unit diagonal.
    """
    _require_numpy()
    phases = np.asarray(phase_matrix, dtype=float)
    n_samples = phases.shape[1]
    e = np.exp(1j * phases)
    plv = np.abs(e @ e.conj().T) / float(max(1, n_samples))
    np.fill_diagonal(plv, 1.0)
    return np.clip(plv, 0.0, 1.0)


def phase_offsets(phase_matrix: "np.ndarray") -> "np.ndarray":
    """Mean phase offset of each channel from the collective rhythm.

    ``θ_j = angle(⟨e^{i(φ_j(t) - Ψ(t))}⟩_t)`` where ``Ψ(t)`` is the mean-field
    phase.  Channels locked to the collective have ``θ_j ≈ 0``; leading/lagging
    channels have ``θ_j ≠ 0``.  The result is a bounded scalar per channel,
    suitable as the TNFR node phase.
    """
    _require_numpy()
    phases = np.asarray(phase_matrix, dtype=float)
    _, psi = mean_field_order(phases)
    relative = phases - psi[np.newaxis, :]
    return np.angle(np.mean(np.exp(1j * relative), axis=1))


def amplitude_pressure(amplitude_matrix: "np.ndarray") -> "np.ndarray":
    """Phase-independent ΔNFR proxy from the analytic amplitude envelope.

    ``ΔNFR_j = |⟨A_j⟩_t - ⟨A⟩| / (⟨A⟩ + ε)`` — the relative deviation of each
    channel's mean envelope from the network mean.  Because it uses amplitude
    (not phase), it does not trivially reproduce the phase-gradient field.
    """
    _require_numpy()
    amp = np.asarray(amplitude_matrix, dtype=float)
    mean_amp = np.mean(amp, axis=1)
    global_mean = float(np.mean(mean_amp))
    return np.abs(mean_amp - global_mean) / (global_mean + _EPS)


# ---------------------------------------------------------------------------
# Coupling graph
# ---------------------------------------------------------------------------


def build_coupling_graph(
    phase_matrix: "np.ndarray",
    amplitude_matrix: "np.ndarray",
    *,
    k_neighbours: int,
) -> Any:
    """Build the phase-locking coupling graph for one window.

    Nodes are channels.  Edges connect each channel to its ``k_neighbours`` most
    strongly phase-locked partners (coupling distance ``1 − PLV``).  Each node
    carries the measured phase offset (``phase``/``theta``) and the amplitude
    pressure (``dnfr``/``delta_nfr``) so the canonical tetrad field functions
    operate on real telemetry.
    """
    _require_numpy()
    _require_networkx()
    phases = np.asarray(phase_matrix, dtype=float)
    amps = np.asarray(amplitude_matrix, dtype=float)
    n_channels = phases.shape[0]

    plv = phase_locking_matrix(phases)
    distance = 1.0 - plv
    np.fill_diagonal(distance, 0.0)
    theta = phase_offsets(phases)
    pressure = amplitude_pressure(amps)

    G = nx.Graph()
    for j in range(n_channels):
        G.add_node(
            j,
            phase=float(theta[j]),
            theta=float(theta[j]),
            dnfr=float(pressure[j]),
            delta_nfr=float(pressure[j]),
        )

    k = max(1, min(int(k_neighbours), n_channels - 1))
    for j in range(n_channels):
        order = np.argsort(distance[j])
        added = 0
        for idx in order:
            neighbour = int(idx)
            if neighbour == j:
                continue
            G.add_edge(j, neighbour, distance=float(distance[j, neighbour]))
            added += 1
            if added >= k:
                break
    return G


def _mean_abs_dict(values: Mapping[Any, float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean([abs(v) for v in values.values()]))


# ---------------------------------------------------------------------------
# Windowed tetrad series
# ---------------------------------------------------------------------------


def _as_channel_matrix(signals: "np.ndarray") -> "np.ndarray":
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2:
        raise ValueError("signals must have shape (n_channels, n_samples)")
    return data


def multichannel_window_series(
    signals: "np.ndarray", *, config: MultichannelConfig | None = None
) -> MultichannelWindowSeries:
    """Compute the per-window spatial tetrad and synchronisation baselines.

    Parameters
    ----------
    signals:
        Array of shape ``(n_channels, n_samples)``.
    config:
        Pipeline configuration (window, step, neighbours, optional band-pass).
    """
    _require_numpy()
    cfg = config or MultichannelConfig()
    data = _as_channel_matrix(signals)
    n_channels, n_samples = data.shape
    if n_channels < 3:
        raise ValueError("multi-channel analysis needs >= 3 channels")
    if n_samples < cfg.window:
        raise ValueError(
            f"signal length ({n_samples}) shorter than window ({cfg.window})"
        )

    ends: list[int] = []
    grad: list[float] = []
    kphi: list[float] = []
    xi: list[float] = []
    phis: list[float] = []
    order: list[float] = []
    plv_mean: list[float] = []
    dispersion: list[float] = []

    upper = np.triu_indices(n_channels, k=1)
    start = 0
    while start + cfg.window <= n_samples:
        seg = data[:, start : start + cfg.window]
        phase, amplitude = phase_amplitude_matrices(
            seg, sampling_rate=cfg.sampling_rate, bandpass=cfg.bandpass
        )
        graph = build_coupling_graph(
            phase, amplitude, k_neighbours=cfg.k_neighbours
        )
        grad.append(_mean_abs_dict(compute_phase_gradient(graph)))
        kphi.append(_mean_abs_dict(compute_phase_curvature(graph)))
        xi.append(float(estimate_coherence_length(graph)))
        phis.append(_mean_abs_dict(compute_structural_potential(graph)))

        order.append(kuramoto_order_parameter(phase))
        plv = phase_locking_matrix(phase)
        plv_mean.append(float(np.mean(plv[upper])))
        theta = phase_offsets(phase)
        dispersion.append(float(1.0 - np.abs(np.mean(np.exp(1j * theta)))))

        ends.append(start + cfg.window - 1)
        start += cfg.step

    return MultichannelWindowSeries(
        window_end=np.asarray(ends, dtype=int),
        grad_phi=np.asarray(grad, dtype=float),
        k_phi=np.asarray(kphi, dtype=float),
        xi_c=np.asarray(xi, dtype=float),
        phi_s=np.asarray(phis, dtype=float),
        order_parameter=np.asarray(order, dtype=float),
        mean_plv=np.asarray(plv_mean, dtype=float),
        phase_dispersion=np.asarray(dispersion, dtype=float),
    )


# ---------------------------------------------------------------------------
# Regime discrimination (AUC)
# ---------------------------------------------------------------------------


def _average_ranks(sorted_values: "np.ndarray") -> "np.ndarray":
    n = sorted_values.size
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _rank_auc(values: "np.ndarray", labels: "np.ndarray") -> float:
    """Mann–Whitney ROC-AUC (tie-aware); ``labels`` is a boolean array."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=bool)
    mask = np.isfinite(v)
    v = v[mask]
    y = y[mask]
    n_pos = int(np.count_nonzero(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(v, kind="mergesort")
    ranks = _average_ranks(v[order])
    rank_of = np.empty_like(ranks)
    rank_of[order] = ranks
    auc = (rank_of[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _discriminative_auc(values: "np.ndarray", labels: "np.ndarray") -> float:
    auc = _rank_auc(values, labels)
    return max(auc, 1.0 - auc)


_TNFR_CHANNELS = ("grad_phi", "k_phi", "xi_c", "phi_s")
_BASELINE_CHANNELS = ("order_parameter", "mean_plv", "phase_dispersion")


def _window_labels(
    labels: Sequence[float], series: MultichannelWindowSeries, window: int
) -> "np.ndarray":
    """Majority binary label per window (label at each window's sample span)."""
    arr = np.asarray(labels, dtype=float)
    out = np.empty(series.window_end.size, dtype=bool)
    for i, end in enumerate(series.window_end):
        start = int(end) - window + 1
        segment = arr[max(0, start) : int(end) + 1]
        out[i] = bool(np.mean(segment) >= 0.5) if segment.size else False
    return out


def evaluate_synchrony_discrimination(
    signals: "np.ndarray",
    labels: Sequence[float],
    *,
    config: MultichannelConfig | None = None,
) -> SynchronyDiscrimination:
    """Compare TNFR tetrad vs synchrony baselines at separating two regimes.

    ``labels`` is a per-sample binary regime label (0/1) aligned with the
    signal samples.  Each window receives its majority label; for every
    indicator the discriminative ROC-AUC against the window labels is reported.

    The Kuramoto order parameter is the gold-standard baseline.  ``|∇φ|`` is
    partially redundant with ``1 − R``; the genuinely distinct TNFR fields are
    ``ξ_C`` and ``K_φ``, so the honest question is whether they *add*
    discriminative power, not whether the tetrad beats a strawman.
    """
    _require_numpy()
    cfg = config or MultichannelConfig()
    series = multichannel_window_series(signals, config=cfg)
    window_labels = _window_labels(labels, series, cfg.window)

    channels = {
        "grad_phi": series.grad_phi,
        "k_phi": series.k_phi,
        "xi_c": series.xi_c,
        "phi_s": series.phi_s,
        "order_parameter": series.order_parameter,
        "mean_plv": series.mean_plv,
        "phase_dispersion": series.phase_dispersion,
    }
    auc = {
        name: _discriminative_auc(values, window_labels)
        for name, values in channels.items()
    }

    def _best(group: tuple[str, ...]) -> tuple[str, float]:
        best_name = max(group, key=lambda n: auc[n])
        return best_name, auc[best_name]

    best_tnfr = _best(_TNFR_CHANNELS)
    best_baseline = _best(_BASELINE_CHANNELS)
    n_pos = int(np.count_nonzero(window_labels))

    gap = best_tnfr[1] - best_baseline[1]
    if best_tnfr[1] < 0.6 and best_baseline[1] < 0.6:
        interpretation = (
            "Neither the TNFR tetrad nor the synchrony baselines separate the "
            "two regimes well (all AUC < 0.6); the regimes may not differ in "
            "phase-coupling structure."
        )
    elif abs(gap) < 0.03:
        interpretation = (
            f"TNFR '{best_tnfr[0]}' (AUC={best_tnfr[1]:.3f}) and baseline "
            f"'{best_baseline[0]}' (AUC={best_baseline[1]:.3f}) are comparable "
            "regime discriminators."
        )
    elif gap > 0:
        interpretation = (
            f"TNFR '{best_tnfr[0]}' (AUC={best_tnfr[1]:.3f}) separates the "
            f"regimes more strongly than the best baseline '{best_baseline[0]}' "
            f"(AUC={best_baseline[1]:.3f}); Δ={gap:+.3f}."
        )
    else:
        interpretation = (
            f"Baseline '{best_baseline[0]}' (AUC={best_baseline[1]:.3f}) "
            f"separates the regimes more strongly than the best TNFR channel "
            f"'{best_tnfr[0]}' (AUC={best_tnfr[1]:.3f}); Δ={gap:+.3f}."
        )

    return SynchronyDiscrimination(
        indicators=tuple(channels),
        auc=auc,
        tnfr_indicators=_TNFR_CHANNELS,
        baseline_indicators=_BASELINE_CHANNELS,
        best_tnfr=best_tnfr,
        best_baseline=best_baseline,
        n_windows=int(series.window_end.size),
        n_positive_windows=n_pos,
        interpretation=interpretation,
        metadata={
            "n_channels": int(_as_channel_matrix(signals).shape[0]),
            "config": {
                "window": cfg.window,
                "step": cfg.step,
                "k_neighbours": cfg.k_neighbours,
                "sampling_rate": cfg.sampling_rate,
                "bandpass": list(cfg.bandpass) if cfg.bandpass else None,
            },
        },
    )
