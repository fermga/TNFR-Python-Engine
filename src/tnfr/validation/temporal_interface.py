"""TNFR temporal-interface analysis for phase-native time series.

This module extends TNFR Structural Interface Theory from the *static spatial*
setting (records -> k-NN graph -> injected binary phase) to the *temporal*
setting that the framework is actually designed for: a real, time-observed
signal whose phase is measured (not injected) and whose structural-field tetrad
is tracked as the system approaches a transition.

Pipeline
--------
``real time series -> Hilbert instantaneous phase -> delay-embedding proximity
graph -> per-window TNFR tetrad (|∇φ|, K_φ, ξ_C, Φ_s) -> trend toward a known
transition``, compared against the *recognised* early-warning-signal (EWS)
baselines from the critical-slowing-down literature (rolling variance and
lag-1 autocorrelation; Scheffer et al. 2009, Dakos et al. 2012).

Scope and honesty
-----------------
- The instantaneous phase here is **measured** from the signal via the analytic
  (Hilbert) transform.  For a phase-native observable such as power-grid
  frequency (frequency = dφ/dt), this is a genuine phase, not a label encoded
  as a phase.  This is the qualitative difference from
  :mod:`tnfr.validation.structural_interface`.
- The classical EWS baselines (variance, lag-1 autocorrelation) are the
  established indicators of an approaching bifurcation.  They are included so
  the comparison is fair: any TNFR claim must beat or match them, not a strawman.
- Critical slowing down is a property of systems *slowly* approaching a fold/
  transcritical/Hopf bifurcation.  When a signal does not approach such a
  bifurcation, neither the classical indicators nor the TNFR tetrad are expected
  to show a rising trend; a null/flat result is a correct, honest outcome.
- All functions are read-only telemetry: the only graph mutation is the
  construction of new graphs and the setting of the requested node attributes
  (``phase``/``theta`` and ``dnfr``) at build time.

References
----------
- ``src/tnfr/validation/structural_interface.py`` — static spatial counterpart
- ``src/tnfr/physics/fields.py`` — canonical tetrad field functions
- AGENTS.md §"Telemetry & Structural Field Tetrad"
- Scheffer et al., "Early-warning signals for critical transitions",
  Nature 461 (2009); Dakos et al., PLoS ONE 7(7):e41010 (2012).
"""

from __future__ import annotations

import math
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

from ..physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)
from .structural_interface import build_knn_graph

__all__ = [
    "TemporalInterfaceConfig",
    "WindowTetradSeries",
    "EarlyWarningComparison",
    "hilbert_instantaneous_phase",
    "delay_embedding",
    "local_structural_pressure",
    "build_temporal_proximity_graph",
    "window_tetrad_series",
    "rolling_variance",
    "rolling_lag1_autocorrelation",
    "kendall_tau",
    "evaluate_early_warning",
]


def _require_numpy() -> None:
    if np is None:  # pragma: no cover - core dependency guard
        raise RuntimeError("numpy is required for temporal-interface analysis")


def _require_networkx() -> None:
    if nx is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("networkx is required for temporal-interface analysis")


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalInterfaceConfig:
    """Configuration for the temporal-interface pipeline.

    Parameters
    ----------
    embedding_dim:
        Takens delay-embedding dimension (number of lagged coordinates).
    embedding_tau:
        Delay (in samples) between successive embedding coordinates.
    k_neighbours:
        Neighbours per node in the embedding proximity graph.
    window:
        Number of signal samples per analysis window.
    step:
        Stride (in samples) between successive windows.
    """

    embedding_dim: int = 3
    embedding_tau: int = 1
    k_neighbours: int = 8
    window: int = 240
    step: int = 30

    def __post_init__(self) -> None:
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if self.embedding_tau < 1:
            raise ValueError("embedding_tau must be >= 1")
        if self.k_neighbours < 1:
            raise ValueError("k_neighbours must be >= 1")
        if self.window < 8:
            raise ValueError("window must be >= 8 samples")
        if self.step < 1:
            raise ValueError("step must be >= 1")


@dataclass(frozen=True)
class WindowTetradSeries:
    """Per-window TNFR tetrad telemetry plus the matched EWS baselines.

    All arrays are aligned by index; entry ``i`` corresponds to the window whose
    most-recent sample is ``window_end[i]``.
    """

    window_end: "np.ndarray"
    grad_phi: "np.ndarray"
    k_phi: "np.ndarray"
    xi_c: "np.ndarray"
    phi_s: "np.ndarray"
    variance: "np.ndarray"
    lag1_autocorr: "np.ndarray"

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "window_end": [int(v) for v in self.window_end],
            "grad_phi": [float(v) for v in self.grad_phi],
            "k_phi": [float(v) for v in self.k_phi],
            "xi_c": [float(v) for v in self.xi_c],
            "phi_s": [float(v) for v in self.phi_s],
            "variance": [float(v) for v in self.variance],
            "lag1_autocorr": [float(v) for v in self.lag1_autocorr],
        }


@dataclass(frozen=True)
class EarlyWarningComparison:
    """Honest comparison of TNFR tetrad trends against EWS baselines.

    ``trends`` maps each indicator name to its Kendall-τ trend strength computed
    over the windows that end before ``transition_end`` (the pre-transition
    portion).  A higher positive τ means a stronger rising trend ahead of the
    transition, which is the standard early-warning criterion.
    """

    indicators: tuple[str, ...]
    trends: Mapping[str, float]
    tnfr_indicators: tuple[str, ...]
    baseline_indicators: tuple[str, ...]
    best_tnfr: tuple[str, float]
    best_baseline: tuple[str, float]
    n_pre_transition_windows: int
    interpretation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        bt, bv = self.best_tnfr
        bb, bbv = self.best_baseline
        return (
            f"best TNFR: {bt}={bv:+.3f} | best baseline: {bb}={bbv:+.3f} | "
            f"pre-transition windows={self.n_pre_transition_windows}"
        )


# ---------------------------------------------------------------------------
# Phase extraction and embedding
# ---------------------------------------------------------------------------


def hilbert_instantaneous_phase(signal: Sequence[float]) -> "np.ndarray":
    """Return the measured instantaneous phase φ(t) of ``signal``.

    Uses the analytic signal from the discrete Hilbert transform (FFT-based,
    identical to ``scipy.signal.hilbert``) so there is no hard SciPy dependency.
    The signal mean is removed first because the analytic phase is only
    meaningful for an oscillatory (zero-mean) component.
    """
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    n = x.size
    if n < 2:
        return np.zeros(n, dtype=float)
    x = x - float(np.mean(x))
    spectrum = np.fft.fft(x)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * h)
    return np.angle(analytic)


def delay_embedding(signal: Sequence[float], *, dim: int, tau: int) -> "np.ndarray":
    """Takens delay embedding of a scalar series.

    Row ``i`` is ``[x[i], x[i+tau], ..., x[i+(dim-1)*tau]]``.  The most-recent
    sample of row ``i`` is at index ``i + (dim-1)*tau``.
    """
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    span = (dim - 1) * tau
    rows = x.size - span
    if rows < 1:
        raise ValueError("signal too short for the requested embedding")
    out = np.empty((rows, dim), dtype=float)
    for d in range(dim):
        out[:, d] = x[d * tau : d * tau + rows]
    return out


def local_structural_pressure(
    signal: Sequence[float], *, smoothing: int = 5
) -> "np.ndarray":
    """Phase-independent ΔNFR proxy: deviation from the local smooth trend.

    The reorganization pressure ΔNFR at a sample is estimated as the absolute
    residual ``|x[t] - local_mean(t)|`` against a centred moving average.  This
    is independent of the measured phase, so it does not trivially reproduce the
    phase-gradient field; it feeds Φ_s and ξ_C as a genuine structural source.
    """
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    n = x.size
    w = max(1, int(smoothing))
    if w == 1 or n == 0:
        return np.abs(x - float(np.mean(x)) if n else x)
    kernel = np.ones(w, dtype=float) / float(w)
    smooth = np.convolve(x, kernel, mode="same")
    return np.abs(x - smooth)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_temporal_proximity_graph(
    window_signal: Sequence[float],
    *,
    phase: Sequence[float],
    pressure: Sequence[float],
    config: TemporalInterfaceConfig,
) -> Any:
    """Build a delay-embedding proximity graph for one analysis window.

    Each node is a delay-embedding vector (a short trajectory segment); edges
    connect nearby trajectory states.  Per node we set the **measured** phase
    (``phase``/``theta``) and the structural pressure ``dnfr`` so the canonical
    tetrad field functions operate on real telemetry.

    Parameters
    ----------
    window_signal:
        The raw signal samples in the window (used for the embedding geometry).
    phase:
        Measured instantaneous phase aligned with ``window_signal``.
    pressure:
        ΔNFR proxy aligned with ``window_signal``.
    config:
        Pipeline configuration (embedding + neighbour count).
    """
    _require_numpy()
    _require_networkx()
    x = np.asarray(window_signal, dtype=float)
    phi = np.asarray(phase, dtype=float)
    press = np.asarray(pressure, dtype=float)
    if not (x.size == phi.size == press.size):
        raise ValueError("window_signal, phase and pressure must be aligned")

    vectors = delay_embedding(x, dim=config.embedding_dim, tau=config.embedding_tau)
    span = (config.embedding_dim - 1) * config.embedding_tau
    # Most-recent-sample index for each embedding row.
    recent = np.arange(vectors.shape[0]) + span

    feature_keys = [f"d{d}" for d in range(config.embedding_dim)]
    records = [
        {key: float(vectors[i, d]) for d, key in enumerate(feature_keys)}
        for i in range(vectors.shape[0])
    ]
    k = min(config.k_neighbours, max(1, len(records) - 1))
    G = build_knn_graph(records, feature_keys, k=k, standardize=True)

    for node in G.nodes():
        t = int(recent[node])
        ph = float(phi[t])
        G.nodes[node]["phase"] = ph
        G.nodes[node]["theta"] = ph
        G.nodes[node]["dnfr"] = float(press[t])
        G.nodes[node]["delta_nfr"] = float(press[t])
    return G


# ---------------------------------------------------------------------------
# Per-window tetrad telemetry
# ---------------------------------------------------------------------------


def _mean_abs(values: Mapping[Any, float]) -> float:
    if not values:
        return 0.0
    return float(np.mean([abs(float(v)) for v in values.values()]))


def window_tetrad_series(
    signal: Sequence[float],
    *,
    config: TemporalInterfaceConfig | None = None,
) -> WindowTetradSeries:
    """Compute the TNFR tetrad and matched EWS baselines over rolling windows.

    Returns aligned arrays for the tetrad channels (mean |∇φ|, mean |K_φ|, ξ_C,
    mean |Φ_s|) and the classical EWS baselines (variance, lag-1 autocorrelation)
    so they can be compared on identical windows.
    """
    _require_numpy()
    cfg = config or TemporalInterfaceConfig()
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < cfg.window:
        raise ValueError("signal shorter than a single window")

    phase = hilbert_instantaneous_phase(x)
    pressure = local_structural_pressure(x)

    ends: list[int] = []
    grad: list[float] = []
    kphi: list[float] = []
    xic: list[float] = []
    phis: list[float] = []
    var: list[float] = []
    ac1: list[float] = []

    start = 0
    while start + cfg.window <= n:
        stop = start + cfg.window
        seg = x[start:stop]
        seg_phase = phase[start:stop]
        seg_press = pressure[start:stop]

        G = build_temporal_proximity_graph(
            seg, phase=seg_phase, pressure=seg_press, config=cfg
        )
        grad.append(_mean_abs(compute_phase_gradient(G)))
        kphi.append(_mean_abs(compute_phase_curvature(G)))
        xi = estimate_coherence_length(G)
        xic.append(float(xi) if math.isfinite(xi) else float("nan"))
        phis.append(_mean_abs(compute_structural_potential(G)))

        var.append(float(np.var(seg)))
        ac1.append(_lag1_autocorr(seg))

        ends.append(stop - 1)
        start += cfg.step

    return WindowTetradSeries(
        window_end=np.asarray(ends, dtype=int),
        grad_phi=np.asarray(grad, dtype=float),
        k_phi=np.asarray(kphi, dtype=float),
        xi_c=np.asarray(xic, dtype=float),
        phi_s=np.asarray(phis, dtype=float),
        variance=np.asarray(var, dtype=float),
        lag1_autocorr=np.asarray(ac1, dtype=float),
    )


# ---------------------------------------------------------------------------
# Classical early-warning-signal baselines
# ---------------------------------------------------------------------------


def _lag1_autocorr(segment: "np.ndarray") -> float:
    seg = np.asarray(segment, dtype=float)
    if seg.size < 3:
        return float("nan")
    seg = seg - float(np.mean(seg))
    denom = float(np.dot(seg, seg))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(seg[:-1], seg[1:]) / denom)


def rolling_variance(
    signal: Sequence[float], *, window: int, step: int
) -> "np.ndarray":
    """Rolling variance — the canonical critical-slowing-down indicator."""
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    out: list[float] = []
    start = 0
    while start + window <= x.size:
        out.append(float(np.var(x[start : start + window])))
        start += step
    return np.asarray(out, dtype=float)


def rolling_lag1_autocorrelation(
    signal: Sequence[float], *, window: int, step: int
) -> "np.ndarray":
    """Rolling lag-1 autocorrelation — the second canonical CSD indicator."""
    _require_numpy()
    x = np.asarray(signal, dtype=float)
    out: list[float] = []
    start = 0
    while start + window <= x.size:
        out.append(_lag1_autocorr(x[start : start + window]))
        start += step
    return np.asarray(out, dtype=float)


def kendall_tau(series: Sequence[float]) -> float:
    """Kendall-τ trend strength of ``series`` against its index.

    This is the standard scalar used in the EWS literature to quantify whether
    an indicator rises ahead of a transition.  Implemented with a τ-b correction
    for ties so it is well-defined on short, noisy windows; NaNs are ignored.
    """
    _require_numpy()
    y = np.asarray(series, dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
    m = y.size
    if m < 3:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_y = 0
    for i in range(m - 1):
        dy = y[i + 1 :] - y[i]
        # time is strictly increasing, so no ties in the x (time) variable
        concordant += int(np.count_nonzero(dy > 0))
        discordant += int(np.count_nonzero(dy < 0))
        ties_y += int(np.count_nonzero(dy == 0))
    n0 = m * (m - 1) / 2.0
    denom = math.sqrt((n0 - ties_y) * n0)
    if denom <= 0.0:
        return float("nan")
    return float((concordant - discordant) / denom)


# ---------------------------------------------------------------------------
# Honest comparison
# ---------------------------------------------------------------------------

_TNFR_CHANNELS = ("grad_phi", "k_phi", "xi_c", "phi_s")
_BASELINE_CHANNELS = ("variance", "lag1_autocorr")


def evaluate_early_warning(
    signal: Sequence[float],
    *,
    transition_index: int | None = None,
    config: TemporalInterfaceConfig | None = None,
) -> EarlyWarningComparison:
    """Compare TNFR tetrad trends with EWS baselines ahead of a transition.

    Parameters
    ----------
    signal:
        The (real) time series.
    transition_index:
        Sample index of the transition/event.  Only windows whose most-recent
        sample is strictly before this index contribute to the trend (the
        pre-transition portion).  If ``None``, all windows are used and the
        result describes the trend over the whole record.
    config:
        Pipeline configuration.

    Returns
    -------
    EarlyWarningComparison
        Kendall-τ trend strength for every indicator, plus the best TNFR and
        best baseline indicators, and an honest interpretation string.
    """
    _require_numpy()
    cfg = config or TemporalInterfaceConfig()
    series = window_tetrad_series(signal, config=cfg)
    data = series.as_dict()
    ends = np.asarray(data["window_end"], dtype=int)

    if transition_index is None:
        pre_mask = np.ones(ends.shape, dtype=bool)
    else:
        pre_mask = ends < int(transition_index)
    n_pre = int(np.count_nonzero(pre_mask))

    trends: dict[str, float] = {}
    for name in (*_TNFR_CHANNELS, *_BASELINE_CHANNELS):
        values = np.asarray(data[name], dtype=float)[pre_mask]
        trends[name] = kendall_tau(values)

    def _best(channels: tuple[str, ...]) -> tuple[str, float]:
        best_name = channels[0]
        best_val = trends[best_name]
        for name in channels:
            val = trends[name]
            if math.isnan(best_val) or (not math.isnan(val) and val > best_val):
                best_name, best_val = name, val
        return best_name, (0.0 if math.isnan(best_val) else best_val)

    best_tnfr = _best(_TNFR_CHANNELS)
    best_baseline = _best(_BASELINE_CHANNELS)

    if n_pre < 3:
        interpretation = (
            "Insufficient pre-transition windows for a trend estimate; "
            "increase the record length or reduce the step."
        )
    else:
        gap = best_tnfr[1] - best_baseline[1]
        if abs(gap) < 0.05:
            interpretation = (
                "TNFR tetrad and classical EWS baselines show comparable "
                "pre-transition trends (no decisive difference)."
            )
        elif gap > 0:
            interpretation = (
                f"TNFR channel '{best_tnfr[0]}' shows a stronger rising "
                f"pre-transition trend than the best classical baseline "
                f"'{best_baseline[0]}' (Δτ={gap:+.3f})."
            )
        else:
            interpretation = (
                f"Classical baseline '{best_baseline[0]}' shows a stronger "
                f"rising pre-transition trend than the best TNFR channel "
                f"'{best_tnfr[0]}' (Δτ={gap:+.3f})."
            )

    return EarlyWarningComparison(
        indicators=(*_TNFR_CHANNELS, *_BASELINE_CHANNELS),
        trends=trends,
        tnfr_indicators=_TNFR_CHANNELS,
        baseline_indicators=_BASELINE_CHANNELS,
        best_tnfr=best_tnfr,
        best_baseline=best_baseline,
        n_pre_transition_windows=n_pre,
        interpretation=interpretation,
        metadata={
            "n_windows": int(ends.size),
            "transition_index": transition_index,
            "config": {
                "embedding_dim": cfg.embedding_dim,
                "embedding_tau": cfg.embedding_tau,
                "k_neighbours": cfg.k_neighbours,
                "window": cfg.window,
                "step": cfg.step,
            },
        },
    )
