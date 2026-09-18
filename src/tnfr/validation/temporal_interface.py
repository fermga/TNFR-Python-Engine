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
- The analytic-signal phase is a declared signal descriptor. The Hilbert phase
  of a frequency trace is not automatically the oscillator phase whose
  derivative generated that frequency. A physical observation map is separate.
- Default whole-record processing is retrospective. Prospective processing
  transforms only the declared warmup/window/latency block available at each
  emission time; it does not certify a future event or physical regime.
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
import hashlib
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
    "TemporalWarningCalibration",
    "ProspectiveWarningComparison",
    "calibrate_temporal_warning",
    "evaluate_prospective_warning",
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
        for name in ("embedding_dim", "embedding_tau", "k_neighbours", "window", "step"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer, not a boolean")
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
        if (self.embedding_dim - 1) * self.embedding_tau >= self.window - 1:
            raise ValueError("window must contain at least two embedding vectors")


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
    mode: str = "retrospective"
    available_at: "np.ndarray | None" = None
    warmup_samples: int = 0
    latency_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_end": [int(v) for v in self.window_end],
            "grad_phi": [float(v) for v in self.grad_phi],
            "k_phi": [float(v) for v in self.k_phi],
            "xi_c": [float(v) for v in self.xi_c],
            "phi_s": [float(v) for v in self.phi_s],
            "variance": [float(v) for v in self.variance],
            "lag1_autocorr": [float(v) for v in self.lag1_autocorr],
            "mode": self.mode,
            "available_at": (
                [int(v) for v in self.available_at]
                if self.available_at is not None else None
            ),
            "warmup_samples": self.warmup_samples,
            "latency_samples": self.latency_samples,
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
    mode: str = "retrospective",
    warmup_samples: int = 0,
    latency_samples: int = 0,
) -> WindowTetradSeries:
    """Compute the TNFR tetrad and matched EWS baselines over rolling windows.

    Returns aligned arrays for the tetrad channels (mean |∇φ|, mean |K_φ|, ξ_C,
    mean |Φ_s|) and the classical EWS baselines (variance, lag-1 autocorrelation)
    so they can be compared on identical windows. The default is descriptive
    whole-record processing. In ``prospective`` mode each window uses only its
    preceding warmup, itself and the declared latency samples. Its result may
    be used only at ``available_at`` (inclusive), not at ``window_end`` when
    latency is nonzero. Windows without the complete declared context are not
    emitted. This is causal emission of telemetry, not a forecast certificate.
    """
    _require_numpy()
    cfg = config or TemporalInterfaceConfig()
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be a finite one-dimensional series; preserve gaps separately")
    if mode not in ("retrospective", "prospective"):
        raise ValueError("mode must be retrospective or prospective")
    for name, value in (("warmup_samples", warmup_samples), ("latency_samples", latency_samples)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if mode == "retrospective" and (warmup_samples or latency_samples):
        raise ValueError("warmup and latency belong to prospective processing")
    n = x.size
    if n < cfg.window:
        raise ValueError("signal shorter than a single window")

    if mode == "retrospective":
        phase = hilbert_instantaneous_phase(x)
        pressure = local_structural_pressure(x)

    ends: list[int] = []
    availability: list[int] = []
    grad: list[float] = []
    kphi: list[float] = []
    xic: list[float] = []
    phis: list[float] = []
    var: list[float] = []
    ac1: list[float] = []

    start = warmup_samples
    while start + cfg.window + latency_samples <= n:
        stop = start + cfg.window
        if mode == "prospective":
            context = x[start - warmup_samples:stop + latency_samples]
            phase = hilbert_instantaneous_phase(context)
            pressure = local_structural_pressure(context)
            seg_phase = phase[warmup_samples:warmup_samples + cfg.window]
            seg_press = pressure[warmup_samples:warmup_samples + cfg.window]
        else:
            seg_phase = phase[start:stop]
            seg_press = pressure[start:stop]
        seg = x[start:stop]

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
        availability.append(stop + latency_samples - 1 if mode == "prospective" else n - 1)
        start += cfg.step

    return WindowTetradSeries(
        window_end=np.asarray(ends, dtype=int),
        grad_phi=np.asarray(grad, dtype=float),
        k_phi=np.asarray(kphi, dtype=float),
        xi_c=np.asarray(xic, dtype=float),
        phi_s=np.asarray(phis, dtype=float),
        variance=np.asarray(var, dtype=float),
        lag1_autocorr=np.asarray(ac1, dtype=float),
        mode=mode,
        available_at=np.asarray(availability, dtype=int),
        warmup_samples=warmup_samples,
        latency_samples=latency_samples,
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
    """Retrospectively compare trends; selection is descriptive, not held-out.

    Whole-record Hilbert/smoothing can use post-event samples. Use
    ``evaluate_prospective_warning`` for separately calibrated channel choices
    and features emitted using only their available context.

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
            "mode": "retrospective",
            "channel_selection": "same_record_descriptive",
            "prospective_prediction": False,
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


@dataclass(frozen=True)
class TemporalWarningCalibration:
    """Channel choices learned once from one declared calibration run.

    This freezes feature extraction and selection, not an event classifier or
    a physical state map. Run identifiers and the content digest make accidental
    reuse visible; independence of the acquisition runs remains a protocol duty.
    """

    config: TemporalInterfaceConfig
    warmup_samples: int
    latency_samples: int
    tnfr_channel: str
    baseline_channel: str
    calibration_run_id: str
    calibration_sha256: str
    calibration_samples: int


@dataclass(frozen=True)
class ProspectiveWarningComparison:
    """Evaluation of frozen channels with explicit unavailable outcomes."""

    tnfr_channel: str
    baseline_channel: str
    tnfr_trend: float | None
    baseline_trend: float | None
    tnfr_valid_windows: int
    baseline_valid_windows: int
    available_at: tuple[int, ...]
    status: str
    reason: str
    calibration_run_id: str
    evaluation_run_id: str


def _run_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run identity must be a nonempty string")
    return value


def _signal_digest(signal: "np.ndarray") -> str:
    return hashlib.sha256(np.asarray(signal, dtype="<f8").tobytes()).hexdigest()


def _resolved_trend(values: "np.ndarray") -> tuple[float | None, int]:
    count = int(np.count_nonzero(np.isfinite(values)))
    trend = kendall_tau(values)
    return (float(trend) if math.isfinite(trend) else None), count


def calibrate_temporal_warning(
    signal: Sequence[float],
    *,
    calibration_run_id: str,
    config: TemporalInterfaceConfig | None = None,
    warmup_samples: int = 0,
    latency_samples: int = 0,
) -> TemporalWarningCalibration:
    """Select finite trend channels using calibration data alone.

    Whole independent runs must be reserved for evaluation. Feature windows
    have the same causal emission semantics as the eventual evaluator.
    No calibration channel with an undefined trend is selected by fallback.
    """
    run_id = _run_id(calibration_run_id)
    cfg = config or TemporalInterfaceConfig()
    series = window_tetrad_series(
        signal, config=cfg, mode="prospective", warmup_samples=warmup_samples,
        latency_samples=latency_samples,
    )

    def select(channels: tuple[str, ...]) -> str:
        choices = [(name, _resolved_trend(getattr(series, name))[0]) for name in channels]
        finite = [(name, value) for name, value in choices if value is not None]
        if not finite:
            raise ValueError("calibration has no resolved trend in a required channel family")
        return max(finite, key=lambda item: item[1])[0]

    return TemporalWarningCalibration(
        config=cfg, warmup_samples=warmup_samples, latency_samples=latency_samples,
        tnfr_channel=select(_TNFR_CHANNELS), baseline_channel=select(_BASELINE_CHANNELS),
        calibration_run_id=run_id, calibration_sha256=_signal_digest(np.asarray(signal)),
        calibration_samples=len(signal),
    )


def evaluate_prospective_warning(
    signal: Sequence[float],
    *,
    calibration: TemporalWarningCalibration,
    evaluation_run_id: str,
    transition_index: int | None = None,
) -> ProspectiveWarningComparison:
    """Evaluate frozen channels without fitting or same-record selection.

    If supplied, ``transition_index`` is a declared evaluation cutoff: only
    emissions strictly before it are scored. It is not a detected/predicted
    event. The suffix after that cutoff is not transformed or validated.
    Descriptive trend agreement is not an event-prediction certificate.
    """
    if not isinstance(calibration, TemporalWarningCalibration):
        raise TypeError("calibration must be a TemporalWarningCalibration")
    run_id = _run_id(evaluation_run_id)
    if run_id == calibration.calibration_run_id:
        raise ValueError("calibration and evaluation require different run identities")
    if (calibration.tnfr_channel not in _TNFR_CHANNELS
            or calibration.baseline_channel not in _BASELINE_CHANNELS):
        raise ValueError("calibration contains unknown channels")
    stop = len(signal)
    if transition_index is not None:
        if type(transition_index) is not int or not 0 <= transition_index <= stop:
            raise ValueError("transition_index must be an integer inside the supplied record")
        stop = transition_index
    prefix = np.asarray(signal[:stop], dtype=float)
    if prefix.ndim != 1 or not np.all(np.isfinite(prefix)):
        raise ValueError("evaluation prefix must be finite and one-dimensional")
    if _signal_digest(prefix) == calibration.calibration_sha256:
        raise ValueError("evaluation content repeats the calibration record")
    required = (calibration.warmup_samples + calibration.config.window
                + calibration.latency_samples)
    if len(prefix) < required:
        series = None
    else:
        series = window_tetrad_series(
            prefix, config=calibration.config, mode="prospective",
            warmup_samples=calibration.warmup_samples,
            latency_samples=calibration.latency_samples,
        )
    tnfr, n_tnfr = (None, 0) if series is None else _resolved_trend(
        getattr(series, calibration.tnfr_channel)
    )
    baseline, n_baseline = (None, 0) if series is None else _resolved_trend(
        getattr(series, calibration.baseline_channel)
    )
    resolved = tnfr is not None and baseline is not None
    return ProspectiveWarningComparison(
        tnfr_channel=calibration.tnfr_channel,
        baseline_channel=calibration.baseline_channel,
        tnfr_trend=tnfr, baseline_trend=baseline,
        tnfr_valid_windows=n_tnfr, baseline_valid_windows=n_baseline,
        available_at=() if series is None else tuple(int(x) for x in series.available_at),
        status="descriptive_evaluation" if resolved else "unavailable",
        reason=("Frozen-channel trends; no event or physical-regime certificate."
                if resolved else "Insufficient finite windows or a degenerate selected trend."),
        calibration_run_id=calibration.calibration_run_id, evaluation_run_id=run_id,
    )
