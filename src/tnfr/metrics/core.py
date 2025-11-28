"""Basic metrics orchestrator."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, NamedTuple, cast

from ..utils import CallbackEvent, callback_manager
from ..constants import get_param
from ..glyph_history import append_metric, ensure_history
from ..telemetry import ensure_nu_f_telemetry
from ..units import get_hz_bridge
from ..telemetry.verbosity import (
    TELEMETRY_VERBOSITY_DEFAULT,
    TELEMETRY_VERBOSITY_LEVELS,
    TelemetryVerbosity,
)
from ..types import (
    GlyphSelector,
    NodeId,
    SelectorPreselectionChoices,
    SelectorPreselectionMetrics,
    SelectorPreselectionPayload,
    TNFRGraph,
    TraceCallback,
    TraceFieldFn,
    TraceFieldMap,
    TraceFieldRegistry,
)
from ..utils import get_logger
from .coherence import (
    GLYPH_LOAD_STABILIZERS_KEY,
    _aggregate_si,
    _track_stability,
    _update_coherence,
    _update_phase_sync,
    _update_sigma,
    register_coherence_callbacks,
)
from .diagnosis import register_diagnosis_callbacks
from .glyph_timing import GlyphMetricsHistory, _compute_advanced_metrics
from .reporting import (
    Tg_by_node,
    Tg_global,
    glyph_top,
    glyphogram_series,
    latency_series,
)

logger = get_logger(__name__)

__all__ = [
    "TNFRGraph",
    "NodeId",
    "GlyphSelector",
    "SelectorPreselectionMetrics",
    "SelectorPreselectionChoices",
    "SelectorPreselectionPayload",
    "TraceCallback",
    "TraceFieldFn",
    "TraceFieldMap",
    "TraceFieldRegistry",
    "_metrics_step",
    "register_metrics_callbacks",
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
]


class MetricsVerbositySpec(NamedTuple):
    """Runtime configuration for metrics verbosity tiers."""

    name: str
    enable_phase_sync: bool
    enable_sigma: bool
    enable_aggregate_si: bool
    enable_advanced: bool
    attach_coherence_hooks: bool
    attach_diagnosis_hooks: bool
    enable_nu_f: bool


METRICS_VERBOSITY_DEFAULT = TELEMETRY_VERBOSITY_DEFAULT

_METRICS_VERBOSITY_PRESETS: dict[str, MetricsVerbositySpec] = {}


def _register_metrics_preset(spec: MetricsVerbositySpec) -> None:
    if spec.name not in TELEMETRY_VERBOSITY_LEVELS:
        raise ValueError(
            "Unknown metrics verbosity '%s'; use %s"
            % (
                spec.name,
                ", ".join(TELEMETRY_VERBOSITY_LEVELS),
            )
        )
    _METRICS_VERBOSITY_PRESETS[spec.name] = spec


_register_metrics_preset(
    MetricsVerbositySpec(
        name=TelemetryVerbosity.BASIC.value,
        enable_phase_sync=False,
        enable_sigma=False,
        enable_aggregate_si=False,
        enable_advanced=False,
        attach_coherence_hooks=False,
        attach_diagnosis_hooks=False,
        enable_nu_f=False,
    )
)

_detailed_spec = MetricsVerbositySpec(
    name=TelemetryVerbosity.DETAILED.value,
    enable_phase_sync=True,
    enable_sigma=True,
    enable_aggregate_si=True,
    enable_advanced=False,
    attach_coherence_hooks=True,
    attach_diagnosis_hooks=False,
    enable_nu_f=True,
)
_register_metrics_preset(_detailed_spec)
_register_metrics_preset(
    _detailed_spec._replace(
        name=TelemetryVerbosity.DEBUG.value,
        enable_advanced=True,
        attach_diagnosis_hooks=True,
        enable_nu_f=True,
    )
)

_METRICS_BASE_HISTORY_KEYS = ("C_steps", "stable_frac", "delta_Si", "B")
_METRICS_PHASE_HISTORY_KEYS = ("phase_sync", "kuramoto_R")
_METRICS_SIGMA_HISTORY_KEYS = (
    GLYPH_LOAD_STABILIZERS_KEY,
    "glyph_load_disr",
    "sense_sigma_x",
    "sense_sigma_y",
    "sense_sigma_mag",
    "sense_sigma_angle",
)
_METRICS_SI_HISTORY_KEYS = ("Si_mean", "Si_hi_frac", "Si_lo_frac")
_METRICS_NU_F_HISTORY_KEYS = (
    "nu_f_rate_hz_str",
    "nu_f_rate_hz",
    "nu_f_ci_lower_hz_str",
    "nu_f_ci_upper_hz_str",
    "nu_f_ci_lower_hz",
    "nu_f_ci_upper_hz",
)


def _update_nu_f_snapshot(
    G: TNFRGraph,
    hist: MutableMapping[str, Any],
    *,
    record_history: bool,
) -> None:
    """Refresh νf telemetry snapshot and optionally persist it in history."""

    accumulator = ensure_nu_f_telemetry(G, confidence_level=None)
    snapshot = accumulator.snapshot(graph=G)
    payload = snapshot.as_payload()
    bridge: float | None
    try:
        bridge = float(get_hz_bridge(G))
    except (TypeError, ValueError, KeyError):
        bridge = None
    else:
        payload["hz_bridge"] = bridge

    telemetry = G.graph.setdefault("telemetry", {})
    if not isinstance(telemetry, MutableMapping):
        telemetry = {}
        G.graph["telemetry"] = telemetry
    telemetry["nu_f_snapshot"] = payload
    telemetry["nu_f_bridge"] = bridge

    if record_history:
        append_metric(hist, "nu_f_rate_hz_str", snapshot.rate_hz_str)
        append_metric(hist, "nu_f_rate_hz", snapshot.rate_hz)
        append_metric(hist, "nu_f_ci_lower_hz_str", snapshot.ci_lower_hz_str)
        append_metric(hist, "nu_f_ci_upper_hz_str", snapshot.ci_upper_hz_str)
        append_metric(hist, "nu_f_ci_lower_hz", snapshot.ci_lower_hz)
        append_metric(hist, "nu_f_ci_upper_hz", snapshot.ci_upper_hz)

    G.graph["_nu_f_snapshot_payload"] = payload


def _resolve_metrics_verbosity(cfg: Mapping[str, Any]) -> MetricsVerbositySpec:
    """Return the preset matching ``cfg['verbosity']``."""

    raw_value = cfg.get("verbosity", METRICS_VERBOSITY_DEFAULT)
    key = str(raw_value).lower()
    spec = _METRICS_VERBOSITY_PRESETS.get(key)
    if spec is not None:
        return spec
    logger.warning(
        "Unknown METRICS verbosity '%s'; falling back to '%s'",
        raw_value,
        METRICS_VERBOSITY_DEFAULT,
    )
    return _METRICS_VERBOSITY_PRESETS[METRICS_VERBOSITY_DEFAULT]


def _metrics_step(G: TNFRGraph, ctx: dict[str, Any] | None = None) -> None:
    """Update operational TNFR metrics per step."""

    del ctx

    cfg = cast(Mapping[str, Any], get_param(G, "METRICS"))
    if not cfg.get("enabled", True):
        return

    spec = _resolve_metrics_verbosity(cfg)
    hist = ensure_history(G)
    if "glyph_load_estab" in hist:
        raise ValueError(
            "History payloads using 'glyph_load_estab' are no longer supported. "
            "Rename the series to 'glyph_load_stabilizers' before loading the graph."
        )
    metrics_sentinel_key = "_metrics_history_id"
    history_id = id(hist)
    if G.graph.get(metrics_sentinel_key) != history_id:
        for key in _METRICS_BASE_HISTORY_KEYS:
            hist.setdefault(key, [])
        if spec.enable_phase_sync:
            for key in _METRICS_PHASE_HISTORY_KEYS:
                hist.setdefault(key, [])
        if spec.enable_sigma:
            for key in _METRICS_SIGMA_HISTORY_KEYS:
                hist.setdefault(key, [])
        if spec.enable_aggregate_si:
            for key in _METRICS_SI_HISTORY_KEYS:
                hist.setdefault(key, [])
        if spec.enable_nu_f:
            for key in _METRICS_NU_F_HISTORY_KEYS:
                hist.setdefault(key, [])
        G.graph[metrics_sentinel_key] = history_id

    dt = float(get_param(G, "DT"))
    eps_dnfr = float(get_param(G, "EPS_DNFR_STABLE"))
    eps_depi = float(get_param(G, "EPS_DEPI_STABLE"))
    t = float(G.graph.get("_t", 0.0))

    _update_coherence(G, hist)

    raw_jobs = cfg.get("n_jobs")
    metrics_jobs: int | None
    try:
        metrics_jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        metrics_jobs = None
    else:
        if metrics_jobs <= 0:
            metrics_jobs = None

    _track_stability(
        G,
        hist,
        dt,
        eps_dnfr,
        eps_depi,
        n_jobs=metrics_jobs,
    )
    if spec.enable_phase_sync or spec.enable_sigma:
        try:
            if spec.enable_phase_sync:
                _update_phase_sync(G, hist)
            if spec.enable_sigma:
                _update_sigma(G, hist)
        except (KeyError, AttributeError, TypeError) as exc:
            logger.debug("observer update failed: %s", exc)

    if hist.get("C_steps") and hist.get("stable_frac"):
        append_metric(
            hist,
            "iota",
            hist["C_steps"][-1] * hist["stable_frac"][-1],
        )

    if spec.enable_aggregate_si:
        _aggregate_si(G, hist, n_jobs=metrics_jobs)

    _update_nu_f_snapshot(G, hist, record_history=spec.enable_nu_f)

    if spec.enable_advanced:
        _compute_advanced_metrics(
            G,
            cast(GlyphMetricsHistory, hist),
            t,
            dt,
            cfg,
            n_jobs=metrics_jobs,
        )


def register_metrics_callbacks(G: TNFRGraph) -> None:
    """Attach canonical metrics callbacks according to graph configuration."""

    cfg = cast(Mapping[str, Any], get_param(G, "METRICS"))
    spec = _resolve_metrics_verbosity(cfg)
    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=_metrics_step,
        name="metrics_step",
    )
    if spec.attach_coherence_hooks:
        register_coherence_callbacks(G)
    if spec.attach_diagnosis_hooks:
        register_diagnosis_callbacks(G)
