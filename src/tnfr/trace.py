"""Trace logging.

Field helpers avoid unnecessary copying by reusing dictionaries stored on
the graph whenever possible.  Callers are expected to treat returned
structures as immutable snapshots.

Immutability Guarantees
-----------------------
Trace field producers return mappings wrapped in ``MappingProxyType`` to
prevent accidental mutation. These proxies enforce immutability while avoiding
unnecessary data copying. Consumers that need to modify trace data should
create mutable copies using ``dict(proxy)`` or merge patterns like
``{**proxy1, **proxy2, "new_key": value}``.

Example safe mutation patterns::

    # Get immutable trace data
    result = gamma_field(G)
    gamma_proxy = result["gamma"]

    # Cannot mutate directly (TypeError will be raised)
    # gamma_proxy["new_key"] = value  # ❌ Error!

    # Safe pattern: create mutable copy
    mutable = dict(gamma_proxy)
    mutable["new_key"] = value  # ✓ OK

    # Safe pattern: merge with new data
    combined = {**gamma_proxy, "new_key": value}  # ✓ OK
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol, cast

from .constants import TRACE
from .glyph_history import append_metric, count_glyphs, ensure_history
from .metrics.sense_index import _normalise_si_sensitivity_mapping
from .telemetry.verbosity import (
    TELEMETRY_VERBOSITY_DEFAULT,
    TelemetryVerbosity,
)
from .types import (
    SigmaVector,
    TNFRGraph,
    TraceCallback,
    TraceFieldFn,
    TraceFieldMap,
    TraceFieldRegistry,
    TraceMetadata,
    TraceSnapshot,
)
from .utils import cached_import, get_graph_mapping, is_non_string_sequence
from .utils.callbacks import CallbackSpec


class _KuramotoFn(Protocol):
    def __call__(self, G: TNFRGraph) -> tuple[float, float]: ...


class _SigmaVectorFn(Protocol):
    def __call__(self, G: TNFRGraph, weight_mode: str | None = None) -> SigmaVector: ...


class TraceFieldSpec(NamedTuple):
    """Declarative specification for a trace field producer."""

    name: str
    phase: str
    producer: TraceFieldFn
    tiers: tuple[TelemetryVerbosity, ...]


TRACE_VERBOSITY_DEFAULT = TELEMETRY_VERBOSITY_DEFAULT
TRACE_VERBOSITY_PRESETS: dict[str, tuple[str, ...]] = {}
_TRACE_CAPTURE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "glyphs": "glyph_counts",
    }
)


def _canonical_capture_name(name: str) -> str:
    """Return the canonical capture field name for ``name``."""

    stripped = name.strip()
    alias = _TRACE_CAPTURE_ALIASES.get(stripped)
    if alias is not None:
        return alias

    lowered = stripped.lower()
    alias = _TRACE_CAPTURE_ALIASES.get(lowered)
    if alias is not None:
        return alias

    return stripped


def _normalise_capture_spec(raw: Any) -> set[str]:
    """Coerce custom capture payloads to a ``set`` of field names."""

    if raw is None:
        return set()
    if isinstance(raw, Mapping):
        return {_canonical_capture_name(str(name)) for name in raw.keys()}
    if isinstance(raw, str):
        return {_canonical_capture_name(raw)}
    if isinstance(raw, Iterable):
        return {_canonical_capture_name(str(name)) for name in raw}
    return {_canonical_capture_name(str(raw))}


def _resolve_trace_capture(cfg: Mapping[str, Any]) -> set[str]:
    """Return the capture set declared by ``cfg`` respecting verbosity."""

    if "capture" in cfg:
        return _normalise_capture_spec(cfg.get("capture"))

    raw_verbosity = cfg.get("verbosity", TRACE_VERBOSITY_DEFAULT)
    verbosity = str(raw_verbosity).lower()
    fields = TRACE_VERBOSITY_PRESETS.get(verbosity)
    if fields is None:
        warnings.warn(
            (
                "Unknown TRACE verbosity %r; falling back to %s"
                % (raw_verbosity, TRACE_VERBOSITY_DEFAULT)
            ),
            UserWarning,
            stacklevel=3,
        )
        fields = TRACE_VERBOSITY_PRESETS[TRACE_VERBOSITY_DEFAULT]
    return set(fields)


def _kuramoto_fallback(G: TNFRGraph) -> tuple[float, float]:
    return 0.0, 0.0


kuramoto_R_psi: _KuramotoFn = cast(
    _KuramotoFn,
    cached_import("tnfr.gamma", "kuramoto_R_psi", fallback=_kuramoto_fallback),
)


def _sigma_fallback(G: TNFRGraph, _weight_mode: str | None = None) -> SigmaVector:
    """Return a null sigma vector regardless of ``_weight_mode``."""

    return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0}


# Public exports for this module
__all__ = (
    "CallbackSpec",
    "TraceFieldSpec",
    "TraceMetadata",
    "TraceSnapshot",
    "register_trace",
    "register_trace_field",
    "_callback_names",
    "gamma_field",
    "grammar_field",
)

# -------------------------
# Helpers
# -------------------------


def _trace_setup(
    G: TNFRGraph,
) -> tuple[
    Mapping[str, Any] | None,
    set[str],
    dict[str, Any] | None,
    str | None,
]:
    """Prepare common configuration for trace snapshots.

    Returns the active configuration, capture set, history and key under
    which metadata will be stored. If tracing is disabled returns
    ``(None, set(), None, None)``.
    """

    cfg_raw = G.graph.get("TRACE", TRACE)
    cfg = cfg_raw if isinstance(cfg_raw, Mapping) else TRACE
    if not cfg.get("enabled", True):
        return None, set(), None, None

    capture = _resolve_trace_capture(cfg)
    hist = ensure_history(G)
    key = cast(str | None, cfg.get("history_key", "trace_meta"))
    return cfg, capture, hist, key


def _callback_names(
    callbacks: Mapping[str, CallbackSpec] | Iterable[CallbackSpec],
) -> list[str]:
    """Return callback names from ``callbacks``."""
    if isinstance(callbacks, Mapping):
        callbacks = callbacks.values()
    return [
        cb.name if cb.name is not None else str(getattr(cb.func, "__name__", "fn"))
        for cb in callbacks
    ]


EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


def mapping_field(G: TNFRGraph, graph_key: str, out_key: str) -> TraceMetadata:
    """Copy mappings from ``G.graph`` into trace output."""
    mapping = get_graph_mapping(G, graph_key, f"G.graph[{graph_key!r}] is not a mapping; ignoring")
    if mapping is None:
        return {}
    return {out_key: mapping}


# -------------------------
# Builders
# -------------------------


def _new_trace_meta(
    G: TNFRGraph, phase: str
) -> tuple[TraceSnapshot, set[str], dict[str, Any] | None, str | None] | None:
    """Initialise trace metadata for a ``phase``.

    Wraps :func:`_trace_setup` and creates the base structure with timestamp
    and current phase. Returns ``None`` if tracing is disabled.
    """

    cfg, capture, hist, key = _trace_setup(G)
    if not cfg:
        return None

    meta: TraceSnapshot = {"t": float(G.graph.get("_t", 0.0)), "phase": phase}
    return meta, capture, hist, key


# -------------------------
# Snapshots
# -------------------------


def _trace_capture(G: TNFRGraph, phase: str, fields: TraceFieldMap) -> None:
    """Capture ``fields`` for ``phase`` and store the snapshot.

    A :class:`TraceSnapshot` is appended to the configured history when
    tracing is active. If there is no active history or storage key the
    capture is silently ignored.
    """

    res = _new_trace_meta(G, phase)
    if not res:
        return

    meta, capture, hist, key = res
    if not capture:
        return
    for name, getter in fields.items():
        if name in capture:
            meta.update(getter(G))
    if hist is None or key is None:
        return
    append_metric(hist, key, meta)


# -------------------------
# Registry
# -------------------------

TRACE_FIELDS: TraceFieldRegistry = {}


def register_trace_field(phase: str, name: str, func: TraceFieldFn) -> None:
    """Register ``func`` to populate trace field ``name`` during ``phase``."""

    TRACE_FIELDS.setdefault(phase, {})[name] = func


def gamma_field(G: TNFRGraph) -> TraceMetadata:
    """Expose γ-field metadata stored under ``G.graph['GAMMA']``."""

    return mapping_field(G, "GAMMA", "gamma")


def grammar_field(G: TNFRGraph) -> TraceMetadata:
    """Expose canonical grammar metadata for trace emission."""

    return mapping_field(G, "GRAMMAR_CANON", "grammar")


def dnfr_weights_field(G: TNFRGraph) -> TraceMetadata:
    return mapping_field(G, "DNFR_WEIGHTS", "dnfr_weights")


def selector_field(G: TNFRGraph) -> TraceMetadata:
    sel = G.graph.get("glyph_selector")
    selector_name = getattr(sel, "__name__", str(sel)) if sel else None
    return {"selector": selector_name}


def _si_weights_field(G: TNFRGraph) -> TraceMetadata:
    weights = mapping_field(G, "_Si_weights", "si_weights")
    if weights:
        return weights
    return {"si_weights": EMPTY_MAPPING}


def _si_sensitivity_field(G: TNFRGraph) -> TraceMetadata:
    mapping = get_graph_mapping(
        G,
        "_Si_sensitivity",
        "G.graph['_Si_sensitivity'] is not a mapping; ignoring",
    )
    if mapping is None:
        return {"si_sensitivity": EMPTY_MAPPING}

    normalised = _normalise_si_sensitivity_mapping(mapping, warn=True)

    if normalised != mapping:
        G.graph["_Si_sensitivity"] = normalised

    return {"si_sensitivity": MappingProxyType(normalised)}


def si_weights_field(G: TNFRGraph) -> TraceMetadata:
    """Return sense-plane weights and sensitivity."""

    weights = _si_weights_field(G)
    sensitivity = _si_sensitivity_field(G)
    return {**weights, **sensitivity}


def callbacks_field(G: TNFRGraph) -> TraceMetadata:
    cb = G.graph.get("callbacks")
    if not isinstance(cb, Mapping):
        return {}
    out: dict[str, list[str] | None] = {}
    for phase, cb_map in cb.items():
        if isinstance(cb_map, Mapping) or is_non_string_sequence(cb_map):
            out[phase] = _callback_names(cb_map)
        else:
            out[phase] = None
    return {"callbacks": out}


def thol_state_field(G: TNFRGraph) -> TraceMetadata:
    th_open = 0
    for _, nd in G.nodes(data=True):
        st = nd.get("_GRAM", {})
        if st.get("thol_open", False):
            th_open += 1
    return {"thol_open_nodes": th_open}


def kuramoto_field(G: TNFRGraph) -> TraceMetadata:
    R, psi = kuramoto_R_psi(G)
    return {"kuramoto": {"R": float(R), "psi": float(psi)}}


def sigma_field(G: TNFRGraph) -> TraceMetadata:
    sigma_vector_from_graph: _SigmaVectorFn = cast(
        _SigmaVectorFn,
        cached_import(
            "tnfr.sense",
            "sigma_vector_from_graph",
            fallback=_sigma_fallback,
        ),
    )
    sv = sigma_vector_from_graph(G)
    return {
        "sigma": {
            "x": float(sv.get("x", 0.0)),
            "y": float(sv.get("y", 0.0)),
            "mag": float(sv.get("mag", 0.0)),
            "angle": float(sv.get("angle", 0.0)),
        }
    }


def glyph_counts_field(G: TNFRGraph) -> TraceMetadata:
    """Return structural operator application count snapshot.

    Provides a snapshot of which structural operator symbols (glyphs) have been
    applied in the current step. ``count_glyphs`` already produces a fresh
    mapping so no additional copy is taken. Treat the returned mapping as read-only.
    """

    cnt = count_glyphs(G, window=1)
    return {"glyphs": cnt}


TRACE_FIELD_SPECS: tuple[TraceFieldSpec, ...] = (
    TraceFieldSpec(
        name="gamma",
        phase="before",
        producer=gamma_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="grammar",
        phase="before",
        producer=grammar_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="selector",
        phase="before",
        producer=selector_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="dnfr_weights",
        phase="before",
        producer=dnfr_weights_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="si_weights",
        phase="before",
        producer=si_weights_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="callbacks",
        phase="before",
        producer=callbacks_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="thol_open_nodes",
        phase="before",
        producer=thol_state_field,
        tiers=(
            TelemetryVerbosity.BASIC,
            TelemetryVerbosity.DETAILED,
            TelemetryVerbosity.DEBUG,
        ),
    ),
    TraceFieldSpec(
        name="kuramoto",
        phase="after",
        producer=kuramoto_field,
        tiers=(TelemetryVerbosity.DETAILED, TelemetryVerbosity.DEBUG),
    ),
    TraceFieldSpec(
        name="sigma",
        phase="after",
        producer=sigma_field,
        tiers=(TelemetryVerbosity.DETAILED, TelemetryVerbosity.DEBUG),
    ),
    TraceFieldSpec(
        name="glyph_counts",
        phase="after",
        producer=glyph_counts_field,
        tiers=(TelemetryVerbosity.DEBUG,),
    ),
)

TRACE_VERBOSITY_PRESETS = {
    level.value: tuple(spec.name for spec in TRACE_FIELD_SPECS if level in spec.tiers)
    for level in TelemetryVerbosity
}

for spec in TRACE_FIELD_SPECS:
    register_trace_field(spec.phase, spec.name, spec.producer)

# -------------------------
# API
# -------------------------


def register_trace(G: TNFRGraph) -> None:
    """Enable before/after-step snapshots and dump operational metadata to history.

    Trace snapshots are stored as :class:`TraceSnapshot` entries in
    ``G.graph['history'][TRACE.history_key]`` with:
      - gamma: active Γi(R) specification
      - grammar: canonical grammar configuration
      - selector: glyph selector name
      - dnfr_weights: ΔNFR mix declared in the engine
      - si_weights: α/β/γ weights and Si sensitivity
      - callbacks: callbacks registered per phase (if in
        ``G.graph['callbacks']``)
      - thol_open_nodes: how many nodes have an open THOL block
      - kuramoto: network ``(R, ψ)``
      - sigma: global sense-plane vector
      - glyphs: glyph counts after the step

    Field helpers reuse graph dictionaries and expect them to be treated as
    immutable snapshots by consumers.
    """
    if G.graph.get("_trace_registered"):
        return

    from .utils import callback_manager

    for phase in TRACE_FIELDS.keys():
        event = f"{phase}_step"

        def _make_cb(ph: str) -> TraceCallback:
            def _cb(graph: TNFRGraph, ctx: dict[str, Any]) -> None:
                del ctx

                _trace_capture(graph, ph, TRACE_FIELDS.get(ph, {}))

            return _cb

        callback_manager.register_callback(
            G, event=event, func=_make_cb(phase), name=f"trace_{phase}"
        )

    G.graph["_trace_registered"] = True
