"""Trace logging.

Field helpers avoid unnecessary copying by reusing dictionaries stored on
the graph whenever possible.  Callers are expected to treat returned
structures as immutable snapshots.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, NamedTuple
from collections.abc import Iterable, Mapping, Sequence

from .constants import TRACE
from .glyph_history import ensure_history, count_glyphs, append_metric
from .import_utils import optional_import
from .helpers.cache import get_graph_mapping


class _KuramotoFn(Protocol):
    def __call__(self, G: Any) -> tuple[float, float]: ...


class _SigmaVectorFn(Protocol):
    def __call__(
        self, G: Any, weight_mode: str | None = None
    ) -> dict[str, float]: ...


class CallbackSpec(NamedTuple):
    """Specification for a registered callback."""

    name: str | None
    func: Callable[..., Any]


def _kuramoto_fallback(G: Any) -> tuple[float, float]:
    return 0.0, 0.0


kuramoto_R_psi: _KuramotoFn = optional_import(
    "tnfr.gamma.kuramoto_R_psi", fallback=_kuramoto_fallback
)


def _sigma_fallback(
    G: Any, weight_mode: str | None = None
) -> dict[str, float]:
    return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0}


# Public exports for this module
__all__ = (
    "CallbackSpec",
    "register_trace",
    "_callback_names",
    "gamma_field",
    "grammar_field",
)

# -------------------------
# Helpers
# -------------------------


def _trace_setup(
    G,
) -> tuple[
    Optional[dict[str, Any]], set[str], Optional[dict[str, Any]], Optional[str]
]:
    """Common configuration for trace snapshots.

    Returns the active configuration, capture set, history and key under
    which metadata will be stored. If tracing is disabled returns
    ``(None, set(), None, None)``.
    """

    cfg = G.graph.get("TRACE", TRACE)
    if not cfg.get("enabled", True):
        return None, set(), None, None

    capture: set[str] = set(cfg.get("capture", []))
    hist = ensure_history(G)
    key = cfg.get("history_key", "trace_meta")
    return cfg, capture, hist, key


def _callback_names(
    callbacks: Mapping[str, CallbackSpec] | Iterable[CallbackSpec],
) -> list[str]:
    """Return callback names from ``callbacks``."""
    if isinstance(callbacks, Mapping):
        callbacks = callbacks.values()
    return [cb.name or getattr(cb.func, "__name__", "fn") for cb in callbacks]


def mapping_field(G, graph_key: str, out_key: str) -> dict[str, Any]:
    """Helper to copy mappings from ``G.graph`` into trace output."""
    mapping = get_graph_mapping(
        G, graph_key, f"G.graph[{graph_key!r}] no es un mapeo; se ignora"
    )
    return {out_key: mapping} if mapping is not None else {}


def make_mapping_field(
    graph_key: str, out_key: str
) -> Callable[[Any], dict[str, Any]]:
    """Return a field function reading ``graph_key`` into ``out_key``."""

    def field(G):
        return mapping_field(G, graph_key, out_key)

    return field


# -------------------------
# Builders
# -------------------------


def _new_trace_meta(
    G, phase: str
) -> Optional[
    tuple[dict[str, Any], set[str], Optional[dict[str, Any]], Optional[str]]
]:
    """Initialise trace metadata for a ``phase``.

    Wraps :func:`_trace_setup` and creates the base structure with timestamp
    and current phase. Returns ``None`` if tracing is disabled.
    """

    cfg, capture, hist, key = _trace_setup(G)
    if not cfg:
        return None

    meta: dict[str, Any] = {"t": float(G.graph.get("_t", 0.0)), "phase": phase}
    return meta, capture, hist, key


# -------------------------
# Snapshots
# -------------------------


def _trace_capture(
    G, phase: str, fields: dict[str, Callable[[Any], dict[str, Any]]]
) -> None:
    """Capture ``fields`` for a ``phase`` and store the snapshot.

    If there is no active history or storage key the capture is silently
    ignored.
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


gamma_field = make_mapping_field("GAMMA", "gamma")


grammar_field = make_mapping_field("GRAMMAR_CANON", "grammar")


dnfr_weights_field = make_mapping_field("DNFR_WEIGHTS", "dnfr_weights")


def selector_field(G):
    sel = G.graph.get("glyph_selector")
    return {"selector": getattr(sel, "__name__", str(sel)) if sel else None}


_si_weights_field = make_mapping_field("_Si_weights", "si_weights")


_si_sensitivity_field = make_mapping_field("_Si_sensitivity", "si_sensitivity")


def si_weights_field(G):
    """Return sense-plane weights and sensitivity."""

    return {
        **(_si_weights_field(G) or {"si_weights": {}}),
        **(_si_sensitivity_field(G) or {"si_sensitivity": {}}),
    }


def callbacks_field(G):
    cb = G.graph.get("callbacks")
    if not isinstance(cb, Mapping):
        return {}
    out = {}
    for phase, cb_map in cb.items():
        if isinstance(cb_map, Mapping):
            out[phase] = _callback_names(cb_map)
        elif isinstance(cb_map, Sequence) and not isinstance(
            cb_map, (str, bytes)
        ):
            out[phase] = _callback_names(cb_map)
        else:
            out[phase] = None
    return {"callbacks": out}


def thol_state_field(G):
    th_open = 0
    for _, nd in G.nodes(data=True):
        st = nd.get("_GRAM", {})
        if st.get("thol_open", False):
            th_open += 1
    return {"thol_open_nodes": th_open}


def kuramoto_field(G):
    R, psi = kuramoto_R_psi(G)
    return {"kuramoto": {"R": float(R), "psi": float(psi)}}


def sigma_field(G):
    sigma_vector_from_graph: _SigmaVectorFn = optional_import(
        "tnfr.sense.sigma_vector_from_graph", fallback=_sigma_fallback
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


def glyph_counts_field(G):
    """Return glyph count snapshot.

    ``count_glyphs`` already produces a fresh mapping so no additional copy
    is taken.  Treat the returned mapping as read-only.
    """

    cnt = count_glyphs(G, window=1)
    return {"glyphs": cnt}


TRACE_FIELDS_BEFORE = {
    "gamma": gamma_field,
    "grammar": grammar_field,
    "selector": selector_field,
    "dnfr_weights": dnfr_weights_field,
    "si_weights": si_weights_field,
    "callbacks": callbacks_field,
    "thol_state": thol_state_field,
}


TRACE_FIELDS_AFTER = {
    "kuramoto": kuramoto_field,
    "sigma": sigma_field,
    "glyph_counts": glyph_counts_field,
}


def _trace_before(G, *args, **kwargs):
    _trace_capture(G, "before", TRACE_FIELDS_BEFORE)


def _trace_after(G, *args, **kwargs):
    _trace_capture(G, "after", TRACE_FIELDS_AFTER)


# -------------------------
# API
# -------------------------


def register_trace(G) -> None:
    """Enable before/after-step snapshots and dump operational metadata
    to history.

    Stores in ``G.graph['history'][TRACE.history_key]`` a list of
    entries ``{'phase': 'before'|'after', ...}`` with:
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

    from .callback_utils import register_callback

    register_callback(
        G, event="before_step", func=_trace_before, name="trace_before"
    )
    register_callback(
        G, event="after_step", func=_trace_after, name="trace_after"
    )

    G.graph["_trace_registered"] = True
