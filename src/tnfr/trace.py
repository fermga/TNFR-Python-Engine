"""Trace logging."""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Protocol
import warnings


class _KuramotoFn(Protocol):
    def __call__(self, G: Any) -> tuple[float, float]:
        ...


class _SigmaVectorFn(Protocol):
    def __call__(
        self, G: Any, weight_mode: str | None = None
    ) -> Dict[str, float]:
        ...

from .constants import TRACE
from .callback_utils import register_callback
from .glyph_history import ensure_history, count_glyphs

try:
    from .gamma import kuramoto_R_psi as _kuramoto_R_psi
except ImportError:  # pragma: no cover
    def _kuramoto_R_psi(G: Any) -> tuple[float, float]:
        return 0.0, 0.0

kuramoto_R_psi: _KuramotoFn = _kuramoto_R_psi

try:
    from .sense import sigma_vector_from_graph as _sigma_vector_from_graph
except ImportError:  # pragma: no cover
    def _sigma_vector_from_graph(
        G: Any, weight_mode: str | None = None
    ) -> Dict[str, float]:
        return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0.0}

sigma_vector_from_graph: _SigmaVectorFn = _sigma_vector_from_graph

# -------------------------
# Helpers
# -------------------------

def _trace_setup(G) -> tuple[Optional[Dict[str, Any]], List[str], Optional[Dict[str, Any]], Optional[str]]:
    """Common configuration for trace snapshots.

    Returns the active configuration, capture list, history and key under
    which metadata will be stored. If tracing is disabled returns
    ``(None, [], None, None)``.
    """

    cfg = G.graph.get("TRACE", TRACE)
    if not cfg.get("enabled", True):
        return None, [], None, None

    capture: List[str] = list(cfg.get("capture", []))
    hist = ensure_history(G)
    key = cfg.get("history_key", "trace_meta")
    return cfg, capture, hist, key


def _callback_names(callbacks: list) -> list[str]:
    names: list[str] = []
    for item in callbacks:
        if isinstance(item, tuple):
            if not item:
                # skip empty tuples
                continue
            first = item[0]
            if isinstance(first, str):
                name = first
            else:
                # no explicit name, fall back to the function's name
                func = first if callable(first) else (item[1] if len(item) > 1 else None)
                name = getattr(func, "__name__", "fn")
        else:
            name = getattr(item, "__name__", "fn")
        names.append(name)
    return names

# -------------------------
# Builders
# -------------------------

def _new_trace_meta(
    G, phase: str
) -> Optional[tuple[Dict[str, Any], List[str], Optional[Dict[str, Any]], Optional[str]]]:
    """Initialise trace metadata for a ``phase``.

    Wraps :func:`_trace_setup` and creates the base structure with timestamp
    and current phase. Returns ``None`` if tracing is disabled.
    """

    cfg, capture, hist, key = _trace_setup(G)
    if not cfg:
        return None

    meta: Dict[str, Any] = {"t": float(G.graph.get("_t", 0.0)), "phase": phase}
    return meta, capture, hist, key

# -------------------------
# Snapshots
# -------------------------

def _trace_capture(
    G, phase: str, fields: Dict[str, Callable[[Any], Dict[str, Any]]]
) -> None:
    """Capture ``fields`` for a ``phase`` and store the snapshot.

    If there is no active history or storage key the capture is silently
    ignored.
    """

    res = _new_trace_meta(G, phase)
    if not res:
        return

    meta, capture, hist, key = res
    for name, getter in fields.items():
        if name in capture:
            meta.update(getter(G))
    if hist is None or key is None:
        return
    hist.setdefault(key, []).append(meta)


def gamma_field(G):
    gam = G.graph.get("GAMMA", {})
    if not isinstance(gam, dict):
        if gam is not None:
            warnings.warn(
                "G.graph['GAMMA'] no es un mapeo; se ignora",
                UserWarning,
                stacklevel=2,
            )
        return {}
    return {"gamma": dict(gam)}


def grammar_field(G):
    gram = G.graph.get("GRAMMAR_CANON", {})
    if not isinstance(gram, dict):
        if gram is not None:
            warnings.warn(
                "G.graph['GRAMMAR_CANON'] no es un mapeo; se ignora",
                UserWarning,
                stacklevel=2,
            )
        return {}
    return {"grammar": dict(gram)}


def selector_field(G):
    sel = G.graph.get("glyph_selector")
    return {"selector": getattr(sel, "__name__", str(sel)) if sel else None}


def dnfr_weights_field(G):
    mix = G.graph.get("DNFR_WEIGHTS")
    return {"dnfr_weights": dict(mix)} if isinstance(mix, dict) else {}


def si_weights_field(G):
    return {
        "si_weights": dict(G.graph.get("_Si_weights", {})),
        "si_sensitivity": dict(G.graph.get("_Si_sensitivity", {})),
    }


def callbacks_field(G):
    cb = G.graph.get("callbacks")
    if not isinstance(cb, dict):
        return {}
    out = {
        phase: _callback_names(cb_list) if isinstance(cb_list, list) else None
        for phase, cb_list in cb.items()
    }
    return {"callbacks": out}


def thol_state_field(G):
    th_open = 0
    for n in G.nodes():
        st = G.nodes[n].get("_GRAM", {})
        if st.get("thol_open", False):
            th_open += 1
    return {"thol_open_nodes": th_open}


def kuramoto_field(G):
    R, psi = kuramoto_R_psi(G)
    return {"kuramoto": {"R": float(R), "psi": float(psi)}}


def sigma_field(G):
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
    cnt = count_glyphs(G, window=1)
    return {"glyphs": dict(cnt)}


def _trace_before(G, *args, **kwargs):
    fields = {
        "gamma": gamma_field,
        "grammar": grammar_field,
        "selector": selector_field,
        "dnfr_weights": dnfr_weights_field,
        "si_weights": si_weights_field,
        "callbacks": callbacks_field,
        "thol_state": thol_state_field,
    }
    _trace_capture(G, "before", fields)


def _trace_after(G, *args, **kwargs):
    fields = {
        "kuramoto": kuramoto_field,
        "sigma": sigma_field,
        "glyph_counts": glyph_counts_field,
    }
    _trace_capture(G, "after", fields)


# -------------------------
# API
# -------------------------

def register_trace(G) -> None:
    """Enable before/after-step snapshots and dump operational metadata to history.

    Stores in ``G.graph['history'][TRACE.history_key]`` a list of entries
    ``{'phase': 'before'|'after', ...}`` with:
      - gamma: active Γi(R) specification
      - grammar: canonical grammar configuration
      - selector: glyph selector name
      - dnfr_weights: ΔNFR mix declared in the engine
      - si_weights: α/β/γ weights and Si sensitivity
      - callbacks: callbacks registered per phase (if in ``G.graph['callbacks']``)
      - thol_open_nodes: how many nodes have an open THOL block
      - kuramoto: network ``(R, ψ)``
      - sigma: global sense-plane vector
      - glyphs: glyph counts after the step
    """
    if G.graph.get("_trace_registered"):
        return

    register_callback(G, event="before_step", func=_trace_before, name="trace_before")
    register_callback(G, event="after_step", func=_trace_after, name="trace_after")

    G.graph["_trace_registered"] = True
