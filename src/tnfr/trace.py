"""Registro de trazas."""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from .constants import TRACE
from .helpers import register_callback, ensure_history, count_glyphs

try:
    from .gamma import kuramoto_R_psi
except ImportError:  # pragma: no cover
    def kuramoto_R_psi(G):
        return 0.0, 0.0

try:
    from .sense import sigma_vector_from_graph
except ImportError:  # pragma: no cover
    def sigma_vector_from_graph(G, *args, **kwargs):
        return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0}

# -------------------------
# Helpers
# -------------------------

def _trace_setup(G) -> tuple[Optional[Dict[str, Any]], List[str], Optional[Dict[str, Any]], Optional[str]]:
    """Configuración común para los snapshots de trazas.

    Retorna la configuración activa, la lista de capturas, el history y la
    clave bajo la que se guardará la metadata. Si el tracing está deshabilitado
    retorna ``(None, [], None, None)``.
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
            name = item[0]
            if not isinstance(name, str):
                func = item[1] if len(item) > 1 else None
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
    """Inicializa la metadata de trace para una ``phase``.

    Envuelve :func:`_trace_setup` y crea la estructura base con la
    marca temporal y la fase actual. Si el tracing está deshabilitado
    retorna ``None``.
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
    """Captura ``fields`` para una ``phase`` y guarda el snapshot."""

    res = _new_trace_meta(G, phase)
    if not res:
        return

    meta, capture, hist, key = res
    for name, getter in fields.items():
        if name in capture:
            meta.update(getter(G))
    hist.setdefault(key, []).append(meta)


def gamma_field(G):
    return {"gamma": dict(G.graph.get("GAMMA", {}))}


def grammar_field(G):
    return {"grammar": dict(G.graph.get("GRAMMAR_CANON", {}))}


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


def glifo_counts_field(G):
    cnt = count_glyphs(G, window=1)
    return {"glifos": dict(cnt)}


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
        "glifo_counts": glifo_counts_field,
    }
    _trace_capture(G, "after", fields)


# -------------------------
# API
# -------------------------

def register_trace(G) -> None:
    """Activa snapshots before/after step y vuelca metadatos operativos en history.

    Guarda en G.graph['history'][TRACE.history_key] una lista de entradas {'phase': 'before'|'after', ...} con:
      - gamma: especificación activa de Γi(R)
      - grammar: configuración de gramática canónica
      - selector: nombre del selector glífico
      - dnfr_weights: mezcla ΔNFR declarada en el motor
      - si_weights: pesos α/β/γ y sensibilidad de Si
      - callbacks: callbacks registrados por fase (si están en G.graph['callbacks'])
      - thol_open_nodes: cuántos nodos tienen bloque THOL abierto
      - kuramoto: (R, ψ) de la red
      - sigma: vector global del plano del sentido
      - glifos: conteos por glifo tras el paso
    """
    if G.graph.get("_trace_registered"):
        return

    register_callback(G, event="before_step", func=_trace_before, name="trace_before")
    register_callback(G, event="after_step", func=_trace_after, name="trace_after")

    G.graph["_trace_registered"] = True
