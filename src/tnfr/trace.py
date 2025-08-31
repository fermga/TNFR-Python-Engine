from __future__ import annotations
from typing import Any, Dict, List, Optional
from collections import Counter

from .constants import TRACE
from .helpers import register_callback, ensure_history, last_glifo

try:
    from .gamma import kuramoto_R_psi
except ImportError:  # pragma: no cover
    def kuramoto_R_psi(G):
        return 0.0, 0.0

try:
    from .sense import sigma_vector_global
except ImportError:  # pragma: no cover
    def sigma_vector_global(G, *args, **kwargs):
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

# -------------------------
# Snapshots
# -------------------------

def _trace_before(G, *args, **kwargs):
    cfg, capture, hist, key = _trace_setup(G)
    if not cfg:
        return

    meta: Dict[str, Any] = {"t": float(G.graph.get("_t", 0.0)), "phase": "before"}

    if "gamma" in capture:
        meta["gamma"] = dict(G.graph.get("GAMMA", {}))

    if "grammar" in capture:
        meta["grammar"] = dict(G.graph.get("GRAMMAR_CANON", {}))

    if "selector" in capture:
        sel = G.graph.get("glyph_selector")
        meta["selector"] = getattr(sel, "__name__", str(sel)) if sel else None

    if "dnfr_weights" in capture:
        mix = G.graph.get("DNFR_WEIGHTS")
        if isinstance(mix, dict):
            meta["dnfr_weights"] = dict(mix)

    if "si_weights" in capture:
        meta["si_weights"] = dict(G.graph.get("_Si_weights", {}))
        meta["si_sensitivity"] = dict(G.graph.get("_Si_sensitivity", {}))

    if "callbacks" in capture:
        # si el motor guarda los callbacks, exponer nombres por fase
        cb = G.graph.get("callbacks")
        if isinstance(cb, dict):
            out = {}
            for phase, callbacks in cb.items():
                if isinstance(callbacks, list):
                    names: List[str] = []
                    for item in callbacks:
                        if isinstance(item, tuple):
                            name = item[0]
                            if not isinstance(name, str):
                                func = item[1] if len(item) > 1 else None
                                name = getattr(func, "__name__", "fn")
                        else:
                            name = getattr(item, "__name__", "fn")
                        names.append(name)
                    out[phase] = names
                else:
                    out[phase] = None
            meta["callbacks"] = out

    if "thol_state" in capture:
        # cuántos nodos tienen bloque THOL abierto
        th_open = 0
        for n in G.nodes():
            st = G.nodes[n].get("_GRAM", {})
            if st.get("thol_open", False):
                th_open += 1
        meta["thol_open_nodes"] = th_open

    hist.setdefault(key, []).append(meta)


def _trace_after(G, *args, **kwargs):
    cfg, capture, hist, key = _trace_setup(G)
    if not cfg:
        return

    meta: Dict[str, Any] = {"t": float(G.graph.get("_t", 0.0)), "phase": "after"}

    if "kuramoto" in capture:
        R, psi = kuramoto_R_psi(G)
        meta["kuramoto"] = {"R": float(R), "psi": float(psi)}

    if "sigma" in capture:
        sv = sigma_vector_global(G)
        meta["sigma"] = {
            "x": float(sv.get("x", 0.0)),
            "y": float(sv.get("y", 0.0)),
            "mag": float(sv.get("mag", 0.0)),
            "angle": float(sv.get("angle", 0.0)),
        }

    if "glifo_counts" in capture:
        cnt = Counter()
        for n in G.nodes():
            g = last_glifo(G.nodes[n])
            if g:
                cnt[g] += 1
        meta["glifos"] = dict(cnt)

    hist.setdefault(key, []).append(meta)


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

    register_callback(G, when="before_step", func=_trace_before, name="trace_before")
    register_callback(G, when="after_step", func=_trace_after, name="trace_after")

    G.graph["_trace_registered"] = True
