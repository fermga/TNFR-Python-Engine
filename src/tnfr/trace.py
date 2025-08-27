from __future__ import annotations
from typing import Any, Dict, List, Optional
from collections import Counter

from .constants import DEFAULTS
from .helpers import register_callback, ensure_history, last_glifo

try:
    from .gamma import kuramoto_R_psi
except Exception:  # pragma: no cover
    def kuramoto_R_psi(G):
        return 0.0, 0.0

try:
    from .sense import sigma_vector_global
except Exception:  # pragma: no cover
    def sigma_vector_global(G, *args, **kwargs):
        return {"x": 1.0, "y": 0.0, "mag": 1.0, "angle": 0.0, "n": 0}

# -------------------------
# Defaults
# -------------------------
DEFAULTS.setdefault("TRACE", {
    "enabled": True,
    "capture": ["gamma", "grammar", "selector", "dnfr_weights", "si_weights", "callbacks", "thol_state", "sigma", "kuramoto", "glifo_counts"],
    "history_key": "trace_meta",
})

# -------------------------
# Helpers
# -------------------------

# -------------------------
# Snapshots
# -------------------------

def _trace_before(G, *args, **kwargs):
    if not G.graph.get("TRACE", DEFAULTS["TRACE"]).get("enabled", True):
        return
    cfg = G.graph.get("TRACE", DEFAULTS["TRACE"])
    capture: List[str] = list(cfg.get("capture", []))
    hist = ensure_history(G)
    key = cfg.get("history_key", "trace_meta")

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
        cb = G.graph.get("_callbacks")
        if isinstance(cb, dict):
            out = {k: [getattr(f, "__name__", "fn") for (_, f, *_rest) in v] if isinstance(v, list) else None for k, v in cb.items()}
            meta["callbacks"] = out

    if "thol_state" in capture:
        # cuántos nodos tienen bloque T’HOL abierto
        th_open = 0
        for n in G.nodes():
            st = G.nodes[n].get("_GRAM", {})
            if st.get("thol_open", False):
                th_open += 1
        meta["thol_open_nodes"] = th_open

    hist.setdefault(key, []).append(meta)


def _trace_after(G, *args, **kwargs):
    if not G.graph.get("TRACE", DEFAULTS["TRACE"]).get("enabled", True):
        return
    cfg = G.graph.get("TRACE", DEFAULTS["TRACE"])
    capture: List[str] = list(cfg.get("capture", []))
    hist = ensure_history(G)
    key = cfg.get("history_key", "trace_meta")

    meta: Dict[str, Any] = {"t": float(G.graph.get("_t", 0.0)), "phase": "after"}

    if "kuramoto" in capture:
        R, psi = kuramoto_R_psi(G)
        meta["kuramoto"] = {"R": float(R), "psi": float(psi)}

    if "sigma" in capture:
        sv = sigma_vector_global(G)
        meta["sigma"] = {"x": float(sv.get("x", 1.0)), "y": float(sv.get("y", 0.0)), "mag": float(sv.get("mag", 1.0)), "angle": float(sv.get("angle", 0.0))}

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
      - callbacks: callbacks registrados por fase (si están en G.graph['_callbacks'])
      - thol_open_nodes: cuántos nodos tienen bloque T’HOL abierto
      - kuramoto: (R, ψ) de la red
      - sigma: vector global del plano del sentido
      - glifos: conteos por glifo tras el paso
    """
    register_callback(G, when="before_step", func=_trace_before, name="trace_before")
    register_callback(G, when="after_step", func=_trace_after, name="trace_after")
