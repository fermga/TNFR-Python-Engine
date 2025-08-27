from __future__ import annotations
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
import statistics

from .constants import DEFAULTS
from .helpers import _get_attr, clamp01, register_callback
from .sense import GLYPHS_CANONICAL

# -------------
# DEFAULTS
# -------------
DEFAULTS.setdefault("METRICS", {
    "enabled": True,
    "save_by_node": True,     # guarda Tg por nodo (más pesado)
    "normalize_series": False # glifograma normalizado a fracción por paso
})

# -------------
# Utilidades internas
# -------------

def _ensure_history(G):
    if "history" not in G.graph:
        G.graph["history"] = {}
    return G.graph["history"]


def _last_glifo(nd: Dict[str, Any]) -> str | None:
    hist = nd.get("hist_glifos")
    if not hist:
        return None
    try:
        return list(hist)[-1]
    except Exception:
        return None


# -------------
# Estado nodal para Tg
# -------------

def _tg_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Estructura interna por nodo para acumular tiempos de corrida por glifo.
    Campos: curr (glifo actual), run (tiempo acumulado en el glifo actual)
    """
    st = nd.setdefault("_Tg", {"curr": None, "run": 0.0})
    st.setdefault("curr", None)
    st.setdefault("run", 0.0)
    return st


# -------------
# Callback principal: actualizar métricas por paso
# -------------

def _metrics_step(G, *args, **kwargs):
    """Actualiza métricas operativas TNFR por paso.

    - Tg (tiempo glífico): sumatoria de corridas por glifo (global y por nodo).
    - Índice de latencia: fracción de nodos en SH’A.
    - Glifograma: conteo o fracción por glifo en el paso.

    Todos los resultados se guardan en G.graph['history'].
    """
    if not G.graph.get("METRICS", DEFAULTS.get("METRICS", {})).get("enabled", True):
        return

    hist = _ensure_history(G)
    dt = float(G.graph.get("DT", 1.0))
    t = float(G.graph.get("_t", 0.0))

    # --- Glifograma (conteos por glifo este paso) ---
    counts = Counter()

    # --- Índice de latencia: proporción de nodos en SH’A ---
    n_total = 0
    n_latent = 0

    # --- Tg: acumular corridas por nodo ---
    save_by_node = bool(G.graph.get("METRICS", DEFAULTS["METRICS"]).get("save_by_node", True))
    tg_total = hist.setdefault("Tg_total", defaultdict(float))  # tiempo total por glifo (global)
    tg_by_node = hist.setdefault("Tg_by_node", {})             # nodo → {glifo: [runs,...]}

    for n in G.nodes():
        nd = G.nodes[n]
        g = _last_glifo(nd)
        if not g:
            continue

        n_total += 1
        if g == "SH’A":
            n_latent += 1

        counts[g] += 1

        st = _tg_state(nd)
        # Si seguimos en el mismo glifo, acumulamos; si cambiamos, cerramos corrida
        if st["curr"] is None:
            st["curr"] = g
            st["run"] = dt
        elif g == st["curr"]:
            st["run"] += dt
        else:
            # cerramos corrida anterior
            prev = st["curr"]
            dur = float(st["run"])
            tg_total[prev] += dur
            if save_by_node:
                rec = tg_by_node.setdefault(n, defaultdict(list))
                rec[prev].append(dur)
            # reiniciamos corrida
            st["curr"] = g
            st["run"] = dt

    # Al final del paso, no cerramos la corrida actual: se cerrará cuando cambie.

    # Guardar glifograma (conteos crudos y normalizados)
    norm = bool(G.graph.get("METRICS", DEFAULTS["METRICS"]).get("normalize_series", False))
    row = {"t": t}
    total = max(1, sum(counts.values()))
    for g in GLYPHS_CANONICAL:
        c = counts.get(g, 0)
        row[g] = (c / total) if norm else c
    hist.setdefault("glifogram", []).append(row)

    # Guardar índice de latencia
    li = (n_latent / max(1, n_total)) if n_total else 0.0
    hist.setdefault("latency_index", []).append({"t": t, "value": li})


# -------------
# Registro del callback
# -------------

def register_metrics_callbacks(G) -> None:
    register_callback(G, when="after_step", func=_metrics_step, name="metrics_step")


# -------------
# Consultas / reportes
# -------------

def Tg_global(G, normalize: bool = True) -> Dict[str, float]:
    """Tiempo glífico total por clase. Si normalize=True, devuelve fracciones del total."""
    hist = _ensure_history(G)
    tg_total: Dict[str, float] = hist.get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    if normalize:
        return {g: float(tg_total.get(g, 0.0)) / total for g in GLYPHS_CANONICAL}
    return {g: float(tg_total.get(g, 0.0)) for g in GLYPHS_CANONICAL}


def Tg_by_node(G, n, normalize: bool = False) -> Dict[str, float | List[float]]:
    """Resumen por nodo: si normalize, devuelve medias por glifo; si no, lista de corridas."""
    hist = _ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    if not normalize:
        # convertir default dict → list para serializar
        return {g: list(rec.get(g, [])) for g in GLYPHS_CANONICAL}
    out = {}
    for g in GLYPHS_CANONICAL:
        runs = rec.get(g, [])
        out[g] = float(statistics.mean(runs)) if runs else 0.0
        
    return out


def latency_series(G) -> Dict[str, List[float]]:
    hist = _ensure_history(G)
    xs = hist.get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glifogram_series(G) -> Dict[str, List[float]]:
    hist = _ensure_history(G)
    xs = hist.get("glifogram", [])
    if not xs:
        return {"t": []}
    out = {"t": [float(x.get("t", i)) for i, x in enumerate(xs)]}
    for g in GLYPHS_CANONICAL:
        out[g] = [float(x.get(g, 0.0)) for x in xs]
    return out


def glyph_top(G, k: int = 3) -> List[Tuple[str, float]]:
    """Top-k glifos por Tg_global (fracción)."""
    tg = Tg_global(G, normalize=True)
    return sorted(tg.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(k))]


def glyph_dwell_stats(G, n) -> Dict[str, Dict[str, float]]:
    """Estadísticos por nodo: mean/median/max de corridas por glifo."""
    hist = _ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    out = {}
    for g in GLYPHS_CANONICAL:
        runs = list(rec.get(g, []))
        if not runs:
            out[g] = {"mean": 0.0, "median": 0.0, "max": 0.0, "count": 0}
        else:
            out[g] = {
                "mean": float(statistics.mean(runs)),
                "median": float(statistics.median(runs)),
                "max": float(max(runs)),
                "count": int(len(runs)),
            }
    return out
