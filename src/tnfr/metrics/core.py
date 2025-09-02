"""Métricas básicas."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Tuple
import heapq

from ..constants import METRIC_DEFAULTS, ALIAS_EPI, METRICS
from ..callback_utils import register_callback
from ..glyph_history import ensure_history, last_glifo
from ..helpers import get_attr
from ..constants_glifos import GLYPHS_CANONICAL, GLYPH_GROUPS
from .coherence import register_coherence_callbacks
from .diagnosis import register_diagnosis_callbacks


LATENT_GLYPH = "SHA"
TgCurr = "curr"
TgRun = "run"


# -------------
# Utilidades internas
# -------------


def _tg_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Estructura interna por nodo para acumular tiempos de corrida por glifo.
    Campos: curr (glifo actual), run (tiempo acumulado en el glifo actual)
    """
    return nd.setdefault("_Tg", {TgCurr: None, TgRun: 0.0})


# -------------
# Helpers de métricas
# -------------


def _update_tg(G, hist, dt, save_by_node: bool):
    """Acumula tiempos glíficos por nodo y devuelve conteos y latencia.

    ``save_by_node`` controla si se registran las corridas por nodo.
    """
    counts = Counter()
    n_total = 0
    n_latent = 0

    tg_total = hist.setdefault("Tg_total", defaultdict(float))
    tg_by_node = hist.setdefault("Tg_by_node", {})

    last = last_glifo
    tg_state = _tg_state
    latent = LATENT_GLYPH
    curr_key = TgCurr
    run_key = TgRun

    for n, nd in G.nodes(data=True):
        g = last(nd)
        if not g:
            continue

        n_total += 1
        if g == latent:
            n_latent += 1

        counts[g] += 1

        st = tg_state(nd)
        if st[curr_key] is None:
            st[curr_key] = g
            st[run_key] = dt
        elif g == st[curr_key]:
            st[run_key] += dt
        else:
            prev = st[curr_key]
            dur = float(st[run_key])
            tg_total[prev] += dur
            if save_by_node:
                rec = tg_by_node.setdefault(n, defaultdict(list))
                rec[prev].append(dur)
            st[curr_key] = g
            st[run_key] = dt

    return counts, n_total, n_latent


def _update_glifogram(G, hist, counts, t):
    """Registra el glifograma del paso a partir de los conteos."""
    normalize_series = bool(
        G.graph.get("METRICS", METRICS).get("normalize_series", False)
    )
    row = {"t": t}
    total = max(1, sum(counts.values()))
    for g in GLYPHS_CANONICAL:
        c = counts.get(g, 0)
        row[g] = (c / total) if normalize_series else c
    hist.setdefault("glifogram", []).append(row)


def _update_latency_index(G, hist, n_total, n_latent, t):
    """Añade el índice de latencia a la historia."""
    li = n_latent / max(1, n_total)
    hist.setdefault("latency_index", []).append({"t": t, "value": li})


def _update_epi_support(G, hist, t, thr):
    """Calcula soporte y norma de la EPI."""
    total = 0.0
    count = 0
    for n in G.nodes():
        epi_val = abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        if epi_val >= thr:
            total += epi_val
            count += 1
    epi_norm = (total / count) if count else 0.0
    hist.setdefault("EPI_support", []).append(
        {"t": t, "size": count, "epi_norm": float(epi_norm)}
    )


def _update_morph_metrics(G, hist, counts, t):
    """Registra métricas morfosintácticas basadas en conteos glíficos."""
    def get_count(keys):
        return sum(counts.get(k, 0) for k in keys)
    total = max(1, sum(counts.values()))
    id_val = get_count(GLYPH_GROUPS.get("ID", ())) / total
    cm_val = get_count(GLYPH_GROUPS.get("CM", ())) / total
    ne_val = get_count(GLYPH_GROUPS.get("NE", ())) / total
    num = get_count(GLYPH_GROUPS.get("PP_num", ()))
    den = get_count(GLYPH_GROUPS.get("PP_den", ()))
    pp_val = 0.0 if den == 0 else num / den
    hist.setdefault("morph", []).append(
        {"t": t, "ID": id_val, "CM": cm_val, "NE": ne_val, "PP": pp_val}
    )


# -------------
# Callback principal: actualizar métricas por paso
# -------------


def _metrics_step(G, *args, **kwargs):
    """Actualiza métricas operativas TNFR por paso.

    Orquesta la actualización del glifograma, índice de latencia, Tg,
    soporte EPI y métricas morfosintácticas. Todos los resultados se
    guardan en ``G.graph['history']``.
    """
    if not G.graph.get("METRICS", METRICS).get("enabled", True):
        return

    hist = ensure_history(G)
    dt = float(G.graph.get("DT", 1.0))
    t = float(G.graph.get("_t", 0.0))
    thr = float(G.graph.get("EPI_SUPPORT_THR", METRIC_DEFAULTS.get("EPI_SUPPORT_THR", 0.0)))

    save_by_node = bool(G.graph.get("METRICS", METRICS).get("save_by_node", True))
    counts, n_total, n_latent = _update_tg(G, hist, dt, save_by_node)
    _update_glifogram(G, hist, counts, t)
    _update_latency_index(G, hist, n_total, n_latent, t)
    _update_epi_support(G, hist, t, thr)
    _update_morph_metrics(G, hist, counts, t)


# -------------
# Registro del callback
# -------------


def register_metrics_callbacks(G) -> None:
    register_callback(G, event="after_step", func=_metrics_step, name="metrics_step")
    # Nuevas funcionalidades canónicas
    register_coherence_callbacks(G)
    register_diagnosis_callbacks(G)


# -------------
# Consultas / reportes
# -------------


def Tg_global(G, normalize: bool = True) -> Dict[str, float]:
    """Tiempo glífico total por clase. Si normalize=True, devuelve fracciones del total."""
    hist = ensure_history(G)
    tg_total: Dict[str, float] = hist.tracked_get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    if normalize:
        return {g: float(tg_total.get(g, 0.0)) / total for g in GLYPHS_CANONICAL}
    return {g: float(tg_total.get(g, 0.0)) for g in GLYPHS_CANONICAL}


def Tg_by_node(G, n, normalize: bool = False) -> Dict[str, float | List[float]]:
    """Resumen por nodo: si normalize, devuelve medias por glifo; si no, lista de corridas."""
    hist = ensure_history(G)
    rec = hist.tracked_get("Tg_by_node", {}).get(n, {})
    if not normalize:
        # convertir default dict → list para serializar
        return {g: list(rec.get(g, [])) for g in GLYPHS_CANONICAL}
    out = {}
    for g in GLYPHS_CANONICAL:
        runs = rec.get(g, [])
        out[g] = float(mean(runs)) if runs else 0.0

    return out


def latency_series(G) -> Dict[str, List[float]]:
    hist = ensure_history(G)
    xs = hist.tracked_get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glifogram_series(G) -> Dict[str, List[float]]:
    hist = ensure_history(G)
    xs = hist.tracked_get("glifogram", [])
    if not xs:
        return {"t": []}
    out = {"t": [float(x.get("t", i)) for i, x in enumerate(xs)]}
    for g in GLYPHS_CANONICAL:
        out[g] = [float(x.get(g, 0.0)) for x in xs]
    return out


def glyph_top(G, k: int = 3) -> List[Tuple[str, float]]:
    """Top-k operadores estructurales por Tg_global (fracción)."""
    tg = Tg_global(G, normalize=True)
    return heapq.nlargest(max(1, int(k)), tg.items(), key=lambda kv: kv[1])


def glyph_dwell_stats(G, n) -> Dict[str, Dict[str, float]]:
    """Estadísticos por nodo: mean/median/max de corridas por glifo."""
    hist = ensure_history(G)
    rec = hist.tracked_get("Tg_by_node", {}).get(n, {})
    out = {}
    for g in GLYPHS_CANONICAL:
        runs = list(rec.get(g, []))
        if not runs:
            out[g] = {"mean": 0.0, "median": 0.0, "max": 0.0, "count": 0}
        else:
            out[g] = {
                "mean": float(mean(runs)),
                "median": float(median(runs)),
                "max": float(max(runs)),
                "count": int(len(runs)),
            }
    return out
