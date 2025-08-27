from __future__ import annotations
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
import statistics
import csv
import json

from .constants import DEFAULTS
from .helpers import register_callback, ensure_history, last_glifo
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


# -------------
# Estado nodal para Tg
# -------------

def _tg_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Estructura interna por nodo para acumular tiempos de corrida por glifo.
    Campos: curr (glifo actual), run (tiempo acumulado en el glifo actual)
    """
    return nd.setdefault("_Tg", {"curr": None, "run": 0.0})


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

    hist = ensure_history(G)
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
        g = last_glifo(nd)
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
    hist = ensure_history(G)
    tg_total: Dict[str, float] = hist.get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    if normalize:
        return {g: float(tg_total.get(g, 0.0)) / total for g in GLYPHS_CANONICAL}
    return {g: float(tg_total.get(g, 0.0)) for g in GLYPHS_CANONICAL}


def Tg_by_node(G, n, normalize: bool = False) -> Dict[str, float | List[float]]:
    """Resumen por nodo: si normalize, devuelve medias por glifo; si no, lista de corridas."""
    hist = ensure_history(G)
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
    hist = ensure_history(G)
    xs = hist.get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glifogram_series(G) -> Dict[str, List[float]]:
    hist = ensure_history(G)
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
    hist = ensure_history(G)
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


# -----------------------------
# Export history to CSV/JSON
# -----------------------------

def export_history(G, base_path: str, fmt: str = "csv") -> None:
    """Vuelca glifograma y traza σ(t) a archivos CSV o JSON compactos."""
    hist = ensure_history(G)
    glifo = glifogram_series(G)
    sigma_mag = hist.get("sense_sigma_mag", [])
    sigma = {
        "t": list(range(len(sigma_mag))),
        "sigma_x": hist.get("sense_sigma_x", []),
        "sigma_y": hist.get("sense_sigma_y", []),
        "mag": sigma_mag,
        "angle": hist.get("sense_sigma_angle", []),
    }
    fmt = fmt.lower()
    if fmt == "csv":
        with open(base_path + "_glifogram.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", *GLYPHS_CANONICAL])
            ts = glifo.get("t", [])
            default_col = [0] * len(ts)
            for i, t in enumerate(ts):
                row = [t] + [glifo.get(g, default_col)[i] for g in GLYPHS_CANONICAL]
                writer.writerow(row)
        with open(base_path + "_sigma.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "mag", "angle"])
            for i, t in enumerate(sigma["t"]):
                writer.writerow([t, sigma["sigma_x"][i], sigma["sigma_y"][i], sigma["mag"][i], sigma["angle"][i]])
    else:
        data = {"glifogram": glifo, "sigma": sigma}
        with open(base_path + ".json", "w") as f:
            json.dump(data, f)
