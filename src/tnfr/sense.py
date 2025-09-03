"""Sense calculations."""
from __future__ import annotations
from typing import Dict, Iterable, List
import math
import warnings
from collections import Counter

import networkx as nx

from .constants import ALIAS_SI, ALIAS_EPI, SIGMA
from .helpers import (
    get_attr,
    clamp01,
)
from .callback_utils import register_callback
from .glyph_history import ensure_history, last_glyph, count_glyphs
from .constants_glyphs import (
    ANGLE_MAP,
    ESTABILIZADORES,
    DISRUPTIVOS,
    GLYPHS_CANONICAL,
    GLYPHS_CANONICAL_SET,
)

# -------------------------
# Canon: orden circular de glyphs y ángulos
# -------------------------

# Glyphs relevantes para el plano Σ de observadores de sentido
SIGMA_ANGLE_KEYS: tuple[str, ...] = tuple(ESTABILIZADORES + DISRUPTIVOS)

GLYPH_UNITS: Dict[str, complex] = {
    g: complex(math.cos(a), math.sin(a)) for g, a in ANGLE_MAP.items()
}

# -------------------------
# Utilidades básicas
# -------------------------

def glyph_angle(g: str) -> float:
    return float(ANGLE_MAP.get(g, 0.0))


def glyph_unit(g: str) -> complex:
    return GLYPH_UNITS.get(g, 1+0j)


def _weight(nd, mode: str) -> float:
    if mode == "Si":
        return clamp01(get_attr(nd, ALIAS_SI, 0.5))
    if mode == "EPI":
        return max(0.0, get_attr(nd, ALIAS_EPI, 0.0))
    return 1.0


def _node_weight(nd, weight_mode: str) -> tuple[str, float, complex] | None:
    """Return ``(glyph, weight, weighted_unit)`` or ``None`` if no glyph."""
    g = last_glyph(nd)
    if not g:
        return None
    w = _weight(nd, weight_mode)
    z = glyph_unit(g) * w  # precompute weighted unit vector
    return g, w, z


def _sigma_cfg(G):
    return G.graph.get("SIGMA", SIGMA)


# -------------------------
# σ por nodo y σ global
# -------------------------


def _sigma_from_vectors(
    vectors: Iterable[complex], fallback_angle: float = 0.0
) -> tuple[Dict[str, float], int]:
    """Normalise a series of complex vectors in the σ-plane."""

    acc = complex(0.0, 0.0)
    cnt = 0
    for z in vectors:
        acc += z
        cnt += 1

    if cnt <= 0:
        vec = {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": float(fallback_angle)}
        return vec, 0

    x, y = acc.real / cnt, acc.imag / cnt
    mag = math.hypot(x, y)
    ang = math.atan2(y, x) if mag > 0 else float(fallback_angle)
    vec = {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang)}
    return vec, cnt


def _sigma_from_pairs(
    pairs: Iterable[tuple[str, float]], fallback_angle: float = 0.0
) -> tuple[Dict[str, float], int]:
    """Convenience to compute σ from ``(glyph, weight)`` pairs."""

    vectors = (glyph_unit(g) * float(w) for g, w in pairs)
    return _sigma_from_vectors(vectors, fallback_angle)

def sigma_vector_node(G, n, weight_mode: str | None = None) -> Dict[str, float] | None:
    cfg = _sigma_cfg(G)
    nd = G.nodes[n]
    nw = _node_weight(nd, weight_mode or cfg.get("weight", "Si"))
    if not nw:
        return None
    g, w, z = nw
    vec, _ = _sigma_from_vectors([z], glyph_angle(g))
    vec.update({"glyph": g, "w": float(w)})
    return vec


def sigma_vector(dist: Dict[str, float]) -> Dict[str, float]:
    """Compute Σ⃗ from a glyph distribution.

    ``dist`` may contain raw counts or proportions. Values are normalised with
    respect to glyphs relevant to the σ plane and the Cartesian components,
    magnitude and resulting angle are obtained. If the distribution provides
    no weight on the relevant glyphs the zero vector is returned.
    """

    total = math.fsum(float(dist.get(k, 0.0)) for k in SIGMA_ANGLE_KEYS)
    if total <= 0:
        return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0}

    factor = len(SIGMA_ANGLE_KEYS) / total
    z_iter = (
        glyph_unit(k) * float(dist.get(k, 0.0)) * factor
        for k in SIGMA_ANGLE_KEYS
    )
    vec, _ = _sigma_from_vectors(z_iter)
    return vec


def sigma_vector_from_graph(G: nx.Graph, weight_mode: str | None = None) -> Dict[str, float]:
    """Vector global del plano del sentido σ para un grafo.

    Parameters
    ----------
    G:
        Grafo de NetworkX con estados por nodo.
    weight_mode:
        Cómo ponderar cada nodo ("Si", "EPI" o ``None`` para peso unitario).

    Returns
    -------
    Dict[str, float]
        Componentes cartesianas, magnitud y ángulo del vector promedio.
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("sigma_vector_from_graph requiere un networkx.Graph")

    cfg = _sigma_cfg(G)
    weight_mode = weight_mode or cfg.get("weight", "Si")
    z_iter = (
        nw[2]
        for _, nd in G.nodes(data=True)
        if (nw := _node_weight(nd, weight_mode))
    )
    vec, n = _sigma_from_vectors(z_iter)
    vec["n"] = n
    return vec


def sigma_vector_global(G: nx.Graph, weight_mode: str | None = None) -> Dict[str, float]:
    """Alias de :func:`sigma_vector_from_graph`.

    .. deprecated:: 4.5.3
       Use :func:`sigma_vector_from_graph` en su lugar.
    """

    warnings.warn(
        "sigma_vector_global está deprecada; use sigma_vector_from_graph",
        DeprecationWarning,
        stacklevel=2,
    )
    return sigma_vector_from_graph(G, weight_mode)


# -------------------------
# Historia / series
# -------------------------

def push_sigma_snapshot(G, t: float | None = None) -> None:
    cfg = _sigma_cfg(G)
    if not cfg.get("enabled", True):
        return
    hist = ensure_history(G)
    key = cfg.get("history_key", "sigma_global")

    # Global
    sv = sigma_vector_from_graph(G, cfg.get("weight", "Si"))

    # Suavizado exponencial (EMA) opcional
    alpha = float(cfg.get("smooth", 0.0))
    if alpha > 0 and hist.get(key):
        prev = hist[key][-1]
        x = (1-alpha)*prev["x"] + alpha*sv["x"]
        y = (1-alpha)*prev["y"] + alpha*sv["y"]
        mag = math.hypot(x, y)
        ang = math.atan2(y, x)
        sv = {"x": x, "y": y, "mag": mag, "angle": ang, "n": sv.get("n", 0)}

    sv["t"] = float(G.graph.get("_t", 0.0) if t is None else t)

    hist.setdefault(key, []).append(sv)

    # Conteo de glyphs por paso (útil para rosa glífica)
    counts = count_glyphs(G, last_only=True)
    hist.setdefault("sigma_counts", []).append({"t": sv["t"], **counts})

    # Trayectoria por nodo (opcional)
    if cfg.get("per_node", False):
        per = hist.setdefault("sigma_per_node", {})
        for n, nd in G.nodes(data=True):
            g = last_glyph(nd)
            if not g:
                continue
            a = glyph_angle(g)
            d = per.setdefault(n, [])
            d.append({"t": sv["t"], "g": g, "angle": a})


# -------------------------
# Registro como callback automático (after_step)
# -------------------------

def register_sigma_callback(G) -> None:
    register_callback(G, event="after_step", func=push_sigma_snapshot, name="sigma_snapshot")


# -------------------------
# Series de utilidad
# -------------------------

def sigma_series(G, key: str | None = None) -> Dict[str, List[float]]:
    cfg = _sigma_cfg(G)
    key = key or cfg.get("history_key", "sigma_global")
    hist = G.graph.get("history", {})
    xs = hist.get(key, [])
    if not xs:
        return {"t": [], "angle": [], "mag": []}
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "angle": [float(x["angle"]) for x in xs],
        "mag": [float(x["mag"]) for x in xs],
    }


def sigma_rose(G, steps: int | None = None) -> Dict[str, int]:
    """Histogram of glyphs in the last ``steps`` steps (or all)."""
    hist = G.graph.get("history", {})
    counts = hist.get("sigma_counts", [])
    if not counts:
        return {g: 0 for g in GLYPHS_CANONICAL}
    if steps is None or steps >= len(counts):
        agg = Counter()
        for row in counts:
            agg.update({k: v for k, v in row.items() if k != "t"})
        out = {g: int(agg.get(g, 0)) for g in GLYPHS_CANONICAL}
        return out
    agg = Counter()
    for row in counts[-int(steps):]:
        agg.update({k: v for k, v in row.items() if k != "t"})
    return {g: int(agg.get(g, 0)) for g in GLYPHS_CANONICAL}
