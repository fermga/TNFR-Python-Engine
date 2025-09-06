"""Sense calculations."""

from __future__ import annotations
from typing import Dict, Iterable, List
import math

import networkx as nx

from .constants import ALIAS_SI, ALIAS_EPI, SIGMA
from .alias import get_attr
from .helpers import clamp01
from .callback_utils import register_callback
from .glyph_history import ensure_history, last_glyph, count_glyphs, append_metric
from .constants_glyphs import (
    ANGLE_MAP,
    GLYPHS_CANONICAL,
)

# -------------------------
# Canon: orden circular de glyphs y ángulos
# -------------------------

GLYPH_UNITS: Dict[str, complex] = {
    g: complex(math.cos(a), math.sin(a)) for g, a in ANGLE_MAP.items()
}

# -------------------------
# Utilidades básicas
# -------------------------


def glyph_angle(g: str) -> float:
    """Return angle for glyph ``g``.

    Raises ``KeyError`` if ``g`` is not registered in :data:`ANGLE_MAP`.
    """
    try:
        return float(ANGLE_MAP[g])
    except KeyError as e:
        raise KeyError(f"Glyph desconocido: {g}") from e


def glyph_unit(g: str) -> complex:
    """Return unit vector for glyph ``g``.

    Raises ``KeyError`` if ``g`` is not registered in :data:`ANGLE_MAP`.
    """
    try:
        return GLYPH_UNITS[g]
    except KeyError as e:
        raise KeyError(f"Glyph desconocido: {g}") from e


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


def _sigma_from_iterable(
    values: Iterable[complex] | complex, fallback_angle: float = 0.0
) -> tuple[Dict[str, float], int]:
    """Normalise complex vectors in el plano σ.

    ``values`` puede ser un complejo individual o un iterable de ellos.
    """

    try:
        iterator = iter(values)
    except TypeError:
        iterator = iter([values])

    cnt = 0
    acc = complex(0.0, 0.0)
    for z in iterator:
        if not isinstance(z, complex):
            raise TypeError("values must be an iterable of complex numbers")
        cnt += 1
        acc += z

    if cnt <= 0:
        vec = {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": float(fallback_angle)}
        return vec, 0

    x, y = acc.real / cnt, acc.imag / cnt
    mag = math.hypot(x, y)
    ang = math.atan2(y, x) if mag > 0 else float(fallback_angle)
    vec = {
        "x": float(x),
        "y": float(y),
        "mag": float(mag),
        "angle": float(ang),
    }
    return vec, cnt


def _sigma_from_pairs(
    pairs: Iterable[tuple[str, float]], fallback_angle: float = 0.0
) -> tuple[Dict[str, float], int]:
    """Backward-compatible wrapper to compute σ from ``(glyph, weight)`` pairs."""

    vectors = (glyph_unit(g) * float(w) for g, w in pairs)
    return _sigma_from_iterable(vectors, fallback_angle)


# Retro-compatibilidad
_sigma_from_vectors = _sigma_from_iterable


def sigma_vector_node(
    G, n, weight_mode: str | None = None
) -> Dict[str, float] | None:
    cfg = _sigma_cfg(G)
    nd = G.nodes[n]
    nw = _node_weight(nd, weight_mode or cfg.get("weight", "Si"))
    if not nw:
        return None
    g, w, z = nw
    vec, _ = _sigma_from_iterable(z, glyph_angle(g))
    vec.update({"glyph": g, "w": float(w)})
    return vec


def sigma_vector(dist: Dict[str, float]) -> tuple[Dict[str, float], int]:
    """Compute Σ⃗ from a glyph distribution.

    ``dist`` may contain raw counts or proportions. All ``(glyph, weight)``
    pairs are forwarded to :func:`_sigma_from_pairs` and the resulting vector
    together with the number of processed pairs are returned.
    """

    vectors = (glyph_unit(g) * float(w) for g, w in dist.items())
    return _sigma_from_iterable(vectors)


def sigma_vector_from_graph(
    G: nx.Graph, weight_mode: str | None = None
) -> Dict[str, float]:
    """Global vector in the σ sense plane for a graph.

    Parameters
    ----------
    G:
        NetworkX graph with per-node states.
    weight_mode:
        How to weight each node ("Si", "EPI" or ``None`` for unit weight).

    Returns
    -------
    Dict[str, float]
        Cartesian components, magnitude and angle of the average vector.
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("sigma_vector_from_graph requiere un networkx.Graph")

    cfg = _sigma_cfg(G)
    weight_mode = weight_mode or cfg.get("weight", "Si")
    vectors = (
        nw[2]
        for _, nd in G.nodes(data=True)
        if (nw := _node_weight(nd, weight_mode))
    )
    vec, n = _sigma_from_iterable(vectors)
    vec["n"] = n
    return vec


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
        x = (1 - alpha) * prev["x"] + alpha * sv["x"]
        y = (1 - alpha) * prev["y"] + alpha * sv["y"]
        mag = math.hypot(x, y)
        ang = math.atan2(y, x)
        sv = {"x": x, "y": y, "mag": mag, "angle": ang, "n": sv.get("n", 0)}

    sv["t"] = float(G.graph.get("_t", 0.0) if t is None else t)

    append_metric(hist, key, sv)

    # Conteo de glyphs por paso (útil para rosa glífica)
    counts = count_glyphs(G, last_only=True)
    append_metric(hist, "sigma_counts", {"t": sv["t"], **counts})

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
    register_callback(
        G, event="after_step", func=push_sigma_snapshot, name="sigma_snapshot"
    )


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
        agg: Dict[str, int] = {}
        for row in counts:
            for k, v in row.items():
                if k != "t":
                    agg[k] = int(agg.get(k, 0)) + int(v)
        return {g: int(agg.get(g, 0)) for g in GLYPHS_CANONICAL}
    agg: Dict[str, int] = {}
    start = -int(steps)
    for row in counts[start:]:
        for k, v in row.items():
            if k != "t":
                agg[k] = int(agg.get(k, 0)) + int(v)
    return {g: int(agg.get(g, 0)) for g in GLYPHS_CANONICAL}
