"""Sense calculations."""

from __future__ import annotations
from typing import TypeVar
from collections.abc import Iterable, Mapping
import math
from collections import Counter
from itertools import tee

import networkx as nx  # type: ignore[import-untyped]

from .constants import get_aliases, get_graph_param
from .alias import get_attr
from .helpers.numeric import clamp01, kahan_sum_nd
from .utils import get_numpy
from .callback_utils import CallbackEvent, callback_manager
from .glyph_history import (
    ensure_history,
    last_glyph,
    count_glyphs,
    append_metric,
)
from .constants_glyphs import (
    ANGLE_MAP,
    GLYPHS_CANONICAL,
)
# -------------------------
# Canon: orden circular de glyphs y ángulos
# -------------------------

GLYPH_UNITS: dict[str, complex] = {
    g: complex(math.cos(a), math.sin(a)) for g, a in ANGLE_MAP.items()
}

__all__ = (
    "GLYPH_UNITS",
    "glyph_angle",
    "glyph_unit",
    "sigma_vector_node",
    "sigma_vector",
    "sigma_vector_from_graph",
    "push_sigma_snapshot",
    "register_sigma_callback",
    "sigma_rose",
)

# -------------------------
# Utilidades básicas
# -------------------------


T = TypeVar("T")


def _resolve_glyph(g: str, mapping: Mapping[str, T]) -> T:
    """Return ``mapping[g]`` or raise ``KeyError`` with a standard message."""

    try:
        return mapping[g]
    except KeyError as e:  # pragma: no cover - small helper
        raise KeyError(f"Glyph desconocido: {g}") from e


def glyph_angle(g: str) -> float:
    """Return angle for glyph ``g``."""

    return float(_resolve_glyph(g, ANGLE_MAP))


def glyph_unit(g: str) -> complex:
    """Return unit vector for glyph ``g``."""

    return _resolve_glyph(g, GLYPH_UNITS)


ALIAS_SI = get_aliases("SI")
ALIAS_EPI = get_aliases("EPI")

MODE_FUNCS = {
    "Si": lambda nd: clamp01(get_attr(nd, ALIAS_SI, 0.5)),
    "EPI": lambda nd: max(0.0, get_attr(nd, ALIAS_EPI, 0.0)),
}


def _weight(nd, mode: str) -> float:
    return MODE_FUNCS.get(mode, lambda _: 1.0)(nd)


def _node_weight(nd, weight_mode: str) -> tuple[str, float, complex] | None:
    """Return ``(glyph, weight, weighted_unit)`` or ``None`` if no glyph."""
    g = last_glyph(nd)
    if not g:
        return None
    w = _weight(nd, weight_mode)
    z = glyph_unit(g) * w  # precompute weighted unit vector
    return g, w, z


def _sigma_cfg(G):
    return get_graph_param(G, "SIGMA", dict)


def _to_complex(val: complex | float | int) -> complex:
    """Return ``val`` as complex, promoting real numbers."""

    if isinstance(val, complex):
        return val
    if isinstance(val, (int, float)):
        return complex(val, 0.0)
    raise TypeError("values must be an iterable of real or complex numbers")


def _empty_sigma(fallback_angle: float) -> dict[str, float]:
    """Return an empty σ-vector with ``fallback_angle``.

    Helps centralise the default structure returned when no values are
    available for σ calculations.
    """

    return {
        "x": 0.0,
        "y": 0.0,
        "mag": 0.0,
        "angle": float(fallback_angle),
        "n": 0,
    }


# -------------------------
# σ por nodo y σ global
# -------------------------


def _sigma_from_iterable(
    values: Iterable[complex | float | int] | complex | float | int,
    fallback_angle: float = 0.0,
) -> dict[str, float]:
    """Normalise vectors in the σ-plane.

    ``values`` may contain complex or real numbers; real inputs are promoted to
    complex with zero imaginary part. The returned dictionary includes the
    number of processed values under the ``"n"`` key.
    """

    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, bytearray, Mapping)):
        iterator = iter(values)
    else:
        iterator = iter((values,))

    np = get_numpy()
    if np is not None:
        iterator, np_iter = tee(iterator)
        arr = np.fromiter((_to_complex(v) for v in np_iter), dtype=np.complex128)
        cnt = int(arr.size)
        if cnt == 0:
            return _empty_sigma(fallback_angle)
        x = float(np.mean(arr.real))
        y = float(np.mean(arr.imag))
        mag = float(np.hypot(x, y))
        ang = float(np.arctan2(y, x)) if mag > 0 else float(fallback_angle)
        return {
            "x": x,
            "y": y,
            "mag": mag,
            "angle": ang,
            "n": cnt,
        }
    cnt = 0

    def pair_iter():
        nonlocal cnt
        for val in iterator:
            z = _to_complex(val)
            cnt += 1
            yield (z.real, z.imag)

    sum_x, sum_y = kahan_sum_nd(pair_iter(), dims=2)

    if cnt == 0:
        return _empty_sigma(fallback_angle)

    x = sum_x / cnt
    y = sum_y / cnt
    mag = math.hypot(x, y)
    ang = math.atan2(y, x) if mag > 0 else float(fallback_angle)
    return {
        "x": float(x),
        "y": float(y),
        "mag": float(mag),
        "angle": float(ang),
        "n": cnt,
    }


def _ema_update(
    prev: dict[str, float], current: dict[str, float], alpha: float
) -> dict[str, float]:
    """Exponential moving average update for σ vectors."""
    x = (1 - alpha) * prev["x"] + alpha * current["x"]
    y = (1 - alpha) * prev["y"] + alpha * current["y"]
    mag = math.hypot(x, y)
    ang = math.atan2(y, x)
    return {"x": x, "y": y, "mag": mag, "angle": ang, "n": current.get("n", 0)}


def _sigma_from_nodes(
    nodes: Iterable[dict], weight_mode: str, fallback_angle: float = 0.0
) -> tuple[dict[str, float], list[tuple[str, float, complex]]]:
    """Aggregate weighted glyph vectors for ``nodes``.

    Returns the aggregated σ vector and the list of ``(glyph, weight, vector)``
    triples used in the calculation.
    """

    nws = [nw for nd in nodes if (nw := _node_weight(nd, weight_mode))]
    sv = _sigma_from_iterable((nw[2] for nw in nws), fallback_angle)
    return sv, nws


def sigma_vector_node(
    G, n, weight_mode: str | None = None
) -> dict[str, float] | None:
    cfg = _sigma_cfg(G)
    nd = G.nodes[n]
    weight_mode = weight_mode or cfg.get("weight", "Si")
    sv, nws = _sigma_from_nodes([nd], weight_mode)
    if not nws:
        return None
    g, w, _ = nws[0]
    if sv["mag"] == 0:
        sv["angle"] = glyph_angle(g)
    sv.update({"glyph": g, "w": float(w)})
    return sv


def sigma_vector(dist: dict[str, float]) -> dict[str, float]:
    """Compute Σ⃗ from a glyph distribution.

    ``dist`` may contain raw counts or proportions. All ``(glyph, weight)``
    pairs are converted to vectors and passed to :func:`_sigma_from_iterable`.
    The resulting vector includes the number of processed pairs under ``n``.
    """

    vectors = (glyph_unit(g) * float(w) for g, w in dist.items())
    return _sigma_from_iterable(vectors)


def sigma_vector_from_graph(
    G: nx.Graph, weight_mode: str | None = None
) -> dict[str, float]:
    """Global vector in the σ sense plane for a graph.

    Parameters
    ----------
    G:
        NetworkX graph with per-node states.
    weight_mode:
        How to weight each node ("Si", "EPI" or ``None`` for unit weight).

    Returns
    -------
    dict[str, float]
        Cartesian components, magnitude and angle of the average vector.
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("sigma_vector_from_graph requiere un networkx.Graph")

    cfg = _sigma_cfg(G)
    weight_mode = weight_mode or cfg.get("weight", "Si")
    sv, _ = _sigma_from_nodes(
        (nd for _, nd in G.nodes(data=True)), weight_mode
    )
    return sv


# -------------------------
# Historia / series
# -------------------------


def push_sigma_snapshot(G, t: float | None = None) -> None:
    cfg = _sigma_cfg(G)
    if not cfg.get("enabled", True):
        return

    # Cache local de la historia para evitar llamadas repetidas
    hist = ensure_history(G)
    key = cfg.get("history_key", "sigma_global")

    weight_mode = cfg.get("weight", "Si")
    sv = sigma_vector_from_graph(G, weight_mode)

    # Suavizado exponencial (EMA) opcional
    alpha = float(cfg.get("smooth", 0.0))
    if alpha > 0 and hist.get(key):
        sv = _ema_update(hist[key][-1], sv, alpha)

    current_t = float(G.graph.get("_t", 0.0) if t is None else t)
    sv["t"] = current_t

    append_metric(hist, key, sv)

    # Conteo de glyphs por paso (útil para rosa glífica)
    counts = count_glyphs(G, last_only=True)
    append_metric(hist, "sigma_counts", {"t": current_t, **counts})

    # Trayectoria por nodo (opcional)
    if cfg.get("per_node", False):
        per = hist.setdefault("sigma_per_node", {})
        for n, nd in G.nodes(data=True):
            g = last_glyph(nd)
            if not g:
                continue
            d = per.setdefault(n, [])
            d.append({"t": current_t, "g": g, "angle": glyph_angle(g)})


# -------------------------
# Registro como callback automático (after_step)
# -------------------------


def register_sigma_callback(G) -> None:
    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=push_sigma_snapshot,
        name="sigma_snapshot",
    )


def sigma_rose(G, steps: int | None = None) -> dict[str, int]:
    """Histogram of glyphs in the last ``steps`` steps (or all)."""
    hist = ensure_history(G)
    counts = hist.get("sigma_counts", [])
    if not counts:
        return {g: 0 for g in GLYPHS_CANONICAL}
    if steps is not None:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        rows = (
            counts if steps >= len(counts) else counts[-steps:]
        )  # noqa: E203
    else:
        rows = counts
    counter = Counter()
    for row in rows:
        for k, v in row.items():
            if k != "t":
                counter[k] += int(v)
    return {g: int(counter.get(g, 0)) for g in GLYPHS_CANONICAL}
