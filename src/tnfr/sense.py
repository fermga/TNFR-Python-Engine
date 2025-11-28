"""Sense calculations and structural operator symbol vector analysis.

This module implements the sense index (Si) calculation and related vector
operations for analyzing the distribution of structural operator applications.

The 'glyph rose' visualization represents the distribution of structural operator
symbols in a circular format, where each glyph corresponds to an angle representing
the associated structural operator.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from itertools import tee
from typing import Any, Callable, TypeVar

import networkx as nx

from .alias import get_attr
from .utils import CallbackEvent, callback_manager
from .config.constants import (
    ANGLE_MAP,
    GLYPHS_CANONICAL,
)
from .constants import get_graph_param
from .constants.aliases import ALIAS_EPI, ALIAS_SI
from .glyph_history import append_metric, count_glyphs, ensure_history
from .glyph_runtime import last_glyph
from .utils import clamp01, kahan_sum_nd
from .types import NodeId, SigmaVector, TNFRGraph
from .utils import get_numpy

# -------------------------
# Canon: circular glyph order and angles
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
# Basic utilities
# -------------------------

T = TypeVar("T")


def _resolve_glyph(g: str, mapping: Mapping[str, T]) -> T:
    """Return ``mapping[g]`` or raise ``KeyError`` with a standard message."""

    try:
        return mapping[g]
    except KeyError as e:  # pragma: no cover - small helper
        raise KeyError(f"Unknown glyph: {g}") from e


def glyph_angle(g: str) -> float:
    """Return the canonical angle for structural operator symbol ``g``.

    Each structural operator symbol (glyph) is mapped to a specific angle
    in the circular representation used for sense vector calculations.
    """

    return float(_resolve_glyph(g, ANGLE_MAP))


def glyph_unit(g: str) -> complex:
    """Return the unit vector for structural operator symbol ``g``.

    Each structural operator symbol (glyph) corresponds to a unit vector
    in the complex plane, used for aggregating operator applications.
    """

    return _resolve_glyph(g, GLYPH_UNITS)


MODE_FUNCS: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "Si": lambda nd: clamp01(get_attr(nd, ALIAS_SI, 0.5)),
    "EPI": lambda nd: max(0.0, get_attr(nd, ALIAS_EPI, 0.0)),
}


def _weight(nd: Mapping[str, Any], mode: str) -> float:
    return MODE_FUNCS.get(mode, lambda _: 1.0)(nd)


def _node_weight(nd: Mapping[str, Any], weight_mode: str) -> tuple[str, float, complex] | None:
    """Return ``(glyph, weight, weighted_unit)`` or ``None`` if no glyph."""
    g = last_glyph(nd)
    if not g:
        return None
    w = _weight(nd, weight_mode)
    z = glyph_unit(g) * w  # precompute weighted unit vector
    return g, w, z


def _sigma_cfg(G: TNFRGraph) -> dict[str, Any]:
    return get_graph_param(G, "SIGMA", dict)


def _to_complex(val: complex | float | int) -> complex:
    """Return ``val`` as complex, promoting real numbers."""

    if isinstance(val, complex):
        return val
    if isinstance(val, (int, float)):
        return complex(val, 0.0)
    raise TypeError("values must be an iterable of real or complex numbers")


def _empty_sigma(fallback_angle: float) -> SigmaVector:
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
# σ per node and global σ
# -------------------------


def _sigma_from_iterable(
    values: Iterable[complex | float | int] | complex | float | int,
    fallback_angle: float = 0.0,
) -> SigmaVector:
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
            "x": float(x),
            "y": float(y),
            "mag": float(mag),
            "angle": float(ang),
            "n": int(cnt),
        }
    cnt = 0

    def pair_iter() -> Iterator[tuple[float, float]]:
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
        "n": int(cnt),
    }


def _ema_update(prev: SigmaVector, current: SigmaVector, alpha: float) -> SigmaVector:
    """Exponential moving average update for σ vectors."""
    x = (1 - alpha) * prev["x"] + alpha * current["x"]
    y = (1 - alpha) * prev["y"] + alpha * current["y"]
    mag = math.hypot(x, y)
    ang = math.atan2(y, x)
    return {
        "x": float(x),
        "y": float(y),
        "mag": float(mag),
        "angle": float(ang),
        "n": int(current["n"]),
    }


def _sigma_from_nodes(
    nodes: Iterable[Mapping[str, Any]],
    weight_mode: str,
    fallback_angle: float = 0.0,
) -> tuple[SigmaVector, list[tuple[str, float, complex]]]:
    """Aggregate weighted glyph vectors for ``nodes``.

    Returns the aggregated σ vector and the list of ``(glyph, weight, vector)``
    triples used in the calculation.
    """

    nws = [nw for nd in nodes if (nw := _node_weight(nd, weight_mode))]
    sv = _sigma_from_iterable((nw[2] for nw in nws), fallback_angle)
    return sv, nws


def sigma_vector_node(
    G: TNFRGraph, n: NodeId, weight_mode: str | None = None
) -> SigmaVector | None:
    """Return the σ vector for node ``n`` using the configured weighting."""

    cfg = _sigma_cfg(G)
    nd = G.nodes[n]
    weight_mode = weight_mode or cfg.get("weight", "Si")
    sv, nws = _sigma_from_nodes([nd], weight_mode)
    if not nws:
        return None
    g, w, _ = nws[0]
    if sv["mag"] == 0:
        sv["angle"] = glyph_angle(g)
    sv["glyph"] = g
    sv["w"] = float(w)
    return sv


def sigma_vector(dist: Mapping[str, float]) -> SigmaVector:
    """Compute Σ⃗ from a glyph distribution.

    ``dist`` may contain raw counts or proportions. All ``(glyph, weight)``
    pairs are converted to vectors and passed to :func:`_sigma_from_iterable`.
    The resulting vector includes the number of processed pairs under ``n``.
    """

    vectors = (glyph_unit(g) * float(w) for g, w in dist.items())
    return _sigma_from_iterable(vectors)


def sigma_vector_from_graph(G: TNFRGraph, weight_mode: str | None = None) -> SigmaVector:
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
        raise TypeError("sigma_vector_from_graph requires a networkx.Graph")

    cfg = _sigma_cfg(G)
    weight_mode = weight_mode or cfg.get("weight", "Si")
    sv, _ = _sigma_from_nodes((nd for _, nd in G.nodes(data=True)), weight_mode)
    return sv


# -------------------------
# History / series
# -------------------------


def push_sigma_snapshot(G: TNFRGraph, t: float | None = None) -> None:
    """Record a global σ snapshot (and optional per-node traces) for ``G``."""

    cfg = _sigma_cfg(G)
    if not cfg.get("enabled", True):
        return

    # Local history cache to avoid repeated lookups
    hist = ensure_history(G)
    key = cfg.get("history_key", "sigma_global")

    weight_mode = cfg.get("weight", "Si")
    sv = sigma_vector_from_graph(G, weight_mode)

    # Optional exponential smoothing (EMA)
    alpha = float(cfg.get("smooth", 0.0))
    if alpha > 0 and hist.get(key):
        sv = _ema_update(hist[key][-1], sv, alpha)

    current_t = float(G.graph.get("_t", 0.0) if t is None else t)
    sv["t"] = current_t

    append_metric(hist, key, sv)

    # Glyph count per step (useful for the glyph rose)
    counts = count_glyphs(G, last_only=True)
    append_metric(hist, "sigma_counts", {"t": current_t, **counts})

    # Optional per-node trajectory
    if cfg.get("per_node", False):
        per = hist.setdefault("sigma_per_node", {})
        for n, nd in G.nodes(data=True):
            g = last_glyph(nd)
            if not g:
                continue
            d = per.setdefault(n, [])
            d.append({"t": current_t, "g": g, "angle": glyph_angle(g)})


# -------------------------
# Register as an automatic callback (after_step)
# -------------------------


def register_sigma_callback(G: TNFRGraph) -> None:
    """Attach :func:`push_sigma_snapshot` to the ``AFTER_STEP`` callback bus."""

    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=push_sigma_snapshot,
        name="sigma_snapshot",
    )


def sigma_rose(G: TNFRGraph, steps: int | None = None) -> dict[str, int]:
    """Histogram of glyphs in the last ``steps`` steps (or all)."""
    hist = ensure_history(G)
    counts = hist.get("sigma_counts", [])
    if not counts:
        return {g: 0 for g in GLYPHS_CANONICAL}
    if steps is not None:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        rows = counts if steps >= len(counts) else counts[-steps:]  # noqa: E203
    else:
        rows = counts
    counter = Counter()
    for row in rows:
        for k, v in row.items():
            if k != "t":
                counter[k] += int(v)
    return {g: int(counter.get(g, 0)) for g in GLYPHS_CANONICAL}
