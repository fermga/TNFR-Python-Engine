from __future__ import annotations

from typing import Any, Iterable, Sequence, Dict
from statistics import fmean, StatisticsError
import math

from ..import_utils import get_numpy, import_nodonx
from ..alias import get_attr

__all__ = [
    "clamp",
    "clamp01",
    "list_mean",
    "kahan_sum",
    "angle_diff",
    "neighbor_mean",
    "neighbor_phase_mean",
    "neighbor_phase_mean_list",
]


def clamp(x: float, a: float, b: float) -> float:
    """Return ``x`` clamped to the ``[a, b]`` interval."""
    return max(a, min(b, x))


def clamp01(x: float) -> float:
    """Clamp ``x`` to the ``[0,1]`` interval."""
    return clamp(float(x), 0.0, 1.0)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    """Return the arithmetic mean of ``xs`` or ``default`` if empty."""
    try:
        return float(fmean(xs))
    except (StatisticsError, ValueError, TypeError):
        return float(default)


def kahan_sum(values: Iterable[float]) -> float:
    """Return the precise sum of ``values`` using Kahan compensation."""
    total = 0.0
    c = 0.0
    for v in values:
        y = float(v) - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def angle_diff(a: float, b: float) -> float:
    """Return the minimal difference between two angles in radians."""
    return (float(a) - float(b) + math.pi) % math.tau - math.pi


def neighbor_mean(G, n, aliases: tuple[str, ...], default: float = 0.0) -> float:
    """Mean of ``aliases`` attribute among neighbours of ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def _neighbor_phase_mean(node, trig) -> float:
    """Internal helper delegating to :func:`neighbor_phase_mean_list`."""
    fallback = trig.theta.get(node.n, 0.0)
    neigh = node.G[node.n]
    np = get_numpy()
    return neighbor_phase_mean_list(
        neigh, trig.cos, trig.sin, np=np, fallback=fallback
    )


def _phase_mean_from_iter(
    it: Iterable[tuple[float, float] | None], fallback: float
) -> float:
    x = y = 0.0
    cx = cy = 0.0  # Kahan compensation terms
    count = 0
    for cs in it:
        if cs is None:
            continue
        cos_val, sin_val = cs
        tx = cos_val - cx
        vx = x + tx
        cx = (vx - x) - tx
        x = vx
        ty = sin_val - cy
        vy = y + ty
        cy = (vy - y) - ty
        y = vy
        count += 1
    if count == 0:
        return fallback
    return math.atan2(y, x)


def neighbor_phase_mean_list(
    neigh: Sequence[Any],
    cos_th: Dict[Any, float],
    sin_th: Dict[Any, float],
    np=None,
    fallback: float = 0.0,
) -> float:
    """Return circular mean of neighbour phases from cosine/sine mappings.

    When ``np`` (NumPy) is provided, ``np.fromiter`` is used to compute the
    averages. Otherwise, the mean is computed using the pure-Python
    :func:`_phase_mean_from_iter` helper.
    """
    deg = len(neigh)
    if deg == 0:
        return fallback
    if np is not None:
        cos_vals = np.fromiter((cos_th[v] for v in neigh), dtype=float, count=deg)
        sin_vals = np.fromiter((sin_th[v] for v in neigh), dtype=float, count=deg)
        mean_cos = float(cos_vals.mean())
        mean_sin = float(sin_vals.mean())
        return float(np.arctan2(mean_sin, mean_cos))
    return _phase_mean_from_iter(((cos_th[v], sin_th[v]) for v in neigh), fallback)


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases.

    The :class:`NodoNX` import is cached after the first call.
    """
    NodoNX = import_nodonx()
    node = NodoNX(obj, n) if n is not None else obj
    if getattr(node, "G", None) is None:
        raise TypeError("neighbor_phase_mean requires nodes bound to a graph")
    from ..metrics_utils import get_trig_cache

    trig = get_trig_cache(node.G)
    return _neighbor_phase_mean(node, trig)
