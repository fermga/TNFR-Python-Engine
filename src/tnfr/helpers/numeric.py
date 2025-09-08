from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Sequence
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
    """Return a compensated sum of ``values`` using Kahan summation.

    The implementation follows the Kahan–Babuška (Neumaier) algorithm,
    which keeps track of a small correction term to reduce floating point
    error. It is more accurate than the built-in :func:`sum` while being
    cheaper than :func:`math.fsum`.
    """
    total = 0.0
    c = 0.0
    for v in values:
        t = total + v
        if abs(total) >= abs(v):
            c += (total - t) + v
        else:
            c += (v - t) + total
        total = t
    return float(total + c)


def angle_diff(a: float, b: float) -> float:
    """Return the minimal difference between two angles in radians."""
    return (float(a) - float(b) + math.pi) % math.tau - math.pi


def neighbor_mean(
    G, n, aliases: tuple[str, ...], default: float = 0.0
) -> float:
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
    total_cos = 0.0
    comp_cos = 0.0
    total_sin = 0.0
    comp_sin = 0.0
    found = False
    for cs in it:
        if cs is None:
            continue
        found = True
        cos_val, sin_val = cs
        # Kahan summation for cosine
        y = cos_val - comp_cos
        t = total_cos + y
        comp_cos = (t - total_cos) - y
        total_cos = t
        # Kahan summation for sine
        y = sin_val - comp_sin
        t = total_sin + y
        comp_sin = (t - total_sin) - y
        total_sin = t
    if not found:
        return fallback
    total_cos += comp_cos
    total_sin += comp_sin
    return math.atan2(total_sin, total_cos)


def neighbor_phase_mean_list(
    neigh: Sequence[Any],
    cos_th: dict[Any, float],
    sin_th: dict[Any, float],
    np=None,
    fallback: float = 0.0,
) -> float:
    """Return circular mean of neighbour phases from cosine/sine mappings.

    When ``np`` (NumPy) is provided, ``np.fromiter`` is used to compute the
    averages. Otherwise, the mean is computed using the pure-Python
    :func:`_phase_mean_from_iter` helper.
    """
    deg = len(neigh)
    if np is not None and deg > 0:
        pairs = np.fromiter(
            (val for v in neigh for val in (cos_th[v], sin_th[v])),
            dtype=float,
            count=deg * 2,
        ).reshape(deg, 2)
        mean_cos, mean_sin = pairs.mean(axis=0)
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
