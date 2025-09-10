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
    "kahan_sum_nd",
    "kahan_sum",
    "kahan_sum2d",
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


def kahan_sum_nd(values: Iterable[Sequence[float]], dims: int) -> tuple[float, ...]:
    """Return compensated sums of ``values`` with ``dims`` components.

    Each component of the tuples in ``values`` is summed independently using the
    Kahan–Babuška (Neumaier) algorithm to reduce floating point error.
    """
    if dims < 1:
        raise ValueError("dims must be >= 1")
    totals = [0.0] * dims
    comps = [0.0] * dims
    for vs in values:
        for i in range(dims):
            v = vs[i]
            t = totals[i] + v
            if abs(totals[i]) >= abs(v):
                comps[i] += (totals[i] - t) + v
            else:
                comps[i] += (v - t) + totals[i]
            totals[i] = t
    return tuple(float(totals[i] + comps[i]) for i in range(dims))


def kahan_sum(values: Iterable[float]) -> float:
    """Return a compensated sum of ``values`` using Kahan summation."""
    (result,) = kahan_sum_nd(((v,) for v in values), dims=1)
    return result


def kahan_sum2d(values: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """Return compensated sums of paired ``values`` using Kahan summation."""
    return kahan_sum_nd(values, dims=2)


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
    """Return circular mean from an iterator of cosine/sine pairs.

    ``it`` yields optional ``(cos, sin)`` tuples; ``None`` entries are ignored.
    The iterator is consumed directly and the components are accumulated with a
    running Kahan–Babuška summation to avoid storing intermediate results. If
    no valid pairs are found ``fallback`` is returned.
    """
    found = False
    total_cos = total_sin = 0.0
    comp_cos = comp_sin = 0.0
    for cs in it:
        if cs is None:
            continue
        found = True
        cos_val, sin_val = cs
        # Accumulate cosine component
        t = total_cos + cos_val
        if abs(total_cos) >= abs(cos_val):
            comp_cos += (total_cos - t) + cos_val
        else:
            comp_cos += (cos_val - t) + total_cos
        total_cos = t
        # Accumulate sine component
        t = total_sin + sin_val
        if abs(total_sin) >= abs(sin_val):
            comp_sin += (total_sin - t) + sin_val
        else:
            comp_sin += (sin_val - t) + total_sin
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

    When ``np`` (NumPy) is provided, a vectorised approach computes the
    averages. Otherwise, the mean is computed using the pure-Python
    :func:`_phase_mean_from_iter` helper which uses a running Kahan
    summation for stable accumulation.
    """
    deg = len(neigh)
    if np is not None and deg > 0:
        pairs = np.fromiter(
            (c for v in neigh for c in (cos_th[v], sin_th[v])),
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
