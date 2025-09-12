"""Numeric helper functions and compensated summation utilities."""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Sequence
from statistics import fmean, StatisticsError, pvariance
import math

from ..import_utils import get_numpy, import_nodonx
from ..alias import get_attr

__all__ = (
    "clamp",
    "clamp01",
    "within_range",
    "list_mean",
    "list_pvariance",
    "kahan_sum_nd",
    "kahan_sum",
    "kahan_sum2d",
    "accumulate_cos_sin",
    "angle_diff",
    "neighbor_mean",
    "neighbor_phase_mean",
    "neighbor_phase_mean_list",
)


def clamp(x: float, a: float, b: float) -> float:
    """Return ``x`` clamped to the ``[a, b]`` interval."""
    return max(a, min(b, x))


def clamp01(x: float) -> float:
    """Clamp ``x`` to the ``[0,1]`` interval."""
    return clamp(float(x), 0.0, 1.0)


def within_range(val: float, lower: float, upper: float, tol: float = 1e-9) -> bool:
    """Return ``True`` if ``val`` lies in ``[lower, upper]`` within ``tol``.

    The comparison uses absolute differences instead of :func:`math.isclose`.
    """

    v = float(val)
    return lower <= v <= upper or abs(v - lower) <= tol or abs(v - upper) <= tol


def _norm01(x: float, lo: float, hi: float) -> float:
    """Normalize ``x`` to the unit interval given bounds.

    ``lo`` and ``hi`` delimit the original value range. When ``hi`` is not
    greater than ``lo`` the function returns ``0.0`` to avoid division by
    zero. The result is clamped to ``[0,1]``.
    """

    if hi <= lo:
        return 0.0
    return clamp01((float(x) - float(lo)) / (float(hi) - float(lo)))


def _similarity_abs(a: float, b: float, lo: float, hi: float) -> float:
    """Return absolute similarity of ``a`` and ``b`` over ``[lo, hi]``.

    It computes ``1`` minus the normalized absolute difference between
    ``a`` and ``b``. Values are scaled using :func:`_norm01` so the result
    falls within ``[0,1]``.
    """

    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, hi - lo)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    """Return the arithmetic mean of ``xs`` or ``default`` if empty."""
    try:
        return fmean(xs)
    except StatisticsError:
        return float(default)


def list_pvariance(xs: Iterable[float], default: float = 0.0) -> float:
    """Return the population variance of ``xs`` or ``default`` if empty."""
    np = get_numpy()
    if np is not None:
        arr = np.fromiter(xs, dtype=float)
        return float(np.var(arr)) if arr.size else float(default)
    try:
        return pvariance(xs)
    except StatisticsError:
        return float(default)


def kahan_sum_nd(
    values: Iterable[Sequence[float]], dims: int
) -> tuple[float, ...]:
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
    np = get_numpy()
    neigh = node.G[node.n]
    return neighbor_phase_mean_list(
        neigh, trig.cos, trig.sin, np=np, fallback=fallback
    )


def accumulate_cos_sin(
    it: Iterable[tuple[float, float] | None],
) -> tuple[float, float, bool]:
    """Accumulate cosine and sine pairs with compensated summation.

    ``it`` yields optional ``(cos, sin)`` tuples. Entries with ``None``
    components are ignored. The returned values are the compensated sums of
    cosines and sines along with a flag indicating whether any pair was
    processed.
    """
    sum_cos = 0.0
    sum_sin = 0.0
    comp_cos = 0.0
    comp_sin = 0.0
    processed = False
    for cs in it:
        if cs is None:
            continue
        c, s = cs
        if c is None or s is None:
            continue
        processed = True
        t = sum_cos + c
        if abs(sum_cos) >= abs(c):
            comp_cos += (sum_cos - t) + c
        else:
            comp_cos += (c - t) + sum_cos
        sum_cos = t

        t = sum_sin + s
        if abs(sum_sin) >= abs(s):
            comp_sin += (sum_sin - t) + s
        else:
            comp_sin += (s - t) + sum_sin
        sum_sin = t

    return sum_cos + comp_cos, sum_sin + comp_sin, processed


def _accumulate_cos_sin(
    it: Iterable[tuple[float, float] | None],
) -> tuple[float, float, bool]:
    """Legacy wrapper for :func:`accumulate_cos_sin`."""
    return accumulate_cos_sin(it)


def _phase_mean_from_iter(
    it: Iterable[tuple[float, float] | None], fallback: float
) -> float:
    """Return circular mean from an iterator of cosine/sine pairs.

    ``it`` yields optional ``(cos, sin)`` tuples. ``fallback`` is returned if
    no valid pairs are processed.
    """
    sum_cos, sum_sin, processed = accumulate_cos_sin(it)
    if not processed:
        return fallback
    return math.atan2(sum_sin, sum_cos)


def neighbor_phase_mean_list(
    neigh: Sequence[Any],
    cos_th: dict[Any, float],
    sin_th: dict[Any, float],
    np=None,
    fallback: float = 0.0,
) -> float:
    """Return circular mean of neighbour phases from cosine/sine mappings.

    When ``np`` (NumPy) is provided, ``np.fromiter`` is used to build cosine and
    sine arrays directly from a generator, avoiding the creation of an
    intermediate Python list of ``(cos, sin)`` pairs. Otherwise, cosine and
    sine values are accumulated via :func:`accumulate_cos_sin` using a running
    Kahan summation.
    """
    if np is None:
        np = get_numpy()

    def _iter_pairs():
        for v in neigh:
            c = cos_th.get(v)
            s = sin_th.get(v)
            if c is not None and s is not None:
                yield c, s

    pairs = _iter_pairs()

    if np is not None:
        flat_pairs = (val for pair in pairs for val in pair)
        arr = np.fromiter(flat_pairs, dtype=float)
        if arr.size:
            cos_arr = arr[0::2]
            sin_arr = arr[1::2]
            mean_cos = float(np.mean(cos_arr))
            mean_sin = float(np.mean(sin_arr))
            return float(np.arctan2(mean_sin, mean_cos))
        return fallback

    sum_cos, sum_sin, processed = accumulate_cos_sin(pairs)
    if not processed:
        return fallback
    return math.atan2(sum_sin, sum_cos)


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
