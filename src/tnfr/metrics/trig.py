"""Trigonometric helpers shared across metrics and helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import tee
from typing import Any

from ..import_utils import cached_import, get_numpy

__all__ = (
    "accumulate_cos_sin",
    "_phase_mean_from_iter",
    "_neighbor_phase_mean_core",
    "_neighbor_phase_mean_generic",
    "neighbor_phase_mean_list",
    "neighbor_phase_mean",
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


def _neighbor_phase_mean_core(
    neigh: Sequence[Any],
    cos_map: dict[Any, float],
    sin_map: dict[Any, float],
    np,
    fallback: float,
) -> float:
    """Return circular mean of neighbour phases given trig mappings."""

    def _iter_pairs():
        for v in neigh:
            c = cos_map.get(v)
            s = sin_map.get(v)
            if c is not None and s is not None:
                yield c, s

    pairs = _iter_pairs()

    if np is not None:
        cos_iter, sin_iter = tee(pairs, 2)
        cos_arr = np.fromiter((c for c, _ in cos_iter), dtype=float)
        sin_arr = np.fromiter((s for _, s in sin_iter), dtype=float)
        if cos_arr.size:
            mean_cos = float(np.mean(cos_arr))
            mean_sin = float(np.mean(sin_arr))
            return float(np.arctan2(mean_sin, mean_cos))
        return fallback

    sum_cos, sum_sin, processed = accumulate_cos_sin(pairs)
    if not processed:
        return fallback
    return math.atan2(sum_sin, sum_cos)


def _neighbor_phase_mean_generic(
    obj,
    cos_map: dict[Any, float] | None = None,
    sin_map: dict[Any, float] | None = None,
    np=None,
    fallback: float = 0.0,
) -> float:
    """Internal helper delegating to :func:`_neighbor_phase_mean_core`.

    ``obj`` may be either a node bound to a graph or a sequence of neighbours.
    When ``cos_map`` and ``sin_map`` are ``None`` the function assumes ``obj`` is
    a node and obtains the required trigonometric mappings from the cached
    structures. Otherwise ``obj`` is treated as an explicit neighbour
    sequence and ``cos_map``/``sin_map`` must be provided.
    """

    if np is None:
        np = get_numpy()

    if cos_map is None or sin_map is None:
        node = obj
        if getattr(node, "G", None) is None:
            raise TypeError(
                "neighbor_phase_mean requires nodes bound to a graph"
            )
        from .trigonometry import get_trig_cache

        trig = get_trig_cache(node.G)
        fallback = trig.theta.get(node.n, fallback)
        cos_map = trig.cos
        sin_map = trig.sin
        neigh = node.G[node.n]
    else:
        neigh = obj

    return _neighbor_phase_mean_core(neigh, cos_map, sin_map, np, fallback)


def neighbor_phase_mean_list(
    neigh: Sequence[Any],
    cos_th: dict[Any, float],
    sin_th: dict[Any, float],
    np=None,
    fallback: float = 0.0,
) -> float:
    """Return circular mean of neighbour phases from cosine/sine mappings.

    This is a thin wrapper over :func:`_neighbor_phase_mean_generic` that
    operates on explicit neighbour lists.
    """

    return _neighbor_phase_mean_generic(
        neigh, cos_map=cos_th, sin_map=sin_th, np=np, fallback=fallback
    )


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases.

    The :class:`NodoNX` import is cached after the first call.
    """

    NodoNX = cached_import("tnfr.node", "NodoNX")
    if NodoNX is None:
        raise ImportError("NodoNX is unavailable")
    node = NodoNX(obj, n) if n is not None else obj
    return _neighbor_phase_mean_generic(node)
