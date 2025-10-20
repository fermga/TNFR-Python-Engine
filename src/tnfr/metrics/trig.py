"""Trigonometric helpers shared across metrics and helpers.

This module focuses on mathematical utilities (means, compensated sums, etc.).
Caching of cosine/sine values lives in :mod:`tnfr.metrics.trig_cache`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import tee
from typing import TYPE_CHECKING, Any, overload, cast

from ..helpers.numeric import kahan_sum_nd
from ..utils import cached_import, get_numpy
from ..types import NodeId, Phase, TNFRGraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..node import NodeProtocol

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

    processed = False

    def iter_real_pairs() -> Iterator[tuple[float, float]]:
        nonlocal processed
        for cs in it:
            if cs is None:
                continue
            c, s = cs
            if c is None or s is None:
                continue
            try:
                c_val = float(c)
                s_val = float(s)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(c_val) and math.isfinite(s_val)):
                continue
            processed = True
            yield (c_val, s_val)

    sum_cos, sum_sin = kahan_sum_nd(iter_real_pairs(), dims=2)

    if not processed:
        return 0.0, 0.0, False

    return sum_cos, sum_sin, True


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
    np: Any | None,
    fallback: float,
) -> float:
    """Return circular mean of neighbour phases given trig mappings."""

    def _iter_pairs() -> Iterator[tuple[float, float]]:
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
    obj: "NodeProtocol" | Sequence[Any],
    cos_map: dict[Any, float] | None = None,
    sin_map: dict[Any, float] | None = None,
    np: Any | None = None,
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
        node = cast("NodeProtocol", obj)
        if getattr(node, "G", None) is None:
            raise TypeError(
                "neighbor_phase_mean requires nodes bound to a graph"
            )
        from .trig_cache import get_trig_cache

        trig = get_trig_cache(node.G)
        fallback = trig.theta.get(node.n, fallback)
        cos_map = trig.cos
        sin_map = trig.sin
        neigh = node.G[node.n]
    else:
        neigh = cast(Sequence[Any], obj)

    return _neighbor_phase_mean_core(neigh, cos_map, sin_map, np, fallback)


def neighbor_phase_mean_list(
    neigh: Sequence[Any],
    cos_th: dict[Any, float],
    sin_th: dict[Any, float],
    np: Any | None = None,
    fallback: float = 0.0,
) -> float:
    """Return circular mean of neighbour phases from cosine/sine mappings.

    This is a thin wrapper over :func:`_neighbor_phase_mean_generic` that
    operates on explicit neighbour lists.
    """

    return _neighbor_phase_mean_generic(
        neigh, cos_map=cos_th, sin_map=sin_th, np=np, fallback=fallback
    )


@overload
def neighbor_phase_mean(obj: "NodeProtocol", n: None = ...) -> Phase:
    ...


@overload
def neighbor_phase_mean(obj: TNFRGraph, n: NodeId) -> Phase:
    ...


def neighbor_phase_mean(
    obj: "NodeProtocol" | TNFRGraph, n: NodeId | None = None
) -> Phase:
    """Circular mean of neighbour phases for ``obj``.

    Parameters
    ----------
    obj:
        Either a :class:`~tnfr.node.NodeProtocol` instance bound to a graph or a
        :class:`~tnfr.types.TNFRGraph` from which the node ``n`` will be wrapped.
    n:
        Optional node identifier. Required when ``obj`` is a graph. Providing a
        node identifier for a node object raises :class:`TypeError`.
    """

    NodeNX = cached_import("tnfr.node", "NodeNX")
    if NodeNX is None:
        raise ImportError("NodeNX is unavailable")
    if n is None:
        if hasattr(obj, "nodes"):
            raise TypeError(
                "neighbor_phase_mean requires a node identifier when passing a graph"
            )
        node = obj
    else:
        if hasattr(obj, "nodes"):
            node = NodeNX(obj, n)
        else:
            raise TypeError(
                "neighbor_phase_mean received a node and an explicit identifier"
            )
    return _neighbor_phase_mean_generic(node)
