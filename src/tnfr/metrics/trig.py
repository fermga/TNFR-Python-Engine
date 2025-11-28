"""Trigonometric helpers shared across metrics and helpers.

This module focuses on mathematical utilities (means, compensated sums, etc.).
Caching of cosine/sine values lives in :mod:`tnfr.metrics.trig_cache`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import tee
from typing import TYPE_CHECKING, Any, cast, overload

from ..utils import kahan_sum_nd
from ..types import NodeId, Phase, TNFRGraph
from ..utils import cached_import, get_numpy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..node import NodeProtocol

__all__ = (
    "accumulate_cos_sin",
    "_phase_mean_from_iter",
    "_neighbor_phase_mean_core",
    "_neighbor_phase_mean_generic",
    "neighbor_phase_mean_bulk",
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


def _phase_mean_from_iter(it: Iterable[tuple[float, float] | None], fallback: float) -> float:
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
    """Compute the neighbour phase mean via :func:`_neighbor_phase_mean_core`.

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
            raise TypeError("neighbor_phase_mean requires nodes bound to a graph")
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


def neighbor_phase_mean_bulk(
    edge_src: Any,
    edge_dst: Any,
    *,
    cos_values: Any,
    sin_values: Any,
    theta_values: Any,
    node_count: int,
    np: Any,
    neighbor_cos_sum: Any | None = None,
    neighbor_sin_sum: Any | None = None,
    neighbor_counts: Any | None = None,
    mean_cos: Any | None = None,
    mean_sin: Any | None = None,
) -> tuple[Any, Any]:
    """Vectorised neighbour phase means for all nodes in a graph.

    Parameters
    ----------
    edge_src, edge_dst:
        Arrays describing the source (neighbour) and destination (node) indices
        for each edge contribution. They must have matching shapes.
    cos_values, sin_values:
        Arrays containing the cosine and sine values of each node's phase. The
        arrays must be indexed using the same positional indices referenced by
        ``edge_src``.
    theta_values:
        Array with the baseline phase for each node. Positions that do not have
        neighbours reuse this baseline as their mean phase.
    node_count:
        Total number of nodes represented in ``theta_values``.
    np:
        Numpy module used to materialise the vectorised operations.

    Optional buffers
    -----------------
    neighbor_cos_sum, neighbor_sin_sum, neighbor_counts, mean_cos, mean_sin:
        Preallocated arrays sized ``node_count`` reused to accumulate the
        neighbour cosine/sine sums, neighbour sample counts, and the averaged
        cosine/sine vectors. When omitted, the helper materialises fresh
        buffers that match the previous semantics.

    Returns
    -------
    tuple[Any, Any]
        Tuple ``(mean_theta, has_neighbors)`` where ``mean_theta`` contains the
        circular mean of neighbour phases for every node and ``has_neighbors``
        is a boolean mask identifying which nodes contributed at least one
        neighbour sample.
    """

    if node_count <= 0:
        empty_mean = np.zeros(0, dtype=float)
        return empty_mean, empty_mean.astype(bool)

    edge_src_arr = np.asarray(edge_src, dtype=np.intp)
    edge_dst_arr = np.asarray(edge_dst, dtype=np.intp)

    if edge_src_arr.shape != edge_dst_arr.shape:
        raise ValueError("edge_src and edge_dst must share the same shape")

    theta_arr = np.asarray(theta_values, dtype=float)
    if theta_arr.ndim != 1 or theta_arr.size != node_count:
        raise ValueError("theta_values must be a 1-D array matching node_count")

    cos_arr = np.asarray(cos_values, dtype=float)
    sin_arr = np.asarray(sin_values, dtype=float)
    if cos_arr.ndim != 1 or cos_arr.size != node_count:
        raise ValueError("cos_values must be a 1-D array matching node_count")
    if sin_arr.ndim != 1 or sin_arr.size != node_count:
        raise ValueError("sin_values must be a 1-D array matching node_count")

    edge_count = edge_dst_arr.size

    def _coerce_buffer(buffer: Any | None, *, name: str) -> tuple[Any, bool]:
        if buffer is None:
            return None, False
        arr = np.array(buffer, dtype=float, copy=False)
        if arr.ndim != 1 or arr.size != node_count:
            raise ValueError(f"{name} must be a 1-D array sized node_count")
        arr.fill(0.0)
        return arr, True

    neighbor_cos_sum, has_cos_buffer = _coerce_buffer(neighbor_cos_sum, name="neighbor_cos_sum")
    neighbor_sin_sum, has_sin_buffer = _coerce_buffer(neighbor_sin_sum, name="neighbor_sin_sum")
    neighbor_counts, has_count_buffer = _coerce_buffer(neighbor_counts, name="neighbor_counts")

    if edge_count:
        cos_bincount = np.bincount(
            edge_dst_arr,
            weights=cos_arr[edge_src_arr],
            minlength=node_count,
        )
        sin_bincount = np.bincount(
            edge_dst_arr,
            weights=sin_arr[edge_src_arr],
            minlength=node_count,
        )
        count_bincount = np.bincount(
            edge_dst_arr,
            minlength=node_count,
        ).astype(float, copy=False)

        if not has_cos_buffer:
            neighbor_cos_sum = cos_bincount
        else:
            np.copyto(neighbor_cos_sum, cos_bincount)

        if not has_sin_buffer:
            neighbor_sin_sum = sin_bincount
        else:
            np.copyto(neighbor_sin_sum, sin_bincount)

        if not has_count_buffer:
            neighbor_counts = count_bincount
        else:
            np.copyto(neighbor_counts, count_bincount)
    else:
        if neighbor_cos_sum is None:
            neighbor_cos_sum = np.zeros(node_count, dtype=float)
        if neighbor_sin_sum is None:
            neighbor_sin_sum = np.zeros(node_count, dtype=float)
        if neighbor_counts is None:
            neighbor_counts = np.zeros(node_count, dtype=float)

    has_neighbors = neighbor_counts > 0.0

    mean_cos, _ = _coerce_buffer(mean_cos, name="mean_cos")
    mean_sin, _ = _coerce_buffer(mean_sin, name="mean_sin")

    if mean_cos is None:
        mean_cos = np.zeros(node_count, dtype=float)
    if mean_sin is None:
        mean_sin = np.zeros(node_count, dtype=float)

    if edge_count:
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(
                neighbor_cos_sum,
                neighbor_counts,
                out=mean_cos,
                where=has_neighbors,
            )
            np.divide(
                neighbor_sin_sum,
                neighbor_counts,
                out=mean_sin,
                where=has_neighbors,
            )

    mean_theta = np.where(has_neighbors, np.arctan2(mean_sin, mean_cos), theta_arr)
    return mean_theta, has_neighbors


@overload
def neighbor_phase_mean(obj: "NodeProtocol", n: None = ...) -> Phase: ...


@overload
def neighbor_phase_mean(obj: TNFRGraph, n: NodeId) -> Phase: ...


def neighbor_phase_mean(obj: "NodeProtocol" | TNFRGraph, n: NodeId | None = None) -> Phase:
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
            raise TypeError("neighbor_phase_mean requires a node identifier when passing a graph")
        node = obj
    else:
        if hasattr(obj, "nodes"):
            node = NodeNX(obj, n)
        else:
            raise TypeError("neighbor_phase_mean received a node and an explicit identifier")
    return _neighbor_phase_mean_generic(node)
