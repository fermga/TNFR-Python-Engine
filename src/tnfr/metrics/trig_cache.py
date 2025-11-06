"""Trigonometric caches for TNFR metrics.

The cosine/sine storage helpers live here to keep :mod:`tnfr.metrics.trig`
focused on pure mathematical utilities (phase means, compensated sums, etc.).
"""

from __future__ import annotations

from ..compat.dataclass import dataclass
import hashlib
import math
import struct
from typing import Any, Iterable, Mapping

from ..alias import get_theta_attr
from ..types import GraphLike, NodeAttrMap
from ..utils import edge_version_cache, get_numpy

__all__ = ("TrigCache", "compute_theta_trig", "get_trig_cache", "_compute_trig_python")


@dataclass(slots=True)
class TrigCache:
    """Container for cached trigonometric values per node."""

    cos: dict[Any, float]
    sin: dict[Any, float]
    theta: dict[Any, float]
    theta_checksums: dict[Any, bytes]
    order: tuple[Any, ...]
    cos_values: Any
    sin_values: Any
    theta_values: Any
    index: dict[Any, int]
    edge_src: Any | None = None
    edge_dst: Any | None = None


def _iter_theta_pairs(
    nodes: Iterable[tuple[Any, NodeAttrMap | float]],
) -> Iterable[tuple[Any, float]]:
    """Yield ``(node, θ)`` pairs from ``nodes``."""

    for n, data in nodes:
        if isinstance(data, Mapping):
            yield n, get_theta_attr(data, 0.0) or 0.0
        else:
            yield n, float(data)


def _compute_trig_python(
    nodes: Iterable[tuple[Any, NodeAttrMap | float]],
) -> TrigCache:
    """Compute trigonometric mappings using pure Python."""

    pairs = list(_iter_theta_pairs(nodes))

    cos_th: dict[Any, float] = {}
    sin_th: dict[Any, float] = {}
    thetas: dict[Any, float] = {}
    theta_checksums: dict[Any, bytes] = {}
    order_list: list[Any] = []

    for n, th in pairs:
        order_list.append(n)
        thetas[n] = th
        cos_th[n] = math.cos(th)
        sin_th[n] = math.sin(th)
        theta_checksums[n] = _theta_checksum(th)

    order = tuple(order_list)
    cos_values = tuple(cos_th[n] for n in order)
    sin_values = tuple(sin_th[n] for n in order)
    theta_values = tuple(thetas[n] for n in order)
    index = {n: i for i, n in enumerate(order)}

    return TrigCache(
        cos=cos_th,
        sin=sin_th,
        theta=thetas,
        theta_checksums=theta_checksums,
        order=order,
        cos_values=cos_values,
        sin_values=sin_values,
        theta_values=theta_values,
        index=index,
        edge_src=None,
        edge_dst=None,
    )


def compute_theta_trig(
    nodes: Iterable[tuple[Any, NodeAttrMap | float]],
    np: Any | None = None,
) -> TrigCache:
    """Return trigonometric mappings of ``θ`` per node."""

    if np is None:
        np = get_numpy()
    if np is None or not all(hasattr(np, attr) for attr in ("fromiter", "cos", "sin")):
        return _compute_trig_python(nodes)

    pairs = list(_iter_theta_pairs(nodes))
    if not pairs:
        return TrigCache(
            cos={},
            sin={},
            theta={},
            theta_checksums={},
            order=(),
            cos_values=(),
            sin_values=(),
            theta_values=(),
            index={},
            edge_src=None,
            edge_dst=None,
        )

    node_list, theta_vals = zip(*pairs)
    node_list = tuple(node_list)
    theta_arr = np.fromiter(theta_vals, dtype=float)
    cos_arr = np.cos(theta_arr)
    sin_arr = np.sin(theta_arr)

    cos_th = dict(zip(node_list, map(float, cos_arr)))
    sin_th = dict(zip(node_list, map(float, sin_arr)))
    thetas = dict(zip(node_list, map(float, theta_arr)))
    theta_checksums = {node: _theta_checksum(float(theta)) for node, theta in pairs}
    index = {n: i for i, n in enumerate(node_list)}
    return TrigCache(
        cos=cos_th,
        sin=sin_th,
        theta=thetas,
        theta_checksums=theta_checksums,
        order=node_list,
        cos_values=cos_arr,
        sin_values=sin_arr,
        theta_values=theta_arr,
        index=index,
        edge_src=None,
        edge_dst=None,
    )


def _build_trig_cache(G: GraphLike, np: Any | None = None) -> TrigCache:
    """Construct trigonometric cache for ``G``."""

    return compute_theta_trig(G.nodes(data=True), np=np)


def get_trig_cache(
    G: GraphLike,
    *,
    np: Any | None = None,
    cache_size: int | None = 128,
) -> TrigCache:
    """Return cached cosines and sines of ``θ`` per node.

    This function maintains a cache of trigonometric values to avoid repeated
    cos(θ) and sin(θ) computations across Si, coherence, and ΔNFR calculations.
    The cache uses version-based invalidation triggered by theta attribute changes.

    Cache Strategy
    --------------
    - **Key**: ``("_trig", version)`` where version increments on theta changes
    - **Invalidation**: Checksum-based detection of theta attribute updates
    - **Capacity**: Controlled by ``cache_size`` parameter (default: 128)
    - **Scope**: Graph-wide, shared across all metrics computations

    The cache maintains both dict (for sparse access) and array (for vectorized
    operations) representations of the trigonometric values.

    Parameters
    ----------
    G : GraphLike
        Graph whose node theta attributes are cached.
    np : Any or None, optional
        NumPy module for array-based storage. Falls back to dict if None.
    cache_size : int or None, optional
        Maximum cache entries. Default: 128. None for unlimited.

    Returns
    -------
    TrigCache
        Container with cos/sin mappings and optional array representations.
        See TrigCache dataclass for field documentation.
    """

    if np is None:
        np = get_numpy()
    graph = G.graph
    version = graph.setdefault("_trig_version", 0)
    key = ("_trig", version)

    def builder() -> TrigCache:
        return _build_trig_cache(G, np=np)

    trig = edge_version_cache(G, key, builder, max_entries=cache_size)
    current_checksums = _graph_theta_checksums(G)
    trig_checksums = getattr(trig, "theta_checksums", None)
    if trig_checksums is None:
        trig_checksums = {}

    # Checksum-based invalidation: detect theta attribute changes
    if trig_checksums != current_checksums:
        version = version + 1
        graph["_trig_version"] = version
        key = ("_trig", version)
        trig = edge_version_cache(G, key, builder, max_entries=cache_size)
        trig_checksums = getattr(trig, "theta_checksums", None)
        if trig_checksums is None:
            trig_checksums = {}
        if trig_checksums != current_checksums:
            current_checksums = _graph_theta_checksums(G)
            if trig_checksums != current_checksums:
                return trig
    return trig


def _theta_checksum(theta: float) -> bytes:
    """Return a deterministic checksum for ``theta``."""

    packed = struct.pack("!d", float(theta))
    return hashlib.blake2b(packed, digest_size=8).digest()


def _graph_theta_checksums(G: GraphLike) -> dict[Any, bytes]:
    """Return checksum snapshot of the graph's current ``θ`` values."""

    checksums: dict[Any, bytes] = {}
    for node, theta in _iter_theta_pairs(G.nodes(data=True)):
        checksums[node] = _theta_checksum(theta)
    return checksums
