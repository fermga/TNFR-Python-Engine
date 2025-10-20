"""Trigonometric caches for TNFR metrics.

The cosine/sine storage helpers live here to keep :mod:`tnfr.metrics.trig`
focused on pure mathematical utilities (phase means, compensated sums, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..alias import get_theta_attr
from ..types import GraphLike
from ..utils import edge_version_cache, get_numpy

__all__ = ("TrigCache", "compute_theta_trig", "get_trig_cache", "_compute_trig_python")


@dataclass(slots=True)
class TrigCache:
    """Container for cached trigonometric values per node."""

    cos: dict[Any, float]
    sin: dict[Any, float]
    theta: dict[Any, float]


def _iter_theta_pairs(
    nodes: Iterable[tuple[Any, Mapping[str, Any] | float]],
) -> Iterable[tuple[Any, float]]:
    """Yield ``(node, θ)`` pairs from ``nodes``."""

    for n, data in nodes:
        if isinstance(data, Mapping):
            yield n, get_theta_attr(data, 0.0) or 0.0
        else:
            yield n, float(data)


def _compute_trig_python(
    nodes: Iterable[tuple[Any, Mapping[str, Any] | float]],
) -> TrigCache:
    """Compute trigonometric mappings using pure Python."""

    cos_th: dict[Any, float] = {}
    sin_th: dict[Any, float] = {}
    thetas: dict[Any, float] = {}
    for n, th in _iter_theta_pairs(nodes):
        thetas[n] = th
        cos_th[n] = math.cos(th)
        sin_th[n] = math.sin(th)
    return TrigCache(cos=cos_th, sin=sin_th, theta=thetas)


def compute_theta_trig(
    nodes: Iterable[tuple[Any, Mapping[str, Any] | float]],
    np: Any | None = None,
) -> TrigCache:
    """Return trigonometric mappings of ``θ`` per node."""

    if np is None:
        np = get_numpy()
    if np is None or not all(hasattr(np, attr) for attr in ("fromiter", "cos", "sin")):
        return _compute_trig_python(nodes)

    pairs = list(_iter_theta_pairs(nodes))
    if not pairs:
        return TrigCache(cos={}, sin={}, theta={})

    node_list, theta_vals = zip(*pairs)
    theta_arr = np.fromiter(theta_vals, dtype=float)
    cos_arr = np.cos(theta_arr)
    sin_arr = np.sin(theta_arr)

    cos_th = dict(zip(node_list, map(float, cos_arr)))
    sin_th = dict(zip(node_list, map(float, sin_arr)))
    thetas = dict(zip(node_list, map(float, theta_arr)))
    return TrigCache(cos=cos_th, sin=sin_th, theta=thetas)


def _build_trig_cache(G: GraphLike, np: Any | None = None) -> TrigCache:
    """Construct trigonometric cache for ``G``."""

    return compute_theta_trig(G.nodes(data=True), np=np)


def get_trig_cache(
    G: GraphLike,
    *,
    np: Any | None = None,
    cache_size: int | None = 128,
) -> TrigCache:
    """Return cached cosines and sines of ``θ`` per node."""

    if np is None:
        np = get_numpy()
    version = G.graph.setdefault("_trig_version", 0)
    key = ("_trig", version)
    return edge_version_cache(
        G,
        key,
        lambda: _build_trig_cache(G, np=np),
        max_entries=cache_size,
    )
