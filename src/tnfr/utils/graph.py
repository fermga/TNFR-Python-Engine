"""Utilities for graph-level bookkeeping shared by TNFR components."""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

from ..types import GraphLike, TNFRGraph

__all__ = (
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
    "GraphLike",
)


def get_graph(
    obj: GraphLike | TNFRGraph | MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Return the graph-level metadata mapping for ``obj``.

    ``obj`` must be a :class:`~tnfr.types.TNFRGraph` instance or fulfil the
    :class:`~tnfr.types.GraphLike` protocol. The function normalises access to
    the ``graph`` attribute exposed by ``networkx``-style graphs and wrappers,
    always returning the underlying metadata mapping. A pre-extracted mapping
    is also accepted for legacy call sites.
    """

    graph = getattr(obj, "graph", None)
    if graph is not None:
        return graph
    if isinstance(obj, MutableMapping):
        return obj
    raise TypeError("Unsupported graph object: metadata mapping not accessible")


def get_graph_mapping(
    G: GraphLike | TNFRGraph | MutableMapping[str, Any], key: str, warn_msg: str
) -> Mapping[str, Any] | None:
    """Return an immutable view of ``G``'s stored mapping for ``key``.

    The ``G`` argument follows the :class:`~tnfr.types.GraphLike` protocol, is
    a concrete :class:`~tnfr.types.TNFRGraph` or provides the metadata mapping
    directly. The helper validates that the stored value is a mapping before
    returning a read-only proxy.
    """

    graph = get_graph(G)
    getter = getattr(graph, "get", None)
    if getter is None:
        return None

    data = getter(key)
    if data is None:
        return None
    if not isinstance(data, Mapping):
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
        return None
    return MappingProxyType(data)


def mark_dnfr_prep_dirty(G: GraphLike | TNFRGraph | MutableMapping[str, Any]) -> None:
    """Flag ΔNFR preparation data as stale by marking ``G.graph``.

    ``G`` is constrained to the :class:`~tnfr.types.GraphLike` protocol, a
    concrete :class:`~tnfr.types.TNFRGraph` or an explicit metadata mapping,
    ensuring the metadata storage is available for mutation.
    """

    graph = get_graph(G)
    graph["_dnfr_prep_dirty"] = True


def supports_add_edge(graph: GraphLike | TNFRGraph) -> bool:
    """Return ``True`` if ``graph`` exposes an ``add_edge`` method.

    The ``graph`` parameter must implement :class:`~tnfr.types.GraphLike` or be
    a :class:`~tnfr.types.TNFRGraph`, aligning runtime expectations with the
    type contract enforced throughout the engine.
    """

    return hasattr(graph, "add_edge")
