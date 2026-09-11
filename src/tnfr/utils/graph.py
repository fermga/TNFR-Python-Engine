"""Utilities for graph-level bookkeeping shared by TNFR components."""

from __future__ import annotations

import hashlib
import warnings
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

from ..types import GraphLike, TNFRGraph

__all__ = (
    "get_graph",
    "get_graph_mapping",
    "get_graph_hash",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
    "GraphLike",
)


def get_graph_hash(G: Any, weight: str | None = "weight") -> str:
    """Return a hash of the graph topology (nodes and edges).

    This hash is sensitive to:
    - Node set (sorted)
    - Edge set (sorted)
    - Edge weights (if weight attribute provided)
    - Graph directedness

    It is used for caching topology-dependent computations like
    Laplacian spectra or distance matrices.
    """
    hasher = hashlib.blake2b(digest_size=16)

    # Hash directedness
    is_directed = G.is_directed() if hasattr(G, "is_directed") else False
    hasher.update(b"directed" if is_directed else b"undirected")

    # Hash nodes (sorted)
    # We use str(n) to handle arbitrary node types
    nodes = sorted(str(n) for n in G.nodes())
    for n in nodes:
        hasher.update(n.encode("utf-8"))

    # Hash edges (sorted)
    edges = []
    if hasattr(G, "edges"):
        for u, v, d in G.edges(data=True):
            w = d.get(weight, 1.0) if weight else 1.0
            # Canonical edge order for undirected
            u_str, v_str = str(u), str(v)
            if not is_directed and u_str > v_str:
                u_str, v_str = v_str, u_str
            edges.append((u_str, v_str, float(w)))

    edges.sort()
    for u, v, w in edges:
        hasher.update(f"{u}:{v}:{w}".encode("utf-8"))

    return hasher.hexdigest()


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
