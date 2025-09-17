from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from ..cache import (
    _node_repr,
    _node_repr_digest,
    node_set_checksum,
)
from ..graph_utils import get_graph

__all__ = (
    "cached_node_list",
    "ensure_node_index_map",
    "ensure_node_offset_map",
)


@dataclass(slots=True)
class NodeCache:
    """Container for cached node data."""

    checksum: str
    nodes: tuple[Any, ...]
    sorted_nodes: tuple[Any, ...] | None = None
    idx: dict[Any, int] | None = None
    offset: dict[Any, int] | None = None

    @property
    def n(self) -> int:
        return len(self.nodes)


def _update_node_cache(
    graph: Any,
    nodes: tuple[Any, ...],
    key: str,
    *,
    checksum: str,
    sorted_nodes: tuple[Any, ...] | None = None,
) -> None:
    """Store ``nodes`` and ``checksum`` in ``graph`` under ``key``."""
    graph[f"{key}_cache"] = NodeCache(
        checksum=checksum, nodes=nodes, sorted_nodes=sorted_nodes
    )
    graph[f"{key}_checksum"] = checksum


def _refresh_node_list_cache(
    G: nx.Graph,
    graph: Any,
    *,
    sort_nodes: bool,
    current_n: int,
) -> tuple[Any, ...]:
    """Refresh the cached node list and return the nodes."""
    nodes = tuple(G.nodes())
    checksum = node_set_checksum(G, nodes, store=True)
    sorted_nodes = tuple(sorted(nodes, key=_node_repr)) if sort_nodes else None
    _update_node_cache(
        graph,
        nodes,
        "_node_list",
        checksum=checksum,
        sorted_nodes=sorted_nodes,
    )
    graph["_node_list_len"] = current_n
    return nodes


def _reuse_node_list_cache(
    graph: Any,
    cache: NodeCache,
    nodes: tuple[Any, ...],
    sorted_nodes: tuple[Any, ...] | None,
    *,
    sort_nodes: bool,
    new_checksum: str | None,
) -> None:
    """Reuse existing node cache and record its checksum if missing."""
    checksum = cache.checksum if new_checksum is None else new_checksum
    if sort_nodes and sorted_nodes is None:
        sorted_nodes = tuple(sorted(nodes, key=_node_repr))
    _update_node_cache(
        graph,
        nodes,
        "_node_list",
        checksum=checksum,
        sorted_nodes=sorted_nodes,
    )


def _cache_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Cache and return the tuple of nodes for ``G``.

    A :class:`NodeCache` instance is stored in ``G.graph`` under
    ``"_node_list_cache"``. The cache refreshes when the node set changes or when
    the optional ``"_node_list_dirty"`` flag is set to ``True``.
    """
    graph = get_graph(G)
    cache: NodeCache | None = graph.get("_node_list_cache")
    nodes = cache.nodes if cache else None
    sorted_nodes = cache.sorted_nodes if cache else None
    stored_len = graph.get("_node_list_len")
    current_n = G.number_of_nodes()
    dirty = bool(graph.pop("_node_list_dirty", False))

    # Determine if inexpensive checks already mark the cache as invalid
    invalid = nodes is None or stored_len != current_n or dirty
    new_checksum: str | None = None

    if not invalid and cache:
        # Only compute the checksum when the quick checks pass.
        # This avoids hashing the node set unless we suspect a change.
        new_checksum = node_set_checksum(G)
        invalid = cache.checksum != new_checksum

    sort_nodes = bool(graph.get("SORT_NODES", False))

    if invalid:
        nodes = _refresh_node_list_cache(
            G, graph, sort_nodes=sort_nodes, current_n=current_n
        )
    elif cache and "_node_list_checksum" not in graph:
        _reuse_node_list_cache(
            graph,
            cache,
            nodes,
            sorted_nodes,
            sort_nodes=sort_nodes,
            new_checksum=new_checksum,
        )
    else:
        if sort_nodes and sorted_nodes is None and cache is not None:
            cache.sorted_nodes = tuple(sorted(nodes, key=_node_repr))
    return nodes


def cached_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Public wrapper returning the cached node tuple for ``G``."""
    return _cache_node_list(G)


def _ensure_node_map(G, *, attrs: tuple[str, ...], sort: bool = False) -> dict[Any, int]:
    """Return cached node-to-index/offset mappings stored on ``NodeCache``.

    ``attrs`` selects the attributes on :class:`NodeCache` used to store the
    mapping(s). ``sort`` controls whether nodes are ordered by their string
    representation before assigning indices.
    """
    graph = G.graph
    _cache_node_list(G)
    cache: NodeCache = graph["_node_list_cache"]

    missing = [attr for attr in attrs if getattr(cache, attr) is None]
    if missing:
        if sort:
            nodes = cache.sorted_nodes
            if nodes is None:
                nodes = cache.sorted_nodes = tuple(sorted(cache.nodes, key=_node_repr))
        else:
            nodes = cache.nodes
        mappings: dict[str, dict[Any, int]] = {attr: {} for attr in missing}
        for idx, node in enumerate(nodes):
            for attr in missing:
                mappings[attr][node] = idx
        for attr in missing:
            setattr(cache, attr, mappings[attr])
    return getattr(cache, attrs[0])


def ensure_node_index_map(G) -> dict[Any, int]:
    """Return cached node-to-index mapping for ``G``."""
    return _ensure_node_map(G, attrs=("idx",), sort=False)


def ensure_node_offset_map(G) -> dict[Any, int]:
    """Return cached node-to-offset mapping for ``G``."""
    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, attrs=("offset",), sort=sort)

