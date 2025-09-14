from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import networkx as nx  # type: ignore[import-untyped]

from ..json_utils import json_dumps
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Key used to store the node set checksum in a graph's ``graph`` attribute.
NODE_SET_CHECKSUM_KEY = "_node_set_checksum_cache"

__all__ = (
    "NODE_SET_CHECKSUM_KEY",
    "get_graph",
    "get_graph_mapping",
    "node_set_checksum",
    "stable_json",
    "cached_node_list",
    "ensure_node_index_map",
    "ensure_node_offset_map",
)


def get_graph(obj: Any) -> Any:
    """Return ``obj.graph`` if available or ``obj`` otherwise."""
    return getattr(obj, "graph", obj)


def get_graph_mapping(
    G: Any, key: str, warn_msg: str
) -> Mapping[str, Any] | None:
    """Return an immutable view of ``G.graph[key]`` if it is a mapping.

    The mapping is wrapped in :class:`types.MappingProxyType` to prevent
    accidental modification. ``warn_msg`` is emitted via :func:`warnings.warn`
    when the stored value is not a mapping. ``None`` is returned when the key is
    absent or invalid.
    """
    data = G.graph.get(key)
    if data is None:
        return None
    if not isinstance(data, Mapping):
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
        return None
    return MappingProxyType(data)


def _stable_json(obj: Any) -> str:
    """Return a JSON string with deterministic ordering."""
    return json_dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        to_bytes=False,
    )


def stable_json(obj: Any) -> str:
    """Public wrapper returning a stable JSON string for ``obj``."""
    return _stable_json(obj)


@lru_cache(maxsize=1024)
def _node_repr_digest(obj: Any) -> tuple[str, bytes]:
    """Return cached stable representation and digest for ``obj``.

    This single helper centralises caching for node representations and their
    digests, ensuring both values stay in sync.
    """
    repr_ = _stable_json(obj)
    digest = hashlib.blake2b(repr_.encode("utf-8"), digest_size=16).digest()
    return repr_, digest


def clear_node_repr_cache() -> None:
    """Clear cached node representations used for checksums."""
    _node_repr_digest.cache_clear()


def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""
    return _node_repr_digest(n)[0]


def _hash_node(obj: Any) -> bytes:
    """Return a stable digest for ``obj`` used in node checksums."""
    return _node_repr_digest(obj)[1]


def _iter_node_digests(
    nodes: Iterable[Any], *, presorted: bool
) -> Iterable[bytes]:
    """Yield node digests in a deterministic order.

    When ``presorted`` is ``True`` the nodes are assumed to already be sorted
    in a stable manner and their digests are yielded directly. Otherwise,
    the tuple of representation and digest provided by
    :func:`_node_repr_digest` is used to avoid redundant computation.
    """
    if presorted:
        for node in nodes:
            yield _node_repr_digest(node)[1]
    else:
        for _, digest in sorted(
            (_node_repr_digest(n) for n in nodes), key=lambda x: x[0]
        ):
            yield digest


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


def _node_set_checksum_no_nodes(
    G: nx.Graph,
    graph: Any,
    *,
    presorted: bool,
    store: bool,
) -> str:
    """Checksum helper when no explicit node set is provided.

    This isolates the flow that hashes the whole graph, preventing duplicate
    lookups and repeated ``current_nodes`` assignments.
    """
    nodes_view = G.nodes()
    current_nodes = frozenset(nodes_view)
    cached = graph.get(NODE_SET_CHECKSUM_KEY)
    if cached and len(cached) == 3 and cached[2] == current_nodes:
        return cached[1]

    hasher = hashlib.blake2b(digest_size=16)
    for digest in _iter_node_digests(nodes_view, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    if store:
        token = checksum[:16]
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum, current_nodes)
    return checksum


def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = None,
    *,
    presorted: bool = False,
    store: bool = True,
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set.

    Nodes are serialised using :func:`_node_repr`. The helper
    :func:`_iter_node_digests` yields their digests in a deterministic order,
    handling the ``presorted`` and unsorted cases. When ``store`` is ``True``
    the final checksum is cached under ``NODE_SET_CHECKSUM_KEY`` to avoid
    recomputation for unchanged graphs.
    """
    graph = get_graph(G)
    if nodes is None:
        return _node_set_checksum_no_nodes(
            G, graph, presorted=presorted, store=store
        )

    hasher = hashlib.blake2b(digest_size=16)

    # Generate digests in stable order; `_iter_node_digests` sorts when needed
    # unless `presorted` indicates the nodes are already ordered.
    for digest in _iter_node_digests(nodes, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    if store:
        token = checksum[:16]
        cached = graph.get(NODE_SET_CHECKSUM_KEY)
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum)
    return checksum


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

