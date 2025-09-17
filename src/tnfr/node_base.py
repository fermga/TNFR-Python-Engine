from __future__ import annotations

import weakref
from threading import RLock
from typing import Iterable, MutableMapping

from .helpers import cached_node_list, ensure_node_offset_map

_NODE_LOCK = RLock()
_PLAIN_GRAPH_REGISTRY: dict[int, set[int]] = {}
_NODE_LOOKUP: "weakref.WeakValueDictionary[int, NodeBase]" = weakref.WeakValueDictionary()
_NODE_REGISTRATIONS: "weakref.WeakKeyDictionary[NodeBase, int]" = weakref.WeakKeyDictionary()
_NODE_FINALIZERS: "weakref.WeakKeyDictionary[NodeBase, object]" = weakref.WeakKeyDictionary()


def _remove_node_from_graph(node_id: int, graph_id: int) -> None:
    with _NODE_LOCK:
        registry = _PLAIN_GRAPH_REGISTRY.get(graph_id)
        if registry is None:
            return
        registry.discard(node_id)
        if not registry:
            _PLAIN_GRAPH_REGISTRY.pop(graph_id, None)


class NodeBase:
    """Common helpers for TNFR nodes."""

    graph: object
    epi_kind: str

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "graph":
            self._sync_graph_registration(value)

    def _sync_graph_registration(self, graph: object) -> None:
        with _NODE_LOCK:
            node_id = id(self)
            graph_id_old = _NODE_REGISTRATIONS.pop(self, None)
            if graph_id_old is not None:
                registry = _PLAIN_GRAPH_REGISTRY.get(graph_id_old)
                if registry is not None:
                    registry.discard(node_id)
                    if not registry:
                        _PLAIN_GRAPH_REGISTRY.pop(graph_id_old, None)
            finalizer = _NODE_FINALIZERS.pop(self, None)
            if finalizer is not None:
                finalizer.detach()
            if graph is None or hasattr(graph, "nodes"):
                return
            data = graph.graph if hasattr(graph, "graph") else graph
            if isinstance(data, MutableMapping):
                graph_id_new = id(data)
                registry = _PLAIN_GRAPH_REGISTRY.setdefault(graph_id_new, set())
                registry.add(node_id)
                _NODE_LOOKUP[node_id] = self
                _NODE_REGISTRATIONS[self] = graph_id_new
                _NODE_FINALIZERS[self] = weakref.finalize(
                    self, _remove_node_from_graph, node_id, graph_id_new
                )

    def _glyph_storage(self) -> MutableMapping[str, object]:
        """Return mapping holding glyph history for this node."""
        raise NotImplementedError

    def offset(self) -> int:
        """Return positional offset for this node within its graph."""
        mapping = ensure_node_offset_map(self.graph)
        return mapping.get(self, 0)

    def all_nodes(self) -> Iterable["NodeBase"]:
        """Iterate all nodes registered on ``self.graph``."""
        graph = self.graph
        data = graph.graph if hasattr(graph, "graph") else graph
        nodes = data.get("_all_nodes")
        if nodes is not None:
            return nodes

        target_graph = None
        if hasattr(graph, "nodes"):
            target_graph = graph
        else:
            maybe_graph = getattr(self, "G", None)
            if maybe_graph is not None and hasattr(maybe_graph, "graph"):
                try:
                    if maybe_graph.graph is data:
                        target_graph = maybe_graph
                except AttributeError:
                    pass

        if target_graph is not None:
            return cached_node_list(target_graph)

        cache = data.get("_node_list_cache")
        if cache is not None:
            cached = getattr(cache, "nodes", None)
            if cached is not None:
                return cached

        if isinstance(data, MutableMapping) and not hasattr(graph, "nodes"):
            self._sync_graph_registration(graph)
            with _NODE_LOCK:
                registry = _PLAIN_GRAPH_REGISTRY.get(id(data))
                if registry:
                    nodes: list[NodeBase] = []
                    cleaned = False
                    for node_id in list(registry):
                        node_obj = _NODE_LOOKUP.get(node_id)
                        if node_obj is None:
                            registry.discard(node_id)
                            cleaned = True
                            continue
                        nodes.append(node_obj)
                    if cleaned and not registry:
                        _PLAIN_GRAPH_REGISTRY.pop(id(data), None)
                    if nodes:
                        return tuple(nodes)

        return [self]
