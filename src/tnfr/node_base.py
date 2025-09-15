from __future__ import annotations

from typing import Iterable, MutableMapping
from .helpers import ensure_node_offset_map


class NodeBase:
    """Common helpers for TNFR nodes."""

    graph: object
    epi_kind: str

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
        return nodes if nodes is not None else [self]
