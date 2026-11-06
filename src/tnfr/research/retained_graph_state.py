"""Reconstruct an explicitly recorded primitive graph and execution context.

Finite tagged containers do not authenticate execution or invoke a producer.
Declared caches remain absent; scientific admission belongs to consuming owners.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS
from ._recorded_graph_containers import (
    _require,
    decode_recorded_container,
    encode_recorded_container,
)

_STORAGE = frozenset({"graph", "_node", "_adj", "__networkx_cache__"})
_VIEWS = frozenset({"nodes", "edges", "degree", "adj"})


def _is_recorded_node_label(value):
    """Admit exact integer identities and nonempty canonical child labels."""
    return type(value) is int or (type(value) is str and bool(value))


def _admit_omissions(omissions, graph):
    for index, items in enumerate(omissions):
        _require(type(items) is tuple, "omission inventories must be tuples")
        fields = {"key", "type", "scope"} | ({"owner"} if index == 0 else set())
        seen = set()
        allowed = (GRAPH_RUNTIME_CACHE_KEYS, _VIEWS, {"__networkx_cache__"})[index]
        for item in items:
            _require(
                type(item) is dict
                and set(item) == fields
                and all(type(value) is str and value for value in item.values()),
                "invalid omission metadata",
            )
            key = item["key"]
            _require(
                key in allowed and key not in seen, "invalid or duplicate omitted key"
            )
            if index == 0:
                _require(key not in graph.graph, "omitted cache is also supplied")
                _require(
                    item["owner"] == "tnfr.utils.cache.GRAPH_RUNTIME_CACHE_KEYS",
                    "invalid cache owner",
                )
            seen.add(key)


@dataclass(frozen=True)
class RecordedGraphReconstruction:
    """A fresh graph, preserving recorded persistent context and omissions."""

    graph: nx.Graph
    omitted_rebuildable_caches: tuple
    omitted_networkx_views: tuple
    omitted_networkx_cache: tuple


def restore_recorded_graph_state(record) -> RecordedGraphReconstruction:
    """Rebuild a simple graph with integer or nonempty string node identities.

    Preserve neighbor order and shared undirected edge attributes. Only the
    last-operator instance field is restored. This is a new preparation from
    recorded inputs, not a complete authenticated checkpoint.
    """
    data = decode_recorded_container(record)
    expected = {
        "graph_type",
        "node_order",
        "neighbor_order",
        "graph",
        "nodes",
        "edges",
        "object_state",
        "omitted_rebuildable_caches",
        "omitted_networkx_views",
        "omitted_networkx_cache",
    }
    _require(
        type(data) is dict and set(data) == expected, "invalid graph capture fields"
    )
    _require(
        data["graph_type"] == "networkx.classes.graph.Graph",
        "simple undirected graph required",
    )
    order = data["node_order"]
    _require(
        type(order) is tuple
        and order
        and all(_is_recorded_node_label(node) for node in order),
        "ordered integer or nonempty string node labels required",
    )
    _require(len(set(order)) == len(order), "duplicate node label")
    _require(type(data["graph"]) is dict, "graph attributes must be a mapping")
    graph = nx.Graph()
    graph.graph.update(data["graph"])
    rows = data["nodes"]
    _require(
        type(rows) is tuple and len(rows) == len(order),
        "node records must match node order",
    )
    for node, row in zip(order, rows, strict=True):
        _require(
            type(row) is tuple
            and len(row) == 2
            and _is_recorded_node_label(row[0])
            and row[0] == node
            and type(row[1]) is dict,
            "node association mismatch",
        )
        graph.add_node(node, **row[1])
    _require(type(data["edges"]) is tuple, "edge records must be ordered")
    for row in data["edges"]:
        _require(type(row) is tuple and len(row) == 3, "invalid edge record")
        left, right, attributes = row
        _require(
            _is_recorded_node_label(left)
            and _is_recorded_node_label(right)
            and left in graph
            and right in graph
            and left != right,
            "invalid edge endpoints",
        )
        _require(
            type(attributes) is dict and not graph.has_edge(left, right),
            "invalid or duplicate edge",
        )
        graph.add_edge(left, right, **attributes)
    neighbors = data["neighbor_order"]
    _require(
        type(neighbors) is tuple and len(neighbors) == len(order),
        "neighbor-order records must match nodes",
    )
    for node, row in zip(order, neighbors, strict=True):
        _require(
            type(row) is tuple
            and len(row) == 2
            and _is_recorded_node_label(row[0])
            and row[0] == node,
            "neighbor association mismatch",
        )
        sequence = row[1]
        _require(
            type(sequence) is tuple
            and all(_is_recorded_node_label(item) for item in sequence),
            "ordered neighbor labels required",
        )
        _require(
            len(set(sequence)) == len(sequence) and set(sequence) == set(graph[node]),
            "neighbor support mismatch",
        )
        graph._adj[node] = {
            neighbor: graph._adj[node][neighbor] for neighbor in sequence
        }
    _require(
        tuple(graph.edges) == tuple((row[0], row[1]) for row in data["edges"]),
        "recorded edge order disagrees with restored node/neighbor order",
    )
    instance = data["object_state"]
    _require(
        type(instance) is dict and set(instance) <= {"_last_operator_applied"},
        "unsupported instance state",
    )
    if instance:
        _require(
            type(instance["_last_operator_applied"]) is str,
            "last operator must be text",
        )
        graph._last_operator_applied = instance["_last_operator_applied"]
    omissions = tuple(
        data[key]
        for key in (
            "omitted_rebuildable_caches",
            "omitted_networkx_views",
            "omitted_networkx_cache",
        )
    )
    _admit_omissions(omissions, graph)
    return RecordedGraphReconstruction(graph, *omissions)


def record_graph_state(graph):
    """Capture persistent state with declared cache omissions, without events.

    BEPI uses shared signed admission and canonical mapping serialization.
    """
    _require(type(graph) is nx.Graph, "simple undirected Graph required")
    instance = vars(graph)
    extras = set(instance) - _STORAGE - _VIEWS
    _require(extras <= {"_last_operator_applied"}, "unsupported instance state")

    # Capture instance inventory before asking NetworkX for any cached views.
    def inventory(items, scope, *, owner=None):
        result = []
        for key, value in items:
            row = {
                "key": key,
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "scope": scope,
            }
            if owner is not None:
                row["owner"] = owner
            result.append(row)
        return tuple(result)

    views = inventory(
        ((key, value) for key, value in instance.items() if key in _VIEWS),
        "view of retained graph storage",
    )
    native = inventory(
        (
            (key, value)
            for key, value in instance.items()
            if key == "__networkx_cache__"
        ),
        "NetworkX rebuildable cache",
    )
    caches = inventory(
        (
            (key, value)
            for key, value in graph.graph.items()
            if key in GRAPH_RUNTIME_CACHE_KEYS
        ),
        "rebuildable payload only; configuration and event state retained",
        owner="tnfr.utils.cache.GRAPH_RUNTIME_CACHE_KEYS",
    )
    # Raw storage iteration avoids populating a cached NetworkX view on input.
    order = tuple(graph._node)
    seen, edges = set(), []
    for left in order:
        for right, attributes in graph._adj[left].items():
            if right not in seen:
                edges.append((left, right, attributes))
        seen.add(left)
    state = {
        "graph_type": "networkx.classes.graph.Graph",
        "node_order": order,
        "neighbor_order": tuple((node, tuple(graph._adj[node])) for node in order),
        "graph": {
            key: value
            for key, value in graph.graph.items()
            if key not in GRAPH_RUNTIME_CACHE_KEYS
        },
        "nodes": tuple(graph._node.items()),
        "edges": tuple(edges),
        "object_state": {key: instance[key] for key in extras},
        "omitted_rebuildable_caches": caches,
        "omitted_networkx_views": views,
        "omitted_networkx_cache": native,
    }
    recorded = encode_recorded_container(state)
    restore_recorded_graph_state(recorded)
    return recorded
