"""Shared validation and label-preserving topology scaffolds for the SDK.

These helpers construct graph support, not a dynamical Coupling operation.
Operator execution retains its separate live U3 phase gate.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import networkx as nx


def nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def positive_integer(value: Any, name: str) -> int:
    value = nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def probability(value: Any) -> float:
    if not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("connection probability must be finite and between 0 and 1")
    return float(value)


def ring_edges(nodes: list[Any]) -> list[tuple[Any, Any]]:
    if len(nodes) < 2:
        return []
    if len(nodes) == 2:
        return [(nodes[0], nodes[1])]
    return [(node, nodes[(index + 1) % len(nodes)]) for index, node in enumerate(nodes)]


def small_world_edges(nodes: list[Any], k: int, p: float, seed: Any) -> list[tuple[Any, Any]]:
    k = nonnegative_integer(k, "k")
    p = probability(p)
    graph = nx.watts_strogatz_graph(len(nodes), k, p, seed=seed)
    return [(nodes[u], nodes[v]) for u, v in graph.edges()]


def contact_degree(size: int, value: Any) -> int:
    """Validate an exactly realizable integer mean degree of a simple graph."""
    size = positive_integer(size, "people")
    degree = nonnegative_integer(value, "connections_per_person")
    if degree >= size:
        raise ValueError("connections_per_person must be smaller than people")
    if size * degree % 2:
        raise ValueError("people * connections_per_person must be even for an exact mean degree")
    return degree


def rewired_contact_edges(
    nodes: list[Any], degree: int, p: float, seed: Any,
) -> list[tuple[Any, Any]]:
    """Rewire a ring scaffold while preserving its exact mean contact degree.

    Odd degree uses an opposite-node matching on an even-sized ring. Rewiring
    preserves edge count, not individual degrees or connectedness. This is an
    operational support graph; no phase-coupling operation is performed here.
    """
    degree = contact_degree(len(nodes), degree)
    p = probability(p)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    for offset in range(1, degree // 2 + 1):
        graph.add_edges_from((index, (index + offset) % len(nodes)) for index in graph)
    if degree % 2:
        midpoint = len(nodes) // 2
        graph.add_edges_from((index, index + midpoint) for index in range(midpoint))
    rng = nx.utils.create_py_random_state(seed)
    for source, target in list(graph.edges()):
        if rng.random() < p:
            candidates = [node for node in graph if node != source and not graph.has_edge(source, node)]
            if candidates:
                replacement = rng.choice(candidates)
                graph.remove_edge(source, target)
                graph.add_edge(source, replacement)
    return [(nodes[source], nodes[target]) for source, target in graph.edges()]


def hierarchical_edges(
    nodes: list[Any], depth: int,
) -> tuple[list[tuple[Any, Any]], dict[Any, int]]:
    """Build connected graph layers with parent links and within-layer rings.

    Depth one is a peer ring. Otherwise the first node is the root, and the
    remaining nodes are spread as evenly as possible over subsequent layers.
    Each child has one parent in the preceding layer. Layers are graph metadata,
    not nested EPIs or a claim of multiscale physical coherence.
    """
    depth = positive_integer(depth, "hierarchy_depth")
    if depth > len(nodes):
        raise ValueError("hierarchy_depth cannot exceed agents")
    if depth == 1:
        layers = [nodes]
    else:
        width, extra = divmod(len(nodes) - 1, depth - 1)
        layers = [nodes[:1]]
        cursor = 1
        for index in range(depth - 1):
            size = width + (index < extra)
            layers.append(nodes[cursor:cursor + size])
            cursor += size
    edges = []
    levels = {}
    for level, layer in enumerate(layers):
        levels.update((node, level) for node in layer)
        edges.extend(ring_edges(layer))
        if level:
            parents = layers[level - 1]
            edges.extend((parents[index % len(parents)], node) for index, node in enumerate(layer))
    return edges, levels


def grid_edges(nodes: list[Any], rows: int | None, cols: int | None) -> list[tuple[Any, Any]]:
    for name, value in (("rows", rows), ("cols", cols)):
        if value is not None and nonnegative_integer(value, name) == 0:
            raise ValueError(f"{name} must be positive")
    n = len(nodes)
    if not n:
        return []
    if rows is None and cols is None:
        rows = max(1, math.isqrt(n))
    if rows is None:
        rows = (n + cols - 1) // cols
    if cols is None:
        cols = (n + rows - 1) // rows
    if rows * cols < n:
        raise ValueError("grid capacity must accommodate every existing node")
    edges = []
    for index, node in enumerate(nodes):
        if index % cols + 1 < cols and index + 1 < n:
            edges.append((node, nodes[index + 1]))
        if index + cols < n:
            edges.append((node, nodes[index + cols]))
    return edges
