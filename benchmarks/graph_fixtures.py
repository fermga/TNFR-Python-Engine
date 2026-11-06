"""Supplied graph preparations shared by auxiliary research instruments.

The simplex gasket is a prescribed combinatorial construction, not a THOL
invocation or a mechanism selecting a physical dimension. Its boundary corners
are part of the preparation and must survive each recursive gluing step.
"""

from __future__ import annotations

from operator import index

import networkx as nx


def sierpinski_simplex(m: int, levels: int) -> tuple[nx.Graph, list]:
    """Return a corner-glued ``K_m`` gasket and its ordered outer corners.

    ``m >= 2`` and ``levels >= 0`` are supplied integer construction parameters.
    Labels and insertion order follow the original recursive benchmark fixture;
    no sorting, random choice or engine evolution is performed.
    """
    if isinstance(m, bool) or isinstance(levels, bool):
        raise TypeError("m and levels must be integers, not Boolean flags")
    m, levels = index(m), index(levels)
    if m < 2 or levels < 0:
        raise ValueError("m must be at least 2 and levels must be nonnegative")
    return _simplex_gasket(m, levels)


def _simplex_gasket(m: int, levels: int) -> tuple[nx.Graph, list]:
    if levels == 0:
        return nx.complete_graph(m), list(range(m))
    subgraph, subcorners = _simplex_gasket(m, levels - 1)
    graph = nx.Graph()
    copies = []
    for i in range(m):
        mapping = {node: (i, node) for node in subgraph.nodes}
        graph.add_nodes_from(mapping[node] for node in subgraph.nodes)
        graph.add_edges_from((mapping[u], mapping[v]) for u, v in subgraph.edges)
        copies.append([mapping[corner] for corner in subcorners])
    parent = {node: node for node in graph.nodes}

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for i in range(m):
        for j in range(i + 1, m):
            root_a, root_b = find(copies[i][j]), find(copies[j][i])
            if root_a != root_b:
                parent[root_b] = root_a

    merged = nx.Graph()
    for u, v in graph.edges:
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            merged.add_edge(root_u, root_v)
    return merged, [find(copies[i][i]) for i in range(m)]
