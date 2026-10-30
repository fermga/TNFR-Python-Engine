"""Exact central-contact geometry for reflection-even C9 components.

Private callers admit component counts and simple unit contacts before entry.
The layer mass includes every attached edge; no class coefficient, phase
origin, preparation or theorem verdict is supplied by this geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

import networkx as nx

from ..mathematics._exact_linear_algebra import ExactSquareMatrix


@dataclass(frozen=True)
class _PortGeometry:
    component_count: int
    contacts: tuple[tuple[int, int], ...]
    contact_degrees: tuple[int, ...]
    layer_masses: tuple[Q, ...]
    internal_laplacian: ExactSquareMatrix
    joined_laplacian: ExactSquareMatrix
    normalized_interior_matrix: ExactSquareMatrix
    normalized_form_matrix: ExactSquareMatrix
    connected: bool
    contact_diameter: int | None


def _central_port_geometry(component_count, contacts) -> _PortGeometry:
    graph = nx.Graph()
    graph.add_nodes_from(range(component_count))
    graph.add_edges_from(contacts)
    degrees = tuple(graph.degree[i] for i in range(component_count))
    masses = tuple(Q(value) for degree in degrees for value in (2 + degree, 4, 4, 4, 4))
    size = 5 * component_count
    internal = [[Q(0) for _ in range(size)] for _ in range(size)]

    def add_edge(matrix, left, right, weight):
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight

    for component in range(component_count):
        for layer in range(4):
            left = 5 * component + layer
            add_edge(internal, left, left + 1, Q(2))
    joined = [row.copy() for row in internal]
    for left, right in contacts:
        add_edge(joined, 5 * left, 5 * right, Q(1))
    connected = nx.is_connected(graph)
    return _PortGeometry(
        component_count=component_count,
        contacts=contacts,
        contact_degrees=degrees,
        layer_masses=masses,
        internal_laplacian=tuple(map(tuple, internal)),
        joined_laplacian=tuple(map(tuple, joined)),
        normalized_interior_matrix=tuple(
            tuple(value / masses[i] for value in row) for i, row in enumerate(internal)
        ),
        normalized_form_matrix=tuple(
            tuple(value / masses[i] for value in row) for i, row in enumerate(joined)
        ),
        connected=connected,
        contact_diameter=nx.diameter(graph) if connected else None,
    )


def _component_interior_matrix(geometry, component) -> ExactSquareMatrix:
    """Embed one already admitted component's normalized internal block."""
    return tuple(
        tuple(value if i // 5 == component else Q(0) for value in row)
        for i, row in enumerate(geometry.normalized_interior_matrix)
    )
