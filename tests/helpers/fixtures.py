"""Shared test fixtures for operator generation and validation.

This module provides reusable fixtures to avoid duplication across
test modules for operator factories, validators, and sequence generation.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)

@pytest.fixture
def seed_graph_factory() -> Callable[..., nx.Graph]:
    """Return a factory for creating deterministic test graphs.

    Returns a function that creates graphs with seeded random initialization
    of TNFR structural attributes (θ, EPI, νf, ΔNFR).
    """

    def _create(
        *,
        num_nodes: int,
        edge_probability: float,
        seed: int,
    ) -> nx.Graph:
        """Create a graph with deterministic TNFR attributes."""
        graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
        inject_defaults(graph)
        graph.graph.setdefault("RANDOM_SEED", seed)

        twopi = 2.0 * math.pi
        for node, data in graph.nodes(data=True):
            base = seed + int(node)
            theta = ((base * 0.017) % twopi) - math.pi
            epi = math.sin(base * 0.031) * 0.45
            vf = 0.35 + 0.05 * ((base % 11) / 10.0)
            data[THETA_KEY] = theta
            data[EPI_PRIMARY] = epi
            data[VF_PRIMARY] = vf
            data[DNFR_PRIMARY] = 0.0

        return graph

    return _create

@pytest.fixture
def homogeneous_graph_factory() -> Callable[..., nx.Graph]:
    """Return a factory for creating homogeneous graphs.

    All nodes in the created graph share the same EPI and νf values,
    which should result in zero ΔNFR under gradient-based dynamics.
    """

    def _create(
        *,
        num_nodes: int,
        edge_probability: float,
        seed: int,
        epi_value: float = 0.0,
        vf_value: float = 1.0,
    ) -> nx.Graph:
        """Create a homogeneous graph with uniform EPI and νf."""
        graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
        inject_defaults(graph)

        for _, data in graph.nodes(data=True):
            data[EPI_PRIMARY] = epi_value
            data[VF_PRIMARY] = vf_value
            data[DNFR_PRIMARY] = 0.0

        return graph

    return _create

@pytest.fixture
def bicluster_graph_factory() -> Callable[..., tuple[nx.Graph, tuple[list, list]]]:
    """Return a factory for creating complete bipartite graphs with contrasting clusters.

    Returns a function that creates bipartite graphs where each cluster
    has distinct EPI and νf values, creating structural gradients.
    """

    def _create(
        *,
        cluster_size: int,
        epi_left: float,
        epi_right: float,
        vf_left: float,
        vf_right: float,
    ) -> tuple[nx.Graph, tuple[list, list]]:
        """Create a bipartite graph with contrasting clusters."""
        graph = nx.complete_bipartite_graph(cluster_size, cluster_size)
        inject_defaults(graph)

        left_nodes = list(range(cluster_size))
        right_nodes = list(range(cluster_size, 2 * cluster_size))

        for node in left_nodes:
            data = graph.nodes[node]
            data[EPI_PRIMARY] = epi_left
            data[VF_PRIMARY] = vf_left
            data[DNFR_PRIMARY] = 0.0

        for node in right_nodes:
            data = graph.nodes[node]
            data[EPI_PRIMARY] = epi_right
            data[VF_PRIMARY] = vf_right
            data[DNFR_PRIMARY] = 0.0

        return graph, (left_nodes, right_nodes)

    return _create

@pytest.fixture
def operator_sequence_factory() -> Callable[..., list[Any]]:
    """Return a factory for creating structural operator sequences.

    Creates valid sequences of TNFR operators for testing
    run_sequence execution paths.
    """

    def _create(operator_names: list[str]) -> list[Any]:
        """Create operator instances from canonical names."""
        from tnfr.structural import (
            Coherence,
            Emission,
            Reception,
            Resonance,
            Silence,
            Dissonance,
            Mutation,
            Transition,
        )

        operator_map = {
            "coherence": Coherence,
            "emission": Emission,
            "reception": Reception,
            "resonance": Resonance,
            "silence": Silence,
            "dissonance": Dissonance,
            "mutation": Mutation,
            "transition": Transition,
        }

        operators = []
        for name in operator_names:
            op_class = operator_map.get(name.lower())
            if op_class is None:
                raise ValueError(f"Unknown operator: {name}")
            operators.append(op_class())

        return operators

    return _create
