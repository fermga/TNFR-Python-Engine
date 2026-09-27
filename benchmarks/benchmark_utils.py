"""Benchmark Utilities for TNFR Research
==========================================

Shared supplied-graph preparation for the curvature safety illustration.
Operator admission and execution belong to the production engine.

Status: RESEARCH - Support infrastructure for validation experiments
"""

from __future__ import annotations

import math
import random

# Import TNFR core
import sys
from pathlib import Path

import networkx as nx

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.config import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
    inject_defaults,
)


def create_tnfr_topology(topology: str, n_nodes: int, seed: int) -> nx.Graph:
    """Create a network topology with proper TNFR initialization.

    Parameters
    ----------
    topology : str
        One of: 'ring', 'scale_free', 'ws', 'tree', 'grid'
    n_nodes : int
        Number of nodes
    seed : int
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Graph with TNFR defaults injected but nodes not yet initialized
    """
    if topology == "ring":
        G = nx.cycle_graph(n_nodes)
    elif topology == "scale_free":
        G = nx.scale_free_graph(n_nodes, seed=seed).to_undirected()
    elif topology == "ws":  # small-world
        k = min(4, n_nodes - 1) if n_nodes > 1 else 0
        G = nx.watts_strogatz_graph(n_nodes, k=k, p=0.3, seed=seed)
    elif topology == "tree":
        if n_nodes <= 1:
            G = nx.Graph()
            G.add_node(0)
        else:
            height = max(1, int(math.log2(n_nodes)))
            G = nx.balanced_tree(r=2, h=height)
            if G.number_of_nodes() > n_nodes:
                nodes_to_remove = list(G.nodes)[n_nodes:]
                G.remove_nodes_from(nodes_to_remove)
    elif topology == "grid":
        if n_nodes <= 1:
            G = nx.Graph()
            G.add_node(0)
        else:
            side = max(2, int(math.sqrt(n_nodes)))
            G = nx.grid_2d_graph(side, side)
            # Convert to integer node labels
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            if G.number_of_nodes() > n_nodes:
                nodes_to_remove = list(G.nodes)[n_nodes:]
                G.remove_nodes_from(nodes_to_remove)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Inject TNFR defaults into graph
    inject_defaults(G)
    G.graph["RANDOM_SEED"] = seed

    return G


def initialize_tnfr_nodes(
    G: nx.Graph,
    nu_f: float = 1.0,
    epi_range: tuple[float, float] = (0.2, 0.8),
    seed: int = 42,
) -> None:
    """Initialize node attributes with proper TNFR primary keys.

    Parameters
    ----------
    G : nx.Graph
        Graph to initialize (must have inject_defaults already called)
    nu_f : float
        Structural frequency (νf) for all nodes
    epi_range : tuple[float, float]
        (min, max) for the supplied signed scalar EPI initialization
    seed : int
        Random seed
    """
    rng = random.Random(seed)

    for node in G.nodes:
        G.nodes[node][EPI_PRIMARY] = rng.uniform(*epi_range)
        G.nodes[node][VF_PRIMARY] = nu_f
        G.nodes[node][THETA_PRIMARY] = rng.uniform(0.0, 2 * math.pi)
        G.nodes[node][DNFR_PRIMARY] = rng.uniform(0.01, 0.05)
        # NOTE: Do NOT add legacy aliases ('phase', 'delta_nfr')
        # The fields.py module will find the correct aliases automatically
        # Adding redundant aliases causes desync issues


__all__ = ["create_tnfr_topology", "initialize_tnfr_nodes"]
