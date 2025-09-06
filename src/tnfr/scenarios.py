"""Scenario generation."""

from __future__ import annotations
import networkx as nx

from .constants import inject_defaults
from .initialization import init_node_attrs


def build_graph(
    n: int = 24,
    topology: str = "ring",
    seed: int | None = 1,
    p: float | None = None,
):
    """Build a graph with canonical initialization.

    For ``topology="erdos"`` the probability of edge creation can be adjusted
    via ``p``. If omitted, a default of ``3.0 / n`` is used.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "erdos":
        prob = p if p is not None else 3.0 / n
        if not 0.0 <= prob <= 1.0:
            raise ValueError("p must be between 0 and 1")
        G = nx.gnp_random_graph(n, prob, seed=seed)
    else:
        valid = ["ring", "complete", "erdos"]
        raise ValueError(
            f"Invalid topology '{topology}'. "
            f"Valid options are: {', '.join(valid)}"
        )

    # Valores canónicos para inicialización
    inject_defaults(G)
    if seed is not None:
        G.graph["RANDOM_SEED"] = int(seed)
    init_node_attrs(G, override=True)
    return G
