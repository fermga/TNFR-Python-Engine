"""Scenario generation."""

from __future__ import annotations
import networkx as nx  # type: ignore[import-untyped]

from .constants import inject_defaults
from .initialization import init_node_attrs

VALID_TOPOLOGIES = ("ring", "complete", "erdos")

__all__ = ["build_graph"]


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

    topology = topology.lower()
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(
            f"Invalid topology '{topology}'. "
            f"Accepted options are: {', '.join(VALID_TOPOLOGIES)}"
        )

    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    else:  # topology == "erdos"
        prob = float(p) if p is not None else 3.0 / n
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"p must be between 0 and 1; received {prob}")
        G = nx.gnp_random_graph(n, prob, seed=seed)

    # Valores canónicos para inicialización
    inject_defaults(G)
    if seed is not None:
        G.graph["RANDOM_SEED"] = int(seed)
    init_node_attrs(G, override=True)
    return G
