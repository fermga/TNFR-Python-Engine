from __future__ import annotations
import networkx as nx

from .constants import inject_defaults, DEFAULTS
from .initialization import init_node_attrs


def build_graph(n: int = 24, topology: str = "ring", seed: int | None = 1):
    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "erdos":
        G = nx.gnp_random_graph(n, 3.0 / n, seed=seed)
    else:
        valid = ["ring", "complete", "erdos"]
        raise ValueError(f"Invalid topology '{topology}'. Valid options are: {', '.join(valid)}")

    # Valores canónicos para inicialización
    inject_defaults(G, DEFAULTS)
    if seed is not None:
        G.graph.setdefault("RANDOM_SEED", int(seed))
    init_node_attrs(G, override=True)
    return G
