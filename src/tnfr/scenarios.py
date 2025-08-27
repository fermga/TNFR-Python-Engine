from __future__ import annotations
from typing import Any
import random
import networkx as nx

from .constants import inject_defaults, DEFAULTS


def build_graph(n: int = 24, topology: str = "ring", seed: int | None = 1):
    rng = random.Random(seed)
    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "erdos":
        G = nx.gnp_random_graph(n, 3.0 / n, seed=seed)
    else:
        G = nx.path_graph(n)

    for i in G.nodes():
        nd = G.nodes[i]
        nd.setdefault("EPI", rng.uniform(0.1, 0.3))
        nd.setdefault("νf", rng.uniform(0.8, 1.2))
        nd.setdefault("θ", rng.uniform(-3.1416, 3.1416))
        nd.setdefault("Si", rng.uniform(0.4, 0.7))

    inject_defaults(G, DEFAULTS)
    return G
