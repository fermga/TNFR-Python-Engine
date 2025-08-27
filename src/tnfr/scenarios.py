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

    # Valores canónicos para inicialización
    inject_defaults(G, DEFAULTS)
    vf_min = float(G.graph.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(G.graph.get("VF_MAX", DEFAULTS["VF_MAX"]))
    th_min = float(G.graph.get("INIT_THETA_MIN", DEFAULTS.get("INIT_THETA_MIN", -3.1416)))
    th_max = float(G.graph.get("INIT_THETA_MAX", DEFAULTS.get("INIT_THETA_MAX", 3.1416)))

    for i in G.nodes():
        nd = G.nodes[i]
        nd.setdefault("EPI", rng.uniform(0.1, 0.3))
        nd.setdefault("νf", rng.uniform(vf_min, vf_max))
        nd.setdefault("θ", rng.uniform(th_min, th_max))
        nd.setdefault("Si", rng.uniform(0.4, 0.7))

    return G
