"""Fresh graph fixtures for auxiliary field and balance diagnostics."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from tnfr.constants import inject_defaults


def make_diagnostic_field_graph(
    n: int = 30,
    topology: str = "watts_strogatz",
    seed: int = 42,
) -> nx.Graph:
    """Reproduce the historical seeded field fixture without sharing state.

    Stored phase, frequency and pressure aliases and random-draw order are
    intentional compatibility inputs. String EPI values are diagnostic labels;
    tests that execute scalar nodal operators supply numeric EPI separately.
    This factory neither supplies an evolution law nor validates a trajectory.
    """
    rng = np.random.default_rng(seed)

    if topology == "watts_strogatz":
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    elif topology == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif topology == "grid":
        side = int(math.sqrt(n))
        G = nx.grid_2d_graph(side, side)
    else:
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)

    inject_defaults(G)

    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[node]["EPI"] = f"epi_{node}"

    return G
