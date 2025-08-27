import networkx as nx
import pytest

from tnfr.constants import attach_defaults, merge_overrides
from tnfr.dynamics import update_epi_via_nodal_equation


def test_gamma_linear_integration():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    attach_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_linear", "beta": 1.0, "R0": 0.0})
    for n in G.nodes():
        G.nodes[n]["νf"] = 1.0
        G.nodes[n]["ΔNFR"] = 0.0
        G.nodes[n]["θ"] = 0.0
        G.nodes[n]["EPI"] = 0.0
    update_epi_via_nodal_equation(G, dt=1.0)
    assert pytest.approx(G.nodes[0]["EPI"], rel=1e-6) == 1.0
    assert pytest.approx(G.nodes[1]["EPI"], rel=1e-6) == 1.0
