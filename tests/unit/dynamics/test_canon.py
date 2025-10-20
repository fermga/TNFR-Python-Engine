"""Pruebas de canon."""

import networkx as nx
from tnfr.dynamics import validate_canon
from tnfr.constants import VF_KEY, THETA_KEY, inject_defaults
from tnfr.initialization import init_node_attrs


def test_build_graph_vf_within_limits():
    G = nx.cycle_graph(10)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    vf_min = G.graph["VF_MIN"]
    vf_max = G.graph["VF_MAX"]
    for n in G.nodes():
        vf = G.nodes[n][VF_KEY]
        assert vf_min <= vf <= vf_max


def test_validate_canon_clamps():
    G = nx.cycle_graph(5)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    for n in G.nodes():
        nd = G.nodes[n]
        nd[VF_KEY] = 2.0
        nd["EPI"] = 2.0
        nd[THETA_KEY] = 5.0
    validate_canon(G)
    vf_min = G.graph["VF_MIN"]
    vf_max = G.graph["VF_MAX"]
    epi_min = G.graph["EPI_MIN"]
    epi_max = G.graph["EPI_MAX"]
    for n in G.nodes():
        nd = G.nodes[n]
        assert vf_min <= nd[VF_KEY] <= vf_max
        assert epi_min <= nd["EPI"] <= epi_max
        assert -3.1416 <= nd[THETA_KEY] <= 3.1416
