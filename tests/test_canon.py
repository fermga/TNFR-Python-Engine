from tnfr.scenarios import build_graph
from tnfr.dynamics import validate_canon


def test_build_graph_vf_within_limits():
    G = build_graph(n=10, topology="ring", seed=42)
    vf_min = G.graph["VF_MIN"]
    vf_max = G.graph["VF_MAX"]
    for n in G.nodes():
        vf = G.nodes[n]["νf"]
        assert vf_min <= vf <= vf_max


def test_validate_canon_clamps():
    G = build_graph(n=5, topology="ring", seed=1)
    for n in G.nodes():
        nd = G.nodes[n]
        nd["νf"] = 2.0
        nd["EPI"] = 2.0
        nd["θ"] = 5.0
    validate_canon(G)
    vf_min = G.graph["VF_MIN"]
    vf_max = G.graph["VF_MAX"]
    epi_min = G.graph["EPI_MIN"]
    epi_max = G.graph["EPI_MAX"]
    for n in G.nodes():
        nd = G.nodes[n]
        assert vf_min <= nd["νf"] <= vf_max
        assert epi_min <= nd["EPI"] <= epi_max
        assert -3.1416 <= nd["θ"] <= 3.1416
