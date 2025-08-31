import pytest

from tnfr.constants import attach_defaults
from tnfr.dynamics import step


def test_vf_converge_to_neighbor_average_when_stable(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    attach_defaults(G)
    # configuraciones para estabilidad
    G.graph["DNFR_WEIGHTS"] = {"phase": 1.0, "epi": 0.0, "vf": 0.0, "topo": 0.0}
    G.graph["VF_ADAPT_TAU"] = 2
    G.graph["VF_ADAPT_MU"] = 0.5
    for n in G.nodes():
        nd = G.nodes[n]
        nd["θ"] = 0.0
        nd["EPI"] = 0.0
    G.nodes[0]["νf"] = 0.2
    G.nodes[1]["νf"] = 1.0

    for _ in range(3):
        step(G, use_Si=True, apply_glyphs=False)

    assert G.nodes[0]["νf"] == pytest.approx(0.6)
    assert G.nodes[1]["νf"] == pytest.approx(0.6)
