import networkx as nx

from tnfr.constants import attach_defaults
from tnfr.dynamics import step
from tnfr.gamma import GAMMA_REGISTRY


def test_history_delta_si_and_B():
    G = nx.Graph()
    G.add_node(0, EPI=0.0, νf=0.5, θ=0.0)
    attach_defaults(G)
    step(G, apply_glyphs=False)
    step(G, apply_glyphs=False)
    hist = G.graph.get("history", {})
    assert "delta_Si" in hist and len(hist["delta_Si"]) >= 2
    assert "B" in hist and len(hist["B"]) >= 2


def test_gamma_kuramoto_tanh_registry():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    attach_defaults(G)
    G.nodes[0]["θ"] = 0.0
    G.nodes[1]["θ"] = 0.0
    cfg = {"type": "kuramoto_tanh", "beta": 0.5, "k": 2.0, "R0": 0.0}
    val = GAMMA_REGISTRY["kuramoto_tanh"](G, 0, 0.0, cfg)
    assert abs(val) <= cfg["beta"]
