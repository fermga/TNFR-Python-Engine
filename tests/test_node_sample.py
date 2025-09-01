"""Pruebas de node sample."""
from tnfr.dynamics import step
from tnfr.constants import attach_defaults
import networkx as nx


def _build_graph(n):
    G = nx.Graph()
    attach_defaults(G)
    for i in range(n):
        G.add_node(i, **{"θ": 0.0, "EPI": 0.0})
    return G


def test_node_sample_large_graph():
    G = _build_graph(80)
    G.graph["UM_CANDIDATE_COUNT"] = 10
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert isinstance(sample, list)
    assert len(sample) == 10
    assert set(sample).issubset(set(G.nodes()))


def test_node_sample_small_graph():
    G = _build_graph(20)
    G.graph["UM_CANDIDATE_COUNT"] = 5
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert len(sample) == len(G.nodes())
