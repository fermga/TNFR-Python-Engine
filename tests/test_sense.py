import math
import networkx as nx
import pytest

from tnfr.sense import sigma_vector_node, sigma_vector_global
from tnfr.types import Glyph


def _make_graph():
    G = nx.Graph()
    G.add_node(0, hist_glifos=[Glyph.AL.value], Si=1.0, EPI=2.0)
    G.add_node(1, Si=0.3, EPI=1.5)
    return G


def test_sigma_vector_node_paths():
    G = _make_graph()
    sv_si = sigma_vector_node(G, 0)
    assert sv_si and sv_si["glifo"] == Glyph.AL.value
    assert sv_si["w"] == 1.0
    assert sigma_vector_node(G, 1) is None
    sv_epi = sigma_vector_node(G, 0, weight_mode="EPI")
    assert sv_epi["w"] == 2.0
    assert sv_epi["mag"] == pytest.approx(2 * sv_si["mag"])


def test_sigma_vector_global_paths():
    G = _make_graph()
    sv_si = sigma_vector_global(G)
    sv_epi = sigma_vector_global(G, weight_mode="EPI")
    assert sv_si["n"] == 1
    assert sv_epi["n"] == 1
    assert sv_epi["mag"] == pytest.approx(2 * sv_si["mag"])
