"""Pruebas de count glyphs."""
import networkx as nx
from collections import deque, Counter

from tnfr.glyph_history import count_glyphs
from tnfr.constants import ALIAS_EPI_KIND

def test_count_glyphs_last_only_and_window():
    G = nx.Graph()
    G.add_node(0, hist_glifos=deque(["A", "B"]))
    G.add_node(1, **{ALIAS_EPI_KIND[0]: "C"})

    last = count_glyphs(G, last_only=True)
    assert last == Counter({"B": 1, "C": 1})

    recent = count_glyphs(G, window=2)
    assert recent == Counter({"A": 1, "B": 1})
