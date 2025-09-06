"""Pruebas para ``inject_defaults`` con tuplas mutables."""

import networkx as nx

from tnfr.constants import inject_defaults, DEFAULTS
import tnfr.constants as const


def test_mutating_graph_tuple_does_not_affect_defaults(monkeypatch):
    tup = ([1], {"a": 1})
    monkeypatch.setitem(const._DEFAULTS_COMBINED, "_test_tuple", tup)
    G = nx.Graph()
    inject_defaults(G)
    assert G.graph["_test_tuple"] is not const._DEFAULTS_COMBINED["_test_tuple"]
    G.graph["_test_tuple"][0].append(2)
    G.graph["_test_tuple"][1]["a"] = 2
    assert const._DEFAULTS_COMBINED["_test_tuple"] == tup
    assert DEFAULTS["_test_tuple"] == tup
