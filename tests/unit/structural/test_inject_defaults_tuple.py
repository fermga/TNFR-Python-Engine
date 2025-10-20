"""Pruebas para ``inject_defaults`` con tuplas mutables."""


from tnfr.constants import inject_defaults, DEFAULTS


def test_mutating_graph_tuple_does_not_affect_defaults(monkeypatch, graph_canon):
    tup = ([1], {"a": 1})
    new_defaults = DEFAULTS | {"_test_tuple": tup}
    G = graph_canon()
    inject_defaults(G, defaults=new_defaults)
    assert G.graph["_test_tuple"] is not new_defaults["_test_tuple"]
    G.graph["_test_tuple"][0].append(2)
    G.graph["_test_tuple"][1]["a"] = 2
    assert new_defaults["_test_tuple"] == tup
    assert "_test_tuple" not in DEFAULTS
