import networkx as nx
from collections import deque

from tnfr.constants import attach_defaults
from tnfr.grammar import (
    enforce_canonical_grammar,
    on_applied_glifo,
    AL, EN, IL, OZ, ZHIR, THOL, SHA, NUL, NAV,
)


def make_graph():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0)
    attach_defaults(G)
    return G


def test_compatibility_fallback():
    G = make_graph()
    from collections import deque

    nd = G.nodes[0]
    nd['hist_glifos'] = deque([AL])
    assert enforce_canonical_grammar(G, 0, IL) == EN


def test_precondition_oz_to_zhir():
    G = make_graph()
    from collections import deque

    nd = G.nodes[0]
    nd['hist_glifos'] = deque([NAV])
    nd['ΔNFR'] = 0.0
    assert enforce_canonical_grammar(G, 0, ZHIR) == OZ
    nd['hist_glifos'] = deque([OZ])
    assert enforce_canonical_grammar(G, 0, ZHIR) == ZHIR


def test_thol_closure():
    G = make_graph()
    from collections import deque

    nd = G.nodes[0]
    nd['hist_glifos'] = deque([THOL])
    on_applied_glifo(G, 0, THOL)
    st = nd['_GRAM']
    st['thol_len'] = 2
    nd['ΔNFR'] = 0.0
    nd['Si'] = 0.7
    assert enforce_canonical_grammar(G, 0, EN) == NUL
    nd['Si'] = 0.1
    st['thol_len'] = 2
    assert enforce_canonical_grammar(G, 0, EN) == SHA
