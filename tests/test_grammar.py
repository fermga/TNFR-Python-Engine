import networkx as nx
from collections import deque

from tnfr.constants import attach_defaults
from tnfr.grammar import (
    enforce_canonical_grammar,
    on_applied_glifo,
    apply_glyph_with_grammar,
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


def test_lag_counters_enforced():
    G = make_graph()
    G.graph['AL_MAX_LAG'] = 2
    G.graph['EN_MAX_LAG'] = 2
    from tnfr.dynamics import step

    for _ in range(6):
        step(G)
        hist = G.graph['history']
        assert all(v <= 2 for v in hist['since_AL'].values())
        assert all(v <= 2 for v in hist['since_EN'].values())


def test_apply_glyph_with_grammar_equivalence():
    G_manual = make_graph()
    G_func = make_graph()

    # Aplicación manual
    g_eff = enforce_canonical_grammar(G_manual, 0, ZHIR)
    from tnfr.operators import aplicar_glifo
    aplicar_glifo(G_manual, 0, g_eff, window=1)
    on_applied_glifo(G_manual, 0, g_eff)

    # Aplicación mediante helper
    apply_glyph_with_grammar(G_func, [0], ZHIR, 1)

    assert G_manual.nodes[0] == G_func.nodes[0]


def test_apply_glyph_with_grammar_multiple_nodes():
    G = nx.Graph()
    G.add_node(0, theta=0.0)
    G.add_node(1)
    attach_defaults(G)
    from collections import deque
    G.nodes[0]['hist_glifos'] = deque([OZ])

    apply_glyph_with_grammar(G, [0, 1], ZHIR, 1)

    assert G.nodes[0]['hist_glifos'][-1] == ZHIR
    assert G.nodes[1]['hist_glifos'][-1] == OZ
