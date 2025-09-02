"""Pruebas de grammar."""
from collections import deque

from tnfr.constants import attach_defaults
from tnfr.grammar import (
    enforce_canonical_grammar,
    on_applied_glifo,
    apply_glyph_with_grammar,
    AL, EN, IL, OZ, ZHIR, THOL, SHA, NUL, NAV,
)


def test_compatibility_fallback(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    nd = G.nodes[0]
    nd['hist_glifos'] = deque([AL])
    assert enforce_canonical_grammar(G, 0, IL) == EN


def test_precondition_oz_to_zhir(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    nd = G.nodes[0]
    nd['hist_glifos'] = deque([NAV])
    nd['ΔNFR'] = 0.0
    assert enforce_canonical_grammar(G, 0, ZHIR) == OZ
    nd['hist_glifos'] = deque([OZ])
    assert enforce_canonical_grammar(G, 0, ZHIR) == ZHIR


def test_thol_closure(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
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


def test_repeat_window_and_force(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    nd = G.nodes[0]
    nd['hist_glifos'] = deque([ZHIR.value, 'OZ'])
    G.graph['GRAMMAR'] = {
        'window': 3,
        'avoid_repeats': ['ZHIR'],
        'force_dnfr': 0.5,
        'force_accel': 0.8,
        'fallbacks': {'ZHIR': 'NAV'},
    }
    G.graph['_sel_norms'] = {'dnfr_max': 1.0, 'accel_max': 1.0}

    nd['ΔNFR'] = 0.0
    nd['d2EPI_dt2'] = 0.0
    assert enforce_canonical_grammar(G, 0, ZHIR) == NAV

    nd['ΔNFR'] = 0.6
    assert enforce_canonical_grammar(G, 0, ZHIR) == ZHIR

    nd['ΔNFR'] = 0.0
    nd['d2EPI_dt2'] = 0.9
    assert enforce_canonical_grammar(G, 0, ZHIR) == ZHIR




def test_repeat_invalid_fallback_string(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    nd = G.nodes[0]
    nd['hist_glifos'] = deque([ZHIR.value])
    G.graph['GRAMMAR'] = {
        'window': 3,
        'avoid_repeats': ['ZHIR'],
        'fallbacks': {'ZHIR': 'NOPE'},
    }
    assert enforce_canonical_grammar(G, 0, ZHIR) == IL


def test_repeat_invalid_fallback_type(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    nd = G.nodes[0]
    nd['hist_glifos'] = deque([ZHIR.value])
    obj = object()
    G.graph['GRAMMAR'] = {
        'window': 3,
        'avoid_repeats': ['ZHIR'],
        'fallbacks': {'ZHIR': obj},
    }
    assert enforce_canonical_grammar(G, 0, ZHIR) == IL

def test_lag_counters_enforced(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph['AL_MAX_LAG'] = 2
    G.graph['EN_MAX_LAG'] = 2
    from tnfr.dynamics import step

    for _ in range(6):
        step(G)
        hist = G.graph['history']
        assert all(v <= 2 for v in hist['since_AL'].values())
        assert all(v <= 2 for v in hist['since_EN'].values())


def test_apply_glyph_with_grammar_equivalence(graph_canon):
    G_manual = graph_canon()
    G_manual.add_node(0)
    attach_defaults(G_manual)
    G_func = graph_canon()
    G_func.add_node(0)
    attach_defaults(G_func)

    # Aplicación manual
    g_eff = enforce_canonical_grammar(G_manual, 0, ZHIR)
    from tnfr.operators import aplicar_glifo
    aplicar_glifo(G_manual, 0, g_eff, window=1)
    on_applied_glifo(G_manual, 0, g_eff)

    # Aplicación mediante helper
    apply_glyph_with_grammar(G_func, [0], ZHIR, 1)

    assert G_manual.nodes[0] == G_func.nodes[0]


def test_apply_glyph_with_grammar_multiple_nodes(graph_canon):
    G = graph_canon()
    G.add_node(0, theta=0.0)
    G.add_node(1)
    attach_defaults(G)
    G.nodes[0]['hist_glifos'] = deque([OZ])

    apply_glyph_with_grammar(G, [0, 1], ZHIR, 1)

    assert G.nodes[0]['hist_glifos'][-1] == ZHIR
    assert G.nodes[1]['hist_glifos'][-1] == OZ


def test_apply_glyph_with_grammar_accepts_iterables(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    attach_defaults(G)
    G.nodes[0]['hist_glifos'] = deque([OZ])
    G.nodes[1]['hist_glifos'] = deque([OZ])
    apply_glyph_with_grammar(G, G.nodes(), ZHIR, 1)
    assert G.nodes[0]['hist_glifos'][-1] == ZHIR
    assert G.nodes[1]['hist_glifos'][-1] == ZHIR

    G2 = graph_canon()
    G2.add_nodes_from([0, 1])
    attach_defaults(G2)
    G2.nodes[0]['hist_glifos'] = deque([OZ])
    G2.nodes[1]['hist_glifos'] = deque([OZ])
    apply_glyph_with_grammar(G2, (n for n in G2.nodes()), ZHIR, 1)
    assert G2.nodes[0]['hist_glifos'][-1] == ZHIR
    assert G2.nodes[1]['hist_glifos'][-1] == ZHIR
