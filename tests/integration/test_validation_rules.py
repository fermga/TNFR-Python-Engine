"""Tests for :mod:`tnfr.validation.rules`."""

from collections import deque

import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.types import Glyph
from tnfr.config.operator_names import DISSONANCE, MUTATION
from tnfr.validation import GrammarContext, glyph_function_name, rules as rules_mod
from tnfr.validation.soft_filters import maybe_force
from tnfr.validation.compatibility import CANON_COMPAT, CANON_FALLBACK


def _graph_with_node():
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0)
    return G


def test_maybe_force_recovers_original_when_dnfr_high():
    G = _graph_with_node()
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.ZHIR.value])
    G.graph["GRAMMAR"] = {"force_dnfr": 0.5}
    G.graph["_sel_norms"] = {"dnfr_max": 1.0}
    nd["ΔNFR"] = 0.6
    ctx = GrammarContext.from_graph(G)

    forced = maybe_force(
        ctx,
        0,
        Glyph.NAV,
        Glyph.ZHIR,
        rules_mod.normalized_dnfr,
        "force_dnfr",
    )

    assert glyph_function_name(forced) == MUTATION


def test_check_compatibility_returns_fallback_glyph():
    G = _graph_with_node()
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.AL])
    ctx = GrammarContext.from_graph(G)

    result = rules_mod._check_compatibility(ctx, 0, Glyph.ZHIR)

    successors = {glyph_function_name(g) for g in CANON_COMPAT[Glyph.AL]}
    assert MUTATION not in successors
    assert glyph_function_name(result) == glyph_function_name(CANON_FALLBACK[Glyph.AL])


def test_check_oz_to_zhir_requires_recent_oz():
    G = _graph_with_node()
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.NAV])
    nd["ΔNFR"] = 0.0
    G.graph["_sel_norms"] = {"dnfr_max": 1.0}
    ctx = GrammarContext.from_graph(G)

    result = rules_mod._check_oz_to_zhir(ctx, 0, Glyph.ZHIR)

    assert glyph_function_name(result) == DISSONANCE
