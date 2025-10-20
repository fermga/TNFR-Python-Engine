"""Tests for :mod:`tnfr.validation.rules`."""

from collections import deque

import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.types import Glyph
from tnfr.validation import rules as rules_mod
from tnfr.validation.compatibility import CANON_COMPAT, CANON_FALLBACK
from tnfr.validation.grammar import GrammarContext


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

    forced = rules_mod._maybe_force(
        ctx,
        0,
        Glyph.NAV,
        Glyph.ZHIR,
        rules_mod.normalized_dnfr,
        "force_dnfr",
    )

    assert forced == Glyph.ZHIR


def test_check_compatibility_returns_fallback_glyph():
    G = _graph_with_node()
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.AL])
    ctx = GrammarContext.from_graph(G)

    result = rules_mod._check_compatibility(ctx, 0, Glyph.ZHIR)

    assert Glyph.ZHIR not in CANON_COMPAT[Glyph.AL]
    assert result == CANON_FALLBACK[Glyph.AL]


def test_check_oz_to_zhir_requires_recent_oz():
    G = _graph_with_node()
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.NAV])
    nd["ΔNFR"] = 0.0
    G.graph["_sel_norms"] = {"dnfr_max": 1.0}
    ctx = GrammarContext.from_graph(G)

    result = rules_mod._check_oz_to_zhir(ctx, 0, Glyph.ZHIR)

    assert result == Glyph.OZ
