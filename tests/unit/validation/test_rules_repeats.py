"""Tests for repetition avoidance in :mod:`tnfr.validation.soft_filters`."""

from __future__ import annotations

from collections import deque

from tnfr.config.operator_names import RESONANCE, SILENCE
from tnfr.types import Glyph
from tnfr.validation.soft_filters import check_repeats
from tnfr.validation import GrammarContext, glyph_function_name


def _ctx_with_node(graph_canon, cfg_soft):
    """Return a grammar context and node id seeded with ``cfg_soft``."""

    G = graph_canon()
    node_id = 0
    G.add_node(node_id)
    ctx = GrammarContext(G=G, cfg_soft=cfg_soft, cfg_canon={}, norms={})
    return ctx, node_id


def test_check_repeats_swaps_to_configured_fallback(graph_canon):
    cfg_soft = {
        "window": 3,
        "avoid_repeats": [Glyph.RA.value],
        "fallbacks": {Glyph.RA.value: Glyph.SHA.value},
    }
    ctx, node_id = _ctx_with_node(graph_canon, cfg_soft)
    nd = ctx.G.nodes[node_id]
    nd["glyph_history"] = deque([Glyph.IL.value, Glyph.RA.value], maxlen=3)

    swapped = check_repeats(ctx, node_id, Glyph.RA)

    assert glyph_function_name(swapped) == SILENCE


def test_check_repeats_leaves_non_recent_candidate(graph_canon):
    cfg_soft = {
        "window": 4,
        "avoid_repeats": [Glyph.RA.value],
        "fallbacks": {Glyph.RA.value: Glyph.SHA.value},
    }
    ctx, node_id = _ctx_with_node(graph_canon, cfg_soft)
    nd = ctx.G.nodes[node_id]
    nd["glyph_history"] = deque([Glyph.OZ.value, Glyph.IL.value], maxlen=4)

    cand = check_repeats(ctx, node_id, Glyph.RA)

    assert glyph_function_name(cand) == RESONANCE


def test_check_repeats_with_zero_window_passthrough(graph_canon):
    cfg_soft = {
        "window": 0,
        "avoid_repeats": [Glyph.RA.value],
        "fallbacks": {Glyph.RA.value: Glyph.SHA.value},
    }
    ctx, node_id = _ctx_with_node(graph_canon, cfg_soft)
    nd = ctx.G.nodes[node_id]
    nd["glyph_history"] = deque([Glyph.RA.value], maxlen=1)

    cand = check_repeats(ctx, node_id, Glyph.RA)

    assert glyph_function_name(cand) == RESONANCE
