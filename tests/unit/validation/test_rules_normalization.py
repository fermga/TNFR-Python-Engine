"""Focused tests for :mod:`tnfr.validation.rules` normalisation helpers."""

from __future__ import annotations

from collections import deque

import pytest

from tnfr.constants import get_aliases
from tnfr.types import Glyph
from tnfr.validation import rules
from tnfr.validation.compatibility import CANON_FALLBACK
from tnfr.validation.grammar import GrammarContext


@pytest.fixture
def seeded_context(graph_canon):
    """Return a grammar context with a single seeded node."""

    G = graph_canon()
    node_id = 0
    G.add_node(node_id)
    nd = G.nodes[node_id]
    # Provide baseline telemetry to tweak in tests.
    accel_key = get_aliases("D2EPI")[0]
    dnfr_key = get_aliases("DNFR")[0]
    si_key = get_aliases("SI")[0]
    nd[accel_key] = 0.25
    nd[dnfr_key] = 0.4
    nd[si_key] = 2.0
    # Prime a glyph history so fallbacks act on canonical glyphs when needed.
    nd["glyph_history"] = deque([Glyph.AL.value])
    return G, node_id, accel_key, dnfr_key, si_key


def test_norm_helpers_handle_defaults_and_bounds(seeded_context):
    G, node_id, accel_key, dnfr_key, si_key = seeded_context
    nd = G.nodes[node_id]

    # With no explicit norms we fall back to 1.0 and keep absolute ratios.
    ctx_missing = GrammarContext.from_graph(G)
    assert rules.get_norm(ctx_missing, "accel_max") == 1.0
    direct = rules._norm_attr(ctx_missing, nd, get_aliases("D2EPI"), "accel_max")
    assert direct == pytest.approx(0.25)
    assert rules._accel_norm(ctx_missing, nd) == pytest.approx(0.25)

    # Saturated SI values clamp to bounds.
    assert rules._si(nd) == 1.0
    nd[si_key] = -0.5
    assert rules._si(nd) == 0.0

    # Tight maxima clamp the ratio to 1.0.
    G.graph["_sel_norms"] = {"accel_max": 0.1, "dnfr_max": 0.5}
    ctx_small = GrammarContext.from_graph(G)
    assert rules._accel_norm(ctx_small, nd) == 1.0
    nd[dnfr_key] = 1.0
    assert rules.normalized_dnfr(ctx_small, nd) == 1.0

    # Zero and negative maxima avoid division and default to safe values.
    G.graph["_sel_norms"].update({"accel_max": 0.0, "dnfr_max": -1.0})
    ctx_bad = GrammarContext.from_graph(G)
    assert rules.get_norm(ctx_bad, "accel_max") == 1.0
    assert rules.normalized_dnfr(ctx_bad, nd) == 0.0


def test_coerce_glyph_and_invalid_values():
    assert rules.coerce_glyph("RA") == Glyph.RA
    assert rules.coerce_glyph(Glyph.IL) == Glyph.IL
    unknown = object()
    assert rules.coerce_glyph("??") == "??"
    assert rules.coerce_glyph(unknown) is unknown


def test_glyph_fallback_prefers_custom_and_canon():
    fallbacks = {"OZ": "VAL", "RA": Glyph.IL, "??": "??"}

    assert rules.glyph_fallback("OZ", fallbacks) == Glyph.VAL
    assert rules.glyph_fallback("RA", fallbacks) == Glyph.IL
    assert rules.glyph_fallback("SHA", fallbacks) == CANON_FALLBACK[Glyph.SHA]
    assert rules.glyph_fallback("??", fallbacks) == "??"
