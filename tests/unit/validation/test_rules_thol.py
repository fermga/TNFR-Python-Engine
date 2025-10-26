"""Tests for THOL closure behaviour in :mod:`tnfr.validation.rules`."""

from __future__ import annotations

from tnfr.types import Glyph
from tnfr.validation.grammar import GrammarContext
from tnfr.validation import rules


def _ctx_with_node(
    graph_canon,
    *,
    thol_min_len: int,
    thol_max_len: int,
    thol_close_dnfr: float,
    si_high: float,
    dnfr_max: float,
):
    """Create a grammar context with a single node and custom canon settings."""

    G = graph_canon()
    node_id = 0
    G.add_node(node_id)
    ctx = GrammarContext(
        G=G,
        cfg_soft={},
        cfg_canon={
            "thol_min_len": thol_min_len,
            "thol_max_len": thol_max_len,
            "thol_close_dnfr": thol_close_dnfr,
            "si_high": si_high,
        },
        norms={"dnfr_max": dnfr_max},
    )
    return ctx, node_id


def test_thol_closure_waits_until_min_length(graph_canon):
    ctx, node_id = _ctx_with_node(
        graph_canon,
        thol_min_len=3,
        thol_max_len=6,
        thol_close_dnfr=0.2,
        si_high=0.75,
        dnfr_max=1.0,
    )
    nd = ctx.G.nodes[node_id]
    nd["ΔNFR"] = 0.9  # High ΔNFR keeps the block from closing early.
    nd["Si"] = 0.4
    state = {"thol_open": True, "thol_len": 0}

    cand = rules._check_thol_closure(ctx, node_id, Glyph.RA, state)

    assert cand == Glyph.RA
    assert state["thol_len"] == 1


def test_thol_closure_triggers_at_max_length_with_high_si(graph_canon):
    ctx, node_id = _ctx_with_node(
        graph_canon,
        thol_min_len=2,
        thol_max_len=5,
        thol_close_dnfr=0.1,
        si_high=0.7,
        dnfr_max=1.0,
    )
    nd = ctx.G.nodes[node_id]
    nd["ΔNFR"] = 0.8
    nd["Si"] = 0.85  # Exceeds si_high so the closure should emit Glyph.NUL.
    state = {"thol_open": True, "thol_len": 4}

    glyph = rules._check_thol_closure(ctx, node_id, Glyph.RA, state)

    assert glyph == Glyph.NUL
    assert state["thol_len"] == 5


def test_thol_closure_low_dnfr_uses_sha_when_si_below_threshold(graph_canon):
    ctx, node_id = _ctx_with_node(
        graph_canon,
        thol_min_len=3,
        thol_max_len=6,
        thol_close_dnfr=0.25,
        si_high=0.8,
        dnfr_max=1.0,
    )
    nd = ctx.G.nodes[node_id]
    nd["ΔNFR"] = 0.1  # Normalised value 0.1 <= close threshold.
    nd["Si"] = 0.3  # Below si_high so closure should choose Glyph.SHA.
    state = {"thol_open": True, "thol_len": 2}

    glyph = rules._check_thol_closure(ctx, node_id, Glyph.RA, state)

    assert glyph == Glyph.SHA
    assert state["thol_len"] == 3


def test_thol_closure_low_dnfr_with_high_si_prefers_nul(graph_canon):
    ctx, node_id = _ctx_with_node(
        graph_canon,
        thol_min_len=3,
        thol_max_len=6,
        thol_close_dnfr=0.2,
        si_high=0.65,
        dnfr_max=1.0,
    )
    nd = ctx.G.nodes[node_id]
    nd["ΔNFR"] = 0.05  # Normalised value 0.05 <= close threshold.
    nd["Si"] = 0.72  # Above si_high so closure should emit Glyph.NUL.
    state = {"thol_open": True, "thol_len": 2}

    glyph = rules._check_thol_closure(ctx, node_id, Glyph.RA, state)

    assert glyph == Glyph.NUL
    assert state["thol_len"] == 3
