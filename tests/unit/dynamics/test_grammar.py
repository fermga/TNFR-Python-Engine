"""Unit tests for grammar utilities that manage glyph selection sequences."""

from collections import defaultdict, deque

import pytest

import tnfr.dynamics.selectors as selectors
from tnfr.constants import inject_defaults
from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    MUTATION,
    RECEPTION,
    TRANSITION,
)
from tnfr.dynamics import _choose_glyph
from tnfr.operators import apply_glyph
from tnfr.types import Glyph
from tnfr.validation import (
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    glyph_function_name,
    MutationPreconditionError,
    TholClosureError,
    GrammarConfigurationError,
)
from tnfr.validation import record_grammar_violation as _record_violation


def test_compatibility_fallback(graph_canon):
    """Test compatibility transitions according to canonical TNFR grammar.

    After AL (emission), IL (coherence) is now valid according to the
    canonical compatibility table. Previously this would fallback to EN,
    but the updated grammar (issue #X) explicitly lists AL → IL as compatible.
    """
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.AL])
    # AL → IL is now compatible according to canonical TNFR grammar
    assert glyph_function_name(enforce_canonical_grammar(G, 0, Glyph.IL)) == COHERENCE


def test_precondition_oz_to_zhir(graph_canon):
    """Test oz-to-zhir fallback behavior.

    When mutation is attempted without recent dissonance and low ΔNFR,
    the grammar enforcer returns DISSONANCE as a fallback glyph rather
    than raising an error. This maintains TNFR invariant §3.4: operator
    closure requires valid preconditions, fulfilled by automatic injection.
    """
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.NAV])
    nd["ΔNFR"] = 0.0
    # Should return DISSONANCE as fallback, not MUTATION
    result = enforce_canonical_grammar(G, 0, Glyph.ZHIR)
    assert (
        glyph_function_name(result) == DISSONANCE
    ), "Expected DISSONANCE fallback when mutation attempted without prerequisites"
    # With recent dissonance, mutation should be allowed
    nd["glyph_history"] = deque([Glyph.OZ])
    assert glyph_function_name(enforce_canonical_grammar(G, 0, Glyph.ZHIR)) == MUTATION


def test_thol_closure(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.THOL])
    on_applied_glyph(G, 0, Glyph.THOL)
    st = nd["_GRAM"]
    st["thol_len"] = 2
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.7
    with pytest.raises(TholClosureError) as excinfo:
        enforce_canonical_grammar(G, 0, Glyph.EN)
    err = excinfo.value
    assert err.window == st["thol_len"]
    assert err.order[-1] == RECEPTION
    assert err.candidate == RECEPTION
    nd["Si"] = 0.1
    st["thol_len"] = 2
    result = enforce_canonical_grammar(G, 0, Glyph.NUL)
    assert result == Glyph.NUL


def test_repeat_window_and_force(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.ZHIR.value, "OZ"])
    G.graph["GRAMMAR"] = {
        "window": 3,
        "avoid_repeats": ["ZHIR"],
        "force_dnfr": 0.5,
        "force_accel": 0.8,
        "fallbacks": {"ZHIR": "NAV"},
    }
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}

    nd["ΔNFR"] = 0.0
    nd["d2EPI_dt2"] = 0.0
    assert (
        glyph_function_name(enforce_canonical_grammar(G, 0, Glyph.ZHIR)) == TRANSITION
    )

    nd["ΔNFR"] = 0.6
    assert glyph_function_name(enforce_canonical_grammar(G, 0, Glyph.ZHIR)) == MUTATION

    nd["ΔNFR"] = 0.0
    nd["d2EPI_dt2"] = 0.9
    assert glyph_function_name(enforce_canonical_grammar(G, 0, Glyph.ZHIR)) == MUTATION


def test_repeat_invalid_fallback_string(graph_canon):
    """When fallback is invalid string, use canonical fallback instead of raising error."""
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.ZHIR.value])
    G.graph["GRAMMAR"] = {
        "window": 3,
        "avoid_repeats": ["ZHIR"],
        "fallbacks": {"ZHIR": "NOPE"},  # Invalid fallback
    }
    # Should use canonical fallback (coherence) instead of raising error
    result = enforce_canonical_grammar(G, 0, Glyph.ZHIR)
    assert result == Glyph.IL  # IL is coherence, the canonical fallback for ZHIR


def test_repeat_invalid_fallback_type(graph_canon):
    """When fallback is non-string object, raise GrammarConfigurationError."""
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.ZHIR.value])
    obj = object()
    G.graph["GRAMMAR"] = {
        "window": 3,
        "avoid_repeats": ["ZHIR"],
        "fallbacks": {"ZHIR": obj},  # Invalid type
    }
    with pytest.raises(GrammarConfigurationError) as excinfo:
        enforce_canonical_grammar(G, 0, Glyph.ZHIR)
    err = excinfo.value
    assert "fallbacks.ZHIR" in str(err)
    assert "not of type 'string'" in str(err)


def test_choose_glyph_records_violation(graph_canon, monkeypatch):
    """Test that choose_glyph applies fallback behavior instead of raising error.

    When selector returns ZHIR (mutation) without prerequisites, the grammar
    enforcer returns DISSONANCE as fallback. The selector should accept this
    and use it instead, maintaining TNFR operator closure (§3.4).
    """
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.NAV])
    nd["ΔNFR"] = 0.0

    def selector(_, __):
        return Glyph.ZHIR

    h_al = defaultdict(int)
    h_en = defaultdict(int)

    # Should get DISSONANCE fallback instead of error
    result = _choose_glyph(G, 0, selector, True, h_al, h_en, 10, 10)
    assert (
        glyph_function_name(result) == DISSONANCE
    ), "Expected DISSONANCE fallback when selector chooses mutation without prerequisites"


def test_apply_glyph_with_grammar_records_violation(graph_canon):
    """Test that apply_glyph_with_grammar applies fallback behavior.

    When applying ZHIR (mutation) without prerequisites, the grammar
    enforcer returns DISSONANCE as fallback. The function should apply
    the fallback glyph instead of raising an error, maintaining TNFR
    operator closure (§3.4).
    """
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.NAV])
    nd["ΔNFR"] = 0.0

    # Should apply DISSONANCE fallback instead of error
    apply_glyph_with_grammar(G, [0], Glyph.ZHIR, 1)

    # Check that DISSONANCE was applied, not MUTATION
    history = nd.get("glyph_history", deque())
    assert len(history) > 0, "Expected glyph to be applied"
    last_glyph = history[-1]
    assert (
        glyph_function_name(last_glyph) == DISSONANCE
    ), "Expected DISSONANCE fallback to be applied instead of MUTATION"


def test_canonical_enforcement_with_string_history(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph.setdefault("GRAMMAR_CANON", {})["enabled"] = True

    nd = G.nodes[0]
    nd["glyph_history"] = deque([Glyph.AL.value])

    def selector(_, __):
        return "IL"

    h_al = defaultdict(int)
    h_en = defaultdict(int)

    result = _choose_glyph(G, 0, selector, True, h_al, h_en, 10, 10)

    assert isinstance(result, str)
    # AL → IL is now compatible according to canonical TNFR grammar
    assert glyph_function_name(result) == COHERENCE


def test_lag_counters_enforced(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["AL_MAX_LAG"] = 2
    G.graph["EN_MAX_LAG"] = 2
    from tnfr.dynamics import step

    for _ in range(6):
        step(G)
        hist = G.graph["history"]
        assert all(v <= 2 for v in hist["since_AL"].values())
        assert all(v <= 2 for v in hist["since_EN"].values())


def test_apply_glyph_with_grammar_equivalence(graph_canon):
    G_manual = graph_canon()
    G_manual.add_node(0)
    inject_defaults(G_manual)
    G_manual.nodes[0]["glyph_history"] = deque(
        [Glyph.OZ]
    )  # Add dissonance precondition
    G_func = graph_canon()
    G_func.add_node(0)
    inject_defaults(G_func)
    G_func.nodes[0]["glyph_history"] = deque([Glyph.OZ])  # Add dissonance precondition

    # Manual application
    g_eff = enforce_canonical_grammar(G_manual, 0, Glyph.ZHIR)

    apply_glyph(G_manual, 0, g_eff, window=1)
    on_applied_glyph(G_manual, 0, g_eff)

    # Application via helper
    apply_glyph_with_grammar(G_func, [0], Glyph.ZHIR, 1)

    assert G_manual.nodes[0] == G_func.nodes[0]


def test_apply_glyph_with_grammar_multiple_nodes(graph_canon):
    G = graph_canon()
    G.add_node(0, theta=0.0)
    G.add_node(1)
    inject_defaults(G)
    G.nodes[0]["glyph_history"] = deque([Glyph.OZ])
    G.nodes[1]["glyph_history"] = deque(
        [Glyph.OZ]
    )  # Add dissonance precondition for node 1 too

    apply_glyph_with_grammar(G, [0, 1], Glyph.ZHIR, 1)

    assert glyph_function_name(G.nodes[0]["glyph_history"][-1]) == MUTATION
    assert (
        glyph_function_name(G.nodes[1]["glyph_history"][-1]) == MUTATION
    )  # Both should get mutation now


def test_apply_glyph_with_grammar_accepts_iterables(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    G.nodes[0]["glyph_history"] = deque([Glyph.OZ])
    G.nodes[1]["glyph_history"] = deque([Glyph.OZ])
    apply_glyph_with_grammar(G, G.nodes(), Glyph.ZHIR, 1)
    assert glyph_function_name(G.nodes[0]["glyph_history"][-1]) == MUTATION
    assert glyph_function_name(G.nodes[1]["glyph_history"][-1]) == MUTATION

    G2 = graph_canon()
    G2.add_nodes_from([0, 1])
    inject_defaults(G2)
    G2.nodes[0]["glyph_history"] = deque([Glyph.OZ])
    G2.nodes[1]["glyph_history"] = deque([Glyph.OZ])
    apply_glyph_with_grammar(G2, (n for n in G2.nodes()), Glyph.ZHIR, 1)
    assert glyph_function_name(G2.nodes[0]["glyph_history"][-1]) == MUTATION
    assert glyph_function_name(G2.nodes[1]["glyph_history"][-1]) == MUTATION


def test_apply_glyph_with_grammar_defaults_window_from_graph(graph_canon, monkeypatch):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)

    sentinel = 5
    G.graph["GLYPH_HYSTERESIS_WINDOW"] = sentinel

    captured = {}

    def fake_apply_glyph(graph, node_id, glyph, *, window=None):
        captured["window"] = window

    monkeypatch.setattr("tnfr.operators.apply_glyph", fake_apply_glyph)

    apply_glyph_with_grammar(G, [0], Glyph.AL)

    assert captured["window"] == sentinel
