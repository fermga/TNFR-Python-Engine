"""Tests for the operator grammar DSL and helpers."""

from collections import deque

import networkx as nx
import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
    operator_display_name,
)
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.validation import (
    GrammarContext,
    MutationPreconditionError,
    RepeatWindowError,
    TholClosureError,
    SequenceSyntaxError,
    SequenceValidationResult,
    ValidationOutcome,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    glyph_function_name,
    on_applied_glyph,
    parse_sequence,
    validate_sequence,
)
from tnfr.types import Glyph

def _canonical_sequence() -> list[str]:
    return [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]

def test_validate_sequence_success() -> None:
    result = validate_sequence(_canonical_sequence())
    assert isinstance(result, ValidationOutcome)
    assert isinstance(result, SequenceValidationResult)
    assert result.passed
    assert result.message == "ok"
    assert result.tokens == tuple(_canonical_sequence())
    assert result.canonical_tokens == tuple(_canonical_sequence())
    assert result.artifacts["canonical_tokens"] == tuple(_canonical_sequence())
    assert result.metadata["unknown_tokens"] == frozenset()
    assert result.summary["tokens"] == tuple(_canonical_sequence())

def test_validate_sequence_via_keyword_argument() -> None:
    result = validate_sequence(names=_canonical_sequence())
    assert result.passed
    assert result.metadata["has_intermediate"]

def test_validate_sequence_rejects_missing_argument() -> None:
    with pytest.raises(TypeError) as excinfo:
        validate_sequence()
    assert "missing required argument" in str(excinfo.value)

def test_validate_sequence_rejects_unexpected_keyword() -> None:
    with pytest.raises(TypeError) as excinfo:
        validate_sequence(_canonical_sequence(), legacy=True)
    assert "unexpected keyword" in str(excinfo.value)

def test_validate_sequence_requires_string_tokens() -> None:
    result = validate_sequence([EMISSION, RECEPTION, 42])
    assert not result.passed
    assert result.message == "tokens must be str"
    assert result.error is not None
    assert result.summary["error"]["index"] == 2

def test_validate_sequence_requires_valid_start() -> None:
    result = validate_sequence([RECEPTION, COHERENCE, RESONANCE, SILENCE])
    assert not result.passed
    assert "must start" in result.message

def test_validate_sequence_requires_intermediate_segment() -> None:
    result = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE])
    assert not result.passed
    assert "missing" in result.message
    assert result.metadata["has_reception"]
    assert result.metadata["has_coherence"]

def test_validate_sequence_requires_known_tokens() -> None:
    result = validate_sequence([*_canonical_sequence(), "unknown"])
    assert not result.passed
    assert "unknown tokens" in result.message
    assert "unknown" in result.message

def test_validate_sequence_reports_first_unknown_token_index() -> None:
    probe = [EMISSION, "UNKNOWN", RECEPTION, COHERENCE, RESONANCE, SILENCE]
    result = validate_sequence(probe)
    assert not result.passed
    assert result.summary["error"]["index"] == 1
    assert result.summary["error"]["token"] == "UNKNOWN"

def test_validate_sequence_requires_thol_closure() -> None:
    result = validate_sequence(
        [EMISSION, RECEPTION, COHERENCE, SELF_ORGANIZATION, RESONANCE, TRANSITION]
    )
    assert not result.passed
    assert operator_display_name(SELF_ORGANIZATION) in result.message

def test_parse_sequence_returns_result() -> None:
    parsed = parse_sequence(_canonical_sequence())
    assert parsed.passed
    assert parsed.subject == tuple(_canonical_sequence())
    assert parsed.metadata["has_intermediate"]

def test_parse_sequence_propagates_errors() -> None:
    """Test that parse_sequence raises appropriate errors for invalid sequences.
    
    With enhanced compatibility validation, the error may be caught as either:
    - "incompatible" transition (if transition rules fail first)
    - "missing" segment (if the sequence reaches finalization)
    """
    with pytest.raises(SequenceSyntaxError) as excinfo:
        parse_sequence([EMISSION, RECEPTION, TRANSITION])
    # Either error is valid - both indicate structural problems
    error_msg = str(excinfo.value)
    assert "missing" in error_msg or "incompatible" in error_msg

def _make_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd.setdefault("glyph_history", deque())
    return G

def test_grammar_context_isolates_default_configurations() -> None:
    g1 = nx.Graph()
    g2 = nx.Graph()

    ctx_one = GrammarContext.from_graph(g1)
    ctx_two = GrammarContext.from_graph(g2)

    ctx_one.cfg_soft["fallbacks"]["ZHIR"] = "CUSTOM"

    default_value = DEFAULTS["GRAMMAR"]["fallbacks"]["ZHIR"]

    assert ctx_two.cfg_soft["fallbacks"]["ZHIR"] == default_value
    assert DEFAULTS["GRAMMAR"]["fallbacks"]["ZHIR"] == default_value

def test_enforce_canonical_grammar_skips_unknown_tokens() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    result = enforce_canonical_grammar(G, 0, "UNKNOWN", ctx)
    assert result == "UNKNOWN"

def test_enforce_canonical_grammar_returns_structural_name_for_text_input() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)

    result = enforce_canonical_grammar(G, 0, EMISSION, ctx)

    assert result == EMISSION

def test_enforce_canonical_grammar_respects_thol_state() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    on_applied_glyph(G, 0, Glyph.THOL)
    nd = G.nodes[0]
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.7
    ctx.cfg_canon.update({"thol_min_len": 0, "thol_max_len": 1, "thol_close_dnfr": 1.0})
    st = nd["_GRAM"]
    st["thol_len"] = 2
    with pytest.raises(TholClosureError) as excinfo:
        enforce_canonical_grammar(G, 0, Glyph.EN, ctx)
    err = excinfo.value
    assert err.order[-1] == RECEPTION
    expected_si_high = float(ctx.cfg_canon.get("si_high", 0.66))
    assert err.context["si"] == pytest.approx(nd["Si"])
    assert err.context["si_high"] == pytest.approx(expected_si_high)

def test_enforce_canonical_grammar_prefers_silence_when_si_high() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    on_applied_glyph(G, 0, SELF_ORGANIZATION)
    nd = G.nodes[0]
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.92
    ctx.cfg_canon.update(
        {"thol_min_len": 0, "thol_max_len": 1, "thol_close_dnfr": 1.0, "si_high": 0.7}
    )
    st = nd["_GRAM"]
    st["thol_len"] = 2

    result = enforce_canonical_grammar(G, 0, Glyph.NUL, ctx)

    assert result == Glyph.SHA

def test_enforce_canonical_grammar_prefers_contraction_when_si_low() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    on_applied_glyph(G, 0, SELF_ORGANIZATION)
    nd = G.nodes[0]
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.18
    ctx.cfg_canon.update(
        {"thol_min_len": 0, "thol_max_len": 1, "thol_close_dnfr": 1.0, "si_high": 0.4}
    )
    st = nd["_GRAM"]
    st["thol_len"] = 2

    result = enforce_canonical_grammar(G, 0, Glyph.SHA, ctx)

    assert result == Glyph.NUL

def test_enforce_canonical_grammar_accepts_canonical_strings() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    on_applied_glyph(G, 0, SELF_ORGANIZATION)
    nd = G.nodes[0]
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.7
    ctx.cfg_canon.update({"thol_min_len": 0, "thol_max_len": 1, "thol_close_dnfr": 1.0})
    st = nd["_GRAM"]
    st["thol_len"] = 2

    with pytest.raises(TholClosureError) as excinfo:
        enforce_canonical_grammar(G, 0, RECEPTION, ctx)

    err = excinfo.value
    assert err.order[-1] == RECEPTION

def test_mutation_precondition_error_uses_structural_order() -> None:
    """Test that mutation fallback uses correct structural labels.
    
    When mutation is attempted without prerequisites, the grammar enforcer
    returns DISSONANCE as fallback. This test verifies the fallback is
    applied correctly with structural semantics preserved.
    """
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    nd = G.nodes[0]
    history = nd["glyph_history"]
    history.append(Glyph.AL.value)
    history.append(Glyph.NAV)
    history.append(Glyph.REMESH.value)
    nd["ΔNFR"] = 0.0

    # Should return DISSONANCE fallback instead of raising error
    result = enforce_canonical_grammar(G, 0, Glyph.ZHIR, ctx)
    result_name = glyph_function_name(result)
    assert result_name == DISSONANCE, \
        f"Expected DISSONANCE fallback, got {result_name}"

def test_thol_closure_error_uses_structural_order() -> None:
    G = _make_graph()
    ctx = GrammarContext.from_graph(G)
    nd = G.nodes[0]
    history = nd["glyph_history"]
    history.append("THOL")
    history.append(Glyph.AL.value)
    on_applied_glyph(G, 0, Glyph.THOL)
    st = nd["_GRAM"]
    st["thol_len"] = 2
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.7

    with pytest.raises(TholClosureError) as excinfo:
        enforce_canonical_grammar(G, 0, Glyph.EN, ctx)

    err = excinfo.value
    assert err.order == (SELF_ORGANIZATION, EMISSION, RECEPTION)
    assert err.candidate == RECEPTION

def test_on_applied_glyph_canonical_strings_toggle_thol_state() -> None:
    G = _make_graph()
    on_applied_glyph(G, 0, SELF_ORGANIZATION)
    st = G.nodes[0]["_GRAM"]
    assert st["thol_open"]
    assert st["thol_len"] == 0

    on_applied_glyph(G, 0, SILENCE)
    assert not st["thol_open"]
    assert st["thol_len"] == 0

def test_apply_glyph_with_grammar_invokes_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    G = _make_graph()
    captured: dict[str, object] = {}

    def fake_apply(graph, node_id, glyph, *, window=None):
        captured["args"] = (graph, node_id, glyph, window)

    monkeypatch.setattr(
        "tnfr.validation.enforce_canonical_grammar",
        lambda graph, node, cand, ctx=None: cand,
    )
    monkeypatch.setattr("tnfr.operators.apply_glyph", fake_apply, raising=False)

    apply_glyph_with_grammar(G, [0], Glyph.AL, window=7)

    assert captured["args"][0] is G
    assert captured["args"][1] == 0
    assert captured["args"][2] == Glyph.AL
    assert captured["args"][3] == 7

def test_apply_glyph_with_grammar_accepts_glyph_instances() -> None:
    G = _make_graph()

    apply_glyph_with_grammar(G, [0], Glyph.AL)

    history = tuple(G.nodes[0]["glyph_history"])
    assert history[-1] == Glyph.AL.value

def test_apply_glyph_with_grammar_translates_canonical_strings() -> None:
    G = _make_graph()

    apply_glyph_with_grammar(G, [0], EMISSION)

    history = tuple(G.nodes[0]["glyph_history"])
    assert history[-1] == Glyph.AL.value

def test_repeat_window_error_uses_structural_names() -> None:
    G = _make_graph()
    nd = G.nodes[0]
    nd["glyph_history"].extend([Glyph.AL.value])
    grammar_cfg = G.graph.setdefault("GRAMMAR", {})
    grammar_cfg.update({
        "window": 2,
        "avoid_repeats": [Glyph.AL.value],
        "fallbacks": {Glyph.AL.value: Glyph.AL.value},
    })

    with pytest.raises(RepeatWindowError) as excinfo:
        apply_glyph_with_grammar(G, [0], Glyph.AL)

    err = excinfo.value
    assert "emission" in str(err)
    assert err.candidate == EMISSION
    assert err.order is not None
    assert err.order[-1] == EMISSION
    assert err.context.get("fallback") == EMISSION

    telemetry = G.graph["telemetry"]["grammar_errors"][-1]
    assert telemetry["candidate"] == EMISSION
    assert telemetry["order"][-1] == EMISSION
    history = telemetry["context"]["history"]
    assert tuple(history) == (EMISSION,)

def test_apply_glyph_with_grammar_canonical_string_violation() -> None:
    G = _make_graph()
    nd = G.nodes[0]
    nd["ΔNFR"] = 0.0
    nd["Si"] = 0.7
    on_applied_glyph(G, 0, SELF_ORGANIZATION)
    st = nd["_GRAM"]
    st["thol_len"] = 2
    G.graph.setdefault("GRAMMAR_CANON", {}).update(
        {"thol_min_len": 0, "thol_max_len": 1, "thol_close_dnfr": 1.0}
    )

    with pytest.raises(TholClosureError):
        apply_glyph_with_grammar(G, [0], RECEPTION, 1)
