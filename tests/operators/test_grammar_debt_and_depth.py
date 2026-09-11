"""Causal U2 accounting and real Recursivity metadata regressions."""

from __future__ import annotations

from collections import deque

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.glyph_history import push_glyph
from tnfr.operators.definitions import (
    Coherence, Emission, Expansion, Recursivity, Silence,
)
from tnfr.operators.grammar_core import GrammarValidator
from tnfr.operators.grammar_dynamics import (
    validate_candidate, validate_sequence_incremental,
)
from tnfr.operators.grammar_memoization import validate_sequence_optimized


def _graph(history=()):
    graph = nx.path_graph(2)
    inject_defaults(graph)
    for node in graph:
        graph.nodes[node].update(EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0)
    graph.nodes[0]["glyph_history"] = deque(history, maxlen=30)
    return graph


@pytest.mark.parametrize("initial_stabilizers", [0, 1, 8])
def test_later_stabilization_does_not_erase_over_capacity_prefix(initial_stabilizers):
    sequence = (
        [Emission()] + [Coherence()] * initial_stabilizers
        + [Expansion()] * 3 + [Coherence(), Silence()]
    )
    valid, message = GrammarValidator.validate_convergence(sequence)
    assert not valid
    assert "debt" in message.lower()
    assert not validate_sequence_optimized(sequence)[0]


def test_two_destabilizers_are_within_derived_capacity():
    sequence = [Emission(), Expansion(), Expansion(), Coherence(), Silence()]
    assert GrammarValidator.validate_convergence(sequence)[0]


@pytest.mark.parametrize("window", [1, 6, 20])
def test_neutral_history_does_not_compensate_pressure(window):
    graph = _graph(["IL", "VAL", "VAL"] + ["EN"] * 12)
    result = validate_candidate(graph, 0, "VAL", window=window)
    assert not result.allowed
    assert any(violation.rule == "U2" for violation in result.violations)


def test_stabilization_does_not_bank_credit_for_future_destabilizers():
    graph = _graph(["IL"] * 8 + ["VAL", "VAL"])
    assert not validate_candidate(graph, 0, "VAL").allowed


def test_stabilizer_can_repair_preexisting_excess_debt():
    graph = _graph(["VAL"] * 5)
    assert validate_candidate(graph, 0, "IL").allowed
    assert not validate_candidate(graph, 0, "VAL").allowed


@pytest.mark.parametrize("history_window", [0, 1, 7])
def test_pressure_survives_bounded_history_eviction(history_window):
    graph = _graph()
    for glyph in ["VAL", "VAL"] + ["EN"] * 12:
        push_glyph(graph.nodes[0], glyph, history_window)
    assert not validate_candidate(graph, 0, "VAL").allowed


def test_shadow_validation_restores_history_and_debt():
    graph = _graph()
    for glyph in ["VAL", "VAL"] + ["EN"] * 12:
        push_glyph(graph.nodes[0], glyph, 1)
    original = dict(graph.nodes[0])
    history = graph.nodes[0]["glyph_history"]
    results = validate_sequence_incremental(graph, 0, ["IL", "VAL"])
    assert all(result.allowed for result in results)
    assert dict(graph.nodes[0]) == original
    assert graph.nodes[0]["glyph_history"] is history
    assert not validate_candidate(graph, 0, "VAL").allowed


def test_selector_records_each_executed_glyph_once():
    from tnfr.dynamics.selectors import _apply_glyphs

    graph = _graph()
    graph.graph["GRAMMAR_CANON"] = {"enabled": True}
    _apply_glyphs(graph, lambda _graph, _node: "VAL", {})
    assert list(graph.nodes[0]["glyph_history"]) == ["VAL"]


@pytest.mark.parametrize("depth", [2, 3, 6])
def test_real_recursivity_depth_requires_scale_stabilizer(depth):
    sequence = [Emission(), Recursivity(depth=depth), Silence()]
    assert not GrammarValidator().validate(sequence)[0]
    assert not validate_sequence_optimized(sequence)[0]
    sequence.insert(1, Coherence())
    assert GrammarValidator().validate(sequence)[0]
    assert validate_sequence_optimized(sequence)[0]


def test_u5_na_describes_only_the_current_sequence():
    valid, message = GrammarValidator.validate_multiscale_coherence(
        [Emission(), Recursivity(depth=1), Silence()]
    )
    assert valid
    assert "dead code" not in message
    assert "no operator currently exposes" not in message


@pytest.mark.parametrize("depth", [True, 1.5, float("nan"), float("inf")])
def test_recursivity_rejects_nonintegral_depth(depth):
    with pytest.raises((TypeError, ValueError)):
        Recursivity(depth=depth)


def test_live_invalid_depth_is_not_silently_treated_as_shallow():
    recursive = Recursivity(depth=1)
    sequence = [Emission(), Coherence(), recursive, Silence()]
    assert validate_sequence_optimized(sequence)[0]
    recursive.depth = float("nan")
    assert not validate_sequence_optimized(sequence)[0]


@pytest.mark.parametrize("token", ["VAL", "val", "Glyph.VAL", "Expansion"])
def test_serialized_glyph_aliases_preserve_debt(token):
    graph = _graph([token, token])
    assert not validate_candidate(graph, 0, "VAL").allowed


def test_debt_can_be_reset_from_deliberately_replaced_history():
    from tnfr.operators.grammar_debt import reset_debt_from_history

    graph = _graph()
    for glyph in ["VAL", "VAL"]:
        push_glyph(graph.nodes[0], glyph, 1)
    assert not validate_candidate(graph, 0, "VAL").allowed
    graph.nodes[0]["glyph_history"] = []
    reset_debt_from_history(graph.nodes[0])
    assert validate_candidate(graph, 0, "VAL").allowed


def test_invalid_saved_debt_reconstructs_available_history():
    from tnfr.operators.grammar_debt import U2_DEBT_KEY

    graph = _graph(["VAL", "VAL"])
    graph.nodes[0][U2_DEBT_KEY] = float("nan")
    assert not validate_candidate(graph, 0, "VAL").allowed


@pytest.mark.parametrize("history", [None, 5, "VAL", b"VAL"])
def test_recording_discards_malformed_legacy_history(history):
    graph = _graph()
    graph.nodes[0]["glyph_history"] = history
    push_glyph(graph.nodes[0], "EN", 7)
    assert list(graph.nodes[0]["glyph_history"]) == ["EN"]
    assert validate_candidate(graph, 0, "VAL").allowed


def test_legacy_full_sequence_catalog_passes_canonical_grammar():
    from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES

    for name, entry in CANONICAL_SEQUENCES.items():
        valid, messages = validate_sequence_optimized(entry.glyphs)
        assert valid, (name, messages)


@pytest.mark.parametrize("window", [3, 6, 20])
def test_prior_coherence_is_lifetime_context_not_recent_context(window):
    history = ["AL", "IL"] + ["EN"] * 6 + ["VAL", "THOL", "VAL"]
    graph = _graph(history)
    assert validate_sequence_optimized(history + ["ZHIR", "IL", "SHA"])[0]
    assert validate_candidate(graph, 0, "ZHIR", window=window).allowed


@pytest.mark.parametrize("history_window", [3, 6])
def test_prior_coherence_survives_trace_eviction(history_window):
    graph = _graph()
    for glyph in ["IL"] + ["EN"] * 8 + ["VAL", "THOL", "VAL"]:
        push_glyph(graph.nodes[0], glyph, history_window)
    assert "IL" not in graph.nodes[0]["glyph_history"]
    assert validate_candidate(graph, 0, "ZHIR").allowed
    for _ in range(3):
        push_glyph(graph.nodes[0], "EN", history_window)
    result = validate_candidate(graph, 0, "ZHIR")
    assert not result.allowed
    assert any(v.rule == "U4b" and "recent destabilizer" in v.message
               for v in result.violations)


def test_deliberate_trace_reset_clears_lifetime_coherence():
    from tnfr.operators.grammar_debt import reset_grammar_state_from_history

    graph = _graph()
    push_glyph(graph.nodes[0], "IL", 0)
    graph.nodes[0]["glyph_history"] = ["VAL", "THOL", "VAL"]
    assert validate_candidate(graph, 0, "ZHIR").allowed
    reset_grammar_state_from_history(graph.nodes[0])
    result = validate_candidate(graph, 0, "ZHIR")
    assert not result.allowed
    assert any(v.rule == "U4b" and "prior IL" in v.message
               for v in result.violations)


@pytest.mark.parametrize("existing_marker", [False, True])
def test_shadow_coherence_survives_window_and_restores_state(existing_marker):
    from tnfr.operators.grammar_debt import PRIOR_COHERENCE_KEY

    graph = _graph()
    if existing_marker:
        graph.nodes[0][PRIOR_COHERENCE_KEY] = False
    original = dict(graph.nodes[0])
    history = graph.nodes[0]["glyph_history"]
    sequence = ["IL"] + ["EN"] * 8 + ["VAL", "THOL", "VAL", "ZHIR"]
    assert all(r.allowed for r in validate_sequence_incremental(graph, 0, sequence))
    assert dict(graph.nodes[0]) == original
    assert graph.nodes[0]["glyph_history"] is history


def test_shadow_restores_coherence_marker_after_validation_exception(monkeypatch):
    import tnfr.operators.grammar_dynamics as dynamics

    graph = _graph(["IL"])
    original = dict(graph.nodes[0])

    def fail(*args, **kwargs):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(dynamics, "validate_candidate", fail)
    with pytest.raises(RuntimeError, match="injected"):
        validate_sequence_incremental(graph, 0, ["VAL"])
    assert dict(graph.nodes[0]) == original


def test_explicit_application_callback_records_lifetime_coherence():
    from tnfr.operators.grammar_application import on_applied_glyph

    graph = _graph()
    graph.nodes[0]["glyph_history"] = deque(maxlen=3)
    for glyph in ["Coherence"] + ["EN"] * 8 + ["VAL", "THOL", "VAL"]:
        on_applied_glyph(graph, 0, glyph)
    assert validate_candidate(graph, 0, "ZHIR").allowed


@pytest.mark.parametrize("reader", ["public", "modular", "readiness", "telemetry"])
def test_mutation_readers_preserve_evicted_coherence(reader):
    from tnfr.operators.preconditions import validate_mutation
    from tnfr.operators.preconditions.mutation import (
        diagnose_mutation_readiness, validate_grammar_u4b,
    )

    graph = _graph()
    graph.graph.update(ZHIR_REQUIRE_IL_PRECEDENCE=True, ZHIR_REQUIRE_DESTABILIZER=True)
    graph.nodes[0]["nu_f"] = 1.0
    graph.nodes[0]["epi_history"] = [0.1, 0.5]
    for glyph in ["IL"] + ["EN"] * 8 + ["VAL", "THOL", "VAL"]:
        push_glyph(graph.nodes[0], glyph, 3)
    if reader == "public":
        validate_mutation(graph, 0)
    elif reader == "modular":
        validate_grammar_u4b(graph, 0)
    elif reader == "readiness":
        assert diagnose_mutation_readiness(graph, 0)["checks"]["il_precedence"]["passed"]
    else:
        from tnfr.operators.metrics_structural import mutation_metrics

        validate_grammar_u4b(graph, 0)
        assert mutation_metrics(graph, 0, theta_before=0.0, epi_before=0.6)["il_precedence_found"]


@pytest.mark.parametrize("token", ["IL", "il", "Glyph.IL", "Coherence"])
def test_serialized_coherence_aliases_are_remembered_without_trace(token):
    from tnfr.operators.grammar_debt import node_has_prior_coherence

    graph = _graph()
    push_glyph(graph.nodes[0], token, 0)
    assert not graph.nodes[0]["glyph_history"]
    assert node_has_prior_coherence(graph.nodes[0])


@pytest.mark.parametrize("history_type", [list, tuple, deque])
def test_replayable_history_containers_have_identical_grammar_decisions(history_type):
    graph = _graph()
    history = history_type(["VAL", "VAL"])
    graph.nodes[0]["glyph_history"] = history
    original = dict(graph.nodes[0])
    assert not validate_candidate(graph, 0, "VAL").allowed
    assert validate_candidate(graph, 0, "IL").allowed
    results = validate_sequence_incremental(graph, 0, ["VAL", "IL", "VAL"])
    assert [result.allowed for result in results] == [False, True, True]
    assert dict(graph.nodes[0]) == original
    assert graph.nodes[0]["glyph_history"] is history
    assert list(history) == ["VAL", "VAL"]


@pytest.mark.parametrize("history_type", [iter, lambda items: (item for item in items)])
@pytest.mark.parametrize("validation", ["candidate", "shadow"])
def test_one_shot_histories_are_rejected_without_consumption(history_type, validation):
    graph = _graph()
    history = history_type(["VAL", "VAL"])
    graph.nodes[0]["glyph_history"] = history
    original = dict(graph.nodes[0])
    with pytest.raises(ValueError, match="replayable"):
        if validation == "candidate":
            validate_candidate(graph, 0, "VAL")
        else:
            validate_sequence_incremental(graph, 0, ["VAL"])
    assert dict(graph.nodes[0]) == original
    assert graph.nodes[0]["glyph_history"] is history
    assert list(history) == ["VAL", "VAL"]
