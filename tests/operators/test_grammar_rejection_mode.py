"""Explicit grammar rejection preserves admission and shared executor semantics."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators import grammar_dynamics
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_debt import U2_DEBT_KEY
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import OPERATOR_MAJOR_GAUSS_SEIDEL, TWO_PHASE_JACOBI
from tnfr.operators.word_execution import execute_network_operator_stage, run_network_sequence


def _graph():
    graph = nx.path_graph(2)
    graph.graph["RANDOM_SEED"] = 19
    for node in graph:
        graph.nodes[node].update({
            ALIAS_EPI[0]: 0.5,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": [],
        })
    return graph


def _state(graph):
    return (
        deepcopy(dict(graph.graph)),
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(list(graph.edges(data=True))),
        hasattr(graph, "_last_operator_applied"),
        getattr(graph, "_last_operator_applied", None),
    )


def _forbid_alternative(*args, **kwargs):
    raise AssertionError("rejection must not calculate a replacement")


@pytest.mark.parametrize("mode", [None, "fallback"])
def test_default_and_explicit_compatibility_mode_retain_replacement(mode):
    graph = _graph()
    if mode is not None:
        graph.graph["GRAMMAR_REJECTION_MODE"] = mode

    Dissonance()(graph, 0)

    assert list(graph.nodes[0]["glyph_history"]) == ["IL"]


def test_standalone_strict_rejection_never_calculates_or_applies_a_replacement(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    before = _state(graph)
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    with pytest.raises(StructuralGrammarError) as error:
        Dissonance()(graph, 0)

    assert error.value.rule == "U4a"
    assert error.value.candidate == "OZ"
    assert _state(graph) == before


@pytest.mark.parametrize("mode", [None, True, False, 0, 1, [], {}])
def test_non_string_mode_rejected_even_for_an_admitted_candidate(mode):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = mode
    before = _state(graph)

    with pytest.raises(TypeError, match="GRAMMAR_REJECTION_MODE"):
        grammar_dynamics.enforce_grammar_on_glyph(graph, 0, "IL")

    assert _state(graph) == before


@pytest.mark.parametrize("mode", ["", "RAISE", "raise ", "Fallback", "reject"])
def test_unknown_mode_rejected_even_for_an_admitted_candidate(mode):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = mode
    before = _state(graph)

    with pytest.raises(ValueError, match="GRAMMAR_REJECTION_MODE"):
        Coherence()(graph, 0)

    assert _state(graph) == before


def test_strict_mode_does_not_replace_an_unknown_operator(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    with pytest.raises(StructuralGrammarError) as error:
        grammar_dynamics.enforce_grammar_on_glyph(graph, 0, "unknown")

    assert error.value.rule == "SYNTAX"


def test_filter_remains_an_ordered_admissible_set_without_selection(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    before = _state(graph)
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    assert grammar_dynamics.filter_candidates(graph, 0, ["EN", "OZ", "IL"]) == ["EN", "IL"]
    assert _state(graph) == before


def test_validated_word_cannot_be_downgraded_by_fallback_setting(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    graph.nodes[0]["glyph_history"] = ["VAL", "VAL"]
    graph.nodes[0][U2_DEBT_KEY] = 2
    step = ValidatedSequence([Emission(), Dissonance(), Coherence(), Silence()]).step(1)
    before = _state(graph)
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    with pytest.raises(StructuralGrammarError, match="debt"):
        Dissonance()(graph, 0, sequence_context=step)

    assert _state(graph) == before


def test_strict_stage_rejects_a_later_target_before_any_commit(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    graph.nodes[0]["glyph_history"] = ["IL"]
    before = _state(graph)
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    with pytest.raises(StructuralGrammarError) as error:
        execute_network_operator_stage(graph, Dissonance(), (0, 1))

    assert error.value.rule == "U4a"
    assert _state(graph) == before


def test_admitted_strict_stage_keeps_two_phase_jacobi_semantics(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    for node in graph:
        graph.nodes[node]["glyph_history"] = ["IL"]
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    result = execute_network_operator_stage(graph, Dissonance(), (0, 1))

    assert result.schedule == TWO_PHASE_JACOBI
    assert result.glyph == "OZ"
    assert all(list(graph.nodes[node]["glyph_history"]) == ["IL", "OZ"] for node in graph)


def test_compatibility_stage_retains_operator_major_replacement():
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"

    result = execute_network_operator_stage(graph, Dissonance(), (0, 1))

    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert all(list(graph.nodes[node]["glyph_history"]) == ["IL"] for node in graph)


def test_disabling_word_validation_does_not_disable_strict_live_rejection(monkeypatch):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    before = _state(graph)
    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", _forbid_alternative)

    with pytest.raises(StructuralGrammarError):
        run_network_sequence(graph, ["dissonance"], validate=False)

    assert _state(graph) == before
