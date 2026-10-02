"""Extra word preferences are explicit and never bypass canonical admission."""

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.grammar_patterns import parse_sequence, validate_sequence
from tnfr.operators.word_execution import run_network_sequence


@pytest.mark.parametrize(
    "word,context",
    [
        (["emission", "expansion", "self_organization", "transition"], None),
        (["coherence", "coherence", "silence"], {"initial_epi_nonzero": True}),
    ],
)
def test_core_profile_removes_only_extra_pair_and_thol_preferences(word, context):
    assert not validate_sequence(word, context=context).passed
    result = validate_sequence(word, context=context, compatibility_profile="core")
    assert result.passed, result.message
    assert result.metadata["compatibility_profile"] == "core"
    assert result.metadata["canonical_word_checked"]
    assert parse_sequence(word, context=context, compatibility_profile="core").passed


@pytest.mark.parametrize(
    "word",
    [
        ["coherence", "silence"],  # no existing-form context
        ["emission", "coherence"],  # no closure
        ["emission", "dissonance", "mutation", "coherence", "silence"],
        ["emission", "coherence", "expansion", "expansion", "expansion", "silence"],
    ],
)
def test_core_profile_keeps_initialization_closure_context_and_debt(word):
    assert not validate_sequence(word, compatibility_profile="core").passed


@pytest.mark.parametrize("value", [True, 0, [], "", "CORE", "derived", "disabled"])
def test_invalid_profile_is_rejected_even_for_an_empty_word(value):
    with pytest.raises((TypeError, ValueError), match="compatibility_profile"):
        validate_sequence([], compatibility_profile=value)


def test_explicit_argument_overrides_context_profile():
    word = ["coherence", "coherence", "silence"]
    context = {"initial_epi_nonzero": True, "compatibility_profile": "core"}
    assert validate_sequence(word, context=context).passed
    assert not validate_sequence(
        word, context=context, compatibility_profile="legacy"
    ).passed


def test_existing_word_executor_carries_explicit_profile_without_replacements():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=1.0, nu_f=1.0, theta=0.0, delta_nfr=0.1)
    word = ["coherence", "coherence", "silence"]
    context = {"initial_epi_nonzero": True}
    with pytest.raises(TNFRValueError, match="adjacency policy"):
        run_network_sequence(graph, word, context=context)
    assert all(not graph.nodes[node].get("glyph_history") for node in graph)
    # Historical pattern recognition remains advisory and cannot veto the word.
    with pytest.warns(UserWarning, match="coherence_coherence_antipattern"):
        run_network_sequence(
            graph,
            word,
            context={**context, "compatibility_profile": "core"},
        )
    for node in graph:
        assert list(graph.nodes[node]["glyph_history"])[-3:] == ["IL", "IL", "SHA"]
