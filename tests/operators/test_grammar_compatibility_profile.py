"""Extra word preferences are explicit and never bypass canonical admission."""

from itertools import product

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import (
    Dissonance,
    Emission,
    Mutation,
    Recursivity,
    Silence,
)
from tnfr.operators.grammar_core import GrammarValidator
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import parse_sequence, validate_sequence
from tnfr.operators.operator_contracts import iter_contracts
from tnfr.operators.registry import get_operator_class
from tnfr.operators.word_execution import run_network_sequence


@pytest.mark.parametrize(
    "word,context",
    [
        (["emission", "expansion", "self_organization", "transition"], None),
        (["coherence", "coherence", "silence"], {"initial_epi_nonzero": True}),
        (["coupling", "coupling", "silence"], {"initial_epi_nonzero": True}),
        (
            ["expansion", "coherence"] + ["coupling"] * 16 + ["silence"],
            {"initial_epi_nonzero": True},
        ),
    ],
)
def test_core_profile_removes_only_extra_pair_and_thol_preferences(word, context):
    assert not validate_sequence(
        word, context=context, compatibility_profile="legacy"
    ).passed
    result = validate_sequence(word, context=context, compatibility_profile="core")
    assert validate_sequence(word, context=context) == result
    assert result.passed, result.message
    assert result.metadata["compatibility_profile"] == "core"
    assert result.metadata["canonical_word_checked"]
    assert parse_sequence(word, context=context, compatibility_profile="core").passed
    operators = [get_operator_class(token)() for token in word]
    assert ValidatedSequence(operators, context=context).names == tuple(word)
    with pytest.raises(TNFRValueError, match="legacy"):
        ValidatedSequence(
            operators, context={**(context or {}), "compatibility_profile": "legacy"}
        )


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
        run_network_sequence(
            graph, word, context={**context, "compatibility_profile": "legacy"}
        )
    assert all(not graph.nodes[node].get("glyph_history") for node in graph)
    # Historical pattern recognition remains advisory and cannot veto the word.
    with pytest.warns(UserWarning, match="coherence_coherence_antipattern"):
        run_network_sequence(
            graph,
            word,
            context=context,
        )
    for node in graph:
        assert list(graph.nodes[node]["glyph_history"])[-3:] == ["IL", "IL", "SHA"]


def test_typed_word_honors_explicit_legacy_preferences():
    word = [Emission(), Silence()]
    admitted = ValidatedSequence(word)
    assert admitted.names == ("emission", "silence")
    assert admitted.compatibility_profile == "core"
    with pytest.raises(TNFRValueError, match="legacy.*adjacency policy"):
        ValidatedSequence(word, context={"compatibility_profile": "legacy"})


@pytest.mark.parametrize("profile", [True, 0, [], "", "CORE", "disabled"])
def test_typed_profile_is_admitted_even_before_empty_word_checks(profile):
    with pytest.raises((TypeError, ValueError), match="compatibility_profile"):
        ValidatedSequence([], context={"compatibility_profile": profile})


@pytest.mark.parametrize("profile", ["core", "legacy"])
def test_typed_profiles_preserve_declared_recursion_depth(profile):
    word = [Emission(), Recursivity(depth=2), Silence()]
    with pytest.raises(TNFRValueError, match="U5"):
        ValidatedSequence(word, context={"compatibility_profile": profile})


def test_diagnostic_fragment_waiver_never_creates_executable_context():
    context = {"initial_epi_nonzero": True, "diagnostic": True}
    result = validate_sequence(["dissonance", "mutation"], context=context)
    assert result.passed and not result.metadata["canonical_word_checked"]
    with pytest.raises(TNFRValueError, match="diagnostic.*executable"):
        ValidatedSequence([Dissonance(), Mutation()], context=context)


def test_default_facade_and_typed_words_match_all_registered_adjacent_pairs():
    names = tuple(contract.name for contract in iter_contracts())
    for left, right in product(names, repeat=2):
        tokens = ("emission", left, right, "silence")
        operators = [get_operator_class(name)() for name in tokens]
        expected, _ = GrammarValidator().validate(operators)
        result = validate_sequence(tokens)
        assert result.passed is expected, tokens
        assert result.metadata["compatibility_profile"] == "core"
        assert result.metadata["canonical_word_checked"]
        if expected:
            assert ValidatedSequence(operators).names == tokens
        else:
            with pytest.raises(TNFRValueError):
                ValidatedSequence(operators)


def test_network_executor_admits_the_typed_word_once(monkeypatch):
    calls = []
    validate = GrammarValidator.validate_checks

    def count(self, sequence, *args, **kwargs):
        calls.append(tuple(sequence))
        return validate(self, sequence, *args, **kwargs)

    monkeypatch.setattr(GrammarValidator, "validate_checks", count)
    run_network_sequence(nx.Graph(), ["emission", "silence"], cycles=0)
    assert len(calls) == 1
    assert isinstance(calls[0][0], Emission)
    assert isinstance(calls[0][1], Silence)
