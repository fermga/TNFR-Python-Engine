"""Public string entry points cannot bypass canonical causal word rules."""

from types import SimpleNamespace

import pytest

from tnfr.operators.grammar_core import GrammarValidator
from tnfr.operators.grammar_patterns import parse_sequence, validate_sequence
from tnfr.operators.grammar_types import SequenceSyntaxError


@pytest.mark.parametrize("word,rule", [
    (["emission", "dissonance", "mutation", "coherence", "silence"], "U4b"),
    (["emission", "coherence", "expansion", "resonance", "expansion",
      "resonance", "expansion", "coherence", "silence"], "U2"),
])
def test_strings_and_parser_reject_missing_history_or_excess_prefix_debt(word, rule):
    valid, _ = GrammarValidator().validate([SimpleNamespace(name=name) for name in word])
    assert not valid
    result = validate_sequence(word)
    assert not result.passed
    assert result.metadata["canonical_word_checked"]
    assert rule in result.message
    with pytest.raises(SequenceSyntaxError, match=rule):
        parse_sequence(word)


@pytest.mark.parametrize("word", [
    ["emission", "coherence", "dissonance", "mutation", "coherence", "silence"],
    ["emission", "resonance", "silence"],
    ["emission", "coherence", "expansion", "resonance", "expansion",
     "coherence", "silence"],
])
def test_parser_and_validator_share_valid_words_without_unconditional_stabilizer(word):
    result = validate_sequence(word)
    assert result.passed, result.message
    assert parse_sequence(iter(word)).canonical_tokens == tuple(word)
    assert result.metadata["canonical_word_checked"]
    assert not result.metadata["diagnostic_probe"]


def test_initialized_context_does_not_supply_a_prior_coherence_history():
    result = validate_sequence(
        ["dissonance", "mutation", "coherence", "silence"],
        context={"initial_epi_nonzero": True},
    )
    assert not result.passed
    assert "prior IL" in result.message


def test_diagnostic_waiver_is_exact_and_explicit():
    word = ["dissonance", "mutation"]
    assert not validate_sequence(word).passed
    assert not validate_sequence(word, context={"initial_epi_nonzero": True}).passed
    context = {"initial_epi_nonzero": True, "diagnostic": True}
    result = validate_sequence(word, context=context)
    assert result.passed
    assert result.metadata["diagnostic_probe"]
    assert not result.metadata["canonical_word_checked"]
    assert not validate_sequence(word + ["silence"], context=context).passed


def test_additional_legacy_thol_policy_is_not_conflated_with_canonical_core():
    word = ["emission", "expansion", "self_organization", "transition"]
    assert GrammarValidator().validate([SimpleNamespace(name=name) for name in word])[0]
    result = validate_sequence(word)
    assert not result.passed
    assert "self_organization requires terminal closure" in result.message


@pytest.mark.parametrize("word,index,token", [
    (["coherence", "silence"], 0, "coherence"),
    (["emission", "coherence"], 1, "coherence"),
    (["emission", "expansion", "self_organization", "transition"], 3, "transition"),
    (["emission", "unknown", "silence"], 1, "unknown"),
    (["emission", 3, "silence"], 1, 3),
    (["emission", "coherence", "coherence", "silence"], 2, "coherence"),
])
def test_parser_preserves_local_error_position(word, index, token):
    result = validate_sequence(word)
    assert not result.passed
    with pytest.raises(SequenceSyntaxError) as caught:
        parse_sequence(word)
    assert caught.value.index == index
    assert caught.value.token == token
