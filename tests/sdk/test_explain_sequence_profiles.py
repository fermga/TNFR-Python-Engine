"""SDK word explanations preserve notation and the selected runtime policy."""

from copy import deepcopy
from types import MappingProxyType

import pytest

from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.sdk import TNFR


def test_default_core_and_explicit_legacy_explain_the_same_word_differently():
    word = ["AL", "SHA"]
    core = TNFR.explain_sequence(word, context={"compatibility_profile": "core"})
    default = TNFR.explain_sequence(word)
    legacy = TNFR.explain_sequence(word, context={"compatibility_profile": "legacy"})
    assert core["valid"] is True
    assert core["compatibility_profile"] == "core"
    assert core == default
    assert legacy["valid"] is False
    assert legacy["compatibility_profile"] == "legacy"
    assert "adjacency policy" in legacy["message"]
    assert core["notation"] == legacy["notation"] == "AL → SHA"
    assert core["operators"] == legacy["operators"] == ["Emission", "Silence"]


@pytest.mark.parametrize(
    "word",
    [
        ["AL", "IL", "SHA"],
        ["emission", "coherence", "silence"],
        ["Emission", "Coherence", "Silence"],
    ],
)
def test_names_and_glyphs_share_simple_ordered_notation(word):
    report = TNFR.explain_sequence(word)
    assert report["notation"] == "AL → IL → SHA"
    assert report["valid"] is True
    assert report["compatibility_profile"] == "core"


def test_notation_preserves_repeated_steps_and_rejected_order():
    forward = TNFR.explain_sequence(["AL", "IL", "IL", "SHA"])
    backward = TNFR.explain_sequence(["SHA", "IL", "IL", "AL"])
    assert forward["notation"] == "AL → IL → IL → SHA"
    assert backward["notation"] == "SHA → IL → IL → AL"
    assert backward["valid"] is False
    assert backward["operators"] == ["Silence", "Coherence", "Coherence", "Emission"]


def test_declared_initialization_uses_shared_boolean_context_without_mutation():
    context = {"compatibility_profile": "core", "initial_epi_nonzero": "true"}
    before = deepcopy(context)
    report = TNFR.explain_sequence(["IL", "SHA"], context=MappingProxyType(context))
    assert report["valid"] is True
    assert report["starts_with_generator"] is False
    assert context == before
    false_context = {**context, "initial_epi_nonzero": "false"}
    assert TNFR.explain_sequence(["IL", "SHA"], context=false_context)["valid"] is False


@pytest.mark.parametrize(
    "context",
    [
        True,
        [],
        {"compatibility_profile": True},
        {"compatibility_profile": "manuscript"},
        {"initial_epi_nonzero": "unknown"},
        {"diagnostic": "unknown"},
    ],
)
def test_invalid_context_is_reported_without_claiming_a_selected_profile(context):
    with pytest.raises((TypeError, ValueError)) as failure:
        validate_sequence(["emission", "silence"], context=context)
    report = TNFR.explain_sequence(["AL", "SHA"], context=context)
    assert report["valid"] is False
    assert report["compatibility_profile"] is None
    assert report["message"] == str(failure.value)
    assert report["notation"] == "AL → SHA"


@pytest.mark.parametrize(
    "word", [["unknown"], ["AL → SHA"], ["A'L", "SH'A"], ["THOL[AL]", "SHA"]]
)
def test_unknown_symbols_remain_rejected_instead_of_becoming_a_text_parser(word):
    with pytest.raises(KeyError):
        TNFR.explain_sequence(word, context={"compatibility_profile": "core"})


def test_empty_sequence_is_still_a_rejected_word():
    report = TNFR.explain_sequence([], context={"compatibility_profile": "core"})
    assert report["valid"] is False
    assert report["notation"] == ""
    assert report["compatibility_profile"] == "core"
    assert report["operators"] == []


@pytest.mark.parametrize("profile", ["legacy", "core"])
def test_word_explanation_never_claims_manuscript_state_or_trajectory_admission(
    profile,
):
    report = TNFR.explain_sequence(
        ["AL", "IL", "SHA"], context={"compatibility_profile": profile}
    )
    assert report["valid"] is True
    assert report["validation_scope"] == "word_policy"
    assert report["manuscript_syntax_assessed"] is False
    assert report["state_checks_assessed"] is False
    assert report["trajectory_checks_assessed"] is False
    assert report["execution_admission_assessed"] is False
    assert {
        "live_operator_preconditions",
        "operator_endpoint_receipts",
        "aligned_potential_reference",
        "full_state_closure",
    } <= set(report["missing_evidence"])
