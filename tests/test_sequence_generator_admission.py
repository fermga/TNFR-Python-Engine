"""Heuristic proposal scores never replace complete-word admission."""

from copy import deepcopy

import pytest

from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.sdk.study import list_sequences
from tnfr.tools import sequence_generator as generation
from tnfr.tools.domain_templates import DOMAIN_TEMPLATES, get_template
from tnfr.tools.sequence_generator import ContextualSequenceGenerator


@pytest.mark.parametrize(
    "domain,objective",
    [
        (domain, objective)
        for domain, entries in DOMAIN_TEMPLATES.items()
        for objective in entries
    ],
)
def test_every_domain_result_obeys_word_and_length_constraints(domain, objective):
    result = ContextualSequenceGenerator(seed=0).generate_for_context(
        domain, objective, max_length=8, min_health=1.0
    )
    context = result.metadata["grammar_context"]
    outcome = validate_sequence(result.sequence, context=context)
    assert outcome.passed and outcome.metadata["canonical_word_checked"]
    assert 0 < len(result.sequence) <= 8
    assert context == {
        "compatibility_profile": "core",
        "initial_epi_nonzero": False,
        "diagnostic": False,
    }
    assert result.metadata["compatibility_profile"] == "core"
    assert result.metadata["constraint_met"] == (result.health_score >= 1.0)


def test_high_scoring_unclosed_healing_proposal_is_not_offered(monkeypatch):
    generator = ContextualSequenceGenerator(seed=0)
    invalid = [
        "emission",
        "resonance",
        "expansion",
        "silence",
        "transition",
        "coherence",
    ]
    assert generator.health_analyzer.analyze_health(invalid).overall_health > 0.70
    assert not validate_sequence(invalid).passed
    monkeypatch.setattr(generator, "_generate_variations", lambda *a, **k: [invalid])
    result = generator.generate_for_context("therapeutic", "healing_cycle")
    assert result.sequence != invalid
    assert result.metadata["candidate_origin"].startswith("registered_recipe:")
    name = result.metadata["candidate_origin"].split(":", 1)[1]
    assert result.sequence == list_sequences(name)["operators"]
    assert validate_sequence(result.sequence).passed


@pytest.mark.parametrize("pattern", ["BOOTSTRAP", "bootstrap"])
def test_pattern_case_and_completion_are_both_enforced(pattern):
    result = ContextualSequenceGenerator(seed=0).generate_for_pattern(
        pattern, max_length=4, min_health=1.0
    )
    assert result.sequence == ["emission", "coupling", "coherence", "silence"]
    assert validate_sequence(result.sequence).passed
    assert result.metadata["target_pattern"] == "BOOTSTRAP"
    assert result.metadata["pattern_matched"] is True
    assert result.metadata["constraint_met"] is False


def test_fallback_cannot_drop_requested_pattern_or_length():
    generator = ContextualSequenceGenerator(seed=0)
    with pytest.raises(ValueError, match="No complete word"):
        generator.generate_for_pattern("BOOTSTRAP", max_length=2, min_health=0)
    with pytest.raises(ValueError, match="No complete word"):
        generator.generate_for_context(
            "therapeutic",
            "crisis_intervention",
            max_length=2,
            min_health=0,
            required_pattern="BOOTSTRAP",
        )


def test_explicit_profile_and_inherited_form_are_consumed(monkeypatch):
    generator = ContextualSequenceGenerator(seed=0)
    monkeypatch.setattr(
        generation, "get_template", lambda *args: ["coherence", "silence"]
    )
    monkeypatch.setattr(generator, "_generate_variations", lambda base, *a, **k: [base])
    monkeypatch.setattr(generator, "_registered_candidates", lambda: [])
    with pytest.raises(ValueError, match="No complete word"):
        generator.generate_for_context("therapeutic", "healing_cycle", min_health=0)
    supplied = {"initial_epi_nonzero": "true", "compatibility_profile": "legacy"}
    result = generator.generate_for_context(
        "therapeutic", "healing_cycle", min_health=0, grammar_context=supplied
    )
    assert result.sequence == ["coherence", "silence"]
    assert result.metadata["grammar_context"]["initial_epi_nonzero"] is True
    assert result.metadata["compatibility_profile"] == "legacy"
    assert supplied["initial_epi_nonzero"] == "true"
    monkeypatch.setattr(
        generation, "get_template", lambda *args: ["emission", "silence"]
    )
    assert generator.generate_for_context(
        "therapeutic", "healing_cycle", min_health=0
    ).sequence == ["emission", "silence"]
    with pytest.raises(ValueError, match="No complete word"):
        generator.generate_for_context(
            "therapeutic",
            "healing_cycle",
            min_health=0,
            grammar_context={"compatibility_profile": "legacy"},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grammar_context": {"diagnostic": True}},
        {"grammar_context": {"initial_epi_nonzero": "unknown"}},
        {"grammar_context": {"compatibility_profile": True}},
        {"max_length": True},
        {"max_length": 0},
        {"min_health": float("nan")},
        {"min_health": True},
        {"required_pattern": "NOT_A_PATTERN"},
    ],
)
def test_invalid_request_rejects_before_sampling(kwargs):
    generator = ContextualSequenceGenerator(seed=42)
    before = generator._rng.getstate()
    with pytest.raises((TypeError, ValueError)):
        generator.generate_for_context("therapeutic", "healing_cycle", **kwargs)
    assert generator._rng.getstate() == before


def test_improvement_admits_result_and_rejects_unresolved_transformer():
    generator = ContextualSequenceGenerator(seed=0)
    word, recommendations = generator.improve_sequence(
        ["emission", "resonance"], target_health=1.0, max_length=3
    )
    assert word == ["emission", "resonance", "silence"]
    assert validate_sequence(word).passed
    assert any("Word admission context:" in item for item in recommendations)
    assert any("threshold was not reached" in item for item in recommendations)
    with pytest.raises(ValueError, match="No complete word"):
        generator.improve_sequence(["mutation"])


def test_proposal_reader_is_detached_and_generation_does_not_rewrite_templates():
    original = deepcopy(DOMAIN_TEMPLATES)
    proposal = get_template("therapeutic", "healing_cycle")
    proposal.clear()
    assert get_template("therapeutic", "healing_cycle")
    ContextualSequenceGenerator(seed=0).generate_for_context(
        "therapeutic", "healing_cycle"
    )
    assert DOMAIN_TEMPLATES == original
