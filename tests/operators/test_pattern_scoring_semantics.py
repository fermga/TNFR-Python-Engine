"""Tests for heuristic pattern-prior naming and compatibility."""

from __future__ import annotations

import pytest

import tnfr.operators.patterns as patterns_module
from tnfr.operators.grammar import StructuralPattern
from tnfr.operators.pattern_detection import analyze_sequence
from tnfr.operators.patterns import AdvancedPatternDetector


def _therapeutic_sequence() -> list[str]:
    return [
        "reception",
        "emission",
        "dissonance",
        "self_organization",
        "coherence",
        "silence",
    ]


def test_pattern_weights_are_labeled_as_scoring_priors() -> None:
    detector = AdvancedPatternDetector()

    analysis = detector.analyze_sequence_composition(_therapeutic_sequence())

    canonical = analysis["pattern_scoring_weights"]
    compatibility = analysis["coherence_weights"]
    assert canonical == compatibility
    assert canonical is compatibility
    assert analysis["weight_semantics"] == (
        "heuristic_pattern_prior_not_canonical_C_t"
    )
    assert canonical["therapeutic"] > 1.0
    assert analysis["weighted_scores"]["therapeutic"] > 1.0


def test_public_analysis_preserves_the_scoped_compatibility_key() -> None:
    analysis = analyze_sequence(_therapeutic_sequence())

    assert analysis["pattern_scoring_weights"] is analysis["coherence_weights"]
    assert analysis["weight_semantics"] == (
        "heuristic_pattern_prior_not_canonical_C_t"
    )

def test_historical_private_weight_method_delegates_to_canonical_name() -> None:
    detector = AdvancedPatternDetector()

    assert detector._coherence_weights() == detector._pattern_scoring_weights()


@pytest.mark.parametrize(
    "invalid_weight", [True, -0.1, float("nan"), float("inf"), 10**400]
)
def test_pattern_weight_validation_rejects_invalid_priors(
    monkeypatch: pytest.MonkeyPatch,
    invalid_weight,
) -> None:
    monkeypatch.setitem(
        patterns_module._PATTERN_SCORING_WEIGHTS,
        StructuralPattern.THERAPEUTIC,
        invalid_weight,
    )

    with pytest.raises(ValueError, match="pattern-scoring weight"):
        AdvancedPatternDetector().analyze_sequence_composition(
            _therapeutic_sequence()
        )
