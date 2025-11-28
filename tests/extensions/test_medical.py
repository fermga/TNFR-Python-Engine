"""Tests for medical domain extension."""

import pytest
from tnfr.extensions.medical import MedicalExtension
from tnfr.extensions.medical.health_analyzers import TherapeuticHealthAnalyzer
import networkx as nx


def test_medical_extension_domain_name():
    """Test medical extension domain name."""
    ext = MedicalExtension()
    assert ext.get_domain_name() == "medical"


def test_medical_extension_patterns():
    """Test medical extension has required patterns."""
    ext = MedicalExtension()
    patterns = ext.get_pattern_definitions()

    assert "therapeutic_alliance" in patterns
    assert "crisis_intervention" in patterns
    assert "integration_phase" in patterns


def test_therapeutic_alliance_pattern():
    """Test therapeutic alliance pattern definition."""
    ext = MedicalExtension()
    patterns = ext.get_pattern_definitions()

    alliance = patterns["therapeutic_alliance"]
    assert alliance.name == "therapeutic_alliance"
    assert alliance.sequence == ["emission", "reception", "coherence", "resonance"]
    assert len(alliance.use_cases) >= 3
    assert len(alliance.examples) >= 3


def test_pattern_health_scores():
    """Test all patterns meet health requirements."""
    ext = MedicalExtension()
    patterns = ext.get_pattern_definitions()

    for pattern_name, pattern in patterns.items():
        for example in pattern.examples:
            health = example["health_metrics"]
            assert health["C_t"] > 0.75, f"{pattern_name} example has C_t too low"
            assert health["Si"] > 0.70, f"{pattern_name} example has Si too low"


def test_medical_health_analyzers():
    """Test medical extension provides health analyzers."""
    ext = MedicalExtension()
    analyzers = ext.get_health_analyzers()

    assert "therapeutic" in analyzers
    assert analyzers["therapeutic"] == TherapeuticHealthAnalyzer


def test_therapeutic_health_analyzer():
    """Test therapeutic health analyzer produces valid metrics."""
    analyzer = TherapeuticHealthAnalyzer()
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    sequence = ["emission", "reception", "coherence", "resonance"]
    metrics = analyzer.analyze_therapeutic_health(G, sequence)

    assert "healing_potential" in metrics
    assert "trauma_safety" in metrics
    assert "therapeutic_alliance" in metrics

    # All metrics should be in [0, 1]
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_healing_potential_calculation():
    """Test healing potential metric calculation."""
    analyzer = TherapeuticHealthAnalyzer()
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])

    # Sequence with growth and stability
    sequence = ["expansion", "coherence", "resonance"]
    metrics = analyzer.analyze_therapeutic_health(G, sequence)

    assert metrics["healing_potential"] > 0.5


def test_trauma_safety_calculation():
    """Test trauma safety metric calculation."""
    analyzer = TherapeuticHealthAnalyzer()
    G = nx.Graph()

    # Safe sequence (no destabilizing ops)
    safe_sequence = ["emission", "reception", "coherence"]
    safe_metrics = analyzer.analyze_therapeutic_health(G, safe_sequence)

    # Risky sequence (destabilizing without safety)
    risky_sequence = ["dissonance", "mutation"]
    risky_metrics = analyzer.analyze_therapeutic_health(G, risky_sequence)

    assert safe_metrics["trauma_safety"] > risky_metrics["trauma_safety"]


def test_therapeutic_alliance_strength():
    """Test therapeutic alliance strength calculation."""
    analyzer = TherapeuticHealthAnalyzer()
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Strong connection sequence
    strong_sequence = ["emission", "reception", "coupling", "resonance", "coherence"]
    strong_metrics = analyzer.analyze_therapeutic_health(G, strong_sequence)

    # Weak connection sequence
    weak_sequence = ["silence"]
    weak_metrics = analyzer.analyze_therapeutic_health(G, weak_sequence)

    assert strong_metrics["therapeutic_alliance"] > weak_metrics["therapeutic_alliance"]


def test_medical_cookbook_recipes():
    """Test medical extension provides cookbook recipes."""
    ext = MedicalExtension()
    recipes = ext.get_cookbook_recipes()

    assert "crisis_stabilization" in recipes
    assert "trust_building" in recipes
    assert "insight_integration" in recipes


def test_crisis_stabilization_recipe():
    """Test crisis stabilization recipe structure."""
    ext = MedicalExtension()
    recipes = ext.get_cookbook_recipes()

    crisis_recipe = recipes["crisis_stabilization"]
    assert crisis_recipe.sequence == ["dissonance", "silence", "coherence", "resonance"]
    assert "suggested_nf" in crisis_recipe.parameters
    assert "success_rate" in crisis_recipe.validation


def test_medical_metadata():
    """Test medical extension metadata."""
    ext = MedicalExtension()
    metadata = ext.get_metadata()

    assert metadata["domain"] == "medical"
    assert "version" in metadata
    assert "use_cases" in metadata
    assert len(metadata["use_cases"]) > 0
