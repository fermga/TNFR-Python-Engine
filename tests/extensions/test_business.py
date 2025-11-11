"""Tests for business domain extension."""

import pytest
from tnfr.extensions.business import BusinessExtension
from tnfr.extensions.business.health_analyzers import ProcessHealthAnalyzer
import networkx as nx


def test_business_extension_domain_name():
    """Test business extension domain name."""
    ext = BusinessExtension()
    assert ext.get_domain_name() == "business"


def test_business_extension_patterns():
    """Test business extension has required patterns."""
    ext = BusinessExtension()
    patterns = ext.get_pattern_definitions()
    
    assert "change_management" in patterns
    assert "workflow_optimization" in patterns
    assert "team_alignment" in patterns


def test_change_management_pattern():
    """Test change management pattern definition."""
    ext = BusinessExtension()
    patterns = ext.get_pattern_definitions()
    
    change = patterns["change_management"]
    assert change.name == "change_management"
    assert len(change.sequence) == 5
    assert len(change.use_cases) >= 3
    assert len(change.examples) >= 3


def test_pattern_health_scores():
    """Test all patterns meet health requirements."""
    ext = BusinessExtension()
    patterns = ext.get_pattern_definitions()
    
    for pattern_name, pattern in patterns.items():
        for example in pattern.examples:
            health = example["health_metrics"]
            assert health["C_t"] > 0.75, f"{pattern_name} example has C_t too low"
            assert health["Si"] > 0.70, f"{pattern_name} example has Si too low"


def test_business_health_analyzers():
    """Test business extension provides health analyzers."""
    ext = BusinessExtension()
    analyzers = ext.get_health_analyzers()
    
    assert "process" in analyzers
    assert analyzers["process"] == ProcessHealthAnalyzer


def test_process_health_analyzer():
    """Test process health analyzer produces valid metrics."""
    analyzer = ProcessHealthAnalyzer()
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    sequence = ["reception", "contraction", "coupling", "resonance"]
    metrics = analyzer.analyze_process_health(G, sequence)
    
    assert "efficiency_potential" in metrics
    assert "change_readiness" in metrics
    assert "alignment_strength" in metrics
    
    # All metrics should be in [0, 1]
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_efficiency_potential_calculation():
    """Test efficiency potential metric calculation."""
    analyzer = ProcessHealthAnalyzer()
    G = nx.path_graph(5)  # Linear path
    
    # Optimization sequence
    sequence = ["reception", "contraction", "coupling", "resonance"]
    metrics = analyzer.analyze_process_health(G, sequence)
    
    assert metrics["efficiency_potential"] > 0.5


def test_change_readiness_calculation():
    """Test change readiness metric calculation."""
    analyzer = ProcessHealthAnalyzer()
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # Change-ready sequence
    ready_sequence = ["dissonance", "mutation", "coherence", "coupling"]
    ready_metrics = analyzer.analyze_process_health(G, ready_sequence)
    
    # Non-change sequence
    stable_sequence = ["coherence", "silence"]
    stable_metrics = analyzer.analyze_process_health(G, stable_sequence)
    
    assert ready_metrics["change_readiness"] > stable_metrics["change_readiness"]


def test_alignment_strength_calculation():
    """Test alignment strength calculation."""
    analyzer = ProcessHealthAnalyzer()
    G = nx.complete_graph(4)  # Fully connected
    
    # Strong alignment sequence
    strong_sequence = ["emission", "reception", "coupling", "resonance", "coherence"]
    strong_metrics = analyzer.analyze_process_health(G, strong_sequence)
    
    # Weak alignment sequence
    weak_sequence = ["contraction"]
    weak_metrics = analyzer.analyze_process_health(G, weak_sequence)
    
    assert strong_metrics["alignment_strength"] > weak_metrics["alignment_strength"]


def test_business_cookbook_recipes():
    """Test business extension provides cookbook recipes."""
    ext = BusinessExtension()
    recipes = ext.get_cookbook_recipes()
    
    assert "change_initiative" in recipes
    assert "process_improvement" in recipes
    assert "team_alignment_meeting" in recipes


def test_change_initiative_recipe():
    """Test change initiative recipe structure."""
    ext = BusinessExtension()
    recipes = ext.get_cookbook_recipes()
    
    change_recipe = recipes["change_initiative"]
    assert len(change_recipe.sequence) == 5
    assert "suggested_nf" in change_recipe.parameters
    assert "success_rate" in change_recipe.validation


def test_business_metadata():
    """Test business extension metadata."""
    ext = BusinessExtension()
    metadata = ext.get_metadata()
    
    assert metadata["domain"] == "business"
    assert "version" in metadata
    assert "use_cases" in metadata
    assert len(metadata["use_cases"]) > 0
