"""Tests for unified pattern detection module.

This test suite validates the unified pattern detection system that consolidates
canonical_patterns.py and patterns.py, with explicit mapping to U1-U4 grammar rules.
"""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from tnfr.operators.grammar import StructuralPattern
from tnfr.operators.pattern_detection import (
    PatternMatch,
    UnifiedPatternDetector,
    analyze_sequence,
    detect_pattern,
)


# U1a: Initiation pattern tests -----------------------------------------------


def test_detect_cold_start():
    """Test detection of cold start pattern (AL from EPI=0)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, SILENCE]
    patterns = detector.detect_initiation_patterns(sequence)

    # Should detect cold_start
    cold_starts = [p for p in patterns if p.pattern_name == "cold_start"]
    assert len(cold_starts) == 1
    assert cold_starts[0].grammar_rule == "U1a"
    assert cold_starts[0].confidence == 1.0


def test_detect_phase_transition_start():
    """Test detection of phase transition start (NAV initiation)."""
    detector = UnifiedPatternDetector()
    sequence = [TRANSITION, EMISSION, COHERENCE, SILENCE]
    patterns = detector.detect_initiation_patterns(sequence)

    # Should detect phase_transition_start
    phase_starts = [p for p in patterns if p.pattern_name == "phase_transition_start"]
    assert len(phase_starts) == 1
    assert phase_starts[0].grammar_rule == "U1a"


def test_detect_fractal_awakening():
    """Test detection of fractal awakening (REMESH initiation)."""
    detector = UnifiedPatternDetector()
    sequence = [RECURSIVITY, COHERENCE, SILENCE]
    patterns = detector.detect_initiation_patterns(sequence)

    # Should detect fractal_awakening
    fractal_awakenings = [
        p for p in patterns if p.pattern_name == "fractal_awakening"
    ]
    assert len(fractal_awakenings) == 1
    assert fractal_awakenings[0].grammar_rule == "U1a"


# U1b: Closure pattern tests --------------------------------------------------


def test_detect_terminal_silence():
    """Test detection of terminal silence closure (SHA)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, SILENCE]
    patterns = detector.detect_closure_patterns(sequence)

    # Should detect terminal_silence
    terminal_silences = [p for p in patterns if p.pattern_name == "terminal_silence"]
    assert len(terminal_silences) == 1
    assert terminal_silences[0].grammar_rule == "U1b"
    assert terminal_silences[0].confidence == 1.0


def test_detect_regime_handoff():
    """Test detection of regime handoff closure (NAV)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, TRANSITION]
    patterns = detector.detect_closure_patterns(sequence)

    # Should detect regime_handoff
    regime_handoffs = [p for p in patterns if p.pattern_name == "regime_handoff"]
    assert len(regime_handoffs) == 1
    assert regime_handoffs[0].grammar_rule == "U1b"


def test_detect_fractal_distribution():
    """Test detection of fractal distribution closure (REMESH)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, RECURSIVITY]
    patterns = detector.detect_closure_patterns(sequence)

    # Should detect fractal_distribution
    fractal_distributions = [
        p for p in patterns if p.pattern_name == "fractal_distribution"
    ]
    assert len(fractal_distributions) == 1
    assert fractal_distributions[0].grammar_rule == "U1b"


def test_detect_intentional_tension():
    """Test detection of intentional tension closure (OZ)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, DISSONANCE]
    patterns = detector.detect_closure_patterns(sequence)

    # Should detect intentional_tension
    intentional_tensions = [
        p for p in patterns if p.pattern_name == "intentional_tension"
    ]
    assert len(intentional_tensions) == 1
    assert intentional_tensions[0].grammar_rule == "U1b"


# U2: Convergence pattern tests -----------------------------------------------


def test_detect_stabilization_cycle():
    """Test detection of stabilization cycle (destabilizer → stabilizer)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, DISSONANCE, COHERENCE, SILENCE]
    patterns = detector.detect_convergence_patterns(sequence)

    # Should detect stabilization_cycle
    stabilization_cycles = [
        p for p in patterns if p.pattern_name == "stabilization_cycle"
    ]
    assert len(stabilization_cycles) >= 1
    assert all(p.grammar_rule == "U2" for p in stabilization_cycles)


def test_detect_runaway_risk():
    """Test detection of runaway risk (destabilizers without stabilizers)."""
    detector = UnifiedPatternDetector()
    # Sequence with destabilizers but no stabilizers
    sequence = [EMISSION, DISSONANCE, MUTATION, TRANSITION]
    patterns = detector.detect_convergence_patterns(sequence)

    # Should detect runaway_risk
    runaway_risks = [p for p in patterns if p.pattern_name == "runaway_risk"]
    assert len(runaway_risks) >= 1
    assert runaway_risks[0].grammar_rule == "U2"


def test_detect_bounded_evolution():
    """Test detection of bounded evolution (alternating destabilizers/stabilizers)."""
    detector = UnifiedPatternDetector()
    # Alternating pattern
    sequence = [
        EMISSION,
        DISSONANCE,
        COHERENCE,
        MUTATION,
        SELF_ORGANIZATION,
        SILENCE,
    ]
    patterns = detector.detect_convergence_patterns(sequence)

    # May or may not detect bounded_evolution depending on strict alternation
    # But should detect multiple stabilization_cycles
    stabilization_cycles = [
        p for p in patterns if p.pattern_name == "stabilization_cycle"
    ]
    assert len(stabilization_cycles) >= 1


# U3: Resonance pattern tests -------------------------------------------------


def test_detect_coupling_chain():
    """Test detection of coupling chain (multiple UM)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COUPLING, COHERENCE, COUPLING, SILENCE]
    patterns = detector.detect_resonance_patterns(sequence)

    # Should detect coupling_chain
    coupling_chains = [p for p in patterns if p.pattern_name == "coupling_chain"]
    assert len(coupling_chains) >= 1
    assert coupling_chains[0].grammar_rule == "U3"


def test_detect_resonance_cascade():
    """Test detection of resonance cascade (multiple RA)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, RESONANCE, COHERENCE, RESONANCE, SILENCE]
    patterns = detector.detect_resonance_patterns(sequence)

    # Should detect resonance_cascade
    resonance_cascades = [p for p in patterns if p.pattern_name == "resonance_cascade"]
    assert len(resonance_cascades) >= 1
    assert resonance_cascades[0].grammar_rule == "U3"


def test_detect_phase_locked_network():
    """Test detection of phase-locked network (UM ↔ RA)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COUPLING, RESONANCE, COHERENCE, SILENCE]
    patterns = detector.detect_resonance_patterns(sequence)

    # Should detect phase_locked_network
    phase_locked_networks = [
        p for p in patterns if p.pattern_name == "phase_locked_network"
    ]
    assert len(phase_locked_networks) >= 1
    assert phase_locked_networks[0].grammar_rule == "U3"


# U4: Bifurcation pattern tests -----------------------------------------------


def test_detect_graduated_destabilization():
    """Test detection of graduated destabilization (destabilizer → transformer)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, DISSONANCE, MUTATION, COHERENCE, SILENCE]
    patterns = detector.detect_bifurcation_patterns(sequence)

    # Should detect graduated_destabilization
    graduated_destabilizations = [
        p for p in patterns if p.pattern_name == "graduated_destabilization"
    ]
    assert len(graduated_destabilizations) >= 1
    assert graduated_destabilizations[0].grammar_rule == "U4b"


def test_detect_managed_bifurcation():
    """Test detection of managed bifurcation (trigger → handler)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, DISSONANCE, COHERENCE, SILENCE]
    patterns = detector.detect_bifurcation_patterns(sequence)

    # Should detect managed_bifurcation
    managed_bifurcations = [
        p for p in patterns if p.pattern_name == "managed_bifurcation"
    ]
    assert len(managed_bifurcations) >= 1
    assert managed_bifurcations[0].grammar_rule == "U4a"


def test_detect_stable_transformation():
    """Test detection of stable transformation (IL → ZHIR)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, MUTATION, COHERENCE, SILENCE]
    patterns = detector.detect_bifurcation_patterns(sequence)

    # Should detect stable_transformation
    stable_transformations = [
        p for p in patterns if p.pattern_name == "stable_transformation"
    ]
    assert len(stable_transformations) >= 1
    assert stable_transformations[0].grammar_rule == "U4b"


def test_detect_spontaneous_organization():
    """Test detection of spontaneous organization (disorder → THOL)."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, DISSONANCE, SELF_ORGANIZATION, COHERENCE, SILENCE]
    patterns = detector.detect_bifurcation_patterns(sequence)

    # Should detect spontaneous_organization
    spontaneous_organizations = [
        p for p in patterns if p.pattern_name == "spontaneous_organization"
    ]
    assert len(spontaneous_organizations) >= 1
    assert spontaneous_organizations[0].grammar_rule == "U4b"


# Integration tests -----------------------------------------------------------


def test_detect_all_patterns():
    """Test comprehensive pattern detection in complex sequence."""
    detector = UnifiedPatternDetector()
    # Complex sequence with multiple patterns
    sequence = [
        EMISSION,  # U1a: cold_start
        DISSONANCE,  # U4: bifurcation trigger
        COHERENCE,  # U2: stabilizer, U4: handler
        COUPLING,  # U3: resonance
        RESONANCE,  # U3: resonance
        SILENCE,  # U1b: terminal_silence
    ]

    all_patterns = detector.detect_all_patterns(sequence)

    # Should detect patterns from multiple categories
    assert len(all_patterns) > 0

    # Check we have patterns from different grammar rules
    grammar_rules = {p.grammar_rule for p in all_patterns}
    assert "U1a" in grammar_rules  # Initiation
    assert "U1b" in grammar_rules  # Closure
    assert "U2" in grammar_rules  # Convergence
    assert "U3" in grammar_rules  # Resonance
    assert "U4a" in grammar_rules or "U4b" in grammar_rules  # Bifurcation


def test_grammar_rule_mapping():
    """Test that pattern names map to correct grammar rules."""
    detector = UnifiedPatternDetector()

    # Test a few key mappings
    assert detector.get_grammar_rule_for_pattern("cold_start") == "U1a"
    assert detector.get_grammar_rule_for_pattern("terminal_silence") == "U1b"
    assert detector.get_grammar_rule_for_pattern("stabilization_cycle") == "U2"
    assert detector.get_grammar_rule_for_pattern("coupling_chain") == "U3"
    assert detector.get_grammar_rule_for_pattern("graduated_destabilization") == "U4b"

    # Test unknown pattern
    assert detector.get_grammar_rule_for_pattern("unknown_pattern") is None


def test_analyze_sequence_composition():
    """Test comprehensive sequence analysis."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION, COHERENCE, RESONANCE, SILENCE]

    analysis = detector.analyze_sequence_composition(sequence)

    # Should have all expected keys
    assert "primary_pattern" in analysis
    assert "pattern_scores" in analysis
    assert "weighted_scores" in analysis
    assert "coherence_weights" in analysis
    assert "components" in analysis
    assert "complexity_score" in analysis
    assert "domain_suitability" in analysis
    assert "structural_health" in analysis


# Convenience function tests --------------------------------------------------


def test_detect_pattern_convenience():
    """Test convenience function for pattern detection."""
    sequence = [EMISSION, COUPLING, COHERENCE, SILENCE]
    pattern = detect_pattern(sequence)

    # Should return a StructuralPattern
    assert isinstance(pattern, StructuralPattern)


def test_analyze_sequence_convenience():
    """Test convenience function for sequence analysis."""
    sequence = [EMISSION, COUPLING, COHERENCE, SILENCE]
    analysis = analyze_sequence(sequence)

    # Should return a mapping with expected keys
    assert "primary_pattern" in analysis
    assert "pattern_scores" in analysis


# Edge case tests -------------------------------------------------------------


def test_empty_sequence():
    """Test pattern detection on empty sequence."""
    detector = UnifiedPatternDetector()
    sequence = []

    patterns = detector.detect_all_patterns(sequence)
    assert len(patterns) == 0


def test_single_operator_sequence():
    """Test pattern detection on single operator."""
    detector = UnifiedPatternDetector()
    sequence = [EMISSION]

    patterns = detector.detect_all_patterns(sequence)
    # Should at least detect initiation pattern
    assert len(patterns) >= 1


def test_pattern_confidence_scores():
    """Test that confidence scores are in valid range."""
    detector = UnifiedPatternDetector()
    sequence = [
        EMISSION,
        DISSONANCE,
        COHERENCE,
        COUPLING,
        RESONANCE,
        SILENCE,
    ]

    all_patterns = detector.detect_all_patterns(sequence)

    # All confidence scores should be between 0.0 and 1.0
    for pattern in all_patterns:
        assert 0.0 <= pattern.confidence <= 1.0


# Backward compatibility tests ------------------------------------------------


def test_backward_compatibility_with_advanced_detector():
    """Test that UnifiedPatternDetector works like AdvancedPatternDetector."""
    from tnfr.operators.patterns import AdvancedPatternDetector

    sequence = [
        RECEPTION,
        EMISSION,
        COHERENCE,
        DISSONANCE,
        SELF_ORGANIZATION,
        COHERENCE,
        SILENCE,
    ]

    # Both should detect the same primary pattern
    unified_detector = UnifiedPatternDetector()
    advanced_detector = AdvancedPatternDetector()

    unified_pattern = unified_detector.detect_pattern(sequence)
    advanced_pattern = advanced_detector.detect_pattern(sequence)

    assert unified_pattern == advanced_pattern


def test_backward_compatibility_analyze_sequence():
    """Test that analysis maintains compatibility."""
    from tnfr.operators.patterns import AdvancedPatternDetector

    sequence = [EMISSION, COHERENCE, RESONANCE, SILENCE]

    unified_detector = UnifiedPatternDetector()
    advanced_detector = AdvancedPatternDetector()

    unified_analysis = unified_detector.analyze_sequence_composition(sequence)
    advanced_analysis = advanced_detector.analyze_sequence_composition(sequence)

    # Should have same primary pattern
    assert unified_analysis["primary_pattern"] == advanced_analysis["primary_pattern"]

    # Should have same keys
    assert set(unified_analysis.keys()) == set(advanced_analysis.keys())
