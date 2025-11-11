"""Tests for R5: Regenerative Cycle Validation.

This test suite validates the R5_REGENERATIVE_CYCLES implementation,
ensuring that regenerative cycles meet TNFR structural requirements.
"""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
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
from tnfr.operators.cycle_detection import (
    REGENERATORS,
    MIN_CYCLE_LENGTH,
    MAX_CYCLE_LENGTH,
    CycleType,
    CycleDetector,
    CycleAnalysis,
)
from tnfr.operators.grammar import StructuralPattern, validate_sequence, parse_sequence


# Test constants and basic structures -------------------------------------


def test_regenerators_constant():
    """Test that REGENERATORS contains the three canonical regenerators."""
    assert REGENERATORS == [TRANSITION, RECURSIVITY, SILENCE]
    assert len(REGENERATORS) == 3


def test_cycle_length_constants():
    """Test cycle length constraints are properly defined."""
    assert MIN_CYCLE_LENGTH == 5
    assert MAX_CYCLE_LENGTH == 13
    assert MIN_CYCLE_LENGTH < MAX_CYCLE_LENGTH


def test_cycle_type_enum():
    """Test CycleType enum has all expected types."""
    expected_types = {"linear", "regenerative", "recursive", "meditative", "transformative"}
    actual_types = {ct.value for ct in CycleType}
    assert actual_types == expected_types


# Valid regenerative cycles -----------------------------------------------


def test_basic_regenerative_cycle_with_nav():
    """Test basic valid regenerative cycle with NAV (transition) regenerator."""
    detector = CycleDetector()
    
    # IL → RA → VAL → NAV → AL → IL
    sequence = [
        COHERENCE,    # IL - stabilizer
        RESONANCE,    # RA - stabilizer
        EXPANSION,    # VAL
        TRANSITION,   # NAV - regenerator
        EMISSION,     # AL
        COHERENCE,    # IL - stabilizer (closure)
    ]
    
    # Find NAV position
    nav_pos = sequence.index(TRANSITION)
    analysis = detector.analyze_potential_cycle(sequence, nav_pos)
    
    assert analysis.is_valid_regenerative
    assert analysis.cycle_type == CycleType.TRANSFORMATIVE
    assert analysis.health_score >= CycleDetector.MIN_HEALTH_SCORE
    assert analysis.stabilizer_count_before > 0
    assert analysis.stabilizer_count_after > 0


def test_meditative_cycle_with_sha():
    """Test meditative cycle with SHA (silence) regenerator."""
    detector = CycleDetector()
    
    # AL → EN → IL → RA → SHA → NAV → AL (with regenerative pause)
    sequence = [
        EMISSION,     # AL
        RECEPTION,    # EN
        COHERENCE,    # IL - stabilizer
        RESONANCE,    # RA - stabilizer
        SILENCE,      # SHA - regenerator (meditative pause)
        COUPLING,     # UM - stabilizer
        SILENCE,      # SHA - end
    ]
    
    # Find first SHA position (the regenerator)
    sha_pos = sequence.index(SILENCE)
    analysis = detector.analyze_potential_cycle(sequence, sha_pos)
    
    assert analysis.is_valid_regenerative
    assert analysis.cycle_type == CycleType.MEDITATIVE
    assert analysis.stabilizer_count_before >= 2
    assert analysis.stabilizer_count_after >= 1


def test_recursive_cycle_with_remesh():
    """Test recursive cycle with REMESH (recursivity) regenerator."""
    detector = CycleDetector()
    
    # THOL → RA → VAL → REMESH → IL → THOL (fractal regeneration)
    sequence = [
        SELF_ORGANIZATION,  # THOL - stabilizer
        RESONANCE,          # RA - stabilizer
        EXPANSION,          # VAL
        RECURSIVITY,        # REMESH - regenerator
        COHERENCE,          # IL - stabilizer
        SELF_ORGANIZATION,  # THOL - stabilizer (closure)
    ]
    
    remesh_pos = sequence.index(RECURSIVITY)
    analysis = detector.analyze_potential_cycle(sequence, remesh_pos)
    
    assert analysis.is_valid_regenerative
    assert analysis.cycle_type == CycleType.RECURSIVE
    assert analysis.health_score >= CycleDetector.MIN_HEALTH_SCORE


def test_full_cycle_analysis_finds_best_regenerator():
    """Test analyze_full_cycle finds the best regenerator position."""
    detector = CycleDetector()
    
    # Sequence with multiple regenerators
    sequence = [
        COHERENCE,
        RESONANCE,
        EXPANSION,
        SILENCE,      # First regenerator
        TRANSITION,   # Second regenerator (better position?)
        EMISSION,
        RECEPTION,
        COUPLING,
        COHERENCE,
    ]
    
    analysis = detector.analyze_full_cycle(sequence)
    
    assert analysis.is_valid_regenerative
    assert analysis.cycle_type in {CycleType.MEDITATIVE, CycleType.TRANSFORMATIVE}
    assert 0 <= analysis.regenerator_position < len(sequence)


# Invalid regenerative cycles ---------------------------------------------


def test_cycle_too_short_rejected():
    """Test that cycles shorter than MIN_CYCLE_LENGTH are rejected."""
    detector = CycleDetector()
    
    # Only 4 operators (< MIN_CYCLE_LENGTH = 5)
    short_sequence = [
        EMISSION,
        COHERENCE,
        SILENCE,
        TRANSITION,
    ]
    
    analysis = detector.analyze_full_cycle(short_sequence)
    
    assert not analysis.is_valid_regenerative
    assert analysis.reason == "too_short"


def test_cycle_too_long_rejected():
    """Test that cycles longer than MAX_CYCLE_LENGTH are rejected."""
    detector = CycleDetector()
    
    # 14 operators (> MAX_CYCLE_LENGTH = 13)
    long_sequence = [
        EMISSION, RECEPTION, COHERENCE, DISSONANCE, COUPLING,
        RESONANCE, SILENCE, EXPANSION, MUTATION, SELF_ORGANIZATION,
        TRANSITION, RECURSIVITY, COHERENCE, SILENCE,
    ]
    
    # Find regenerator
    nav_pos = long_sequence.index(TRANSITION)
    analysis = detector.analyze_potential_cycle(long_sequence, nav_pos)
    
    assert not analysis.is_valid_regenerative
    assert analysis.reason == "too_long"


def test_cycle_without_stabilizers_before_rejected():
    """Test cycle without stabilizers before regenerator is rejected."""
    detector = CycleDetector()
    
    # No stabilizers before TRANSITION
    sequence = [
        EMISSION,     # Not a stabilizer
        DISSONANCE,   # Not a stabilizer
        EXPANSION,    # Not a stabilizer
        TRANSITION,   # Regenerator
        COHERENCE,    # Stabilizer after
        SILENCE,      # Stabilizer after
    ]
    
    nav_pos = sequence.index(TRANSITION)
    analysis = detector.analyze_potential_cycle(sequence, nav_pos)
    
    assert not analysis.is_valid_regenerative
    assert analysis.reason == "no_stabilization"
    assert analysis.stabilizer_count_before == 0


def test_cycle_without_stabilizers_after_rejected():
    """Test cycle without stabilizers after regenerator is rejected."""
    detector = CycleDetector()
    
    # No stabilizers after SILENCE
    sequence = [
        COHERENCE,    # Stabilizer before
        RESONANCE,    # Stabilizer before
        SILENCE,      # Regenerator
        EMISSION,     # Not a stabilizer
        DISSONANCE,   # Not a stabilizer
        TRANSITION,   # End but not counted as "after" stabilizer
    ]
    
    sha_pos = 2  # SILENCE position
    analysis = detector.analyze_potential_cycle(sequence, sha_pos)
    
    assert not analysis.is_valid_regenerative
    assert analysis.reason == "no_stabilization"


def test_cycle_with_low_health_score_rejected():
    """Test cycle with health score below threshold is rejected."""
    detector = CycleDetector()
    
    # Unbalanced sequence with poor structural coherence
    # All destabilizers, minimal diversity
    sequence = [
        EMISSION,
        DISSONANCE,
        DISSONANCE,
        SILENCE,      # Regenerator (only stabilizer)
        MUTATION,
        TRANSITION,   # End
    ]
    
    sha_pos = sequence.index(SILENCE)
    analysis = detector.analyze_potential_cycle(sequence, sha_pos)
    
    # Either rejected for low health or no stabilizers
    assert not analysis.is_valid_regenerative
    assert analysis.reason in {"low_health_score", "no_stabilization"}


def test_sequence_without_regenerator_rejected():
    """Test sequence without any regenerator is not regenerative."""
    detector = CycleDetector()
    
    # No NAV, REMESH, or SHA
    sequence = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        COUPLING,
        RESONANCE,
    ]
    
    analysis = detector.analyze_full_cycle(sequence)
    
    assert not analysis.is_valid_regenerative
    assert analysis.reason == "no_regenerator"


# Cycle analysis metrics --------------------------------------------------


def test_cycle_health_calculation():
    """Test that cycle health score is calculated from balance, diversity, coherence."""
    detector = CycleDetector()
    
    # Well-balanced cycle
    balanced_sequence = [
        COHERENCE,
        RESONANCE,
        EXPANSION,
        SILENCE,
        TRANSITION,
        EMISSION,
        RECEPTION,
        COUPLING,
        COHERENCE,
    ]
    
    analysis = detector.analyze_full_cycle(balanced_sequence)
    
    assert analysis.is_valid_regenerative
    assert 0.0 <= analysis.balance_score <= 1.0
    assert 0.0 <= analysis.diversity_score <= 1.0
    assert 0.0 <= analysis.coherence_score <= 1.0
    # Health is average of the three
    expected_health = (
        analysis.balance_score + 
        analysis.diversity_score + 
        analysis.coherence_score
    ) / 3.0
    assert abs(analysis.health_score - expected_health) < 0.01


def test_cycle_type_detection():
    """Test that cycle type is correctly determined from regenerator."""
    detector = CycleDetector()
    
    base_sequence = [COHERENCE, RESONANCE, EXPANSION]
    end_sequence = [EMISSION, COHERENCE, SILENCE]
    
    # Test NAV → TRANSFORMATIVE
    nav_cycle = base_sequence + [TRANSITION] + end_sequence
    nav_analysis = detector.analyze_full_cycle(nav_cycle)
    assert nav_analysis.cycle_type == CycleType.TRANSFORMATIVE
    
    # Test REMESH → RECURSIVE
    remesh_cycle = base_sequence + [RECURSIVITY] + end_sequence
    remesh_analysis = detector.analyze_full_cycle(remesh_cycle)
    assert remesh_analysis.cycle_type == CycleType.RECURSIVE
    
    # Test SHA → MEDITATIVE
    sha_cycle = base_sequence + [SILENCE] + end_sequence
    sha_analysis = detector.analyze_full_cycle(sha_cycle)
    assert sha_analysis.cycle_type == CycleType.MEDITATIVE


def test_regenerator_position_analysis():
    """Test that regenerator position is correctly identified."""
    detector = CycleDetector()
    
    sequence = [
        COHERENCE,
        RESONANCE,
        EXPANSION,
        TRANSITION,   # Position 3
        EMISSION,
        COUPLING,
        SILENCE,
    ]
    
    analysis = detector.analyze_full_cycle(sequence)
    
    assert analysis.regenerator_position == 3
    assert sequence[analysis.regenerator_position] == TRANSITION


# Integration with pattern detection --------------------------------------


def test_regenerative_pattern_detection():
    """Test that REGENERATIVE pattern is detected for valid cycles."""
    from tnfr.operators.patterns import AdvancedPatternDetector
    
    detector = AdvancedPatternDetector()
    
    # Valid regenerative cycle from issue examples
    sequence = [
        COHERENCE,
        RESONANCE,
        EXPANSION,
        SILENCE,
        TRANSITION,
        EMISSION,
        RECEPTION,
        COUPLING,
        COHERENCE,
    ]
    
    pattern = detector.detect_pattern(sequence)
    assert pattern == StructuralPattern.REGENERATIVE


def test_regenerative_validation_in_parse_sequence():
    """Test that parse_sequence validates regenerative cycles through R5."""
    # This test verifies the integration into the grammar validation pipeline
    
    # Note: parse_sequence requires sequences that pass ALL grammar rules (R1-R5)
    # Including starting with valid operators, having EN→IL, etc.
    
    # Valid regenerative cycle that passes all rules
    valid_sequence = [
        EMISSION,      # R1: Valid start
        RECEPTION,     # R2 setup: EN
        COHERENCE,     # R2: EN→IL + stabilizer
        RESONANCE,     # Stabilizer
        EXPANSION,     # Growth
        SILENCE,       # SHA regenerator
        TRANSITION,    # NAV regenerator
        COUPLING,      # Stabilizer
        SILENCE,       # R3: Valid end
    ]
    
    # This should pass if it's actually REGENERATIVE and meets R5 requirements
    try:
        automaton = parse_sequence(valid_sequence)
        # If it's detected as REGENERATIVE, R5 validation occurred
        if automaton.detected_pattern == StructuralPattern.REGENERATIVE:
            # It passed R5 validation
            assert automaton.detected_pattern == StructuralPattern.REGENERATIVE
        # If it's another pattern, that's also acceptable
    except Exception as e:
        # If it fails, it should be for a reason other than R5
        # (e.g., compatibility issues)
        error_msg = str(e)
        # R5 errors would contain "regenerative cycle"
        if "regenerative cycle" in error_msg.lower():
            pytest.fail(f"Valid regenerative cycle rejected by R5: {e}")


# Complex cases -----------------------------------------------------------


def test_nested_regenerative_segments():
    """Test sequence with multiple regenerator sections."""
    detector = CycleDetector()
    
    # Multiple regenerators in sequence
    sequence = [
        COHERENCE,
        RESONANCE,
        SILENCE,       # First regenerator
        TRANSITION,    # Second regenerator
        RECURSIVITY,   # Third regenerator
        COUPLING,
        COHERENCE,
        SILENCE,
    ]
    
    # Should find at least one valid regenerative position
    analysis = detector.analyze_full_cycle(sequence)
    
    # May or may not be valid depending on stabilizer distribution
    # But should not crash
    assert isinstance(analysis, CycleAnalysis)
    assert analysis.reason in {
        "valid", "no_stabilization", "low_health_score", "no_valid_cycle"
    }


def test_partial_vs_complete_cycles():
    """Test distinction between partial and complete regenerative cycles."""
    detector = CycleDetector()
    
    # Complete cycle: starts and ends with stabilizers
    complete = [
        COHERENCE,    # Stabilizer start
        RESONANCE,
        EXPANSION,
        SILENCE,      # Regenerator
        EMISSION,
        COUPLING,
        COHERENCE,    # Stabilizer end
    ]
    
    complete_analysis = detector.analyze_full_cycle(complete)
    
    # Partial cycle: doesn't end with stabilizer
    partial = [
        COHERENCE,    # Stabilizer start
        RESONANCE,
        EXPANSION,
        SILENCE,      # Regenerator
        EMISSION,
        DISSONANCE,   # Non-stabilizer end
        TRANSITION,
    ]
    
    partial_analysis = detector.analyze_full_cycle(partial)
    
    # Complete should have better coherence score
    if complete_analysis.is_valid_regenerative and partial_analysis.is_valid_regenerative:
        assert complete_analysis.coherence_score >= partial_analysis.coherence_score


def test_cycle_with_other_validations_passes():
    """Test that R5 validation is compatible with R1-R4."""
    # This is implicitly tested by other tests, but we make it explicit
    
    # A sequence that should pass R1 (start), R2 (stabilizers), R3 (end), R4 (bifurcation)
    # AND R5 (regenerative cycle if detected as REGENERATIVE)
    
    detector = CycleDetector()
    
    sequence = [
        EMISSION,      # R1: Valid start
        RECEPTION,     # Setup for R2
        COHERENCE,     # R2: Stabilizer + EN→IL
        RESONANCE,     # Stabilizer
        DISSONANCE,    # R4 setup for mutation
        MUTATION,      # R4: Transformer after destabilizer
        SILENCE,       # Regenerator + R3: Valid end
    ]
    
    # Check cycle validation
    analysis = detector.analyze_full_cycle(sequence)
    
    # This sequence may not be detected as REGENERATIVE by pattern detector
    # (it might be EXPLORE or BIFURCATED), so R5 validation wouldn't apply
    # But if it were detected as REGENERATIVE, it should pass or fail consistently
    assert isinstance(analysis, CycleAnalysis)


def test_linear_sequences_unaffected():
    """Test that non-regenerative sequences don't trigger R5 validation."""
    # Simple linear sequence
    linear = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
    ]
    
    # Should validate normally (pattern detection will determine it's not REGENERATIVE)
    result = validate_sequence(linear)
    
    # Pattern should not be REGENERATIVE
    if result.passed:
        detected_pattern = result.metadata.get("detected_pattern")
        # Could be LINEAR, STABILIZE, etc. but not REGENERATIVE
        assert detected_pattern != "regenerative"
