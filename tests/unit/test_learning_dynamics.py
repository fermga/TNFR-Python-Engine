"""Tests for adaptive learning dynamics (AL + T'HOL).

This test suite validates the canonical learning sequences and metrics
as specified in the TNFR operational manual.
"""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from tnfr.dynamics.learning import AdaptiveLearningSystem
from tnfr.metrics.learning_metrics import (
    compute_consolidation_index,
    compute_learning_efficiency,
    compute_learning_plasticity,
    glyph_history_to_operator_names,
)
from tnfr.operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Mutation,
    Reception,
    Recursivity,
    SelfOrganization,
    Silence,
)
from tnfr.operators.grammar import StructuralPattern
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.structural import create_nfr, run_sequence
from tnfr.types import Glyph


# Test learning sequences pattern detection ------------------------------------


def test_basic_learning_pattern_detection():
    """Test detection of basic learning pattern: AL→EN→IL→SHA."""
    detector = AdvancedPatternDetector()
    sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
    pattern = detector.detect_pattern(sequence)
    assert pattern == StructuralPattern.BASIC_LEARNING


def test_deep_learning_pattern_detection():
    """Test detection of deep learning pattern: AL→EN→IL→OZ→THOL→IL→SHA."""
    detector = AdvancedPatternDetector()
    sequence = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        SELF_ORGANIZATION,
        COHERENCE,
        SILENCE,
    ]
    pattern = detector.detect_pattern(sequence)
    assert pattern == StructuralPattern.DEEP_LEARNING


def test_exploratory_learning_pattern_detection():
    """Test detection of exploratory learning: AL→EN→IL→OZ→THOL→RA→IL→SHA."""
    detector = AdvancedPatternDetector()
    sequence = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        SELF_ORGANIZATION,
        RESONANCE,
        COHERENCE,
        SILENCE,
    ]
    pattern = detector.detect_pattern(sequence)
    assert pattern == StructuralPattern.EXPLORATORY_LEARNING


def test_consolidation_cycle_pattern_detection():
    """Test detection of consolidation cycle: AL→EN→IL→REMESH."""
    detector = AdvancedPatternDetector()
    sequence = [EMISSION, RECEPTION, COHERENCE, RECURSIVITY]
    pattern = detector.detect_pattern(sequence)
    assert pattern == StructuralPattern.CONSOLIDATION_CYCLE


def test_adaptive_mutation_pattern_detection():
    """Test detection of adaptive mutation: AL→EN→IL→OZ→ZHIR→NAV."""
    detector = AdvancedPatternDetector()
    sequence = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        MUTATION,
        TRANSITION,
    ]
    pattern = detector.detect_pattern(sequence)
    assert pattern == StructuralPattern.ADAPTIVE_MUTATION


# Test learning metrics ---------------------------------------------------------


def test_learning_plasticity_empty_history():
    """Test plasticity with no operator history."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    plasticity = compute_learning_plasticity(G, node, window=10)
    assert plasticity == 0.0


def test_learning_plasticity_with_reorganization():
    """Test plasticity increases with reorganization operators."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    # Apply operators that increase plasticity (grammar-compliant sequence)
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Dissonance(), SelfOrganization(), Silence()])
    plasticity = compute_learning_plasticity(G, node, window=10)
    assert plasticity > 0.0
    assert plasticity <= 1.0


def test_consolidation_index_empty_history():
    """Test consolidation with no operator history."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    consolidation = compute_consolidation_index(G, node, window=10)
    assert consolidation == 0.0


def test_consolidation_index_with_stabilization():
    """Test consolidation increases with stabilization operators."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    # Apply operators that increase consolidation (grammar-compliant sequence)
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    consolidation = compute_consolidation_index(G, node, window=10)
    assert consolidation > 0.0
    assert consolidation <= 1.0


def test_learning_efficiency_no_ops():
    """Test efficiency with no operators applied."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    G.nodes[node]["epi_initial"] = 0.3
    efficiency = compute_learning_efficiency(G, node)
    assert efficiency == 0.0


def test_learning_efficiency_with_ops():
    """Test efficiency after applying operators."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    G.nodes[node]["epi_initial"] = 0.3
    # Apply grammar-compliant sequence with stabilizer
    run_sequence(G, node, [Emission(), Reception(), Coherence()])
    efficiency = compute_learning_efficiency(G, node)
    assert efficiency >= 0.0


def test_plasticity_consolidation_balance():
    """Test that plasticity and consolidation are complementary."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    
    # Phase 1: High plasticity (reorganization) - grammar-compliant
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Dissonance(), SelfOrganization(), Silence()])
    plasticity1 = compute_learning_plasticity(G, node, window=10)
    consolidation1 = compute_consolidation_index(G, node, window=10)
    
    # Phase 2: Add more consolidation - grammar-compliant
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    plasticity2 = compute_learning_plasticity(G, node, window=10)
    consolidation2 = compute_consolidation_index(G, node, window=10)
    
    # Consolidation should increase
    assert consolidation2 > consolidation1
    # Plasticity proportion should decrease as we add more stabilizers
    assert plasticity2 < plasticity1


# Test AdaptiveLearningSystem ---------------------------------------------------


def test_adaptive_learning_system_init():
    """Test initialization of AdaptiveLearningSystem."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node, learning_rate=0.8)
    assert system.G is G
    assert system.node == node
    assert system.learning_rate == 0.8
    assert system.consolidation_threshold == 0.7


def test_learn_from_input_no_dissonance():
    """Test learning from consonant input (no reorganization needed)."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node, learning_rate=1.0)
    
    # Stimulus close to current EPI - no dissonance
    system.learn_from_input(stimulus=0.35, consolidate=True)
    
    # Check that operators were applied - convert glyphs to operator names
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    assert len(history) > 0
    # Should have AL, EN, IL, SHA
    assert EMISSION in history
    assert RECEPTION in history
    assert COHERENCE in history
    assert SILENCE in history


def test_learn_from_input_with_dissonance():
    """Test learning from dissonant input (reorganization triggered)."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node, learning_rate=0.5)
    
    # Stimulus far from current EPI - dissonance
    system.learn_from_input(stimulus=0.9, consolidate=True)
    
    # Check that operators were applied - convert glyphs to operator names
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    assert len(history) > 0
    # Should have AL, EN, IL, OZ, THOL, IL, SHA
    assert EMISSION in history
    assert RECEPTION in history
    assert DISSONANCE in history
    assert SELF_ORGANIZATION in history
    assert COHERENCE in history
    assert SILENCE in history


def test_consolidate_memory():
    """Test memory consolidation cycle."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node)
    
    system.consolidate_memory()
    
    # Check that consolidation operators were applied - convert glyphs to operator names
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    assert EMISSION in history
    assert RECEPTION in history
    assert COHERENCE in history
    assert RECURSIVITY in history


def test_adaptive_cycle():
    """Test adaptive learning cycle with iterations."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node, consolidation_threshold=0.5)
    
    system.adaptive_cycle(num_iterations=3)
    
    # Check that operators were applied - convert glyphs to operator names
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    assert len(history) > 0
    # Should have multiple AL and THOL applications
    assert history.count(EMISSION) >= 3
    assert history.count(SELF_ORGANIZATION) >= 3


def test_deep_learning_cycle():
    """Test deep learning cycle execution."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node)
    
    system.deep_learning_cycle()
    
    # Verify sequence: AL→EN→IL→OZ→THOL→IL→(SHA or NUL) - grammar may choose closure
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    
    # First 6 operators must match exactly
    expected_prefix = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        SELF_ORGANIZATION,
        COHERENCE,
    ]
    assert history[:6] == expected_prefix
    
    # Last operator must be valid T'HOL closure (SILENCE or CONTRACTION)
    from tnfr.config.operator_names import CONTRACTION
    assert len(history) == 7
    assert history[6] in [SILENCE, CONTRACTION], f"Expected SILENCE or CONTRACTION, got {history[6]}"


def test_exploratory_learning_cycle():
    """Test exploratory learning cycle execution."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node)
    
    system.exploratory_learning_cycle()
    
    # Verify sequence: AL→EN→IL→OZ→THOL→RA→IL→SHA - convert glyphs to operator names
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    expected = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        SELF_ORGANIZATION,
        RESONANCE,
        COHERENCE,
        SILENCE,
    ]
    assert history == expected


def test_adaptive_mutation_cycle():
    """Test adaptive mutation cycle execution."""
    G, node = create_nfr("learner", epi=0.3, vf=1.0)
    system = AdaptiveLearningSystem(G, node)
    
    system.adaptive_mutation_cycle()
    
    # Verify sequence: AL→EN→IL→OZ→ZHIR→NAV - convert glyphs to operator names
    glyphs = list(G.nodes[node].get("glyph_history", []))
    history = glyph_history_to_operator_names(glyphs)
    expected = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        MUTATION,
        TRANSITION,
    ]
    assert history == expected
    history = list(G.nodes[node].get("glyph_history", []))
    expected = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        MUTATION,
        TRANSITION,
    ]
    assert history == expected


# Integration tests -------------------------------------------------------------


def test_full_learning_workflow():
    """Test complete learning workflow with metrics tracking."""
    G, node = create_nfr("learner", epi=0.2, vf=1.0)
    G.nodes[node]["epi_initial"] = 0.2
    
    system = AdaptiveLearningSystem(G, node, learning_rate=0.5)
    
    # Phase 1: Initial learning (dissonant input)
    system.learn_from_input(stimulus=0.8, consolidate=False)
    plasticity1 = compute_learning_plasticity(G, node)
    assert plasticity1 > 0.0  # Should show plasticity
    
    # Phase 2: Consolidation
    system.consolidate_memory()
    consolidation = compute_consolidation_index(G, node)
    assert consolidation > 0.0  # Should show consolidation
    
    # Phase 3: Measure efficiency
    efficiency = compute_learning_efficiency(G, node)
    assert efficiency >= 0.0


def test_learning_pattern_coherence_weights():
    """Test that learning patterns have correct coherence weights."""
    detector = AdvancedPatternDetector()
    
    # Deep learning should have higher weight than basic
    assert (
        detector.COHERENCE_WEIGHTS["DEEP_LEARNING"]
        > detector.COHERENCE_WEIGHTS["BASIC_LEARNING"]
    )
    
    # Exploratory learning should have same weight as deep learning
    assert (
        detector.COHERENCE_WEIGHTS["EXPLORATORY_LEARNING"]
        == detector.COHERENCE_WEIGHTS["DEEP_LEARNING"]
    )
    
    # Consolidation should be level 1 (compositional)
    assert detector.COHERENCE_WEIGHTS["CONSOLIDATION_CYCLE"] == 1.0
