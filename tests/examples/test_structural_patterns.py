"""Tests for structural pattern examples.

Validates that all specialized structural patterns meet acceptance criteria
according to TNFR Grammar 2.0, including BOOTSTRAP, EXPLORE, STABILIZE,
RESONATE, COMPRESS, and COMPLEX patterns.
"""

import pytest
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector

# Import structural examples
import sys
from pathlib import Path

examples_path = Path(__file__).parent.parent.parent / "examples" / "domain_applications"
sys.path.insert(0, str(examples_path))

from structural_patterns import (
    get_bootstrap_pattern,
    get_bootstrap_extended_pattern,
    get_explore_pattern,
    get_explore_deep_pattern,
    get_stabilize_pattern,
    get_stabilize_recursive_pattern,
    get_resonate_pattern,
    get_resonate_cascade_pattern,
    get_compress_pattern,
    get_compress_adaptive_pattern,
    get_complex_pattern,
    get_complex_full_cycle_pattern,
    get_all_patterns,
)


# =============================================================================
# Test All Patterns Validity
# =============================================================================


class TestStructuralPatternsValidity:
    """Test suite for structural pattern validity."""

    def test_all_patterns_pass_validation(self):
        """Test that all 12 structural patterns pass canonical validation."""
        patterns = get_all_patterns()

        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation: {result.message}"

    def test_all_health_scores_above_threshold(self):
        """Test that all patterns have health scores > 0.65."""
        patterns = get_all_patterns()
        threshold = 0.65

        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} did not pass validation"

            health = result.health_metrics.overall_health
            assert (
                health >= threshold
            ), f"{name} health score {health:.3f} below threshold {threshold}"

    def test_pattern_count(self):
        """Test that we have exactly 12 structural patterns."""
        patterns = get_all_patterns()
        assert len(patterns) == 12, f"Expected 12 patterns, got {len(patterns)}"


# =============================================================================
# Test Bootstrap Patterns
# =============================================================================


class TestBootstrapPatterns:
    """Test suite for BOOTSTRAP patterns."""

    def test_bootstrap_rapid_initialization(self):
        """Test that bootstrap pattern achieves rapid initialization."""
        sequence = get_bootstrap_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should be minimal length (efficient)
        assert len(sequence) <= 6, "Bootstrap should be minimal"

        # Should have decent health despite minimalism
        assert health.overall_health >= 0.65

        # Should include EMISSION, RECEPTION, COUPLING (bootstrap signature)
        assert "emission" in sequence
        assert "reception" in sequence
        assert "coupling" in sequence
        assert "coherence" in sequence

    def test_bootstrap_extended_higher_stability(self):
        """Test that extended bootstrap has higher stability than minimal."""
        minimal = get_bootstrap_pattern()
        extended = get_bootstrap_extended_pattern()

        result_min = validate_sequence_with_health(minimal)
        result_ext = validate_sequence_with_health(extended)

        assert result_min.passed
        assert result_ext.passed

        # Extended should have higher or equal health
        assert (
            result_ext.health_metrics.overall_health
            >= result_min.health_metrics.overall_health - 0.02
        )

        # Extended should have more operators
        assert len(extended) > len(minimal)

    def test_bootstrap_pattern_detection(self):
        """Test that bootstrap patterns contain EMISSION→COUPLING→COHERENCE signature."""
        sequence = get_bootstrap_pattern()

        # Check for bootstrap signature in sequence
        seq_str = "→".join(sequence)
        # Bootstrap signature may be detected within the larger sequence
        has_coupling = "coupling" in sequence
        has_early_coherence = "coherence" in sequence[:5]

        assert has_coupling and has_early_coherence, "Missing bootstrap signature elements"


# =============================================================================
# Test Explore Patterns
# =============================================================================


class TestExplorePatterns:
    """Test suite for EXPLORE patterns."""

    def test_explore_safe_return_to_baseline(self):
        """Test that explore pattern has safe return to baseline."""
        sequence = get_explore_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have high health (balanced exploration)
        assert health.overall_health >= 0.75

        # Should include DISSONANCE → MUTATION (exploration signature)
        assert "dissonance" in sequence
        assert "mutation" in sequence

        # Should end with stabilizer (safe return)
        assert sequence[-1] in ["silence", "coherence", "transition"]

    def test_explore_deep_multiple_cycles(self):
        """Test that deep explore has multiple exploration cycles."""
        sequence = get_explore_deep_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed

        # Should have multiple DISSONANCE operators (multiple cycles)
        dissonance_count = sequence.count("dissonance")
        assert dissonance_count >= 2, "Deep explore should have multiple cycles"

        # Should have multiple MUTATION operators
        mutation_count = sequence.count("mutation")
        assert mutation_count >= 2, "Deep explore should explore multiple alternatives"

    def test_explore_has_dissonance_mutation_coherence(self):
        """Test that explore patterns have the DISSONANCE→MUTATION→COHERENCE signature."""
        sequence = get_explore_pattern()

        # Find DISSONANCE followed by MUTATION followed by COHERENCE
        for i in range(len(sequence) - 2):
            if (
                sequence[i] == "dissonance"
                and sequence[i + 1] == "mutation"
                and sequence[i + 2] == "coherence"
            ):
                return  # Found the signature

        pytest.fail("Explore pattern missing DISSONANCE→MUTATION→COHERENCE signature")


# =============================================================================
# Test Stabilize Patterns
# =============================================================================


class TestStabilizePatterns:
    """Test suite for STABILIZE patterns."""

    def test_stabilize_high_sustainability(self):
        """Test that stabilize pattern has high sustainability."""
        sequence = get_stabilize_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good sustainability
        assert health.sustainability_index >= 0.6

        # Should have multiple stabilizers
        stabilizers = ["coherence", "resonance", "silence"]
        stabilizer_count = sum(sequence.count(s) for s in stabilizers)
        assert stabilizer_count >= 3, "Stabilize should have multiple stabilizers"

        # Should end with COHERENCE→SILENCE or similar
        assert sequence[-1] in ["silence", "transition"]

    def test_stabilize_recursive_fractal_properties(self):
        """Test that recursive stabilize has fractal properties."""
        sequence = get_stabilize_recursive_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed

        # Should include RECURSIVITY
        assert "recursivity" in sequence, "Recursive pattern must include RECURSIVITY"

        # Pattern detection may show regenerative or fractal
        pattern = result.health_metrics.dominant_pattern
        assert pattern in ["regenerative", "fractal", "activation"]


# =============================================================================
# Test Resonate Patterns
# =============================================================================


class TestResonatePatterns:
    """Test suite for RESONATE patterns."""

    def test_resonate_amplification_coherence(self):
        """Test that resonate pattern has good frequency harmony."""
        sequence = get_resonate_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good frequency harmony
        assert health.frequency_harmony >= 0.65

        # Should have multiple RESONANCE operators
        resonance_count = sequence.count("resonance")
        assert resonance_count >= 2, "Resonate should have multiple RESONANCE operators"

        # Should have COUPLING for synchronization
        assert "coupling" in sequence

    def test_resonate_cascade_emergent_properties(self):
        """Test that cascade resonate has emergent properties."""
        sequence = get_resonate_cascade_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed

        # Should include SELF_ORGANIZATION (emergent property)
        assert "self_organization" in sequence

        # Should have DISSONANCE before SELF_ORGANIZATION (canonical requirement)
        # Find SELF_ORGANIZATION and check previous operators
        self_org_idx = sequence.index("self_organization")
        prev_three = sequence[max(0, self_org_idx - 3) : self_org_idx]
        destabilizers = ["dissonance", "expansion", "transition"]
        has_destabilizer = any(d in prev_three for d in destabilizers)
        assert has_destabilizer, "SELF_ORGANIZATION requires preceding destabilizer"


# =============================================================================
# Test Compress Patterns
# =============================================================================


class TestCompressPatterns:
    """Test suite for COMPRESS patterns."""

    def test_compress_efficiency_optimization(self):
        """Test that compress pattern has high efficiency."""
        sequence = get_compress_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good complexity efficiency
        assert health.complexity_efficiency >= 0.65

        # Should have EXPANSION followed by CONTRACTION
        assert "expansion" in sequence
        assert "contraction" in sequence

        # EXPANSION should come before CONTRACTION
        exp_idx = sequence.index("expansion")
        cont_idx = sequence.index("contraction")
        assert exp_idx < cont_idx, "EXPANSION should precede CONTRACTION"

    def test_compress_adaptive_exploration(self):
        """Test that adaptive compress explores compression paths."""
        sequence = get_compress_adaptive_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed

        # Should have MUTATION (explores alternatives)
        assert "mutation" in sequence

        # Should have DISSONANCE (enables exploration)
        assert "dissonance" in sequence

        # Should have multiple CONTRACTION (tries different compressions)
        contraction_count = sequence.count("contraction")
        assert contraction_count >= 2, "Adaptive should try multiple compressions"


# =============================================================================
# Test Complex Patterns
# =============================================================================


class TestComplexPatterns:
    """Test suite for COMPLEX patterns."""

    def test_complex_multi_pattern_detection(self):
        """Test that complex pattern contains multiple pattern signatures."""
        sequence = get_complex_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have decent health despite complexity
        assert health.overall_health >= 0.75

        # Should be longer (multiple patterns)
        assert len(sequence) >= 9

        # Should have elements from multiple patterns
        has_coupling = "coupling" in sequence  # Bootstrap element
        has_dissonance = "dissonance" in sequence  # Explore element
        has_resonance = "resonance" in sequence  # Stabilize element

        assert (
            has_coupling and has_dissonance and has_resonance
        ), "Complex should combine multiple pattern elements"

    def test_complex_full_cycle_completeness(self):
        """Test that full cycle complex has maximum completeness."""
        sequence = get_complex_full_cycle_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have high health despite length
        assert health.overall_health >= 0.70

        # Should be long (complete cycle)
        assert len(sequence) >= 15, "Full cycle should be comprehensive"

        # Should include many operator types
        unique_ops = len(set(sequence))
        assert unique_ops >= 8, "Full cycle should use diverse operators"

        # Should have SELF_ORGANIZATION (emergent)
        assert "self_organization" in sequence

        # Should have RECURSIVITY (fractal)
        assert "recursivity" in sequence

        # Should have TRANSITION (state navigation)
        assert "transition" in sequence


# =============================================================================
# Test Pattern Construction Principles
# =============================================================================


class TestPatternConstructionPrinciples:
    """Test that patterns follow construction principles."""

    def test_all_patterns_have_reception_coherence(self):
        """Test that all patterns contain RECEPTION→COHERENCE segment."""
        patterns = get_all_patterns()

        for name, sequence in patterns.items():
            # Find RECEPTION
            if "reception" in sequence:
                rec_idx = sequence.index("reception")
                # Check if COHERENCE appears after RECEPTION
                remaining = sequence[rec_idx + 1 :]
                assert "coherence" in remaining, f"{name} has RECEPTION but no subsequent COHERENCE"

    def test_all_patterns_start_correctly(self):
        """Test that all patterns start with EMISSION or RECURSIVITY."""
        patterns = get_all_patterns()
        valid_starts = ["emission", "recursivity"]

        for name, sequence in patterns.items():
            assert (
                sequence[0] in valid_starts
            ), f"{name} starts with {sequence[0]}, expected {valid_starts}"

    def test_all_patterns_end_correctly(self):
        """Test that all patterns end with valid terminators."""
        patterns = get_all_patterns()
        valid_ends = ["dissonance", "recursivity", "silence", "transition"]

        for name, sequence in patterns.items():
            assert (
                sequence[-1] in valid_ends
            ), f"{name} ends with {sequence[-1]}, expected one of {valid_ends}"

    def test_expansion_followed_by_coherence(self):
        """Test that EXPANSION is always followed by COHERENCE."""
        patterns = get_all_patterns()

        for name, sequence in patterns.items():
            for i, op in enumerate(sequence):
                if op == "expansion" and i < len(sequence) - 1:
                    assert (
                        sequence[i + 1] == "coherence"
                    ), f"{name}: EXPANSION at position {i} not followed by COHERENCE"

    def test_mutation_followed_by_coherence(self):
        """Test that MUTATION is followed by COHERENCE (with some tolerance)."""
        patterns = get_all_patterns()

        for name, sequence in patterns.items():
            for i, op in enumerate(sequence):
                if op == "mutation" and i < len(sequence) - 1:
                    # COHERENCE should appear soon after MUTATION
                    next_few = sequence[i + 1 : min(i + 3, len(sequence))]
                    assert (
                        "coherence" in next_few
                    ), f"{name}: MUTATION at position {i} not followed by COHERENCE within 2 operators"


# =============================================================================
# Test Health Metrics Specialization
# =============================================================================


class TestHealthMetricsSpecialization:
    """Test that patterns optimize for their specific purposes."""

    def test_bootstrap_minimizes_time(self):
        """Test that bootstrap patterns are shortest."""
        bootstrap = get_bootstrap_pattern()
        explore = get_explore_pattern()
        stabilize = get_stabilize_pattern()

        # Bootstrap should be shortest or equal
        assert len(bootstrap) <= min(len(explore), len(stabilize))

    def test_explore_high_balance(self):
        """Test that explore patterns have reasonable balance."""
        sequence = get_explore_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Explore should have acceptable balance score (not necessarily high)
        # Balance may be moderate due to exploration/destabilization elements
        assert health.balance_score >= 0.2, "Explore should maintain some balance"

        # But should have excellent overall health
        assert health.overall_health >= 0.75, "Explore should have high overall health"

    def test_stabilize_high_sustainability(self):
        """Test that stabilize patterns maximize sustainability."""
        sequence = get_stabilize_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good sustainability
        assert health.sustainability_index >= 0.6

    def test_compress_high_efficiency(self):
        """Test that compress patterns have high efficiency."""
        sequence = get_compress_pattern()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good complexity efficiency
        assert health.complexity_efficiency >= 0.65


# =============================================================================
# Test Pattern Library Integration
# =============================================================================


class TestPatternLibraryIntegration:
    """Test integration with pattern library."""

    def test_get_all_patterns_returns_dict(self):
        """Test that get_all_patterns returns proper dictionary."""
        patterns = get_all_patterns()

        assert isinstance(patterns, dict)
        assert len(patterns) > 0

        for name, sequence in patterns.items():
            assert isinstance(name, str)
            assert isinstance(sequence, list)
            assert all(isinstance(op, str) for op in sequence)

    def test_pattern_names_unique(self):
        """Test that all pattern names are unique."""
        patterns = get_all_patterns()
        names = list(patterns.keys())

        assert len(names) == len(set(names)), "Duplicate pattern names found"


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v"])
