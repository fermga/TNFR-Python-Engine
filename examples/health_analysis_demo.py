"""Demonstration of sequence health analysis with TNFR structural metrics.

This example shows how to use the SequenceHealthAnalyzer to evaluate
and compare different operator sequences, providing quantitative insights
into sequence quality, coherence, balance, and sustainability.
"""

from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.config.operator_names import (
    EMISSION,
    RECEPTION,
    COHERENCE,
    DISSONANCE,
    COUPLING,
    RESONANCE,
    SILENCE,
    EXPANSION,
    CONTRACTION,
    SELF_ORGANIZATION,
    MUTATION,
    TRANSITION,
    RECURSIVITY,
)


def print_health_report(sequence, health):
    """Print a formatted health report for a sequence."""
    print(f"\n{'=' * 70}")
    print(f"Sequence: {' → '.join(sequence)}")
    print(f"{'=' * 70}")
    print(f"Pattern: {health.dominant_pattern.upper()}")
    print(f"Length: {health.sequence_length} operators")
    print(f"\n--- Structural Health Metrics ---")
    print(f"Overall Health:         {health.overall_health:.2f}")
    print(f"Coherence Index:        {health.coherence_index:.2f}")
    print(f"Balance Score:          {health.balance_score:.2f}")
    print(f"Sustainability:         {health.sustainability_index:.2f}")
    print(f"Complexity Efficiency:  {health.complexity_efficiency:.2f}")
    print(f"Frequency Harmony:      {health.frequency_harmony:.2f}")
    print(f"Pattern Completeness:   {health.pattern_completeness:.2f}")
    print(f"Transition Smoothness:  {health.transition_smoothness:.2f}")
    
    if health.recommendations:
        print(f"\n--- Recommendations ---")
        for i, rec in enumerate(health.recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print(f"\n✓ No structural issues detected - excellent health!")
    
    # Health interpretation
    if health.overall_health >= 0.85:
        print(f"\n🌟 EXCELLENT: Highly coherent and sustainable structure")
    elif health.overall_health >= 0.70:
        print(f"\n✓ GOOD: Solid structural quality with minor improvements possible")
    elif health.overall_health >= 0.50:
        print(f"\n⚠ FAIR: Moderate quality, consider addressing recommendations")
    else:
        print(f"\n⚠️ POOR: Significant structural issues detected")


def example_1_basic_activation():
    """Example 1: Basic activation sequence - simple and effective."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Activation Sequence")
    print("=" * 70)
    
    sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
    
    analyzer = SequenceHealthAnalyzer()
    health = analyzer.analyze_health(sequence)
    
    print_health_report(sequence, health)


def example_2_therapeutic_sequence():
    """Example 2: Therapeutic sequence - includes transformation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Therapeutic Sequence (with transformation)")
    print("=" * 70)
    
    sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, RESONANCE, SILENCE]
    
    analyzer = SequenceHealthAnalyzer()
    health = analyzer.analyze_health(sequence)
    
    print_health_report(sequence, health)


def example_3_suboptimal_sequence():
    """Example 3: Suboptimal sequence - imbalanced and unstable."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Suboptimal Sequence (imbalanced)")
    print("=" * 70)
    
    # Many destabilizers, few stabilizers, doesn't end with stabilizer
    sequence = [EMISSION, DISSONANCE, DISSONANCE, EXPANSION, MUTATION]
    
    analyzer = SequenceHealthAnalyzer()
    health = analyzer.analyze_health(sequence)
    
    print_health_report(sequence, health)


def example_4_regenerative_cycle():
    """Example 4: Regenerative cycle with transition."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Regenerative Cycle")
    print("=" * 70)
    
    sequence = [EMISSION, RECEPTION, COHERENCE, COUPLING, RESONANCE, TRANSITION]
    
    analyzer = SequenceHealthAnalyzer()
    health = analyzer.analyze_health(sequence)
    
    print_health_report(sequence, health)


def example_5_comparison():
    """Example 5: Comparing alternative sequences."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comparing Alternative Sequences")
    print("=" * 70)
    
    option_a = [EMISSION, RECEPTION, COHERENCE, SILENCE]  # Simple
    option_b = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]  # With resonance
    option_c = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]  # Transformative
    
    analyzer = SequenceHealthAnalyzer()
    
    health_a = analyzer.analyze_health(option_a)
    health_b = analyzer.analyze_health(option_b)
    health_c = analyzer.analyze_health(option_c)
    
    print("\n--- Option A: Simple Activation ---")
    print(f"Overall Health: {health_a.overall_health:.2f}")
    print(f"Pattern: {health_a.dominant_pattern}")
    
    print("\n--- Option B: With Resonance ---")
    print(f"Overall Health: {health_b.overall_health:.2f}")
    print(f"Pattern: {health_b.dominant_pattern}")
    
    print("\n--- Option C: Transformative ---")
    print(f"Overall Health: {health_c.overall_health:.2f}")
    print(f"Pattern: {health_c.dominant_pattern}")
    
    # Determine best option
    scores = [
        ("Option A (Simple)", health_a.overall_health),
        ("Option B (Resonance)", health_b.overall_health),
        ("Option C (Transformative)", health_c.overall_health),
    ]
    best = max(scores, key=lambda x: x[1])
    
    print(f"\n🏆 Best Option: {best[0]} with health score {best[1]:.2f}")


def example_6_validation_integration():
    """Example 6: Using validate_sequence_with_health API."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Integration with Validation API")
    print("=" * 70)
    
    # Valid sequence
    valid_sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
    
    result = validate_sequence_with_health(valid_sequence)
    
    if result.passed:
        print(f"\n✓ Sequence is VALID")
        print(f"  Overall Health: {result.health_metrics.overall_health:.2f}")
        print(f"  Pattern: {result.health_metrics.dominant_pattern}")
        print(f"  Coherence: {result.health_metrics.coherence_index:.2f}")
    else:
        print(f"\n✗ Sequence is INVALID: {result.message}")
    
    # Invalid sequence (for comparison)
    invalid_sequence = [RECEPTION, COHERENCE, SILENCE]  # Doesn't start with valid operator
    
    result_invalid = validate_sequence_with_health(invalid_sequence)
    
    print(f"\n--- Invalid Sequence ---")
    print(f"Sequence: {' → '.join(invalid_sequence)}")
    if not result_invalid.passed:
        print(f"✗ Invalid: {result_invalid.message}")
        print(f"  Health metrics: {'Not computed' if result_invalid.health_metrics is None else 'Available'}")


def example_7_long_sequence_analysis():
    """Example 7: Analyzing a complex long sequence."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Complex Long Sequence")
    print("=" * 70)
    
    # Comprehensive sequence using many operators
    sequence = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        COUPLING,
        RESONANCE,
        EXPANSION,
        CONTRACTION,
        SELF_ORGANIZATION,
        MUTATION,
        TRANSITION,
    ]
    
    analyzer = SequenceHealthAnalyzer()
    health = analyzer.analyze_health(sequence)
    
    print_health_report(sequence, health)
    
    # Additional analysis
    print(f"\n--- Detailed Analysis ---")
    print(f"Sequence uses {len(set(sequence))} unique operators")
    print(f"Structural value vs complexity: {health.complexity_efficiency:.2%}")
    print(f"Sustainability forecast: {'Sustainable' if health.sustainability_index > 0.7 else 'May degrade'}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TNFR Sequence Health Analysis Demonstration")
    print("Evaluating structural quality of operator sequences")
    print("=" * 70)
    
    example_1_basic_activation()
    example_2_therapeutic_sequence()
    example_3_suboptimal_sequence()
    example_4_regenerative_cycle()
    example_5_comparison()
    example_6_validation_integration()
    example_7_long_sequence_analysis()
    
    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Health metrics provide quantitative assessment of sequence quality")
    print("- Use validate_sequence_with_health() for integrated validation + health analysis")
    print("- Compare alternatives to select the most coherent structural pattern")
    print("- Recommendations guide optimization of suboptimal sequences")
    print("=" * 70)


if __name__ == "__main__":
    main()
