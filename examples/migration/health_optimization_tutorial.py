"""Interactive tutorial for health-based sequence optimization.

This guided tutorial teaches you how to use Grammar 2.0 health metrics
to optimize operator sequences for better coherence, balance, and sustainability.
"""

from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.config.operator_names import (
    EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE,
    COUPLING, RESONANCE, SELF_ORGANIZATION, TRANSITION, MUTATION
)
from tools.migration.sequence_upgrader import SequenceUpgrader


def print_health_breakdown(sequence: list, title: str = "Health Analysis"):
    """Print detailed health breakdown for a sequence."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"\nSequence: {' → '.join(sequence)}")
    
    result = validate_sequence_with_health(sequence)
    
    if not result.passed:
        print(f"\n❌ INVALID: {result.message}")
        return
    
    health = result.health_metrics
    pattern = result.metadata.get('detected_pattern', 'unknown')
    
    print(f"\n✓ Valid | Pattern: {pattern.upper()}")
    print("\n--- Health Metrics (0.0 = poor, 1.0 = excellent) ---")
    print(f"  Overall Health:         {health.overall_health:.2f}  {get_health_emoji(health.overall_health)}")
    print(f"  Coherence Index:        {health.coherence_index:.2f}  {get_health_emoji(health.coherence_index)}")
    print(f"  Balance Score:          {health.balance_score:.2f}  {get_health_emoji(health.balance_score)}")
    print(f"  Sustainability:         {health.sustainability_index:.2f}  {get_health_emoji(health.sustainability_index)}")
    print(f"  Complexity Efficiency:  {health.complexity_efficiency:.2f}  {get_health_emoji(health.complexity_efficiency)}")
    print(f"  Frequency Harmony:      {health.frequency_harmony:.2f}  {get_health_emoji(health.frequency_harmony)}")
    print(f"  Pattern Completeness:   {health.pattern_completeness:.2f}  {get_health_emoji(health.pattern_completeness)}")
    print(f"  Transition Smoothness:  {health.transition_smoothness:.2f}  {get_health_emoji(health.transition_smoothness)}")
    
    if health.recommendations:
        print("\n--- Recommendations for Improvement ---")
        for i, rec in enumerate(health.recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✨ Excellent! No improvements needed.")
    
    return health


def get_health_emoji(score: float) -> str:
    """Get emoji for health score."""
    if score >= 0.85:
        return "🌟"
    elif score >= 0.70:
        return "✓"
    elif score >= 0.50:
        return "⚠️"
    else:
        return "❌"


def lesson_1_understanding_metrics():
    """Lesson 1: Understanding what each metric means."""
    print("\n" + "█" * 80)
    print("  LESSON 1: Understanding Health Metrics")
    print("█" * 80)
    
    print("\n📚 Health metrics provide quantitative assessment of sequence quality:")
    print("\n1. COHERENCE INDEX: Quality of transitions between operators")
    print("   - Measures structural flow and compatibility")
    print("   - Higher = smoother, more natural transitions")
    
    print("\n2. BALANCE SCORE: Equilibrium of stabilizers vs destabilizers")
    print("   - Stabilizers: COHERENCE, SILENCE, RESONANCE, COUPLING")
    print("   - Destabilizers: DISSONANCE, MUTATION, CONTRACTION")
    print("   - Balanced sequences are more sustainable")
    
    print("\n3. SUSTAINABILITY INDEX: Capacity for long-term maintenance")
    print("   - Considers termination, balance, and completeness")
    print("   - Higher = sequence can maintain coherence over time")
    
    print("\n4. FREQUENCY HARMONY: Smoothness of structural frequency transitions")
    print("   - High: Smooth frequency progression (e.g., Medium → High → Medium)")
    print("   - Low: Jarring jumps (e.g., Zero → High)")
    
    # Demonstrate with examples
    print("\n--- Example: Well-Balanced Sequence ---")
    balanced = [EMISSION, RECEPTION, COHERENCE, SILENCE]
    print_health_breakdown(balanced, "Well-Balanced Activation")
    
    print("\n--- Example: Unbalanced Sequence ---")
    unbalanced = [EMISSION, DISSONANCE, MUTATION]
    print_health_breakdown(unbalanced, "Unbalanced (Too Many Destabilizers)")


def lesson_2_iterative_optimization():
    """Lesson 2: Iteratively improving a sequence."""
    print("\n" + "█" * 80)
    print("  LESSON 2: Iterative Optimization Process")
    print("█" * 80)
    
    print("\n📚 Learn to improve sequences step-by-step based on recommendations.")
    
    # Start with a problematic sequence
    print("\n--- Step 1: Initial Sequence ---")
    sequence = [EMISSION, DISSONANCE]
    health = print_health_breakdown(sequence, "Initial Sequence (Problematic)")
    
    print("\n--- Step 2: Apply First Recommendation ---")
    print("  Recommendation: Add stabilizer after destabilizer")
    sequence_v2 = [EMISSION, DISSONANCE, COHERENCE]
    health_v2 = print_health_breakdown(sequence_v2, "After Adding COHERENCE")
    
    print(f"\n  Improvement: {health.overall_health:.2f} → {health_v2.overall_health:.2f} (+{health_v2.overall_health - health.overall_health:.2f})")
    
    print("\n--- Step 3: Add Proper Terminator ---")
    print("  Recommendation: End with terminator for sustainability")
    sequence_v3 = [EMISSION, DISSONANCE, COHERENCE, SILENCE]
    health_v3 = print_health_breakdown(sequence_v3, "After Adding SILENCE")
    
    print(f"\n  Improvement: {health_v2.overall_health:.2f} → {health_v3.overall_health:.2f} (+{health_v3.overall_health - health_v2.overall_health:.2f})")
    
    print("\n✨ Final Result: Transformed problematic sequence into healthy one!")
    print(f"   Original:  {' → '.join([EMISSION, DISSONANCE])}")
    print(f"   Optimized: {' → '.join(sequence_v3)}")
    print(f"   Health: {health.overall_health:.2f} → {health_v3.overall_health:.2f}")


def lesson_3_automatic_upgrader():
    """Lesson 3: Using the automatic sequence upgrader."""
    print("\n" + "█" * 80)
    print("  LESSON 3: Automatic Sequence Upgrader")
    print("█" * 80)
    
    print("\n📚 The SequenceUpgrader can automatically optimize sequences.")
    
    # Example 1: Simple upgrade
    print("\n--- Example 1: Basic Upgrade ---")
    original = [EMISSION, RECEPTION]
    
    upgrader = SequenceUpgrader(target_health=0.75)
    result = upgrader.upgrade_sequence(original)
    
    print(f"\nOriginal:  {' → '.join(result.original_sequence)}")
    print(f"Upgraded:  {' → '.join(result.upgraded_sequence)}")
    
    if result.original_health and result.upgraded_health:
        print(f"Health:    {result.original_health:.2f} → {result.upgraded_health:.2f}")
    
    if result.improvements:
        print("\nImprovements applied:")
        for imp in result.improvements:
            print(f"  • {imp}")
    
    # Example 2: THOL fix
    print("\n--- Example 2: THOL Validation Fix ---")
    problematic = [EMISSION, RECEPTION, SELF_ORGANIZATION]
    
    result = upgrader.upgrade_sequence(problematic)
    
    print(f"\nOriginal:  {' → '.join(result.original_sequence)}")
    print(f"Upgraded:  {' → '.join(result.upgraded_sequence)}")
    
    if result.improvements:
        print("\nImprovements applied:")
        for imp in result.improvements:
            print(f"  • {imp}")
    
    # Example 3: Target health
    print("\n--- Example 3: Optimize to Target Health ---")
    basic = [COHERENCE, EMISSION]
    
    upgrader_high = SequenceUpgrader(target_health=0.80)
    result = upgrader_high.improve_to_target(basic, max_iterations=3)
    
    print(f"\nOriginal:  {' → '.join(result.original_sequence)}")
    print(f"Upgraded:  {' → '.join(result.upgraded_sequence)}")
    
    if result.original_health and result.upgraded_health:
        print(f"Health:    {result.original_health:.2f} → {result.upgraded_health:.2f}")
        print(f"Target:    0.80 {'✓' if result.upgraded_health >= 0.80 else '(not reached)'}")


def lesson_4_pattern_aware_optimization():
    """Lesson 4: Optimizing for specific patterns."""
    print("\n" + "█" * 80)
    print("  LESSON 4: Pattern-Aware Optimization")
    print("█" * 80)
    
    print("\n📚 Different patterns have different health profiles.")
    
    patterns = [
        ("MINIMAL", [EMISSION, COHERENCE]),
        ("LINEAR", [EMISSION, RECEPTION, COHERENCE]),
        ("HIERARCHICAL", [EMISSION, COUPLING, COHERENCE, RESONANCE]),
        ("THERAPEUTIC", [RECEPTION, EMISSION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE]),
    ]
    
    print("\n--- Pattern Health Comparison ---")
    for pattern_name, sequence in patterns:
        result = validate_sequence_with_health(sequence)
        if result.passed:
            health = result.health_metrics
            detected = result.metadata.get('detected_pattern', 'unknown')
            print(f"\n{pattern_name:15} (detected: {detected:12}): Health = {health.overall_health:.2f}")
            print(f"  Sequence: {' → '.join(sequence)}")
            print(f"  Balance: {health.balance_score:.2f} | Frequency: {health.frequency_harmony:.2f}")


def lesson_5_custom_optimization():
    """Lesson 5: Custom optimization strategies."""
    print("\n" + "█" * 80)
    print("  LESSON 5: Custom Optimization Strategies")
    print("█" * 80)
    
    print("\n📚 Tailor optimization to your specific needs.")
    
    print("\n--- Strategy 1: Maximize Coherence ---")
    print("  Focus: High coherence_index and transition_smoothness")
    sequence = [EMISSION, RECEPTION, COHERENCE, COUPLING, RESONANCE, SILENCE]
    health = print_health_breakdown(sequence, "Coherence-Optimized")
    
    print("\n--- Strategy 2: Balanced Transformation ---")
    print("  Focus: Balance destabilizers with stabilizers")
    sequence = [RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
    health = print_health_breakdown(sequence, "Balance-Optimized")
    
    print("\n--- Strategy 3: Sustainable Growth ---")
    print("  Focus: High sustainability_index and pattern_completeness")
    sequence = [EMISSION, RECEPTION, COHERENCE, COUPLING, RESONANCE, SILENCE]
    health = print_health_breakdown(sequence, "Sustainability-Optimized")


def main():
    """Run complete health optimization tutorial."""
    print("\n" + "█" * 80)
    print("  GRAMMAR 2.0: HEALTH OPTIMIZATION TUTORIAL")
    print("█" * 80)
    print("\n  Learn to use health metrics for sequence optimization")
    
    lesson_1_understanding_metrics()
    lesson_2_iterative_optimization()
    lesson_3_automatic_upgrader()
    lesson_4_pattern_aware_optimization()
    lesson_5_custom_optimization()
    
    print("\n" + "=" * 80)
    print("  🎓 Tutorial Complete!")
    print("=" * 80)
    print("\n  Key Takeaways:")
    print("  • Health metrics provide objective sequence quality assessment")
    print("  • Iterative improvement based on recommendations")
    print("  • Automatic upgrader for common optimizations")
    print("  • Different patterns optimize for different goals")
    print("\n  Next Steps:")
    print("  1. Try optimizing your own sequences")
    print("  2. Explore pattern_upgrade_examples.py for pattern construction")
    print("  3. Review docs/HEALTH_METRICS_GUIDE.md for deep dive")
    print("=" * 80)


if __name__ == "__main__":
    main()
