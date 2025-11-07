"""Before/After comparison examples for Grammar 2.0 migration.

This example demonstrates side-by-side comparisons of sequences
before and after Grammar 2.0 upgrades, showing the improvements
in validation, health metrics, and pattern detection.
"""

from tnfr.operators.grammar import validate_sequence, validate_sequence_with_health
from tnfr.config.operator_names import (
    EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE,
    SELF_ORGANIZATION, COUPLING, RESONANCE, TRANSITION
)


def print_comparison(title: str, before_seq: list, after_seq: list):
    """Print side-by-side comparison of sequences."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    
    # Before (Grammar 1.0 style)
    print("\n🔴 BEFORE (Grammar 1.0):")
    print(f"   Sequence: {' → '.join(before_seq)}")
    
    try:
        before_result = validate_sequence(before_seq)
        print(f"   Valid: {before_result.passed}")
        if not before_result.passed:
            print(f"   Error: {before_result.message}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # After (Grammar 2.0 style)
    print("\n🟢 AFTER (Grammar 2.0):")
    print(f"   Sequence: {' → '.join(after_seq)}")
    
    try:
        after_result = validate_sequence_with_health(after_seq)
        print(f"   Valid: {after_result.passed}")
        
        if after_result.passed:
            health = after_result.health_metrics
            pattern = after_result.metadata.get('detected_pattern', 'unknown')
            
            print(f"   Pattern: {pattern.upper()}")
            print(f"   Overall Health: {health.overall_health:.2f}")
            print(f"   Coherence Index: {health.coherence_index:.2f}")
            print(f"   Balance Score: {health.balance_score:.2f}")
            print(f"   Frequency Harmony: {health.frequency_harmony:.2f}")
            
            if health.recommendations:
                print(f"   Recommendations:")
                for rec in health.recommendations:
                    print(f"     • {rec}")
    except Exception as e:
        print(f"   Error: {e}")


def example_1_thol_fix():
    """Example 1: Fixing SELF_ORGANIZATION validation."""
    before = [EMISSION, RECEPTION, SELF_ORGANIZATION, COHERENCE]
    after = [EMISSION, DISSONANCE, SELF_ORGANIZATION, COHERENCE]
    
    print_comparison(
        "Example 1: SELF_ORGANIZATION Validation Fix",
        before,
        after
    )
    
    print("\n📝 What changed:")
    print("   • Added DISSONANCE before SELF_ORGANIZATION")
    print("   • THOL now has required destabilizer within 3-operator window")
    print("   • Sequence now passes validation")


def example_2_frequency_smoothing():
    """Example 2: Smoothing frequency transitions."""
    before = [SILENCE, EMISSION, COHERENCE]
    after = [SILENCE, TRANSITION, EMISSION, COHERENCE]
    
    print_comparison(
        "Example 2: Frequency Transition Smoothing",
        before,
        after
    )
    
    print("\n📝 What changed:")
    print("   • Added TRANSITION between SILENCE and EMISSION")
    print("   • Smooth frequency progression: Zero → Medium → High → Medium")
    print("   • Improved frequency harmony score")


def example_3_balance_improvement():
    """Example 3: Balancing stabilizers and destabilizers."""
    before = [EMISSION, DISSONANCE, DISSONANCE]
    after = [EMISSION, DISSONANCE, COHERENCE, SILENCE]
    
    print_comparison(
        "Example 3: Operator Balance Improvement",
        before,
        after
    )
    
    print("\n📝 What changed:")
    print("   • Removed duplicate DISSONANCE")
    print("   • Added COHERENCE to stabilize after destabilizer")
    print("   • Added SILENCE terminator")
    print("   • Better stabilizer/destabilizer balance")


def example_4_health_optimization():
    """Example 4: Full health optimization."""
    before = [EMISSION, RECEPTION]
    after = [EMISSION, RECEPTION, COHERENCE, COUPLING, RESONANCE, SILENCE]
    
    print_comparison(
        "Example 4: Health Optimization",
        before,
        after
    )
    
    print("\n📝 What changed:")
    print("   • Extended minimal sequence with structural operators")
    print("   • Added COHERENCE for stabilization")
    print("   • Added COUPLING → RESONANCE for network propagation")
    print("   • Added SILENCE terminator")
    print("   • Transformed LINEAR pattern to RESONATE pattern")


def example_5_pattern_upgrade():
    """Example 5: Pattern-specific upgrade."""
    before = [RECEPTION, EMISSION, COHERENCE]
    after = [
        RECEPTION, EMISSION, COHERENCE,
        DISSONANCE, SELF_ORGANIZATION, COHERENCE
    ]
    
    print_comparison(
        "Example 5: Pattern Upgrade (MINIMAL → THERAPEUTIC)",
        before,
        after
    )
    
    print("\n📝 What changed:")
    print("   • Extended basic activation with transformation phase")
    print("   • Added DISSONANCE → SELF_ORGANIZATION → COHERENCE")
    print("   • Pattern upgraded from MINIMAL to THERAPEUTIC")
    print("   • Demonstrates domain-specific pattern construction")


def show_adoption_strategies():
    """Show different adoption strategies."""
    print("\n" + "=" * 80)
    print("  ADOPTION STRATEGIES")
    print("=" * 80)
    
    print("\n🔵 Strategy 1: Conservative (No Changes)")
    print("   Keep using validate_sequence() - all existing code works")
    sequence = [EMISSION, COHERENCE]
    result = validate_sequence(sequence)
    print(f"   {' → '.join(sequence)}: {result.passed}")
    
    print("\n🟡 Strategy 2: Progressive (Opt-in Health)")
    print("   Switch to validate_sequence_with_health() for metrics")
    result = validate_sequence_with_health(sequence)
    print(f"   {' → '.join(sequence)}: Health = {result.health_metrics.overall_health:.2f}")
    
    print("\n🟢 Strategy 3: Advanced (Full Optimization)")
    print("   Use upgrader tools and advanced patterns")
    from tools.migration.sequence_upgrader import SequenceUpgrader
    
    upgrader = SequenceUpgrader(target_health=0.80)
    upgrade_result = upgrader.upgrade_sequence(sequence)
    print(f"   Original: {' → '.join(upgrade_result.original_sequence)}")
    print(f"   Upgraded: {' → '.join(upgrade_result.upgraded_sequence)}")
    if upgrade_result.upgraded_health:
        print(f"   Health: {upgrade_result.original_health:.2f} → {upgrade_result.upgraded_health:.2f}")


def main():
    """Run all comparison examples."""
    print("\n" + "█" * 80)
    print("  GRAMMAR 2.0 MIGRATION: BEFORE/AFTER COMPARISON")
    print("█" * 80)
    
    example_1_thol_fix()
    example_2_frequency_smoothing()
    example_3_balance_improvement()
    example_4_health_optimization()
    example_5_pattern_upgrade()
    show_adoption_strategies()
    
    print("\n" + "=" * 80)
    print("  📚 Next Steps:")
    print("=" * 80)
    print("  1. Review docs/MIGRATION_GUIDE_2.0.md for complete guide")
    print("  2. Run migration_checker.py on your codebase")
    print("  3. Try health_optimization_tutorial.py for guided optimization")
    print("  4. Explore pattern_upgrade_examples.py for pattern construction")
    print("=" * 80)


if __name__ == "__main__":
    main()
