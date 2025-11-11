"""Introduction to regenerative cycles in Grammar 2.0.

This example introduces the regenerative cycle concept - self-sustaining
sequences that can maintain coherence over time through strategic
regenerator operators.
"""

from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.cycle_detection import CycleDetector, CycleType
from tnfr.config.operator_names import (
    EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE,
    COUPLING, RESONANCE, SELF_ORGANIZATION, TRANSITION,
    RECURSIVITY, EXPANSION, CONTRACTION
)


def analyze_cycle(sequence: list, regenerator_index: int, title: str):
    """Analyze and display regenerative cycle properties."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    
    print(f"\nSequence: {' → '.join(sequence)}")
    print(f"Regenerator: {sequence[regenerator_index]} (position {regenerator_index})")
    
    # Validate overall sequence
    result = validate_sequence_with_health(sequence)
    
    if not result.passed:
        print(f"\n❌ INVALID SEQUENCE: {result.message}")
        return
    
    # Analyze cycle
    detector = CycleDetector()
    analysis = detector.analyze_potential_cycle(sequence, regenerator_index)
    
    print(f"\n✓ Valid Sequence")
    print(f"Pattern: {result.metadata.get('detected_pattern', 'unknown').upper()}")
    print(f"Overall Health: {result.health_metrics.overall_health:.2f}")
    
    print(f"\n--- Regenerative Cycle Analysis ---")
    print(f"Is Valid Regenerative: {analysis.is_valid_regenerative}")
    
    if analysis.is_valid_regenerative:
        print(f"Cycle Type: {analysis.cycle_type.value.upper()}")
        print(f"Health Score: {analysis.health_score:.2f}")
        print(f"Stabilizers Before: {analysis.stabilizers_before}")
        print(f"Stabilizers After: {analysis.stabilizers_after}")
        print(f"Balance: {'✓ Good' if analysis.stabilizers_before > 0 and analysis.stabilizers_after > 0 else '⚠️ Needs improvement'}")
    else:
        print("\nReasons for invalidity:")
        if len(sequence) < 5:
            print("  • Sequence too short (minimum 5 operators)")
        if sequence[regenerator_index].lower() not in {"transition", "recursivity", "silence"}:
            print(f"  • {sequence[regenerator_index]} is not a valid regenerator")
        if analysis.stabilizers_before == 0:
            print("  • No stabilizers before regenerator")
        if analysis.stabilizers_after == 0:
            print("  • No stabilizers after regenerator")


def example_1_basic_regenerative():
    """Example 1: Basic regenerative cycle with TRANSITION."""
    sequence = [
        COHERENCE, SILENCE,           # Initial stability
        TRANSITION,                   # Regenerator
        EMISSION, RECEPTION,          # Activity
        COHERENCE, SILENCE            # Return to stability
    ]
    
    analyze_cycle(sequence, 2, "Example 1: Basic Regenerative Cycle (TRANSITION)")
    
    print("\n📝 Structure:")
    print("   Phase 1 (Stability): COHERENCE → SILENCE")
    print("   Phase 2 (Regenerate): TRANSITION (regenerator)")
    print("   Phase 3 (Activity): EMISSION → RECEPTION")
    print("   Phase 4 (Stability): COHERENCE → SILENCE")
    
    print("\n💡 This creates a sustainable loop:")
    print("   Stability → Regenerate → Activity → Stability → ...")


def example_2_silence_regenerative():
    """Example 2: Regenerative cycle with SILENCE."""
    sequence = [
        EMISSION, RECEPTION, COHERENCE,  # Activity phase
        SILENCE,                          # Regenerator (pause)
        COUPLING, RESONANCE, COHERENCE    # Propagation phase
    ]
    
    analyze_cycle(sequence, 3, "Example 2: Regenerative Cycle (SILENCE)")
    
    print("\n📝 Structure:")
    print("   Phase 1 (Activity): EMISSION → RECEPTION → COHERENCE")
    print("   Phase 2 (Pause): SILENCE (regenerator)")
    print("   Phase 3 (Propagate): COUPLING → RESONANCE → COHERENCE")
    
    print("\n💡 Silence as regenerator:")
    print("   Allows system to consolidate before next phase")
    print("   Creates natural rhythm: act → pause → propagate")


def example_3_recursivity_regenerative():
    """Example 3: Advanced regenerative with RECURSIVITY."""
    sequence = [
        RECEPTION, EMISSION, COHERENCE,      # Initial activation
        COUPLING, RESONANCE,                 # Network propagation
        RECURSIVITY,                         # Regenerator (fractal)
        EMISSION, COHERENCE, SILENCE         # Stabilization
    ]
    
    analyze_cycle(sequence, 5, "Example 3: Regenerative Cycle (RECURSIVITY)")
    
    print("\n📝 Structure:")
    print("   Phase 1 (Activate): RECEPTION → EMISSION → COHERENCE")
    print("   Phase 2 (Propagate): COUPLING → RESONANCE")
    print("   Phase 3 (Recursivity): RECURSIVITY (regenerator)")
    print("   Phase 4 (Stabilize): EMISSION → COHERENCE → SILENCE")
    
    print("\n💡 Recursivity as regenerator:")
    print("   Enables fractal/hierarchical regeneration")
    print("   Pattern repeats at different scales")


def example_4_transformative_cycle():
    """Example 4: Transformative regenerative cycle."""
    sequence = [
        COHERENCE, SILENCE,                  # Stable start
        TRANSITION,                          # Regenerator
        DISSONANCE, SELF_ORGANIZATION,       # Transformation
        COHERENCE, RESONANCE, SILENCE        # Stabilize and complete
    ]
    
    analyze_cycle(sequence, 2, "Example 4: Transformative Regenerative Cycle")
    
    print("\n📝 Structure:")
    print("   Phase 1 (Stability): COHERENCE → SILENCE")
    print("   Phase 2 (Transition): TRANSITION (regenerator)")
    print("   Phase 3 (Transform): DISSONANCE → SELF_ORGANIZATION")
    print("   Phase 4 (Complete): COHERENCE → RESONANCE → SILENCE")
    
    print("\n💡 Use case:")
    print("   Therapeutic or healing processes")
    print("   Cycles of growth and transformation")
    print("   Sustainable innovation loops")


def example_5_invalid_cycles():
    """Example 5: Common mistakes in regenerative cycles."""
    print("\n" + "=" * 80)
    print("  Example 5: Common Mistakes in Regenerative Cycles")
    print("=" * 80)
    
    print("\n❌ Mistake 1: Too Short")
    short = [COHERENCE, TRANSITION, EMISSION]
    print(f"Sequence: {' → '.join(short)}")
    print("Issue: Only 3 operators (minimum is 5)")
    
    print("\n❌ Mistake 2: Wrong Regenerator")
    wrong_regen = [COHERENCE, EMISSION, COHERENCE, EMISSION, SILENCE]
    print(f"Sequence: {' → '.join(wrong_regen)}")
    print("Issue: EMISSION at position 1 is not a valid regenerator")
    print("Valid regenerators: TRANSITION, RECURSIVITY, SILENCE")
    
    print("\n❌ Mistake 3: No Stabilizers Before")
    no_before = [EMISSION, TRANSITION, COHERENCE, SILENCE, COHERENCE]
    print(f"Sequence: {' → '.join(no_before)}")
    print("Issue: No stabilizers before TRANSITION")
    print("Need: COHERENCE, SILENCE, RESONANCE, or COUPLING before regenerator")
    
    print("\n❌ Mistake 4: No Stabilizers After")
    no_after = [COHERENCE, SILENCE, TRANSITION, EMISSION, DISSONANCE]
    print(f"Sequence: {' → '.join(no_after)}")
    print("Issue: No stabilizers after TRANSITION")
    print("Need: At least one stabilizer after regenerator")


def design_your_cycle():
    """Guide for designing custom regenerative cycles."""
    print("\n" + "=" * 80)
    print("  DESIGNING YOUR REGENERATIVE CYCLE")
    print("=" * 80)
    
    print("\n📋 Requirements Checklist:")
    print("  ✓ Minimum 5 operators")
    print("  ✓ Contains regenerator: TRANSITION, RECURSIVITY, or SILENCE")
    print("  ✓ Stabilizers before regenerator (COHERENCE, SILENCE, RESONANCE, COUPLING)")
    print("  ✓ Stabilizers after regenerator")
    print("  ✓ Overall health score > 0.6")
    
    print("\n🎯 Design Pattern:")
    print("  1. START with stability (COHERENCE, SILENCE)")
    print("  2. REGENERATOR (TRANSITION, RECURSIVITY, or SILENCE)")
    print("  3. ACTIVITY phase (your domain operations)")
    print("  4. RETURN to stability (COHERENCE, SILENCE)")
    
    print("\n💡 Regenerator Selection Guide:")
    print("\n  TRANSITION - For state changes and phase shifts")
    print("    Example: Learning cycles, workflow stages")
    print("    Pattern: Stable → Transition → New State → Stable")
    
    print("\n  RECURSIVITY - For fractal/hierarchical patterns")
    print("    Example: Nested processes, multi-scale systems")
    print("    Pattern: Activate → Recursivity → Sub-patterns → Integrate")
    
    print("\n  SILENCE - For consolidation and rhythm")
    print("    Example: Work-rest cycles, batch processing")
    print("    Pattern: Activity → Silence → Next Activity → Silence")


def real_world_examples():
    """Real-world regenerative cycle applications."""
    print("\n" + "=" * 80)
    print("  REAL-WORLD REGENERATIVE CYCLES")
    print("=" * 80)
    
    print("\n🧬 Example: Cell Regeneration Cycle")
    cell_cycle = [
        COHERENCE, SILENCE,              # G0/G1 phase (stable)
        TRANSITION,                      # Checkpoint
        EMISSION, EXPANSION,             # S phase (DNA replication)
        SELF_ORGANIZATION,               # G2/M (organization)
        COHERENCE, SILENCE               # Return to stable
    ]
    print(f"Sequence: {' → '.join(cell_cycle)}")
    result = validate_sequence_with_health(cell_cycle)
    if result.passed:
        print(f"Health: {result.health_metrics.overall_health:.2f}")
    
    print("\n🎓 Example: Learning Cycle")
    learning_cycle = [
        RECEPTION, EMISSION, COHERENCE,  # Learn new material
        SILENCE,                         # Consolidation (sleep/rest)
        RESONANCE, COUPLING, COHERENCE   # Practice/integration
    ]
    print(f"Sequence: {' → '.join(learning_cycle)}")
    result = validate_sequence_with_health(learning_cycle)
    if result.passed:
        print(f"Health: {result.health_metrics.overall_health:.2f}")
    
    print("\n🏢 Example: Team Sprint Cycle")
    sprint_cycle = [
        COHERENCE, COUPLING,             # Planning/alignment
        TRANSITION,                      # Sprint start
        EMISSION, RECEPTION,             # Work/collaboration
        SELF_ORGANIZATION,               # Adaptation
        COHERENCE, SILENCE               # Review/retrospective
    ]
    print(f"Sequence: {' → '.join(sprint_cycle)}")
    result = validate_sequence_with_health(sprint_cycle)
    if result.passed:
        print(f"Health: {result.health_metrics.overall_health:.2f}")


def main():
    """Run complete regenerative cycles introduction."""
    print("\n" + "█" * 80)
    print("  GRAMMAR 2.0: REGENERATIVE CYCLES INTRODUCTION")
    print("█" * 80)
    
    example_1_basic_regenerative()
    example_2_silence_regenerative()
    example_3_recursivity_regenerative()
    example_4_transformative_cycle()
    example_5_invalid_cycles()
    design_your_cycle()
    real_world_examples()
    
    print("\n" + "=" * 80)
    print("  🎓 Key Concepts:")
    print("=" * 80)
    print("  • Regenerative cycles are self-sustaining sequences")
    print("  • Require: 5+ operators, regenerator, balanced stabilizers")
    print("  • Three regenerator types: TRANSITION, RECURSIVITY, SILENCE")
    print("  • Pattern: Stability → Regenerate → Activity → Stability")
    print("  • Health score must exceed 0.6")
    print("\n  Applications:")
    print("  • Sustainable processes (biological, organizational)")
    print("  • Rhythmic patterns (work-rest, learn-practice)")
    print("  • Self-maintaining systems (homeostasis, adaptation)")
    print("\n  Next Steps:")
    print("  1. Design your first regenerative cycle")
    print("  2. Test with CycleDetector.analyze_potential_cycle()")
    print("  3. Review examples/regenerative_cycles.py for more examples")
    print("=" * 80)


if __name__ == "__main__":
    main()
