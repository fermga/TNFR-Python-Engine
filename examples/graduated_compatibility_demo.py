"""Example demonstrating graduated compatibility levels in TNFR.

This example shows how the graduated compatibility system works with 4 levels:
- EXCELLENT: Optimal structural progression
- GOOD: Acceptable structural progression
- CAUTION: Contextually dependent, requires validation (generates warnings)
- AVOID: Incompatible, violates structural coherence (raises errors)
"""

from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RESONANCE,
    SILENCE,
)
from tnfr.operators.grammar import validate_sequence
from tnfr.validation import CompatibilityLevel, get_compatibility_level


def demonstrate_graduated_compatibility():
    """Demonstrate the 4 graduated compatibility levels."""
    
    print("=" * 70)
    print("GRADUATED COMPATIBILITY DEMONSTRATION")
    print("=" * 70)
    print()
    
    # EXCELLENT transitions
    print("1. EXCELLENT Transitions (Optimal Structural Progression)")
    print("-" * 70)
    
    excellent_pairs = [
        (EMISSION, COHERENCE, "Emission → Coherence: initiation → stabilization"),
        (RECEPTION, COHERENCE, "Reception → Coherence: anchoring → stabilization"),
        (DISSONANCE, MUTATION, "Dissonance → Mutation: tension → transformation"),
    ]
    
    for prev, next_op, description in excellent_pairs:
        level = get_compatibility_level(prev, next_op)
        print(f"  {description}")
        print(f"  Level: {level.value.upper()}")
        print()
    
    # GOOD transitions
    print("2. GOOD Transitions (Acceptable Structural Progression)")
    print("-" * 70)
    
    good_pairs = [
        (EMISSION, RESONANCE, "Emission → Resonance: initiation → amplification"),
        (COHERENCE, SILENCE, "Coherence → Silence: stabilization → pause"),
        (RESONANCE, EMISSION, "Resonance → Emission: amplification → re-initiation"),
    ]
    
    for prev, next_op, description in good_pairs:
        level = get_compatibility_level(prev, next_op)
        print(f"  {description}")
        print(f"  Level: {level.value.upper()}")
        print()
    
    # CAUTION transitions
    print("3. CAUTION Transitions (Contextually Dependent)")
    print("-" * 70)
    print("  These transitions are allowed but generate warnings for careful review.")
    print()
    
    caution_pairs = [
        (EMISSION, DISSONANCE, "Emission → Dissonance: direct tension after initiation"),
        (COHERENCE, MUTATION, "Coherence → Mutation: transformation after stabilization"),
        (DISSONANCE, DISSONANCE, "Dissonance → Dissonance: repeated tension"),
    ]
    
    for prev, next_op, description in caution_pairs:
        level = get_compatibility_level(prev, next_op)
        print(f"  {description}")
        print(f"  Level: {level.value.upper()}")
        print()
    
    # AVOID transitions
    print("4. AVOID Transitions (Incompatible)")
    print("-" * 70)
    print("  These transitions violate structural coherence and raise errors.")
    print()
    
    avoid_pairs = [
        (SILENCE, DISSONANCE, "Silence → Dissonance: pause → tension (contradictory)"),
        (COHERENCE, EMISSION, "Coherence → Emission: cannot re-initiate after stabilizing"),
    ]
    
    for prev, next_op, description in avoid_pairs:
        level = get_compatibility_level(prev, next_op)
        print(f"  {description}")
        print(f"  Level: {level.value.upper()}")
        print()


def demonstrate_sequence_validation():
    """Demonstrate how graduated compatibility affects sequence validation."""
    
    print("=" * 70)
    print("SEQUENCE VALIDATION WITH GRADUATED COMPATIBILITY")
    print("=" * 70)
    print()
    
    # Example 1: Excellent/Good sequence (passes without warnings)
    print("Example 1: High-quality sequence (EXCELLENT + GOOD transitions)")
    print("-" * 70)
    sequence1 = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
    print(f"Sequence: {' → '.join(sequence1)}")
    
    result1 = validate_sequence(sequence1)
    if result1.passed:
        print("✓ Validation: PASSED")
        print("  No warnings - all transitions are excellent or good")
    else:
        print(f"✗ Validation: FAILED - {result1.message}")
    print()
    
    # Example 2: Sequence with CAUTION transition (passes with warnings)
    print("Example 2: Sequence with caution transition")
    print("-" * 70)
    sequence2 = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE]
    print(f"Sequence: {' → '.join(sequence2)}")
    print("Note: COHERENCE → DISSONANCE is a CAUTION-level transition")
    
    result2 = validate_sequence(sequence2)
    if result2.passed:
        print("✓ Validation: PASSED")
        print("  ⚠️  Warning generated for CAUTION transition (check logs)")
    else:
        print(f"✗ Validation: FAILED - {result2.message}")
    print()
    
    # Example 3: Sequence with AVOID transition (fails)
    print("Example 3: Invalid sequence with avoid transition")
    print("-" * 70)
    sequence3 = [EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE]
    print(f"Sequence: {' → '.join(sequence3)}")
    print("Note: SILENCE → DISSONANCE is an AVOID-level transition")
    
    result3 = validate_sequence(sequence3)
    if result3.passed:
        print("✓ Validation: PASSED")
    else:
        print(f"✗ Validation: FAILED - {result3.message}")
        print(f"  Error at position {result3.error.index}: {result3.error.token}")
    print()


def show_operator_compatibility_summary():
    """Show compatibility levels for all transitions from EMISSION."""
    
    print("=" * 70)
    print("COMPATIBILITY MATRIX FOR EMISSION")
    print("=" * 70)
    print()
    
    from tnfr.validation import GRADUATED_COMPATIBILITY
    
    emission_levels = GRADUATED_COMPATIBILITY[EMISSION]
    
    for level_name in ["excellent", "good", "caution", "avoid"]:
        operators = emission_levels[level_name]
        if operators:
            print(f"{level_name.upper()}:")
            for op in sorted(operators):
                print(f"  - EMISSION → {op.upper()}")
            print()


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_graduated_compatibility()
    print("\n")
    demonstrate_sequence_validation()
    print("\n")
    show_operator_compatibility_summary()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The graduated compatibility system provides:
  
  1. ✓ More precise validation than binary compatible/incompatible
  2. ⚠️  Contextual warnings for risky but allowed transitions
  3. ✗ Clear errors for truly incompatible transitions
  4. 📊 Better sequence quality metrics based on transition levels
  
This enables developers to write more robust TNFR sequences while
maintaining the flexibility needed for complex structural patterns.
""")
