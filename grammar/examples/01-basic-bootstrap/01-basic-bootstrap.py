#!/usr/bin/env python
"""
Grammar Example 01: Basic Bootstrap Pattern

Demonstrates U1a (Initiation) and U1b (Closure) constraints.

This is the simplest valid TNFR sequence - a basic bootstrap pattern that:
1. Starts with a generator (U1a)
2. Stabilizes the structure
3. Ends with a closure (U1b)

Pattern: [Generator → Stabilizer → Closure]
Health: Excellent (minimal, focused, complete)
"""

from tnfr.operators.definitions import (
    Emission,     # Generator (U1a)
    Coherence,    # Stabilizer
    Silence,      # Closure (U1b)
)
from tnfr.operators.grammar import validate_grammar
import networkx as nx

print("="*70)
print(" " * 15 + "Grammar Example 01: Basic Bootstrap")
print("="*70)
print()

# ============================================================================
# Example 1: Valid Bootstrap from EPI=0
# ============================================================================

print("Example 1: Valid Bootstrap (Starting from EPI=0)")
print("-" * 70)

sequence_1 = [
    Emission(),    # U1a: Generator (creates EPI from vacuum)
    Coherence(),   # Stabilizes the new structure
    Silence()      # U1b: Closure (freezes evolution)
]

print("Sequence: Emission → Coherence → Silence")
print()
print("Why this works:")
print("  ✓ U1a: Emission is a generator (can create from EPI=0)")
print("  ✓ U1b: Silence is a closure (provides terminal endpoint)")
print("  ✓ Coherence stabilizes without needing U2 (no destabilizers)")
print()

try:
    is_valid = validate_grammar(sequence_1, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    print()
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
    print()

# ============================================================================
# Example 2: Invalid - No Generator
# ============================================================================

print("Example 2: Invalid Bootstrap (No generator when EPI=0)")
print("-" * 70)

sequence_2 = [
    Coherence(),   # NOT a generator!
    Silence()
]

print("Sequence: Coherence → Silence")
print()
print("Why this fails:")
print("  ✗ U1a violation: Coherence is NOT a generator")
print("  ✗ Cannot evolve from EPI=0 without generator")
print("  ✗ ∂EPI/∂t undefined at EPI=0")
print()

try:
    is_valid = validate_grammar(sequence_2, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Example 3: Valid - Starting from EPI>0
# ============================================================================

print("Example 3: Valid Bootstrap (Starting from existing EPI)")
print("-" * 70)

sequence_3 = [
    Coherence(),   # No generator needed when EPI>0
    Silence()      # Closure
]

print("Sequence: Coherence → Silence")
print()
print("Why this works:")
print("  ✓ U1a: No generator needed when epi_initial > 0")
print("  ✓ U1b: Silence is a closure")
print("  ✓ Can operate on existing structure without creation")
print()

try:
    is_valid = validate_grammar(sequence_3, epi_initial=1.0)  # EPI > 0!
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Summary
# ============================================================================

print("="*70)
print(" " * 25 + "Summary")
print("="*70)
print()
print("Key Lessons:")
print()
print("1. U1a (Initiation):")
print("   • When EPI=0, MUST start with generator {Emission, Transition, Recursivity}")
print("   • Generators create structure from null/dormant states")
print("   • When EPI>0, no generator needed")
print()
print("2. U1b (Closure):")
print("   • All sequences MUST end with closure {Silence, Transition, Recursivity, Dissonance}")
print("   • Closures provide coherent endpoints")
print("   • Like action potentials need repolarization")
print()
print("3. Bootstrap Pattern:")
print("   • Simplest valid pattern: Generator → Stabilizer → Closure")
print("   • Creates structure, stabilizes it, provides endpoint")
print("   • Foundation for more complex sequences")
print()
print("Next: See 02-intermediate-exploration.py for U2 (Convergence)")
print()
print("="*70)
