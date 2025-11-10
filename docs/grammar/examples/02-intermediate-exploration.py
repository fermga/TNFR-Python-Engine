#!/usr/bin/env python
"""
Grammar Example 02: Intermediate Exploration

Demonstrates U2 (Convergence & Boundedness) constraint.

This example shows controlled destabilization with stabilization:
1. Destabilizers increase |ΔNFR|
2. Must include stabilizers to prevent divergence
3. ∫νf·ΔNFR dt must converge

Pattern: [Generator → Destabilizer → Stabilizer → Closure]
Health: Good (balanced feedback)
"""

from tnfr.operators.definitions import (
    Emission,       # Generator (U1a)
    Dissonance,     # Destabilizer (U2)
    Coherence,      # Stabilizer (U2)
    Silence,        # Closure (U1b)
    Expansion,      # Destabilizer (U2)
)
from tnfr.operators.grammar import validate_grammar

print("="*70)
print(" " * 10 + "Grammar Example 02: Intermediate Exploration")
print("="*70)
print()

# ============================================================================
# Example 1: Valid Exploration with Destabilizer + Stabilizer
# ============================================================================

print("Example 1: Valid Exploration (Dissonance + Coherence)")
print("-" * 70)

sequence_1 = [
    Emission(),     # Generator (U1a)
    Dissonance(),   # Destabilizer - increases |ΔNFR|
    Coherence(),    # Stabilizer - prevents divergence (U2)
    Silence()       # Closure (U1b)
]

print("Sequence: Emission → Dissonance → Coherence → Silence")
print()
print("Why this works:")
print("  ✓ U1a: Emission is generator")
print("  ✓ U2: Dissonance (destabilizer) + Coherence (stabilizer)")
print("  ✓ U1b: Silence is closure")
print()
print("Physics:")
print("  • Dissonance increases |ΔNFR| (positive feedback)")
print("  • Without Coherence: ∫νf·ΔNFR dt → ∞ (divergence)")
print("  • With Coherence: integral converges (bounded evolution)")
print()

try:
    is_valid = validate_grammar(sequence_1, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Example 2: Invalid - Destabilizer without Stabilizer
# ============================================================================

print("Example 2: Invalid Exploration (Destabilizer without stabilizer)")
print("-" * 70)

sequence_2 = [
    Emission(),
    Dissonance(),   # Destabilizer
    Silence()       # No stabilizer!
]

print("Sequence: Emission → Dissonance → Silence")
print()
print("Why this fails:")
print("  ✗ U2 violation: Destabilizer without stabilizer")
print("  ✗ |ΔNFR| grows unbounded (exponential)")
print("  ✗ ∫νf·ΔNFR dt diverges → fragmentation")
print()
print("Physics failure:")
print("  dΔNFR/dt > 0 always (only positive feedback)")
print("  → ΔNFR(t) ~ e^(λt)")
print("  → System loses coherence")
print()

try:
    is_valid = validate_grammar(sequence_2, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Example 3: Multiple Destabilizers
# ============================================================================

print("Example 3: Multiple Destabilizers (still need ONE stabilizer)")
print("-" * 70)

sequence_3 = [
    Emission(),
    Dissonance(),   # Destabilizer 1
    Expansion(),    # Destabilizer 2
    Coherence(),    # Stabilizer (covers both)
    Silence()
]

print("Sequence: Emission → Dissonance → Expansion → Coherence → Silence")
print()
print("Why this works:")
print("  ✓ Multiple destabilizers allowed")
print("  ✓ One stabilizer can handle multiple destabilizers")
print("  ✓ Coherence provides sufficient negative feedback")
print()
print("Physics:")
print("  • Both Dissonance and Expansion increase |ΔNFR|")
print("  • Cumulative positive feedback")
print("  • Coherence stabilizes entire accumulated gradient")
print()

try:
    is_valid = validate_grammar(sequence_3, epi_initial=0.0)
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
print("1. U2 (Convergence & Boundedness):")
print("   • Destabilizers {Dissonance, Mutation, Expansion} increase |ΔNFR|")
print("   • Must include stabilizers {Coherence, SelfOrganization}")
print("   • Without stabilizers: ∫νf·ΔNFR dt → ∞ (divergence)")
print()
print("2. Destabilizers:")
print("   • Dissonance (OZ): Strong positive feedback")
print("   • Mutation (ZHIR): Phase transformation")
print("   • Expansion (VAL): Increases dimensionality")
print()
print("3. Stabilizers:")
print("   • Coherence (IL): Direct negative feedback")
print("   • SelfOrganization (THOL): Emergent self-limiting")
print()
print("4. Exploration Pattern:")
print("   • Generator → Destabilizer(s) → Stabilizer → Closure")
print("   • Controlled instability with recovery")
print("   • Enables adaptation while maintaining coherence")
print()
print("Next: See 03-advanced-bifurcation.py for U4 (Bifurcation)")
print()
print("="*70)
