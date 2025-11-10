#!/usr/bin/env python
"""
Grammar Example 03: Advanced Bifurcation

Demonstrates U4 (Bifurcation Dynamics) constraints.

This example shows:
- U4a: Bifurcation triggers need handlers
- U4b: Transformers need context (recent destabilizer + prior coherence for Mutation)

Pattern: [Generator → Coherence → Destabilizer → Transformer → Stabilizer → Closure]
Health: Advanced (controlled phase transitions)
"""

from tnfr.operators.definitions import (
    Emission,           # Generator (U1a)
    Coherence,          # Stabilizer + Handler
    Dissonance,         # Destabilizer + Bifurcation trigger
    Mutation,           # Transformer + Bifurcation trigger
    SelfOrganization,   # Stabilizer + Handler + Transformer
    Silence,            # Closure (U1b)
)
from tnfr.operators.grammar import validate_grammar

print("="*70)
print(" " * 10 + "Grammar Example 03: Advanced Bifurcation")
print("="*70)
print()

# ============================================================================
# Example 1: Valid Mutation with Full Context
# ============================================================================

print("Example 1: Valid Mutation (with context)")
print("-" * 70)

sequence_1 = [
    Emission(),           # Generator (U1a)
    Coherence(),          # Prior coherence for stable base (U4b requirement for ZHIR)
    Dissonance(),         # Recent destabilizer (U4b) + Bifurcation trigger (U4a)
    Mutation(),           # Transformer (U4b) + Bifurcation trigger (U4a)
    Coherence(),          # Stabilizer (U2) + Handler (U4a)
    Silence()             # Closure (U1b)
]

print("Sequence: Emission → Coherence → Dissonance → Mutation → Coherence → Silence")
print()
print("Why this works:")
print("  ✓ U1a: Emission is generator")
print("  ✓ U4b: Coherence before Mutation (stable base)")
print("  ✓ U4b: Dissonance is recent destabilizer (within ~3 ops)")
print("  ✓ U4a: Mutation (trigger) + Coherence (handler)")
print("  ✓ U2: Dissonance + Mutation (destabilizers) + Coherence (stabilizer)")
print("  ✓ U1b: Silence is closure")
print()
print("Physics:")
print("  • Coherence provides stable base for transformation")
print("  • Dissonance elevates |ΔNFR| above bifurcation threshold")
print("  • Mutation performs phase transition (θ → θ')")
print("  • Final Coherence manages post-bifurcation dynamics")
print()

try:
    is_valid = validate_grammar(sequence_1, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Example 2: Invalid - Mutation without Prior Coherence
# ============================================================================

print("Example 2: Invalid Mutation (no prior coherence)")
print("-" * 70)

sequence_2 = [
    Emission(),
    Dissonance(),         # Recent destabilizer
    Mutation(),           # No prior Coherence!
    Coherence(),
    Silence()
]

print("Sequence: Emission → Dissonance → Mutation → Coherence → Silence")
print()
print("Why this fails:")
print("  ✗ U4b violation: Mutation needs prior Coherence")
print("  ✗ No stable base for transformation")
print("  ✗ Like crystal growth without seed")
print()

try:
    is_valid = validate_grammar(sequence_2, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Example 3: Invalid - Mutation without Recent Destabilizer
# ============================================================================

print("Example 3: Invalid Mutation (no recent destabilizer)")
print("-" * 70)

sequence_3 = [
    Emission(),
    Coherence(),
    Mutation(),           # No recent destabilizer!
    Coherence(),
    Silence()
]

print("Sequence: Emission → Coherence → Mutation → Coherence → Silence")
print()
print("Why this fails:")
print("  ✗ U4b violation: Mutation needs recent destabilizer")
print("  ✗ |ΔNFR| not elevated enough for phase transition")
print("  ✗ Insufficient threshold energy")
print()

try:
    is_valid = validate_grammar(sequence_3, epi_initial=0.0)
    print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
except ValueError as e:
    print(f"Validation: ✗ FAIL - {e}")
print()

# ============================================================================
# Example 4: Valid SelfOrganization with Context
# ============================================================================

print("Example 4: Valid SelfOrganization (with context)")
print("-" * 70)

sequence_4 = [
    Emission(),
    Dissonance(),         # Recent destabilizer (U4b)
    SelfOrganization(),   # Transformer + Handler
    Coherence(),          # Additional stabilizer
    Silence()
]

print("Sequence: Emission → Dissonance → SelfOrganization → Coherence → Silence")
print()
print("Why this works:")
print("  ✓ U4b: Dissonance is recent destabilizer")
print("  ✓ U4a: Dissonance (trigger) + SelfOrganization (handler)")
print("  ✓ U2: Dissonance (destabilizer) + SelfOrganization + Coherence (stabilizers)")
print("  ✓ SelfOrganization creates autopoietic structures")
print()

try:
    is_valid = validate_grammar(sequence_4, epi_initial=0.0)
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
print("1. U4a (Bifurcation Triggers Need Handlers):")
print("   • Triggers: {Dissonance, Mutation}")
print("   • Handlers: {SelfOrganization, Coherence}")
print("   • Uncontrolled bifurcations → chaos")
print()
print("2. U4b (Transformers Need Context):")
print("   • Transformers: {Mutation, SelfOrganization}")
print("   • Need recent destabilizer within ~3 operators")
print("   • Mutation additionally needs prior Coherence")
print("   • Threshold energy requirement")
print()
print("3. Mutation Requirements:")
print("   • Prior Coherence (stable base)")
print("   • Recent destabilizer (threshold energy)")
print("   • Handler after (manage transition)")
print()
print("4. SelfOrganization:")
print("   • Acts as both stabilizer AND transformer")
print("   • Creates autopoietic sub-EPIs")
print("   • Self-limiting through boundaries")
print()
print("5. Bifurcation Pattern:")
print("   • Generator → Coherence → Destabilizer → Transformer → Handler → Closure")
print("   • Controlled phase transitions")
print("   • Enables qualitative state changes")
print()
print("Next: See ../08-QUICK-REFERENCE.md for complete cheat sheet")
print()
print("="*70)
