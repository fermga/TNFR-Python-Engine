"""Example demonstrating U5: Multi-Scale Coherence grammar rule.

This example shows how deep hierarchical structures (REMESH with depth>1)
require scale stabilizers to preserve coherence across levels.

Physical Basis
--------------
From coherence conservation:
    C_parent ≥ α · Σ C_child_i
    
Where α ∈ [0.1, 0.4] captures coupling efficiency losses in hierarchy.
"""

from tnfr.operators.definitions import (
    Coherence,
    Emission,
    Recursivity,
    SelfOrganization,
    Silence,
)
from tnfr.operators.unified_grammar import UnifiedGrammarValidator

print("=" * 70)
print("U5: Multi-Scale Coherence - Example")
print("=" * 70)
print()

# Example 1: Shallow recursion (depth=1) - No stabilizer needed
print("Example 1: Shallow recursion (depth=1)")
print("-" * 70)
sequence_shallow = [
    Emission(),
    Recursivity(depth=1),  # Shallow recursion, single level
    Silence(),
]

valid, messages = UnifiedGrammarValidator.validate(sequence_shallow, epi_initial=0.0)
print(f"Sequence: AL → REMESH(depth=1) → SHA")
print(f"Valid: {valid}")
print(f"U5 Status: {[m for m in messages if m.startswith('U5:')][0]}")
print()
print("✓ Shallow recursion passes - no multi-scale hierarchy")
print()

# Example 2: Deep recursion (depth=3) without stabilizer - FAILS
print("Example 2: Deep recursion (depth=3) WITHOUT stabilizer")
print("-" * 70)
sequence_deep_fail = [
    Emission(),
    Recursivity(depth=3),  # Deep hierarchy - 3 levels
    Silence(),
]

valid, messages = UnifiedGrammarValidator.validate(sequence_deep_fail, epi_initial=0.0)
print(f"Sequence: AL → REMESH(depth=3) → SHA")
print(f"Valid: {valid}")
print(f"U5 Status: {[m for m in messages if m.startswith('U5:')][0]}")
print()
print("✗ Deep recursion fails - coherence fragments across 3 levels")
print("  Physics: C_parent < α·ΣC_child → fragmentation")
print()

# Example 3: Deep recursion WITH Coherence (IL) - PASSES
print("Example 3: Deep recursion (depth=3) WITH IL stabilizer")
print("-" * 70)
sequence_deep_il = [
    Emission(),
    Recursivity(depth=3),
    Coherence(),  # Scale stabilizer within ±3 window
    Silence(),
]

valid, messages = UnifiedGrammarValidator.validate(sequence_deep_il, epi_initial=0.0)
print(f"Sequence: AL → REMESH(depth=3) → IL → SHA")
print(f"Valid: {valid}")
print(f"U5 Status: {[m for m in messages if m.startswith('U5:')][0]}")
print()
print("✓ Deep recursion passes - IL stabilizes all 3 hierarchical levels")
print("  Physics: C_parent ≥ α·ΣC_child → bounded evolution")
print()

# Example 4: Deep recursion WITH Self-organization (THOL) - PASSES
print("Example 4: Deep recursion (depth=3) WITH THOL stabilizer")
print("-" * 70)
sequence_deep_thol = [
    Emission(),
    Recursivity(depth=3),
    SelfOrganization(),  # Requires U4b context (destabilizer)
    Silence(),
]

# This will fail U4b (THOL needs destabilizer), so let's add one
from tnfr.operators.definitions import Dissonance

sequence_deep_thol_fixed = [
    Emission(),
    Dissonance(),  # U4b: provides context for THOL transformer
    Recursivity(depth=3),
    SelfOrganization(),  # Scale stabilizer + transformer
    Silence(),
]

valid, messages = UnifiedGrammarValidator.validate(
    sequence_deep_thol_fixed, epi_initial=0.0
)
print(f"Sequence: AL → OZ → REMESH(depth=3) → THOL → SHA")
print(f"Valid: {valid}")
print(f"U5 Status: {[m for m in messages if m.startswith('U5:')][0]}")
print()
print("✓ Deep recursion passes - THOL provides autopoietic closure")
print("  Physics: Self-limiting boundaries at each hierarchical level")
print()

# Example 5: Very deep hierarchy (depth=10)
print("Example 5: Very deep recursion (depth=10)")
print("-" * 70)
sequence_very_deep = [
    Emission(),
    Recursivity(depth=10),  # 10 hierarchical levels
    Coherence(),
    Silence(),
]

valid, messages = UnifiedGrammarValidator.validate(sequence_very_deep, epi_initial=0.0)
print(f"Sequence: AL → REMESH(depth=10) → IL → SHA")
print(f"Valid: {valid}")
print(f"U5 Status: {[m for m in messages if m.startswith('U5:')][0]}")
print()
print("✓ Even very deep hierarchies work with stabilizers")
print("  Physics: α decreases with N, but IL maintains C_parent ≥ α·ΣC_child")
print()

# Example 6: Stabilizer window demonstration
print("Example 6: Stabilizer window (±3 operators)")
print("-" * 70)
from tnfr.operators.definitions import Transition

# Stabilizer too far from REMESH
sequence_window_fail = [
    Emission(),
    Coherence(),  # Position 1
    Transition(),
    Transition(),
    Transition(),
    Transition(),  # More than 3 ops away
    Recursivity(depth=3),  # Position 6
    Silence(),
]

valid, messages = UnifiedGrammarValidator.validate(sequence_window_fail, epi_initial=0.0)
print(f"Sequence: AL → IL → NAV → NAV → NAV → NAV → REMESH(depth=3) → SHA")
print(f"Valid: {valid}")
print(f"U5 Status: {[m for m in messages if m.startswith('U5:')][0]}")
print()
print("✗ Stabilizer outside ±3 window doesn't satisfy U5")
print("  Physics: Beyond ΔNFR decay timescale (~3 operators)")
print()

# Summary
print("=" * 70)
print("Summary: U5 Multi-Scale Coherence")
print("=" * 70)
print()
print("Physical Principle: C_parent ≥ α · Σ C_child_i")
print("  α = (1/√N) · η_phase(N) · η_coupling(N) ∈ [0.1, 0.4]")
print()
print("Requirement:")
print("  - Shallow REMESH (depth=1): No stabilizer needed")
print("  - Deep REMESH (depth>1): Scale stabilizers {IL, THOL} within ±3 ops")
print()
print("Dimensionality:")
print("  - U1-U4: TEMPORAL (operator sequences in time)")
print("  - U5: SPATIAL (hierarchical nesting in structure)")
print()
print("Independence: U5 operates on different dimension than U2+U4b")
print("  Test case: [AL, REMESH(depth=3), SHA]")
print("    U2: ✓ (no destabilizers)")
print("    U4b: ✓ (REMESH not transformer)")
print("    U5: ✗ (multi-scale hierarchy without stabilization)")
print()
print("Canonicity: STRONG - Derived from coherence conservation principle")
print("=" * 70)
