#!/usr/bin/env python
"""Example demonstrating migration from C1-C3 to unified U1-U4 grammar.

Note: U5 (Multi-Scale Coherence) was added later (2025-11-10) extending the
unified grammar to cover hierarchical stabilization across REMESH depth>1.

This script shows how to update code from the old C1-C3 grammar system
to the new unified U1-U4 grammar system (later extended by U5 multi-scale coherence).
"""

import warnings

# ============================================================================
# OLD APPROACH (C1-C3) - Still works but deprecated
# ============================================================================

print("=" * 70)
print("OLD APPROACH: Using deprecated C1-C3 functions")
print("=" * 70)

from tnfr.operators.grammar import (
    validate_c1_existence,
    validate_c2_boundedness,
    validate_c3_threshold,
)

# Enable deprecation warnings
warnings.simplefilter('always', DeprecationWarning)

# Example sequence
sequence = ['emission', 'dissonance', 'coherence', 'mutation', 'silence']

print(f"\nValidating sequence: {sequence}\n")

# Old way - separate C1-C3 checks (will emit warnings)
c1_valid = validate_c1_existence(sequence)
print(f"  C1 (Existence & Closure): {c1_valid}")

c2_valid = validate_c2_boundedness(sequence)
print(f"  C2 (Boundedness): {c2_valid}")

c3_valid = validate_c3_threshold(sequence)
print(f"  C3 (Threshold Physics): {c3_valid}")

print(f"\n  Overall valid (C1-C3): {c1_valid and c2_valid and c3_valid}")

# ============================================================================
# NEW APPROACH (U1-U5) - Recommended for new code (U5 concerns multi-scale REMESH stabilization)
# ============================================================================

print("\n" + "=" * 70)
print("NEW APPROACH: Using unified U1-U5 grammar (temporal + multi-scale)")
print("=" * 70)

from tnfr.operators.grammar import UnifiedGrammarValidator, validate_unified
from tnfr.operators.definitions import (
    Emission, Dissonance, Coherence, Mutation, Silence
)

# Create operator instances
ops = [Emission(), Dissonance(), Coherence(), Mutation(), Silence()]

print(f"\nValidating sequence with unified grammar:\n")

# New way - unified validation with detailed messages
valid, messages = UnifiedGrammarValidator.validate(ops, epi_initial=0.0)

print("  Unified validation messages:")
for msg in messages:
    print(f"    {msg}")

print(f"\n  Overall valid (U1-U4): {valid}")

# Convenience function for simple boolean result
print(f"\n  Using validate_unified(): {validate_unified(ops, epi_initial=0.0)}")

# ============================================================================
# UNDERSTANDING THE MAPPING
# ============================================================================

print("\n" + "=" * 70)
print("C1-C3 → U1-U4 MAPPING")
print("=" * 70)

mapping = """
Old Grammar (C1-C3)              →  Unified Grammar (U1-U4)
───────────────────────────────────────────────────────────────────
C1: EXISTENCE & CLOSURE          →  U1: STRUCTURAL INITIATION & CLOSURE
    - Must start with generator      - U1a: Start with generator if EPI=0
    - Must end with closure          - U1b: End with closure operator

C2: BOUNDEDNESS                  →  U2: CONVERGENCE & BOUNDEDNESS
    - Destabilizers need              - If destabilizers present, include
      stabilizers                       stabilizers (IL or THOL)

C3: THRESHOLD PHYSICS            →  U4: BIFURCATION DYNAMICS
    - Transformers need context       - U4a: Bifurcation triggers need handlers
                                      - U4b: Transformers need recent destabilizer

NEW: Not in C1-C3                →  U3: RESONANT COUPLING
                                      - Coupling/resonance need phase check
"""

print(mapping)

# ============================================================================
# OPERATOR SETS
# ============================================================================

print("=" * 70)
print("OPERATOR SETS")
print("=" * 70)

from tnfr.operators.grammar import (
    UNIFIED_GENERATORS,
    UNIFIED_CLOSURES,
    UNIFIED_STABILIZERS,
    UNIFIED_DESTABILIZERS,
)

print(f"""
Generators (U1a):     {sorted(UNIFIED_GENERATORS)}
Closures (U1b):       {sorted(UNIFIED_CLOSURES)}
Stabilizers (U2):     {sorted(UNIFIED_STABILIZERS)}
Destabilizers (U2):   {sorted(UNIFIED_DESTABILIZERS)}
""")

# ============================================================================
# MIGRATION CHECKLIST
# ============================================================================

print("=" * 70)
print("MIGRATION CHECKLIST")
print("=" * 70)

checklist = """
✓ Replace separate C1-C3 checks with UnifiedGrammarValidator.validate()
✓ Use validate_unified() for simple boolean validation
✓ Import operator sets from unified_grammar (UNIFIED_GENERATORS, etc.)
✓ Update documentation to reference UNIFIED_GRAMMAR_RULES.md
✓ Test with actual operator instances (not just string names)
✓ Review U3 (Resonant Coupling) - new constraint not in C1-C3

For complete migration guide, see:
  - UNIFIED_GRAMMAR_RULES.md
  - src/tnfr/operators/unified_grammar.py
"""

print(checklist)
