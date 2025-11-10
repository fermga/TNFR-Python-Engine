#!/usr/bin/env python3
"""
U2: CONVERGENCE & BOUNDEDNESS Examples

Demonstrates stabilizer-destabilizer balance for integral convergence.

Physics Basis:
- Without stabilizers: ∫νf·ΔNFR dt → ∞ (diverges)
- With stabilizers: Integral converges, coherence preserved

From integrated nodal equation:
  EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf·ΔNFR dτ

Run: python u2-convergence-examples.py
"""

from tnfr.operators.grammar import validate_grammar, GrammarValidator
from tnfr.operators.definitions import (
    Emission,
    Dissonance,
    Mutation,
    Expansion,
    Coherence,
    SelfOrganization,
    Silence,
)


def example_u2_valid():
    """Valid U2: Destabilizers balanced by stabilizers."""
    print("\n" + "=" * 60)
    print("U2 VALID EXAMPLES: Balanced Sequences")
    print("=" * 60)

    examples = [
        (
            "Single destabilizer + stabilizer",
            [Emission(), Dissonance(), Coherence(), Silence()],
        ),
        (
            "Mutation + Self-organization",
            [
                Emission(),
                Coherence(),
                Dissonance(),
                Mutation(),
                SelfOrganization(),
                Silence(),
            ],
        ),
        (
            "Multiple destabilizers + stabilizer",
            [Emission(), Dissonance(), Expansion(), Coherence(), Silence()],
        ),
        (
            "Interleaved stabilizers",
            [
                Emission(),
                Dissonance(),
                Coherence(),
                Expansion(),
                Coherence(),
                Silence(),
            ],
        ),
    ]

    for name, sequence in examples:
        is_valid, message = GrammarValidator.validate(sequence, epi_initial=0.0)
        print(f"\n{name}:")
        print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
        print(f"  Valid: {is_valid}")
        print(f"  Convergence: {message}")


def example_u2_invalid():
    """Invalid U2: Destabilizers without stabilizers."""
    print("\n" + "=" * 60)
    print("U2 INVALID EXAMPLES: Unbalanced Sequences")
    print("=" * 60)

    examples = [
        ("Dissonance alone", [Emission(), Dissonance(), Silence()]),
        ("Mutation alone", [Emission(), Mutation(), Silence()]),
        ("Multiple destabilizers", [Emission(), Dissonance(), Expansion(), Silence()]),
    ]

    for name, sequence in examples:
        try:
            is_valid = validate_grammar(sequence, epi_initial=0.0)
            print(f"\n{name}: SHOULD HAVE FAILED but got {is_valid}")
        except ValueError as e:
            print(f"\n{name}:")
            print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
            print(f"  ✓ Correctly rejected: {str(e)[:80]}...")


def example_u2_not_applicable():
    """When U2 doesn't apply."""
    print("\n" + "=" * 60)
    print("U2 NOT APPLICABLE: No Destabilizers")
    print("=" * 60)

    # No destabilizers = no convergence risk
    sequence = [Emission(), Coherence(), Silence()]
    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)

    print("\nSequence with no destabilizers:")
    print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
    print(f"  Valid: {is_valid}")
    print(f"  Message: {msg}")
    print("\n  ℹ U2 only checks sequences containing destabilizers")


def example_operator_classification():
    """Show which operators are destabilizers vs stabilizers."""
    print("\n" + "=" * 60)
    print("OPERATOR CLASSIFICATION: Destabilizers vs Stabilizers")
    print("=" * 60)

    print("\nDESTABILIZERS (increase |ΔNFR|):")
    print("  - Dissonance (OZ): Explicit dissonance")
    print("  - Mutation (ZHIR): Phase transformation")
    print("  - Expansion (VAL): Increases structural complexity")
    print("  → Positive feedback → May cause divergence")

    print("\nSTABILIZERS (reduce |ΔNFR|):")
    print("  - Coherence (IL): Direct coherence restoration")
    print("  - Self-organization (THOL): Autopoietic boundaries")
    print("  → Negative feedback → Ensures convergence")


def example_ordering_matters():
    """Demonstrate that stabilizer order matters."""
    print("\n" + "=" * 60)
    print("ANTI-PATTERN: Ordering Matters")
    print("=" * 60)

    # Stabilizer before destabilizer - less effective
    sequence1 = [Emission(), Coherence(), Dissonance(), Silence()]
    print("\nStabilizer BEFORE destabilizer:")
    print(f"  Sequence: {[op.__class__.__name__ for op in sequence1]}")
    print("  ⚠ Coherence cannot prevent later dissonance")
    print("  ⚠ Passes U2 but less effective in practice")

    # Stabilizer after destabilizer - effective
    sequence2 = [Emission(), Dissonance(), Coherence(), Silence()]
    print("\nStabilizer AFTER destabilizer:")
    print(f"  Sequence: {[op.__class__.__name__ for op in sequence2]}")
    print("  ✓ Coherence bounds dissonance growth")
    print("  ✓ Better control of |ΔNFR|")


def example_masking_antipattern():
    """Anti-pattern: Multiple destabilizers, single weak stabilizer."""
    print("\n" + "=" * 60)
    print("ANTI-PATTERN: Masking with Weak Stabilizers")
    print("=" * 60)

    sequence = [
        Emission(),
        Dissonance(),  # +ΔNFR
        Expansion(),  # ++ΔNFR
        Mutation(),  # +++ΔNFR
        Coherence(),  # -ΔNFR (may not be sufficient!)
        Silence(),
    ]

    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)
    print("\nMultiple destabilizers, single stabilizer:")
    print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
    print(f"  Grammar valid: {is_valid}")
    print("  ⚠ WARNING: Technically passes U2...")
    print("  ⚠ But integral may still be large!")
    print("  ⚠ Better: Add more stabilizers or reduce destabilizers")


def example_interleaving_pattern():
    """Good pattern: Interleave stabilizers with destabilizers."""
    print("\n" + "=" * 60)
    print("GOOD PATTERN: Interleaved Stabilizers")
    print("=" * 60)

    sequence = [
        Emission(),
        Dissonance(),
        Coherence(),  # Bound first destabilizer
        Expansion(),
        Coherence(),  # Bound second destabilizer
        Mutation(),
        SelfOrganization(),  # Bound third destabilizer
        Silence(),
    ]

    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)
    print("\nInterleaved stabilizers:")
    print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
    print(f"  Valid: {is_valid}")
    print("  ✓ Each destabilizer has nearby stabilizer")
    print("  ✓ Better |ΔNFR| control throughout sequence")
    print("  ✓ Lower risk of divergence")


def main():
    """Run all U2 examples."""
    print("=" * 60)
    print("U2: CONVERGENCE & BOUNDEDNESS")
    print("Executable Examples with Physics Traceability")
    print("=" * 60)

    example_u2_valid()
    example_u2_invalid()
    example_u2_not_applicable()
    example_operator_classification()
    example_ordering_matters()
    example_masking_antipattern()
    example_interleaving_pattern()

    print("\n" + "=" * 60)
    print("Examples complete! Demonstrates ∫νf·ΔNFR dt convergence.")
    print("=" * 60)


if __name__ == "__main__":
    main()
