#!/usr/bin/env python3
"""
U4: BIFURCATION DYNAMICS Examples

Demonstrates U4a (triggers need handlers) and U4b (transformers need context).

Physics Basis:
- U4a: Bifurcations (∂²EPI/∂t² > τ) need control mechanisms
- U4b: Phase transitions need threshold energy (recent destabilizer)
        ZHIR specifically needs stable base (prior coherence)

Run: python u4-bifurcation-examples.py
"""

from tnfr.operators.grammar import validate_grammar, GrammarValidator
from tnfr.operators.definitions import (
    Emission,
    Reception,
    Dissonance,
    Mutation,
    Expansion,
    Coherence,
    SelfOrganization,
    Silence,
)


def example_u4a_valid():
    """Valid U4a: Bifurcation triggers with handlers."""
    print("\n" + "=" * 60)
    print("U4a VALID EXAMPLES: Triggers with Handlers")
    print("=" * 60)

    examples = [
        (
            "Dissonance + Coherence",
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
            "Multiple triggers with handlers",
            [
                Emission(),
                Dissonance(),
                Coherence(),
                Mutation(),
                SelfOrganization(),
                Silence(),
            ],
        ),
    ]

    for name, sequence in examples:
        is_valid, message = GrammarValidator.validate(sequence, epi_initial=0.0)
        print(f"\n{name}:")
        print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
        print(f"  Valid: {is_valid}")
        print(f"  Bifurcation control: {message}")


def example_u4a_invalid():
    """Invalid U4a: Triggers without handlers."""
    print("\n" + "=" * 60)
    print("U4a INVALID EXAMPLES: Uncontrolled Bifurcations")
    print("=" * 60)

    examples = [
        ("Dissonance alone", [Emission(), Dissonance(), Silence()]),
        ("Mutation alone", [Emission(), Mutation(), Silence()]),
    ]

    for name, sequence in examples:
        try:
            is_valid = validate_grammar(sequence, epi_initial=0.0)
            print(f"\n{name}: SHOULD HAVE FAILED but got {is_valid}")
        except ValueError as e:
            print(f"\n{name}:")
            print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
            print(f"  ✓ Correctly rejected: {str(e)[:80]}...")


def example_u4b_valid():
    """Valid U4b: Transformers with proper context."""
    print("\n" + "=" * 60)
    print("U4b VALID EXAMPLES: Transformers with Context")
    print("=" * 60)

    examples = [
        (
            "Mutation with context",
            [
                Emission(),
                Coherence(),  # Stable base
                Dissonance(),  # Recent destabilizer
                Mutation(),  # Transformer
                Coherence(),
                Silence(),
            ],
        ),
        (
            "Self-organization with context",
            [
                Emission(),
                Dissonance(),  # Recent destabilizer
                SelfOrganization(),  # Transformer
                Coherence(),
                Silence(),
            ],
        ),
        (
            "Minimal window",
            [
                Emission(),
                Coherence(),
                Dissonance(),
                Mutation(),  # Within ~3 ops
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


def example_u4b_invalid():
    """Invalid U4b: Transformers without context."""
    print("\n" + "=" * 60)
    print("U4b INVALID EXAMPLES: Missing Context")
    print("=" * 60)

    examples = [
        (
            "Mutation without destabilizer",
            [Emission(), Coherence(), Mutation(), Silence()],
        ),
        (
            "Mutation without prior coherence",
            [Emission(), Dissonance(), Mutation(), Coherence(), Silence()],
        ),
        (
            "Destabilizer too far",
            [
                Emission(),
                Dissonance(),
                Reception(),
                Reception(),
                Reception(),
                Mutation(),
                Silence(),
            ],
        ),
    ]

    for name, sequence in examples:
        try:
            is_valid = validate_grammar(sequence, epi_initial=0.0)
            print(f"\n{name}: SHOULD HAVE FAILED but got {is_valid}")
        except ValueError as e:
            print(f"\n{name}:")
            print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
            print(f"  ✓ Correctly rejected: {str(e)[:80]}...")


def example_operator_classification():
    """Show bifurcation-related operator sets."""
    print("\n" + "=" * 60)
    print("OPERATOR CLASSIFICATION: Bifurcation Dynamics")
    print("=" * 60)

    print("\nBIFURCATION TRIGGERS (may cause ∂²EPI/∂t² > τ):")
    print("  - Dissonance (OZ): Controlled instability")
    print("  - Mutation (ZHIR): Phase transformation")
    print("  → Create conditions for bifurcation")

    print("\nBIFURCATION HANDLERS:")
    print("  - Self-organization (THOL): Creates new structure")
    print("  - Coherence (IL): Stabilizes transition")
    print("  → Manage reorganization into new attractor")

    print("\nTRANSFORMERS (execute bifurcations):")
    print("  - Mutation (ZHIR): Phase transition")
    print("  - Self-organization (THOL): Autopoietic emergence")
    print("  → Require elevated ΔNFR for threshold crossing")


def example_zhir_requirements():
    """Explain ZHIR-specific requirements."""
    print("\n" + "=" * 60)
    print("ZHIR (MUTATION) SPECIFIC REQUIREMENTS")
    print("=" * 60)

    print("\nMutation needs TWO things:")
    print("  1. PRIOR Coherence (IL): Stable base to jump from")
    print("  2. RECENT Destabilizer: Threshold energy")

    print("\nWhy stable base?")
    print("  - Mutation is a PHASE TRANSITION")
    print("  - Need stable configuration to transform from")
    print("  - Like solid → liquid needs defined solid structure")

    print("\nWhy recent destabilizer?")
    print("  - Need elevated ΔNFR for threshold crossing")
    print("  - ΔEPI/Δt > ξ required")
    print("  - Energy from destabilizer provides this")

    print("\nPattern:")
    print("  [... Coherence ... Destabilizer ... Mutation ...]")
    print("       ^^^^^^ stable base  ^^^^^^ energy  ^^^^^^ transform")


def example_antipattern_cascade():
    """Anti-pattern: Uncontrolled bifurcation cascade."""
    print("\n" + "=" * 60)
    print("ANTI-PATTERN: Bifurcation Cascade")
    print("=" * 60)

    print("\nBAD: Multiple triggers without handlers")
    print("""
    sequence = [
        Emission(),
        Dissonance(),  # Trigger 1
        Mutation(),    # Trigger 2 - cascade!
        Silence()
    ]
    """)
    print("  ✗ System may enter chaotic regime")
    print("  ✗ Multiple bifurcations unmanaged")

    print("\nGOOD: Handler between triggers")
    print("""
    sequence = [
        Emission(),
        Coherence(),        # Stable base
        Dissonance(),       # Trigger 1
        Coherence(),        # Handler
        Mutation(),         # Trigger 2
        SelfOrganization(), # Handler
        Silence()
    ]
    """)
    print("  ✓ Each bifurcation controlled")
    print("  ✓ System guided through transitions")


def example_antipattern_window():
    """Anti-pattern: Transformer outside context window."""
    print("\n" + "=" * 60)
    print("ANTI-PATTERN: Context Window Violation")
    print("=" * 60)

    print("\nPROBLEM: Transformer too far from destabilizer")
    print("""
    sequence = [
        Emission(),
        Dissonance(),  # Position 1
        Reception(),   # ΔNFR decays...
        Reception(),   # More decay...
        Reception(),   # Even more decay...
        Mutation(),    # Position 5 - insufficient ΔNFR!
        Silence()
    ]
    """)
    print("  ✗ Window is ~3 operators")
    print("  ✗ Destabilizer at position 1 too far for position 5")
    print("  ✗ ΔNFR has decayed below threshold")

    print("\nSOLUTION: Keep transformer within ~3 ops of destabilizer")
    print("""
    sequence = [
        Emission(),
        Coherence(),
        Dissonance(),  # Position 2
        Mutation(),    # Position 3 - within window!
        Coherence(),
        Silence()
    ]
    """)


def example_handler_selection():
    """Discuss which handler for which trigger."""
    print("\n" + "=" * 60)
    print("HANDLER SELECTION: Best Practices")
    print("=" * 60)

    print("\nFor Dissonance (OZ):")
    print("  - Coherence (IL): Direct stabilization")
    print("  - Self-organization (THOL): If creating new structure")

    print("\nFor Mutation (ZHIR):")
    print("  - Self-organization (THOL): Preferred")
    print("    → Mutation creates new phase, THOL organizes it")
    print("  - Coherence (IL): Also valid")
    print("    → Stabilizes the new phase")

    print("\nGeneral rule:")
    print("  - Use THOL when new structure emerges")
    print("  - Use IL when stabilizing existing structure")


def main():
    """Run all U4 examples."""
    print("=" * 60)
    print("U4: BIFURCATION DYNAMICS")
    print("Executable Examples with Physics Traceability")
    print("=" * 60)

    # U4a examples
    example_u4a_valid()
    example_u4a_invalid()

    # U4b examples
    example_u4b_valid()
    example_u4b_invalid()

    # Classification and requirements
    example_operator_classification()
    example_zhir_requirements()

    # Anti-patterns
    example_antipattern_cascade()
    example_antipattern_window()
    example_handler_selection()

    print("\n" + "=" * 60)
    print("Examples complete! Bifurcations under control.")
    print("=" * 60)


if __name__ == "__main__":
    main()
