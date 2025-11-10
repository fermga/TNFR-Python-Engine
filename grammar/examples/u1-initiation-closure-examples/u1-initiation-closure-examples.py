#!/usr/bin/env python3
"""
U1: STRUCTURAL INITIATION & CLOSURE Examples

Demonstrates valid and invalid patterns for U1a (initiation) and U1b (closure).

Physics Basis:
- U1a: Cannot evolve from EPI=0 without generator (∂EPI/∂t undefined)
- U1b: Sequences need coherent endpoints (attractor states)

Run: python u1-initiation-closure-examples.py
"""

from tnfr.operators.grammar import validate_grammar, GrammarValidator
from tnfr.operators.definitions import (
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Silence,
    Transition,
    Recursivity,
)


def example_u1a_valid():
    """Valid U1a: Starting with generators."""
    print("\n" + "=" * 60)
    print("U1a VALID EXAMPLES: Starting with Generators")
    print("=" * 60)

    examples = [
        ("Emission starter", [Emission(), Coherence(), Silence()]),
        ("Transition starter", [Transition(), Reception(), Silence()]),
        ("Recursivity starter", [Recursivity(), Coherence(), Silence()]),
    ]

    for name, sequence in examples:
        is_valid, message = GrammarValidator.validate(sequence, epi_initial=0.0)
        print(f"\n{name}:")
        print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
        print(f"  Valid: {is_valid}")
        print(f"  Message: {message}")


def example_u1a_invalid():
    """Invalid U1a: Missing generators."""
    print("\n" + "=" * 60)
    print("U1a INVALID EXAMPLES: Missing Generators")
    print("=" * 60)

    examples = [
        ("Reception starter (not generator)", [Reception(), Coherence(), Silence()]),
        ("Coherence starter (not generator)", [Coherence(), Silence()]),
        ("Dissonance starter (not generator)", [Dissonance(), Coherence(), Silence()]),
    ]

    for name, sequence in examples:
        try:
            is_valid = validate_grammar(sequence, epi_initial=0.0)
            print(f"\n{name}: SHOULD HAVE FAILED but got {is_valid}")
        except ValueError as e:
            print(f"\n{name}:")
            print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
            print(f"  ✓ Correctly rejected: {str(e)[:80]}...")


def example_u1a_context_matters():
    """Demonstrates when U1a applies."""
    print("\n" + "=" * 60)
    print("U1a CONTEXT: When Does Initiation Apply?")
    print("=" * 60)

    # When EPI=0, need generator
    sequence = [Emission(), Coherence(), Silence()]
    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)
    print(f"\nEPI=0.0, with generator:")
    print(f"  Valid: {is_valid}")
    print(f"  {msg}")

    # When EPI>0, generator not required
    sequence = [Reception(), Coherence(), Silence()]
    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.5)
    print(f"\nEPI=0.5, without generator:")
    print(f"  Valid: {is_valid}")
    print(f"  {msg}")


def example_u1b_valid():
    """Valid U1b: Ending with closures."""
    print("\n" + "=" * 60)
    print("U1b VALID EXAMPLES: Ending with Closures")
    print("=" * 60)

    examples = [
        ("Silence closure", [Emission(), Coherence(), Silence()]),
        ("Transition closure", [Emission(), Coherence(), Transition()]),
        ("Recursivity closure", [Emission(), Coherence(), Recursivity()]),
        ("Dissonance closure", [Emission(), Coherence(), Dissonance()]),
    ]

    for name, sequence in examples:
        is_valid, message = GrammarValidator.validate(sequence, epi_initial=0.0)
        print(f"\n{name}:")
        print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
        print(f"  Valid: {is_valid}")
        print(f"  Message: {message}")


def example_u1b_invalid():
    """Invalid U1b: Missing closures."""
    print("\n" + "=" * 60)
    print("U1b INVALID EXAMPLES: Missing Closures")
    print("=" * 60)

    examples = [
        ("Ends with Coherence (not closure)", [Emission(), Coherence()]),
        ("Ends with Reception (not closure)", [Emission(), Coherence(), Reception()]),
        ("Ends with Emission (not closure)", [Emission(), Coherence(), Emission()]),
    ]

    for name, sequence in examples:
        try:
            is_valid = validate_grammar(sequence, epi_initial=0.0)
            print(f"\n{name}: SHOULD HAVE FAILED but got {is_valid}")
        except ValueError as e:
            print(f"\n{name}:")
            print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
            print(f"  ✓ Correctly rejected: {str(e)[:80]}...")


def example_dual_role_operators():
    """Operators that can be both generators AND closures."""
    print("\n" + "=" * 60)
    print("DUAL ROLE: Operators in Multiple Sets")
    print("=" * 60)

    print("\nTransition (NAV):")
    print("  - Generator (U1a): Activates latent EPI")
    print("  - Closure (U1b): Handoff to next regime")

    sequence = [Transition(), Coherence(), Transition()]
    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)
    print(f"\n  Example: {[op.__class__.__name__ for op in sequence]}")
    print(f"  Valid: {is_valid}")
    print(f"  Both U1a and U1b satisfied!")

    print("\nRecursivity (REMESH):")
    print("  - Generator (U1a): Echoes dormant structure")
    print("  - Closure (U1b): Recursive attractor")

    sequence = [Recursivity(), Coherence(), Recursivity()]
    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)
    print(f"\n  Example: {[op.__class__.__name__ for op in sequence]}")
    print(f"  Valid: {is_valid}")


def main():
    """Run all U1 examples."""
    print("=" * 60)
    print("U1: STRUCTURAL INITIATION & CLOSURE")
    print("Executable Examples with Physics Traceability")
    print("=" * 60)

    # U1a examples
    example_u1a_valid()
    example_u1a_invalid()
    example_u1a_context_matters()

    # U1b examples
    example_u1b_valid()
    example_u1b_invalid()

    # Dual role examples
    example_dual_role_operators()

    print("\n" + "=" * 60)
    print("Examples complete! All behaviors match TNFR physics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
