#!/usr/bin/env python3
"""
U3: RESONANT COUPLING Examples

Demonstrates phase verification requirement for coupling/resonance operators.

Physics Basis:
- Resonance requires phase compatibility: |φᵢ - φⱼ| ≤ Δφ_max
- Antiphase → destructive interference (non-physical)
- Based on AGENTS.md Invariant #5 + wave physics

Run: python u3-resonant-coupling-examples.py
"""

import networkx as nx
import numpy as np

from tnfr.operators.grammar import validate_grammar, GrammarValidator


def create_test_graph(theta1, theta2):
    """Create a simple graph with two nodes at specified phases."""
    G = nx.Graph()
    G.add_node(0, theta=theta1, vf=1.0, EPI=0.5, DNFR=0.0)
    G.add_node(1, theta=theta2, vf=1.0, EPI=0.6, DNFR=0.0)
    return G


def example_phase_compatibility():
    """Demonstrate phase compatibility checking."""
    print("\n" + "=" * 60)
    print("PHASE COMPATIBILITY: Valid Coupling Conditions")
    print("=" * 60)

    cases = [
        ("In phase", 0.0, 0.0, True),
        ("Small difference", 0.0, 0.3, True),
        ("Near threshold", 0.0, np.pi / 2 - 0.1, True),
        ("At threshold", 0.0, np.pi / 2, False),
        ("Beyond threshold", 0.0, np.pi / 2 + 0.1, False),
        ("Antiphase", 0.0, np.pi, False),
    ]

    for name, theta1, theta2, should_pass in cases:
        G = create_test_graph(theta1, theta2)
        delta_phi = abs(theta1 - theta2)

        print(f"\n{name}:")
        print(f"  θ₁ = {theta1:.3f}, θ₂ = {theta2:.3f}")
        print(f"  Δφ = {delta_phi:.3f} rad")

        # Try applying coupling operator
        from tnfr.operators.definitions import Coupling

        try:
            # Coupling operator should check phase during application
            op = Coupling()
            # Apply the operator - it will check phase internally
            op(G, 0, 1)
            
            if should_pass:
                print(f"  ✓ Coupling allowed (Δφ < π/2)")
            else:
                print(f"  ⚠ Should have failed but passed!")
        except (ValueError, RuntimeError, Exception) as e:
            if not should_pass:
                print(f"  ✓ Correctly rejected: {str(e)[:60]}...")
            else:
                print(f"  ✗ Should have passed but failed: {str(e)[:60]}...")



def example_coupling_resonance_operators():
    """Show which operators require phase verification."""
    print("\n" + "=" * 60)
    print("OPERATORS REQUIRING PHASE VERIFICATION (U3)")
    print("=" * 60)

    print("\nCoupling (UM):")
    print("  - Creates structural links between nodes")
    print("  - Requires: |φᵢ - φⱼ| ≤ Δφ_max")
    print("  - Effect: φᵢ(t) → φⱼ(t) (phase synchronization)")

    print("\nResonance (RA):")
    print("  - Amplifies and propagates patterns")
    print("  - Requires: Phase compatibility for constructive interference")
    print("  - Effect: Increases effective coupling strength")

    print("\n⚠ MANDATORY per AGENTS.md Invariant #5:")
    print("  'No coupling is valid without explicit phase verification'")


def example_sequence_validation():
    """Show that U3 is checked at sequence level."""
    print("\n" + "=" * 60)
    print("SEQUENCE-LEVEL VALIDATION")
    print("=" * 60)

    from tnfr.operators.definitions import Emission, Coupling, Silence

    sequence = [Emission(), Coupling(), Silence()]
    is_valid, msg = GrammarValidator.validate(sequence, epi_initial=0.0)

    print("\nSequence with coupling operator:")
    print(f"  Sequence: {[op.__class__.__name__ for op in sequence]}")
    print(f"  Grammar: {msg}")
    print("\n  ℹ U3 is META-rule: Documents requirement")
    print("  ℹ Actual phase check happens in operator preconditions")
    print("  ℹ Grammar ensures awareness that check is MANDATORY")


def example_antipattern_no_check():
    """Anti-pattern: Attempting to couple without verification."""
    print("\n" + "=" * 60)
    print("ANTI-PATTERN: Coupling Without Phase Check")
    print("=" * 60)

    print("\nBAD: Direct coupling without verification")
    print("""
    G = create_graph()
    G.nodes[0]['theta'] = 0.0
    G.nodes[1]['theta'] = 3.0  # Could be antiphase!
    
    Coupling()(G, 0, 1)  # ERROR: No phase check
    """)

    print("\nGOOD: Verify phase compatibility first")
    print("""
    from tnfr.operators.grammar import validate_resonant_coupling
    
    G = create_graph()
    G.nodes[0]['theta'] = 0.0
    G.nodes[1]['theta'] = 3.0
    
    validate_resonant_coupling(G, 0, 1)  # Raises if incompatible
    Coupling()(G, 0, 1)  # Safe
    """)


def example_antipattern_phase_drift():
    """Anti-pattern: Ignoring phase drift during sequences."""
    print("\n" + "=" * 60)
    print("ANTI-PATTERN: Phase Drift")
    print("=" * 60)

    print("\nPROBLEM: Phase changes during sequence")
    print("""
    sequence = [
        Emission(),
        Coupling(),    # Phase compatible here
        Mutation(),    # Changes θ!
        Coupling(),    # Phase may no longer be compatible!
        Silence()
    ]
    """)

    print("\nSOLUTION 1: Verify phase after transformations")
    print("  - Re-check phase after Mutation")
    print("  - Or use operators that preserve phase")

    print("\nSOLUTION 2: Couple before phase-changing operators")
    print("""
    sequence = [
        Emission(),
        Coupling(),    # Couple while phases compatible
        Silence(),
        # Then in separate sequence:
        Emission(),
        Mutation(),    # Change phase
        Silence()
    ]
    """)


def example_threshold_considerations():
    """Discuss threshold selection."""
    print("\n" + "=" * 60)
    print("THRESHOLD SELECTION: Δφ_max")
    print("=" * 60)

    print("\nTypical threshold: π/2 radians (~90 degrees)")
    print("\nPhysics basis:")
    print("  - Coupling strength ~ cos(Δφ)")
    print("  - At Δφ = 0: cos(0) = 1 (maximum coupling)")
    print("  - At Δφ = π/2: cos(π/2) = 0 (no coupling)")
    print("  - At Δφ = π: cos(π) = -1 (destructive interference)")

    print("\nPractical considerations:")
    print("  - Δφ < π/4: Strong coupling")
    print("  - π/4 < Δφ < π/2: Weak coupling")
    print("  - Δφ > π/2: Non-physical (rejected)")


def example_wave_interference():
    """Explain wave interference physics."""
    print("\n" + "=" * 60)
    print("PHYSICS: Wave Interference")
    print("=" * 60)

    print("\nTwo oscillators with phases φ₁ and φ₂:")
    print("  x₁(t) = A sin(ωt + φ₁)")
    print("  x₂(t) = A sin(ωt + φ₂)")

    print("\nWhen coupled:")
    print("  - IN PHASE (Δφ ≈ 0): Constructive interference")
    print("    → Amplitudes add: x = 2A sin(ωt)")
    print("    → Information transfer efficient")

    print("\n  - QUADRATURE (Δφ ≈ π/2): Orthogonal")
    print("    → No coherent transfer: x₁ ⊥ x₂")
    print("    → Coupling impossible")

    print("\n  - ANTIPHASE (Δφ ≈ π): Destructive interference")
    print("    → Amplitudes cancel: x = 0")
    print("    → Non-physical for information coupling")


def main():
    """Run all U3 examples."""
    print("=" * 60)
    print("U3: RESONANT COUPLING")
    print("Executable Examples with Physics Traceability")
    print("=" * 60)

    example_phase_compatibility()
    example_coupling_resonance_operators()
    example_sequence_validation()
    example_antipattern_no_check()
    example_antipattern_phase_drift()
    example_threshold_considerations()
    example_wave_interference()

    print("\n" + "=" * 60)
    print("Examples complete! Phase verification is MANDATORY.")
    print("=" * 60)


if __name__ == "__main__":
    main()
