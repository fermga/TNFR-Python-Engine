#!/usr/bin/env python3
"""Example demonstrating the canonical TNFR nodal equation.

This example shows how to use the explicit canonical equation implementation
to validate TNFR theoretical computations.
"""

from tnfr.dynamics.canonical import (
    compute_canonical_nodal_derivative,
    validate_nodal_gradient,
    validate_structural_frequency,
)
from tnfr.structural import create_nfr
from tnfr.dynamics import update_epi_via_nodal_equation


def example_1_basic_canonical_computation():
    """Example 1: Basic canonical equation computation."""
    print("=" * 60)
    print("Example 1: Basic Canonical Equation Computation")
    print("=" * 60)

    # Define TNFR variables (using values within typical ranges)
    nu_f = 0.8  # Structural frequency (Hz_str)
    delta_nfr = 0.4  # Nodal gradient (reorganization operator)

    # Compute using canonical equation: ∂EPI/∂t = νf · ΔNFR(t)
    result = compute_canonical_nodal_derivative(
        nu_f=nu_f, delta_nfr=delta_nfr, validate_units=True
    )

    print(f"\nInputs:")
    print(f"  νf (structural frequency) = {nu_f} Hz_str")
    print(f"  ΔNFR (nodal gradient)     = {delta_nfr}")

    print(f"\nCanonical Equation: ∂EPI/∂t = νf · ΔNFR(t)")
    print(f"Result: ∂EPI/∂t = {result.derivative} Hz_str")
    print(f"Validated: {result.validated}")
    print()


def example_2_unit_validation():
    """Example 2: Unit validation for structural frequency."""
    print("=" * 60)
    print("Example 2: Unit Validation")
    print("=" * 60)

    # Valid inputs
    try:
        vf = validate_structural_frequency(2.5)
        print(f"✓ Valid νf = {vf} Hz_str")
    except ValueError as e:
        print(f"✗ Invalid: {e}")

    # Zero frequency (structural silence)
    try:
        vf = validate_structural_frequency(0.0)
        print(f"✓ Valid νf = {vf} Hz_str (structural silence)")
    except ValueError as e:
        print(f"✗ Invalid: {e}")

    # Invalid: negative frequency
    try:
        vf = validate_structural_frequency(-1.0)
        print(f"✓ Valid νf = {vf} Hz_str")
    except ValueError as e:
        print(f"✗ Invalid νf < 0: {e}")

    # Valid: negative gradient (contraction)
    try:
        dnfr = validate_nodal_gradient(-0.5)
        print(f"✓ Valid ΔNFR = {dnfr} (contraction)")
    except ValueError as e:
        print(f"✗ Invalid: {e}")

    print()


def example_3_expansion_vs_contraction():
    """Example 3: Expansion vs contraction based on ΔNFR sign."""
    print("=" * 60)
    print("Example 3: Expansion vs Contraction")
    print("=" * 60)

    nu_f = 1.0

    # Positive ΔNFR: expansion
    expansion = compute_canonical_nodal_derivative(nu_f, 0.5)
    print(f"\nPositive ΔNFR (expansion):")
    print(f"  ΔNFR = +0.5 → ∂EPI/∂t = {expansion.derivative}")
    print(f"  Result: EPI increases (expansion)")

    # Negative ΔNFR: contraction
    contraction = compute_canonical_nodal_derivative(nu_f, -0.5)
    print(f"\nNegative ΔNFR (contraction):")
    print(f"  ΔNFR = -0.5 → ∂EPI/∂t = {contraction.derivative}")
    print(f"  Result: EPI decreases (contraction)")

    # Zero ΔNFR: equilibrium
    equilibrium = compute_canonical_nodal_derivative(nu_f, 0.0)
    print(f"\nZero ΔNFR (equilibrium):")
    print(f"  ΔNFR = 0.0 → ∂EPI/∂t = {equilibrium.derivative}")
    print(f"  Result: EPI stable (no reorganization)")

    print()


def example_4_structural_silence():
    """Example 4: Structural silence (νf = 0)."""
    print("=" * 60)
    print("Example 4: Structural Silence")
    print("=" * 60)

    # When νf = 0, evolution is frozen regardless of ΔNFR
    for delta_nfr in [0.5, -0.5, 1.0]:
        result = compute_canonical_nodal_derivative(0.0, delta_nfr)
        print(f"\nνf = 0.0, ΔNFR = {delta_nfr:+.1f}")
        print(f"  ∂EPI/∂t = {result.derivative}")
        print(f"  → Structural silence: evolution frozen")

    print()


def example_5_integration_with_graph():
    """Example 5: Integration with TNFR graph."""
    print("=" * 60)
    print("Example 5: Integration with TNFR Graph")
    print("=" * 60)

    # Create a TNFR node (vf must be <= 1.0 by default validation)
    G, node = create_nfr("test_node", epi=1.0, vf=0.8, theta=0.0)

    # Set nodal gradient
    G.nodes[node]["ΔNFR"] = 0.4

    print(f"\nInitial state:")
    print(f"  EPI = {G.nodes[node]['EPI']}")
    print(f"  νf  = {G.nodes[node]['νf']} Hz_str")
    print(f"  ΔNFR = {G.nodes[node]['ΔNFR']}")

    # Manually compute expected derivative
    vf = G.nodes[node]["νf"]
    dnfr = G.nodes[node]["ΔNFR"]
    expected = compute_canonical_nodal_derivative(vf, dnfr, validate_units=False)
    print(f"\nCanonical equation: ∂EPI/∂t = {expected.derivative}")

    # Integrate using TNFR engine
    update_epi_via_nodal_equation(G, dt=0.1)

    print(f"\nAfter integration (dt=0.1):")
    print(f"  EPI = {G.nodes[node]['EPI']:.6f}")
    print(f"  Expected ≈ {1.0 + 0.1 * expected.derivative:.6f}")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TNFR Canonical Nodal Equation Examples")
    print("Equation: ∂EPI/∂t = νf · ΔNFR(t)")
    print("=" * 60 + "\n")

    example_1_basic_canonical_computation()
    example_2_unit_validation()
    example_3_expansion_vs_contraction()
    example_4_structural_silence()
    example_5_integration_with_graph()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
