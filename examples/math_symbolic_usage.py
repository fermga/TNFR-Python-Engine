"""
Example: Using TNFR Symbolic Mathematics for Sequence Analysis.

This example demonstrates how to use the symbolic math module to:
1. Analyze convergence properties of operator sequences
2. Detect bifurcation risks
3. Validate grammar rules (U2, U4)
4. Design safe sequences
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnfr.math import symbolic


def analyze_sequence_safety():
    """Analyze if a sequence is safe according to TNFR physics."""
    print("=" * 70)
    print("EXAMPLE: Operator Sequence Safety Analysis")
    print("=" * 70)
    
    # Scenario: Dissonance operator applied without stabilizer
    print("\n📋 SCENARIO: Applying Dissonance (OZ) without Coherence (IL)")
    print("-" * 70)
    
    # Dissonance increases ΔNFR with positive feedback (λ > 0)
    print("\n1. Checking Convergence (U2 Grammar Rule):")
    growth_rate = 0.15  # Positive (destabilizer effect)
    time_horizon = 10.0
    
    converges, explanation, integral_val = \
        symbolic.check_convergence_exponential(growth_rate, time_horizon)
    
    print(f"   Growth rate (λ): {growth_rate}")
    print(f"   Time horizon: {time_horizon}s")
    print(f"   Result: {explanation}")
    if integral_val:
        print(f"   Integral value: {integral_val:.4f}")
    
    if not converges:
        print("\n   ⚠️ U2 VIOLATION: Sequence needs stabilizers {IL, THOL}")
    
    # Check bifurcation risk
    print("\n2. Checking Bifurcation Risk (U4 Grammar Rule):")
    
    # After dissonance: high ΔNFR and increasing
    at_risk, second_deriv, recommendation = \
        symbolic.evaluate_bifurcation_risk(
            nu_f_val=1.5,         # Moderate frequency
            delta_nfr_val=2.0,    # High gradient (from dissonance)
            d_nu_f_dt=0.3,        # Frequency increasing
            d_delta_nfr_dt=1.2,   # Gradient accelerating
            threshold=1.0
        )
    
    print(f"   νf: 1.5 Hz_str")
    print(f"   ΔNFR: 2.0")
    print(f"   ∂νf/∂t: 0.3")
    print(f"   ∂ΔNFR/∂t: 1.2")
    print(f"   Threshold (τ): 1.0")
    print(f"   ∂²EPI/∂t²: {second_deriv:.4f}")
    print(f"\n   {recommendation}")
    
    # Corrected sequence
    print("\n" + "=" * 70)
    print("🔧 CORRECTED SEQUENCE: [Dissonance, Coherence]")
    print("=" * 70)
    
    print("\n1. After adding Coherence stabilizer:")
    growth_rate_corrected = -0.1  # Negative (stabilizer effect)
    
    converges_c, explanation_c, integral_val_c = \
        symbolic.check_convergence_exponential(
            growth_rate_corrected, time_horizon
        )
    
    print(f"   Growth rate (λ): {growth_rate_corrected}")
    print(f"   Result: {explanation_c}")
    if integral_val_c:
        print(f"   Integral value: {integral_val_c:.4f}")
    
    if converges_c:
        print("\n   ✓ U2 SATISFIED: Integral converges")
    
    print("\n2. After stabilization:")
    at_risk_c, second_deriv_c, recommendation_c = \
        symbolic.evaluate_bifurcation_risk(
            nu_f_val=1.5,
            delta_nfr_val=0.5,    # Reduced by stabilizer
            d_nu_f_dt=0.1,        # Slower growth
            d_delta_nfr_dt=-0.2,  # Now decreasing (negative feedback)
            threshold=1.0
        )
    
    print(f"   νf: 1.5 Hz_str")
    print(f"   ΔNFR: 0.5 (reduced)")
    print(f"   ∂νf/∂t: 0.1")
    print(f"   ∂ΔNFR/∂t: -0.2 (negative feedback)")
    print(f"   ∂²EPI/∂t²: {second_deriv_c:.4f}")
    print(f"\n   {recommendation_c}")
    
    if not at_risk_c:
        print("\n   ✓ U4 SATISFIED: Below bifurcation threshold")
    
    print("\n" + "=" * 70)


def demonstrate_nodal_equation():
    """Demonstrate nodal equation analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Nodal Equation Analysis")
    print("=" * 70)
    
    # Display equation
    print("\n1. Canonical TNFR Nodal Equation:")
    eq = symbolic.get_nodal_equation()
    print(symbolic.pretty_print(eq))
    print(f"\nLaTeX: {symbolic.latex_export(eq)}")
    
    # Solve for simple case
    print("\n2. Analytical Solution (constant parameters):")
    print("   Given: νf = 2.0 Hz_str, ΔNFR = 0.3, EPI(0) = 1.0")
    
    solution = symbolic.solve_nodal_equation_constant_params(
        nu_f_val=2.0,
        delta_nfr_val=0.3,
        EPI_0=1.0,
        t0=0
    )
    
    print(f"\n   Solution: EPI(t) = {solution}")
    
    # Evaluate at specific times
    print("\n   Trajectory:")
    for time in [0, 1, 2, 5, 10]:
        epi_val = solution.subs(symbolic.t, time)
        print(f"   t = {time:2d}s → EPI = {float(epi_val):.3f}")
    
    print("\n   Interpretation:")
    print("   - Linear growth at rate νf·ΔNFR = 2.0·0.3 = 0.6 per second")
    print("   - Node actively reorganizing (νf > 0, ΔNFR > 0)")
    print("   - Stable evolution (both parameters constant)")
    
    print("\n" + "=" * 70)


def main():
    """Run all examples."""
    print("\n🧮 TNFR SYMBOLIC MATHEMATICS - USAGE EXAMPLES")
    print("\n")
    
    # Example 1: Sequence safety
    analyze_sequence_safety()
    
    # Example 2: Nodal equation
    demonstrate_nodal_equation()
    
    print("\n✅ Examples completed successfully")
    print("\nNext steps:")
    print("- Explore src/tnfr/math/symbolic.py for more functions")
    print("- Run tests: pytest tests/test_math_symbolic.py -v")
    print("- See AGENTS.md for TNFR physics foundation")
    print()


if __name__ == "__main__":
    main()
