"""Basic test to validate canonical.py extension with coupled system."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnfr.dynamics.canonical import compute_canonical_nodal_derivative, compute_extended_nodal_system  # noqa: E402


def test_classical_limit():
    """Test that verifies J=0 recovers classical dynamics."""
    print("🧪 Test 1: Classical limit (J_φ = J_ΔNFR = 0)")
    
    # Test parameters
    nu_f = 1.5
    delta_nfr = 0.4
    theta = 0.5
    
    # Classical system
    classical_result = compute_canonical_nodal_derivative(nu_f, delta_nfr)
    
    # Extended system with zero fluxes
    extended_result = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=0.0,            # No phase flux
        j_dnfr_divergence=0.0,  # No divergence
        coupling_strength=1.0,
    )
    
    # Verify ∂EPI/∂t is identical
    classical_depi = classical_result.derivative
    extended_depi = extended_result.classical_derivative
    
    error = abs(classical_depi - extended_depi)
    tolerance = 1e-10
    
    print(f"  ∂EPI/∂t (classical): {classical_depi:.6f}")
    print(f"  ∂EPI/∂t (extended): {extended_depi:.6f}")
    print(f"  Error: {error:.2e}")
    print("  ✅ PASS" if error < tolerance else "  ❌ FAIL")
    
    return error < tolerance


def test_extended_dynamics():
    """Test that verifies extended dynamics with non-zero fluxes."""
    print("\n🧪 Test 2: Extended dynamics with fluxes")
    
    # Parameters with active fluxes
    result = compute_extended_nodal_system(
        nu_f=1.2,
        delta_nfr=0.3,
        theta=0.8,
        j_phi=0.15,            # Active phase flux
        j_dnfr_divergence=-0.05,  # Negative divergence (sink)
        coupling_strength=0.7,
    )
    
    print(f"  ∂EPI/∂t: {result.classical_derivative:.6f}")
    print(f"  ∂θ/∂t: {result.phase_derivative:.6f}")
    print(f"  ∂ΔNFR/∂t: {result.dnfr_derivative:.6f}")
    print(f"  J_φ: {result.j_phi:.6f}")
    print(f"  ∇·J_ΔNFR: {result.j_dnfr_divergence:.6f}")
    print(f"  κ: {result.coupling_strength:.6f}")
    
    # Basic checks
    checks = []
    
    # ∂EPI/∂t should be non-zero (active evolution)
    checks.append(("EPI evolution", abs(result.classical_derivative) > 1e-6))
    
    # ∂θ/∂t should reflect J_φ transport
    checks.append(("Phase transport", abs(result.phase_derivative) > 1e-6))
    
    # ∂ΔNFR/∂t should respond to divergence
    checks.append(("DNFR conservation", result.dnfr_derivative > 0))  # Negative divergence → ΔNFR increases
    
    # Validation should pass
    checks.append(("Validation", result.validated))
    
    all_pass = all(check[1] for check in checks)
    
    for name, passed in checks:
        print(f"  {name}: {'✅' if passed else '❌'}")
    
    return all_pass


def test_validation_errors():
    """Test that validation catches errors."""
    print("\n🧪 Test 3: Error validation")
    
    try:
        # Negative νf must fail
        _ = compute_extended_nodal_system(
            nu_f=-1.0,  # ❌ Negative
            delta_nfr=0.1,
            theta=0.0,
            j_phi=0.0,
            j_dnfr_divergence=0.0,
            validate_units=True
        )
        print("  ❌ FAIL: Negative νf not detected")
        return False
    except ValueError as e:
        print(f"  ✅ Negative νf detected: {str(e)[:50]}...")
        
    try:
        # Infinite flux must fail
        _ = compute_extended_nodal_system(
            nu_f=1.0,
            delta_nfr=0.1,
            theta=0.0,
            j_phi=float('inf'),  # ❌ Infinite
            j_dnfr_divergence=0.0,
            validate_units=True
        )
        print("  ❌ FAIL: Infinite J_φ not detected")
        return False
    except ValueError as e:
        print(f"  ✅ Infinite J_φ detected: {str(e)[:50]}...")
        
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("🎯 VALIDATION: canonical.py extended with coupled system")
    print("=" * 60)
    
    results = []
    
    # Test 1: Classical limit
    results.append(test_classical_limit())
    
    # Test 2: Extended dynamics  
    results.append(test_extended_dynamics())
    
    # Test 3: Error validation
    results.append(test_validation_errors())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    test_names = [
    "Classical limit",
    "Extended dynamics", 
    "Error validation"
    ]
    
    for i, (name, passed) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Test {i+1} ({name}): {status}")
    
    total_pass = sum(results)
    print(f"\nResult: {total_pass}/{len(results)} tests passed")
    
    if total_pass == len(results):
        print("🎉 SUCCESS: canonical.py extension working correctly")
        return True
    else:
        print("⚠️ WARNING: Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)